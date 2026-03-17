from typing import *
import os
import copy
import functools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
import imageio
from PIL import Image

# 添加LPIPS相关导入
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("警告: LPIPS未安装，将跳过LPIPS loss计算")
    LPIPS_AVAILABLE = False

from ...modules import sparse as sp
from ...utils.general_utils import dict_reduce
from ...utils.data_utils import cycle, BalancedResumableSampler
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin


class SparseFlowMatchingTrainer(FlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
    """
    
    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        print("Preparing dataloader...")
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        print("Created BalancedResumableSampler")
        
        print("Creating DataLoader...")
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=0,  # 禁用多进程，使用单进程
            pin_memory=False,  # 禁用pin_memory
            drop_last=True,
            persistent_workers=False,  # 禁用persistent_workers
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        print("Created DataLoader successfully")
        
        print("Creating data iterator...")
        self.data_iterator = cycle(self.dataloader)
        print("Data iterator created successfully")
    
    def set_pipeline_model(self, pipeline_model):
        """设置pipeline_model，用于LPIPS loss计算"""
        self.pipeline_model = pipeline_model
        print("已设置pipeline_model用于LPIPS loss计算")
        
    def training_losses(
        self,
        x_0: sp.SparseTensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x ... x C] sparse tensor of the inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        # 确保x_0是独立的tensor，避免重复计算图
        if x_0.feats.requires_grad:
            x_0 = x_0.replace(x_0.feats.detach().clone())
        
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)
        
        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]
        
        # 计算LPIPS loss（使用dataset的decode与渲染）
        lpips_loss = self._compute_lpips_loss(pred, x_0, cond, noise=noise, target=target, **kwargs)
        if lpips_loss is not None:
            terms["lpips"] = lpips_loss
            # 将LPIPS loss加入到总loss中，可以调整权重
            lpips_weight = getattr(self, 'lpips_weight', 0.1)  # 默认权重0.1
            terms["loss"] = terms["loss"] + lpips_weight * lpips_loss
        
        # 调试：检查loss和预测值的统计信息
        if hasattr(self, '_debug_step'):
            self._debug_step += 1
        else:
            self._debug_step = 0
        print(self._debug_step)
        if self._debug_step % 100 == 0:  # 每100步打印一次
            print(f"[训练步骤{self._debug_step}] Loss统计:")
            print(f"  MSE Loss: {terms['mse'].item():.6f}")
            if "lpips" in terms:
                print(f"  LPIPS Loss: {terms['lpips'].item():.6f}")
            print(f"  预测特征范围: [{pred.feats.min().item():.4f}, {pred.feats.max().item():.4f}]")
            print(f"  目标特征范围: [{target.feats.min().item():.4f}, {target.feats.max().item():.4f}]")
            print(f"  预测特征均值: {pred.feats.mean().item():.4f}, 标准差: {pred.feats.std().item():.4f}")
            print(f"  目标特征均值: {target.feats.mean().item():.4f}, 标准差: {target.feats.std().item():.4f}")
          
        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        sampler = self.get_sampler()
        sample_gt = []
        sample = []
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            data = {k: v[:batch].cuda() if not isinstance(v, list) else v[:batch] for k, v in data.items()}
            noise = data['x_0'].replace(torch.randn_like(data['x_0'].feats))
            sample_gt.append(data['x_0'])
            cond_vis.append(self.vis_cond(**data))
            del data['x_0']
            args = self.get_inference_cond(**data)
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=50, cfg_strength=3.0, verbose=verbose,
            )
            sample.append(res.samples)

        sample_gt = sp.sparse_cat(sample_gt)
        sample = sp.sparse_cat(sample)
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample', 'sample_type': 'ground_truth'},
            'sample': {'value': sample, 'type': 'sample', 'sample_type': 'model_generated'},
        }
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))

        # 额外保存latent的PCA可视化
        try:
            save_root = os.path.join(self.output_dir, 'samples', f'step{self.step:07d}')
            os.makedirs(save_root, exist_ok=True)
            # GT PCA
            gt_pca_img = self.dataset._visualize_latent_pca(sample_gt)
            gt_np = (gt_pca_img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(gt_np).save(os.path.join(save_root, f'pca_sample_gt_step{self.step:07d}.png'))
            # Sample PCA
            pred_pca_img = self.dataset._visualize_latent_pca(sample)
            pred_np = (pred_pca_img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(pred_np).save(os.path.join(save_root, f'pca_sample_step{self.step:07d}.png'))
        except Exception:
            pass
        
        # 添加更彻底的测试，参考finetune.py
        #if self.step>=1000:

        self._run_thorough_test(sample_dict)
        
        return sample_dict
    
    def _run_thorough_test(self, sample_dict):
        """运行更彻底的测试，参考finetune.py中的测试逻辑"""
        try:
            print("=" * 60)
            print("开始运行彻底测试...")
            print("=" * 60)
            
            # 测试路径配置
            test_image_paths = "/home/xinyue_liang/lxy/dreamposible/1w/2_image_gen/test_imgs_paths.txt"
            results_dir = f"/home/xinyue_liang/lxy/aaa_Trellis/TRELLIS/outputs/thorough_test_{self.step}"
            
            # 检查测试图像路径文件是否存在
            if not os.path.exists(test_image_paths):
                print(f"测试图像路径文件不存在: {test_image_paths}")
                return
            
            # 创建结果目录
            os.makedirs(results_dir, exist_ok=True)
            
            # 读取测试图像路径
            with open(test_image_paths, 'r') as f:
                image_paths = [line.strip() for line in f.readlines()]
            
            print(f"找到 {len(image_paths)} 个测试图像")
            
            # 获取pipeline（如果存在）
            pipeline = None
            try:
                from trellis.pipelines import TrellisImageTo3DPipeline
                pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
                pipeline.cuda()
                
                # 将当前训练的模型替换到pipeline中
                if 'denoiser' in self.models:
                    pipeline.models["slat_flow_model"] = self.models['denoiser']
                    print("已将训练的denoiser模型加载到pipeline中")
                else:
                    print("警告: 未找到denoiser模型")
                    return
                    
            except Exception as e:
                print(f"无法创建pipeline: {e}")
                return
            
            # 对每个测试图像进行推理
            for image_path in image_paths:
                try:
                    print(f"处理图像: {image_path}")
                    
                    # 提取文件名（不包含扩展名）
                    image_filename = os.path.splitext(os.path.basename(image_path))[0]
                    
                    # 加载和预处理图像
                    from PIL import Image
                    image = Image.open(image_path)
                    image = pipeline.preprocess_image(image)
                    
                    # 运行推理
                    try:
                        outputs = pipeline.run(
                            image,
                            seed=2025,
                            preprocess_image=False,
                            formats=["gaussian", "mesh"],
                            sparse_structure_sampler_params={
                                "steps": 12,
                                "cfg_strength": 7.5,
                            },
                            slat_sampler_params={
                                "steps": 12,
                                "cfg_strength": 3,
                            },
                        )
                        # 检查outputs的格式
                        if isinstance(outputs, tuple):
                            print(f"pipeline.run返回了 {len(outputs)} 个值")
                            outputs = outputs[0]  # 取第一个元素作为outputs
                        elif isinstance(outputs, dict):
                            print("pipeline.run返回了字典格式")
                        else:
                            print(f"pipeline.run返回了未知格式: {type(outputs)}")
                            continue
                    except Exception as e:
                        print(f"pipeline.run执行失败: {e}")
                        continue
                    
                    # 生成视频
                    from trellis.utils import render_utils
                    
                    # 检查outputs的结构
                    print(f"outputs类型: {type(outputs)}")
                    if isinstance(outputs, dict):
                        print(f"outputs键: {list(outputs.keys())}")
                        if 'gaussian' in outputs:
                            print(f"gaussian类型: {type(outputs['gaussian'])}")
                            if len(outputs['gaussian']) > 0:
                                print(f"gaussian[0]类型: {type(outputs['gaussian'][0])}")
                    
                    # 确保outputs包含gaussian数据
                    if not isinstance(outputs, dict) or 'gaussian' not in outputs:
                        print("outputs中没有gaussian数据，跳过视频生成")
                        continue
                    
                    if len(outputs['gaussian']) == 0:
                        print("gaussian列表为空，跳过视频生成")
                        continue
                    
                    video = render_utils.render_video(outputs['gaussian'][0])['color']
                    
                    # 处理视频格式
                    import numpy as np
                    import imageio
                    processed_video = []
                    for frame in video:
                        if len(frame.shape) == 3 and frame.shape[0] == 3:  # [C, H, W]
                            frame = frame.permute(1, 2, 0)  # [H, W, C]
                        
                        if isinstance(frame, torch.Tensor):
                            frame = frame.detach().cpu().numpy()
                        
                        frame = np.clip(frame, 0, 1)
                        frame = (frame * 255).astype(np.uint8)
                        processed_video.append(frame)
                    
                    # 保存视频
                    video_path = f"{results_dir}/{image_filename}_video_gs.mp4"
                    imageio.mimsave(video_path, processed_video, fps=30)
                    
                    # 生成单帧图像
                    video_gaussian, image_masks = render_utils.render_around_view(outputs['gaussian'][0], r=1.7)
                    video_gaussian = torch.stack([frame for frame in video_gaussian])
                    image_masks = torch.stack([torch.tensor(np.array(frame)) for frame in image_masks])
                    
                    # 保存单帧图像
                    for frame_idx in range(video_gaussian.shape[0]):
                        image_tensor = video_gaussian[frame_idx]
                        image_mask = image_masks[frame_idx]
                        
                        # 转换为PIL图像
                        image_np = image_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255.
                        image_mask = image_mask.permute(1, 2, 0).detach().cpu()
                        image_mask = torch.cat([image_mask]*3, dim=-1).numpy().astype(np.uint8)
                        
                        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
                        image = Image.fromarray(image_np)
                        image_mask = Image.fromarray(image_mask)
                        
                        # 保存图像
                        image_path_single = f"{results_dir}/{image_filename}_{frame_idx}.png"
                        image.save(image_path_single)
                    
                    print(f"完成图像 {image_filename} 的处理")
                    
                except Exception as e:
                    print(f"处理图像 {image_path} 时出错: {e}")
                    continue
            
            print("=" * 60)
            print(f"彻底测试完成！结果保存在: {results_dir}")
            print("=" * 60)
            
        except Exception as e:
            print(f"彻底测试过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
    
    def _compute_lpips_loss(self, pred, x_0, cond, noise=None, target=None, **kwargs):
        """
        计算LPIPS loss：
        - 使用训练时相同的noise与target，将pred重构成x0_pred
        - 使用dataset的decode_latent与_render_gaussian渲染四宫格图
        - 与由target重构得到的x0_gt渲染图做LPIPS
        
        Args:
            pred: 模型预测的velocity field (Flow Matching)
            x_0: 真实的x0
            cond: 条件信息
            kwargs: 其他参数
            
        Returns:
            LPIPS loss tensor 或 None（如果计算失败）
        """
        if not LPIPS_AVAILABLE:
            return None
            
        try:
            # 初始化LPIPS模型（如果还没有初始化）
            if not hasattr(self, '_lpips_model'):
                self._lpips_model = lpips.LPIPS(net='alex').to(pred.device)
                self._lpips_model.eval()
            
            # 使用训练时的noise与target重构x0
            if noise is None or target is None:
                # 回退：按公式构造
                noise = torch.randn_like(x_0.feats)
                target = self.get_v(x_0, x_0.replace(noise), self.sample_t(x_0.shape[0]).to(x_0.device).float())

            # reconstructed x0 from pred (velocity): x0_pred = noise - pred
            x0_pred_feats = noise.feats - pred.feats
            x0_pred = sp.SparseTensor(feats=x0_pred_feats, coords=x_0.coords)

            # reconstructed x0 from target (ground truth velocity): x0_gt = noise - target
            x0_gt_feats = noise.feats - target.feats
            x0_gt = sp.SparseTensor(feats=x0_gt_feats, coords=x_0.coords)

            # 使用dataset的decode和渲染，内部已做反归一化
            if not hasattr(self.dataset, 'decode_latent') or not hasattr(self.dataset, '_render_gaussian'):
                print("警告: dataset缺少decode_latent或_render_gaussian，跳过LPIPS计算")
                return None

            # 先确保特征维度为8
            def ensure_feat_dim8(st: sp.SparseTensor) -> sp.SparseTensor:
                if st.feats.shape[1] == 8:
                    return st
                if st.feats.shape[1] > 8:
                    return st.replace(st.feats[:, :8])
                padded = torch.zeros(st.feats.shape[0], 8, device=st.feats.device, dtype=st.feats.dtype)
                padded[:, :st.feats.shape[1]] = st.feats
                return st.replace(padded)

            x0_pred = ensure_feat_dim8(x0_pred)
            x0_gt = ensure_feat_dim8(x0_gt)

            # 解码并渲染预测的x0
            decoded_pred = self.dataset.decode_latent_grad(x0_pred, sample_type="model_generated")
            pred_grid = self.dataset._render_gaussian(decoded_pred)  # [3,H,W]

            # 使用样本中的gt_image作为GT四宫格，不再解码x0_gt
            gt_grid = None
            if 'gt_image' in kwargs and kwargs['gt_image'] is not None:
                gt_img_raw = kwargs['gt_image']
                # 处理列表/批次/PIL/ndarray/tensor多种输入
                if isinstance(gt_img_raw, list) and len(gt_img_raw) > 0:
                    gt_img_raw = gt_img_raw[0]
                if isinstance(gt_img_raw, Image.Image):
                    gt_img = torch.tensor(np.array(gt_img_raw)).permute(2, 0, 1).float() / 255.0
                elif isinstance(gt_img_raw, np.ndarray):
                    gt_img = torch.tensor(gt_img_raw).permute(2, 0, 1).float() / 255.0
                elif isinstance(gt_img_raw, torch.Tensor):
                    if gt_img_raw.dim() == 4:
                        gt_img = gt_img_raw[0]
                    else:
                        gt_img = gt_img_raw
                    gt_img = gt_img.float()
                    if gt_img.max() > 1.5:
                        gt_img = gt_img / 255.0
                else:
                    gt_img = None

                if gt_img is not None:
                    gt_img = gt_img.to(pred_grid.device)
                    # 尺寸对齐
                    if gt_img.shape[-2:] != pred_grid.shape[-2:]:
                        gt_img = F.interpolate(gt_img.unsqueeze(0), size=pred_grid.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
                    gt_grid = gt_img.clamp(0, 1)

            if gt_grid is None:
                return None

            if pred_grid is None or gt_grid is None:
                return None

            # LPIPS期望[-1,1]
            pred_input = (pred_grid.unsqueeze(0) * 2 - 1).clamp(-1, 1)
            gt_input = (gt_grid.unsqueeze(0) * 2 - 1).clamp(-1, 1)

            with torch.cuda.amp.autocast(enabled=False):
                lpips_loss = self._lpips_model(pred_input, gt_input).mean()

            return lpips_loss
            
        except Exception as e:
            print(f"LPIPS loss计算失败: {e}")
            import traceback
            traceback.print_exc()
            return None


class SparseFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, SparseFlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """
    pass


class TextConditionedSparseFlowMatchingCFGTrainer(TextConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse text-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        text_cond_model(str): Text conditioning model.
    """
    pass


class ImageConditionedSparseFlowMatchingCFGTrainer(ImageConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
    """
    pass
