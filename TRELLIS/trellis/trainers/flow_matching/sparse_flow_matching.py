from typing import *
import os
import copy
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
import imageio
from PIL import Image

from ...modules import sparse as sp
from ...utils.general_utils import dict_reduce
from ...utils.data_utils import cycle, BalancedResumableSampler
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin


# ===== 全局缓存 CLIP 模型，避免重复加载 =====
_CLIP_MODEL = None
_CLIP_AVAILABLE = True
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
_DINO_MODEL = None

def get_clip_model(device: torch.device):
    global _CLIP_MODEL, _CLIP_AVAILABLE, _CLIP_MEAN, _CLIP_STD
    if not _CLIP_AVAILABLE:
        return None
    if _CLIP_MODEL is None:
        try:
            import clip  # OpenAI CLIP
            _CLIP_MODEL, _ = clip.load('ViT-B/32', device=device, jit=False)
            for p in _CLIP_MODEL.parameters():
                p.requires_grad_(False)
            _CLIP_MODEL.eval()
            _CLIP_MEAN = _CLIP_MEAN.to(device)
            _CLIP_STD = _CLIP_STD.to(device)
        except Exception:
            print("警告: CLIP未安装或加载失败，CLIP损失将回退为MSE")
            _CLIP_AVAILABLE = False
            _CLIP_MODEL = None
    else:
        _CLIP_MODEL = _CLIP_MODEL.to(device).eval()
        _CLIP_MEAN = _CLIP_MEAN.to(device)
        _CLIP_STD = _CLIP_STD.to(device)
    return _CLIP_MODEL


def get_dino_model(device: torch.device):
    global _DINO_MODEL
    if _DINO_MODEL is None:
        try:
            _DINO_MODEL = torch.hub.load(
                repo_or_dir='facebookresearch/dinov3',
                model='dinov3_vitl16',
                source='github',
            )
            for p in _DINO_MODEL.parameters():
                p.requires_grad_(False)
            _DINO_MODEL.eval()
        except Exception:
            print("警告: DINOv3加载失败，DINOv3损失将回退为0")
            _DINO_MODEL = None
    if _DINO_MODEL is not None:
        _DINO_MODEL = _DINO_MODEL.to(device).eval()
    return _DINO_MODEL


def _extract_dino_features_resized(img_chw: torch.Tensor, model, target_q: int, with_grad: bool) -> torch.Tensor:
    bchw = img_chw.unsqueeze(0)
    bchw = F.interpolate(bchw, size=(target_q, target_q), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=bchw.device, dtype=bchw.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=bchw.device, dtype=bchw.dtype).view(1, 3, 1, 1)
    x = (bchw - mean) / std
    if with_grad:
        feats_list = model.get_intermediate_layers(x, n=1, reshape=True, norm=True)
    else:
        with torch.no_grad():
            feats_list = model.get_intermediate_layers(x, n=1, reshape=True, norm=True)
    feats = feats_list[-1].squeeze(0)
    feats = F.normalize(feats, p=2, dim=0)
    return feats


def _one_to_one_greedy_max_matching(similarity: torch.Tensor):
    na, nb = similarity.shape
    work = similarity.clone()
    neg_inf = torch.tensor(-1e9, dtype=work.dtype, device=work.device)
    vals = []
    for _ in range(min(na, nb)):
        v, idx = work.view(-1).max(dim=0)
        if v.item() <= -1e8:
            break
        r = int((idx // nb).item())
        c = int((idx % nb).item())
        vals.append(v)
        work[r, :] = neg_inf
        work[:, c] = neg_inf
    if len(vals) == 0:
        return torch.zeros(0, dtype=similarity.dtype, device=similarity.device)
    return torch.stack(vals)


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
    
    def __init__(self, *args, **kwargs):
        # 是否将输入图像作为condition传入训练中的decoder（基于DINOv2）
        self.decoder_use_image_condition = kwargs.pop('decoder_use_image_condition', False)
        # 是否使用交叉注意力融合（比加性偏置更强）
        self.decoder_use_image_cross_attn = kwargs.pop('decoder_use_image_cross_attn', False)
        # 条件融合强度（初始缩放）
        self.decoder_condition_scale = float(kwargs.pop('decoder_condition_scale', 0.5))
        
        super().__init__(*args, **kwargs)
        
        # 可选：使用输入图像的 DINOv2 语义特征对 decoder 进行条件化
        if getattr(self, 'decoder_use_image_condition', False) and 'decoder' in getattr(self, 'training_models', {}):
            try:
                if not hasattr(self, '_dinov2_model'):
                    import torch
                    print("🔄 加载 DINOv2 模型用于图像条件...")
                    self._dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
                    self._dinov2_model.eval()
                    for p in self._dinov2_model.parameters():
                        p.requires_grad_(False)
                # 包装 decoder
                device = next(self.training_models['decoder'].parameters()).device
                self.training_models['decoder'] = _ImageConditionedDecoderWrapper(
                    base_decoder=self.training_models['decoder'],
                    dinov2_model=self._dinov2_model,
                    use_cross_attn=self.decoder_use_image_cross_attn,
                    init_scale=self.decoder_condition_scale,
                ).to(device)
                print("✅ 已启用使用 DINOv2 的图像条件 Decoder 包装")
                # 调试：标记已启用图像条件
                self._decoder_condition_enabled = True
            except Exception as e:
                print(f"⚠️ 启用图像条件 Decoder 失败: {e}")
        
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
        """训练损失：crop CLIP + DINOv3。"""
        def _to_sparse(v):
            if isinstance(v, sp.SparseTensor):
                return v
            if isinstance(v, torch.Tensor):
                return x_0.replace(v)
            raise TypeError(f"Unsupported sparse input type: {type(v)}")

        if x_0.feats.requires_grad:
            x_0 = x_0.replace(x_0.feats.detach().clone())

        if 'slat' not in kwargs:
            raise ValueError("需要在kwargs中提供 `slat`。")
        loss_feats = _to_sparse(kwargs['slat'])
        target_feats = loss_feats

        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        loss_feats_noisy = self.diffuse(loss_feats, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)

        model_name = list(self.training_models.keys())[0] if self.training_models else 'denoiser'
        model = self.training_models[model_name]
        pred = model(loss_feats_noisy, t * 1000, cond)
        target = self.get_v(target_feats, noise, t)

        total_loss, clip_crop_loss, dinov3_loss = self._compute_clip_dinov3_render_loss(
            pred, x_0, cond, noise=noise, target=target, **kwargs
        )

        terms = edict()
        terms["clip_crop"] = clip_crop_loss
        terms["dinov3"] = dinov3_loss
        terms["loss"] = total_loss

        if hasattr(self, '_debug_step'):
            self._debug_step += 1
        else:
            self._debug_step = 0
        print(
            f"[训练步骤{self._debug_step}] loss={terms['loss'].item():.6f}, "
            f"clip_crop={terms['clip_crop'].item():.6f}, dinov3={terms['dinov3'].item():.6f}"
        )

        return terms, {}

    def _compute_dinov3_match_loss(self, pred_img: torch.Tensor, gt_img: torch.Tensor) -> torch.Tensor:
        """
        DINOv3 细粒度匹配损失。
        输入 pred_img/gt_img: [3,H,W], 值域[0,1]。
        """
        try:
            device = pred_img.device
            dino_model = get_dino_model(device)
            if dino_model is None:
                return torch.tensor(0.0, device=device)

            target_q = int(getattr(self, 'dino_target_q', 224))
            white_bg_mask = bool(getattr(self, 'dino_white_bg_mask', False))

            pred = pred_img.clamp(0, 1)
            gt = gt_img.clamp(0, 1)
            feat_pred = _extract_dino_features_resized(pred, dino_model, target_q=target_q, with_grad=True)
            feat_gt = _extract_dino_features_resized(gt, dino_model, target_q=target_q, with_grad=False)

            c, hp, wp = feat_pred.shape
            _, hg, wg = feat_gt.shape
            pred_flat = feat_pred.permute(1, 2, 0).reshape(-1, c)
            gt_flat = feat_gt.permute(1, 2, 0).reshape(-1, c)
            keep_pred = torch.ones(hp * wp, dtype=torch.bool, device=device)
            keep_gt = torch.ones(hg * wg, dtype=torch.bool, device=device)

            if white_bg_mask:
                thr = 250
                pred_np = (pred.detach().permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                gt_np = (gt.detach().permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                pred_bg = (pred_np[..., 0] >= thr) & (pred_np[..., 1] >= thr) & (pred_np[..., 2] >= thr)
                gt_bg = (gt_np[..., 0] >= thr) & (gt_np[..., 1] >= thr) & (gt_np[..., 2] >= thr)
                pred_fg = torch.from_numpy(~pred_bg).float().unsqueeze(0).unsqueeze(0).to(device)
                gt_fg = torch.from_numpy(~gt_bg).float().unsqueeze(0).unsqueeze(0).to(device)
                keep_pred = F.interpolate(pred_fg, size=(hp, wp), mode='nearest').view(-1) > 0.5
                keep_gt = F.interpolate(gt_fg, size=(hg, wg), mode='nearest').view(-1) > 0.5

            if keep_pred.sum().item() == 0 or keep_gt.sum().item() == 0:
                return torch.tensor(1.0, device=device)

            pred_sel = pred_flat[keep_pred]
            gt_sel = gt_flat[keep_gt]
            sims = pred_sel @ gt_sel.t()
            matched_vals = _one_to_one_greedy_max_matching(sims)
            if matched_vals.numel() == 0:
                return torch.tensor(1.0, device=device)
            return 1.0 - matched_vals.mean()
        except Exception as e:
            print(f"DINOv3损失计算异常: {e}")
            return pred_img.sum() * 0.0

    def _compute_clip_dinov3_render_loss(self, pred, x_0, cond, noise=None, target=None, **kwargs):
        """
        从 3D latent 渲染 2D 图后，仅计算 crop-CLIP + DINOv3。
        """
        device = pred.device
        # Keep zero loss attached to graph to avoid backward failure in fallback path.
        zero = pred.feats.sum() * 0.0
        try:
            if noise is None or target is None:
                noise = torch.randn_like(x_0.feats)
                target = self.get_v(x_0, x_0.replace(noise), self.sample_t(x_0.shape[0]).to(x_0.device).float())

            x0_pred = sp.SparseTensor(feats=noise.feats - pred.feats, coords=x_0.coords)

            def ensure_feat_dim8(st: sp.SparseTensor) -> sp.SparseTensor:
                if st.feats.shape[1] == 8:
                    return st
                if st.feats.shape[1] > 8:
                    return st.replace(st.feats[:, :8])
                padded = torch.zeros(st.feats.shape[0], 8, device=st.feats.device, dtype=st.feats.dtype)
                padded[:, :st.feats.shape[1]] = st.feats
                return st.replace(padded)

            x0_pred = ensure_feat_dim8(x0_pred)
            if not hasattr(self.dataset, 'decode_latent') or not hasattr(self.dataset, '_render_gaussian'):
                return zero, zero, zero

            if hasattr(self, 'train_decoder') and self.train_decoder and 'decoder' in self.training_models:
                use_cond_flag = getattr(self, 'decoder_use_image_condition', False)
                cond_image = kwargs.get('image', None) if use_cond_flag else None
                if isinstance(cond_image, list) and len(cond_image) > 0:
                    cond_image = cond_image[0]
                decoded_pred = self.training_models['decoder'](x0_pred, cond_image) if (use_cond_flag and cond_image is not None) else self.training_models['decoder'](x0_pred)
            else:
                decoded_pred = self.dataset.decode_latent_grad(x0_pred, sample_type="model_generated")
            pred_grid = self.dataset._render_gaussian(decoded_pred).clamp(0, 1)

            gt_grid = None
            gt_img_raw = kwargs.get('gt_image', kwargs.get('gt_image_style', None))
            if isinstance(gt_img_raw, list) and len(gt_img_raw) > 0:
                gt_img_raw = gt_img_raw[0]
            if isinstance(gt_img_raw, Image.Image):
                gt_img = torch.tensor(np.array(gt_img_raw)).permute(2, 0, 1).float() / 255.0
            elif isinstance(gt_img_raw, np.ndarray):
                gt_img = torch.tensor(gt_img_raw).permute(2, 0, 1).float() / 255.0
            elif isinstance(gt_img_raw, torch.Tensor):
                gt_img = gt_img_raw[0] if gt_img_raw.dim() == 4 else gt_img_raw
                gt_img = gt_img.float()
                if gt_img.max() > 1.5:
                    gt_img = gt_img / 255.0
            else:
                gt_img = None

            if gt_img is not None:
                gt_img = gt_img.to(pred_grid.device)
                if gt_img.shape[-2:] != pred_grid.shape[-2:]:
                    gt_img = F.interpolate(gt_img.unsqueeze(0), size=pred_grid.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
                gt_grid = gt_img.clamp(0, 1)

            if gt_grid is None:
                return zero, zero, zero

            clip_crop_loss = self._compute_clip_loss(pred_grid, gt_grid)
            dinov3_loss = self._compute_dinov3_match_loss(pred_grid, gt_grid)
            lambda_clip = float(getattr(self, 'lambda_clip', kwargs.get('lambda_clip', 1.0)))
            fine_match_weight = float(getattr(self, 'fine_match_weight', kwargs.get('fine_match_weight', 1.0)))
            total_loss = lambda_clip * clip_crop_loss + fine_match_weight * dinov3_loss
            return total_loss, clip_crop_loss, dinov3_loss
        except Exception as e:
            print(f"CLIP+DINOv3损失计算失败: {e}")
            return zero, zero, zero
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
        use_random_seed: bool = True,
    ) -> Dict:
        # 设置随机种子以确保每次采样不同的数据
        import time
        import random
        import numpy as np
        
        
        # 使用固定的随机种子（基于步数），确保在numpy允许的范围内
        current_seed = self.step #% (2**32)
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        random.seed(current_seed)
        
      
        
        # 直接使用原dataset（避免deepcopy导致的tensor复制问题）
        dataloader = DataLoader(
            self.dataset,
            batch_size=num_samples,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        sampler = self.get_sampler()
        sample = []
        cond_vis = []
        original_images = []  # 存储原始图像
        ground_truth_images = []  # 存储GT图像
        # 获取一个完整的batch数据
        data = next(iter(dataloader))
        data = {k: v.cuda() if not isinstance(v, list) else v for k, v in data.items()}
        
        # 确保数据量足够
        actual_batch_size = data['x_0'].shape[0] if hasattr(data['x_0'], 'shape') else len(data['x_0'])
        if actual_batch_size < num_samples:
            print(f"警告: 实际batch大小({actual_batch_size})小于请求的样本数({num_samples})")
            num_samples = actual_batch_size
        
        # 只处理前num_samples个样本
        data = {k: v[:num_samples].cuda() if not isinstance(v, list) else v[:num_samples] for k, v in data.items()}
        
        # 生成随机噪声作为采样起点
        if use_random_seed:
            torch.manual_seed((current_seed) % (2**32))  # 使用当前种子
        layout_tensor = data['slat']
        noise = layout_tensor.replace(torch.randn_like(layout_tensor.feats))
        
        #print(f"🔍 采样起点: 使用latent布局({layout_tensor.shape})但随机特征作为noise")
        
        # 保存原始图像（如果存在）
        if 'image' in data:
            original_images.append(data['image'])
        # 保存GT图像（如果存在）
        if 'gt_image' in data:
            ground_truth_images.append(data['gt_image'])
        
        # 确保cond和x_0对应：先获取cond，再删除x_0
        cond_vis.append(self.vis_cond(**data))
        
        # 创建用于推理的数据副本，确保cond和x_0的对应关系
        inference_data = {k: v for k, v in data.items() if k != 'x_0'}
        args = self.get_inference_cond(**inference_data)
     
             
    
        
        self.models['denoiser'].eval()
        res_trained = sampler.sample(
            self.models['denoiser'],
            noise=noise,
            **args,
            steps=50, cfg_strength=3.0, verbose=verbose,
        )
        sample_trained = [res_trained.samples]
        self.models['denoiser'].train()

        # 处理原始数据样本（不是生成的）
        sample_slat_original = sp.sparse_cat([layout_tensor]) if layout_tensor is not None else None
            
        sample_trained = sp.sparse_cat(sample_trained)
        # 采样后使用训练中的decoder进行解码渲染（带可选图像condition），用于与训练路径对齐
        try:
            if hasattr(self, 'training_models') and 'decoder' in self.training_models and hasattr(self.dataset, '_render_gaussian'):
                from torchvision.utils import save_image
                dec = self.training_models['decoder']
                # 取一张条件图像作为condition（若开启decoder_use_image_condition）
                cond_image = None
                if getattr(self, 'decoder_use_image_condition', False) and 'image' in data:
                    cond_image = data['image']
                    if isinstance(cond_image, list) and len(cond_image) > 0:
                        cond_image = cond_image[0]
                # 解码（打印是否使用了condition）
                used_cond = getattr(self, 'decoder_use_image_condition', False) and (cond_image is not None)
                print(f"[run_snapshot] decoder_use_image_condition={getattr(self, 'decoder_use_image_condition', False)}, used_cond={used_cond}")
                decoded_pred = dec(sample_trained, cond_image) if used_cond else dec(sample_trained)
                # 渲染
                pred_grid_sample = self.dataset._render_gaussian(decoded_pred)
                # 保存
                save_root = os.path.join(self.output_dir, 'samples', f'step{self.step:07d}')
                os.makedirs(save_root, exist_ok=True)
                save_image(pred_grid_sample.clamp(0,1), os.path.join(save_root, f'sample_trained_conditioned.png'))
        except Exception:
            pass
        
        # 修改保存逻辑：保存多种样本类型
        sample_dict = {}
        
        # 添加原始数据样本（非生成）
        if sample_slat_original is not None:
            sample_dict['sample_slat'] = {'value': sample_slat_original, 'type': 'sample', 'sample_type': 'original_data_slat'}
        
        # 添加按照当前训练方法生成的样本
        sample_dict['sample_trained'] = {'value': sample_trained, 'type': 'sample', 'sample_type': 'trained_method'}
       
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))

        # 额外保存condition图像
        try:
            save_root = os.path.join(self.output_dir, 'samples', f'step{self.step:07d}')
            os.makedirs(save_root, exist_ok=True)
            
            # 保存条件图像（如果存在）
            if original_images:
                try:
                    # 将条件图像保存为网格
                    from torchvision.utils import save_image
                    cond_images = torch.cat(original_images, dim=0)
                    save_image(cond_images, os.path.join(save_root, f'condition_images_step{self.step:07d}.png'), 
                              nrow=min(4, len(original_images)), normalize=True)
                    print(f"条件图像已保存到: {os.path.join(save_root, f'condition_images_step{self.step:07d}.png')}")
                except Exception as e:
                    print(f"保存条件图像时出错: {e}")
            # 保存GT图像（如果存在）
            if ground_truth_images:
                try:
                    from torchvision.utils import save_image
                    gt_images = torch.cat(ground_truth_images, dim=0)
                    save_image(gt_images, os.path.join(save_root, f'gt_images_step{self.step:07d}.png'),
                               nrow=min(4, len(ground_truth_images)), normalize=True)
                    print(f"GT图像已保存到: {os.path.join(save_root, f'gt_images_step{self.step:07d}.png')}")
                except Exception as e:
                    print(f"保存GT图像时出错: {e}")
        except Exception as e:
            print(f"保存可视化时出错: {e}")
            pass
        
        # 添加更彻底的测试，参考finetune.py
        if self.step%1000 == 0 and self.step>0:
            self._run_thorough_test(sample_dict)
        
        return sample_dict
    

    
    def _v_to_xstart_eps(self, x_t, t, v):
        """从velocity计算x_0和eps（与FlowEulerSampler保持一致）"""
        sigma_min = getattr(self, 'sigma_min', 0.002)
        eps = (1 - t) * v + x_t
        x_0 = (1 - sigma_min) * x_t - (sigma_min + (1 - sigma_min) * t) * v
        return x_0, eps
    
    def _run_thorough_test(self, sample_dict):
        """运行更彻底的测试，参考finetune.py中的测试逻辑"""
        print("=" * 60)
        print("开始运行彻底测试...")
        print("=" * 60)
        
        # 测试路径配置
        test_image_paths = "/home/xinyue_liang/lxy/dreamposible/1w/2_image_gen/test_imgs_paths.txt"
        # 保存到当前训练输出目录下
        results_dir = os.path.join(self.output_dir, 'thorough_test', f'step{self.step:07d}')
        
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
        from trellis.pipelines import TrellisImageTo3DPipeline
        
        # 直接使用当前正在训练的模型，而不是加载checkpoint文件
        print("直接使用当前正在训练的模型进行测试")
        
        # 创建pipeline并直接使用当前训练的模型
        pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "/home/xinyue_liang/.cache/huggingface/hub/models--JeffreyXiang--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/"
        )
        
        # 将当前训练的模型状态注入到pipeline中，确保所有可学习权重都被传递
        if hasattr(self, 'models') and 'denoiser' in self.models:
            current_model = self.models['denoiser']
            pipeline_model = pipeline.models['slat_flow_model']
            
            print(f"🔍 [模型类型检查] 当前训练模型: {type(current_model).__name__}")
            print(f"🔍 [模型类型检查] Pipeline模型: {type(pipeline_model).__name__}")
            
            # 如果pipeline模型类型不匹配，需要替换为相同类型
            if type(current_model).__name__ != type(pipeline_model).__name__:
                print(f"⚠️ 模型类型不匹配，将pipeline模型替换为训练模型类型")
                import copy
                # 深拷贝当前训练模型作为pipeline模型
                new_pipeline_model = copy.deepcopy(current_model)
                # 将原始预训练权重加载到新模型中（除了新增的参数）
                original_state_dict = pipeline_model.state_dict()
                missing_keys, unexpected_keys = new_pipeline_model.load_state_dict(original_state_dict, strict=False)
                print(f"📊 预训练权重加载: 缺失{len(missing_keys)}个键, 多余{len(unexpected_keys)}个键")
                # 然后加载训练权重
                current_state_dict = current_model.state_dict()
                new_pipeline_model.load_state_dict(current_state_dict, strict=True)
                # 替换pipeline中的模型
                pipeline.models['slat_flow_model'] = new_pipeline_model
                pipeline_model = new_pipeline_model
                print(f"✅ 已替换pipeline模型为: {type(pipeline_model).__name__}")
            else:
                # 获取当前训练模型的状态字典
                current_state_dict = current_model.state_dict()
                # 加载状态字典到pipeline模型，使用strict=False以处理新增的参数
                missing_keys, unexpected_keys = pipeline_model.load_state_dict(current_state_dict, strict=False)
            
            # 只有在模型类型匹配时才检查missing/unexpected keys
            if type(current_model).__name__ == type(pipeline.models['slat_flow_model']).__name__:
                if missing_keys:
                    print(f"警告: 以下键在pipeline模型中缺失: {missing_keys}")
                if unexpected_keys:
                    print(f"警告: 以下键在pipeline模型中多余: {unexpected_keys}")
            
            # 验证权重是否正确加载
            print("已将当前训练的模型状态（包括所有可学习权重）注入到pipeline中")
            
            # 检查关键参数是否存在
            if hasattr(pipeline_model, 'ref_gamma_params'):
                if isinstance(pipeline_model.ref_gamma_params, nn.ModuleDict):
                    num_layers = len(pipeline_model.ref_gamma_params)
                    all_params = []
                    for param in pipeline_model.ref_gamma_params.parameters():
                        all_params.append(param.data.view(-1))
                    if all_params:
                        all_params_tensor = torch.cat(all_params)
                        param_min = all_params_tensor.min().item()
                        param_max = all_params_tensor.max().item()
                        total_params = all_params_tensor.numel()
                        print(f"✅ ref_gamma_params MLP已加载: {num_layers}层, 总参数={total_params}, 值范围=[{param_min:.6f}, {param_max:.6f}]")
                    else:
                        print("✅ ref_gamma_params MLP已加载但无参数")
                else:
                    print(f"✅ ref_gamma_params已加载: 形状={pipeline_model.ref_gamma_params.shape}, 值范围=[{pipeline_model.ref_gamma_params.min().item():.6f}, {pipeline_model.ref_gamma_params.max().item():.6f}]")
            else:
                print("⚠️ pipeline模型中没有ref_gamma_params")
            
            # 检查第一个block的权重是否匹配
            current_first_block_weight = current_model.blocks[0].self_attn.to_qkv.weight
            pipeline_first_block_weight = pipeline_model.blocks[0].self_attn.to_qkv.weight
            weights_match = torch.allclose(current_first_block_weight, pipeline_first_block_weight, atol=1e-6)
            print(f"✅ 第一个block权重匹配: {weights_match}")
            
            if not weights_match:
                diff = (current_first_block_weight - pipeline_first_block_weight).abs().max().item()
                print(f"⚠️ 权重差异: {diff:.8f}")
            
            print(f"📊 当前训练模型类型: {type(current_model).__name__}")
            print(f"📊 Pipeline模型类型: {type(pipeline_model).__name__}")
            
            print("✅ 彻底测试使用标准采样路径")
        else:
            print("警告: 当前训练器中没有找到denoiser模型")
        
        pipeline.cuda()
        
        # 复用dataset中已经加载的decoder
        if hasattr(self.dataset, 'ss_dec') and self.dataset.ss_dec is not None:
            # 直接复用dataset中已经加载的decoder
            decoder = self.dataset.ss_dec
            print("复用dataset中已加载的decoder")
            
            # 将decoder替换到pipeline中
            if hasattr(pipeline, 'models') and 'slat_decoder_gs' in pipeline.models:
                pipeline.models['slat_decoder_gs'] = decoder
                print("已将dataset的decoder加载到pipeline中")
            else:
                print("警告: pipeline中没有找到slat_decoder_gs模型位置")
                
        elif hasattr(self.dataset, 'ss_dec_path'):
            # 如果dataset中没有加载decoder，但配置了路径，则加载
            ss_dec_path = self.dataset.ss_dec_path
            print(f"dataset中decoder未加载，使用配置路径: {ss_dec_path}")
            
            # 触发dataset的decoder加载
            self.dataset._loading_ss_dec()
            
            if hasattr(self.dataset, 'ss_dec') and self.dataset.ss_dec is not None:
                decoder = self.dataset.ss_dec
                
                # 将decoder替换到pipeline中
                if hasattr(pipeline, 'models') and 'slat_decoder_gs' in pipeline.models:
                    pipeline.models['slat_decoder_gs'] = decoder
                    print("已将dataset的decoder加载到pipeline中")
                else:
                    print("警告: pipeline中没有找到slat_decoder_gs模型位置")
            else:
                print("dataset加载decoder失败，使用预训练权重")
                from trellis import models
                decoder = models.from_pretrained('microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')
                decoder = decoder.cuda().eval()
                
                # 将decoder替换到pipeline中
                if hasattr(pipeline, 'models') and 'slat_decoder_gs' in pipeline.models:
                    pipeline.models['slat_decoder_gs'] = decoder
                    print("已将预训练decoder加载到pipeline中")
                else:
                    print("警告: pipeline中没有找到sparse_structure_decoder模型位置")
                
        else:
            print("警告: dataset中没有找到decoder配置信息")
            
        # 对每个测试图像进行推理
        for image_path in image_paths:
            print(f"处理图像: {image_path}")
            
            # 提取文件名（不包含扩展名）
            image_filename = os.path.splitext(os.path.basename(image_path))[0]
            
            # 加载和预处理图像
            from PIL import Image
            image = Image.open(image_path)
            image = pipeline.preprocess_image(image)
            
            # 运行推理
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
            
            print("=" * 60)
            print(f"彻底测试完成！结果保存在: {results_dir}")
            print("=" * 60)
    
    def _compute_clip_loss(self, pred_img: torch.Tensor, gt_img: torch.Tensor) -> torch.Tensor:
        """中心细节裁剪 CLIP 图像-图像损失（无全局项，提效聚焦重心）
        - 输入: pred_img/gt_img [3,H,W], 值域[0,1]（四宫格图像）
        - 每个宫格一次中心裁剪（固定比例）-> 224 -> CLIP
        - 允许梯度回传到pred_img，CLIP参数仍冻结
        """
        try:
            device = pred_img.device
            clip_model = get_clip_model(device)
            if clip_model is None:
                return F.mse_loss(pred_img, gt_img)

            active_quadrants = [0, 1, 2, 3]
            H, W = pred_img.shape[-2:]
            h, w = H // 2, W // 2

            # 随机多裁剪参数
            num_crops = 6
            min_scale, max_scale = 0.4, 1.0

            losses = []
            for quadrant_idx in active_quadrants:
                row = quadrant_idx // 2
                col = quadrant_idx % 2
                pred_quad = pred_img[:, row*h:(row+1)*h, col*w:(col+1)*w].unsqueeze(0).clamp(0, 1)
                gt_quad = gt_img[:, row*h:(row+1)*h, col*w:(col+1)*w].unsqueeze(0).clamp(0, 1)

                _, _, qH, qW = pred_quad.shape
                for _ in range(num_crops):
                    scale = float(torch.empty(1).uniform_(min_scale, max_scale).item())
                    ch = max(32, int(qH * scale))
                    cw = max(32, int(qW * scale))
                    ch = min(ch, qH)
                    cw = min(cw, qW)
                    y0 = 0 if qH == ch else int(torch.randint(0, qH - ch + 1, (1,)).item())
                    x0 = 0 if qW == cw else int(torch.randint(0, qW - cw + 1, (1,)).item())
                    v_crop = pred_quad[:, :, y0:y0+ch, x0:x0+cw]
                    g_crop = gt_quad[:, :, y0:y0+ch, x0:x0+cw]
                    v_r = F.interpolate(v_crop, size=(224, 224), mode='bilinear', align_corners=False)
                    g_r = F.interpolate(g_crop, size=(224, 224), mode='bilinear', align_corners=False)
                    v_n = (v_r - _CLIP_MEAN.to(v_r.dtype)) / _CLIP_STD.to(v_r.dtype)
                    g_n = (g_r - _CLIP_MEAN.to(g_r.dtype)) / _CLIP_STD.to(g_r.dtype)
                    f_v = clip_model.encode_image(v_n)
                    f_g = clip_model.encode_image(g_n)
                    f_v = F.normalize(f_v, dim=-1)
                    f_g = F.normalize(f_g, dim=-1)
                    sim = (f_v * f_g).sum(dim=-1)
                    loss_crop = 1.0 - sim.mean()
                    losses.append(loss_crop.view([]))

            if len(losses) == 0:
                return F.mse_loss(pred_img, gt_img)
            loss = torch.stack(losses).mean()
            return loss
        except Exception as e:
            print(f"CLIP损失计算异常: {e}")
            return F.mse_loss(pred_img, gt_img)

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


class _ImageConditionedDecoderWrapper(torch.nn.Module):
    """将 DINOv2 的图像全局特征融合到稀疏 latent 上，再调用原始 decoder。
    - 冻结 DINOv2，只提取全局特征（[1024]）
    - 小型 MLP 将全局特征映射到 latent 维度，加性融合
    - 透明包装：保持原 decoder 的接口
    """
    def __init__(self, base_decoder: torch.nn.Module, dinov2_model):
        super().__init__()
        self.base_decoder = base_decoder
        self.dinov2 = dinov2_model
        # 估计 latent 维度
        latent_dim = 8
        try:
            if hasattr(base_decoder, 'input_layer') and hasattr(base_decoder.input_layer, 'in_features'):
                latent_dim = int(base_decoder.input_layer.in_features)
        except Exception:
            pass
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, latent_dim),
        )
        self.register_buffer('_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, latent_sparse, image_tensor=None):
        if image_tensor is None:
            return self.base_decoder(latent_sparse)
        try:
            img = image_tensor
            if isinstance(img, list) and len(img) > 0:
                img = img[0]
            if isinstance(img, Image.Image):
                import numpy as np
                img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            if img.dim() == 3:
                img = img.unsqueeze(0)
            img = img.to(next(self.base_decoder.parameters()).device).float()
            if img.max() > 1.5:
                img = img / 255.0
            img_n = (img - self._mean.to(img.dtype)) / self._std.to(img.dtype)
            with torch.no_grad():
                feats = self.dinov2(img_n, is_training=True)
            x_prenorm = feats['x_prenorm']  # [B, T, C]
            num_reg = int(getattr(self.dinov2, 'num_register_tokens', 0))
            patch_tokens = x_prenorm[:, num_reg + 1:]
            global_vec = patch_tokens.mean(dim=1)  # [B, 1024]
            bias = self.proj(global_vec)  # [B, latent_dim]
            fused_feats = latent_sparse.feats + bias[:1].repeat(latent_sparse.feats.shape[0], 1)
            latent_cond = latent_sparse.replace(fused_feats)
            return self.base_decoder(latent_cond)
        except Exception:
            return self.base_decoder(latent_sparse)
