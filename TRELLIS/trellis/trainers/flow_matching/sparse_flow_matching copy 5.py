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

# 可选导入kornia用于SIFT损失
try:
    import kornia
    KORNIA_AVAILABLE = True
except Exception:
    KORNIA_AVAILABLE = False

# 可选导入vision_aided_loss用于GAN损失
try:
    import vision_aided_loss
    VISION_AIDED_LOSS_AVAILABLE = True
except ImportError:
    print("警告: vision_aided_loss未安装，将跳过GAN loss计算")
    VISION_AIDED_LOSS_AVAILABLE = False

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
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化GAN判别器
        if hasattr(self, 'device'):
            self.init_gan_discriminator(self.device)
        else:
            # 如果没有device属性，使用第一个模型的设备
            device = next(self.models.values()).device
            self.init_gan_discriminator(device)
    
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
    
    def init_gan_discriminator(self, device):
        """初始化GAN判别器，用于GAN loss计算"""
        try:
            if VISION_AIDED_LOSS_AVAILABLE:
                # 按照官方用法初始化判别器
                self.net_disc = vision_aided_loss.Discriminator(
                    cv_type='clip', 
                    loss_type="multilevel_sigmoid_s", 
                    device=device
                ).to(device)
                
                # 冻结特征提取器（官方用法）
                self.net_disc.cv_ensemble.requires_grad_(False)
                
                # 为判别器创建优化器
                self.disc_optimizer = torch.optim.Adam(
                    self.net_disc.parameters(),
                    lr=4e-4,  # 判别器学习率通常比生成器小
                    betas=(0.5, 0.999)
                )
                
                print("成功初始化vision_aided_loss判别器用于GAN损失")
                print(f"判别器优化器: Adam(lr=1e-4, betas=(0.5, 0.999))")
                print("特征提取器已冻结")
            else:
                print("警告: vision_aided_loss不可用，无法初始化GAN判别器")
                self.net_disc = None
                self.disc_optimizer = None
        except Exception as e:
            print(f"警告: 初始化GAN判别器失败: {e}")
            print("GAN损失将被跳过")
            self.net_disc = None
            self.disc_optimizer = None
    
    def enable_dpo_mode(self, ref_model, dpo_beta=1.0, sample_same_epsilon=True):
        """
        启用DPO训练模式
        
        Args:
            ref_model: 参考模型
            dpo_beta: DPO损失函数的beta参数
            sample_same_epsilon: 是否使用相同的epsilon
        """
        self.use_dpo = True
        self.ref_model = ref_model
        self.dpo_beta = dpo_beta
        self.sample_same_epsilon = sample_same_epsilon
        print(f"已启用DPO训练模式: beta={dpo_beta}, sample_same_epsilon={sample_same_epsilon}")
    
    def disable_dpo_mode(self):
        """禁用DPO训练模式，回到常规Flow Matching"""
        self.use_dpo = False
        self.ref_model = None
        print("已禁用DPO训练模式，回到常规Flow Matching")
        
    def training_losses_dpo(
        self,
        model,
        ref_model,
        x0_win,
        x0_loss,
        cond=None,
        beta=1.0,
        sample_same_epsilon=True,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算DPO训练损失，包括model和ref_model的loss
        返回: (model_loss, ref_model_loss, combined_loss)
        """
        # Handle SparseTensor lists (参考finetune.py)
        if isinstance(x0_win, list):
            model_losses = []
            ref_losses = []
            combined_losses = []
            
            for i in range(len(x0_win)):
                single_x0_win = x0_win[i]
                single_x0_loss = x0_loss[i]
                single_t = self.sample_t(1).to(single_x0_win.device).float()
                single_cond = cond[i:i+1] if isinstance(cond, torch.Tensor) else cond
                
                eps_win = torch.randn_like(single_x0_win.feats)
                eps_loss = torch.randn_like(single_x0_loss.feats)
                
                # Model forward pass
                xt_win = (1 - single_t.view(-1, 1)) * single_x0_win.feats + single_t.view(-1, 1) * eps_win
                xt_loss = (1 - single_t.view(-1, 1)) * single_x0_loss.feats + single_t.view(-1, 1) * eps_loss
                
                target_win = eps_win - single_x0_win.feats
                target_loss = eps_loss - single_x0_loss.feats
                
                xt_win_sparse = sp.SparseTensor(feats=xt_win, coords=single_x0_win.coords)
                xt_loss_sparse = sp.SparseTensor(feats=xt_loss, coords=single_x0_loss.coords)
                
                pred_win = model(xt_win_sparse, single_t * 1000, single_cond, **kwargs)
                pred_loss = model(xt_loss_sparse, single_t * 1000, single_cond, **kwargs)
                
                loss_w = (pred_win.feats - target_win).pow(2).mean()
                loss_l = (pred_loss.feats - target_loss).pow(2).mean()
                model_diff = loss_w - loss_l
                
                # Ref model forward pass (with no_grad for efficiency)
                with torch.no_grad():
                    if sample_same_epsilon:
                        eps_win_ref = eps_win
                        eps_loss_ref = eps_loss
                    else:
                        eps_win_ref = torch.randn_like(single_x0_win.feats)
                        eps_loss_ref = torch.randn_like(single_x0_loss.feats)
                    
                    xt_win_ref = (1 - single_t.view(-1, 1)) * single_x0_win.feats + single_t.view(-1, 1) * eps_win_ref
                    xt_loss_ref = (1 - single_t.view(-1, 1)) * single_x0_loss.feats + single_t.view(-1, 1) * eps_loss_ref
                    
                    target_win_ref = eps_win_ref - single_x0_win.feats
                    target_loss_ref = eps_loss_ref - single_x0_loss.feats
                    
                    xt_win_ref_sparse = sp.SparseTensor(feats=xt_win_ref, coords=single_x0_win.coords)
                    xt_loss_ref_sparse = sp.SparseTensor(feats=xt_loss_ref, coords=single_x0_loss.coords)
                    
                    pred_win_ref = ref_model(xt_win_ref_sparse, single_t * 1000, single_cond, **kwargs)
                    pred_loss_ref = ref_model(xt_loss_ref_sparse, single_t * 1000, single_cond, **kwargs)
                    
                    loss_w_ref = (pred_win_ref.feats - target_win_ref).pow(2).mean()
                    loss_l_ref = (pred_loss_ref.feats - target_loss_ref).pow(2).mean()
                    ref_diff = loss_w_ref - loss_l_ref
                
                # 计算DPO损失
                improvement = model_diff - ref_diff
                inside_term = -beta * improvement
                combined_loss = -F.logsigmoid(inside_term)
                
                model_losses.append(model_diff)
                ref_losses.append(ref_diff)
                combined_losses.append(combined_loss)
            
            return (torch.stack(model_losses).mean(), 
                    torch.stack(ref_losses).mean(), 
                    torch.stack(combined_losses).mean())
        else:
            # 单个SparseTensor的情况（保持原有逻辑）
            # 确保输入是独立的tensor，避免重复计算图
            if x0_win.feats.requires_grad:
                x0_win = x0_win.replace(x0_win.feats.detach().clone())
            if x0_loss.feats.requires_grad:
                x0_loss = x0_loss.replace(x0_loss.feats.detach().clone())
            
            # 生成噪声
            eps_win = torch.randn_like(x0_win.feats)
            eps_loss = torch.randn_like(x0_loss.feats)
            
            # 采样时间步
            t = self.sample_t(x0_win.shape[0]).to(x0_win.device).float()
            
            # 扩散过程
            x_t_win = self.diffuse(x0_win, t, noise=x0_win.replace(eps_win))
            x_t_loss = self.diffuse(x0_loss, t, noise=x0_loss.replace(eps_loss))
            
            # 获取条件
            cond = self.get_cond(cond, **kwargs)
            
            # Model forward pass
            pred_win = model(x_t_win, t * 1000, cond, **kwargs)
            pred_loss = model(x_t_loss, t * 1000, cond, **kwargs)
            
            # 计算目标
            target_win = self.get_v(x0_win, x0_win.replace(eps_win), t)
            target_loss = self.get_v(x0_loss, x0_loss.replace(eps_loss), t)
            
            # 计算损失
            loss_w = F.mse_loss(pred_win.feats, target_win.feats)
            loss_l = F.mse_loss(pred_loss.feats, target_loss.feats)
            model_diff = loss_w - loss_l
            
            # Ref model forward pass (with no_grad for efficiency)
            with torch.no_grad():
                if sample_same_epsilon:
                    eps_win_ref = eps_win
                    eps_loss_ref = eps_loss
                else:
                    eps_win_ref = torch.randn_like(x0_win.feats)
                    eps_loss_ref = torch.randn_like(x0_loss.feats)
                
                x_t_win_ref = self.diffuse(x0_win, t, noise=x0_win.replace(eps_win_ref))
                x_t_loss_ref = self.diffuse(x0_loss, t, noise=x0_loss.replace(eps_loss_ref))
                
                target_win_ref = self.get_v(x0_win, x0_win.replace(eps_win_ref), t)
                target_loss_ref = self.get_v(x0_loss, x0_loss.replace(eps_loss_ref), t)
                
                pred_win_ref = ref_model(x_t_win_ref, t * 1000, cond, **kwargs)
                pred_loss_ref = ref_model(x_t_loss_ref, t * 1000, cond, **kwargs)
                
                loss_w_ref = F.mse_loss(pred_win_ref.feats, target_win_ref.feats)
                loss_l_ref = F.mse_loss(pred_loss_ref.feats, target_loss_ref.feats)
                ref_diff = loss_w_ref - loss_l_ref
            
            # 计算DPO损失
            improvement = model_diff - ref_diff
            inside_term = -beta * improvement
            combined_loss = -F.logsigmoid(inside_term)
            
            return model_diff, ref_diff, combined_loss

    def _compute_dpo_loss_simple(self, x_0, cond, beta=50.0, sample_same_epsilon=True, **kwargs):
        """
        简化的DPO损失计算，直接使用x_0
        """
        # 获取当前使用的模型名称
        model_name = list(self.training_models.keys())[0] if self.training_models else 'denoiser'
        model = self.training_models[model_name]
        ref_model = self.ref_model
        
        # 生成噪声
        eps = torch.randn_like(x_0.feats)
        
        # 采样时间步
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        
        # 扩散过程
        x_t = self.diffuse(x_0, t, noise=x_0.replace(eps))
        
        # 获取条件
        cond = self.get_cond(cond, **kwargs)
        
        # Model forward pass
        pred = model(x_t, t * 1000, cond, **kwargs)
        
        # 计算目标
        target = self.get_v(x_0, x_0.replace(eps), t)
        
        # 计算模型损失
        model_loss = F.mse_loss(pred.feats, target.feats)
        
        # Ref model forward pass (with no_grad for efficiency)
        with torch.no_grad():
            if sample_same_epsilon:
                eps_ref = eps
            else:
                eps_ref = torch.randn_like(x_0.feats)
            
            x_t_ref = self.diffuse(x_0, t, noise=x_0.replace(eps_ref))
            target_ref = self.get_v(x_0, x_0.replace(eps_ref), t)
            
            pred_ref = ref_model(x_t_ref, t * 1000, cond, **kwargs)
            ref_loss = F.mse_loss(pred_ref.feats, target_ref.feats)
        
        # 计算DPO损失
        improvement = model_loss - ref_loss
        inside_term = -beta * improvement
        dpo_loss = -F.logsigmoid(inside_term)
        
        return model_loss, ref_loss, dpo_loss

    def training_losses(
        self,
        x_0: sp.SparseTensor,
        cond=None,
        use_dpo=False,
        ref_model=None,
        x0_win=None,
        x0_loss=None,
        dpo_beta=1.0,
        sample_same_epsilon=True,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x ... x C] sparse tensor of the inputs.
            cond: The [N x ...] tensor of additional conditions.
            use_dpo: Whether to use DPO training instead of regular flow matching.
            ref_model: Reference model for DPO training.
            x0_win: Winning samples for DPO training.
            x0_loss: Losing samples for DPO training.
            dpo_beta: Beta parameter for DPO loss.
            sample_same_epsilon: Whether to use same epsilon for ref model.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        # DPO模式开关
        print(f"DPO检查: use_dpo={use_dpo}, ref_model={ref_model is not None}")
        if use_dpo and ref_model is not None:
            # 使用DPO训练
            # 获取当前使用的模型名称
            model_name = list(self.training_models.keys())[0] if self.training_models else 'denoiser'
            
            # 直接使用x_0进行DPO训练
            model_loss, ref_loss, dpo_loss = self._compute_dpo_loss_simple(
                x_0=x_0,
                cond=cond,
                beta=dpo_beta,
                sample_same_epsilon=sample_same_epsilon,
                **kwargs
            )
            
            terms = edict()
            terms["model_diff"] = model_loss
            terms["ref_diff"] = ref_loss
            terms["dpo_loss"] = dpo_loss
            terms["loss"] = dpo_loss
            
            # 可选：添加调试信息
            if hasattr(self, '_debug_step'):
                self._debug_step += 1
            else:
                self._debug_step = 0
            
            #if self._debug_step % 100 == 0:
            print(f"[DPO训练步骤{self._debug_step}] 损失统计:")
            print(f"  Model loss: {model_loss.item():.6f}")
            print(f"  Ref loss: {ref_loss.item():.6f}")
            print(f"  Improvement: {(model_loss - ref_loss).item():.6f}")
            print(f"  DPO loss: {dpo_loss.item():.6f}")
            
            # 在DPO模式下也计算多种2D损失函数
            try:
                # 获取当前使用的模型名称
                model_name = list(self.training_models.keys())[0] if self.training_models else 'denoiser'
                model = self.training_models[model_name]
                
                # 生成噪声和时间步
                eps = torch.randn_like(x_0.feats)
                t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
                x_t = self.diffuse(x_0, t, noise=x_0.replace(eps))
                cond_processed = self.get_cond(cond, **kwargs)
                
                # 模型前向传播
                pred = model(x_t, t * 1000, cond_processed, **kwargs)
                target = self.get_v(x_0, x_0.replace(eps), t)
                
                # DPO模式下的损失权重配置（权重为0的损失函数不会计算）
                # 先用两个最重要的损失函数测试（LPIPS + MSE）
                dpo_loss_weights = {
                    'lpips': 10,      # 感知损失，对真实感最重要
                    'sift': 0.0,       # SIFT特征损失（设为0减少计算量）
                    'mse': 1,        # 均方误差，保持像素级准确性
                    'clip': 0.0,       # CLIP Score损失（设为0减少计算量）
                    'ssim': 0.0        # 结构相似性损失（设为0减少计算量）
                }
                
                # 计算多种2D损失函数（只计算权重大于0的）
                lpips_loss = self._compute_lpips_loss(pred, x_0, cond, noise=x_0.replace(eps), target=target, **kwargs)
                additional_losses = self._compute_additional_2d_losses(pred, x_0, cond, noise=x_0.replace(eps), target=target, loss_weights=dpo_loss_weights, **kwargs)
                
                # 计算总损失（只计算权重大于0的损失）
                total_additional_loss = 0.0
                if lpips_loss is not None and dpo_loss_weights.get('lpips', 0) > 0:
                    terms["lpips"] = lpips_loss
                    total_additional_loss += dpo_loss_weights['lpips'] * lpips_loss
                    print(f"  LPIPS loss: {dpo_loss_weights['lpips'] * lpips_loss.item():.6f}")
                
                for loss_name, weight in dpo_loss_weights.items():
                    if weight <= 0:
                        continue  # 跳过权重为0的损失
                    if loss_name in additional_losses:
                        total_additional_loss += weight * additional_losses[loss_name]
                        print(f"  {loss_name.upper()} loss: {weight * additional_losses[loss_name].item():.6f}")
                
                terms["loss"] = terms["loss"] + total_additional_loss
                print(f"  总额外损失: {total_additional_loss.item():.6f}")
                
            except Exception as e:
                print(f"  2D损失计算异常 - {e}")
            
            return terms, {}
        
        # 常规Flow Matching模式
        # 确保x_0是独立的tensor，避免重复计算图
        if x_0.feats.requires_grad:
            x_0 = x_0.replace(x_0.feats.detach().clone())
        
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)
        
        # 获取当前使用的模型名称
        model_name = list(self.training_models.keys())[0] if self.training_models else 'denoiser'
        pred = self.training_models[model_name](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]
        
        # 计算多种2D损失函数，避免生成白色/模糊结果
        lpips_loss = self._compute_lpips_loss(pred, x_0, cond, noise=noise, target=target, **kwargs)
        if lpips_loss is not None:
            terms["lpips"] = lpips_loss
            
            # 组合5种损失函数：LPIPS、SIFT、MSE、CLIP Score、SSIM
            # 权重为0的损失函数不会计算，减少计算量
            # 简化损失函数组合
            loss_weights = {

                'lpips': 1.0,      # 感知损失，对纹理和真实感最重要
                'sift': 0.0,       # SIFT特征损失（几何严格对应，暂时关闭）
                'mse': 1.0,        # 轻微像素级约束，保持基本准确性
                'clip': 0.0,       # CLIP语义损失，关注纹理语义
                'ssim': 0.,       # 结构相似性损失（几何严格对应，暂时关闭）
                'gan': 0.0        # GAN损失，提高真实感
            }
            
            # 计算额外的2D损失函数（只计算权重大于0的）
            additional_losses = self._compute_additional_2d_losses(pred, x_0, cond, noise=noise, target=target, loss_weights=loss_weights, **kwargs)
            
            # 计算总损失（只计算权重大于0的损失）
            total_loss = 0.0
            for loss_name, weight in loss_weights.items():
                if weight <= 0:
                    continue  # 跳过权重为0的损失
                    
                if loss_name == 'lpips' and lpips_loss is not None:
                    total_loss += weight * lpips_loss
                    print(f"  {loss_name.upper()} Loss: {weight * lpips_loss.item():.6f}")
                elif loss_name in additional_losses:
                    total_loss += weight * additional_losses[loss_name]
                    print(f"  {loss_name.upper()} Loss: {weight * additional_losses[loss_name].item():.6f}")
                elif loss_name == 'mse':
                    total_loss += weight * terms["mse"]
                    print(f"  {loss_name.upper()} Loss: {weight * terms['mse'].item():.6f}")
            
            terms["loss"] = total_loss
            print(f"  总损失: {total_loss.item():.6f}")
        else:
            print(f"  LPIPS loss: 计算失败")
        # 调试：检查loss和预测值的统计信息
        if hasattr(self, '_debug_step'):
            self._debug_step += 1
        else:
            self._debug_step = 0
        
        # 每步都打印基本信息
        print(f"[训练步骤{self._debug_step}] Loss统计:")
        print(f"  MSE Loss: {terms['mse'].item():.6f}")
        if "lpips" in terms:
            print(f"  LPIPS Loss: {terms['lpips'].item():.6f}")
        
        # 每100步打印详细信息
        if self._debug_step % 100 == 0:
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
        
      
        
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=num_samples,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        sampler = self.get_sampler()
        sample_gt = []
        sample = []
        sample_ref = []  # 参考模型的样本
        cond_vis = []
        original_images = []  # 存储原始图像
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
        
        # 为每次迭代生成不同的噪声
        if use_random_seed:
            torch.manual_seed((current_seed) % (2**32))  # 使用当前种子
        noise = data['x_0'].replace(torch.randn_like(data['x_0'].feats))
        sample_gt.append(data['x_0'])
        
        # 保存原始图像（如果存在）
        if 'image' in data:
            original_images.append(data['image'])
        
        # 确保cond和x_0对应：先获取cond，再删除x_0
        cond_vis.append(self.vis_cond(**data))
        
        # 创建用于推理的数据副本，确保cond和x_0的对应关系
        inference_data = {k: v for k, v in data.items() if k != 'x_0'}
        args = self.get_inference_cond(**inference_data)
     
             
        # 使用当前训练的模型生成样本
        res = sampler.sample(
            self.models['denoiser'],
            noise=noise,
            **args,
            steps=50, cfg_strength=3.0, verbose=verbose,
        )
        sample.append(res.samples)
        
        # 使用参考模型生成样本（如果存在）
        if hasattr(self, 'ref_model') and self.ref_model is not None:
            res_ref = sampler.sample(
                self.ref_model,
                noise=noise,
                **args,
                steps=50, cfg_strength=3.0, verbose=verbose,
            )
            sample_ref.append(res_ref.samples)
        else:
            # 如果没有参考模型，使用相同的噪声作为占位符
            sample_ref.append(noise)

        sample_gt = sp.sparse_cat(sample_gt)
        sample = sp.sparse_cat(sample)
        sample_ref = sp.sparse_cat(sample_ref)
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample', 'sample_type': 'ground_truth'},
            'sample': {'value': sample, 'type': 'sample', 'sample_type': 'model_generated'},
            'sample_ref': {'value': sample_ref, 'type': 'sample', 'sample_type': 'reference_model'},
        }
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))

        # 额外保存latent的PCA可视化和condition图像
        try:
            save_root = os.path.join(self.output_dir, 'samples', f'step{self.step:07d}')
            os.makedirs(save_root, exist_ok=True)
            # GT PCA
            gt_pca_img = self.dataset._visualize_latent_pca(sample_gt)
            gt_np = (gt_pca_img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(gt_np).save(os.path.join(save_root, f'pca_sample_gt_step{self.step:07d}.png'))
            # Sample PCA (当前训练的模型)
            pred_pca_img = self.dataset._visualize_latent_pca(sample)
            pred_np = (pred_pca_img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(pred_np).save(os.path.join(save_root, f'pca_sample_step{self.step:07d}.png'))
            # Reference Sample PCA (参考模型)
            if hasattr(self, 'ref_model') and self.ref_model is not None:
                ref_pca_img = self.dataset._visualize_latent_pca(sample_ref)
                ref_np = (ref_pca_img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(ref_np).save(os.path.join(save_root, f'pca_sample_ref_step{self.step:07d}.png'))
            
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
        except Exception as e:
            print(f"保存可视化时出错: {e}")
            pass
        
        # 添加更彻底的测试，参考finetune.py
        if self.step%300 == 0: #and self.step>0:
            self._run_thorough_test(sample_dict)
        
        return sample_dict
    
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
        pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        pipeline.cuda()
        
        # 将当前训练的模型替换到pipeline中
        if 'denoiser' in self.models:
            pipeline.models["slat_flow_model"] = self.models['denoiser']
            print("已将训练的denoiser模型加载到pipeline中")
        else:
            print("警告: 未找到denoiser模型")
            return
        
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

            # 将四宫格分割成4个单独的宫格，分别计算LPIPS loss
            # 可以通过修改这个列表来控制哪些宫格参与LPIPS计算
            active_quadrants = [0, 1, 2, 3]  # 0:左上, 1:右上, 2:左下, 3:右下
            
            # 获取单个宫格的尺寸
            H, W = pred_grid.shape[-2:]
            h, w = H // 2, W // 2
            
            lpips_losses = []
            
            for quadrant_idx in active_quadrants:
                # 计算宫格位置
                row = quadrant_idx // 2  # 0或1
                col = quadrant_idx % 2   # 0或1
                
                # 提取对应的宫格
                pred_quadrant = pred_grid[:, row*h:(row+1)*h, col*w:(col+1)*w]
                gt_quadrant = gt_grid[:, row*h:(row+1)*h, col*w:(col+1)*w]
                
                # LPIPS期望[-1,1]
                pred_input = (pred_quadrant.unsqueeze(0) * 2 - 1).clamp(-1, 1)
                gt_input = (gt_quadrant.unsqueeze(0) * 2 - 1).clamp(-1, 1)
                
                with torch.cuda.amp.autocast(enabled=False):
                    quadrant_loss = self._lpips_model(pred_input, gt_input).mean()
                    if quadrant_idx==2:
                        quadrant_loss*=10.0
                    else:
                        quadrant_loss*=0.0
                    lpips_losses.append(quadrant_loss)
                    
            
            # 计算平均LPIPS loss
            if lpips_losses:
                lpips_loss = torch.stack(lpips_losses).mean()
                print(f"  LPIPS loss (宫格{active_quadrants}): {lpips_loss.item():.6f}")
            else:
                lpips_loss = torch.tensor(0.0, device=pred_grid.device)
                print(f"  LPIPS loss: 未计算（无活动宫格）")

            # 调试：前几次打印关键可导性信息
           
            # 可选：叠加SIFT损失（开关：self.sift_weight > 0）
            sift_weight = float(getattr(self, 'sift_weight', 0.0))
            if sift_weight > 0.0:
                # 不要detach，保持可导
                sift_loss = self._compute_sift_loss(pred_grid, gt_grid)
                # 使用相同设备
                sift_loss = sift_loss.to(lpips_loss.device)
                print(sift_loss)
                return lpips_loss#*0.0 + sift_weight * sift_loss
            else:
                return lpips_loss
            
        except Exception as e:
            print(f"LPIPS loss计算失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _compute_additional_2d_losses(self, pred, x_0, cond, noise=None, target=None, loss_weights=None, **kwargs):
        """
        计算额外的2D损失函数，包括SIFT、SSIM、CLIP等
        只计算权重大于0的损失函数，减少计算量
        返回: 包含各种损失的字典
        """
        additional_losses = {}
        
        # 如果没有提供权重配置，使用默认配置
        if loss_weights is None:
            loss_weights = {
                'sift': 1.0,
                'ssim': 1.0,
                'clip': 1.0
            }
        
        try:
            # 获取渲染的图像用于计算2D损失
            if noise is None or target is None:
                noise = torch.randn_like(x_0.feats)
                target = self.get_v(x_0, x_0.replace(noise), self.sample_t(x_0.shape[0]).to(x_0.device).float())

            # 重构x0
            x0_pred_feats = noise.feats - pred.feats
            x0_pred = sp.SparseTensor(feats=x0_pred_feats, coords=x_0.coords)
            
            # 确保特征维度为8
            def ensure_feat_dim8(st: sp.SparseTensor) -> sp.SparseTensor:
                if st.feats.shape[1] == 8:
                    return st
                if st.feats.shape[1] > 8:
                    return st.replace(st.feats[:, :8])
                padded = torch.zeros(st.feats.shape[0], 8, device=st.feats.device, dtype=st.feats.dtype)
                padded[:, :st.feats.shape[1]] = st.feats
                return st.replace(padded)

            x0_pred = ensure_feat_dim8(x0_pred)
            
            # 解码并渲染预测的x0
            if hasattr(self.dataset, 'decode_latent') and hasattr(self.dataset, '_render_gaussian'):
                decoded_pred = self.dataset.decode_latent_grad(x0_pred, sample_type="model_generated")
                pred_grid = self.dataset._render_gaussian(decoded_pred)  # [3,H,W]
                
                # 获取GT图像
                gt_grid = None
                if 'gt_image' in kwargs and kwargs['gt_image'] is not None:
                    gt_img_raw = kwargs['gt_image']
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
                        if gt_img.shape[-2:] != pred_grid.shape[-2:]:
                            gt_img = F.interpolate(gt_img.unsqueeze(0), size=pred_grid.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
                        gt_grid = gt_img.clamp(0, 1)
                
                if pred_grid is not None and gt_grid is not None:
                    # 只计算权重大于0的损失函数
                    
                    # 计算SIFT损失（如果权重>0）
                    if loss_weights.get('sift', 0) > 0:
                        additional_losses['sift'] = self._compute_sift_loss(pred_grid, gt_grid)
                    
                    # 计算SSIM损失（如果权重>0）
                    if loss_weights.get('ssim', 0) > 0:
                        additional_losses['ssim'] = self._compute_ssim_loss(pred_grid, gt_grid)
                    
                    # 计算CLIP Score损失（如果权重>0）
                    if loss_weights.get('clip', 0) > 0:
                        additional_losses['clip'] = self._compute_clip_loss(pred_grid, gt_grid)
                    
                    # 计算GAN损失（如果权重>0）
                    if loss_weights.get('gan', 0) > 0:
                        additional_losses['gan'] = self._compute_gan_loss(pred_grid, gt_grid)
                    
        except Exception as e:
            print(f"计算额外2D损失时出错: {e}")
            # 提供默认值（只包含权重大于0的损失）
            device = pred.device
            additional_losses = {}
            if loss_weights.get('sift', 0) > 0:
                additional_losses['sift'] = torch.tensor(0.1, device=device, requires_grad=True)
            if loss_weights.get('ssim', 0) > 0:
                additional_losses['ssim'] = torch.tensor(0.1, device=device, requires_grad=True)
            if loss_weights.get('clip', 0) > 0:
                additional_losses['clip'] = torch.tensor(0.1, device=device, requires_grad=True)
            if loss_weights.get('gan', 0) > 0:
                additional_losses['gan'] = torch.tensor(0.1, device=device, requires_grad=True)
        
        return additional_losses

    def _compute_ssim_loss(self, pred_img: torch.Tensor, gt_img: torch.Tensor) -> torch.Tensor:
        """计算SSIM损失 - 四宫格分别计算
        输入pred_img/gt_img: [3, H, W], 值域[0,1]（四宫格图像）
        返回: 标量loss张量
        """
        try:
            # 按四宫格分别计算SSIM损失
            active_quadrants = [0, 1, 2, 3]  # 0:左上, 1:右上, 2:左下, 3:右下
            
            # 获取单个宫格的尺寸
            H, W = pred_img.shape[-2:]
            h, w = H // 2, W // 2
            
            ssim_losses = []
            
            for quadrant_idx in active_quadrants:
                # 计算宫格位置
                row = quadrant_idx // 2  # 0或1
                col = quadrant_idx % 2   # 0或1
                
                # 提取对应的宫格
                pred_quadrant = pred_img[:, row*h:(row+1)*h, col*w:(col+1)*w]
                gt_quadrant = gt_img[:, row*h:(row+1)*h, col*w:(col+1)*w]
                
                # 简化的SSIM计算，避免复杂的多尺度计算（针对当前宫格）
            
            # 确保输入维度正确
                if pred_quadrant.dim() != 3 or gt_quadrant.dim() != 3:
                    print(f"SSIM损失宫格{quadrant_idx}输入维度错误: pred={pred_quadrant.shape}, gt={gt_quadrant.shape}")
                    ssim_losses.append(F.mse_loss(pred_quadrant, gt_quadrant))
                    continue
                
                # 单尺度SSIM计算（针对当前宫格）
                mu1 = pred_quadrant.mean(dim=[1, 2], keepdim=True)
                mu2 = gt_quadrant.mean(dim=[1, 2], keepdim=True)
                
                sigma1 = pred_quadrant.var(dim=[1, 2], keepdim=True)
                sigma2 = gt_quadrant.var(dim=[1, 2], keepdim=True)
                
                sigma12 = ((pred_quadrant - mu1) * (gt_quadrant - mu2)).mean(dim=[1, 2], keepdim=True)
            
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
            
            ssim_loss = 1 - ssim.mean()
            
            # 检查损失值是否合理
            if torch.isnan(ssim_loss) or torch.isinf(ssim_loss):
                print(f"SSIM损失宫格{quadrant_idx}值异常: {ssim_loss.item()}, 使用MSE损失")
                ssim_losses.append(F.mse_loss(pred_quadrant, gt_quadrant))
            else:
                ssim_losses.append(ssim_loss)
            
            # 计算平均SSIM损失
            if ssim_losses:
                final_ssim_loss = torch.stack(ssim_losses).mean()
                print(f"  SSIM loss (宫格{active_quadrants}): {final_ssim_loss.item():.6f}")
                return final_ssim_loss
            else:
                return F.mse_loss(pred_img, gt_img)
            
        except Exception as e:
            print(f"SSIM损失计算异常: {e}")
            return F.mse_loss(pred_img, gt_img)



    def _compute_clip_loss(self, pred_img: torch.Tensor, gt_img: torch.Tensor) -> torch.Tensor:
        """计算CLIP Score损失 - 四宫格分别计算
        输入pred_img/gt_img: [3, H, W], 值域[0,1]（四宫格图像）
        返回: 标量loss张量
        """
        try:
            # 按四宫格分别计算CLIP损失
            active_quadrants = [0, 1, 2, 3]  # 0:左上, 1:右上, 2:左下, 3:右下
            
            # 获取单个宫格的尺寸
            H, W = pred_img.shape[-2:]
            h, w = H // 2, W // 2
            
            clip_losses = []
            
            for quadrant_idx in active_quadrants:
                # 计算宫格位置
                row = quadrant_idx // 2  # 0或1
                col = quadrant_idx % 2   # 0或1
                
                # 提取对应的宫格
                pred_quadrant = pred_img[:, row*h:(row+1)*h, col*w:(col+1)*w]
                gt_quadrant = gt_img[:, row*h:(row+1)*h, col*w:(col+1)*w]
                
            # 简化的CLIP Score计算，避免复杂的多尺度计算
            # 使用全局特征相似性作为CLIP Score的近似
            
            # 确保输入维度正确
                if pred_quadrant.dim() != 3 or gt_quadrant.dim() != 3:
                    print(f"CLIP损失宫格{quadrant_idx}输入维度错误: pred={pred_quadrant.shape}, gt={gt_quadrant.shape}")
                    clip_losses.append(F.mse_loss(pred_quadrant, gt_quadrant))
                    continue
                
                # 1. 全局特征相似性（针对当前宫格）
                pred_flat = pred_quadrant.reshape(pred_quadrant.shape[0], -1)
                gt_flat = gt_quadrant.reshape(gt_quadrant.shape[0], -1)
                
                # 检查是否有零向量
                pred_norm = pred_flat.norm(p=2, dim=1, keepdim=True)
                gt_norm = gt_flat.norm(p=2, dim=1, keepdim=True)
                
                if torch.any(pred_norm == 0) or torch.any(gt_norm == 0):
                    print(f"CLIP损失宫格{quadrant_idx}: 检测到零向量，使用MSE损失")
                    clip_losses.append(F.mse_loss(pred_quadrant, gt_quadrant))
                    continue
                
                # 归一化
                pred_normalized = pred_flat / (pred_norm + 1e-8)
                gt_normalized = gt_flat / (gt_norm + 1e-8)
                
                # 计算余弦相似性
                similarity = torch.sum(pred_normalized * gt_normalized, dim=1)
                
                # 2. 颜色分布相似性（简化版本）
                pred_mean = pred_quadrant.mean(dim=[1, 2])  # [3]
                gt_mean = gt_quadrant.mean(dim=[1, 2])      # [3]
                
                color_similarity = 1 - F.l1_loss(pred_mean, gt_mean)
                
                # 综合相似性：全局 + 颜色分布
                combined_similarity = (0.7 * similarity.mean() + 0.3 * color_similarity)
                
                # CLIP损失 = 1 - 相似性
                clip_loss = 1 - combined_similarity
                if quadrant_idx==2:
                    clip_loss*=10.0
                clip_losses.append(clip_loss)
            
            # 计算平均CLIP损失
            if clip_losses:
                final_clip_loss = torch.stack(clip_losses).mean()
                print(f"  CLIP loss (宫格{active_quadrants}): {final_clip_loss.item():.6f}")
                return final_clip_loss
            else:
                return F.mse_loss(pred_img, gt_img)
            
        except Exception as e:
            print(f"CLIP损失计算异常: {e}")
            return F.mse_loss(pred_img, gt_img)

    def _compute_gan_loss(self, pred_img: torch.Tensor, gt_img: torch.Tensor) -> torch.Tensor:
        """计算GAN损失 - 严格按照官方代码方式
        
        官方代码：
        # Update discriminator discr
        lossD = discr(real, for_real=True) + discr(fake, for_real=False)
        lossD.backward()
        
        # Update generator G
        lossG = discr(fake, for_real=False)
        lossG.backward()
        
        输入pred_img/gt_img: [3, H, W], 值域[0,1]（四宫格图像）
        返回: 标量loss张量
        """
        try:
            if not hasattr(self, 'net_disc') or self.net_disc is None:
                print(f"GAN损失: 判别器未初始化，使用MSE损失")
                return F.mse_loss(pred_img, gt_img)
            
            # 按照官方代码：先训练判别器，再训练生成器
            # 将四宫格分割成4个单独的宫格，分别计算
            active_quadrants = [0, 1, 2, 3]  # 0:左上, 1:右上, 2:左下, 3:右下
            
            # 获取单个宫格的尺寸
            H, W = pred_img.shape[-2:]
            h, w = H // 2, W // 2
            
            # 第一步：训练判别器（严格按照官方代码）
            # lossD = discr(real, for_real=True) + discr(fake, for_real=False)
            self.disc_optimizer.zero_grad()
            disc_losses = []
            
            for quadrant_idx in active_quadrants:
                # 计算宫格位置
                row = quadrant_idx // 2  # 0或1
                col = quadrant_idx % 2   # 0或1
                
                # 提取对应的宫格
                pred_quadrant = pred_img[:, row*h:(row+1)*h, col*w:(col+1)*w]
                gt_quadrant = gt_img[:, row*h:(row+1)*h, col*w:(col+1)*w]
            

            
            # 计算平均判别器损失并反向传播
            if disc_losses:
                total_disc_loss = torch.stack(disc_losses).mean()
                total_disc_loss.backward()  # 判别器训练，不需要保留计算图
                self.disc_optimizer.step()
                print(f"  判别器训练完成，平均损失: {total_disc_loss.item():.6f}")
            
            # 第二步：训练生成器（严格按照官方代码）
            # lossG = discr(fake, for_G=True)
            gen_losses = []
            
            for quadrant_idx in active_quadrants:
                # 计算宫格位置
                row = quadrant_idx // 2  # 0或1
                col = quadrant_idx % 2   # 0或1
                
                # 提取对应的宫格
                pred_quadrant = pred_img[:, row*h:(row+1)*h, col*w:(col+1)*w]
                
                # 添加batch维度
                pred_batch = pred_quadrant.unsqueeze(0)  # [1, 3, h, w]
                
                # 确保值域在[0,1]
                pred_batch = torch.clamp(pred_batch, 0, 1)
                
                # 严格按照官方代码训练生成器
                # lossG = discr(fake, for_G=True)
                lossG = self.net_disc(pred_batch, for_G=True)
                
              
                
                gen_losses.append(lossG)
            
            # 计算平均生成器损失
            if gen_losses:
                gan_loss = torch.stack(gen_losses).mean()
                print(f"  GAN loss (宫格{active_quadrants}): {gan_loss.item():.6f}")
                
                # 注意：判别器已经在第一步用lossD训练过了
                # 这里的gan_loss将在主训练循环中用于更新生成器
                
                # 添加详细的GAN训练信息
                if hasattr(self, '_debug_step'):
                    if self._debug_step % 50 == 0:  # 每50步打印详细信息
                        print(f"    🎯 GAN训练状态:")
                        
                        # 计算判别器置信度（用于判断生成器是否在改进）
                        with torch.no_grad():
                            # 对真实图像的判别器输出
                            real_confidence = []
                            fake_confidence = []
                            
                            for quadrant_idx in active_quadrants:
                                row = quadrant_idx // 2
                                col = quadrant_idx % 2
                                pred_quadrant = pred_img[:, row*h:(row+1)*h, col*w:(col+1)*w]
                                gt_quadrant = gt_img[:, row*h:(row+1)*h, col*w:(col+1)*w]
                                
                                pred_batch = pred_quadrant.unsqueeze(0).clamp(0, 1)
                                gt_batch = gt_quadrant.unsqueeze(0).clamp(0, 1)
                                
                                # 判别器对真实图像的置信度
                                real_conf = self.net_disc(gt_batch, for_real=True).mean().item()
                                # 判别器对生成图像的置信度
                                fake_conf = self.net_disc(pred_batch, for_real=False).mean().item()
                                
                                real_confidence.append(real_conf)
                                fake_confidence.append(fake_conf)
                                
                                avg_real_conf = np.mean(real_confidence)
                                avg_fake_conf = np.mean(fake_confidence)
                                
                                print(f"      - 判别器对真实图像置信度: {avg_real_conf:.4f}")
                                print(f"      - 判别器对生成图像置信度: {avg_fake_conf:.4f}")
                                print(f"      - 置信度差距: {avg_real_conf - avg_fake_conf:.4f}")
                                
                                # 判断生成器是否在改进
                                if avg_fake_conf > 0.3:  # 生成图像被判别为真实
                                    print(f"      - ✅ 生成器表现良好，能欺骗判别器")
                                elif avg_fake_conf > 0.1:
                                    print(f"      - ⚠️ 生成器正在改进，但还需努力")
                                else:
                                    print(f"      - ❌ 生成器需要更多训练")
                                
                                # 判断判别器是否平衡
                                if avg_real_conf > 0.8 and avg_fake_conf < 0.2:
                                    print(f"      - ⚠️ 判别器过强，可能阻碍生成器学习")
                                elif avg_real_conf < 0.5:
                                    print(f"      - ⚠️ 判别器过弱，需要更多训练")
                                else:
                                    print(f"      - ✅ 判别器状态平衡")
                            
                            # 检查损失是否异常
                            if gan_loss.item() > 10.0:
                                print(f"    ⚠️ 警告: GAN loss过高 ({gan_loss.item():.6f})，可能导致训练不稳定")
                            elif gan_loss.item() < 0.001:
                                print(f"    ⚠️ 警告: GAN loss过低 ({gan_loss.item():.6f})，可能判别器过强")
            else:
                gan_loss = torch.tensor(0.0, device=pred_img.device)
                print(f"  GAN loss: 未计算（无活动宫格）")
            
            # 检查损失值是否合理
            if torch.isnan(gan_loss) or torch.isinf(gan_loss):
                print(f"GAN损失值异常: {gan_loss.item()}, 使用MSE损失")
                return F.mse_loss(pred_img, gt_img)
            
            return gan_loss
            
        except Exception as e:
            print(f"GAN损失计算异常: {e}")
            return F.mse_loss(pred_img, gt_img)

    def _compute_sift_loss(self, pred_img: torch.Tensor, gt_img: torch.Tensor) -> torch.Tensor:
        """计算SIFT/结构相似性损失 - 四宫格分别计算
        优先使用kornia的SIFT特征匹配均值距离；若不可用，使用基于Sobel梯度的结构损失。
        输入pred_img/gt_img: [3, H, W], 值域[0,1]（四宫格图像）
        返回: 标量loss张量
        """
        device = pred_img.device
        try:
            # 按四宫格分别计算SIFT损失
            active_quadrants = [0, 1, 2, 3]  # 0:左上, 1:右上, 2:左下, 3:右下
            
            # 获取单个宫格的尺寸
            H, W = pred_img.shape[-2:]
            h, w = H // 2, W // 2
            
            sift_losses = []
            
            for quadrant_idx in active_quadrants:
                # 计算宫格位置
                row = quadrant_idx // 2  # 0或1
                col = quadrant_idx % 2   # 0或1
                
                # 提取对应的宫格
                pred_quadrant = pred_img[:, row*h:(row+1)*h, col*w:(col+1)*w]
                gt_quadrant = gt_img[:, row*h:(row+1)*h, col*w:(col+1)*w]
                
                if KORNIA_AVAILABLE:
                    # 针对当前宫格计算SIFT损失
                    render_batch = pred_quadrant.unsqueeze(0)
                    gt_batch = gt_quadrant.unsqueeze(0)
                render_gray = kornia.color.rgb_to_grayscale(render_batch)
                gt_gray = kornia.color.rgb_to_grayscale(gt_batch)

                sift_detector = kornia.feature.SIFTFeature(num_features=200)
                render_lafs, render_resp, render_descs = sift_detector(render_gray)
                gt_lafs, gt_resp, gt_descs = sift_detector(gt_gray)

                if render_descs.shape[1] > 0 and gt_descs.shape[1] > 0:
                    render_desc_2d = render_descs.squeeze(0)
                    gt_desc_2d = gt_descs.squeeze(0)
                    matcher = kornia.feature.DescriptorMatcher('snn', 0.7)
                    dists, indices = matcher(render_desc_2d, gt_desc_2d)
                  
                    if dists.shape[0] > 0:
                            sift_losses.append(dists.mean())
                    else:
                            sift_losses.append(torch.tensor(1.0, device=device))
                else:
                        sift_losses.append(torch.tensor(1.0, device=device))
            else:
                    # 基于Sobel梯度的结构损失（针对当前宫格）
                    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred_quadrant.dtype, device=device).view(1, 1, 3, 3)
                    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred_quadrant.dtype, device=device).view(1, 1, 3, 3)

                    def to_gray(img: torch.Tensor) -> torch.Tensor:
                        return (0.299 * img[0:1] + 0.587 * img[1:2] + 0.114 * img[2:3]).unsqueeze(0)

                    pred_gray = to_gray(pred_quadrant)
                    gt_gray = to_gray(gt_quadrant)

                    pred_gx = F.conv2d(pred_gray, sobel_x, padding=1)
                    pred_gy = F.conv2d(pred_gray, sobel_y, padding=1)
                    gt_gx = F.conv2d(gt_gray, sobel_x, padding=1)
                    gt_gy = F.conv2d(gt_gray, sobel_y, padding=1)

                    pred_mag = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-8)
                    gt_mag = torch.sqrt(gt_gx ** 2 + gt_gy ** 2 + 1e-8)

                    structure_loss = F.l1_loss(pred_mag, gt_mag)
                    pixel_loss = F.mse_loss(pred_quadrant, gt_quadrant)
                    sift_loss = 0.7 * pixel_loss + 0.3 * structure_loss
                    sift_losses.append(sift_loss)
            
            # 计算平均SIFT损失
            if sift_losses:
                final_sift_loss = torch.stack(sift_losses).mean()
                print(f"  SIFT loss (宫格{active_quadrants}): {final_sift_loss.item():.6f}")
                return final_sift_loss
            else:
                return F.mse_loss(pred_img, gt_img)
        except Exception:
            return F.mse_loss(pred_img, gt_img)


    
    def save_discriminator_state(self, save_path: str):
        """保存判别器状态"""
        try:
            if hasattr(self, 'net_disc') and self.net_disc is not None:
                state_dict = {
                    'discriminator': self.net_disc.state_dict(),
                    'discriminator_optimizer': self.disc_optimizer.state_dict() if self.disc_optimizer else None
                }
                torch.save(state_dict, save_path)
                print(f"判别器状态已保存到: {save_path}")
            else:
                print("警告: 判别器未初始化，无法保存状态")
        except Exception as e:
            print(f"保存判别器状态失败: {e}")
    
    def load_discriminator_state(self, load_path: str):
        """加载判别器状态"""
        try:
            if hasattr(self, 'net_disc') and self.net_disc is not None:
                state_dict = torch.load(load_path, map_location=self.net_disc.device)
                
                if 'discriminator' in state_dict:
                    self.net_disc.load_state_dict(state_dict['discriminator'])
                    print(f"判别器状态已从 {load_path} 加载")
                
                if 'discriminator_optimizer' in state_dict and state_dict['discriminator_optimizer'] is not None:
                    if hasattr(self, 'disc_optimizer') and self.disc_optimizer is not None:
                        self.disc_optimizer.load_state_dict(state_dict['discriminator_optimizer'])
                        print(f"判别器优化器状态已从 {load_path} 加载")
            else:
                print("警告: 判别器未初始化，无法加载状态")
        except Exception as e:
            print(f"加载判别器状态失败: {e}")




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
