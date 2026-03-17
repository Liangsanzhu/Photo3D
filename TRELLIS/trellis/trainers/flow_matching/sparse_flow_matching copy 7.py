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


# ===== 全局缓存 CLIP 模型，避免重复加载 =====
_CLIP_MODEL = None
_CLIP_AVAILABLE = True
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

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
        ref_model_path (str, optional): Path to pretrained model for reference model initialization.
            Can be a local checkpoint file (*.pt, *.pth) or Hugging Face model ID.
        ref_model_key (str, optional): Model key to use for reference model. Default: 'denoiser'.
    """
    
    def __init__(self, *args, **kwargs):
        # 提取ref_model相关参数
        self.ref_model_path = kwargs.pop('ref_model_path', None)
        self.ref_model_key = kwargs.pop('ref_model_key', 'denoiser')
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

        # 初始化ref_model：优先使用预训练模型
        self.ref_model = self._init_ref_model()
        
        # 初始化GAN判别器
        if hasattr(self, 'device'):
            self.init_gan_discriminator(self.device)
        else:
            # 如果没有device属性，使用第一个模型的设备
            device = next(self.models.values()).device
            self.init_gan_discriminator(device)
    
    def _init_ref_model(self):
        """初始化参考模型，优先使用预训练模型"""
        import copy
        import torch
        
        # 方案1：如果指定了预训练模型路径，从checkpoint加载
        if self.ref_model_path is not None:
            try:
                print(f"🔄 从预训练checkpoint加载ref_model: {self.ref_model_path}")
                
                # 加载checkpoint
                if self.ref_model_path.endswith('.pt') or self.ref_model_path.endswith('.pth'):
                    # 直接加载checkpoint文件
                    checkpoint = torch.load(self.ref_model_path, map_location='cpu')
                    
                    # 创建模型副本
                    if self.ref_model_key in self.models:
                        ref_model = copy.deepcopy(self.models[self.ref_model_key])
                        
                        # 加载预训练权重
                        if 'model' in checkpoint:
                            state_dict = checkpoint['model']
                        elif 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        elif self.ref_model_key in checkpoint:
                            state_dict = checkpoint[self.ref_model_key]
                        else:
                            state_dict = checkpoint
                        
                        # 过滤掉不匹配的键
                        model_keys = set(ref_model.state_dict().keys())
                        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
                        missing_keys = model_keys - set(filtered_state_dict.keys())
                        
                        ref_model.load_state_dict(filtered_state_dict, strict=False)
                        
                        if missing_keys:
                            print(f"⚠️ 预训练模型中缺少以下键: {list(missing_keys)[:5]}...")
                        
                        # 冻结参数
                        for p in ref_model.parameters():
                            p.requires_grad_(False)
                        ref_model.eval()
                        
                        print(f"✅ 成功从预训练checkpoint加载ref_model")
                        return ref_model
                    else:
                        print(f"❌ 未找到模型键 '{self.ref_model_key}' 在self.models中")
                        
                elif self.ref_model_path.startswith('microsoft/') or self.ref_model_path.startswith('JeffreyXiang/'):
                    # 从Hugging Face Hub加载
                    print(f"🔄 从Hugging Face Hub加载ref_model: {self.ref_model_path}")
                    try:
                        from trellis import models
                        ref_model = models.from_pretrained(self.ref_model_path)
                        
                        # 移到正确设备
                        if self.ref_model_key in self.models:
                            device = next(self.models[self.ref_model_key].parameters()).device
                            ref_model = ref_model.to(device)
                        
                        # 冻结参数
                        for p in ref_model.parameters():
                            p.requires_grad_(False)
                        ref_model.eval()
                        
                        print(f"✅ 成功从Hugging Face Hub加载ref_model")
                        return ref_model
                    except Exception as e:
                        print(f"❌ 从Hugging Face Hub加载失败: {e}")
                        
            except Exception as e:
                print(f"❌ 加载预训练ref_model失败: {e}")
                print(f"🔄 回退到使用当前模型的初始状态")
        
        # 方案2：优先从本地路径加载预训练的TRELLIS模型作为ref_model
        try:
            # 先尝试本地缓存路径
            local_model_paths = [
                "/home/xinyue_liang/.cache/huggingface/hub/models--JeffreyXiang--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/",
                "/home/xinyue_liang/.cache/huggingface/hub/models--JeffreyXiang--TRELLIS-image-large/",
                # 可以添加更多本地路径
            ]
            
            ref_model = None
            for local_path in local_model_paths:
                if os.path.exists(local_path):
                    try:
                        print(f"🔄 从本地路径加载预训练TRELLIS模型: {local_path}")
                        from trellis import TrellisImageTo3DPipeline
                        
                        # 从本地路径加载pipeline
                        pipeline = TrellisImageTo3DPipeline.from_pretrained(local_path)
                        
                        # 提取第二阶段的flow matching模型
                        pretrained_slat_model = None
                        if hasattr(pipeline, 'models') and 'slat_flow_model' in pipeline.models:
                            pretrained_slat_model = pipeline.models['slat_flow_model']
                            print(f"✅ 找到预训练的slat_flow_model，类型: {type(pretrained_slat_model).__name__}")
                        elif hasattr(pipeline, 'slat_flow_model'):
                            pretrained_slat_model = pipeline.slat_flow_model
                            print(f"✅ 找到预训练的slat_flow_model，类型: {type(pretrained_slat_model).__name__}")
                        
                        if pretrained_slat_model is not None:
                            # 使用当前训练模型作为ref_model的架构模板，然后加载预训练权重
                            if self.ref_model_key in self.models:
                                current_model = self.models[self.ref_model_key]
                                print(f"🔄 基于当前训练模型创建ref_model，类型: {type(current_model).__name__}")
                                
                                # 深度复制当前模型作为ref_model
                                import copy
                                ref_model = copy.deepcopy(current_model)
                                
                                # 加载预训练权重（忽略新增的参数如ref_gamma_params）
                                pretrained_state_dict = pretrained_slat_model.state_dict()
                                missing_keys, unexpected_keys = ref_model.load_state_dict(pretrained_state_dict, strict=False)
                                
                                print(f"✅ 成功将预训练权重加载到ElasticSLatFlowModel ref_model中")
                                if missing_keys:
                                    print(f"   缺失的键（使用初始值）: {missing_keys}")
                                if unexpected_keys:
                                    print(f"   预训练模型中多余的键: {unexpected_keys}")
                                break
                            else:
                                print(f"⚠️ 当前模型中未找到{self.ref_model_key}")
                                continue
                        else:
                            print(f"⚠️ 本地模型中未找到slat_flow_model")
                            continue
                    except Exception as e:
                        print(f"❌ 从本地路径 {local_path} 加载失败: {e}")
                        continue
                else:
                    print(f"⚠️ 本地路径不存在: {local_path}")
            
            # 如果本地加载成功
            if ref_model is not None:
                # 移到正确的设备
                if self.ref_model_key in self.models:
                    device = next(self.models[self.ref_model_key].parameters()).device
                    ref_model = ref_model.to(device)
                
                # 冻结所有参数（ref_model不应该被训练）
                for p in ref_model.parameters():
                    p.requires_grad_(False)
                ref_model.eval()
                
                # ref_model不应该有可训练的ref_gamma_params，将其重置为0并冻结
                if hasattr(ref_model, 'ref_gamma_params'):
                    print(f"🔄 冻结ref_model的ref_gamma_params（ref_model不应该有KV融合权重）")
                    if isinstance(ref_model.ref_gamma_params, nn.ModuleDict):
                        # 新版本：ModuleDict
                        for param in ref_model.ref_gamma_params.parameters():
                            param.requires_grad_(False)
                        print(f"  - ref_model.ref_gamma_params MLP已冻结")
                    else:
                        # 旧版本：Parameter
                        with torch.no_grad():
                            ref_model.ref_gamma_params.zero_()
                        ref_model.ref_gamma_params.requires_grad_(False)
                        print(f"  - ref_model.ref_gamma_params已重置为0并冻结")
                
                print(f"✅ 成功设置本地预训练模型作为ref_model")
                print(f"  - ref_model类型: {type(ref_model).__name__}")
                print(f"  - ref_model设备: {ref_model.device}")
                print(f"  - ref_model参数数量: {sum(p.numel() for p in ref_model.parameters()):,}")
                print(f"  - 所有参数已冻结: {all(not p.requires_grad for p in ref_model.parameters())}")
                print(f"  - KV机制支持: reset_bank={hasattr(ref_model, 'reset_bank')}, get_bank={hasattr(ref_model, 'get_bank')}")
                return ref_model
            
            # 如果本地加载失败，回退到Hugging Face
            print(f"🔄 本地加载失败，尝试从Hugging Face加载预训练的TRELLIS模型")
            from trellis import TrellisImageTo3DPipeline
            
            # 加载预训练的TRELLIS pipeline
            pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
            
            # 提取第二阶段的flow matching模型
            if hasattr(pipeline, 'models') and 'slat_flow_model' in pipeline.models:
                ref_model = pipeline.models['slat_flow_model']
            elif hasattr(pipeline, 'slat_flow_model'):
                ref_model = pipeline.slat_flow_model
            else:
                print(f"⚠️ Hugging Face模型中未找到slat_flow_model")
                ref_model = None
                
            if ref_model is not None:
                # 移到正确的设备
                if self.ref_model_key in self.models:
                    device = next(self.models[self.ref_model_key].parameters()).device
                    ref_model = ref_model.to(device)
                
                # 冻结所有参数（ref_model不应该被训练）
                for p in ref_model.parameters():
                    p.requires_grad_(False)
                ref_model.eval()
                
                # 如果预训练模型意外包含ref_gamma_params，将其移除或冻结
                if hasattr(ref_model, 'ref_gamma_params'):
                    print(f"⚠️ 预训练模型包含ref_gamma_params，将其冻结")
                    if isinstance(ref_model.ref_gamma_params, nn.ModuleDict):
                        # 新版本：ModuleDict
                        for param in ref_model.ref_gamma_params.parameters():
                            param.requires_grad_(False)
                        print(f"  - ref_model.ref_gamma_params MLP已冻结")
                    else:
                        # 旧版本：Parameter
                        ref_model.ref_gamma_params.requires_grad_(False)
                        print(f"  - ref_model.ref_gamma_params已冻结")
                
                print(f"✅ 成功从Hugging Face加载预训练TRELLIS模型作为ref_model")
                print(f"  - ref_model类型: {type(ref_model).__name__}")
                print(f"  - ref_model设备: {ref_model.device}")
                print(f"  - ref_model参数数量: {sum(p.numel() for p in ref_model.parameters()):,}")
                print(f"  - 所有参数已冻结: {all(not p.requires_grad for p in ref_model.parameters())}")
                print(f"  - KV机制支持: reset_bank={hasattr(ref_model, 'reset_bank')}, get_bank={hasattr(ref_model, 'get_bank')}")
                return ref_model
                
        except Exception as e:
            print(f"❌ 加载预训练TRELLIS模型失败: {e}")
            print(f"🔄 回退到使用当前模型的初始权重")
        
        # 方案3：如果无法加载预训练模型，则不使用ref_model
        print(f"❌ 无法加载预训练模型作为ref_model")
        print(f"⚠️ 警告：没有有效的ref_model，KV机制将无法使用")
        print(f"💡 建议：请确保预训练模型路径正确，或手动调用set_ref_model_from_pretrained()方法")
        
        # 方案4：无法初始化ref_model
        print(f"⚠️ 无法初始化ref_model，将在采样时使用当前模型作为fallback")
        return None
    
    def set_ref_model_from_pretrained(self, model_path_or_name: str = "JeffreyXiang/TRELLIS-image-large"):
        """
        直接设置预训练模型作为ref_model
        
        Args:
            model_path_or_name: 预训练模型的路径或Hugging Face模型名称
        """
        try:
            print(f"🔄 设置预训练模型作为ref_model: {model_path_or_name}")
            
            if model_path_or_name.endswith('.pt') or model_path_or_name.endswith('.pth'):
                # 从本地checkpoint加载
                print(f"从本地文件加载: {model_path_or_name}")
                checkpoint = torch.load(model_path_or_name, map_location='cpu')
                
                # 创建模型副本
                if self.ref_model_key in self.models:
                    ref_model = copy.deepcopy(self.models[self.ref_model_key])
                    
                    # 加载预训练权重
                    if 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif self.ref_model_key in checkpoint:
                        state_dict = checkpoint[self.ref_model_key]
                    else:
                        state_dict = checkpoint
                    
                    # 过滤不匹配的键
                    model_state_dict = ref_model.state_dict()
                    filtered_state_dict = {}
                    for k, v in state_dict.items():
                        if k in model_state_dict and model_state_dict[k].shape == v.shape:
                            filtered_state_dict[k] = v
                    
                    ref_model.load_state_dict(filtered_state_dict, strict=False)
                    
                    # 移到正确设备并冻结
                    device = next(self.models[self.ref_model_key].parameters()).device
                    ref_model = ref_model.to(device)
                    for p in ref_model.parameters():
                        p.requires_grad_(False)
                    ref_model.eval()
                    
                    self.ref_model = ref_model
                    print(f"✅ 成功从本地文件设置ref_model")
                    return True
                    
            else:
                # 从Hugging Face或本地路径加载
                print(f"从模型路径加载: {model_path_or_name}")
                from trellis import TrellisImageTo3DPipeline
                
                pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path_or_name)
                
                # 提取第二阶段的flow matching模型
                ref_model = None
                if hasattr(pipeline, 'models') and 'slat_flow_model' in pipeline.models:
                    ref_model = pipeline.models['slat_flow_model']
                elif hasattr(pipeline, 'slat_flow_model'):
                    ref_model = pipeline.slat_flow_model
                
                if ref_model is not None:
                    # 移到正确设备并冻结
                    if self.ref_model_key in self.models:
                        device = next(self.models[self.ref_model_key].parameters()).device
                        ref_model = ref_model.to(device)
                    
                    for p in ref_model.parameters():
                        p.requires_grad_(False)
                    ref_model.eval()
                    
                    self.ref_model = ref_model
                    print(f"✅ 成功设置预训练模型作为ref_model")
                    return True
                else:
                    print(f"❌ 预训练模型中未找到slat_flow_model")
                    return False
                    
        except Exception as e:
            print(f"❌ 设置预训练ref_model失败: {e}")
            return False
    
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
    
    def set_ref_model_from_pretrained(self, model_path, model_key='denoiser'):
        """
        从预训练模型设置ref_model
        
        Args:
            model_path: 预训练模型路径，可以是：
                - 本地checkpoint文件路径 (*.pt, *.pth)
                - Hugging Face模型ID (如 'microsoft/TRELLIS-image-large')
            model_key: 要使用的模型键，默认为'denoiser'
        
        Examples:
            # 使用本地checkpoint
            trainer.set_ref_model_from_pretrained('/path/to/pretrained_model.pt')
            
            # 使用Hugging Face模型
            trainer.set_ref_model_from_pretrained('microsoft/TRELLIS-image-large')
            
            # 在训练器初始化时指定
            trainer = SparseFlowMatchingTrainer(
                models, dataset, 
                ref_model_path='microsoft/TRELLIS-image-large',
                **other_args
            )
        """
        self.ref_model_path = model_path
        self.ref_model_key = model_key
        
        # 重新初始化ref_model
        old_ref_model = self.ref_model
        self.ref_model = self._init_ref_model()
        
        if self.ref_model is not None:
            print(f"✅ 成功更新ref_model为预训练模型: {model_path}")
            # 清理旧模型
            if old_ref_model is not None:
                del old_ref_model
        else:
            print(f"❌ 更新ref_model失败，保持原有ref_model")
            self.ref_model = old_ref_model
    
    
  
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
        x_0: sp.SparseTensor,  # 这里实际上是 loss_feats
        cond=None,
        use_dpo=False,
        ref_model=None,
        x0_win=None,
        x0_loss=None,
        dpo_beta=1.0,
        sample_same_epsilon=True,
        use_hinge_gan=False,
        hinge_gan_weight=1.0,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x ... x C] sparse tensor of the loss_feats (structure reference).
            cond: The [N x ...] tensor of additional conditions.
            use_dpo: Whether to use DPO training instead of regular flow matching.
            ref_model: Reference model for DPO training.
            x0_win: Winning samples for DPO training.
            x0_loss: Losing samples for DPO training.
            dpo_beta: Beta parameter for DPO loss.
            sample_same_epsilon: Whether to use same epsilon for ref model.
            use_hinge_gan: Whether to use hinge adversarial training for 3D latent.
            hinge_gan_weight: Weight for hinge GAN loss.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        
        # 常规Flow Matching模式
        # 确保loss_feats是独立的tensor，避免重复计算图
        if x_0.feats.requires_grad:
            x_0 = x_0.replace(x_0.feats.detach().clone())
        
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        loss_feats_noisy = self.diffuse(x_0, t, noise=noise)  # loss_feats 的加噪结果
        cond = self.get_cond(cond, **kwargs)
        #sprint("cond",cond.shape,cond)
        
        # 获取当前使用的模型名称
        model_name = list(self.training_models.keys())[0] if self.training_models else 'denoiser'
        model = self.training_models[model_name]
        # ReferenceNet：写阶段使用冻结ref_model，自注意力KV写入；读阶段用训练模型替换自注意力KV
        # 统一读取 win/loss 键（兼容旧键）
        win_feats = kwargs.get('win', None)
        assert win_feats is not None, '需要在kwargs中提供 feats 或 style（win_feats）用于read阶段'
        
        # 计算win_feats的加噪版本，与后续读操作保持一致
        win_feats_noisy = self.diffuse(win_feats, t, noise=noise)
        
        try:
            with torch.no_grad():
                # 写：ReferenceNet 使用加噪的 win_feats_noisy（与读操作保持一致）
                if hasattr(self.ref_model, 'reset_bank'):
                    self.ref_model.reset_bank()
                
                # 修改：训练时ref_model使用加噪的win_feats_noisy，保持与读操作的一致性
                _ = self.ref_model(win_feats_noisy, t * 1000, cond, bank_mode='write')
                
                if hasattr(self.ref_model, 'get_bank') and hasattr(model, 'set_bank'):
                    bank_k, bank_v = self.ref_model.get_bank()
                    if hasattr(model, 'reset_bank'):
                        model.reset_bank()
                    model.set_bank(bank_k, bank_v)
        except Exception as e:
            print(f"❌ [训练KV] 失败: {str(e)}")
        
        # 读：DenoisingNet 使用相同的 win_feats_noisy（替换自注意力KV）
        pred = model(win_feats_noisy, t * 1000, cond, bank_mode='read')
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(win_feats, noise, t)
        
        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]
        
        # 如果启用了decoder训练，decoder会通过Flow Matching的loss自动训练
        # 因为decoder在LPIPS loss计算中被使用，梯度会自动反传
        if hasattr(self, 'train_decoder') and self.train_decoder and 'decoder' in self.training_models:
            print("✅ Decoder将通过Flow Matching loss进行训练")
        
        # Hinge对抗训练模式
        if use_hinge_gan:
            hinge_loss = self._compute_hinge_gan_loss(pred, x_0, cond, noise=noise, target=target, **kwargs)
            if hinge_loss is not None:
                terms["hinge_gan"] = hinge_loss
              
                print(f"  Hinge GAN Loss: {hinge_gan_weight * hinge_loss.item():.6f}")
        
        # 计算多种2D损失函数，避免生成白色/模糊结果
        # 确保LPIPS使用与pred同一object的外部GT图：优先使用style对应的gt_image
        lpips_kwargs = dict(kwargs)
        if 'gt_image_style' in lpips_kwargs:
            lpips_kwargs['gt_image'] = lpips_kwargs['gt_image_style']
        
        # 根据梯度控制模式计算LPIPS损失
        if hasattr(self, 'decoder_gradient_mode') and self.decoder_gradient_mode == 'dual_pass':
            # 两遍计算模式：第一遍用于优化扩散模型，第二遍用于优化decoder
            print("🔄 使用两遍decoder计算模式")
            
            # 第一遍：冻结decoder参数，只优化扩散模型
            if 'decoder' in self.training_models:
                # 临时冻结decoder参数
                decoder_params = list(self.training_models['decoder'].parameters())
                decoder_grad_states = [p.requires_grad for p in decoder_params]
                for p in decoder_params:
                    p.requires_grad_(False)
            
            lpips_loss_diffusion = self._compute_lpips_loss(pred, x_0, cond, noise=noise, target=target, **lpips_kwargs)
            
            # 恢复decoder参数梯度状态
            if 'decoder' in self.training_models:
                for p, grad_state in zip(decoder_params, decoder_grad_states):
                    p.requires_grad_(grad_state)
            
            # 第二遍：detach pred，只优化decoder
            if 'decoder' in self.training_models and lpips_loss_diffusion is not None:
                pred_detached = pred.replace(pred.feats.detach())
                lpips_loss_decoder = self._compute_lpips_loss(pred_detached, x_0, cond, noise=noise, target=target, **lpips_kwargs)
                
                if lpips_loss_decoder is not None:
                    # 将两遍损失相加，但通过冻结和detach控制梯度流向
                    terms["lpips"] = lpips_loss_diffusion + lpips_loss_decoder
                    terms["lpips_diffusion"] = lpips_loss_diffusion
                    terms["lpips_decoder"] = lpips_loss_decoder
                    print(f"  LPIPS Loss (扩散模型): {lpips_loss_diffusion.item():.6f}")
                    print(f"  LPIPS Loss (Decoder): {lpips_loss_decoder.item():.6f}")
                else:
                    terms["lpips"] = lpips_loss_diffusion
            else:
                terms["lpips"] = lpips_loss_diffusion if lpips_loss_diffusion is not None else 0.0
        else:
            # 默认模式或detach_pred模式
            detach_pred = hasattr(self, 'decoder_gradient_mode') and self.decoder_gradient_mode == 'detach_pred'
            if detach_pred:
                print("✂️ 使用detach_pred模式：截断pred梯度")
                pred_for_lpips = pred.replace(pred.feats.detach())
            else:
                pred_for_lpips = pred
            
            lpips_loss = self._compute_lpips_loss(pred_for_lpips, x_0, cond, noise=noise, target=target, **lpips_kwargs)
            if lpips_loss is not None:
                terms["lpips"] = lpips_loss
            
            # 组合5种损失函数：LPIPS、SIFT、MSE、CLIP Score、SSIM
            # 权重为0的损失函数不会计算，减少计算量
            # 简化损失函数组合
            loss_weights = {

                'lpips': 10.0,      # 感知损失，对纹理和真实感最重要
                'sift': 0.0,       # SIFT特征损失（几何严格对应，暂时关闭）
                'mse': 0.0,        # 轻微像素级约束，保持基本准确性
                'clip': 0.0,       # CLIP语义损失，关注纹理语义
                'ssim': 0.,       # 结构相似性损失（几何严格对应，暂时关闭）
                'gan': 0.0,        # GAN损失，提高真实感
                '3dgan':0.0
            }
            
            # 计算额外的2D损失函数（只计算权重大于0的）
            #additional_losses = self._compute_additional_2d_losses(pred, x_0, cond, noise=noise, target=target, loss_weights=loss_weights, **kwargs)
            
            # 计算总损失（只计算权重大于0的损失）
            total_loss = terms["lpips"] *loss_weights['lpips']
            if False: #for loss_name, weight in loss_weights.items():
                if weight <= 0:
                    pass
                   # continue  # 跳过权重为0的损失
                    
                if loss_name == 'lpips' and lpips_loss is not None:
                    total_loss += weight * lpips_loss
                    print(f"  {loss_name.upper()} Loss: {weight * lpips_loss.item():.6f}")
                elif loss_name in additional_losses:
                    total_loss += weight * additional_losses[loss_name]
                    print(f"  {loss_name.upper()} Loss: {weight * additional_losses[loss_name].item():.6f}")
                elif loss_name == 'mse':
                    total_loss += weight * terms["mse"]
                    print(f"  {loss_name.upper()} Loss: {weight * terms['mse'].item():.6f}")
                elif loss_name == '3dgan':
                    total_loss += weight * terms["hinge_gan"]
                    print(f"  {loss_name.upper()} Loss: {weight * terms['mse'].item():.6f}")
            
            terms["loss"]= terms["mse"]*loss_weights['mse']+terms["lpips"] *loss_weights['lpips']
        # 调试：检查loss和预测值的统计信息
        if hasattr(self, '_debug_step'):
            self._debug_step += 1
        else:
            self._debug_step = 0
        
        # 每步都打印基本信息
        print(f"[训练步骤{self._debug_step}] MSE: {terms['mse'].item():.6f}", end="")
        if "lpips" in terms:
            lpips_val = terms["lpips"]
            if hasattr(lpips_val, 'item'):
                lpips_val = lpips_val.item()
            print(f", LPIPS: {lpips_val:.6f}")
        else:
            print()
        
      
        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        # 定期汇报ref_gamma_params状态
        if hasattr(model, 'ref_gamma_params') and self._debug_step % 50 == 0:
            if isinstance(model.ref_gamma_params, nn.ModuleDict):
                # 新版本：ModuleDict，收集所有参数的统计信息
                all_params = []
                all_grads = []
                for param in model.ref_gamma_params.parameters():
                    all_params.append(param.data.view(-1))
                    if param.grad is not None:
                        all_grads.append(param.grad.view(-1))
                
                if all_params:
                    all_params_tensor = torch.cat(all_params)
                    param_min = all_params_tensor.min().item()
                    param_max = all_params_tensor.max().item()
                    param_mean = all_params_tensor.mean().item()
                    print(f"📊 [Step {self._debug_step}] ref_gamma_params MLP: min={param_min:.6f}, max={param_max:.6f}, mean={param_mean:.6f}")
                    
                    if all_grads:
                        all_grads_tensor = torch.cat(all_grads)
                        grad_norm = all_grads_tensor.norm().item()
                        print(f"    MLP梯度范数: {grad_norm:.6f}")
                    else:
                        print(f"    MLP梯度: None")
            else:
                # 旧版本：Parameter
                param_min = model.ref_gamma_params.min().item()
                param_max = model.ref_gamma_params.max().item()
                param_mean = model.ref_gamma_params.mean().item()
                print(f"📊 [Step {self._debug_step}] ref_gamma_params: min={param_min:.6f}, max={param_max:.6f}, mean={param_mean:.6f}")
                
                if model.ref_gamma_params.grad is not None:
                    grad_norm = model.ref_gamma_params.grad.norm().item()
                    print(f"    梯度范数: {grad_norm:.6f}")
                else:
                    print(f"    梯度: None")
        
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
        sample_ref = []  # 参考模型的样本
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
        
        # 生成随机噪声作为采样起点（使用win的布局但随机特征）
        if use_random_seed:
            torch.manual_seed((current_seed) % (2**32))  # 使用当前种子
        win_tensor = data['win'] 
        noise = win_tensor.replace(torch.randn_like(win_tensor.feats))
        
        print(f"🔍 采样起点: 使用win的布局({win_tensor.shape})但随机特征作为noise")
        
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
     
             
    
        
        # 确保模型没有残留的KV缓存
        if hasattr(self.models['denoiser'], 'reset_bank'):
            self.models['denoiser'].reset_bank()
        if hasattr(self.ref_model, 'reset_bank'):
            self.ref_model.reset_bank()
        
        from trellis.pipelines.samplers import FlowEulerCfgSampler
        kv_sampler = FlowEulerCfgSampler(sigma_min=getattr(self, 'sigma_min', 0.002))
        
        if (hasattr(self.models['denoiser'], 'reset_bank') and 
            hasattr(self.models['denoiser'], 'set_bank') and 
            self.ref_model is not None and
            hasattr(self.ref_model, 'reset_bank') and 
            hasattr(self.ref_model, 'get_bank')):
            
            # 确保采样时模型处于eval模式且不需要梯度
            self.ref_model.eval()
            self.models['denoiser'].eval()
            
            # 临时禁用ref_gamma_params的梯度以节省显存
            original_requires_grad = {}
            if hasattr(self.models['denoiser'], 'ref_gamma_params'):
                if isinstance(self.models['denoiser'].ref_gamma_params, nn.ModuleDict):
                    # 新版本：ModuleDict
                    original_requires_grad['ref_gamma_params'] = {}
                    for name, param in self.models['denoiser'].ref_gamma_params.named_parameters():
                        original_requires_grad['ref_gamma_params'][name] = param.requires_grad
                        param.requires_grad_(False)
                else:
                    # 旧版本：Parameter
                    original_requires_grad['ref_gamma_params'] = self.models['denoiser'].ref_gamma_params.requires_grad
                    self.models['denoiser'].ref_gamma_params.requires_grad_(False)
            
            loss_feats_for_kv = noise
            win_feats_for_kv = noise
            
            res_trained = kv_sampler.sample_with_ref_kv(
                ref_model=self.ref_model,      # 写阶段：使用ref_model
                model=self.models['denoiser'], # 读阶段：使用正在训练的模型
                noise=noise,
                loss_feats=loss_feats_for_kv,
                win_feats=win_feats_for_kv,
                **args,
                steps=50, cfg_strength=3.0, verbose=verbose,
            )
            sample_trained = [res_trained.samples]
            
            # 恢复ref_gamma_params的梯度设置
            if 'ref_gamma_params' in original_requires_grad:
                if isinstance(self.models['denoiser'].ref_gamma_params, nn.ModuleDict):
                    # 新版本：ModuleDict
                    for name, param in self.models['denoiser'].ref_gamma_params.named_parameters():
                        param.requires_grad_(original_requires_grad['ref_gamma_params'][name])
                else:
                    # 旧版本：Parameter
                    self.models['denoiser'].ref_gamma_params.requires_grad_(original_requires_grad['ref_gamma_params'])
            
            # 恢复训练模式
            self.models['denoiser'].train()
            
        else:
            # 确保普通采样时也处于eval模式
            self.models['denoiser'].eval()
            
            res_trained = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=50, cfg_strength=3.0, verbose=verbose,
            )
            sample_trained = [res_trained.samples]
            
            # 恢复训练模式
            self.models['denoiser'].train()

        # 3. 使用原始Trellis方法生成sample_ref（不使用KV写读，直接采样）
        ref_model_to_use = None
        if hasattr(self, 'ref_model') and self.ref_model is not None:
            ref_model_to_use = self.ref_model
            
            # 简化采样状态检查
                
        elif 'denoiser' in self.models:
            ref_model_to_use = self.models['denoiser']
        else:
            print("❌ 错误: 既没有ref_model也没有denoiser模型")
        
        if ref_model_to_use is not None:
            ref_model_to_use.eval()
            # 确保模型是冻结的（对于采样）
            original_requires_grad = {}
            for name, p in ref_model_to_use.named_parameters():
                original_requires_grad[name] = p.requires_grad
                p.requires_grad_(False)
            
            # 重置KV缓存，确保使用原始方法
            if hasattr(ref_model_to_use, 'reset_bank'):
                ref_model_to_use.reset_bank()
                print("✅ 已重置模型的KV缓存（原始方法前）")
            
            # 使用原始采样方法，不经过KV写读
            res_ref = sampler.sample(
                ref_model_to_use,
                noise=noise,
                **args,
                steps=50, cfg_strength=3.0, verbose=verbose,
            )
            sample_ref = [res_ref.samples]
            print(f"✅ sample_ref生成完成，形状: {res_ref.samples.shape}")
            
            # 恢复原始的requires_grad状态
            for name, p in ref_model_to_use.named_parameters():
                p.requires_grad_(original_requires_grad[name])
        else:
            print("⚠️ 警告: 无可用模型生成sample_ref，使用noise作为占位符")
            # 创建一个空的sample_ref
            sample_ref = [noise]  # 使用noise作为占位符
        # 处理原始数据样本（不是生成的）
        if 'win' in data:
            sample_win_original = sp.sparse_cat([data['win']])
        else:
            sample_win_original = None
            
        if 'loss' in data:
            sample_loss_original = sp.sparse_cat([data['loss']])
        else:
            sample_loss_original = None
       
        sample_trained = sp.sparse_cat(sample_trained)
        sample_ref = sp.sparse_cat(sample_ref)

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
        sample_dict = {
            'sample_ref': {'value': sample_ref, 'type': 'sample', 'sample_type': 'reference_model'},
        }
        
        # 添加原始数据样本（非生成）
        if sample_win_original is not None:
            sample_dict['sample_win'] = {'value': sample_win_original, 'type': 'sample', 'sample_type': 'original_data_win'}
        
        if sample_loss_original is not None:
            sample_dict['sample_loss'] = {'value': sample_loss_original, 'type': 'sample', 'sample_type': 'original_data_loss'}
        
        # 添加按照当前训练方法生成的样本（KV写读流程的结果）
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
                    # 新版本：ModuleDict
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
                    # 旧版本：Parameter
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
            
            # 设置ref_model到pipeline中，启用KV机制
            if hasattr(self, 'ref_model') and self.ref_model is not None:
                # 直接设置ref_model属性到pipeline
                pipeline.ref_model = self.ref_model
                print("✅ 已将ref_model设置到pipeline中，启用KV机制")
                
                # 确保slat_sampler也能访问到sample_with_ref_kv方法
                if hasattr(pipeline, 'slat_sampler') and hasattr(pipeline.slat_sampler, 'sample_with_ref_kv'):
                    print("✅ pipeline的slat_sampler支持KV机制")
                else:
                    print("⚠️ pipeline的slat_sampler可能不支持KV机制")
            else:
                print("⚠️ 当前训练器中没有ref_model，彻底测试将不使用KV机制")
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
            if hasattr(self.dataset, 'decode_latent') and hasattr(self.dataset, '_render_gaussian'):
                # 使用训练的decoder进行解码（如果启用）
                if hasattr(self, 'train_decoder') and self.train_decoder and 'decoder' in self.training_models:
                    # 使用训练的decoder（可选图像条件）
                    use_cond_flag = getattr(self, 'decoder_use_image_condition', False)
                    cond_image = kwargs.get('image', None) if use_cond_flag else None
                    if isinstance(cond_image, list) and len(cond_image) > 0:
                        cond_image = cond_image[0]
                    used_cond = use_cond_flag and (cond_image is not None)
                    print(f"[train LPIPS] decoder_use_image_condition={use_cond_flag}, used_cond={used_cond}")
                    decoded_pred = self.training_models['decoder'](x0_pred, cond_image) if used_cond else self.training_models['decoder'](x0_pred)
                    print("✅ 使用训练的decoder进行LPIPS计算")
                else:
                    # 使用dataset中的固定decoder
                    decoded_pred = self.dataset.decode_latent_grad(x0_pred, sample_type="model_generated")
                    print("✅ 使用dataset中的固定decoder进行LPIPS计算")
                
                pred_grid = self.dataset._render_gaussian(decoded_pred)  # [3,H,W]

                # 训练可视化：定期保存 pred_grid 与 gt_grid 方便排查"变白/变黑"
                try:
                    save_every = int(getattr(self, 'save_train_vis_every', 50))
                    if isinstance(pred_grid, torch.Tensor) and save_every > 0 and (self.step % save_every == 0):
                        import os
                        from torchvision.utils import save_image
                        # 为避免覆盖，加入一个自增序号
                        vis_root = os.path.join(self.output_dir, 'samples', f'train_vis_step')
                        os.makedirs(vis_root, exist_ok=True)
                        idx = int(getattr(self, '_train_vis_idx', 0))
                        vis_dir = vis_root
                        os.makedirs(vis_dir, exist_ok=True)
                        # 保存预测渲染
                        save_image(pred_grid.clamp(0,1), os.path.join(vis_dir, 'pred_grid.png'))
                        # 保存对应的condition图片（使用原始输入图像，而非处理后的cond embedding）
                        try:
                            cond_img = None
                            raw_img = kwargs.get('image', None)
                            if isinstance(raw_img, torch.Tensor):
                                cond_img = raw_img
                                if cond_img.dim() == 4:
                                    cond_img = cond_img[0]
                                if cond_img.shape[0] == 1:
                                    cond_img = cond_img.repeat(3,1,1)
                                elif cond_img.shape[0] == 4:
                                    rgb = cond_img[:3]
                                    alpha = cond_img[3:4].clamp(0,1)
                                    cond_img = rgb * alpha + (1 - alpha)
                                if cond_img.max() > 1.5:
                                    cond_img = cond_img / 255.0
                                elif cond_img.min() < 0.0:
                                    cond_img = (cond_img + 1.0) / 2.0
                                if cond_img.shape[-2:] != pred_grid.shape[-2:]:
                                    cond_img = F.interpolate(cond_img.unsqueeze(0), size=pred_grid.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
                            if isinstance(cond_img, torch.Tensor):
                                save_image(cond_img.clamp(0,1), os.path.join(vis_dir, 'cond.png'))
                        except Exception:
                            pass
                        # 若可得GT渲染则一并保存（在下方获得gt_grid后再保存）
                        _pending_gt_path = os.path.join(vis_dir, 'gt_grid.png')
                        # 保存原始输入路径记录
                        try:
                            meta_txt = os.path.join(vis_dir, 'meta.txt')
                            with open(meta_txt, 'w') as f:
                                def _fmt(v):
                                    if isinstance(v, (list, tuple)):
                                        return str(v[0])
                                    return str(v)
                                if 'win_path' in kwargs:
                                    f.write(f"win_path: {_fmt(kwargs['win_path'])}\n")
                                if 'loss_path' in kwargs:
                                    f.write(f"loss_path: {_fmt(kwargs['loss_path'])}\n")
                                if 'cond_path' in kwargs:
                                    f.write(f"cond_path: {_fmt(kwargs['cond_path'])}\n")
                                if 'gt_path' in kwargs:
                                    f.write(f"gt_path: {_fmt(kwargs['gt_path'])}\n")
                        except Exception:
                            pass
                        # 自增序号
                        #self._train_vis_idx = idx + 1
                    else:
                        _pending_gt_path = None
                except Exception as _:
                    _pending_gt_path = None

            # 使用样本中的gt_image作为GT四宫格（外部提供）
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
                    # 可选：四宫格重排，确保与pred的宫格顺序一致（默认[0,1,2,3]）
                    try:
                        order = getattr(self, 'gt_quadrant_order', [0,1,2,3])
                        if isinstance(order, (list, tuple)) and len(order) == 4 and order != [0,1,2,3]:
                            H, W = gt_img.shape[-2:]
                            h, w = H // 2, W // 2
                            quads = [
                                gt_img[:, 0:h,   0:w],   # 0 左上
                                gt_img[:, 0:h,   w:W],   # 1 右上
                                gt_img[:, h:H,   0:w],   # 2 左下
                                gt_img[:, h:H,   w:W],   # 3 右下
                            ]
                            gt_img = torch.zeros_like(gt_img)
                            mapping = {
                                0: (slice(0, h), slice(0, w)),
                                1: (slice(0, h), slice(w, W)),
                                2: (slice(h, H), slice(0, w)),
                                3: (slice(h, H), slice(w, W)),
                            }
                            for dst_idx, src_idx in enumerate(order):
                                ys, xs = mapping[dst_idx]
                                gt_img[:, ys, xs] = quads[src_idx]
                    except Exception as _:
                        pass
                    gt_grid = gt_img.clamp(0, 1)

            if gt_grid is None:
                return None

            if pred_grid is None or gt_grid is None:
                return None

            # 若设置了保存路径，保存gt_grid
            try:
                if 'vis_dir' in locals() and _pending_gt_path is not None:
                    from torchvision.utils import save_image
                    save_image(gt_grid.clamp(0,1), _pending_gt_path)
                    # 额外保存：GT对应的latent渲染（loss/x_0）与WIN/style的latent渲染
                    try:
                        vis_dir = os.path.dirname(_pending_gt_path)
                        # loss/x_0 latent
                        x0_gt_img = self.dataset._visualize_x0_tensor(sp.SparseTensor(feats=x_0.feats, coords=x_0.coords), sample_type="ground_truth")
                        save_image(x0_gt_img.clamp(0,1), os.path.join(vis_dir, 'loss_latent_grid.png'))
                        # win/style latent（若提供）
                    except Exception:
                        pass
            except Exception:
                pass

            # 使用随机裁剪 CLIP 图像-图像损失替代 LPIPS
            clip_loss = self._compute_clip_loss(pred_grid, gt_grid)
            lpips_loss = clip_loss

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
                # 使用训练的decoder进行解码（如果启用）
                if hasattr(self, 'train_decoder') and self.train_decoder and 'decoder' in self.training_models:
                    # 使用训练的decoder（可选图像条件）
                    use_cond_flag = getattr(self, 'decoder_use_image_condition', False)
                    cond_image = kwargs.get('image', None) if use_cond_flag else None
                    if isinstance(cond_image, list) and len(cond_image) > 0:
                        cond_image = cond_image[0]
                    used_cond = use_cond_flag and (cond_image is not None)
                    print(f"[train add2D] decoder_use_image_condition={use_cond_flag}, used_cond={used_cond}")
                    decoded_pred = self.training_models['decoder'](x0_pred, cond_image) if used_cond else self.training_models['decoder'](x0_pred)
                    print("✅ 使用训练的decoder进行2D损失计算")
                else:
                    # 使用dataset中的固定decoder
                    decoded_pred = self.dataset.decode_latent_grad(x0_pred, sample_type="model_generated")
                    print("✅ 使用dataset中的固定decoder进行2D损失计算")
                
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

    def _compute_hinge_gan_loss(self, pred, x_0, cond, noise=None, target=None, **kwargs):
        """
        计算3D latent的hinge对抗损失
        
        Hinge损失公式：
        - 生成器损失: L_G = -E[D(G(z))]
        - 判别器损失: L_D = E[max(0, 1 - D(x_real))] + E[max(0, 1 + D(G(z)))]
        
        Args:
            pred: 模型预测的velocity field (Flow Matching)
            x_0: 真实的x0 (3D latent)
            cond: 条件信息
            noise: 噪声
            target: 目标velocity
            kwargs: 其他参数
            
        Returns:
            Hinge GAN loss tensor 或 None（如果计算失败）
        """
        try:
            # 初始化3D判别器（如果还没有初始化）
            if not hasattr(self, '_3d_discriminator'):
                self._init_3d_discriminator(x_0.device)
            
            if not hasattr(self, '_3d_discriminator') or self._3d_discriminator is None:
                print("警告: 3D判别器未初始化，跳过hinge GAN损失计算")
                return None
            
            # 重构x0_pred和x0_gt
            if noise is None or target is None:
                noise = torch.randn_like(x_0.feats)
                target = self.get_v(x_0, x_0.replace(noise), self.sample_t(x_0.shape[0]).to(x_0.device).float())
            
            # 从pred重构x0_pred
            x0_pred_feats = noise.feats - pred.feats
            x0_pred = sp.SparseTensor(feats=x0_pred_feats, coords=x_0.coords)
            
            # 真实样本：x0_gt
            x0_gt_feats = noise.feats - target.feats
            x0_gt = sp.SparseTensor(feats=x0_gt_feats, coords=x_0.coords)
            
            # 确保特征维度一致
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
            
            # 第一步：训练判别器
            self._3d_disc_optimizer.zero_grad()
            
            # 真实样本的判别器输出
            real_scores = self._3d_discriminator(x0_gt)
            # 生成样本的判别器输出
            fake_scores = self._3d_discriminator(x0_pred.detach())  # 分离梯度
            
            # Hinge判别器损失
            disc_loss_real = torch.mean(F.relu(1.0 - real_scores))
            disc_loss_fake = torch.mean(F.relu(1.0 + fake_scores))
            disc_loss = disc_loss_real + disc_loss_fake
            
            # 判别器反向传播
            disc_loss.backward()
            self._3d_disc_optimizer.step()
            
            # 第二步：训练生成器
            # 生成样本的判别器输出（不分离梯度）
            fake_scores_gen = self._3d_discriminator(x0_pred)
            
            # Hinge生成器损失
            gen_loss = -torch.mean(fake_scores_gen)
            
            # 添加调试信息
            if hasattr(self, '_debug_step'):
                if self._debug_step % 100 == 0:
                    print(f"    🎯 3D Hinge GAN训练状态:")
                    print(f"      - 判别器对真实样本置信度: {real_scores.mean().item():.4f}")
                    print(f"      - 判别器对生成样本置信度: {fake_scores.mean().item():.4f}")
                    print(f"      - 判别器损失: {disc_loss.item():.6f}")
                    print(f"      - 生成器损失: {gen_loss.item():.6f}")
                    
                    # 判断训练状态
                    if real_scores.mean().item() > 0.8 and fake_scores.mean().item() < -0.8:
                        print(f"      - ✅ 判别器状态良好")
                    elif real_scores.mean().item() < 0.5:
                        print(f"      - ⚠️ 判别器对真实样本置信度偏低")
                    elif fake_scores.mean().item() > -0.5:
                        print(f"      - ⚠️ 判别器对生成样本置信度偏高")
            
            return gen_loss
            
        except Exception as e:
            print(f"3D Hinge GAN损失计算失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _init_3d_discriminator(self, device):
        """
        初始化3D latent判别器
        使用简单的MLP网络作为3D latent判别器
        """
        try:
            class Sparse3DDiscriminator(torch.nn.Module):
                def __init__(self, feat_dim=8, hidden_dims=[256, 512, 256, 128]):
                    super().__init__()
                    layers = []
                    input_dim = feat_dim
                    
                    for hidden_dim in hidden_dims:
                        layers.extend([
                            torch.nn.Linear(input_dim, hidden_dim),
                            torch.nn.LeakyReLU(0.2),
                            torch.nn.Dropout(0.3)
                        ])
                        input_dim = hidden_dim
                    
                    # 输出层：单个标量
                    layers.append(torch.nn.Linear(input_dim, 1))
                    
                    self.network = torch.nn.Sequential(*layers)
                
                def forward(self, sparse_tensor):
                    # 输入: SparseTensor with feats [N, feat_dim]
                    # 输出: [N] 判别器分数
                    features = sparse_tensor.feats
                    scores = self.network(features)
                    return scores.squeeze(-1)  # [N]
            
            # 创建判别器
            self._3d_discriminator = Sparse3DDiscriminator(feat_dim=8).to(device)
            
            # 创建判别器优化器
            self._3d_disc_optimizer = torch.optim.Adam(
                self._3d_discriminator.parameters(),
                lr=1e-4,  # 判别器学习率通常比生成器小
                betas=(0.5, 0.999)
            )
            
            print("成功初始化3D latent判别器用于hinge对抗训练")
            print(f"判别器架构: MLP with hidden_dims=[256, 512, 256, 128]")
            print(f"判别器优化器: Adam(lr=1e-4, betas=(0.5, 0.999))")
            
        except Exception as e:
            print(f"初始化3D判别器失败: {e}")
            self._3d_discriminator = None
            self._3d_disc_optimizer = None

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
    
    def save_3d_discriminator_state(self, save_path: str):
        """保存3D判别器状态"""
        try:
            if hasattr(self, '_3d_discriminator') and self._3d_discriminator is not None:
                state_dict = {
                    '3d_discriminator': self._3d_discriminator.state_dict(),
                    '3d_discriminator_optimizer': self._3d_disc_optimizer.state_dict() if self._3d_disc_optimizer else None
                }
                torch.save(state_dict, save_path)
                print(f"3D判别器状态已保存到: {save_path}")
            else:
                print("警告: 3D判别器未初始化，无法保存状态")
        except Exception as e:
            print(f"保存3D判别器状态失败: {e}")
    
    def load_3d_discriminator_state(self, load_path: str):
        """加载3D判别器状态"""
        try:
            if hasattr(self, '_3d_discriminator') and self._3d_discriminator is not None:
                state_dict = torch.load(load_path, map_location=self._3d_discriminator.device)
                
                if '3d_discriminator' in state_dict:
                    self._3d_discriminator.load_state_dict(state_dict['3d_discriminator'])
                    print(f"3D判别器状态已从 {load_path} 加载")
                
                if '3d_discriminator_optimizer' in state_dict and state_dict['3d_discriminator_optimizer'] is not None:
                    if hasattr(self, '_3d_disc_optimizer') and self._3d_disc_optimizer is not None:
                        self._3d_disc_optimizer.load_state_dict(state_dict['3d_discriminator_optimizer'])
                        print(f"3D判别器优化器状态已从 {load_path} 加载")
            else:
                print("警告: 3D判别器未初始化，无法加载状态")
        except Exception as e:
            print(f"加载3D判别器状态失败: {e}")




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
