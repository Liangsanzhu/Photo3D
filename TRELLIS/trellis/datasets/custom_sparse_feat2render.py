import os
import json
from typing import Optional
import numpy as np
import pandas as pd
import torch
from PIL import Image
import utils3d.torch
from ..modules.sparse.basic import SparseTensor
from .components import StandardDatasetBase, ImageConditionedMixin
import torch.nn.functional as F
try:
    from sklearn.decomposition import PCA
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


class CustomSparseFeat2Render(StandardDatasetBase):
    """
    自定义的SparseFeat2Render dataset，与TRELLIS兼容，不依赖CSV文件
    """
    def __init__(
        self,
        roots: str,
        image_size: int = 518,
        gt_image_size: int = 1024,

        model: str = 'dinov2_vitl14_reg',
        resolution: int = 64,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        #ss_dec_path: str ="/home/xinyue_liang/lxy/dreamposible/1w/data/7_decoder_only_training_nocond/ckpt_0050000/decoder.safetensors",#"/home/xinyue_liang/lxy/dreamposible/1w/data/7_decoder_only_training_nocond/ckpt_0004000/decoder.safetensors",#"/home/xinyue_liang/lxy/dreamposible/1w/data/7_decoder_only_training/ckpt_0004000/decoder.safetensors",
        ss_dec_path: str ="/home/xinyue_liang/.cache/huggingface/hub/models--JeffreyXiang--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors",#"/home/xinyue_liang/lxy/dreamposible/1w/data/7_encoded_slat_gpt_reszie_150/checkpoint_10500/decoder.safetensors",
        #ss_dec_path: str ="/home/xinyue_liang/lxy/dreamposible/1w/data/7_decoder_only_training_pixel/ckpt_0001500/decoder.safetensors",
        #ss_dec_path: str ="/home/xinyue_liang/lxy/dreamposible/1w/data/7_decoder_only_training_van/ckpt_0011500/decoder.safetensors",
        enable_decoder_training: bool = True,  # 新增参数控制decoder训练
        image_dir: Optional[str] = None,
        gt_dir: Optional[str] = None,
        slat_dir: Optional[str] = None,
    ):
        self.image_size = image_size
        self.gt_image_size = gt_image_size
        self.model = model
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)
        self.enable_decoder_training = enable_decoder_training  # 保存训练模式标志
        
        # 解码器相关
        self.ss_dec = None  # 当前训练的模型使用的decoder
        self.ss_dec_path = ss_dec_path
        # 启动时加载解码器，避免重复加载
        self._loading_ss_dec()
        
        # 缓存机制
        self._npz_validation_cache = {}  # 缓存NPZ文件验证结果
        
        # 自定义数据路径，换成自己的路径
        self.image_dir = "/home/xinyue_liang/lxy/dreamposible/1w/data/2_gen_input_images"
        self.gt_dir = "/home/xinyue_liang/lxy/dreamposible/1w/data/7_decode_img_gpt_new"
        self.slat_dir = slat_dir or "/home/xinyue_liang/lxy/dreamposible/1w/data/3_trellis_gen"
        
        # 归一化参数，需要自己计算一下数据集里的值
       
        self.normalization = {
            "mean": [-2.1687545776367188, -0.004347046371549368, -0.13352349400520325,
                    -0.08418072760105133, -0.5271206498146057, 0.7238689064979553,
                    -1.1414450407028198, 1.2039363384246826],
            "std": [2.377650737762451, 2.386378288269043, 2.124418020248413,
                   2.1748552322387695, 2.663944721221924, 2.371192216873169,
                   2.6217446327209473, 2.684523105621338]
        }
       
         

        
        # 创建虚拟的metadata.csv以满足StandardDatasetBase的要求
        self._create_virtual_metadata(roots)
        
        # 调用父类初始化
        super().__init__(roots)
        
    def _create_virtual_metadata(self, roots):
        """创建虚拟的metadata来满足StandardDatasetBase的要求"""
        # 获取所有可用的实例
        instances = self._get_available_instances()
        if len(instances) == 0:
            raise RuntimeError(
                "CustomSparseFeat2Render 未找到有效实例。请检查路径:\n"
                f"- slat_dir: {self.slat_dir}\n"
                f"- image_dir: {self.image_dir}\n"
                f"- gt_dir: {self.gt_dir}"
            )
        
        # 创建虚拟的metadata DataFrame，包含所有必需的字段
        metadata_data = []
        for instance_id in instances:
            metadata_data.append({
                'sha256': instance_id,
                f'feature_{self.model}': True,  # 确保有特征
                'aesthetic_score': 5.0,  # 确保美学评分足够高
                'num_voxels': 1000,  # 确保体素数量在范围内
                'cond_rendered': True,  # 确保有条件渲染
                'captions': '["test caption"]',  # 添加captions字段以防需要
            })
        
        # 创建DataFrame并保存到临时文件
        import pandas as pd
        self.metadata_df = pd.DataFrame(
            metadata_data,
            columns=[
                'sha256',
                f'feature_{self.model}',
                'aesthetic_score',
                'num_voxels',
                'cond_rendered',
                'captions',
            ],
        )
        
        # 创建临时目录和metadata文件
        temp_root = roots.split(',')[0]  # 使用第一个root
        os.makedirs(temp_root, exist_ok=True)
        metadata_path = os.path.join(temp_root, 'metadata.csv')
        self.metadata_df.to_csv(metadata_path, index=False)
        
        # 创建loads属性用于BalancedResumableSampler
        # 基于slat_dir的npz实际行数估计每实例负载
        self.loads = []
        for instance_id in instances:
            try:
                low_npz = None
                for potential_dir in [f"{instance_id}", f"grid_{instance_id}"]:
                    test_path = os.path.join(self.slat_dir, potential_dir, "slat.npz")
                    if os.path.exists(test_path):
                        low_npz = test_path
                        break

                if low_npz and os.path.exists(low_npz):
                    with np.load(low_npz) as data:
                        load = int(data['feats'].shape[0]) if 'feats' in data else 1000
                else:
                    load = 1000
            except Exception:
                load = 1000
            self.loads.append(load)
        
    def _loading_ss_dec(self):
        """
        加载稀疏结构解码器
        
        加载策略：
        1. 首先尝试使用 ss_dec_path 指定的路径：
           - 如果是文件路径，直接加载权重文件
           - 如果是目录路径，在目录中查找权重文件
        2. 如果指定路径加载失败，尝试多个预训练模型路径
        3. 如果所有预训练模型都失败，使用随机初始化的解码器
        """
        if self.ss_dec is not None:
            return
        
        try:
            import json
            from trellis import models
            
            # 首先尝试直接使用指定的绝对路径作为权重文件
            print(f"尝试加载指定的解码器路径: {self.ss_dec_path}")
            
            # 检查指定路径是否是直接的权重文件
            if os.path.isfile(self.ss_dec_path):
                # 如果指定的是直接的权重文件
                final_ckpt_path = self.ss_dec_path
                print(f"检测到直接权重文件: {final_ckpt_path}")
                
                # 尝试从权重文件所在目录加载配置文件
                config_dir = os.path.dirname(self.ss_dec_path)
                config_path = os.path.join(config_dir, 'config.json')
                
                if not os.path.exists(config_path):
                    # 如果配置文件不存在，尝试其他可能的配置文件路径
                    alternative_config_paths = [
                        "/home/xinyue_liang/lxy/aaa_Trellis/TRELLIS/configs/vae/custom_sparse_feat2render.json",
                        "/home/xinyue_liang/lxy/aaa_Trellis/TRELLIS/pretrained_weights/config.json",
                        "/home/xinyue_liang/lxy/aaa_Trellis/TRELLIS/configs/vae/sparse_feat2render.json"
                    ]
                    
                    for alt_config_path in alternative_config_paths:
                        if os.path.exists(alt_config_path):
                            config_path = alt_config_path
                            print(f"使用配置文件: {config_path}")
                            break
                    else:
                        print("警告: 未找到配置文件，使用默认配置")
                        config_path = "/home/xinyue_liang/lxy/aaa_Trellis/TRELLIS/configs/vae/custom_sparse_feat2render.json"
                
            else:
                # 如果指定的是目录，尝试在目录中查找权重文件
                config_path = os.path.join(self.ss_dec_path, 'config.json')
                ckpt_path = os.path.join(self.ss_dec_path, 'decoder.pt')
                safetensors_path = os.path.join(self.ss_dec_path, 'decoder.safetensors')
                
                # 确定权重文件
                if os.path.exists(safetensors_path):
                    final_ckpt_path = safetensors_path
                elif os.path.exists(ckpt_path):
                    final_ckpt_path = ckpt_path
                else:
                    # 在目录中查找其他可能的权重文件
                    if os.path.exists(self.ss_dec_path):
                        ckpt_files = [f for f in os.listdir(self.ss_dec_path) 
                                     if f.endswith(('.pt', '.safetensors')) and 
                                     ('decoder' in f.lower() or 'ss_dec' in f.lower())]
                        if ckpt_files:
                            final_ckpt_path = os.path.join(self.ss_dec_path, ckpt_files[0])
                            print(f"在目录中找到权重文件: {final_ckpt_path}")
                        else:
                            raise FileNotFoundError(f"在指定目录中未找到权重文件: {self.ss_dec_path}")
                    else:
                        raise FileNotFoundError(f"指定的路径不存在: {self.ss_dec_path}")
            
            # 加载配置文件
            if os.path.exists(config_path):
                cfg = json.load(open(config_path, 'r'))
                decoder_name = cfg['models']['decoder']['name']
                decoder_args = cfg['models']['decoder']['args']
                decoder = getattr(models, decoder_name)(**decoder_args)
            else:
                print("配置文件不存在，尝试使用预训练模型")
                raise FileNotFoundError("配置文件不存在")
            
            print(f"使用权重文件: {final_ckpt_path}")
            
            # 加载权重
            if final_ckpt_path.endswith('.safetensors'):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(final_ckpt_path)
                except ImportError:
                    state_dict = torch.load(final_ckpt_path, map_location='cpu')
            else:
                state_dict = torch.load(final_ckpt_path, map_location='cpu', weights_only=True)
            
            # 处理权重文件结构
            if isinstance(state_dict, dict):
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                
                # 处理module.前缀 - 这是关键修复
                # 检查是否需要去除module.前缀
                has_module_prefix = any(name.startswith('module.') for name in state_dict.keys())
                if has_module_prefix:
                    print(f"检测到module.前缀，正在去除...")
                    new_state_dict = {}
                    for name, param in state_dict.items():
                        if name.startswith('module.'):
                            new_name = name[7:]  # 去除 'module.' 前缀
                        else:
                            new_name = name
                        new_state_dict[new_name] = param
                    state_dict = new_state_dict
                    print(f"✅ 成功去除module.前缀，参数数量: {len(state_dict)}")
            
            decoder.load_state_dict(state_dict, strict=False)
            # 根据训练模式设置decoder状态
            if self.enable_decoder_training:
                self.ss_dec = decoder.cuda().train()  # 训练模式
                # 启用梯度
                for param in self.ss_dec.parameters():
                    param.requires_grad_(True)
                print("解码器加载成功 - 训练模式")
            else:
                self.ss_dec = decoder.cuda().eval()  # 评估模式
                # 禁用梯度
                for param in self.ss_dec.parameters():
                    param.requires_grad_(False)
                print("解码器加载成功 - 评估模式")
            
        except Exception as e:
            print(f"加载指定解码器失败: {e}")
            print("尝试使用预训练模型...")
            
            try:
                # 尝试加载预训练模型
                from trellis import models
                
                # 尝试多个预训练模型路径
                pretrained_paths = [
                    # Sparse latent decoder expects SparseTensor input.
                    'microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
                    'JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
                    '/home/xinyue_liang/.cache/huggingface/hub/models--JeffreyXiang--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors',
                    # Keep old fallbacks as last resort.
                    'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
                    'JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
                ]
                
                for pretrained_path in pretrained_paths:
                    try:
                        if pretrained_path.endswith('.safetensors'):
                            # 如果是本地文件，需要先创建模型再加载权重
                            import json
                            config_path = "/home/xinyue_liang/lxy/aaa_Trellis/TRELLIS/configs/vae/custom_sparse_feat2render.json"
                            if os.path.exists(config_path):
                                cfg = json.load(open(config_path, 'r'))
                                decoder_name = cfg['models']['decoder']['name']
                                decoder_args = cfg['models']['decoder']['args']
                                decoder = getattr(models, decoder_name)(**decoder_args)
                                
                                from safetensors.torch import load_file
                                state_dict = load_file(pretrained_path)
                                decoder.load_state_dict(state_dict, strict=False)
                                # 根据训练模式设置decoder状态
                                if self.enable_decoder_training:
                                    self.ss_dec = decoder.cuda().train()  # 训练模式
                                    for param in self.ss_dec.parameters():
                                        param.requires_grad_(True)
                                    print(f"成功加载预训练权重: {pretrained_path} - 训练模式")
                                else:
                                    self.ss_dec = decoder.cuda().eval()  # 评估模式
                                    for param in self.ss_dec.parameters():
                                        param.requires_grad_(False)
                                    print(f"成功加载预训练权重: {pretrained_path} - 评估模式")
                                return
                        else:
                            # 如果是HuggingFace模型
                            decoder = models.from_pretrained(pretrained_path)
                            # 根据训练模式设置decoder状态
                            if self.enable_decoder_training:
                                self.ss_dec = decoder.cuda().train()  # 训练模式
                                for param in self.ss_dec.parameters():
                                    param.requires_grad_(True)
                                print(f"成功加载预训练模型: {pretrained_path} - 训练模式")
                            else:
                                self.ss_dec = decoder.cuda().eval()  # 评估模式
                                for param in self.ss_dec.parameters():
                                    param.requires_grad_(False)
                                print(f"成功加载预训练模型: {pretrained_path} - 评估模式")
                            return
                    except Exception as pretrained_e:
                        print(f"尝试预训练路径失败 {pretrained_path}: {pretrained_e}")
                        continue
                
                # 如果所有预训练模型都失败，使用随机初始化的解码器
                print("所有预训练模型加载失败，使用随机初始化的解码器")
                import json
                config_path = "/home/xinyue_liang/lxy/aaa_Trellis/TRELLIS/configs/vae/custom_sparse_feat2render.json"
                if os.path.exists(config_path):
                    cfg = json.load(open(config_path, 'r'))
                    decoder_name = cfg['models']['decoder']['name']
                    decoder_args = cfg['models']['decoder']['args']
                    decoder = getattr(models, decoder_name)(**decoder_args)
                    # 根据训练模式设置decoder状态
                    if self.enable_decoder_training:
                        self.ss_dec = decoder.cuda().train()  # 训练模式
                        for param in self.ss_dec.parameters():
                            param.requires_grad_(True)
                        print("使用随机初始化的解码器 - 训练模式")
                    else:
                        self.ss_dec = decoder.cuda().eval()  # 评估模式
                        for param in self.ss_dec.parameters():
                            param.requires_grad_(False)
                        print("使用随机初始化的解码器 - 评估模式")
                else:
                    self.ss_dec = None
                    print("无法创建解码器")
                    
            except Exception as e2:
                print(f"创建随机初始化解码器也失败: {e2}")
                self.ss_dec = None
    
    def _delete_ss_dec(self):
        """删除解码器以释放内存"""
        del self.ss_dec
        self.ss_dec = None
    
    def clear_cache(self):
        """清理缓存以释放内存"""
        self._npz_validation_cache.clear()
    
    def decode_latent_grad(self, z, batch_size=4, sample_type=None):
        """可传播梯度的解码：不使用@torch.no_grad，且保持ss_dec常驻。
        仅用于训练时需要梯度的场景。"""
        self._loading_ss_dec()
        if self.ss_dec is None:
            raise RuntimeError("解码器加载失败")
        if hasattr(z, 'feats') and hasattr(z, 'coords'):
            # SparseTensor解码（带反归一化）
            if self.normalization is not None:
                mean_tensor = torch.tensor(self.normalization['mean'], device=z.feats.device, dtype=z.feats.dtype)
                std_tensor = torch.tensor(self.normalization['std'], device=z.feats.device, dtype=z.feats.dtype)
                denormalized_feats = z.feats * std_tensor + mean_tensor
                z = z.replace(denormalized_feats)
            ss = self.ss_dec(z)
            return ss
        else:
            # 普通tensor批处理
            ss = []
            zz = z
            if self.normalization is not None:
                zz = z * torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1).to(z.device) + \
                     torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1).to(z.device)
            for i in range(0, zz.shape[0], batch_size):
                ss.append(self.ss_dec(zz[i:i+batch_size]))
            ss = torch.cat(ss, dim=0)
            return ss

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4, sample_type=None):
        """解码潜在表示 - 保持decoder常驻，避免重复加载"""
        # 只在decoder未加载时加载一次
        if not hasattr(self, 'ss_dec') or self.ss_dec is None:
            self._loading_ss_dec()
        
        if self.ss_dec is None:
            raise RuntimeError("解码器加载失败")
        
        if hasattr(z, 'feats') and hasattr(z, 'coords'):
            # SparseTensor解码
            # 调试信息已移除
            
            # 对所有样本进行反归一化
            if self.normalization is not None:
                mean_tensor = torch.tensor(self.normalization['mean'], device=z.feats.device, dtype=z.feats.dtype)
                std_tensor = torch.tensor(self.normalization['std'], device=z.feats.device, dtype=z.feats.dtype)
                denormalized_feats = z.feats * std_tensor + mean_tensor
                
                # 使用replace方法创建新的SparseTensor
                z = z.replace(denormalized_feats)
            else:
                pass
            
            ss = self.ss_dec(z)
            return ss
        else:
            # 普通张量批处理解码
            print(f"普通张量解码，输入形状: {z.shape}")
            ss = []
            if self.normalization is not None:
                z = z * torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1).to(z.device) + torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1).to(z.device)
            
            for i in range(0, z.shape[0], batch_size):
                ss.append(self.ss_dec(z[i:i+batch_size]))
            ss = torch.cat(ss, dim=0)
            return ss
    
    def decode_latent_grad(self, z, batch_size=4, sample_type=None):
        """解码潜在表示 - 保持decoder常驻，避免重复加载"""
        # 只在decoder未加载时加载一次
        if not hasattr(self, 'ss_dec') or self.ss_dec is None:
            self._loading_ss_dec()
        
        if self.ss_dec is None:
            raise RuntimeError("解码器加载失败")
        
        if hasattr(z, 'feats') and hasattr(z, 'coords'):
            # SparseTensor解码
            # 调试信息已移除
            
            # 对所有样本进行反归一化
            if self.normalization is not None:
                mean_tensor = torch.tensor(self.normalization['mean'], device=z.feats.device, dtype=z.feats.dtype)
                std_tensor = torch.tensor(self.normalization['std'], device=z.feats.device, dtype=z.feats.dtype)
                denormalized_feats = z.feats * std_tensor + mean_tensor
                
                # 使用replace方法创建新的SparseTensor
                z = z.replace(denormalized_feats)
            else:
                pass
            
            ss = self.ss_dec(z)
            return ss
        else:
            # 普通张量批处理解码
            print(f"普通张量解码，输入形状: {z.shape}")
            ss = []
            if self.normalization is not None:
                z = z * torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1).to(z.device) + torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1).to(z.device)
            
            for i in range(0, z.shape[0], batch_size):
                ss.append(self.ss_dec(z[i:i+batch_size]))
            ss = torch.cat(ss, dim=0)
            return ss
    def _validate_npz_file(self, npz_path):
        """验证NPZ文件格式是否正确"""
        # 检查缓存
        if npz_path in self._npz_validation_cache:
            return self._npz_validation_cache[npz_path]
        
        try:
            with np.load(npz_path) as data:
                # 检查文件是否包含feats键
                if 'feats' not in data:
                    result = (False, "NPZ文件中没有feats键")
                    self._npz_validation_cache[npz_path] = result
                    return result
                
                # 检查feats是否是numpy数组
                feats = data['feats']
                if not isinstance(feats, np.ndarray):
                    result = (False, "feats不是numpy数组")
                    self._npz_validation_cache[npz_path] = result
                    return result
                
                # 检查feats的形状
                if len(feats.shape) != 2:
                    result = (False, f"feats形状不正确: {feats.shape}")
                    self._npz_validation_cache[npz_path] = result
                    return result
                
                result = (True, "NPZ文件格式正确")
                self._npz_validation_cache[npz_path] = result
                return result
        except Exception as e:
            result = (False, f"NPZ文件验证失败: {str(e)}")
            self._npz_validation_cache[npz_path] = result
            return result
    

    
    def _get_available_instances(self):
        """返回可用的实例ID（仅依赖slat_dir）。"""
        img_ids = set()
        low_ids = set()

        # 从图像目录推断id（可选）
        if os.path.exists(self.image_dir):
            for file in os.listdir(self.image_dir):
                if file.endswith('.png'):
                    img_ids.add(os.path.splitext(file)[0].split("_")[-1])

        # 从slat_dir有效npz+coords推断id（必须）
        if os.path.exists(self.slat_dir):
            for d in os.listdir(self.slat_dir):
                # 如果目录名以grid_开头，提取后面的数字部分作为ID
                if d.startswith('grid_'):
                    iid = d.split('_', 1)[1]  # 提取grid_后面的部分
                else:
                    iid = d
                npz_ok = os.path.exists(os.path.join(self.slat_dir, d, 'slat.npz'))
                coords_ok = os.path.exists(os.path.join(self.slat_dir, d, 'coords.pt'))
                if npz_ok and coords_ok:
                    low_ids.add(iid)

        # 调试：打印一些样本ID来检查格式
        print(f"slat_ids 样本 (前10个): {sorted(list(low_ids))[:10]}")
        if len(img_ids) > 0:
            print(f"img_ids 样本 (前10个): {sorted(list(img_ids))[:10]}")

        # 仅以low_ids为基准，若img_ids非空则进一步与其相交
        valid = set(low_ids)
        print(f"slat_ids 基础数量: {len(valid)}")
        
        if len(img_ids) > 0:
            before_img_filter = len(valid)
            valid = valid & img_ids
            print(f"加入 img_ids 过滤后: {before_img_filter} -> {len(valid)}")

        valid_list = sorted(list(valid))
        # 进一步验证：条件图与GT图是否存在（若image_dir/gt_dir存在）
        if os.path.exists(self.image_dir) and os.path.exists(self.gt_dir):
            filtered = []
            for iid in valid_list:
                iid_str = str(iid)
                cond_ok = False
                gt_ok = False
                # 条件图候选
                cond_candidates = [
                    f"grid_{iid_str}.png", f"{iid_str}.png",
                    f"grid_{iid_str}.jpg", f"{iid_str}.jpg",
                    f"grid_{iid_str}.jpeg", f"{iid_str}.jpeg",
                ]
                for name in cond_candidates:
                    if os.path.exists(os.path.join(self.image_dir, name)):
                        cond_ok = True
                        break
                # GT图候选
                gt_candidates = [
                    f"grid_{iid_str}.png", f"{iid_str}.png",
                    f"grid_{iid_str}.jpg", f"{iid_str}.jpg",
                    f"grid_{iid_str}.jpeg", f"{iid_str}.jpeg",
                ]
                for name in gt_candidates:
                    if os.path.exists(os.path.join(self.gt_dir, name)):
                        gt_ok = True
                        break
                if cond_ok and gt_ok:
                    filtered.append(iid)
            print(f"加入 条件图+GT 存在性校验后: {len(valid_list)} -> {len(filtered)}")
            valid_list = sorted(filtered)
        if len(valid_list) == 0:
            print(f"警告: 未找到任何有效实例，请检查数据路径")
            print(f"slat_dir: {self.slat_dir} (找到 {len(low_ids)} 个实例)")
            if len(img_ids) > 0:
                print(f"image_dir: {self.image_dir} (找到 {len(img_ids)} 个实例)")
        else:
            print(f"找到 {len(valid_list)} 个有效实例")
        return valid_list
    
    def filter_metadata(self, metadata):
        """过滤元数据"""
        stats = {}
        # 如果有feature列，过滤有特征的数据
        if f'feature_{self.model}' in metadata.columns:
            metadata = metadata[metadata[f'feature_{self.model}']]
            stats['With features'] = len(metadata)
        
        # 如果有aesthetic_score列，过滤美学评分
        if 'aesthetic_score' in metadata.columns:
            metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
            stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        
        # 如果有num_voxels列，过滤体素数量
        if 'num_voxels' in metadata.columns:
            metadata = metadata[metadata['num_voxels'] <= self.max_num_voxels]
            stats[f'Num voxels <= {self.max_num_voxels}'] = len(metadata)
        
        return metadata, stats
    
    def get_instance(self, root, instance):
        """获取单个实例的完整数据 - 参考dataset_mul.py的实现"""
        # 使用instance作为instance_id
        instance_id = instance
        #print(f"instance_id: {instance_id}")
        #instance_id=instance_id%10+1660
      
        try:
            image_data = self._get_image(instance_id)
            feat_data = self._get_feat(instance_id)
            
            # 如果特征数据加载失败，返回None让调用者跳过这个实例
            if feat_data is None:
                print(f"实例 {instance_id} 特征数据加载失败，跳过")
                return None
            
            # 兼容ImageConditionedSLat接口
            result = {**image_data, **feat_data}
            
            # 添加cond字段（条件图像）
            result['cond'] = result['image']
            # 记录instance_id
            result['instance_id'] = instance_id
            
            return result
        except Exception as e:
            print(f"获取实例 {instance_id} 失败: {e}")
            return None  # 返回None让调用者跳过这个实例

    def _get_image(self, instance_id):
        """获取图像和相机参数 - 参考dataset_mul.py的实现"""
        # 确保instance_id是字符串类型
        instance_id_str = str(instance_id)
        
        # 查找对应的图像文件，兼容多种命名与后缀
        candidate_names = [
            f"grid_{instance_id_str}.png",
            f"{instance_id_str}.png",
            f"grid_{instance_id_str}.jpg",
            f"{instance_id_str}.jpg",
            f"grid_{instance_id_str}.jpeg",
            f"{instance_id_str}.jpeg",
        ]
        image_file = None
        for name in candidate_names:
            if os.path.exists(os.path.join(self.image_dir, name)):
                image_file = name
                break
        
        if image_file is None:
            # 如果找不到对应的文件，创建一个默认图像
            image = Image.new('RGB', (self.image_size, self.image_size), color=(128, 128, 128))
            gt_image = Image.new('RGB', (self.gt_image_size, self.gt_image_size), color=(128, 128, 128))
        else:
            # 读取条件图像
            cond_path = os.path.join(self.image_dir, image_file)
            image = Image.open(cond_path).convert('RGB')
            
            # 读取GT图像：同样进行多候选匹配
            gt_image = None
            gt_path = None
            gt_candidate_names = [
                f"grid_{instance_id_str}.png",
                f"{instance_id_str}.png",
                f"grid_{instance_id_str}.jpg",
                f"{instance_id_str}.jpg",
                f"grid_{instance_id_str}.jpeg",
                f"{instance_id_str}.jpeg",
            ]
            for name in gt_candidate_names:
                candidate = os.path.join(self.gt_dir, name)
                if os.path.exists(candidate):
                    gt_path = candidate
                    break
            if gt_path is not None:
                gt_image = Image.open(gt_path).convert('RGB')
            else:
                gt_image = Image.new('RGB', (self.image_size, self.image_size), color=(128, 128, 128))
        
        # 调整图像大小
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        gt_image = gt_image.resize((self.gt_image_size, self.gt_image_size), Image.Resampling.LANCZOS)
        #print("gt_path",gt_path)
        #print("cond_path",cond_path)

        # 转换为tensor
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        gt_image = torch.tensor(np.array(gt_image)).permute(2, 0, 1).float() / 255.0
        
        # 生成相机参数
        extrinsics, intrinsics = self._generate_camera_params()
        
        return {
            'image': image,
            'gt_image': gt_image,
            'alpha': torch.ones(1, self.image_size, self.image_size),
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            # 调试：保存图片路径
            'cond_path': cond_path if 'cond_path' in locals() else None,
            'gt_path': gt_path if 'gt_path' in locals() else None,
        }
    
    def _generate_camera_params(self):
        """生成相机参数"""
        fov = 60.0
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(
            torch.tensor(fov * np.pi / 180.0), 
            torch.tensor(fov * np.pi / 180.0)
        )
        
        c2w = torch.eye(4)
        c2w[:3, 3] = torch.tensor([0.0, 0.0, 2.0])
        c2w[:3, 1:3] *= -1
        extrinsics = torch.inverse(c2w)
        
        return extrinsics, intrinsics
    
    def _get_feat(self, instance_id):
        """获取特征数据 - 参考dataset_mul.py的实现"""
        # 确保instance_id是字符串类型
        instance_id_str = str(instance_id)
        
        # 构建路径
        # slat_dir中的目录可能是grid_开头的，需要检查实际存在的目录格式
        slat_main_path = None
        for potential_dir in [f"{instance_id_str}", f"grid_{instance_id_str}"]:
            test_path = os.path.join(self.slat_dir, potential_dir)
            if os.path.exists(test_path):
                slat_main_path = test_path
                break
        
        if slat_main_path is None:
            print(f"找不到 slat_dir 中的实例目录: {instance_id_str}")
            return None

        slat_path = os.path.join(slat_main_path, "slat.npz")
        coords_path = os.path.join(slat_main_path, "coords.pt")

        # 加载slat特征 - 如果文件不存在或加载失败，直接跳过该样本
        if not os.path.exists(slat_path):
            print(f"slat_path not exists: {slat_path} - 跳过该样本")
            return None
        
        # 先验证NPZ文件格式
        is_valid, message = self._validate_npz_file(slat_path)
        if not is_valid:
            print(f"NPZ文件格式错误 {slat_path}: {message} - 跳过该样本")
            return None
        
        try:
            feats = np.load(slat_path)
            slat_feats = torch.from_numpy(feats['feats']).float()
        except Exception as e:
            print(f"加载NPZ文件失败 {slat_path}: {e} - 跳过该样本")
            return None
        
        # 加载坐标
        coords = None
        
        # 优先从slat的NPZ加载坐标，其次使用coords.pt兜底
        try:
            slat_data = np.load(slat_path)
            if 'coords' in slat_data:
                coords = torch.from_numpy(slat_data['coords']).int()
            elif os.path.exists(coords_path):
                coords = torch.load(coords_path, map_location='cpu').int()
        except Exception as e:
            print(f"加载坐标失败: {e}")
            return None

        if coords is None:
            print(f"无法获取坐标（NPZ缺少coords且coords.pt不存在）: {instance_id_str}")
            return None
        
        # 确保坐标是3维的（x, y, z）
        if coords.shape[1] == 4:
            # 如果已经是4维，去掉第一个维度（通常是batch维度）
            coords = coords[:, 1:]
        elif coords.shape[1] != 3:
            raise ValueError(f"Unexpected coords shape: {coords.shape}")
        
        # 确保坐标范围在[0, resolution)内
        coords = torch.clamp(coords, 0, self.resolution - 1)
        
        # 确保特征和坐标数量匹配
        num_coords = coords.shape[0]
        
        # 处理slat特征数量不匹配的情况
        if slat_feats.shape[0] != num_coords:
            print(f"特征数量不匹配: slat={slat_feats.shape[0]}, coords={num_coords}")
            if slat_feats.shape[0] > num_coords:
                # 如果特征数量多于坐标，随机采样匹配的数量
                indices = torch.randperm(slat_feats.shape[0])[:num_coords]
                slat_feats = slat_feats[indices]
                print(f"随机采样slat: {slat_feats.shape[0]} -> {num_coords}")
            else:
                # 如果特征数量少于坐标，随机重复特征
                repeat_count = num_coords - slat_feats.shape[0]
                repeat_indices = torch.randint(0, slat_feats.shape[0], (repeat_count,))
                repeat_feats = slat_feats[repeat_indices]
                slat_feats = torch.cat([slat_feats, repeat_feats], dim=0)
                print(f"随机重复slat: {slat_feats.shape[0] - repeat_count} -> {num_coords}")
        
        # 应用归一化
        if self.normalization is not None:
            mean = torch.tensor(self.normalization['mean'])[None]
            std = torch.tensor(self.normalization['std'])[None]
            slat_feats = (slat_feats - mean) / std
        
        # 当前已不依赖win_dir：风格/结构都使用low特征
        result = {
            'coords': coords,
            'slat': slat_feats,
            'slat_path': slat_path,
        }
        
        return result
    


    @torch.no_grad()
    def visualize_sample(self, sample):
        """可视化样本"""
        # 检查是否是包含sample_gt和sample的完整样本字典
        if isinstance(sample, dict) and 'sample_gt' in sample and 'sample' in sample:
            # 这是训练器传递的完整样本字典，包含真实样本和生成样本
            gt_sample = sample['sample_gt']
            gen_sample = sample['sample']
            
            # 提取feats
            if hasattr(gt_sample['value'], 'feats') and hasattr(gen_sample['value'], 'feats'):
                gt_feats = gt_sample['value'].feats
                gen_feats = gen_sample['value'].feats
                
                # 比较分布
                self.compare_feats_distributions(gen_feats, gt_feats, "当前模型", "真实样本")
            
            # 返回生成样本的可视化
            return self._visualize_x0_tensor(gen_sample['value'], sample_type="model_generated")
        
        # 处理单个样本的情况
        if isinstance(sample, dict):
            # 如果是完整的样本信息（包含value, type, sample_type等）
            if 'value' in sample and 'type' in sample:
                sample_value = sample['value']
                sample_type = sample.get('sample_type', 'unknown')
                
                # 支持latent_pca可视化
                if sample.get('type') == 'latent_pca' or sample_type == 'latent_pca':
                    if hasattr(sample_value, 'feats') and hasattr(sample_value, 'coords'):
                        return self._visualize_latent_pca(sample_value)
                    else:
                        return torch.ones(3, 512, 512)
                
                if hasattr(sample_value, 'feats') and hasattr(sample_value, 'coords'):
                    return self._visualize_x0_tensor(sample_value, sample_type=sample_type)
                else:
                    return sample_value
            elif 'x_0' in sample:
                # 从sample_dict中获取sample_type
                sample_type = sample.get('sample_type', 'model_generated')
                return self._visualize_x0_tensor(sample['x_0'], sample_type=sample_type)
            elif 'image' in sample: 
                return sample['image']
            elif 'gt_image' in sample:
                # 如果有slat，也检查GT的latent
                if 'slat' in sample:
                    gt_tensor = SparseTensor(feats=sample['slat'], coords=sample['coords'])
                    return self._visualize_x0_tensor(gt_tensor, sample_type="ground_truth")
                return sample['gt_image']
            else:
                return torch.ones(3, 512, 512)
        elif hasattr(sample, 'feats') and hasattr(sample, 'coords'):
            # 如果是SparseTensor，假设是模型生成的样本
            return self._visualize_x0_tensor(sample, sample_type="model_generated")
        else: 
            return torch.ones(3, 512, 512)
    

    
    def _visualize_x0_tensor(self, x0_tensor, sample_type=None):
        """解码SparseTensor为3D高斯并渲染为2D图像"""
        if not hasattr(x0_tensor, 'feats') or not hasattr(x0_tensor, 'coords'):
            raise ValueError("x_0 tensor格式不正确，缺少feats或coords属性")
        
        # 检查特征值范围
        feats = x0_tensor.feats
        
        # 只输出关键统计信息
        if sample_type == "model_generated":
            # 按通道统计
            channel_means = feats.mean(dim=0)
            channel_stds = feats.std(dim=0)
            channel_mins = feats.min(dim=0)[0]
            channel_maxs = feats.max(dim=0)[0]
            print(f"[生成样本] 通道均值={[f'{m:.4f}' for m in channel_means]}, 通道标准差={[f'{s:.4f}' for s in channel_stds]}")
        elif sample_type == "ground_truth":
            # 按通道统计
            channel_means = feats.mean(dim=0)
            channel_stds = feats.std(dim=0)
            channel_mins = feats.min(dim=0)[0]
            channel_maxs = feats.max(dim=0)[0]
            print(f"[真实样本] 通道均值={[f'{m:.4f}' for m in channel_means]}, 通道标准差={[f'{s:.4f}' for s in channel_stds]}")
        # 调整特征维度为8
        feat_dim = feats.shape[1]
        if feat_dim != 8:
            if feat_dim < 8:
                new_feats = torch.zeros(feats.shape[0], 8, device=feats.device, dtype=feats.dtype)
                new_feats[:, :feat_dim] = feats
                feats = new_feats
            else:
                feats = feats[:, :8]
        
        # 注意：decode_latent方法已经处理了SparseTensor的反归一化，这里不需要再次反归一化
        
        # 创建新的SparseTensor
        x0_tensor = SparseTensor(feats=feats, coords=x0_tensor.coords)
        
        # 使用当前训练的decoder解码
        decoded_gaussian = self.decode_latent(x0_tensor, sample_type=sample_type)
        
        # 渲染为2D图像
        return self._render_gaussian(decoded_gaussian)

    def _visualize_latent_pca(self, x0_tensor, out_size=512):
        """将SparseTensor的latent用PCA映射到RGB，并按XY投影到2D图像可视化。
        返回: [3, out_size, out_size] 的tensor，范围[0,1]
        """
        feats = x0_tensor.feats.detach().cpu()
        coords = x0_tensor.coords.detach().cpu()

        # 处理坐标维度（可能为[N,3]或[N,4]，若为4则去掉batch维）
        if coords.shape[1] == 4:
            coords_xy = coords[:, 1:3]
        else:
            coords_xy = coords[:, 0:2]

        # PCA到3维
        if _HAS_SKLEARN:
            pca = PCA(n_components=3)
            feats_pca = torch.from_numpy(pca.fit_transform(feats.numpy())).float()
        else:
            # 无sklearn时，用SVD近似前三主成分
            feats_center = feats - feats.mean(0, keepdim=True)
            U, S, Vh = torch.linalg.svd(feats_center, full_matrices=False)
            comps = Vh[:3].T  # [C,3]
            feats_pca = feats_center @ comps  # [N,3]

        # 归一化到[0,1]
        min_vals = feats_pca.min(dim=0).values
        max_vals = feats_pca.max(dim=0).values
        feats_pca_norm = (feats_pca - min_vals) / (max_vals - min_vals + 1e-8)

        # 将颜色填充到2D网格（按XY投影）
        H = W = self.resolution
        img = torch.zeros(H, W, 3)
        cnt = torch.zeros(H, W, 1)

        # 限制坐标范围
        coords_xy = torch.clamp(coords_xy, 0, self.resolution - 1).long()
        for i in range(coords_xy.shape[0]):
            x, y = coords_xy[i, 0].item(), coords_xy[i, 1].item()
            img[y, x] += feats_pca_norm[i]
            cnt[y, x] += 1

        cnt_safe = torch.where(cnt > 0, cnt, torch.ones_like(cnt))
        img = img / cnt_safe
        img = img.permute(2, 0, 1).contiguous()  # [3,H,W]

        # 上采样到out_size
        if img.shape[-1] != out_size or img.shape[-2] != out_size:
            img = F.interpolate(img.unsqueeze(0), size=(out_size, out_size), mode='nearest').squeeze(0)

        img = img.clamp(0, 1)
        return img
    

    

    
    def _render_gaussian(self, decoded_gaussian):
        """渲染3D高斯为2D图像"""
        from trellis.utils import render_utils
        
        if isinstance(decoded_gaussian, (list, tuple)):
            decoded_gaussian = decoded_gaussian[0]
        
        # 调试颜色统计输出已移除
        
        # 渲染4个视角
        video_gaussian, _ = render_utils.render_around_view(decoded_gaussian, r=1.7)
        video_gaussian = torch.stack([frame for frame in video_gaussian])
        
        # 渲染后颜色调试输出已移除
        
        # 选择4个角度生成4宫格
        total_frames = video_gaussian.shape[0]
        selected_indices = [0, total_frames//4, total_frames//2, 3*total_frames//4]

        # 创建4宫格tensor
        H, W = video_gaussian.shape[-2:]
        grid_tensor = torch.zeros(3, H * 2, W * 2, device=video_gaussian.device, dtype=video_gaussian.dtype)

        # 拼接4个视角到2x2网格
        for j, idx in enumerate(selected_indices):
            row = j // 2
            col = j % 2
            frame = video_gaussian[idx]
            grid_tensor[:, row*H:(row+1)*H, col*W:(col+1)*W] = frame

        # 确保值在[0,1]范围内
        grid_tensor = torch.clamp(grid_tensor, 0, 1)
        
        # 如果图像太小，上采样到512x512
        if grid_tensor.shape[-1] < 512 or grid_tensor.shape[-2] < 512:
            grid_tensor = F.interpolate(grid_tensor.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
        
        return grid_tensor

    def compare_feats_distributions(self, generated_feats, ground_truth_feats, sample_name1="生成样本", sample_name2="真实样本"):
        """比较两个样本的feats分布"""
        # 按通道计算差异
        gen_means = generated_feats.mean(dim=0)
        gen_stds = generated_feats.std(dim=0)
        gt_means = ground_truth_feats.mean(dim=0)
        gt_stds = ground_truth_feats.std(dim=0)
        
        # 计算每个通道的差异
        mean_diffs = torch.abs(gen_means - gt_means)
        std_diffs = torch.abs(gen_stds - gt_stds)
        
        # 计算平均差异
        avg_mean_diff = mean_diffs.mean().item()
        avg_std_diff = std_diffs.mean().item()
        max_mean_diff = mean_diffs.max().item()
        max_std_diff = std_diffs.max().item()
        
        # 计算相关性
        try:
            gen_flat = generated_feats.flatten()
            gt_flat = ground_truth_feats.flatten()
            min_len = min(len(gen_flat), len(gt_flat))
            correlation = torch.corrcoef(torch.stack([gen_flat[:min_len], gt_flat[:min_len]]))[0, 1].item()
        except:
            correlation = 0.0
        
        # 输出结果
        print(f"[分布一致性 {sample_name1} vs {sample_name2}] 平均均值差异={avg_mean_diff:.4f}, 平均标准差差异={avg_std_diff:.4f}, 最大均值差异={max_mean_diff:.4f}, 最大标准差差异={max_std_diff:.4f}, 相关系数={correlation:.4f}")
        
        # 判断一致性
        if avg_mean_diff < 0.1 and avg_std_diff < 0.1 and max_mean_diff < 0.2 and max_std_diff < 0.2 and correlation > 0.8:
            print(f"[分布一致性 {sample_name1} vs {sample_name2}] ✓ 分布一致")
        elif avg_mean_diff < 0.3 and avg_std_diff < 0.3 and max_mean_diff < 0.5 and max_std_diff < 0.5 and correlation > 0.6:
            print(f"[分布一致性 {sample_name1} vs {sample_name2}] ○ 分布较为一致")
        else:
            print(f"[分布一致性 {sample_name1} vs {sample_name2}] ✗ 分布不一致")
    


    @staticmethod
    def collate_fn(batch, split_size=None):
        # 过滤掉None值
        valid_batch = [b for b in batch if b is not None]
        
        if not valid_batch:
            raise ValueError("批次中没有有效的数据")
        
        pack = {}
        coords = []
        for i, b in enumerate(valid_batch):
            # 确保坐标是3维的（x, y, z），然后添加batch维度
            if b['coords'].shape[1] == 4:
                # 如果已经是4维，去掉第一个维度（通常是batch维度）
                coords_3d = b['coords'][:, 1:]  # 去掉第一维，保留x, y, z
            elif b['coords'].shape[1] == 3:
                # 已经是3维
                coords_3d = b['coords']
            else:
                raise ValueError(f"Unexpected coords shape: {b['coords'].shape}")
            
            # 添加batch维度
            coords.append(torch.cat([
                torch.full((coords_3d.shape[0], 1), i, dtype=torch.int32), 
                coords_3d
            ], dim=-1))
        
        coords = torch.cat(coords, dim=0)
        slat = torch.cat([b['slat'] for b in valid_batch])

        # slat 即直接读取的 x_0
        pack['slat'] = SparseTensor(coords=coords, feats=slat)
        # 兼容旧训练代码中使用的键
        pack['loss'] = pack['slat']
        pack['x_0'] = pack['slat']
        
        # 这些字段用于可视化，但不传递给模型
        pack['image'] = torch.stack([b['image'] for b in valid_batch])
        pack['gt_image'] = torch.stack([b['gt_image'] for b in valid_batch])
        pack['alpha'] = torch.stack([b['alpha'] for b in valid_batch])
        pack['extrinsics'] = torch.stack([b['extrinsics'] for b in valid_batch])
        pack['intrinsics'] = torch.stack([b['intrinsics'] for b in valid_batch])
        # instance ids
        if 'instance_id' in valid_batch[0]:
            pack['instance_id'] = [b['instance_id'] for b in valid_batch]
        # 调试：路径（按样本顺序的列表）
        if 'cond_path' in valid_batch[0]:
            pack['cond_path'] = [b.get('cond_path', None) for b in valid_batch]
        if 'gt_path' in valid_batch[0]:
            pack['gt_path'] = [b.get('gt_path', None) for b in valid_batch]
        if 'slat_path' in valid_batch[0]:
            pack['slat_path'] = [b.get('slat_path', None) for b in valid_batch]
        
        # 添加cond字段，指向image，以满足ImageConditionedMixin的要求
        pack['cond'] = pack['image']
        
        # 如果指定了split_size，进行batch分割
        if split_size is not None and split_size > 1:
            # 这里可以实现batch分割逻辑，但暂时先返回原始数据
            pass
        
        return pack
