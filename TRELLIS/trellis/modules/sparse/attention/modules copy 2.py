from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import SparseTensor
from .full_attn import sparse_scaled_dot_product_attention
from .serialized_attn import SerializeMode, sparse_serialized_scaled_dot_product_self_attention
from .windowed_attn import sparse_windowed_scaled_dot_product_self_attention
from ...attention import RotaryPositionEmbedder


class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: Union[SparseTensor, torch.Tensor]) -> Union[SparseTensor, torch.Tensor]:
        x_type = x.dtype
        x = x.float()
        if isinstance(x, SparseTensor):
            x = x.replace(F.normalize(x.feats, dim=-1))
        else:
            x = F.normalize(x, dim=-1)            
        return (x * self.gamma * self.scale).to(x_type)


class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "serialized", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        cache_max_tokens: Optional[int] = 8192,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "serialized", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        assert type == "self" or use_rope is False, "Rotary position embeddings only supported for self-attention"
        self.channels = channels
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_sequence = shift_sequence
        self.shift_window = shift_window
        self.serialize_mode = serialize_mode
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm
        self.cache_max_tokens = cache_max_tokens

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        
        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)
        
        # KV缓存机制
        self.cached_k = None
        self.cached_v = None
        self.cached_coords = None

    def clear_cache(self):
        """清除KV缓存"""
        self.cached_k = None
        self.cached_v = None
        self.cached_coords = None

    def _cap_kv_length(self, k: Union[SparseTensor, torch.Tensor], v: Union[SparseTensor, torch.Tensor]) -> Tuple[Union[SparseTensor, torch.Tensor], Union[SparseTensor, torch.Tensor]]:
        """将提供的K/V在dim=0上裁剪到cache_max_tokens长度（若设置）。
        假设dim=0为时间/序列维度，且前部为最新token。"""
        if self.cache_max_tokens is None:
            return k, v
        max_len = self.cache_max_tokens
        if isinstance(k, SparseTensor) and isinstance(v, SparseTensor):
            cur_len = k.feats.shape[0]
            if cur_len <= max_len:
                return k, v
            k = SparseTensor(feats=k.feats[:max_len], coords=k.coords[:max_len])
            v = SparseTensor(feats=v.feats[:max_len], coords=v.coords[:max_len])
            return k, v
        else:
            cur_len = k.shape[0]
            if cur_len <= max_len:
                return k, v
            return k[:max_len], v[:max_len]

    def _truncate_cached(self) -> None:
        """对内部缓存的KV进行长度裁剪。"""
        if self.cache_max_tokens is None or self.cached_k is None or self.cached_v is None:
            return
        if isinstance(self.cached_k, SparseTensor) and isinstance(self.cached_v, SparseTensor):
            cur_len = self.cached_k.feats.shape[0]
            if cur_len > self.cache_max_tokens:
                self.cached_k = SparseTensor(
                    feats=self.cached_k.feats[:self.cache_max_tokens],
                    coords=self.cached_k.coords[:self.cache_max_tokens]
                )
                self.cached_v = SparseTensor(
                    feats=self.cached_v.feats[:self.cache_max_tokens],
                    coords=self.cached_v.coords[:self.cache_max_tokens]
                )
                if self.cached_coords is not None:
                    self.cached_coords = self.cached_coords[:self.cache_max_tokens]
        else:
            cur_len = self.cached_k.shape[0]
            if cur_len > self.cache_max_tokens:
                self.cached_k = self.cached_k[:self.cache_max_tokens]
                self.cached_v = self.cached_v[:self.cache_max_tokens]

    @staticmethod
    def _linear(module: nn.Linear, x: Union[SparseTensor, torch.Tensor]) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            return x.replace(module(x.feats))
        else:
            return module(x)

    @staticmethod
    def _reshape_chs(x: Union[SparseTensor, torch.Tensor], shape: Tuple[int, ...]) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            return x.reshape(*shape)
        else:
            return x.reshape(*x.shape[:2], *shape)

    def _fused_pre(self, x: Union[SparseTensor, torch.Tensor], num_fused: int) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            x_feats = x.feats.unsqueeze(0)
        else:
            x_feats = x
        x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
        return x.replace(x_feats.squeeze(0)) if isinstance(x, SparseTensor) else x_feats

    def _rope(self, qkv: SparseTensor) -> SparseTensor:
        q, k, v = qkv.feats.unbind(dim=1)   # [T, H, C]
        q, k = self.rope(q, k, qkv.coords[:, 1:])
        qkv = qkv.replace(torch.stack([q, k, v], dim=1)) 
        return qkv
    
    def forward(self, x: Union[SparseTensor, torch.Tensor], context: Optional[Union[SparseTensor, torch.Tensor]] = None,
                precomputed_k: Optional[Union[SparseTensor, torch.Tensor]] = None, 
                precomputed_v: Optional[Union[SparseTensor, torch.Tensor]] = None,
                ref_gamma: Union[float, torch.Tensor] = 0.0,
                adaptive_weights: Optional[torch.Tensor] = None) -> Union[SparseTensor, torch.Tensor]:
        if self._type == "self":
            qkv = self._linear(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)
            if self.use_rope:
                qkv = self._rope(qkv)
            if self.qk_rms_norm:
                q, k, v = qkv.unbind(dim=1)
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
                qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
            
            # KV融合逻辑：只在full attention模式下使用融合机制
            q, k_current, v_current = qkv.unbind(dim=1)
            
            # 只在full attention模式下进行KV融合
            if self.attn_mode == "full":
                # 检查是否有缓存的KV
                if self.cached_k is not None and self.cached_v is not None:
                    # 第二次调用：将当前KV与缓存的KV在dim=0上拼接（缓存的放在后面）
                    
                    if isinstance(k_current, SparseTensor) and isinstance(v_current, SparseTensor):
                        # 对于SparseTensor，需要拼接features和coordinates
                        k_fused = SparseTensor(
                            feats=torch.cat([k_current.feats, self.cached_k.feats], dim=0),
                            coords=torch.cat([k_current.coords, self.cached_coords], dim=0)
                        )
                        v_fused = SparseTensor(
                            feats=torch.cat([v_current.feats, self.cached_v.feats], dim=0),
                            coords=torch.cat([v_current.coords, self.cached_coords], dim=0)
                        )
                    else:
                        # 对于普通Tensor，直接拼接
                        k_fused = torch.cat([k_current, self.cached_k], dim=0)
                        v_fused = torch.cat([v_current, self.cached_v], dim=0)
                    """
                    # 改为将历史KV按ref_gamma缩放后与当前KV相加（保持长度不变）
                    if isinstance(k_current, SparseTensor) and isinstance(v_current, SparseTensor) \
                       and isinstance(self.cached_k, SparseTensor) and isinstance(self.cached_v, SparseTensor):
                        # 仅当坐标完全一致时才进行逐元素相加；否则回退为当前KV
                        if (self.cached_coords is not None and
                            self.cached_k.coords.shape == k_current.coords.shape and
                            torch.equal(self.cached_k.coords, k_current.coords)):
                            feats_like = k_current.feats  # [T, H, C]
                            # 计算按head广播的ref_gamma: [H] -> [1,H,1]
                            if isinstance(ref_gamma, torch.Tensor):
                                rg = ref_gamma.to(device=feats_like.device, dtype=feats_like.dtype)
                                if rg.dim() == 0:
                                    rg = rg.view(1, 1, 1)
                                elif rg.dim() == 1 and rg.shape[0] == feats_like.shape[1]:
                                    rg = rg.view(1, -1, 1)
                                else:
                                    rg = None
                            else:
                                rg = torch.tensor(ref_gamma, device=feats_like.device, dtype=feats_like.dtype).view(1, 1, 1)

                            # 仅当通道数一致时相加，避免C不一致报错
                            if (
                                rg is not None and
                                self.cached_k.feats.shape[1] == feats_like.shape[1] and
                                self.cached_k.feats.shape[2] == feats_like.shape[2]
                            ):
                                k_fused = SparseTensor(
                                    feats=feats_like + (rg * self.cached_k.feats),
                                    coords=k_current.coords
                                )
                                v_fused = SparseTensor(
                                    feats=v_current.feats + (rg * self.cached_v.feats),
                                    coords=v_current.coords
                                )
                                print(k_fused.feats.shape, v_fused.feats.shape,rg.shape)
                            else:
                                k_fused = k_current
                                v_fused = v_current
                        else:
                            k_fused = k_current
                            v_fused = v_current
                    else:
                        # 稠密Tensor：形状需完全一致，否则回退为当前KV
                        if isinstance(k_current, torch.Tensor) and isinstance(v_current, torch.Tensor) and \
                           isinstance(self.cached_k, torch.Tensor) and isinstance(self.cached_v, torch.Tensor):
                            if self.cached_k.shape == k_current.shape and self.cached_v.shape == v_current.shape:
                                feats_like = k_current  # [T, H, C]
                                if isinstance(ref_gamma, torch.Tensor):
                                    rg = ref_gamma.to(device=feats_like.device, dtype=feats_like.dtype)
                                    if rg.dim() == 0:
                                        rg = rg.view(1, 1, 1)
                                    elif rg.dim() == 1 and rg.shape[0] == feats_like.shape[1]:
                                        rg = rg.view(1, -1, 1)
                                    else:
                                        rg = None
                                else:
                                    rg = torch.tensor(ref_gamma, device=feats_like.device, dtype=feats_like.dtype).view(1, 1, 1)

                                if rg is not None:
                                    k_fused = k_current + (rg * self.cached_k)
                                    v_fused = v_current + (rg * self.cached_v)
                                else:
                                    k_fused = k_current
                                    v_fused = v_current
                            else:
                                k_fused = k_current
                                v_fused = v_current
                        else:
                            k_fused = k_current
                            v_fused = v_current
                    """
                    # 使用融合后的KV进行full attention计算
                    h = sparse_scaled_dot_product_attention(q=q, k=k_fused, v=v_fused)
                    
                    # 更新缓存：将当前KV添加到缓存中（当前KV放在前面，缓存的放在后面，断开计算图引用）
                    if isinstance(k_current, SparseTensor) and isinstance(v_current, SparseTensor):
                        self.cached_k = SparseTensor(
                            feats=torch.cat([k_current.feats.detach(), self.cached_k.feats], dim=0),
                            coords=torch.cat([k_current.coords.detach(), self.cached_coords], dim=0)
                        )
                        self.cached_v = SparseTensor(
                            feats=torch.cat([v_current.feats.detach(), self.cached_v.feats], dim=0),
                            coords=torch.cat([v_current.coords.detach(), self.cached_coords], dim=0)
                        )
                        self.cached_coords = torch.cat([k_current.coords.detach(), self.cached_coords], dim=0)
                    else:
                        self.cached_k = torch.cat([k_current.detach(), self.cached_k], dim=0)
                        self.cached_v = torch.cat([v_current.detach(), self.cached_v], dim=0)
                    
                    # 裁剪内部缓存长度
                    self._truncate_cached()
                else:
                    # 第一次调用：缓存当前的KV（断开计算图引用）
                    if isinstance(k_current, SparseTensor):
                        self.cached_k = SparseTensor(
                            feats=k_current.feats.detach(),
                            coords=k_current.coords.detach()
                        )
                        self.cached_v = SparseTensor(
                            feats=v_current.feats.detach(),
                            coords=v_current.coords.detach()
                        )
                        self.cached_coords = k_current.coords.detach()
                    else:
                        self.cached_k = k_current.detach()
                        self.cached_v = v_current.detach()
                        self.cached_coords = None
                    
                    # 使用当前KV进行full attention计算
                    h = sparse_scaled_dot_product_attention(qkv)
            else:
                # 非full attention模式：保持原有逻辑，不使用KV融合
                if self.attn_mode == "serialized":
                    h = sparse_serialized_scaled_dot_product_self_attention(
                        qkv, self.window_size, serialize_mode=self.serialize_mode, 
                        shift_sequence=self.shift_sequence, shift_window=self.shift_window
                    )
                elif self.attn_mode == "windowed":
                    h = sparse_windowed_scaled_dot_product_self_attention(
                        qkv, self.window_size, shift_window=self.shift_window
                    )
            
            # 保存当前KV用于外部读取
            self.last_k = k_current
            self.last_v = v_current
        else:
            # Cross-attention：不需要KV缓存机制
            q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            kv = self._linear(self.to_kv, context)
            kv = self._fused_pre(kv, num_fused=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=1)
                k = self.k_rms_norm(k)
                kv = kv.replace(torch.stack([k.feats, v.feats], dim=1))
            
            h = sparse_scaled_dot_product_attention(q, kv)
        h = self._reshape_chs(h, (-1,))
        h = self._linear(self.to_out, h)
        return h
