from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
from ...utils.random_utils import hammersley_sequence
from .decoder_gs import SLatGaussianDecoder
from ...representations import Gaussian
from ..sparse_elastic_mixin import SparseTransformerElasticMixin
from ...modules.sparse.attention.modules import SparseMultiHeadAttention


class DinoCrossAttention(nn.Module):
    """Cross attention module for DINO features conditioning"""
    def __init__(self, model_channels: int, num_heads: int, num_head_channels: int = 64):
        super().__init__()
        self.model_channels = model_channels
        self.num_heads = num_heads
        self.head_dim = num_head_channels
        
        # DINOv3 features are 1024-dim, need to project to model_channels
        self.dino_proj = nn.Linear(1024, model_channels)
        
        # Cross attention
        self.cross_attn = SparseMultiHeadAttention(
            model_channels, num_heads, model_channels, type="cross"
        )
        
        # Gating
        self.gate = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: sp.SparseTensor, dino_feats: torch.Tensor) -> sp.SparseTensor:
        """
        x: SparseTensor [N, C] - decoder features
        dino_feats: [C, Hp, Wp] - DINO features from input image (C=1024 for DINOv3)
        """
        # Ensure dino_feats has the same dtype and device as x.feats
        dino_feats = dino_feats.to(dtype=x.feats.dtype, device=x.feats.device)
        
        # Flatten DINO features: [C, Hp, Wp] -> [Hp*Wp, C]
        dino_flat = dino_feats.permute(1, 2, 0).reshape(-1, dino_feats.shape[0])  # [Hp*Wp, 1024]
        
        # Ensure dino_proj has the same dtype and device as x.feats
        self.dino_proj = self.dino_proj.to(dtype=x.feats.dtype, device=x.feats.device)
        
        # Project DINO features to model_channels
        dino_proj = self.dino_proj(dino_flat)  # [Hp*Wp, model_channels]
        
        # Ensure cross_attn modules have the same dtype and device as x.feats
        self.cross_attn = self.cross_attn.to(dtype=x.feats.dtype, device=x.feats.device)
        
        # Create SparseTensor for DINO features with dummy coordinates
        # We need to create coordinates for each DINO patch
        device = x.feats.device
        batch_size = dino_proj.shape[0]  # Hp*Wp
        
        # Create dummy coordinates for DINO patches (batch_idx, x, y, z)
        # Use a simple grid layout
        grid_size = int(batch_size ** 0.5)  # Assume square grid
        coords = []
        for i in range(batch_size):
            batch_idx = 0  # All DINO patches belong to the same batch
            x_coord = i // grid_size
            y_coord = i % grid_size
            z_coord = 0
            coords.append([batch_idx, x_coord, y_coord, z_coord])
        
        coords = torch.tensor(coords, device=device, dtype=torch.int32)
        dino_sparse = sp.SparseTensor(dino_proj, coords)
        
        # Cross attention with DINO features as context
        attn_out = self.cross_attn(x, context=dino_sparse)
        
        # Gated residual connection
        # Ensure consistent dtype
        gate_output = self.gate * attn_out.feats
        gate_output = gate_output.to(dtype=x.feats.dtype, device=x.feats.device)
        return x.replace(x.feats + gate_output)


class SLatGaussianDecoderDinoCond(SLatGaussianDecoder):
    """
    SLat Gaussian Decoder with DINO cross-attention conditioning.
    Adds cross-attention to input image DINO features at the end of each stage.
    """
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
        enable_dino_cond: bool = True,
    ):
        super().__init__(
            resolution=resolution,
            model_channels=model_channels,
            latent_channels=latent_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
            representation_config=representation_config,
        )
        
        self.enable_dino_cond = enable_dino_cond
        self.dino_feats_cache = None
        
        if enable_dino_cond:
            # Add cross-attention modules at the end of each stage
            self.dino_cross_attns = nn.ModuleList()
            for i in range(num_blocks):
                cross_attn = DinoCrossAttention(
                    model_channels=model_channels,
                    num_heads=num_heads or (model_channels // num_head_channels),
                    num_head_channels=num_head_channels
                )
                self.dino_cross_attns.append(cross_attn)
    
    def set_dino_features(self, dino_feats: torch.Tensor):
        """Set DINO features for conditioning (cached)"""
        self.dino_feats_cache = dino_feats.detach()
    
    def forward(self, x: sp.SparseTensor, dino_feats: Optional[torch.Tensor] = None) -> List[Gaussian]:
        """
        Forward pass with optional DINO conditioning.
        
        Args:
            x: Input sparse tensor
            dino_feats: DINO features from input image [C, Hp, Wp]
        """
        if dino_feats is not None:
            self.set_dino_features(dino_feats)
        
        # Follow base.py forward flow exactly
        h = self.input_layer(x)
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(x.coords[:, 1:])
        h = h.type(self.dtype)
        
        # Forward through blocks with DINO cross-attention
        for i, block in enumerate(self.blocks):
            h = block(h)
            
            # Add DINO cross-attention at the end of each block
            if self.enable_dino_cond and self.dino_feats_cache is not None:
                # Ensure DINO features cache has the same dtype as decoder features
                dino_feats_converted = self.dino_feats_cache.to(dtype=h.feats.dtype, device=h.feats.device)
                h = self.dino_cross_attns[i](h, dino_feats_converted)
        
        # Final processing (from parent class)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return self.to_representation(h)


class ElasticSLatGaussianDecoderDinoCond(SparseTransformerElasticMixin, SLatGaussianDecoderDinoCond):
    """Elastic version with memory management"""
    pass


