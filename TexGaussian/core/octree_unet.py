import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Tuple, Literal
from torch.utils.checkpoint import checkpoint

import ocnn

def ckpt_conv_wrapper(conv_op, x, *args):
    def conv_wrapper(x, dummy_tensor, *args):
        return conv_op(x, *args)

    # The dummy tensor is a workaround when the checkpoint is used for the first conv layer:
    # https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
    dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)

    return checkpoint(
        conv_wrapper, x, dummy, *args, use_reentrant=True)

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float,
        context_dim: int,
        batch_first: bool = True,
    ):
        super().__init__()
        self.cross_norm = torch.nn.LayerNorm(dim)
        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=attn_drop, 
            kdim = context_dim, vdim = context_dim, 
            batch_first=batch_first
        )
    
    def forward(self, data, octree, depth, context = None):
        cross_attn_data = self.cross_norm(data)

        batch_id = octree.batch_id(depth = depth, nempty = True)
        batch_size = octree.batch_size

        cross_attn_results = []

        for i in range(batch_size):
            cross_attn_i = cross_attn_data[batch_id == i]
            cross_attn_i, _ = self.cross_attn(
                query = cross_attn_i.unsqueeze(0), key = context[i:i+1], value = context[i:i+1])
            
            cross_attn_results.append(cross_attn_i.squeeze(0))
        
        cross_attn_results = torch.cat(cross_attn_results, dim = 0)

        data = data + cross_attn_results

        return data
    

class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resample: Literal['default', 'up', 'down'] = 'default',
        groups: int = 32,
        eps: float = 1e-5,
        skip_scale: float = 1, # multiplied to output
        use_checkpoint: bool = True, 
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_scale = skip_scale

        self.use_checkpoint = use_checkpoint

        self.norm1 = ocnn.nn.OctreeGroupNorm(in_channels = in_channels, group = groups, nempty = True)
        self.conv1 = ocnn.nn.OctreeConv(in_channels, out_channels, kernel_size=[3], nempty=True, use_bias=True)

        self.norm2 = ocnn.nn.OctreeGroupNorm(in_channels=out_channels, group=groups, nempty=True)
        self.conv2 = ocnn.nn.OctreeConv(out_channels, out_channels, kernel_size=[3], nempty=True, use_bias=True)

        self.act = F.silu
        
        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = ocnn.modules.Conv1x1(in_channels, out_channels, use_bias = True)


    def forward(self, x, octree, depth):
        res = x

        x = self.norm1(x, octree, depth)

        x = self.act(x)
        
        if self.use_checkpoint:
            x = ckpt_conv_wrapper(self.conv1, x, octree, depth)
        else:
            x = self.conv1(x, octree, depth)

        x = self.norm2(x, octree, depth)
        x = self.act(x)

        if self.use_checkpoint:
            x = ckpt_conv_wrapper(self.conv2, x, octree, depth)
        else:
            x = self.conv2(x, octree, depth)

        x = (x + self.shortcut(res)) * self.skip_scale

        return x


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        downsample: bool = True,
        attention: bool = True,
        num_heads: int = 16,
        context_dim: int = 768,
        skip_scale: float = 1,
        use_checkpoint: bool = True, 
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
 
        nets = []
        attns = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            nets.append(ResnetBlock(in_channels, out_channels, skip_scale=skip_scale, use_checkpoint=use_checkpoint))
            if attention:
                attns.append(CrossAttention(dim = out_channels, num_heads = num_heads, attn_drop = 0.0, context_dim = context_dim, batch_first = True))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.downsample = None
        if downsample:
            self.downsample = ocnn.nn.OctreeConv(out_channels, out_channels, kernel_size=[3], stride=2, nempty = True, use_bias=True)

    def forward(self, x, octree, depth, condition = None):
        xs = []

        for attn, net in zip(self.attns, self.nets):
            x = net(x, octree, depth)
            if attn:
                if self.use_checkpoint:
                    x = checkpoint(attn, x, octree, depth, condition)
                else:
                    x = attn(x, octree, depth, condition)
            xs.append(x)

        if self.downsample:
            if self.use_checkpoint:
                x = ckpt_conv_wrapper(self.downsample, x, octree, depth)
            else:
                x = self.downsample(x, octree, depth)
            xs.append(x)
  
        return x, xs


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        attention: bool = True,
        num_heads: int = 16,
        context_dim: int = 768,
        skip_scale: float = 1,
        use_checkpoint: bool = True, 
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        nets = []
        attns = []
        # first layer
        nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale, use_checkpoint=use_checkpoint))
        # more layers
        for i in range(num_layers):
            nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale, use_checkpoint=use_checkpoint))
            if attention:
                attns.append(CrossAttention(dim = in_channels, num_heads = num_heads, attn_drop = 0.0, context_dim = context_dim, batch_first = True))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        
    def forward(self, x, octree, depth, condition = None):
        x = self.nets[0](x, octree, depth)
        for attn, net in zip(self.attns, self.nets[1:]):
            if attn:
                if self.use_checkpoint:
                    x = checkpoint(attn, x, octree, depth, condition)
                else:
                    x = attn(x, octree, depth, condition)
            x = net(x, octree, depth)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_out_channels: int,
        out_channels: int,
        num_layers: int = 1,
        upsample: bool = True,
        attention: bool = True,
        num_heads: int = 16,
        context_dim: int = 768, 
        skip_scale: float = 1,
        use_checkpoint: bool = True, 
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        nets = []
        attns = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            cskip = prev_out_channels if (i == num_layers - 1) else out_channels
            nets.append(ResnetBlock(cin + cskip, out_channels, skip_scale=skip_scale, use_checkpoint=use_checkpoint))
            if attention:
                attns.append(CrossAttention(dim = out_channels, num_heads = num_heads, attn_drop = 0.0, context_dim = context_dim, batch_first = True))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.upsample = None
        if upsample:
            self.up_pool = ocnn.nn.OctreeUpsample(method="nearest", nempty=True)
            self.upsample = ocnn.nn.OctreeConv(out_channels, out_channels, kernel_size=[3], nempty=True, use_bias=True)

    def forward(self, x, xs, octree, depth, condition = None):

        for attn, net in zip(self.attns, self.nets):
            res_x = xs[-1]
            xs = xs[:-1]
            x = torch.cat([x, res_x], dim=1)
            x = net(x, octree, depth)
            if attn:
                if self.use_checkpoint:
                    x = checkpoint(attn, x, octree, depth, condition)
                else:
                    x = attn(x, octree, depth, condition)
            
        if self.upsample:
            if self.use_checkpoint:
                x = ckpt_conv_wrapper(self.up_pool, x, octree, depth)
                x = ckpt_conv_wrapper(self.upsample, x, octree, depth + 1)
            else:
                x = self.up_pool(x, octree, depth)
                x = self.upsample(x, octree, depth + 1)

        return x


class OctreeUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 32, 
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8, 8),
        down_attention: Tuple[bool, ...] = (False, False, False, False, False),
        mid_attention: bool = False,
        up_attention: Tuple[bool, ...] = (False, False, False, False, False),
        num_heads: int = 16,
        context_dim: int = 768,
        layers_per_block: int = 2,
        skip_scale: float = np.sqrt(0.5),
        use_checkpoint: bool = True,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        down_channels = [model_channels * m for m in channel_mult]
        up_channels = [model_channels * m for m in channel_mult[::-1]]

        # first
        self.conv_in = ocnn.nn.OctreeConv(in_channels, down_channels[0], kernel_size=[3], nempty=True, use_bias=True)

        # down
        down_blocks = []
        cout = down_channels[0]
        self.downsample = []

        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]

            downsample = (i != len(down_channels) - 1)

            self.downsample.append(downsample)

            down_blocks.append(DownBlock(
                cin, cout, 
                num_layers=layers_per_block, 
                downsample = downsample, # not final layer
                attention=down_attention[i],
                num_heads = num_heads,
                context_dim = context_dim,
                skip_scale=skip_scale,
                use_checkpoint=use_checkpoint,
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        # mid
        self.mid_block = MidBlock(
            in_channels = down_channels[-1],
            num_layers = layers_per_block,
            attention=mid_attention,
            num_heads = num_heads,
            context_dim = context_dim,
            skip_scale=skip_scale, 
            use_checkpoint=use_checkpoint,
        )

        # up
        up_blocks = []
        cout = up_channels[0]

        self.upsample = []

        for i in range(len(up_channels)):
            cin = cout
            cout = up_channels[i]
            cskip = down_channels[max(-2 - i, -len(down_channels))] # for assymetric

            upsample = (i != len(up_channels) - 1)
            self.upsample.append(upsample)

            up_blocks.append(UpBlock(
                cin, cskip, cout, 
                num_layers=layers_per_block + 1, # one more layer for up
                upsample=upsample, # not final layer
                attention=up_attention[i],
                skip_scale=skip_scale,
                use_checkpoint = use_checkpoint,
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

        # last
        self.norm_out = ocnn.nn.OctreeGroupNorm(in_channels = up_channels[-1], group = 32, nempty = True)
        self.conv_out = ocnn.nn.OctreeConv(up_channels[-1], out_channels, kernel_size=[3], nempty=True, use_bias=True)
        self.conv = ocnn.nn.OctreeConv(out_channels, out_channels, kernel_size=[3], nempty=True, use_bias=True)

    def forward(self, x, octree, condition = None):
        # x: [B, Cin, H, W]

        input_data = x

        depth = octree.depth

        # first
        if self.use_checkpoint:
            x = ckpt_conv_wrapper(self.conv_in, x, octree, depth)
        else:
            x = self.conv_in(x, octree, depth)
        
        # down
        xss = [x]
        for block, downsample in zip(self.down_blocks, self.downsample):
            x, xs = block(x, octree, depth, condition)
            xss.extend(xs)

            if downsample:
                depth -= 1
        
        # mid
        x = self.mid_block(x, octree, depth, condition)

        # up
        for block, upsample in zip(self.up_blocks, self.upsample):
            xs = xss[-len(block.nets):]
            xss = xss[:-len(block.nets)]
            x = block(x, xs, octree, depth, condition)

            if upsample:
                depth += 1

        # last
        x = self.norm_out(x, octree, depth)
        x = F.silu(x)
        if self.use_checkpoint:
            x = ckpt_conv_wrapper(self.conv_out, x, octree, depth)
        else:
            x = self.conv_out(x, octree, depth) # [B, Cout, H', W']
        
        if self.use_checkpoint:
            x = ckpt_conv_wrapper(self.conv, x, octree, depth)
        else:
            x = self.conv(x, octree, depth)

        assert depth == octree.depth
        assert x.shape[0] == input_data.shape[0]

        return x
