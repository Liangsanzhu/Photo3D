import os
import numpy as np

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from torch.autograd import Variable
from math import exp

import roma
from kiui.op import safe_normalize
from einops import rearrange
import math

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_model, current_model, ema_updater):
    for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
        old_weight, up_weight = ema_params.data, current_params.data
        ema_params.data = ema_updater.update_average(old_weight, up_weight)

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool


def beta_linear_log_snr(t):
    return -torch.log(torch.special.expm1(1e-4 + 10 * (t ** 2)))

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def linear_to_srgb(linear):
    if isinstance(linear, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps)**(5 / 12) - 11) / 200
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError

def srgb_to_linear(srgb):
    if isinstance(srgb, torch.Tensor):
        """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        linear0 = 25 / 323 * srgb
        linear1 = torch.clamp(((200 * srgb + 11) / (211)), min=eps)**(12 / 5)
        return torch.where(srgb <= 0.04045, linear0, linear1)
    elif isinstance(srgb, np.ndarray):
        """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = np.finfo(np.float32).eps
        linear0 = 25 / 323 * srgb
        linear1 = np.maximum(((200 * srgb + 11) / (211)), eps)**(12 / 5)
        return np.where(srgb <= 0.04045, linear0, linear1)
    else:
        raise NotImplementedError

def activation_function():
    return nn.SiLU()

class LearnedSinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def pil2tensor(pil_img, device=torch.device('cpu')):
    tensor_chw = torch.Tensor(np.array(pil_img)).to(device).permute(2, 0, 1) / 255.0
    return tensor_chw.unsqueeze(0)

def save_tensor_image(tensor: torch.Tensor, save_path: str):
    if len(os.path.dirname(save_path)) > 0 and not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)  # [1, c, h, w]-->[c, h, w]
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)  
    tensor = tensor.permute(1, 2, 0).detach().cpu().numpy()  # [c, h, w]-->[h, w, c]
    Image.fromarray((tensor * 255).astype(np.uint8)).save(save_path)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def inpaint_atlas(image, append_mask=None):
    src_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    src_h = src_hls[:, :, 0]
    tgt_range, thres = 150, 1
    lowerb = tgt_range - thres
    upperb = tgt_range + thres
    mask = cv2.inRange(src=src_h, lowerb=lowerb, upperb=upperb)

    if append_mask is not None:
        mask = np.clip(mask + append_mask[..., 0], 0, 1).astype(np.uint8)
    image_inpaint = cv2.inpaint(src=image, inpaintMask=mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
    return image_inpaint
def get_rays(pose, h, w, fovy, opengl=True):

    x, y = torch.meshgrid(
        torch.arange(w, device=pose.device),
        torch.arange(h, device=pose.device),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx + 0.5) / focal,
                (y - cy + 0.5) / focal * (-1.0 if opengl else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl else 1.0),
    )  # [hw, 3]

    rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)  # [hw, 3]
    rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d) # [hw, 3]

    rays_o = rays_o.view(h, w, 3)
    rays_d = safe_normalize(rays_d).view(h, w, 3)

    return rays_o, rays_d

def orbit_camera_jitter(poses, strength=0.1):
    # poses: [B, 4, 4], assume orbit camera in opengl format
    # random orbital rotate

    B = poses.shape[0]
    rotvec_x = poses[:, :3, 1] * strength * np.pi * (torch.rand(B, 1, device=poses.device) * 2 - 1)
    rotvec_y = poses[:, :3, 0] * strength * np.pi / 2 * (torch.rand(B, 1, device=poses.device) * 2 - 1)

    rot = roma.rotvec_to_rotmat(rotvec_x) @ roma.rotvec_to_rotmat(rotvec_y)
    R = rot @ poses[:, :3, :3]
    T = rot @ poses[:, :3, 3:]

    new_poses = poses.clone()
    new_poses[:, :3, :3] = R
    new_poses[:, :3, 3:] = T
    
    return new_poses

def grid_distortion(images, strength=0.5):
    # images: [B, C, H, W]
    # num_steps: int, grid resolution for distortion
    # strength: float in [0, 1], strength of distortion

    B, C, H, W = images.shape

    num_steps = np.random.randint(8, 17)
    grid_steps = torch.linspace(-1, 1, num_steps)

    # have to loop batch...
    grids = []
    for b in range(B):
        # construct displacement
        x_steps = torch.linspace(0, 1, num_steps) # [num_steps], inclusive
        x_steps = (x_steps + strength * (torch.rand_like(x_steps) - 0.5) / (num_steps - 1)).clamp(0, 1) # perturb
        x_steps = (x_steps * W).long() # [num_steps]
        x_steps[0] = 0
        x_steps[-1] = W
        xs = []
        for i in range(num_steps - 1):
            xs.append(torch.linspace(grid_steps[i], grid_steps[i + 1], x_steps[i + 1] - x_steps[i]))
        xs = torch.cat(xs, dim=0) # [W]

        y_steps = torch.linspace(0, 1, num_steps) # [num_steps], inclusive
        y_steps = (y_steps + strength * (torch.rand_like(y_steps) - 0.5) / (num_steps - 1)).clamp(0, 1) # perturb
        y_steps = (y_steps * H).long() # [num_steps]
        y_steps[0] = 0
        y_steps[-1] = H
        ys = []
        for i in range(num_steps - 1):
            ys.append(torch.linspace(grid_steps[i], grid_steps[i + 1], y_steps[i + 1] - y_steps[i]))
        ys = torch.cat(ys, dim=0) # [H]

        # construct grid
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy') # [H, W]
        grid = torch.stack([grid_x, grid_y], dim=-1) # [H, W, 2]

        grids.append(grid)
    
    grids = torch.stack(grids, dim=0).to(images.device) # [B, H, W, 2]

    # grid sample
    images = F.grid_sample(images, grids, align_corners=False)

    return images

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def depth2batch(data, indices):
    seq = data[indices]
    return seq

def get_batch_id(octree, depth_list):
    batch_id = []
    for d in depth_list:
        batch_id.append(octree.batch_id(d))
    batch_id = torch.cat(batch_id, dim=0)
    return batch_id


def get_depth2batch_indices(octree, depth_list, buffer_size=None, mask=None):
  # Rearange data from depth-by-depth to batch-by-batch
    batch_id = get_batch_id(octree, depth_list)
    if buffer_size is not None:
        batch_buffer = torch.arange(
            octree.batch_size, device=octree.device).reshape(-1, 1)
        batch_buffer = batch_buffer.repeat(1, buffer_size).reshape(-1)
        batch_id = torch.cat([batch_buffer, batch_id])
    if mask is not None:
        batch_id = batch_id[~mask]
    batch_id_sorted, indices = torch.sort(batch_id)
    return batch_id_sorted, indices