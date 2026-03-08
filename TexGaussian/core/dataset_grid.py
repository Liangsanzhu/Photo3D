import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Tuple

import trimesh

from external.clip import tokenize
from core.options import Options


def read_prompt_line(text_file: str, idx: int) -> str:
    with open(text_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == idx:
                return line.strip()
    raise IndexError(f"Index {idx} out of range for {text_file}")


def split_grid(image: np.ndarray) -> List[np.ndarray]:
    """Split a 2x2 grid image (RGB or RGBA) into 4 tiles in row-major order.

    Returns list of 4 images [H/2, W/2, C].
    """
    H, W, C = image.shape
    h2, w2 = H // 2, W // 2
    tiles = [
        image[0:h2, 0:w2],
        image[0:h2, w2:W],
        image[h2:H, 0:w2],
        image[h2:H, w2:W],
    ]
    return tiles


def get_4view_cameras(radius: float, fovy: float, yaw_offset_deg: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct 4 canonical camera poses (yaws: 0, 90, 180, 270; pitch=0) with intrinsics from fovy.

    Returns cam_view [V, 4, 4], cam_view_proj [V, 4, 4], cam_pos [V, 3]
    """
    yaws = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
    if abs(yaw_offset_deg) > 1e-6:
        yaw_offset = np.deg2rad(yaw_offset_deg)
        yaws = [y + yaw_offset for y in yaws]
    pitch = 0.0

    def yaw_pitch_to_c2w(yaw: float, pitch: float, r: float) -> np.ndarray:
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        # camera looking at origin, up +Y
        forward = np.array([cy * cp, sp, sy * cp], dtype=np.float32)
        pos = -forward * r
        up = np.array([0, 1, 0], dtype=np.float32)
        z = (pos - np.array([0, 0, 0], dtype=np.float32))
        z = z / (np.linalg.norm(z) + 1e-8)
        x = np.cross(up, z)
        x = x / (np.linalg.norm(x) + 1e-8)
        y = np.cross(z, x)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = x
        c2w[:3, 1] = y
        c2w[:3, 2] = z
        c2w[:3, 3] = pos
        return c2w

    c2ws = [yaw_pitch_to_c2w(y, pitch, radius) for y in yaws]
    cam_poses = torch.from_numpy(np.stack(c2ws, axis=0))

    tan_half_fov = np.tan(0.5 * np.deg2rad(fovy))
    proj = torch.zeros(4, 4, dtype=torch.float32)
    proj[0, 0] = 1 / tan_half_fov
    proj[1, 1] = 1 / tan_half_fov
    proj[2, 2] = (2.5 + 0.5) / (2.5 - 0.5)
    proj[3, 2] = - (2.5 * 0.5) / (2.5 - 0.5)
    proj[2, 3] = 1

    cam_poses[:, :3, 1:3] *= -1
    cam_view = torch.inverse(cam_poses).transpose(1, 2)
    cam_view_proj = cam_view @ proj
    cam_pos = - cam_poses[:, :3, 3]
    return cam_view, cam_view_proj, cam_pos


class GridDataset(Dataset):

    @staticmethod
    def scan_ids(opt: Options) -> List[int]:
        """Scan grid_image_dir for grid_{id}.png and keep ids whose mesh/text exist."""
        ids = []
        if not os.path.isdir(opt.grid_image_dir):
            return ids
        for name in os.listdir(opt.grid_image_dir):
            if not name.startswith('grid_') or not name.endswith('.png'):
                continue
            try:
                sid = int(name[5:-4])
            except Exception:
                continue
            grid_path = os.path.join(opt.grid_image_dir, name)
            mesh_path = os.path.join(opt.mesh_root, str(sid), 'mesh.obj')
            if not os.path.exists(grid_path) or not os.path.exists(mesh_path):
                continue
            if opt.text_file and os.path.exists(opt.text_file):
                try:
                    _ = read_prompt_line(opt.text_file, sid)
                except Exception:
                    continue
            ids.append(sid)
        ids.sort()
        return ids

    def __init__(self, opt: Options, id_list: List[int], training: bool = True):
        self.opt = opt
        self.ids = list(id_list) if id_list is not None else GridDataset.scan_ids(opt)
        self.training = training

    @staticmethod
    def _safe_tokenize_text(text: str):
        """Safely tokenize text for CLIP by truncating if context length exceeds limit.

        Strategy: try original → progressively shorten by ratio → fallback to first N words.
        """
        text = (text or "").strip()
        if not text:
            return tokenize("").squeeze()

        # hard cap chars to avoid pathological long inputs
        if len(text) > 512:
            text = text[:512]

        candidates = [text]
        for frac in [0.9, 0.8, 0.7, 0.6, 0.5]:
            cut = text[: max(1, int(len(text) * frac))]
            candidates.append(cut)
        # fallback by words
        words = text.split()
        if len(words) > 50:
            candidates.append(" ".join(words[:50]))
        if len(words) > 32:
            candidates.append(" ".join(words[:32]))

        last_err = None
        for cand in candidates:
            try:
                return tokenize(cand).squeeze()
            except RuntimeError as e:
                msg = str(e)
                if 'context length' in msg:
                    last_err = e
                    continue
                raise
        # if still failing, re-raise the last context error
        if last_err is not None:
            # fallback to empty to avoid crashing the dataloader
            return tokenize("").squeeze()
        return tokenize(text).squeeze()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = int(self.ids[idx])

        grid_path = os.path.join(self.opt.grid_image_dir, f'grid_{uid}.png')
        img_rgba = cv2.imread(grid_path, cv2.IMREAD_UNCHANGED)
        if img_rgba is None:
            raise FileNotFoundError(grid_path)
        img_rgba = img_rgba.astype(np.float32) / 255.0

        tiles = split_grid(img_rgba)  # 4 tiles
        images = []
        masks = []
        for tile in tiles:
            t = torch.from_numpy(tile).permute(2, 0, 1)  # [C,H,W]
            C = t.shape[0]
            if C == 4:
                mask = t[3:4]
                rgb = t[:3] * mask + (1 - mask)  # white bg
                rgb = rgb[[2, 1, 0]].contiguous()  # bgr->rgb
            elif C == 3:
                mask = torch.ones(1, t.shape[1], t.shape[2], dtype=t.dtype)
                rgb = t[[2, 1, 0]].contiguous()  # bgr->rgb
            else:
                # grayscale fallback
                mask = torch.ones(1, t.shape[1], t.shape[2], dtype=t.dtype)
                rgb = t[:1].expand(3, t.shape[1], t.shape[2])
            images.append(rgb)
            masks.append(mask.squeeze(0))
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)

        # cameras for 4 views (allow yaw offset and radius override)
        cam_radius = self.opt.grid_cam_radius_override if getattr(self.opt, 'grid_cam_radius_override', 0.0) > 0 else self.opt.grid_cam_radius
        cam_view, cam_view_proj, cam_pos = get_4view_cameras(cam_radius, self.opt.grid_fovy, getattr(self.opt, 'grid_yaw_offset_deg', 0.0))

        # resize to output_size
        images_out = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)
        masks_out = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)

        results = {
            'uid': str(uid),
            'images_output': images_out,
            'masks_output': masks_out,
            'cam_view': cam_view,
            'cam_view_proj': cam_view_proj,
            'cam_pos': cam_pos,
        }

        # optional text
        if self.opt.text_file and os.path.exists(self.opt.text_file):
            text = read_prompt_line(self.opt.text_file, uid)
            token = self._safe_tokenize_text(text)
            results['token'] = token

        # mesh points/normals from mesh.obj
        mesh_dir = os.path.join(self.opt.mesh_root, str(uid))
        mesh_path = os.path.join(mesh_dir, 'mesh.obj')
        if os.path.exists(mesh_path):
            mesh = trimesh.load(mesh_path, force='mesh')
            # make a writable copy to avoid read-only arrays side effects
            mesh = mesh.copy()
            # recenter to origin; optional normalize to unit sphere
            mesh.vertices -= mesh.bounding_box.centroid
            if getattr(self.opt, 'normalize_to_unit', False):
                distances = np.linalg.norm(mesh.vertices, axis=1)
                mesh.vertices /= (np.max(distances) + 1e-8)

            # coordinate transform via affine (avoid in-place writes to face_normals)
            R = np.eye(3, dtype=np.float64)
            if self.opt.coord_swap_yz:
                R = R @ np.array([[1, 0, 0],
                                  [0, 0, 1],
                                  [0, 1, 0]], dtype=np.float64)
            if self.opt.coord_flip_x:
                R = np.diag([-1, 1, 1]) @ R
            if self.opt.coord_flip_y:
                R = np.diag([1, -1, 1]) @ R
            if self.opt.coord_flip_z:
                R = np.diag([1, 1, -1]) @ R
            if not np.allclose(R, np.eye(3)):
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                mesh.apply_transform(T)
            pts, face_idx = trimesh.sample.sample_surface(mesh, self.opt.num_points)
            norms = mesh.face_normals[face_idx]
            results['points'] = pts.astype(np.float32)
            results['normals'] = norms.astype(np.float32)
        else:
            # fallback: zeros (模型内部会构建 octree，需要 points/normals)
            results['points'] = np.zeros((self.opt.num_points, 3), dtype=np.float32)
            results['normals'] = np.zeros((self.opt.num_points, 3), dtype=np.float32)

        return results


