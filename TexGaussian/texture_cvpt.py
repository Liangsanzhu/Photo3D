import os
import tyro
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import re

import trimesh
import os
from core.regression_models import TexGaussian
from core.options import AllConfigs, Options
from core.gs import GaussianRenderer
from external.clip import tokenize

from ocnn.octree import Octree, Points
import ocnn

import nvdiffrast.torch as dr

import kiui
from kiui.mesh import Mesh
from kiui.op import uv_padding
from kiui.cam import orbit_camera, get_perspective

from safetensors.torch import load_file

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Converter(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt
        self.device = torch.device("cuda")

        self.tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[2, 3] = 1

        self.gs_renderer = GaussianRenderer(opt)
        
        if self.opt.force_cuda_rast:
            self.glctx = dr.RasterizeCudaContext()
        else:
            self.glctx = dr.RasterizeGLContext()
    
        self.proj = torch.from_numpy(get_perspective(self.opt.fovy)).float().to(self.device)
        self.v = self.f = None
        self.vt = self.ft = None
        self.deform = None

        self.model = TexGaussian(opt, self.device)

        self.pointcloud_dir = self.opt.pointcloud_dir

        if self.opt.use_text:
            text = self.opt.text_prompt

            token = tokenize(text)
            token = token.to(self.device)

            self.text_embedding = self.model.text_encoder.encode(token).float() # [bs, 77, 768]
    
    def normalize_mesh(self):
        self.mesh.vertices = self.mesh.vertices - self.mesh.bounding_box.centroid
        distances = np.linalg.norm(self.mesh.vertices, axis=1)
        self.mesh.vertices /= np.max(distances)

    def load_mesh(self, path, num_samples = 200000):
        self.mesh = trimesh.load(path, force = 'mesh')
        self.normalize_mesh()

        point, idx = trimesh.sample.sample_surface(self.mesh, num_samples)
        normals = self.mesh.face_normals[idx]
        
        points_gt = Points(points = torch.from_numpy(point).float(), normals = torch.from_numpy(normals).float())
        points_gt.clip(min=-1, max=1)

        points = [points_gt]
        points = [pts.cuda(non_blocking = True) for pts in points]
        
        octrees = [self.points2octree(pts) for pts in points]
        octree_in = ocnn.octree.merge_octrees(octrees)

        octree_in.construct_all_neigh()

        xyzb = octree_in.xyzb(depth = octree_in.depth, nempty = True)
        x, y, z, b = xyzb
        xyz = torch.stack([x,y,z], dim = 1)
        octree_in.position = 2 * xyz / (2 ** octree_in.depth) - 1

        self.octree_in = octree_in

        self.input_data = self.octree_in.get_input_feature(feature = self.opt.input_feature, nempty = True) 

    def points2octree(self, points):
        octree = ocnn.octree.Octree(depth = self.opt.input_depth, full_depth = self.opt.full_depth)
        octree.build_octree(points)
        return octree
    
    def load_ckpt(self, ckpt_path):

        print('Start loading checkpoint')

        if ckpt_path.endswith('safetensors'):
            ckpt = load_file(ckpt_path, device='cpu')
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu')
        
        state_dict = self.model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict:
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                print(f'[WARN] unexpected param {k}: {v.shape}')
       
    @torch.no_grad()
    def render_gs(self, pose, use_material = False):
    
        cam_poses = torch.from_numpy(pose).unsqueeze(0).to(self.device)
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]

        batch_id = self.octree_in.batch_id(self.opt.input_depth, nempty = True)
        
        if use_material:
            out = self.gs_renderer.render(self.mr_gaussians, batch_id, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0))
        else:
            out = self.gs_renderer.render(self.gaussians, batch_id, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0))
        
        image = out['image'].squeeze(1).squeeze(0) # [C, H, W]
        alpha = out['alpha'].squeeze(2).squeeze(1).squeeze(0) # [H, W]

        return image, alpha
    
    def render_mesh(self, pose, use_material = False):

        h = w = self.opt.output_size

        v = self.v
        f = self.f

        pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)

        # get v_clip and render rgb
        v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ self.proj.T

        rast, rast_db = dr.rasterize(self.glctx, v_clip, f, (h, w))

        alpha = torch.clamp(rast[..., -1:], 0, 1).contiguous() # [1, H, W, 1]
        alpha = dr.antialias(alpha, rast, v_clip, f).clamp(0, 1).squeeze(-1).squeeze(0) # [H, W] important to enable gradients!
        
        texc, texc_db = dr.interpolate(self.vt.unsqueeze(0), rast, self.ft, rast_db=rast_db, diff_attrs='all')
        if use_material:
            image = torch.sigmoid(dr.texture(self.mr_albedo.unsqueeze(0), texc, uv_da=texc_db)) # [1, H, W, 3]
        else:
            image = torch.sigmoid(dr.texture(self.albedo.unsqueeze(0), texc, uv_da=texc_db)) # [1, H, W, 3]

        image = image.view(1, h, w, 3)
        # image = dr.antialias(image, rast, v_clip, f).clamp(0, 1)
        image = image.squeeze(0).permute(2, 0, 1).contiguous() # [3, H, W]
        image = alpha * image + (1 - alpha)

        return image, alpha
    
    # uv mesh refine
    def fit_mesh_uv(self, iters=1024, resolution=512, texture_resolution=1024, padding=2):

        # Check if output files already exist
        save_dir = os.path.join(self.opt.output_dir, self.opt.texture_name)
        albedo_path = os.path.join(save_dir, 'albedo_mesh.obj')
        mr_path = os.path.join(save_dir, 'mr_mesh.obj')
        
        files_exist = os.path.exists(albedo_path)
        if self.opt.use_material:
            files_exist = files_exist and os.path.exists(mr_path)
        
        if files_exist:
            print(f'[INFO] Mesh files already exist, skipping fit_mesh_uv: {save_dir}')
            return

        if self.opt.use_material:
            _, self.gaussians, self.mr_gaussians = self.model.forward_gaussians(self.input_data, self.octree_in, condition = self.text_embedding, data = None, ema = True)
        else:
            _, self.gaussians = self.model.forward_gaussians(self.input_data, self.octree_in, condition = self.text_embedding, data = None, ema = True)

        self.opt.output_size = resolution

        v = self.mesh.vertices.astype(np.float32)
        f = self.mesh.faces.astype(np.int32)

        self.v = torch.from_numpy(v).contiguous().float().to(self.device)
        self.f = torch.from_numpy(f).contiguous().int().to(self.device)

        # unwrap uv
        print(f"[INFO] uv unwrapping...")
        mesh = Mesh(v=self.v, f=self.f, albedo=None, device=self.device)
        mesh.auto_normal()
        mesh.auto_uv()

        self.vt = mesh.vt
        self.ft = mesh.ft

        # render uv maps
        h = w = texture_resolution
        uv = mesh.vt * 2.0 - 1.0 # uvs to range [-1, 1]
        uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]

        rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(0), mesh.ft, (h, w)) # [1, h, w, 4]
        xyzs, _ = dr.interpolate(mesh.v.unsqueeze(0), rast, mesh.f) # [1, h, w, 3]
        mask, _ = dr.interpolate(torch.ones_like(mesh.v[:, :1]).unsqueeze(0), rast, mesh.f) # [1, h, w, 1]

        # masked query 
        xyzs = xyzs.view(-1, 3)
        mask = (mask > 0).view(-1)
        
        albedo = torch.zeros(h * w, 3, device=self.device, dtype=torch.float32)
        
        albedo = albedo.view(h, w, -1)
        mask = mask.view(h, w)
        albedo = uv_padding(albedo, mask, padding)

        if self.opt.use_material:
            mr_albedo = torch.zeros(h * w, 3, device=self.device, dtype=torch.float32)
        
            mr_albedo = mr_albedo.view(h, w, -1)
            mask = mask.view(h, w)
            mr_albedo = uv_padding(mr_albedo, mask, padding)

        # optimize texture
        self.albedo = nn.Parameter(albedo).to(self.device)

        if self.opt.use_material:
            self.mr_albedo = nn.Parameter(mr_albedo).to(self.device)
        
        optimizer = torch.optim.Adam([
            {'params': self.albedo, 'lr': 1e-1},
        ])

        if self.opt.use_material:
            mr_optimizer = torch.optim.Adam([
                {'params': self.mr_albedo, 'lr': 1e-3},
            ])

        vers = [-89, 89, 0, 0, 0, 0]
        hors = [0, 0, -90, 0, 90, 180]

        rad = self.opt.texture_cam_radius # np.random.uniform(1, 2)

        for (ver, hor) in zip(vers, hors):

            print(f"[INFO] fitting mesh albedo...")
            pbar = tqdm.trange(iters)

            for i in pbar:
                
                pose = orbit_camera(ver, hor, rad)
                
                image_gt, alpha_gt = self.render_gs(pose)
                image_pred, alpha_pred = self.render_mesh(pose)

                if opt.save_image:
                    image_gt_save = image_gt.detach().cpu().numpy()
                    image_gt_save = image_gt_save.transpose(1, 2, 0)
                    kiui.write_image(f'{self.opt.output_dir}/{opt.texture_name}/albedo_gt_images/{i}.jpg', image_gt_save)

                    image_pred_save = image_pred.detach().cpu().numpy()
                    image_pred_save = image_pred_save.transpose(1, 2, 0)
                    kiui.write_image(f'{self.opt.output_dir}/{opt.texture_name}/mesh_albedo_images/{i}.jpg', image_pred_save)

                loss_mse = F.mse_loss(image_pred, image_gt)
                loss = loss_mse

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                pbar.set_description(f"MSE = {loss_mse.item():.6f}")
        
        pbar = tqdm.trange(iters * 2)

        for i in pbar:

            # shrink to front view as we care more about it...
            ver = np.random.randint(-89, 89)
            hor = np.random.randint(-180, 180)
            
            pose = orbit_camera(ver, hor, rad)
            
            image_gt, alpha_gt = self.render_gs(pose)
            image_pred, alpha_pred = self.render_mesh(pose)

            if opt.save_image:
                image_gt_save = image_gt.detach().cpu().numpy()
                image_gt_save = image_gt_save.transpose(1, 2, 0)
                kiui.write_image(f'{self.opt.output_dir}/{opt.texture_name}/albedo_gt_images/{i}.jpg', image_gt_save)

                image_pred_save = image_pred.detach().cpu().numpy()
                image_pred_save = image_pred_save.transpose(1, 2, 0)
                kiui.write_image(f'{self.opt.output_dir}/{opt.texture_name}/mesh_albedo_images/{i}.jpg', image_pred_save)

            loss_mse = F.mse_loss(image_pred, image_gt)
            loss = loss_mse

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(f"MSE = {loss_mse.item():.6f}")
        
        print(f"[INFO] finished fitting mesh albedo!")
    
        if self.opt.use_material:

            for (ver, hor) in zip(vers, hors):

                print(f"[INFO] fitting mesh material...")
                pbar = tqdm.trange(iters)

                for i in pbar:
                    
                    pose = orbit_camera(ver, hor, rad)
                    
                    image_gt, alpha_gt = self.render_gs(pose, use_material = True)
                    image_pred, alpha_pred = self.render_mesh(pose, use_material = True)

                    if opt.save_image:
                        image_gt_save = image_gt.detach().cpu().numpy()
                        image_gt_save = image_gt_save.transpose(1, 2, 0)
                        kiui.write_image(f'{self.opt.output_dir}/{opt.texture_name}/material_gt_images/{i}.jpg', image_gt_save)

                        image_pred_save = image_pred.detach().cpu().numpy()
                        image_pred_save = image_pred_save.transpose(1, 2, 0)
                        kiui.write_image(f'{self.opt.output_dir}/{opt.texture_name}/mesh_material_images/{i}.jpg', image_pred_save)

                    loss_mse = F.mse_loss(image_pred, image_gt)
                    loss = loss_mse

                    loss.backward()

                    mr_optimizer.step()
                    mr_optimizer.zero_grad()

                    pbar.set_description(f"MSE = {loss_mse.item():.6f}")
        
            pbar = tqdm.trange(iters * 2)

            for i in pbar:

                # shrink to front view as we care more about it...
                ver = np.random.randint(-89, 89)
                hor = np.random.randint(-180, 180)
                
                pose = orbit_camera(ver, hor, rad)
                
                image_gt, alpha_gt = self.render_gs(pose, use_material = True)
                image_pred, alpha_pred = self.render_mesh(pose, use_material = True)

                if opt.save_image:
                    image_gt_save = image_gt.detach().cpu().numpy()
                    image_gt_save = image_gt_save.transpose(1, 2, 0)
                    kiui.write_image(f'{self.opt.output_dir}/{opt.texture_name}/material_gt_images/{i}.jpg', image_gt_save)

                    image_pred_save = image_pred.detach().cpu().numpy()
                    image_pred_save = image_pred_save.transpose(1, 2, 0)
                    kiui.write_image(f'{self.opt.output_dir}/{opt.texture_name}/mesh_material_images/{i}.jpg', image_pred_save)

                loss_mse = F.mse_loss(image_pred, image_gt)
                loss = loss_mse

                loss.backward()

                mr_optimizer.step()
                mr_optimizer.zero_grad()

                pbar.set_description(f"MSE = {loss_mse.item():.6f}")
            
            print(f"[INFO] finished fitting mesh material!")

    @torch.no_grad()
    def save_gaussian_prid_grid(self):
        # Ensure gaussians are prepared
        if self.opt.use_material:
            _, self.gaussians, self.mr_gaussians = self.model.forward_gaussians(
                self.input_data, self.octree_in, condition=self.text_embedding, data=None, ema=True
            )
        else:
            _, self.gaussians = self.model.forward_gaussians(
                self.input_data, self.octree_in, condition=self.text_embedding, data=None, ema=True
            )

        # Define 4 canonical views: left, front, right, back
        vers = [0, 0, 0, 0]
        hors = [-90, 0, 90, 180]
        rad = self.opt.texture_cam_radius

        # Collect RGBA views (transparent background)
        from PIL import Image as PILImage
        images = []  # list of PIL RGBA
        for ver, hor in zip(vers, hors):
            pose = orbit_camera(ver, hor, rad)
            image, alpha = self.render_gs(pose, use_material=False)
            rgb = image.detach().cpu().numpy().transpose(1, 2, 0)  # H, W, 3
            a = alpha.detach().cpu().numpy()  # H, W
            rgb = np.clip(rgb, 0.0, 1.0)
            a = np.clip(a, 0.0, 1.0)
            rgba = np.concatenate([rgb, a[..., None]], axis=-1)
            rgba_u8 = (rgba * 255.0).astype(np.uint8)
            images.append(PILImage.fromarray(rgba_u8, mode='RGBA'))

        # Compose 2x2 grid with transparency
        w, h = images[0].size
        canvas = PILImage.new('RGBA', (w * 2, h * 2), (0, 0, 0, 0))
        canvas.paste(images[0], (0, 0), images[0])
        canvas.paste(images[1], (w, 0), images[1])
        canvas.paste(images[2], (0, h), images[2])
        canvas.paste(images[3], (w, h), images[3])

        # Extract id from mesh_path: {id}_final.glb
        base = os.path.basename(self.opt.mesh_path)
        m = re.match(r'(.+?)_final\\.glb$', base)
        if m:
            id_str = m.group(1)
        else:
            id_str = os.path.splitext(base)[0]

        os.makedirs(self.opt.output_dir, exist_ok=True)
        out_path = os.path.join(self.opt.output_dir, f'{id_str}_prid.png')
        
        # Skip if file already exists
        if os.path.exists(out_path):
            print(f'[INFO] File already exists, skipping: {out_path}')
            return
        
        canvas.save(out_path)
        print(f'[INFO] Saved Gaussian 4-view grid to: {out_path}')

    @torch.no_grad()
    def export_mesh(self, save_dir):

        os.makedirs(save_dir, exist_ok=True)

        albedo_path = os.path.join(save_dir, 'albedo_mesh.obj')
        mr_path = os.path.join(save_dir, 'mr_mesh.obj')
        
        # Check if all output files already exist
        files_exist = os.path.exists(albedo_path)
        if self.opt.use_material:
            files_exist = files_exist and os.path.exists(mr_path)
        
        if files_exist:
            print(f'[INFO] Mesh files already exist, skipping export to: {save_dir}')
            return

        v = self.mesh.vertices.astype(np.float32)

        self.v = torch.from_numpy(v).contiguous().float().to(self.device)
        
        mesh = Mesh(v=self.v, f=self.f, vt=self.vt, ft=self.ft, albedo=torch.sigmoid(self.albedo), device=self.device)
        mesh.auto_normal()
        mesh.write(albedo_path)

        if self.opt.use_material:
            
            mr_mesh = Mesh(v=self.v, f=self.f, vt=self.vt, ft=self.ft, albedo=torch.sigmoid(self.mr_albedo), device=self.device)
            mr_mesh.auto_normal()
            mr_mesh.write(mr_path)

if __name__ == "__main__":

    opt = tyro.cli(AllConfigs)

    opt.use_checkpoint = str2bool(opt.use_checkpoint)
    opt.use_material = str2bool(opt.use_material)
    opt.save_image = str2bool(opt.save_image)
    opt.gaussian_loss = str2bool(opt.gaussian_loss)
    opt.use_local_pretrained_ckpt = str2bool(opt.use_local_pretrained_ckpt)

    output_dir = opt.output_dir
    os.makedirs(output_dir, exist_ok=True)

    name = opt.texture_name

    converter = Converter(opt).cuda() 

    converter.load_mesh(opt.mesh_path)
    converter.load_ckpt(opt.ckpt_path)

    # 保存高斯 4 视图格图（如果需要）
    if opt.save_image:
        converter.save_gaussian_prid_grid()

    # 始终尝试导出网格（内部有已存在检查，重复调用会自动跳过）
    # fit_mesh_uv 会先检查 obj 是否已存在，存在则直接返回
    converter.fit_mesh_uv(iters = 1000)
    converter.export_mesh(os.path.join(output_dir, opt.texture_name))
   