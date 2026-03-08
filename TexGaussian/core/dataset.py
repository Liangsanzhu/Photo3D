import os
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from external.clip import tokenize

from core.options import Options

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class CollateBatch:

    def __init__(self):
        pass

    def __call__(self, batch: list):
        assert type(batch) == list

        outputs = {}
        for key in batch[0].keys():
            outputs[key] = [b[key] for b in batch]

            if 'points' in key or 'normals' in key:
                outputs[key] = [torch.from_numpy(a) for a in outputs[key]]
        
            elif 'gaussian' in key:
                outputs[key] = torch.cat(outputs[key], dim = 0)
            
            elif 'uid' in key:
                pass
        
            else:
                outputs[key] = torch.stack(outputs[key])

        return outputs

def collate_func(batch):

    collate_batch = CollateBatch()
    output = collate_batch(batch)

    return output


class TexGaussianDataset(Dataset):

    def __init__(self, opt: Options, training=True):

        self.opt = opt
        self.training = training

        self.items = []

        if self.training:
            with open(opt.trainlist, "r") as file:
                lines = file.readlines()

        else:
            with open(opt.testlist, "r") as file:
                lines = file.readlines()
                lines = lines[:100]

        for line in lines:
            uid = line.strip()
            self.items.append(uid)

        if opt.use_text:
            self.text_prompt = pd.read_csv(opt.text_description)
            self.text_prompt.columns = ['id', 'string']

        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        uid = self.items[idx]

        image_dir = os.path.join(self.opt.image_dir, uid)
        pointcloud_path = os.path.join(self.opt.pointcloud_dir, uid + '.npz')

        if self.opt.gaussian_loss:
            gaussian_path = os.path.join(self.opt.gaussian_dir, uid, 'gaussian.pth')

        results = {}

        results['uid'] = uid

        pointcloud = np.load(pointcloud_path)
        results['points'] = pointcloud['points']
        results['normals'] = pointcloud['normals']

        if self.opt.gaussian_loss:
            gaussian = torch.load(gaussian_path)
            gaussian.requires_grad_(False)
            results['gaussian'] = gaussian

        # load num_views images
        images = []
        masks = []
        if self.opt.use_material:
            mr_images = []
        cam_poses = []

        vid_cnt = 0

        # TODO: choose views, based on your rendering settings
        vids = np.random.permutation(self.opt.total_num_views).tolist()

        camera_path = os.path.join(image_dir, 'cameras.npz')
        cameras = np.load(camera_path)

        for vid in vids:

            image_path = os.path.join(image_dir, f'{vid}.png')

            if self.opt.use_material:
                mr_image_path = os.path.join(image_dir, f'{vid}_mr.png')

            try:
                # TODO: load data (modify self.client here)
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                image = torch.from_numpy(image.astype(np.float32) / 255) # [512, 512, 4] in [0, 1]

                if self.opt.use_material:
                    mr_image = cv2.imread(mr_image_path, cv2.IMREAD_UNCHANGED)
                    mr_image = torch.from_numpy(mr_image.astype(np.float32) / 255) # [512, 512, 4] in [0, 1]

                c2w = cameras['poses'][vid] # [4, 4]
                c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)
            except Exception as e:
                continue

            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb

            if self.opt.use_material:
                mr_image = mr_image.permute(2, 0, 1) # [4, 512, 512]
                mr_image = mr_image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
                mr_image = mr_image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)

            if self.opt.use_material:
                mr_images.append(mr_image)

            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        images = torch.stack(images, dim=0) # [V, C, H, W]

        if self.opt.use_material:
            mr_images = torch.stack(mr_images, dim=0)

        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        if self.opt.use_material:
            results['mr_images_output'] = F.interpolate(mr_images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction

        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]

        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        if self.opt.use_text:
            text = self.text_prompt.loc[self.text_prompt['id'] == uid, 'string']
            text = text.iloc[0]
            token = tokenize(text)
            token = token.squeeze()
            results['token'] = token

        return results
