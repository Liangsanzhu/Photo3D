import tyro
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from safetensors.torch import load_file
import numpy as np
import kiui
import copy
import json
import random
from tqdm import tqdm

from core.options import AllConfigs, Options
from core.regression_models import TexGaussian
from core.dataset_grid import GridDataset, read_prompt_line
from core.dataset import collate_func

_CLIP_MODEL = None
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
_DINO_MODEL = None


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def get_clip_model(device: torch.device):
    global _CLIP_MODEL, _CLIP_MEAN, _CLIP_STD
    if _CLIP_MODEL is None:
        import clip
        _CLIP_MODEL, _ = clip.load('ViT-B/32', device=device, jit=False)
        for p in _CLIP_MODEL.parameters():
            p.requires_grad_(False)
        _CLIP_MODEL.eval()
    else:
        _CLIP_MODEL = _CLIP_MODEL.to(device).eval()
    _CLIP_MEAN = _CLIP_MEAN.to(device)
    _CLIP_STD = _CLIP_STD.to(device)
    return _CLIP_MODEL


def get_dino_model(device: torch.device):
    global _DINO_MODEL
    if _DINO_MODEL is None:
        _DINO_MODEL = torch.hub.load(repo_or_dir='facebookresearch/dinov3', model='dinov3_vitl16', source='github')
        for p in _DINO_MODEL.parameters():
            p.requires_grad_(False)
        _DINO_MODEL.eval()
    _DINO_MODEL = _DINO_MODEL.to(device).eval()
    return _DINO_MODEL


def _extract_dino_features_resized(img_chw: torch.Tensor, model, target_q: int, with_grad: bool) -> torch.Tensor:
    bchw = img_chw.unsqueeze(0)  # [1, 3, H, W]
    bchw = F.interpolate(bchw, size=(target_q, target_q), mode='bilinear', align_corners=False)  # [1, 3, Q, Q]
    mean = torch.tensor([0.485, 0.456, 0.406], device=bchw.device).view(1, 3, 1, 1)  # [1, 3, 1, 1]
    std = torch.tensor([0.229, 0.224, 0.225], device=bchw.device).view(1, 3, 1, 1)  # [1, 3, 1, 1]
    x = (bchw - mean) / std  # [1, 3, Q, Q]
    if with_grad:
        feats_list = model.get_intermediate_layers(x, n=1, reshape=True, norm=True)
    else:
        with torch.no_grad():
            feats_list = model.get_intermediate_layers(x, n=1, reshape=True, norm=True)
    feats = feats_list[-1].squeeze(0)  # [C, Ht, Wt]
    feats = F.normalize(feats, p=2, dim=0)  # [C, Ht, Wt]
    return feats


def _one_to_one_greedy_max_matching(similarity: torch.Tensor):
    na, nb = similarity.shape
    work = similarity.clone()  # [Na, Nb]
    neg_inf = torch.tensor(-1e9, dtype=work.dtype, device=work.device)
    vals = []
    for _ in range(min(na, nb)):
        v, idx = work.view(-1).max(dim=0)
        if v.item() <= -1e8:
            break
        r = int((idx // nb).item())
        c = int((idx % nb).item())
        vals.append(v)
        work[r, :] = neg_inf  # [Nb]
        work[:, c] = neg_inf  # [Na]
    if len(vals) == 0:
        return torch.zeros(0, dtype=similarity.dtype, device=similarity.device)  # [0]
    return torch.stack(vals)  # [M]


def compute_adapt_clip_crop_loss(
    pred_views: torch.Tensor,
    gt_views: torch.Tensor,
    num_crops: int,
    min_scale: float,
    max_scale: float,
    clip_input_size: int,
) -> torch.Tensor:
    clip_model = get_clip_model(pred_views.device)
    n, _, h, w = pred_views.shape  # [N, 3, H, W]
    losses = []
    for i in range(n):
        pred_full = pred_views[i:i + 1].clamp(0, 1)  # [1, 3, H, W]
        gt_full = gt_views[i:i + 1].clamp(0, 1)  # [1, 3, H, W]
        for _ in range(int(num_crops)):
            scale = random.uniform(float(min_scale), float(max_scale))
            ch = min(h, max(32, int(h * scale)))
            cw = min(w, max(32, int(w * scale)))
            y0 = 0 if h == ch else random.randint(0, h - ch)
            x0 = 0 if w == cw else random.randint(0, w - cw)
            pred_crop = pred_full[:, :, y0:y0 + ch, x0:x0 + cw]  # [1, 3, ch, cw]
            gt_crop = gt_full[:, :, y0:y0 + ch, x0:x0 + cw]  # [1, 3, ch, cw]
            pred_resize = F.interpolate(
                pred_crop, size=(clip_input_size, clip_input_size), mode='bilinear', align_corners=False
            )  # [1, 3, S, S]
            gt_resize = F.interpolate(
                gt_crop, size=(clip_input_size, clip_input_size), mode='bilinear', align_corners=False
            )  # [1, 3, S, S]
            pred_norm = (pred_resize - _CLIP_MEAN) / _CLIP_STD  # [1, 3, S, S]
            gt_norm = (gt_resize - _CLIP_MEAN) / _CLIP_STD  # [1, 3, S, S]
            feat_pred = F.normalize(clip_model.encode_image(pred_norm), dim=-1)  # [1, D]
            feat_gt = F.normalize(clip_model.encode_image(gt_norm), dim=-1)  # [1, D]
            sim = (feat_pred * feat_gt).sum(dim=-1).mean()  # []
            losses.append(1.0 - sim)  # []
    if len(losses) == 0:
        return torch.tensor(0.0, device=pred_views.device)
    return torch.stack(losses).mean()  # []


def compute_match_fine_loss(
    pred_views: torch.Tensor,
    gt_views: torch.Tensor,
    target_q: int,
    white_bg_mask: bool,
) -> torch.Tensor:
    device = pred_views.device
    dino_model = get_dino_model(device)
    vals_all = []
    n = pred_views.shape[0]  # [N, 3, H, W]
    for i in range(n):
        pred = pred_views[i].clamp(0, 1)  # [3, H, W]
        gt = gt_views[i].clamp(0, 1)  # [3, H, W]
        feat_pred = _extract_dino_features_resized(pred, dino_model, target_q=int(target_q), with_grad=True)  # [C, Hp, Wp]
        feat_gt = _extract_dino_features_resized(gt, dino_model, target_q=int(target_q), with_grad=False)  # [C, Hg, Wg]
        c, hp, wp = feat_pred.shape
        _, hg, wg = feat_gt.shape
        pred_flat = feat_pred.permute(1, 2, 0).reshape(-1, c)  # [Np, C]
        gt_flat = feat_gt.permute(1, 2, 0).reshape(-1, c)  # [Ng, C]
        keep_pred = torch.ones(hp * wp, dtype=torch.bool, device=device)  # [Np]
        keep_gt = torch.ones(hg * wg, dtype=torch.bool, device=device)  # [Ng]
        if white_bg_mask:
            thr = 250
            pred_np = (pred.detach().permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)  # [H, W, 3]
            gt_np = (gt.detach().permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)  # [H, W, 3]
            pred_bg = (pred_np[..., 0] >= thr) & (pred_np[..., 1] >= thr) & (pred_np[..., 2] >= thr)  # [H, W]
            gt_bg = (gt_np[..., 0] >= thr) & (gt_np[..., 1] >= thr) & (gt_np[..., 2] >= thr)  # [H, W]
            pred_fg = torch.from_numpy(~pred_bg).float().unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
            gt_fg = torch.from_numpy(~gt_bg).float().unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
            keep_pred = F.interpolate(pred_fg, size=(hp, wp), mode='nearest').view(-1) > 0.5  # [Np]
            keep_gt = F.interpolate(gt_fg, size=(hg, wg), mode='nearest').view(-1) > 0.5  # [Ng]
        if keep_pred.sum().item() == 0 or keep_gt.sum().item() == 0:
            continue
        pred_sel = pred_flat[keep_pred]  # [Np2, C]
        gt_sel = gt_flat[keep_gt]  # [Ng2, C]
        sims = pred_sel @ gt_sel.t()  # [Np2, Ng2]
        matched_vals = _one_to_one_greedy_max_matching(sims)  # [M]
        if matched_vals.numel() > 0:
            vals_all.append(matched_vals.mean())  # []
    if len(vals_all) == 0:
        return torch.tensor(1.0, device=device)
    mean_sim = torch.stack(vals_all).mean()  # []
    return 1.0 - mean_sim  # []


def main():
    opt = tyro.cli(Options)

    # 强制设置：不用 gaussian_loss（grid 训练走自定义 loss）
    opt.gaussian_loss = 'False'
    # 开启横向翻转（用于 texture 渲染的可视化）
    opt.texture_flip_horizontal = False

    opt.gaussian_loss = str2bool(opt.gaussian_loss)
    opt.use_text = str2bool(opt.use_text)
    # optional bools from CLI when provided as strings
    if isinstance(opt.use_material, str):
        opt.use_material = str2bool(opt.use_material)
    if hasattr(opt, 'supervise_material') and isinstance(opt.supervise_material, str):
        opt.supervise_material = str2bool(opt.supervise_material)

    os.makedirs(opt.workspace, exist_ok=True)
    accelerator = Accelerator(mixed_precision=opt.mixed_precision, gradient_accumulation_steps=opt.gradient_accumulation_steps)
    device = accelerator.device

    # 文本条件由 CLI 参数控制（--use_text / --text_file）

    # 组装 id 列表
    all_ids = GridDataset.scan_ids(opt)
    if len(all_ids) == 0:
        raise RuntimeError("No valid samples found. Please check grid_image_dir / mesh_root / text_file.")
    # 95/5 划分，或进入单样本过拟合模式
    if getattr(opt, 'overfit_one', False):
        if getattr(opt, 'overfit_id', -1) in all_ids:
            sel = [opt.overfit_id]
        else:
            sel = [all_ids[0]]
        train_ids = sel
        test_ids = sel
        print(f"[overfit] using single id {sel[0]}")
    else:
        split = max(1, int(0.05 * len(all_ids)))
        train_ids = all_ids[:-split] if len(all_ids) > 1 else all_ids
        test_ids = all_ids[-split:] if len(all_ids) > 1 else all_ids

    train_dataset = GridDataset(opt, train_ids, training=True)
    test_dataset = GridDataset(opt, test_ids, training=False)

    # 不随机打乱，便于严格对比
    # 过拟合模式强制单样本、batch_size=1、不要丢弃最后一个batch
    if getattr(opt, 'overfit_one', False):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False, collate_fn=collate_func)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False, collate_fn=collate_func)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.batch_size, pin_memory=True, drop_last=True, collate_fn=collate_func)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False, collate_fn=collate_func)

    # 评估可视化的张数上限（例如保存前 20 张）
    num_eval_vis = 20

    # 为避免模型构造时对 opt 的副作用（如 out_channels 递增），分别使用深拷贝
    opt_for_model = copy.deepcopy(opt)
    model = TexGaussian(opt_for_model, device)

    # 基线：加载原始 PBR 预训练模型用于可视化对比（只做推理，不参与训练）
    opt_for_baseline = copy.deepcopy(opt)
    baseline_model = TexGaussian(opt_for_baseline, device)
    try:
        pbr_ckpt_path = os.path.join(os.path.dirname(__file__), "PBR_model.safetensors")
        if pbr_ckpt_path.endswith('safetensors'):
            pbr_ckpt = load_file(pbr_ckpt_path, device='cpu')
        else:
            pbr_ckpt = torch.load(pbr_ckpt_path, map_location='cpu')

        b_state = baseline_model.state_dict()
        loaded_keys = []
        mismatched = []
        unexpected = []
        for k, v in pbr_ckpt.items():
            if k in b_state:
                if b_state[k].shape == v.shape:
                    b_state[k].copy_(v)
                    loaded_keys.append(k)
                else:
                    mismatched.append((k, tuple(v.shape), tuple(b_state[k].shape)))
            else:
                unexpected.append((k, tuple(v.shape)))
        missing = [k for k in b_state.keys() if k not in pbr_ckpt]
        # 打印与写文件（仅主进程）
        try:
            report = []
            report.append(f"[baseline-ckpt] path={pbr_ckpt_path}")
            report.append(f"  loaded={len(loaded_keys)} mismatched={len(mismatched)} unexpected={len(unexpected)} missing={len(missing)}")
            preview_n = 20
            if len(mismatched) > 0:
                report.append("  mismatched (name, ckpt_shape, model_shape) [preview]:")
                for item in mismatched[:preview_n]:
                    report.append(f"    {item[0]} {item[1]} -> {item[2]}")
            if len(unexpected) > 0:
                report.append("  unexpected (name, ckpt_shape) [preview]:")
                for item in unexpected[:preview_n]:
                    report.append(f"    {item[0]} {item[1]}")
            if len(missing) > 0:
                report.append("  missing (in model not in ckpt) [preview]:")
                for k in missing[:preview_n]:
                    report.append(f"    {k}")
            if accelerator.is_main_process:
                print("\n".join(report))
                # 写完整报告到文件
                os.makedirs(opt.workspace, exist_ok=True)
                out_path = os.path.join(opt.workspace, 'baseline_ckpt_report.txt')
                with open(out_path, 'w') as f:
                    f.write("\n".join(report))
                    if len(mismatched) > preview_n:
                        f.write("\nFULL mismatched list:\n")
                        for item in mismatched:
                            f.write(f"{item[0]} {item[1]} -> {item[2]}\n")
                    if len(unexpected) > preview_n:
                        f.write("\nFULL unexpected list:\n")
                        for item in unexpected:
                            f.write(f"{item[0]} {item[1]}\n")
                    if len(missing) > preview_n:
                        f.write("\nFULL missing list:\n")
                        for k in missing:
                            f.write(f"{k}\n")
        except Exception as e2:
            print(f"[WARN] failed to write baseline ckpt report: {e2}")
        baseline_model.to(device)
        baseline_model.eval()
    except Exception as e:
        print(f"[WARN] failed to load baseline PBR model: {e}")

    # 预训练权重加载（容错，只拷贝形状匹配的参数）。若未传入 resume，则默认加载 PBR 预训练权重。
    if opt.resume is not None:
        print(opt.resume)
        # support directory or file path
        resume_path = opt.resume
        if os.path.isdir(resume_path):
            cand1 = os.path.join(resume_path, 'model.safetensors')
            cand2 = os.path.join(resume_path, 'pytorch_model.bin')
            if os.path.exists(cand1):
                resume_path = cand1
            elif os.path.exists(cand2):
                resume_path = cand2
        if resume_path.endswith('safetensors'):
            ckpt = load_file(resume_path, device='cpu')
        else:
            ckpt = torch.load(resume_path, map_location='cpu')

        print('Start loading checkpoint')
        state_dict = model.state_dict()
        loaded_keys = []
        mismatched = []
        unexpected = []
        for k, v in ckpt.items():
            if k in state_dict:
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                    loaded_keys.append(k)
                else:
                    mismatched.append((k, tuple(v.shape), tuple(state_dict[k].shape)))
            else:
                unexpected.append((k, tuple(v.shape)))
        missing = [k for k in state_dict.keys() if k not in ckpt]
        try:
            report = []
            report.append(f"[train-ckpt] path={opt.resume}")
            report.append(f"  loaded={len(loaded_keys)} mismatched={len(mismatched)} unexpected={len(unexpected)} missing={len(missing)}")
            preview_n = 20
            if len(mismatched) > 0:
                report.append("  mismatched (name, ckpt_shape, model_shape) [preview]:")
                for item in mismatched[:preview_n]:
                    report.append(f"    {item[0]} {item[1]} -> {item[2]}")
            if len(unexpected) > 0:
                report.append("  unexpected (name, ckpt_shape) [preview]:")
                for item in unexpected[:preview_n]:
                    report.append(f"    {item[0]} {item[1]}")
            if len(missing) > 0:
                report.append("  missing (in model not in ckpt) [preview]:")
                for k in missing[:preview_n]:
                    report.append(f"    {k}")
            if accelerator.is_main_process:
                print("\n".join(report))
                out_path = os.path.join(opt.workspace, 'train_ckpt_report.txt')
                with open(out_path, 'w') as f:
                    f.write("\n".join(report))
                    if len(mismatched) > preview_n:
                        f.write("\nFULL mismatched list:\n")
                        for item in mismatched:
                            f.write(f"{item[0]} {item[1]} -> {item[2]}\n")
                    if len(unexpected) > preview_n:
                        f.write("\nFULL unexpected list:\n")
                        for item in unexpected:
                            f.write(f"{item[0]} {item[1]}\n")
                    if len(missing) > preview_n:
                        f.write("\nFULL missing list:\n")
                        for k in missing:
                            f.write(f"{k}\n")
        except Exception as e2:
            print(f"[WARN] failed to write train ckpt report: {e2}")
    else:
        try:
            default_resume = os.path.join(os.path.dirname(__file__), "PBR_model.safetensors")
            print(default_resume)
            if default_resume.endswith('safetensors'):
                ckpt = load_file(default_resume, device='cpu')
            else:
                ckpt = torch.load(default_resume, map_location='cpu')
            print('Start loading checkpoint (default PBR)')
            state_dict = model.state_dict()
            loaded_keys = []
            mismatched = []
            unexpected = []
            for k, v in ckpt.items():
                if k in state_dict:
                    if state_dict[k].shape == v.shape:
                        state_dict[k].copy_(v)
                        loaded_keys.append(k)
                    else:
                        mismatched.append((k, tuple(v.shape), tuple(state_dict[k].shape)))
                else:
                    unexpected.append((k, tuple(v.shape)))
            missing = [k for k in state_dict.keys() if k not in ckpt]
            try:
                report = []
                report.append(f"[train-ckpt-default] path={default_resume}")
                report.append(f"  loaded={len(loaded_keys)} mismatched={len(mismatched)} unexpected={len(unexpected)} missing={len(missing)}")
                preview_n = 20
                if len(mismatched) > 0:
                    report.append("  mismatched (name, ckpt_shape, model_shape) [preview]:")
                    for item in mismatched[:preview_n]:
                        report.append(f"    {item[0]} {item[1]} -> {item[2]}")
                if len(unexpected) > 0:
                    report.append("  unexpected (name, ckpt_shape) [preview]:")
                    for item in unexpected[:preview_n]:
                        report.append(f"    {item[0]} {item[1]}")
                if len(missing) > 0:
                    report.append("  missing (in model not in ckpt) [preview]:")
                    for k in missing[:preview_n]:
                        report.append(f"    {k}")
                if accelerator.is_main_process:
                    print("\n".join(report))
                    out_path = os.path.join(opt.workspace, 'train_ckpt_report.txt')
                    with open(out_path, 'w') as f:
                        f.write("\n".join(report))
                        if len(mismatched) > preview_n:
                            f.write("\nFULL mismatched list:\n")
                            for item in mismatched:
                                f.write(f"{item[0]} {item[1]} -> {item[2]}\n")
                        if len(unexpected) > preview_n:
                            f.write("\nFULL unexpected list:\n")
                            for item in unexpected:
                                f.write(f"{item[0]} {item[1]}\n")
                        if len(missing) > preview_n:
                            f.write("\nFULL missing list:\n")
                            for k in missing:
                                f.write(f"{k}\n")
            except Exception as e2:
                print(f"[WARN] failed to write train ckpt report: {e2}")
        except Exception as e:
            print(f"[WARN] failed to load default PBR checkpoint for training model: {e}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=opt.lr, weight_decay=0.0, betas=(0.9, 0.95))
    total_steps = opt.num_epochs * max(1, len(train_loader))
    pct_start = 10 / max(11, total_steps)#3000 / max(3001, total_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start)

    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, test_loader, scheduler)

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=opt.workspace)
        # 创建可视化输出目录
        opt.pred_image_dir = os.path.join(opt.workspace, 'pred_images')
        opt.gt_image_dir = os.path.join(opt.workspace, 'gt_images')
        opt.eval_pred_image_dir = os.path.join(opt.workspace, 'eval_pred_images')
        opt.eval_gt_image_dir = os.path.join(opt.workspace, 'eval_gt_images')
        # 基线可视化输出目录
        opt.pred_image_dir_pbr = os.path.join(opt.workspace, 'pred_images_pbr')
        opt.eval_pred_image_dir_pbr = os.path.join(opt.workspace, 'eval_pred_images_pbr')
        os.makedirs(opt.pred_image_dir, exist_ok=True)
        os.makedirs(opt.gt_image_dir, exist_ok=True)
        os.makedirs(opt.eval_pred_image_dir, exist_ok=True)
        os.makedirs(opt.eval_gt_image_dir, exist_ok=True)
        os.makedirs(opt.pred_image_dir_pbr, exist_ok=True)
        os.makedirs(opt.eval_pred_image_dir_pbr, exist_ok=True)

        # 训练开始前：先可视化一次训练 batch（当前模型预测与 GT）
        model.eval()
        try:
            first_batch = next(iter(train_loader))
            with torch.no_grad():
                out0 = model(first_batch, ema=True)
                # ---- debug: confirm text usage on train batch (main proc only) ----
                if accelerator.is_main_process and opt.use_text and opt.text_file:
                    try:
                        uids = first_batch.get('uid', [])
                        if isinstance(uids, (list, tuple)):
                            uid_list = [int(u) for u in uids]
                        elif torch.is_tensor(uids):
                            uid_list = [int(x) for x in uids.view(-1).tolist()]
                        else:
                            uid_list = [int(uids)]
                        print(f"[debug-text][train] use_text=True, text_file={opt.text_file}")
                        for uid in uid_list[:4]:
                            try:
                                t = read_prompt_line(opt.text_file, uid)
                                print(f"  id={uid} prompt[:120]={t[:120]}")
                            except Exception as e:
                                print(f"  id={uid} prompt read failed: {e}")
                    except Exception as e:
                        print(f"[debug-text][train] failed to dump prompts: {e}")
                # 基线模型对同一 batch 推理
                try:
                    # 确保数据与模型在同一设备
                    first_batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in first_batch.items()}
                    out0_pbr = baseline_model(first_batch_dev, ema=True)
                    if 'images_pred' in out0_pbr:
                        p_pred = out0_pbr['images_pred'].detach().cpu().numpy()
                        p_pred = p_pred.transpose(0, 3, 1, 4, 2).reshape(-1, p_pred.shape[1] * p_pred.shape[3], 3)
                        kiui.write_image(f"{opt.pred_image_dir_pbr}/step_0.jpg", p_pred)
                except Exception as e:
                    print(f"[WARN] baseline pre-visualization failed: {e}")
            if 'images_pred' in out0 and 'images_output' in first_batch:
                pred_images = out0['images_pred'].detach().cpu().numpy()  # [B, V, 3, H, W]
                pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                kiui.write_image(f"{opt.pred_image_dir}/step_0.jpg", pred_images)

                gt_images = first_batch['images_output'].detach().cpu().numpy()
                gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3)
                kiui.write_image(f"{opt.gt_image_dir}/step_0.jpg", gt_images)
            # debug: dump initial config and simple stats
            if getattr(opt, 'debug_log', False):
                try:
                    dbg = {
                        'coord_swap_yz': getattr(opt, 'coord_swap_yz', None),
                        'coord_flip_x': getattr(opt, 'coord_flip_x', None),
                        'coord_flip_y': getattr(opt, 'coord_flip_y', None),
                        'coord_flip_z': getattr(opt, 'coord_flip_z', None),
                        'grid_yaw_offset_deg': getattr(opt, 'grid_yaw_offset_deg', None),
                        'grid_cam_radius_override': getattr(opt, 'grid_cam_radius_override', None),
                        'grid_fovy': getattr(opt, 'grid_fovy', None),
                    }
                    if 'images_pred' in out0 and 'images_output' in first_batch and 'masks_output' in first_batch:
                        try:
                            pm = [float(out0['images_pred'][:, :, c].mean().item()) for c in range(3)]
                            gm = [float(first_batch['images_output'][:, :, c].mean().item()) for c in range(3)]
                            mc = float(first_batch['masks_output'].mean().item())
                            dbg.update({'pred_rgb_mean': pm, 'gt_rgb_mean': gm, 'mask_coverage': mc})
                        except Exception:
                            pass
                    os.makedirs(os.path.join(opt.workspace, 'debug'), exist_ok=True)
                    with open(os.path.join(opt.workspace, 'debug', 'debug_step_0.json'), 'w') as f:
                        json.dump(dbg, f, indent=2)
                except Exception as e:
                    print(f"[WARN] debug dump failed: {e}")
            # 训练开始前：可视化一次评估集（推理）
            with torch.no_grad():
                for j, eval_data in enumerate(test_loader):
                    eval_out = model(eval_data, ema=True)
                    if 'images_pred' in eval_out:
                        e_pred = eval_out['images_pred'].detach().cpu().numpy()
                        e_pred = e_pred.transpose(0, 3, 1, 4, 2).reshape(-1, e_pred.shape[1] * e_pred.shape[3], 3)
                        kiui.write_image(f"{opt.eval_pred_image_dir}/step_0_{j}.jpg", e_pred)

                        e_gt = eval_data['images_output'].detach().cpu().numpy()
                        e_gt = e_gt.transpose(0, 3, 1, 4, 2).reshape(-1, e_gt.shape[1] * e_gt.shape[3], 3)
                        kiui.write_image(f"{opt.eval_gt_image_dir}/step_0_{j}.jpg", e_gt)
                        # ---- debug: confirm text usage on eval batch ----
                        if j == 0 and accelerator.is_main_process and opt.use_text and opt.text_file:
                            try:
                                uids = eval_data.get('uid', [])
                                if isinstance(uids, (list, tuple)):
                                    uid_list = [int(u) for u in uids]
                                elif torch.is_tensor(uids):
                                    uid_list = [int(x) for x in uids.view(-1).tolist()]
                                else:
                                    uid_list = [int(uids)]
                                print(f"[debug-text][eval] use_text=True, text_file={opt.text_file}")
                                for uid in uid_list[:4]:
                                    try:
                                        t = read_prompt_line(opt.text_file, uid)
                                        print(f"  id={uid} prompt[:120]={t[:120]}")
                                    except Exception as e:
                                        print(f"  id={uid} prompt read failed: {e}")
                            except Exception as e:
                                print(f"[debug-text][eval] failed to dump prompts: {e}")
                        # 基线模型在评估集上的可视化（保存前 num_eval_vis 个）
                        try:
                            if j < num_eval_vis:
                                eval_data_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in eval_data.items()}
                                eval_out_pbr = baseline_model(eval_data_dev, ema=True)
                                if 'images_pred' in eval_out_pbr:
                                    ep_pred = eval_out_pbr['images_pred'].detach().cpu().numpy()
                                    ep_pred = ep_pred.transpose(0, 3, 1, 4, 2).reshape(-1, ep_pred.shape[1] * ep_pred.shape[3], 3)
                                    kiui.write_image(f"{opt.eval_pred_image_dir_pbr}/step_0_{j}.jpg", ep_pred)
                        except Exception as e:
                            print(f"[WARN] baseline eval pre-visualization failed: {e}")
                    if j + 1 >= num_eval_vis:
                        break
        except Exception as e:
            print(f"[WARN] pre-visualization failed: {e}")
        model.train()

    best_psnr = 0
    global_step = 0
    if float(opt.lambda_clip) <= 0.0 and float(opt.fine_match_weight) <= 0.0:
        raise ValueError("Both lambda_clip and fine_match_weight are <= 0. Please set at least one positive.")
    for epoch in range(opt.num_epochs):
        model.train()
        pbar = tqdm(total=len(train_loader), desc=f"epoch {epoch}", disable=not accelerator.is_main_process)
        total_loss = 0
        total_psnr = 0
        t0_iter = time.time()
        for i, data in enumerate(train_loader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                out = model(data)
                pred_views = out['images_pred']  # [B, V, 3, H, W]
                gt_views = data['images_output']  # [B, V, 3, H, W]
                b, v, c, h, w = pred_views.shape
                pred_flat = pred_views.reshape(b * v, c, h, w)  # [B*V, 3, H, W]
                gt_flat = gt_views.reshape(b * v, c, h, w)  # [B*V, 3, H, W]

                adapt_loss = torch.tensor(0.0, device=pred_views.device)
                if float(opt.lambda_clip) > 0.0:
                    adapt_loss = compute_adapt_clip_crop_loss(
                        pred_flat,
                        gt_flat,
                        num_crops=int(opt.clip_num_crops),
                        min_scale=float(opt.clip_min_scale),
                        max_scale=float(opt.clip_max_scale),
                        clip_input_size=int(opt.clip_input_size),
                    )  # []

                match_loss = torch.tensor(0.0, device=pred_views.device)
                if float(opt.fine_match_weight) > 0.0:
                    match_loss = compute_match_fine_loss(
                        pred_flat,
                        gt_flat,
                        target_q=int(opt.fine_match_target_q),
                        white_bg_mask=bool(opt.fine_match_white_bg),
                    )  # []

                loss = float(opt.lambda_clip) * adapt_loss + float(opt.fine_match_weight) * match_loss  # []
                
                accelerator.backward(loss)
                #print(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)
                optimizer.step()
                scheduler.step()
                (model.module.update_EMA() if hasattr(model, 'module') else model.update_EMA())
                total_loss += loss.detach()
                total_psnr += out['psnr'].detach()
                global_step += 1

                if accelerator.is_main_process:
                    try:
                        t1_iter = time.time()
                        iter_time = t1_iter - t0_iter
                        t0_iter = t1_iter
                        cur_loss = float(loss.detach().cpu().item())
                        cur_adapt = float(adapt_loss.detach().cpu().item())
                        cur_match = float(match_loss.detach().cpu().item())
                        cur_psnr = float(out['psnr'].detach().cpu().item())
                        cur_lr = float(optimizer.param_groups[0]['lr'])
                        pbar.set_postfix({
                            'loss': f"{cur_loss:.4f}",
                            'adapt': f"{cur_adapt:.4f}",
                            'match': f"{cur_match:.4f}",
                            'psnr': f"{cur_psnr:.2f}",
                            'lr': f"{cur_lr:.2e}",
                            'time': f"{iter_time:.3f}s",
                        })
                        pbar.update(1)
                    except Exception:
                        pass

            # 每 image_interval 步保存一次可视化并执行一次评估（所有进程参与评估与通信，只有主进程做可视化与输出）
            if (global_step % opt.image_interval == 0):
                # 保存当前 batch 的预测与 GT 可视化（仅主进程）
                if accelerator.is_main_process and 'images_pred' in out and 'images_output' in data:
                    pred_images = out['images_pred'].detach().cpu().numpy()  # [B, V, 3, H, W]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f"{opt.pred_image_dir}/step_{global_step}.jpg", pred_images)

                    gt_images = data['images_output'].detach().cpu().numpy()  # [B, V, 3, H, W]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3)
                    kiui.write_image(f"{opt.gt_image_dir}/step_{global_step}.jpg", gt_images)
                    # 基线模型在当前 batch 上的可视化（仅主进程）
                    try:
                        with torch.no_grad():
                            data_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}
                            out_pbr = baseline_model(data_dev, ema=True)
                        if 'images_pred' in out_pbr:
                            p_pred = out_pbr['images_pred'].detach().cpu().numpy()
                            p_pred = p_pred.transpose(0, 3, 1, 4, 2).reshape(-1, p_pred.shape[1] * p_pred.shape[3], 3)
                            kiui.write_image(f"{opt.pred_image_dir_pbr}/step_{global_step}.jpg", p_pred)
                    except Exception as e:
                        print(f"[WARN] baseline train-step visualization failed: {e}")

                # 评估（所有进程参与）；仅主进程保存可视化
                model.eval()
                eval_psnr_step = 0
                with torch.no_grad():
                    for j, eval_data in enumerate(test_loader):
                        eval_out = model(eval_data, ema=True)
                        eval_psnr_step += eval_out['psnr'].detach()

                        # 保存前 num_eval_vis 个 batch 的可视化（仅主进程）
                        if accelerator.is_main_process and (j < num_eval_vis) and 'images_pred' in eval_out:
                            e_pred = eval_out['images_pred'].detach().cpu().numpy()
                            e_pred = e_pred.transpose(0, 3, 1, 4, 2).reshape(-1, e_pred.shape[1] * e_pred.shape[3], 3)
                            kiui.write_image(f"{opt.eval_pred_image_dir}/step_{global_step}_{j}.jpg", e_pred)

                            e_gt = eval_data['images_output'].detach().cpu().numpy()
                            e_gt = e_gt.transpose(0, 3, 1, 4, 2).reshape(-1, e_gt.shape[1] * e_gt.shape[3], 3)
                            kiui.write_image(f"{opt.eval_gt_image_dir}/step_{global_step}_{j}.jpg", e_gt)
                            # 基线模型在评估集上的可视化（仅保存前 num_eval_vis 个，仅主进程）
                            try:
                                eval_data_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in eval_data.items()}
                                eval_out_pbr = baseline_model(eval_data_dev, ema=True)
                                if 'images_pred' in eval_out_pbr:
                                    ep_pred = eval_out_pbr['images_pred'].detach().cpu().numpy()
                                    ep_pred = ep_pred.transpose(0, 3, 1, 4, 2).reshape(-1, ep_pred.shape[1] * ep_pred.shape[3], 3)
                                    kiui.write_image(f"{opt.eval_pred_image_dir_pbr}/step_{global_step}_{j}.jpg", ep_pred)
                            except Exception as e:
                                print(f"[WARN] baseline eval-step visualization failed: {e}")

                eval_psnr_step = accelerator.gather_for_metrics(eval_psnr_step).mean() / max(1, len(test_loader))
                accelerator.print(f"[step-eval] step {global_step} psnr {eval_psnr_step:.4f}")
                # debug dump at interval（仅主进程写文件）
                if accelerator.is_main_process and getattr(opt, 'debug_log', False) and (global_step % max(1, getattr(opt, 'debug_interval', 100)) == 0):
                    try:
                        with torch.no_grad():
                            if 'images_pred' in out and 'images_output' in data and 'masks_output' in data:
                                pm = [float(out['images_pred'][:, :, c].mean().item()) for c in range(3)]
                                gm = [float(data['images_output'][:, :, c].mean().item()) for c in range(3)]
                                # std for better diagnosis
                                pstd = [float(out['images_pred'][:, :, c].std().item()) for c in range(3)]
                                gstd = [float(data['images_output'][:, :, c].std().item()) for c in range(3)]
                                mc = float(data['masks_output'].mean().item())
                            else:
                                pm, gm, pstd, gstd, mc = None, None, None, None, None
                            lr_cur = float(optimizer.param_groups[0]['lr'])
                            dbg = {
                                'global_step': int(global_step),
                                'eval_psnr_step': float(eval_psnr_step.item()),
                                'lr': lr_cur,
                                'pred_rgb_mean': pm,
                                'pred_rgb_std': pstd,
                                'gt_rgb_mean': gm,
                                'gt_rgb_std': gstd,
                                'mask_coverage': mc,
                            }
                            os.makedirs(os.path.join(opt.workspace, 'debug'), exist_ok=True)
                            with open(os.path.join(opt.workspace, 'debug', f'debug_step_{global_step}.json'), 'w') as f:
                                json.dump(dbg, f, indent=2)
                    except Exception as e:
                        print(f"[WARN] debug dump failed: {e}")
                # 用 step 评估更新 best 模型（仅主进程）
                if accelerator.is_main_process:
                    # always save latest
                    latest_dir = os.path.join(opt.workspace, 'latest_ckpt')
                    os.makedirs(latest_dir, exist_ok=True)
                    accelerator.save_model(model, latest_dir)
                    # update best if improved
                    if eval_psnr_step > best_psnr:
                        best_psnr = eval_psnr_step
                        accelerator.save_model(model, os.path.join(opt.workspace, 'best_ckpt'))
                model.train()
        if accelerator.is_main_process:
            try:
                pbar.close()
            except Exception:
                pass
        total_loss = accelerator.gather_for_metrics(total_loss).mean() / max(1, len(train_loader))
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean() / max(1, len(train_loader))
        if accelerator.is_main_process:
            print(f"[train] epoch {epoch} loss {total_loss.item():.6f} psnr {total_psnr.item():.4f}")
            writer.add_scalar('loss', total_loss, epoch)
            writer.add_scalar('psnr', total_psnr, epoch)

        # 取消按 epoch 的整集评估与外部推理，仅依赖 image_interval 的 step 评估与可视化

    if accelerator.is_main_process:
        writer.close()


if __name__ == "__main__":
    main()
