import tyro
import time
import random
import os
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from core.utils import seed_everything
from core.options import AllConfigs
from core.regression_models import TexGaussian
from accelerate import Accelerator
from safetensors.torch import load_file
from core.dataset import TexGaussianDataset as Dataset
from core.dataset import collate_func

import kiui
import pytz
from datetime import datetime
import os

def get_time():
    beijing_tz = pytz.timezone('Asia/Shanghai')
    beijing_time = datetime.now(beijing_tz)
    timestr = beijing_time.strftime("%Y.%m.%d-%H:%M:%S")
    return timestr

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    opt = tyro.cli(AllConfigs)

    opt.use_checkpoint = str2bool(opt.use_checkpoint)
    opt.use_material = str2bool(opt.use_material)
    opt.gaussian_loss = str2bool(opt.gaussian_loss)
    opt.use_text = str2bool(opt.use_text)
    opt.use_local_pretrained_ckpt = str2bool(opt.use_local_pretrained_ckpt)

    current_time = get_time()

    experiment_name = f"{current_time}_lr_{opt.lr}_num_views_{opt.num_views}"

    opt.workspace = os.path.join(opt.workspace, experiment_name)

    os.makedirs(opt.workspace, exist_ok = True)

    opt.gt_image_dir = os.path.join(opt.workspace, 'gt_images')
    opt.pred_image_dir = os.path.join(opt.workspace, 'pred_images')

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )

    device = accelerator.device

    # model
    model = TexGaussian(opt, device)

    if accelerator.is_main_process:

        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f'Total Params:{trainable_num / 1e6}M')


    # resume
    if opt.resume is not None:
        print(opt.resume)
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')

        # tolerant load (only load matching shapes)
        accelerator.print('Start loading checkpoint')
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict:
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')

    train_dataset = Dataset(opt, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.batch_size,
        collate_fn=collate_func,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = Dataset(opt, training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_func,
        pin_memory=True,
        drop_last=False,
    )

    # optimizer
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad == True], lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))

    total_steps = opt.num_epochs * len(train_dataloader)
    pct_start = 3000 / total_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start)

    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(   # prepare函数会自动把模型、优化器、训练集和测试集都加载到gpu上，不用手动移动了。
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir = opt.workspace)

        file_name = os.path.join(opt.workspace, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(opt.__dict__.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        os.system(f'cp main.py {opt.workspace}')
        os.system(f'cp -r core {opt.workspace}')

    iter_start_time = time.time()

    max_eval_psnr = 0

    # loop
    for epoch in range(opt.num_epochs):
        # train
        model.train()
        total_loss = 0
        total_psnr = 0
        if opt.use_material:
            mr_total_psnr = 0
        for i, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                optimizer.zero_grad()

                out = model(data)
                loss = out['loss']
                psnr = out['psnr']
                gaussian_loss = out['gaussian_loss']
                albedo_loss = out['albedo_loss']
                mask_loss = out['mask_loss']
                lpips_loss = out['lpips_loss']

                if opt.use_material:
                    mr_loss = out['mr_loss']
                    mr_lpips_loss = out['mr_lpips_loss']
                    mr_psnr = out['mr_psnr']

                accelerator.backward(loss)

                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                optimizer.step()
                scheduler.step()
                model.module.update_EMA()

                total_loss += loss.detach()
                total_psnr += psnr.detach()

                if opt.use_material:
                    mr_total_psnr += mr_psnr.detach()

                torch.cuda.empty_cache()

            if accelerator.is_main_process:
                # logging
                if i % 10 == 0:
                    mem_free, mem_total = torch.cuda.mem_get_info()
                    t = (time.time() - iter_start_time)

#                     print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G \
# lr: {optimizer.state_dict()['param_groups'][0]['lr']:.7f} \
# loss: {loss.item():.6f} time: {t:.3f}")

                    if opt.use_material:
                        print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G \
lr: {optimizer.state_dict()['param_groups'][0]['lr']:.7f} loss: {loss.item():.6f} \
gaussian_loss: {gaussian_loss:.6f} albedo_loss: {albedo_loss:.6f} mr_loss: {mr_loss:.6f} \
mask_loss: {mask_loss:.6f} lpips_loss: {lpips_loss:.6f} mr_lpips_loss: {mr_lpips_loss:.6f} time: {t:.3f}")

                    else:
                        print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G \
lr: {optimizer.state_dict()['param_groups'][0]['lr']:.7f} \
loss: {loss.item():.6f} gaussian_loss: {gaussian_loss:.6f} albedo_loss: {albedo_loss:.6f} \
mask_loss: {mask_loss:.6f} lpips_loss: {lpips_loss:.6f} time: {t:.3f}")

                    # writer.add_scalar('loss', loss, epoch * len(train_dataloader) + i)

                    writer.add_scalar('loss', out['loss'], epoch * len(train_dataloader) + i)
                    writer.add_scalar('gaussian_loss', out['gaussian_loss'], epoch * len(train_dataloader) + i)
                    writer.add_scalar('albedo_loss', out['albedo_loss'], epoch * len(train_dataloader) + i)
                    writer.add_scalar('mask_loss', out['mask_loss'], epoch * len(train_dataloader) + i)
                    writer.add_scalar('lpips_loss', out['lpips_loss'], epoch * len(train_dataloader) + i)
                    writer.add_scalar('psnr', out['psnr'], epoch * len(train_dataloader) + i)

                    if opt.use_material:
                        writer.add_scalar('mr_loss', out['mr_loss'], epoch * len(train_dataloader) + i)
                        writer.add_scalar('mr_lpips_loss', out['mr_lpips_loss'], epoch * len(train_dataloader) + i)
                        writer.add_scalar('mr_psnr', out['mr_psnr'], epoch * len(train_dataloader) + i)        

                # save log images
                if i % opt.image_interval == 0:
                    # pred_images = model.module.sample_and_render(data)

                    uid = data['uid']

                    pred_images = out['images_pred'] # [B, 3, output_size, output_size]

                    pred_images = pred_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    if opt.batch_size == 1:
                        kiui.write_image(f'{opt.pred_image_dir}/{epoch}_{uid[0]}.jpg', pred_images)
                    else:
                        kiui.write_image(f'{opt.pred_image_dir}/{epoch}_{i}.jpg', pred_images)

                    if opt.use_material:
                        mr_pred_images = out['mr_images_pred']
                        mr_pred_images = mr_pred_images.detach().cpu().numpy()
                        mr_pred_images = mr_pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, mr_pred_images.shape[1] * mr_pred_images.shape[3], 3)

                        if opt.batch_size == 1:
                            kiui.write_image(f'{opt.pred_image_dir}/{epoch}_mr_{uid[0]}.jpg', mr_pred_images)
                        else:
                            kiui.write_image(f'{opt.pred_image_dir}/{epoch}_mr_{i}.jpg', mr_pred_images)

                    gt_images = data['images_output']

                    gt_images = gt_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3)
                    if opt.batch_size == 1:
                        kiui.write_image(f'{opt.gt_image_dir}/{epoch}_{uid[0]}.jpg', gt_images)
                    else:
                        kiui.write_image(f'{opt.gt_image_dir}/{epoch}_{i}.jpg', gt_images)

                    if opt.use_material:
                        mr_gt_images = data['mr_images_output']

                        mr_gt_images = mr_gt_images.detach().cpu().numpy()
                        mr_gt_images = mr_gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, mr_gt_images.shape[1] * mr_gt_images.shape[3], 3)

                        if opt.batch_size == 1:
                            kiui.write_image(f'{opt.gt_image_dir}/{epoch}_mr_{uid[0]}.jpg', mr_gt_images)
                        else:
                            kiui.write_image(f'{opt.gt_image_dir}/{epoch}_mr_{i}.jpg', mr_gt_images)

        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if opt.use_material:
            mr_total_psnr = accelerator.gather_for_metrics(mr_total_psnr).mean()
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            if opt.use_material:
                mr_total_psnr /= len(train_dataloader)
                accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f} mr_psnr: {mr_total_psnr.item():.4f}")
            else:
                accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")

        # checkpoint
        if epoch % opt.ckpt_interval == 0:
            accelerator.wait_for_everyone()
            accelerator.save_model(model, opt.workspace)

        total_psnr = 0
        if opt.use_material:
            mr_total_psnr = 0

        model.eval()

        # eval
        with torch.no_grad():

            for i, data in enumerate(test_dataloader):

                out = model(data, ema = True)

                psnr = out['psnr']
                total_psnr += psnr.detach()

                if opt.use_material:
                    mr_psnr = out['mr_psnr']
                    mr_total_psnr += mr_psnr.detach()

                # save some images
                if accelerator.is_main_process:
                    uid = data['uid']

                    gt_images = data['images_output']

                    gt_images = gt_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/eval_gt_images/{epoch}_{uid[0]}.jpg', gt_images)

                    pred_images = out['images_pred']
                    pred_images = pred_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/eval_pred_images/{epoch}_{uid[0]}.jpg', pred_images)

                    if opt.use_material:
                        mr_gt_images = data['mr_images_output']
                        mr_gt_images = mr_gt_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        mr_gt_images = mr_gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, mr_gt_images.shape[1] * mr_gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                        kiui.write_image(f'{opt.workspace}/eval_gt_images/{epoch}_mr_{uid[0]}.jpg', mr_gt_images)

                        mr_pred_images = out['mr_images_pred']
                        mr_pred_images = mr_pred_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        mr_pred_images = mr_pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, mr_pred_images.shape[1] * mr_pred_images.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/eval_pred_images/{epoch}_mr_{uid[0]}.jpg', mr_pred_images)

                torch.cuda.empty_cache()

        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if opt.use_material:
            mr_total_psnr = accelerator.gather_for_metrics(mr_total_psnr).mean()
        if accelerator.is_main_process:
            total_psnr /= len(test_dataloader)
            if opt.use_material:
                mr_total_psnr /= len(test_dataloader)
                accelerator.print(f"[eval] epoch: {epoch} psnr: {total_psnr:.4f} mr_psnr: {mr_total_psnr:.4f}")
            else:
                accelerator.print(f"[eval] epoch: {epoch} psnr: {total_psnr:.4f}")

        if total_psnr > max_eval_psnr:
            max_eval_psnr = total_psnr
            accelerator.wait_for_everyone()
            save_path = f'{opt.workspace}/best_ckpt'
            accelerator.save_model(model, save_path)
            

    if accelerator.is_main_process:
        writer.close()

if __name__ == "__main__":
    main()
