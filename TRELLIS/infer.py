import argparse
import json
import os
from pathlib import Path

os.environ["SPCONV_ALGO"] = "native"

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file as safetensors_load

from trellis import models as trellis_models
from trellis.models.structured_latent_vae.decoder_gs_dino_cond import SLatGaussianDecoderDinoCond
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils


_DINOV3_MODEL_CACHE = None


def parse_args():
    parser = argparse.ArgumentParser(description="Single-image TRELLIS inference (open-source clean version)")
    parser.add_argument("--input", type=str, required=True, help="Path to a single input image")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--pretrained", type=str, default="microsoft/TRELLIS-image-large", help="Pretrained TRELLIS model id/path")
    parser.add_argument("--slat-flow-ckpt", type=str, default="", help="Optional custom stage-2 ckpt (.pt/.pth/.safetensors)")
    parser.add_argument("--slat-flow-json", type=str, default="", help="Optional custom stage-2 model json (same model architecture)")
    parser.add_argument("--decoder-ckpt", type=str, default="", help="Optional safetensors path for finetuned decoder")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--ss-cfg", type=float, default=7.5)
    parser.add_argument("--slat-steps", type=int, default=12)
    parser.add_argument("--slat-cfg", type=float, default=3.0)
    parser.add_argument("--camera-radius", type=float, default=1.7)
    parser.add_argument("--grid-size", type=int, default=1024)
    parser.add_argument("--save-ply", action="store_true", help="Save gaussian as .ply")
    parser.add_argument("--use-dino-cond", action="store_true", help="Enable DINOv3 feature injection to decoder")
    return parser.parse_args()


def center_crop_to_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def render_around_view_to_grid(gaussian, radius=1.7, output_size=512):
    video_gaussian, _ = render_utils.render_around_view(gaussian, r=radius)
    video_gaussian = torch.stack([frame for frame in video_gaussian])

    total_frames = video_gaussian.shape[0]
    selected_indices = [0, total_frames // 4, total_frames // 2, 3 * total_frames // 4]

    h, w = video_gaussian.shape[-2:]
    grid_tensor = torch.zeros(3, h * 2, w * 2, device=video_gaussian.device, dtype=video_gaussian.dtype)
    for j, idx in enumerate(selected_indices):
        row = j // 2
        col = j % 2
        grid_tensor[:, row * h:(row + 1) * h, col * w:(col + 1) * w] = video_gaussian[idx]

    grid_tensor = torch.clamp(grid_tensor, 0, 1)
    if grid_tensor.shape[-1] != output_size or grid_tensor.shape[-2] != output_size:
        grid_tensor = F.interpolate(
            grid_tensor.unsqueeze(0),
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return grid_tensor


def get_dinov3_model():
    global _DINOV3_MODEL_CACHE
    if _DINOV3_MODEL_CACHE is None:
        model = torch.hub.load(repo_or_dir="facebookresearch/dinov3", model="dinov3_vitl16", source="github")
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        for p in model.parameters():
            p.requires_grad_(False)
        _DINOV3_MODEL_CACHE = model
    return _DINOV3_MODEL_CACHE


def extract_dinov3_features(img_chw: torch.Tensor, model) -> torch.Tensor:
    bchw = img_chw.unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=bchw.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=bchw.device).view(1, 3, 1, 1)
    x = (bchw - mean) / std
    with torch.no_grad():
        feats_list = model.get_intermediate_layers(x, n=1, reshape=True, norm=True)
    feats = feats_list[-1].squeeze(0)
    feats = F.normalize(feats, p=2, dim=0)
    return feats


def maybe_replace_decoder_with_ckpt(pipeline: TrellisImageTo3DPipeline, decoder_ckpt: str):
    if not decoder_ckpt:
        return
    if not os.path.isfile(decoder_ckpt):
        raise FileNotFoundError(f"Decoder checkpoint not found: {decoder_ckpt}")

    original_decoder = pipeline.models.get("slat_decoder_gs")
    if original_decoder is None:
        raise RuntimeError("slat_decoder_gs not found in pipeline")

    ref_args = original_decoder.__dict__
    new_decoder = SLatGaussianDecoderDinoCond(
        resolution=ref_args.get("resolution", 64),
        model_channels=ref_args.get("model_channels", 768),
        latent_channels=ref_args.get("in_channels", 8),
        num_blocks=ref_args.get("num_blocks", 12),
        num_heads=ref_args.get("num_heads", 12),
        num_head_channels=ref_args.get("model_channels", 768) // ref_args.get("num_heads", 12),
        mlp_ratio=ref_args.get("mlp_ratio", 4.0),
        attn_mode=ref_args.get("attn_mode", "swin"),
        window_size=ref_args.get("window_size", 8),
        pe_mode=ref_args.get("pe_mode", "ape"),
        use_fp16=ref_args.get("use_fp16", False),
        use_checkpoint=ref_args.get("use_checkpoint", False),
        qk_rms_norm=ref_args.get("qk_rms_norm", False),
        representation_config=ref_args.get("rep_config", getattr(original_decoder, "rep_config", None)),
        enable_dino_cond=True,
    ).to(pipeline.device)

    new_decoder.load_state_dict(original_decoder.state_dict(), strict=False)
    new_decoder.load_state_dict(safetensors_load(decoder_ckpt), strict=False)
    pipeline.models["slat_decoder_gs"] = new_decoder.eval()


def _extract_state_dict_from_ckpt(ckpt_obj):
    if not isinstance(ckpt_obj, dict):
        raise RuntimeError("Unsupported .pt/.pth checkpoint format: expected dict-like state_dict.")
    if all(torch.is_tensor(v) for v in ckpt_obj.values()):
        return ckpt_obj
    for key in ("state_dict", "model", "denoiser"):
        v = ckpt_obj.get(key)
        if isinstance(v, dict) and all(torch.is_tensor(t) for t in v.values()):
            return v
    raise RuntimeError("Cannot find a valid state_dict in checkpoint. Tried keys: state_dict/model/denoiser.")


def maybe_replace_slat_flow_model(pipeline: TrellisImageTo3DPipeline, slat_flow_ckpt: str, slat_flow_json: str):
    if not slat_flow_ckpt and not slat_flow_json:
        return
    if not slat_flow_ckpt or not slat_flow_json:
        raise ValueError("Please provide both --slat-flow-ckpt and --slat-flow-json together.")
    if "slat_flow_model" not in pipeline.models:
        raise RuntimeError("slat_flow_model not found in pipeline")

    if not os.path.isfile(slat_flow_ckpt):
        raise FileNotFoundError(f"SLAT flow ckpt not found: {slat_flow_ckpt}")
    if not os.path.isfile(slat_flow_json):
        raise FileNotFoundError(f"SLAT flow json not found: {slat_flow_json}")

    with open(slat_flow_json, "r") as f:
        cfg = json.load(f)
    if "name" in cfg and "args" in cfg:
        model_name = cfg["name"]
        model_args = cfg["args"]
    elif "models" in cfg and "denoiser" in cfg["models"]:
        model_name = cfg["models"]["denoiser"]["name"]
        model_args = cfg["models"]["denoiser"]["args"]
    else:
        raise RuntimeError("Unsupported slat flow json format. Expect {name,args} or {models: {denoiser: ...}}")

    slat_flow_model = getattr(trellis_models, model_name)(**model_args)
    if slat_flow_ckpt.endswith(".safetensors"):
        state_dict = safetensors_load(slat_flow_ckpt)
    else:
        state_dict = _extract_state_dict_from_ckpt(torch.load(slat_flow_ckpt, map_location="cpu", weights_only=False))
    missing, unexpected = slat_flow_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[SLAT flow] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[SLAT flow] Unexpected keys: {len(unexpected)}")

    slat_flow_model = slat_flow_model.to(pipeline.device).eval()
    pipeline.models["slat_flow_model"] = slat_flow_model


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.pretrained)
    pipeline.cuda()

    maybe_replace_slat_flow_model(pipeline, args.slat_flow_ckpt, args.slat_flow_json)
    maybe_replace_decoder_with_ckpt(pipeline, args.decoder_ckpt)

    image = Image.open(str(input_path)).convert("RGB")
    image = center_crop_to_square(image)

    if args.use_dino_cond:
        dino_model = get_dinov3_model()
        img_t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        if torch.cuda.is_available():
            img_t = img_t.cuda()
        dino_feats = extract_dinov3_features(img_t, dino_model)
        pipeline.models["slat_decoder_gs"].set_dino_features(dino_feats)

    outputs = pipeline.run(
        image,
        seed=args.seed,
        sparse_structure_sampler_params={"steps": args.ss_steps, "cfg_strength": args.ss_cfg},
        slat_sampler_params={"steps": args.slat_steps, "cfg_strength": args.slat_cfg},
    )
    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]

    base = input_path.stem
    grid_image = render_around_view_to_grid(
        outputs["gaussian"][0],
        radius=args.camera_radius,
        output_size=args.grid_size,
    )
    grid_np = (grid_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    grid_out = output_dir / f"{base}_grid.png"
    Image.fromarray(grid_np).save(str(grid_out))
    print(f"Saved grid image: {grid_out}")

    if args.save_ply:
        ply_out = output_dir / f"{base}.ply"
        outputs["gaussian"][0].save_ply(str(ply_out))
        print(f"Saved ply: {ply_out}")


if __name__ == "__main__":
    main()

