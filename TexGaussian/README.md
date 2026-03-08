# TexGaussian (Photo3D Training Version)

This repository is a Photo3D-oriented training/inference variant based on TexGaussian, cleaned for open-source release.

- Core code is preserved (`core/`, training scripts, inference scripts).
- Key scripts are preserved: `train_from_grid.py` and `run_batch_texture_cvpt_ours.sh`.
- Intermediate artifacts (debug files, logs, temporary visualization outputs) are removed.
- Checkpoints are intentionally kept outside cleanup.

---

## 1. Environment

Please follow the official TexGaussian setup instructions:
<https://github.com/ymxbj/TexGaussian>

---

## 2. Checkpoint Download

You can directly use the uploaded checkpoint:

- Repository: <https://huggingface.co/LaPetitRose/Photo3D_models>
- File path: `TexGaussian/model.safetensors`
- File page: <https://huggingface.co/LaPetitRose/Photo3D_models/blob/main/TexGaussian/model.safetensors>

Create a local checkpoint directory:

```bash
mkdir -p ckpt
```

Option 1 (recommended, `huggingface-cli`):

```bash
huggingface-cli download LaPetitRose/Photo3D_models \
  TexGaussian/model.safetensors \
  --repo-type model \
  --local-dir .
mv TexGaussian/model.safetensors ckpt/model.safetensors
rm -rf TexGaussian
```

Option 2 (direct URL with `wget`):

```bash
wget -O ckpt/model.safetensors \
  "https://huggingface.co/LaPetitRose/Photo3D_models/resolve/main/TexGaussian/model.safetensors"
```

Use it in commands as:

- Training: `--resume ./ckpt/model.safetensors`
- Inference: `--ckpt_path ./ckpt/model.safetensors`

---

## 3. Data Layout (Grid-Supervised Training)

### 3.0 Download Photo3D-MV and place it under Photo3D

Training data should be downloaded from:
<https://huggingface.co/datasets/LaPetitRose/Photo3D-MV>

Example commands:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login

huggingface-cli download "LaPetitRose/Photo3D-MV" \
  --repo-type dataset \
  --local-dir ./photo3d_mv_release \
  --local-dir-use-symlinks False
```

Extract:

```bash
mkdir -p ./photo3d_mv_unpacked
for f in ./photo3d_mv_release/photo3d_mv_part_*.tar; do
  tar -xf "$f" -C ./photo3d_mv_unpacked
done
```

Copy (or move) the unpacked data into your `Photo3D` project folder:

```bash
mkdir -p ../Photo3D/data
cp -r ./photo3d_mv_unpacked ../Photo3D/data/
```

Then upload to your GitHub repo if needed:

```bash
cd ../Photo3D
git add .
git commit -m "Add Photo3D-MV dataset and update docs"
git push
```

### 3.1 Data quality note (important)

Some samples in Photo3D-MV may have image size/orientation mismatches with the 3D model.  
This mainly comes from limitations of the 2D generator used during data creation.

As 2D generators continue improving, newer models generally preserve structure better.  
We recommend re-optimizing `rgb_grid.png` yourself (for example: "keep geometry unchanged, make texture more realistic"), which can improve final training quality.

### 3.2 Training file layout

`train_from_grid.py` expects the following structure:

```text
<grid_image_dir>/
  grid_0.png
  grid_1.png
  ...

<mesh_root>/
  0/mesh.obj
  1/mesh.obj
  ...

<text_file>
  line 0 -> id=0
  line 1 -> id=1
  ...
```

Notes:

- `grid_{id}.png` is a 2x2 multi-view grid image (automatically split into 4 views in `core/dataset_grid.py`).
- `mesh_root/{id}/mesh.obj` is the corresponding mesh.
- The `id`-th line in `text_file` is the prompt for that sample.

---

## 4. Training

### 4.1 Single-GPU Example

```bash
CUDA_VISIBLE_DEVICES=0 python train_from_grid.py objaverse \
  --workspace ./workspace_grid_photo3d \
  --grid_image_dir /path/to/grid_image_dir \
  --mesh_root /path/to/mesh_root \
  --text_file /path/to/text_optimized_prompts_10000.txt \
  --resume ./ckpt/model.safetensors \
  --batch_size 4 \
  --num_epochs 200 \
  --image_interval 200 \
  --mixed_precision bf16
```

### 4.2 Multi-GPU Example (`accelerate`)

```bash
accelerate launch --num_processes 4 train_from_grid.py objaverse \
  --workspace ./workspace_grid_photo3d \
  --grid_image_dir /path/to/grid_image_dir \
  --mesh_root /path/to/mesh_root \
  --text_file /path/to/text_optimized_prompts_10000.txt \
  --resume ./ckpt/model.safetensors \
  --batch_size 4 \
  --num_epochs 200
```

Training outputs:

- Latest checkpoint: `<workspace>/latest_ckpt/`
- Best checkpoint: `<workspace>/best_ckpt/`
- Visualization/eval images: `<workspace>/pred_images*`, `<workspace>/eval_pred_images*`

---

## 5. Inference

### 5.1 Batch Inference (Recommended)

Run:

```bash
bash run_batch_texture_cvpt_ours.sh
```

Before running, edit the variables at the top of `run_batch_texture_cvpt_ours.sh`:

- `MESH_DIR`: input mesh directory (script scans `*_final.glb`)
- `CAPTIONS_JSON`: JSON mapping `id -> caption`
- `CKPT_PATH`: checkpoint path (for example `./ckpt/model.safetensors`)
- `OUTPUT_DIR`: output directory
- `GPU_IDS`: GPU ids to use

`CAPTIONS_JSON` example:

```json
{
  "0": "a rusty robot",
  "1": "a wooden chair"
}
```

### 5.2 Single-Mesh Inference

```bash
CUDA_VISIBLE_DEVICES=0 python texture_cvpt.py objaverse \
  --mesh_path /path/to/xxxx_final.glb \
  --ckpt_path ./ckpt/model.safetensors \
  --text_prompt "a worn metal robot" \
  --output_dir /path/to/output \
  --texture_name sample \
  --save_image True
```

Typical outputs:

- `/path/to/output/sample/albedo_mesh.obj`
- `/path/to/output/sample/mr_mesh.obj`
- (optional) `/path/to/output/*_prid.png`

---

## 6. Open-Source Packaging Tips

- Do not include intermediate artifacts (debug files, logs, temporary visualizations).
- Keep checkpoints under `ckpt/` and document download instructions.
- Keep only code and configs necessary for reproducibility.

