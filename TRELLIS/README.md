# Photo3D TRELLIS

Setup the environment as the official TRELLIS repo: <https://github.com/microsoft/TRELLIS>

## Download Checkpoints To `ckpt/` (Copy and Run)

```bash
cd Photo3D/TRELLIS
mkdir -p ckpt

huggingface-cli download LaPetitRose/Photo3D_models \
  --repo-type model \
  --include "Trellis/*" \
  --local-dir .

cp -r Trellis/* ckpt/
rm -rf Trellis
```

After this, `ckpt/` should contain files like:

- `ckpt/config.json`
- `ckpt/slat_denoiser.pt`
- `ckpt/decoder.safetensors`

## Run Inference (Copy and Run)

```bash
cd Photo3D/TRELLIS

python infer.py \
  --input ./assets/42.png \
  --output-dir ./output \
  --pretrained microsoft/TRELLIS-image-large \
  --slat-flow-ckpt ./ckpt/slat_denoiser.pt \
  --slat-flow-json ./ckpt/config.json \
  --decoder-ckpt ./ckpt/decoder.safetensors
```

Result image:

- `./output/42_grid.png`

