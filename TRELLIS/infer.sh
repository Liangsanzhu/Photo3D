  python infer.py \
  --input ./assets/52.png \
  --output-dir output/ \
  --pretrained microsoft/TRELLIS-image-large \
  --slat-flow-ckpt /home/xinyue_liang/lxy/Photo3D/TRELLIS/ckpt/slat_denoiser.pt\
  --slat-flow-json /home/xinyue_liang/lxy/Photo3D/TRELLIS/ckpt/config.json \
  --decoder-ckpt /home/xinyue_liang/lxy/Photo3D/TRELLIS/ckpt/decoder.safetensors