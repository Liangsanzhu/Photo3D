cd /home/xinyue_liang/lxy/Photo3D/TRELLIS

SPCONV_ALGO=native CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/generation/custom_slat_flow_img_dit_L_64l8p2_fp16.json \
  --output_dir outputs/custom_train_run \
  --ckpt none \
  --data_dir /home/xinyue_liang/lxy/dreamposible/1w/data/3_trellis_gen_gpt \
  --num_nodes 1 \
  --node_rank 0 \
  --master_addr 127.0.0.1 \
  --master_port 29500