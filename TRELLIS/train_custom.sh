#!/bin/bash

# 使用自定义dataset的训练脚本
SPCONV_ALGO=native CUDA_VISIBLE_DEVICES=3 python train.py \
  --config configs/generation/custom_slat_flow_img_dit_L_64l8p2_fp16.json \
  --output_dir /home/xinyue_liang/lxy/aaa_Trellis/TRELLIS/outputs/new_lpips_concat_kv_decoder_win \
  --data_dir /home/xinyue_liang/lxy/dreamposible/1w/data/human_test \
  --num_nodes 1 \
  --node_rank 0 \
  --master_addr localhost \
  --master_port 12345
