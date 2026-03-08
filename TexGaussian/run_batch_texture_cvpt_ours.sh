#!/usr/bin/env bash
set -euo pipefail

# 配置（请按你的环境修改）
# 输入目录：包含 *_final.glb
MESH_DIR="./data/meshes"
# 文本描述：json 格式，key=id，value=caption
CAPTIONS_JSON="./data/captions.json"
# 推理脚本（默认仓库内）
SCRIPT="./texture_cvpt.py"
# 模型权重
CKPT_PATH="./ckpt/model.safetensors"
# 输出目录
OUTPUT_DIR="./outputs/tex_ours"
TEXTURE_NAME="myname"
DATASET_ARG="objaverse"
GPU_IDS="2"

mkdir -p "${OUTPUT_DIR}"
shopt -s nullglob

num_total=0
num_run=0

for glb in "${MESH_DIR}"/*_final.glb; do
  ((num_total++)) || true
  base="$(basename "${glb}")"
  id="${base%_final.glb}"
  

  # 读取 caption（按 id 为 key）
  caption="$(ID="${id}" CAPTIONS_JSON="${CAPTIONS_JSON}" python3 - <<'PY'
import os, json
id_ = os.environ['ID']
captions_json = os.environ['CAPTIONS_JSON']
with open(captions_json, 'r') as f:
    caps = json.load(f)
print(caps.get(id_, ''), end='')
PY
)"

  if [ -z "${caption}" ]; then
    continue
  fi

  # 检查输出文件是否已存在
  prid_file="${OUTPUT_DIR}/${id}_prid.png"
  if [ -f "${prid_file}" ]; then
    echo "[INFO] Skipping id=${id} (${prid_file} already exists)"
    continue
  fi

  # 转义双引号，确保传参安全
  caption_escaped="${caption//\"/\\\"}"

  echo "[INFO] Running id=${id} | caption=${caption}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" python3 "${SCRIPT}" "${DATASET_ARG}" \
    --texture_name "${TEXTURE_NAME}" \
    --ckpt_path "${CKPT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --save_image True \
    --mesh_path "${glb}" \
    --text_prompt "${caption_escaped}"

  ((num_run++)) || true
done

echo "[INFO] Finished. Total files: ${num_total}, Launched: ${num_run}"


