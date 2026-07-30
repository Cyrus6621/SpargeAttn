#!/usr/bin/env bash
set -euo pipefail

SPARGE_REPO="/home/dangyunkai/yunkai/VLM/VIG-Group/haoyi/ICLR27/sparse_attn/SpargeAttn"
PYTHON_BIN="/data1/dangyunkai/conda_envs/sparge/bin/python"
COLLECTOR="${SPARGE_REPO}/experiment/script/q_sim/collect_q_keyblock_disagreement.py"
OUTPUT_DIR="${SPARGE_REPO}/experiment/output/q_sim/govreport_test_qwen3_8b_32k"
GPU_IDS_TEXT="${GPU_IDS:-0 1 6}"
read -r -a GPU_IDS_ARRAY <<< "${GPU_IDS_TEXT}"
if [[ -n "${SHARD_IDS:-}" ]]; then
  read -r -a SHARD_IDS_ARRAY <<< "${SHARD_IDS}"
else
  SHARD_IDS_ARRAY=()
  for local_index in "${!GPU_IDS_ARRAY[@]}"; do
    SHARD_IDS_ARRAY+=("${local_index}")
  done
fi
if [[ "${#GPU_IDS_ARRAY[@]}" -ne "${#SHARD_IDS_ARRAY[@]}" ]]; then
  echo "GPU_IDS and SHARD_IDS must contain the same number of entries" >&2
  exit 2
fi
NUM_SHARDS="${NUM_SHARDS:-${#GPU_IDS_ARRAY[@]}}"

mkdir -p "${OUTPUT_DIR}/logs"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

pids=()
for local_index in "${!GPU_IDS_ARRAY[@]}"; do
  gpu_id="${GPU_IDS_ARRAY[${local_index}]}"
  shard_id="${SHARD_IDS_ARRAY[${local_index}]}"
  log_file="${OUTPUT_DIR}/logs/collector-shard-${shard_id}-of-${NUM_SHARDS}.log"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${PYTHON_BIN}" "${COLLECTOR}" \
      --output-dir "${OUTPUT_DIR}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-id "${shard_id}" \
      "$@" >"${log_file}" 2>&1 &
  pids+=("$!")
  echo "Started shard ${shard_id}/${NUM_SHARDS} on GPU ${gpu_id}; log=${log_file}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
