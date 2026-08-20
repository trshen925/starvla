#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT"
export HF_HOME="${HF_HOME:-/mnt/pfs/share/pretrained_model/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_CREDENTIAL="${WANDB_CREDENTIAL:-/mnt/pfs/users/shentingrui/.credentials/wandb_api_trshen.txt}"
if [[ -z "${WANDB_API_KEY:-}" && -s "$WANDB_CREDENTIAL" ]]; then
  export WANDB_API_KEY="$(<"$WANDB_CREDENTIAL")"
fi
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TOKENIZERS_PARALLELISM=false

CONFIG="${CONFIG:-examples/realRobots/DROID/train_files/starvla_qwen35_08b_oft_c42_full.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/gen2act/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/root/miniconda3/envs/gen2act/bin/accelerate}"
NUM_GPUS="${NUM_GPUS:-4}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
"$ACCELERATE_BIN" launch --num_processes "$NUM_GPUS" \
  starVLA/training/train_starvla.py --config_yaml "$CONFIG" "$@"
