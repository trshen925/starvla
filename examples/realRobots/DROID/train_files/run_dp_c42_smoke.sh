#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export HF_HOME="${HF_HOME:-/mnt/pfs/share/pretrained_model/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
WANDB_CREDENTIAL="${WANDB_CREDENTIAL:-/mnt/pfs/users/shentingrui/.credentials/wandb_api_trshen.txt}"
if [[ -z "${WANDB_API_KEY:-}" && -s "$WANDB_CREDENTIAL" ]]; then
  export WANDB_API_KEY="$(<"$WANDB_CREDENTIAL")"
fi
ACCELERATE_BIN="${ACCELERATE_BIN:-/root/miniconda3/envs/gen2act/bin/accelerate}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
"$ACCELERATE_BIN" launch --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/realRobots/DROID/train_files/starvla_dp_c42_smoke.yaml "$@"
