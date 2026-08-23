#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export HF_HOME="${HF_HOME:-/mnt/pfs/share/pretrained_model/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false

# The workspace's validated training environment. Override this variable when
# launching from a different conda environment.
ACCELERATE_BIN="${ACCELERATE_BIN:-/root/miniconda3/envs/physctrl/bin/accelerate}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"$ACCELERATE_BIN" launch --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/realRobots/DROID/train_files/starvla_dp_oxe_raw_smoke.yaml "$@"
