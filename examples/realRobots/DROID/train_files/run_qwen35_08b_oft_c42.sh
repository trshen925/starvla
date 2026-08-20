#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

CONFIG="${CONFIG:-examples/realRobots/DROID/train_files/starvla_qwen35_08b_oft_c42.yaml}"
BASE_VLM="${BASE_VLM:-Qwen/Qwen3.5-0.8B}"

accelerate launch --num_processes "${NUM_GPUS:-1}" \
  starVLA/training/train_starvla.py --config_yaml "$CONFIG" \
  --framework.qwenvl.base_vlm "$BASE_VLM" "$@"
