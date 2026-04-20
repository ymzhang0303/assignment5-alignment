#!/usr/bin/env bash
# LR sweep wrapper: Qwen3-1.7B GRPO on Big-Math with LR=3e-5 on cuda:2.
set -euo pipefail
export LR="${LR:-3e-5}"
export DEVICE="${DEVICE:-cuda:2}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/grpo_qwen3_bigmath_lr3e-5}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-grpo_qwen3_bigmath_lr3e-5}"
source "$(dirname "$0")/_base.sh" "$@"
