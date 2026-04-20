#!/usr/bin/env bash
# LR sweep wrapper: Qwen3-1.7B GRPO on Big-Math with LR=1e-4 on cuda:3.
set -euo pipefail
export LR="${LR:-1e-4}"
export DEVICE="${DEVICE:-cuda:3}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/grpo_qwen3_bigmath_lr1e-4}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-grpo_qwen3_bigmath_lr1e-4}"
source "$(dirname "$0")/_base.sh" "$@"
