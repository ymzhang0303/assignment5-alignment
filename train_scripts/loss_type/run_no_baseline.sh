#!/usr/bin/env bash
# Loss-type ablation wrapper: Qwen3-1.7B GRPO on Big-Math, LR=1e-5,
# loss_type=no_baseline (vanilla REINFORCE, no group-mean baseline).
# Runs on cuda:5 by default.
set -euo pipefail
export LR="${LR:-1e-5}"
export DEVICE="${DEVICE:-cuda:5}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/grpo_qwen3_bigmath_no_baseline}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-grpo_qwen3_bigmath_no_baseline}"
export LOSS_TYPE="${LOSS_TYPE:-no_baseline}"
source "$(dirname "$0")/../lr_sweep/_base.sh" "$@"
