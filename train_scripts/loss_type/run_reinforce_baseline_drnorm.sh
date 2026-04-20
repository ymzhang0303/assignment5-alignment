#!/usr/bin/env bash
# Loss-type ablation wrapper: Qwen3-1.7B GRPO on Big-Math, LR=1e-5,
# loss_type=reinforce_with_baseline + length_normalization=masked_normalize
# (Dr-GRPO style "sum / L_max"). Runs on cuda:6 by default.
#
# Holding everything else equal, this isolates the effect of switching the
# sequence-dim reducer from masked_mean (per-token average, biased toward
# shorter rollouts) to masked_normalize (constant per-token weight).
# normalize_constant defaults to sampling_max_tokens (1536) if left unset.
set -euo pipefail
export LR="${LR:-1e-5}"
export DEVICE="${DEVICE:-cuda:6}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/grpo_qwen3_bigmath_reinforce_bl_drnorm}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-grpo_qwen3_bigmath_reinforce_bl_drnorm}"
export LOSS_TYPE="${LOSS_TYPE:-reinforce_with_baseline}"
export LENGTH_NORMALIZATION="${LENGTH_NORMALIZATION:-masked_normalize}"
export NORMALIZE_CONSTANT="${NORMALIZE_CONSTANT:-1536}"
source "$(dirname "$0")/../lr_sweep/_base.sh" "$@"
