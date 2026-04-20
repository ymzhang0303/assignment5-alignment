#!/usr/bin/env bash
# Loss-type ablation wrapper: Qwen3-1.7B GRPO on Big-Math, LR=1e-5,
# loss_type=reinforce_with_baseline, with use_std_normalization=False.
# Runs on cuda:7 by default.
#
# Holding everything else equal to run_reinforce_baseline.sh, this isolates
# the effect of dropping the per-group std denominator from the advantage:
#     A = r - mean(r)        (this run)
# vs  A = (r - mean(r)) / (std(r) + eps)  (default reinforce_with_baseline).
# Removing the std normalization keeps the magnitude of advantages tied to
# the actual reward variance instead of always being O(1) -- groups with
# unanimous outcomes contribute proportionally smaller updates, which is
# the Dr-GRPO recipe.
set -euo pipefail
export LR="${LR:-1e-5}"
export DEVICE="${DEVICE:-cuda:7}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/grpo_qwen3_bigmath_reinforce_bl_no_std}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-grpo_qwen3_bigmath_reinforce_bl_no_std}"
export LOSS_TYPE="${LOSS_TYPE:-reinforce_with_baseline}"
export USE_STD_NORMALIZATION="${USE_STD_NORMALIZATION:-0}"
source "$(dirname "$0")/../lr_sweep/_base.sh" "$@"
