#!/usr/bin/env bash
# Loss-type ablation wrapper: Qwen3-1.7B GRPO on Big-Math, LR=1e-5,
# loss_type=reinforce_with_baseline. Runs on cuda:4 by default.
#
# Re-runs the baseline now that drgrpo_grader._normalize folds Unicode
# math glyphs (×, ⁰-⁹, ⁻⁺, ₀-₉, ...), so 1.656×10⁶ correctly grades as
# equivalent to 1.656×10^{6}.
set -euo pipefail
export LR="${LR:-1e-5}"
export DEVICE="${DEVICE:-cuda:4}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/grpo_qwen3_bigmath_reinforce_bl}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-grpo_qwen3_bigmath_reinforce_bl}"
export LOSS_TYPE="${LOSS_TYPE:-reinforce_with_baseline}"
source "$(dirname "$0")/../lr_sweep/_base.sh" "$@"
