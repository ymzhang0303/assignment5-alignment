#!/usr/bin/env bash
# Exp E: SDPO + GRPO mix for "weak" base models.
#   Paper section 4.5: Qwen3-0.6B benefits from lambda=0.9 GRPO advantage +
#   0.1 SDPO advantage. We're on Qwen3-1.7B which sits in that regime.
#   Implement as: keep SDPO distill loss at weight 1 and add a sizeable PG
#   term (pg_loss_weight=5.0) applied to *all* samples.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PG_LOSS_WEIGHT=5.0
export PG_APPLY_TO_ALL_SAMPLES=1
source "$DIR/_common.sh"

export DEVICE="${DEVICE:-cuda:4}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/sdpo_r3_e_grpo_mix}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-sdpo_r3_e_grpo_mix}"
bash "$COMMON_REPO_ROOT/train_scripts/sdpo/run_sdpo.sh"
