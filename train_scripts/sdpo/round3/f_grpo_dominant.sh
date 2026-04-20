#!/usr/bin/env bash
# Exp F: GRPO-dominant with light SDPO (closest to paper's lambda=0.9 recipe
# for weak models).
#   distill_loss + 20 * pg_loss  ->  GRPO dominates, SDPO is auxiliary.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PG_LOSS_WEIGHT=20.0
export PG_APPLY_TO_ALL_SAMPLES=1
source "$DIR/_common.sh"

export DEVICE="${DEVICE:-cuda:5}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/sdpo_r3_f_grpo_dominant}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-sdpo_r3_f_grpo_dominant}"
bash "$COMMON_REPO_ROOT/train_scripts/sdpo/run_sdpo.sh"
