#!/usr/bin/env bash
# Exp B: stronger teacher regularization (EMA decay=0.01, paper Table 4).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TEACHER_UPDATE_RATE=0.01
source "$DIR/_common.sh"

export DEVICE="${DEVICE:-cuda:1}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/sdpo_r3_b_ema_strong}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-sdpo_r3_b_ema_strong}"
bash "$COMMON_REPO_ROOT/train_scripts/sdpo/run_sdpo.sh"
