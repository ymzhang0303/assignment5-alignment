#!/usr/bin/env bash
# Exp D: isolate the EMA-teacher fix.
#   Keep old LR=1e-5 but add EMA teacher (decay=0.05).
#   Tests whether teacher regularization alone is enough.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LR=1e-5  # back to original LR
source "$DIR/_common.sh"

export DEVICE="${DEVICE:-cuda:3}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/sdpo_r3_d_ema_only}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-sdpo_r3_d_ema_only}"
bash "$COMMON_REPO_ROOT/train_scripts/sdpo/run_sdpo.sh"
