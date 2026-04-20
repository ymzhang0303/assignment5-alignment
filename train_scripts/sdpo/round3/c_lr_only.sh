#!/usr/bin/env bash
# Exp C: isolate the learning-rate fix.
#   LR=1e-6 but keep teacher=current policy (no EMA regularization).
#   Tests whether a 10x lower LR alone is enough to prevent collapse.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TEACHER_UPDATE_RATE=1.0  # back to unregularized teacher
source "$DIR/_common.sh"

export DEVICE="${DEVICE:-cuda:2}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/sdpo_r3_c_lr_only}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-sdpo_r3_c_lr_only}"
bash "$COMMON_REPO_ROOT/train_scripts/sdpo/run_sdpo.sh"
