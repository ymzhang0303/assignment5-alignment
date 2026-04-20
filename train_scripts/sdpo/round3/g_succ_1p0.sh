#!/usr/bin/env bash
# Exp G: keep the strict success threshold (1.0) but otherwise use paper
# recipe. Tests whether our BigMath reward function makes partial-success
# demos (0.5 <= r < 1.0, i.e. format correct only) toxic.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SUCCESS_THRESHOLD=1.0
source "$DIR/_common.sh"

export DEVICE="${DEVICE:-cuda:6}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/sdpo_r3_g_succ_1p0}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-sdpo_r3_g_succ_1p0}"
bash "$COMMON_REPO_ROOT/train_scripts/sdpo/run_sdpo.sh"
