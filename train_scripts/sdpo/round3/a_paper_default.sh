#!/usr/bin/env bash
# Exp A: as close to the paper's reference YAML as we can get.
#   LR=1e-6, EMA teacher (decay=0.05), success_reward_threshold=0.5,
#   pure SDPO (no GRPO mix), no reprompt override.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/_common.sh"

export DEVICE="${DEVICE:-cuda:0}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/sdpo_r3_a_paper_default}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-sdpo_r3_a_paper_default}"
bash "$COMMON_REPO_ROOT/train_scripts/sdpo/run_sdpo.sh"
