#!/usr/bin/env bash
# Exp H: paper recipe + keep thinking traces inside demos.
#   The paper's reference YAML removes thinking from demos by default,
#   but our BigMath setting is all about thinking-style reasoning; removing
#   it produces trivial demos like "<think></think><answer>42</answer>"
#   which we saw drives collapse in round 1.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REMOVE_THINKING_FROM_DEMO=0
source "$DIR/_common.sh"

export DEVICE="${DEVICE:-cuda:7}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/sdpo_r3_h_keep_thinking}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-sdpo_r3_h_keep_thinking}"
bash "$COMMON_REPO_ROOT/train_scripts/sdpo/run_sdpo.sh"
