#!/usr/bin/env bash
# Ablation A: keep <think>...</think> in demonstrations.
# Hypothesis: stripping the chain-of-thought before showing the demo to the
# teacher made every demo look like '<answer>X</answer>', which collapsed the
# teacher's distribution to "skip thinking, emit answer". Putting the demo
# reasoning back in should give the teacher a long-form distribution to
# distill into the student.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

export REMOVE_THINKING_FROM_DEMO=0
# Use the SDPO-default reprompt template (don't override).
unset REPROMPT_TEMPLATE SOLUTION_TEMPLATE

TAG="${TAG:-fix_a_keep_thinking}"
export OUTPUT_DIR="runs/sdpo_ablation_${TAG}"
export WANDB_RUN_NAME="sdpo_ablation_${TAG}"
export DEVICE="${DEVICE:-cuda:2}"

exec bash "$SCRIPT_DIR/../run_sdpo.sh" "$@"
