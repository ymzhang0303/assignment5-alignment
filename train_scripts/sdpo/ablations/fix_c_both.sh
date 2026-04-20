#!/usr/bin/env bash
# Ablation C: combine A + B.
# Hypothesis: the two fixes are complementary -- A keeps the demo's actual
# reasoning visible to the teacher (gives it a long-form *example*), B adds
# an explicit instruction in the user message (long-form *prior*). Both
# should pull the teacher's response distribution toward thinking + answer.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

export REMOVE_THINKING_FROM_DEMO=0
export REPROMPT_TEMPLATE=$'{prompt}{solution}{feedback}\n\nNow correctly solve the original question. First reason carefully step by step inside <think>...</think>, then put your final answer inside <answer>...</answer>.'
unset SOLUTION_TEMPLATE

TAG="${TAG:-fix_c_both}"
export OUTPUT_DIR="runs/sdpo_ablation_${TAG}"
export WANDB_RUN_NAME="sdpo_ablation_${TAG}"
export DEVICE="${DEVICE:-cuda:7}"

exec bash "$SCRIPT_DIR/../run_sdpo.sh" "$@"
