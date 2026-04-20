#!/usr/bin/env bash
# Ablation B: keep stripping demos (so teacher only sees the final answer)
# but EXPLICITLY tell the teacher to think before answering.
# Hypothesis: even with a short, answer-only demo, an explicit instruction
# like "think step by step inside <think>...</think> before answering"
# should keep the teacher's response distribution long-form (because the
# instruction lives in the user message, before the assistant turn). The
# teacher then conditions its per-token distribution on that instruction
# and the distillation target stops collapsing to "no-think + guess".
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

export REMOVE_THINKING_FROM_DEMO=1  # baseline behaviour for this knob
# New reprompt template: explicitly instruct a thinking trace + answer.
# Must contain '{prompt}', '{solution}', '{feedback}' placeholders.
export REPROMPT_TEMPLATE=$'{prompt}{solution}{feedback}\n\nNow correctly solve the original question. First reason carefully step by step inside <think>...</think>, then put your final answer inside <answer>...</answer>.'
unset SOLUTION_TEMPLATE

TAG="${TAG:-fix_b_force_reason}"
export OUTPUT_DIR="runs/sdpo_ablation_${TAG}"
export WANDB_RUN_NAME="sdpo_ablation_${TAG}"
export DEVICE="${DEVICE:-cuda:3}"

exec bash "$SCRIPT_DIR/../run_sdpo.sh" "$@"
