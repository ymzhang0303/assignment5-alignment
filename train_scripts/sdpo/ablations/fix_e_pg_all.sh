#!/usr/bin/env bash
# Ablation E: same as D but apply PG to *every* rollout, not just the
# non-distilled ones. Effectively "GRPO + KL distillation regulariser".
#
# Hypothesis: the strongest anti-collapse signal is having a real reward
# gradient on as many tokens as possible. Distilled samples already get a
# KL gradient toward the teacher; adding the PG term on top should help
# more than it hurts when the teacher is drifting in a wrong direction
# (which is exactly when SDPO collapses).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

export REMOVE_THINKING_FROM_DEMO=1
export REPROMPT_TEMPLATE=$'{prompt}{solution}{feedback}\n\nNow correctly solve the original question. First reason carefully step by step inside <think>...</think>, then put your final answer inside <answer>...</answer>.'

export PG_LOSS_TYPE=grpo_clip
export PG_LOSS_WEIGHT=1.0
export PG_APPLY_TO_ALL_SAMPLES=1  # the difference vs fix_d

TAG="${TAG:-fix_e_pg_all}"
export OUTPUT_DIR="runs/sdpo_ablation_${TAG}"
export WANDB_RUN_NAME="sdpo_ablation_${TAG}"
export DEVICE="${DEVICE:-cuda:3}"

exec bash "$SCRIPT_DIR/../run_sdpo.sh" "$@"
