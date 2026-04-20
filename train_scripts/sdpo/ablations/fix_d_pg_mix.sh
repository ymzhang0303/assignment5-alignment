#!/usr/bin/env bash
# Ablation D: best previous fix (B = force-reasoning instruction in the
# reprompt) PLUS a GRPO-style PG term on samples without a teacher signal.
#
# Hypothesis: the demo-only fixes slowed the collapse but did not stop it
# because there is no signal anywhere in the loss that pulls toward
# *correct* answers -- the KL only matches the teacher, and once the
# teacher (= current policy) drifts, the policy follows. Adding a real
# reward signal on the ~50% of rollouts that lack a successful sibling
# should anchor the policy to correctness.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

# Inherit fix_b's reprompt change.
export REMOVE_THINKING_FROM_DEMO=1
export REPROMPT_TEMPLATE=$'{prompt}{solution}{feedback}\n\nNow correctly solve the original question. First reason carefully step by step inside <think>...</think>, then put your final answer inside <answer>...</answer>.'

# The actual fix.
export PG_LOSS_TYPE=grpo_clip
export PG_LOSS_WEIGHT=1.0
export PG_APPLY_TO_ALL_SAMPLES=0  # only non-distilled samples

TAG="${TAG:-fix_d_pg_mix}"
export OUTPUT_DIR="runs/sdpo_ablation_${TAG}"
export WANDB_RUN_NAME="sdpo_ablation_${TAG}"
export DEVICE="${DEVICE:-cuda:2}"

exec bash "$SCRIPT_DIR/../run_sdpo.sh" "$@"
