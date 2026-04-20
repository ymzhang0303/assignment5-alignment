#!/usr/bin/env bash
# Ablation F: best previous fix (B = force-reasoning instruction) PLUS
# reject any successful sibling whose <think>...</think> content has
# fewer than MIN_DEMO_THINKING_CHARS non-whitespace chars (default 64).
#
# Hypothesis: the seed of the SDPO collapse was demos like
# '<think></think><answer>4</answer>' -- a rollout that skipped thinking
# but happened to guess the right answer. Filtering those out at the
# *demo selection* step should keep teacher prompts conditioned on real
# chains of thought even at later training steps when the policy starts
# producing more no-think rollouts.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

export REMOVE_THINKING_FROM_DEMO=1
export REPROMPT_TEMPLATE=$'{prompt}{solution}{feedback}\n\nNow correctly solve the original question. First reason carefully step by step inside <think>...</think>, then put your final answer inside <answer>...</answer>.'

# The actual fix.
export MIN_DEMO_THINKING_CHARS="${MIN_DEMO_THINKING_CHARS:-64}"

# Pure SDPO (no PG mix) so this isolates the demo-quality lever.
export PG_LOSS_WEIGHT=0.0
export PG_APPLY_TO_ALL_SAMPLES=0

TAG="${TAG:-fix_f_min_demo}"
export OUTPUT_DIR="runs/sdpo_ablation_${TAG}"
export WANDB_RUN_NAME="sdpo_ablation_${TAG}"
export DEVICE="${DEVICE:-cuda:7}"

exec bash "$SCRIPT_DIR/../run_sdpo.sh" "$@"
