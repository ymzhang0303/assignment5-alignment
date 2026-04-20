#!/usr/bin/env bash
# SDPO-final: the winning recipe from the collapse-fix ablations.
#
# Round 2 results (30 steps each, val/answer_reward trajectory):
#   - fix_d (PG only on no-demo samples): 0.45 -> 0.17 -> 0.28 -> 0.29 -> 0.23
#   - fix_e (PG on ALL samples, w=1.0):   0.47 -> 0.58 -> 0.45 -> 0.28 -> 0.36 -> 0.36 -> 0.30  <-- winner
#   - fix_f (min-thinking demo filter):   0.49 -> 0.53 -> 0.05 -> 0.01   (catastrophic)
#
# Recipe = fix_b's reprompt template (force thinking in the instruction) +
# fix_e (PG on every rollout, weight=1.0). Effectively: GRPO with a
# KL-to-self-teacher regulariser whose teacher sees a reprompted message
# that embeds a successful sibling demonstration and an explicit
# instruction to think before answering.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# -------- Winning-recipe knobs (override via env if needed). -------- #
# Force the teacher to reason before answering; strip thinking from demo
# (fix_c showed that combining "keep thinking in demo" + "force thinking in
# instruction" *hurt*; the two fight each other).
export REMOVE_THINKING_FROM_DEMO="${REMOVE_THINKING_FROM_DEMO:-1}"
# Avoid the ${VAR:-$'...'} pattern: bash does not ANSI-C-decode `$'...'`
# inside a parameter-expansion default, and the `}` after `{prompt}`
# inside the literal gets eaten as the outer `${...}` terminator. Use a
# plain-assign with $'...' instead (this is how the ablation scripts
# encode it) so the brace-placeholders survive intact.
if [[ -z "${REPROMPT_TEMPLATE:-}" ]]; then
    export REPROMPT_TEMPLATE=$'{prompt}{solution}{feedback}\n\nNow correctly solve the original question. First reason carefully step by step inside <think>...</think>, then put your final answer inside <answer>...</answer>.'
fi

# Structural fix: GRPO-style PG term, on EVERY rollout (not just
# non-distilled). This is what actually prevented collapse -- the KL alone
# has no anchor to correctness once the teacher (=current policy) drifts.
export PG_LOSS_TYPE="${PG_LOSS_TYPE:-grpo_clip}"
export PG_LOSS_WEIGHT="${PG_LOSS_WEIGHT:-1.0}"
export PG_APPLY_TO_ALL_SAMPLES="${PG_APPLY_TO_ALL_SAMPLES:-1}"

# No demo-quality filter: fix_f showed that MIN_DEMO_THINKING_CHARS>0
# collapses the run because it starves the teacher signal in later steps
# when successful rollouts become rarer.
export MIN_DEMO_THINKING_CHARS="${MIN_DEMO_THINKING_CHARS:-0}"

# Inherited SDPO defaults.
export SDPO_ALPHA="${SDPO_ALPHA:-0.5}"            # JSD
export DISTILLATION_TOPK="${DISTILLATION_TOPK:-100}"
export IS_CLIP="${IS_CLIP:-2.0}"
export SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-1.0}"
export TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-1.0}"
export DONT_REPROMPT_ON_SELF_SUCCESS="${DONT_REPROMPT_ON_SELF_SUCCESS:-1}"

# -------- Training shape (same as baseline GRPO/SDPO runs). -------- #
export LR="${LR:-1e-5}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
export GROUP_SIZE="${GROUP_SIZE:-8}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-64}"
export SAMPLING_MAX_TOKENS="${SAMPLING_MAX_TOKENS:-1536}"
export EPOCHS_PER_ROLLOUT="${EPOCHS_PER_ROLLOUT:-1}"
export USE_STD_NORMALIZATION="${USE_STD_NORMALIZATION:-1}"
export ENABLE_THINKING="${ENABLE_THINKING:-1}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"
export EVAL_EXAMPLES="${EVAL_EXAMPLES:-256}"
export EVAL_EVERY="${EVAL_EVERY:-5}"

# Step count is the main knob the "50 -> 100 -> 200" progression exercises.
export N_GRPO_STEPS="${N_GRPO_STEPS:-50}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_sdpo.sh" "$@"
