#!/usr/bin/env bash
# Shared defaults for the SDPO collapse-fix ablation sweep.
#
# Baseline run (runs/sdpo_qwen3_bigmath) collapsed to ~38-char no-think
# answers by step ~10 because every demo presented to the teacher was just
# `<answer>X</answer>` (median demo length stayed 18 chars all run). Each
# ablation flips one knob and re-runs ~30 steps so we can compare:
#   - val/answer_reward at steps 5/10/15/20 (vs baseline ~ 0.37/0.12/0.11/0.15)
#   - val/response_chars (baseline collapses to ~38 by step 10)
#   - sdpo/sample_with_demo_fraction (baseline collapses to ~0.06-0.25)
#
# All ablations share the same shape, seeds, eval cadence, and tiny step
# budget so per-step numbers are directly comparable.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Match the baseline run shape exactly so val curves are comparable.
export N_GRPO_STEPS="${N_GRPO_STEPS:-30}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
export GROUP_SIZE="${GROUP_SIZE:-8}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-64}"
export SAMPLING_MAX_TOKENS="${SAMPLING_MAX_TOKENS:-1536}"
export EPOCHS_PER_ROLLOUT="${EPOCHS_PER_ROLLOUT:-1}"
export EVAL_EVERY="${EVAL_EVERY:-5}"
export EVAL_EXAMPLES="${EVAL_EXAMPLES:-256}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"
export LR="${LR:-1e-5}"

# Same SDPO defaults as the baseline; ablations only override one knob.
export SDPO_ALPHA="${SDPO_ALPHA:-0.5}"
export DISTILLATION_TOPK="${DISTILLATION_TOPK:-100}"
export IS_CLIP="${IS_CLIP:-2.0}"
export SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-1.0}"
export TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-1.0}"
export PG_LOSS_TYPE="${PG_LOSS_TYPE:-grpo_clip}"
export PG_LOSS_WEIGHT="${PG_LOSS_WEIGHT:-0.0}"
export USE_STD_NORMALIZATION="${USE_STD_NORMALIZATION:-1}"
export DONT_REPROMPT_ON_SELF_SUCCESS="${DONT_REPROMPT_ON_SELF_SUCCESS:-1}"
export ENABLE_THINKING="${ENABLE_THINKING:-1}"

export WANDB_PROJECT="${WANDB_PROJECT:-cs336-sdpo}"
export WANDB_MODE="${WANDB_MODE:-online}"
