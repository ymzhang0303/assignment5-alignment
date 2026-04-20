#!/usr/bin/env bash
# Shared defaults for round-3 SDPO ablations.
#
# Based on careful reading of the SDPO paper (arXiv:2601.20802) and the
# reference config at github.com/lasgroup/SDPO/verl/trainer/config/actor/actor.yaml:
#
#   - paper uses LR = 1e-6 (we were at 1e-5, 10x too high)
#   - paper says non-regularized teacher (teacher=current policy) diverges
#     (Table 4); EMA/trust-region teacher is critical for stability
#   - paper uses success_reward_threshold = 0.5 (not 1.0)
#   - weaker base models benefit from SDPO+GRPO mix (lambda=0.9 GRPO in Qwen3-0.6B);
#     we're on Qwen3-1.7B which is still on the "weak" side of the paper's scaling,
#     so GRPO help is expected
#
# Each round-3 script sources this file, then tweaks one axis.

set -euo pipefail

# Shorter runs so the sweep finishes in reasonable time.
export N_GRPO_STEPS="${N_GRPO_STEPS:-50}"
export EVAL_EVERY="${EVAL_EVERY:-5}"

# Paper's reference LR. 10x lower than our previous runs.
export LR="${LR:-1e-6}"

# Paper reference config:
#   teacher_regularization: ema
#   teacher_update_rate: 0.05 (YAML default) / 0.01 (Table 4)
# This is the critical fix: prior runs used teacher=current policy, which the
# paper explicitly calls out as diverging.
export TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"

# Paper reference (0.5). Prior runs used 1.0, so few demos survived, which
# combined with short-success bias triggered collapse.
export SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-0.5}"

# Back to paper defaults: thinking removed from demos, no "force reason"
# reprompt override (earlier rounds added hacks to fight collapse; now that
# we have a regularized teacher we can use the clean paper recipe).
export REMOVE_THINKING_FROM_DEMO="${REMOVE_THINKING_FROM_DEMO:-1}"
export MIN_DEMO_THINKING_CHARS="${MIN_DEMO_THINKING_CHARS:-0}"
unset REPROMPT_TEMPLATE SOLUTION_TEMPLATE || true

# PG blend. Pure SDPO (0.0) matches paper. Raise to mix in GRPO for weaker
# base models (paper section 4.5).
export PG_LOSS_WEIGHT="${PG_LOSS_WEIGHT:-0.0}"
export PG_APPLY_TO_ALL_SAMPLES="${PG_APPLY_TO_ALL_SAMPLES:-0}"

# EMA teacher adds one extra copy of the policy in GPU memory and another
# during swap; tighten vLLM budget so we don't OOM on the 45% default.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.35}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export COMMON_REPO_ROOT="$REPO_ROOT"
