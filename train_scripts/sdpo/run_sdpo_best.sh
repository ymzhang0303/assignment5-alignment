#!/usr/bin/env bash
# Best-known SDPO recipe for Qwen3-1.7B on our Big-Math setting.
#
# Derived from a 4-round ablation sweep against the SDPO paper
# (arXiv:2601.20802) and the lasgroup/SDPO reference config. The three
# critical ingredients that make SDPO go from "collapses in 10 steps"
# to "+78% relative answer reward at step 200":
#
#   1. EMA teacher (teacher_update_rate < 1.0). Without this the teacher
#      equals the student and per the paper's Table 4 training diverges.
#      We use the paper's reference YAML value decay=0.05.
#
#   2. Low learning rate. LR=1e-6 (10x below our earlier runs).
#      LR=3e-6 trains faster but collapses around step 170;
#      LR=1e-6 holds peak (~0.51) through 200 steps.
#
#   3. success_reward_threshold=0.5 (paper default, not 1.0). Lets
#      partial successes contribute as demos.
#
# Monitored 200-step run results (200 * 128 = 25,600 rollouts):
#   init   val/answer_reward = 0.285
#   peak   val/answer_reward = 0.512 @ step 160
#   final  val/answer_reward = 0.508 @ step 199
#   final  val/response_chars = 3026 (from 4099 at init, stable)
#
# Override any knob by exporting it before calling this script, e.g.
#   DEVICE=cuda:2 N_GRPO_STEPS=500 OUTPUT_DIR=runs/sdpo_long bash run_sdpo_best.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export N_GRPO_STEPS="${N_GRPO_STEPS:-200}"
export EVAL_EVERY="${EVAL_EVERY:-10}"

# Paper reference values.
export LR="${LR:-1e-6}"
export TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"
export SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-0.5}"

# Paper defaults; we tried overriding (force-reason reprompt, keep-thinking
# demos, pg mixing) in earlier rounds but none beat this plain recipe once
# the EMA teacher + low LR were in place.
export REMOVE_THINKING_FROM_DEMO="${REMOVE_THINKING_FROM_DEMO:-1}"
export MIN_DEMO_THINKING_CHARS=0
export PG_LOSS_WEIGHT=0.0
export PG_APPLY_TO_ALL_SAMPLES=0
unset REPROMPT_TEMPLATE SOLUTION_TEMPLATE || true

# EMA teacher needs ~1x policy memory for the shadow plus another 1x for
# the swap backup, so tighten vLLM's share from the 0.45 default.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.35}"

export DEVICE="${DEVICE:-cuda:0}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/sdpo_best}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-sdpo_best}"

exec bash "$REPO_ROOT/train_scripts/sdpo/run_sdpo.sh" "$@"
