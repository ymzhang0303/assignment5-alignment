#!/usr/bin/env bash
# Best-known SDPO recipe for Qwen3-1.7B on our Big-Math setting.
#
# Round-9 winner: **`pgw5_advmask`** — the first SDPO recipe that
# measurably beats GRPO on the smoothed metric.
#
#   run                        tail10   peak       chars   fmt
#   GRPO reinforce_bl          0.622    0.648@165  2271    0.837
#   sdpo_r9_pgw5_advmask       0.627    0.652@160  2486    0.828   (+0.005)
#   (prev best: r6_d)          0.607    0.633@190  2662    0.806
#
# Why it works:
#   * ADV_MASK_DISTILL=1 — AND positive-advantage into the distill mask,
#     so the EMA self-teacher only learns from rollouts that outperform
#     their group mean. Kills the "distilling on wrong-rollout tokens"
#     failure mode of r6_d.
#   * PG_LOSS_WEIGHT=5 (halved from r6_d's 10). Once the distill signal
#     is cleaned by advmask, the teacher's knowledge can contribute ~30%
#     of the gradient magnitude (vs ~6% at pg_w=10) without regressing on
#     verbosity/format. Pure SDPO (pg_w=0) still caps out below 0.52;
#     pg_w=10 drowns the distill in a GRPO shell.
#   * PG_APPLY_TO_ALL_SAMPLES=1 — every rollout feeds the PG term, giving
#     a reward baseline even when no successful demo exists.
#   * TEACHER_UPDATE_RATE=0.05 — EMA teacher (paper value). τ=0.02 was
#     marginally worse; τ=1.0 (no EMA) collapses.
#   * SUCCESS_THRESHOLD=0.5 — paper value. 1.0 (perfect-only) hurts demo
#     coverage without a quality gain.
#   * LR=1e-5 (not 1e-6 from Round 4). With advmask + pg_w=5, the higher
#     LR learns faster without destabilising.
#
# Dominant knobs that *did not* help once advmask was in:
#   * forward-KL (α=0) — consistently regressed on chars and format.
#   * length-penalty (1e-5 or 5e-5) — collapses to short outputs within
#     100 steps because the in-group length delta swamps the correctness
#     signal under std-normalisation.
#
# Override any knob by exporting it before calling this script, e.g.
#   DEVICE=cuda:2 N_GRPO_STEPS=500 OUTPUT_DIR=runs/sdpo_long bash run_sdpo_best.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export N_GRPO_STEPS="${N_GRPO_STEPS:-200}"
export EVAL_EVERY="${EVAL_EVERY:-10}"
export EVAL_EXAMPLES="${EVAL_EXAMPLES:-256}"
export SAMPLING_MAX_TOKENS="${SAMPLING_MAX_TOKENS:-1536}"

export LR="${LR:-1e-5}"
export TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"
export SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-0.5}"
export REMOVE_THINKING_FROM_DEMO="${REMOVE_THINKING_FROM_DEMO:-1}"
export MIN_DEMO_THINKING_CHARS="${MIN_DEMO_THINKING_CHARS:-0}"

export PG_LOSS_WEIGHT="${PG_LOSS_WEIGHT:-5.0}"
export PG_APPLY_TO_ALL_SAMPLES="${PG_APPLY_TO_ALL_SAMPLES:-1}"
export ADV_MASK_DISTILL="${ADV_MASK_DISTILL:-1}"

unset REPROMPT_TEMPLATE SOLUTION_TEMPLATE LENGTH_PENALTY TOKEN_CLIP || true
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"

export DEVICE="${DEVICE:-cuda:0}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/sdpo_best}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-sdpo_best}"

exec bash "$REPO_ROOT/train_scripts/sdpo/run_sdpo.sh" "$@"
