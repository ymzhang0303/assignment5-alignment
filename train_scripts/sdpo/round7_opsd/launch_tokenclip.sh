#!/usr/bin/env bash
# Follow-up to round 7: test OPSD's per-token pointwise divergence clip.
#
# OPSD README:
#   "per-token point-wise KL clipping. We find style tokens (such as 'wait',
#    'think') can exhibit 6-15x higher KL divergence than math-related tokens,
#    and dominates the training signal. Clipping stablizes training and
#    improves performance."
# Their reference value: jsd_token_clip=0.05 for Qwen3-1.7B.
#
# Hypothesis for why it matters for us right now:
#   r6_d_grpoheavy_lr1e5 just beat GRPO (peak 0.625 vs 0.648, last 0.609 vs
#   0.606 — +0.004). It uses JSD distillation on self-demos. If style tokens
#   are dominating its gradient, clipping them should let math tokens drive
#   more learning and push the peak higher.
#
#   r7_b_gt_grpoheavy_lr1e5 (same recipe but with GT teacher) peaked lower
#   (0.543). Suspicion: the GT-teacher prompt says "correct answer is X" so
#   the teacher is extremely peaked at the answer-token positions -> very
#   high local KL at those tokens -> dominates gradient toward "gamble the
#   answer" behavior. Token-clip should directly cure this.
#
# We launch on the 2 GPUs (0, 4) freed by killing the collapsed r7_a and r7_c.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/sdpo_r7

BASE_ENV=(
    N_GRPO_STEPS=200
    EVAL_EVERY=10
    TEACHER_UPDATE_RATE=0.05
    SUCCESS_THRESHOLD=0.5
    REMOVE_THINKING_FROM_DEMO=1
    MIN_DEMO_THINKING_CHARS=0
    GPU_MEM_UTIL=0.35
)

launch() {
    local gpu=$1 name=$2; shift 2
    local log=logs/sdpo_r7/${name}.log
    nohup env "${BASE_ENV[@]}" "$@" \
        DEVICE="cuda:$gpu" \
        OUTPUT_DIR="runs/sdpo_r7_${name}" \
        WANDB_RUN_NAME="sdpo_r7_${name}" \
        bash train_scripts/sdpo/run_sdpo.sh > "$log" 2>&1 < /dev/null &
    echo "[gpu $gpu] $name  pid=$!  log=$log"
    sleep 3
}

# e: OPSD-style token clip on top of the current best (r6_d) recipe --
#    self-demo teacher, not GT teacher. Cleanest "does token clip improve
#    the winning config?" test.
launch 0 e_tokenclip_on_r6d \
    GT_TEACHER=0 \
    LR=1e-5 \
    PG_LOSS_WEIGHT=10.0 \
    PG_APPLY_TO_ALL_SAMPLES=1 \
    TOKEN_CLIP=0.05

# f: token clip + GT teacher, GRPO-heavy. Tests whether the GT-teacher
#    plateau in r7_b (0.543) was due to style-token dominance.
launch 4 f_tokenclip_gt_grpoheavy \
    GT_TEACHER=1 \
    LR=1e-5 \
    PG_LOSS_WEIGHT=10.0 \
    PG_APPLY_TO_ALL_SAMPLES=1 \
    TOKEN_CLIP=0.05

echo
echo "2 token-clip runs launched."
