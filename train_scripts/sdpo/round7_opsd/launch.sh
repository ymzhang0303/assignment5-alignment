#!/usr/bin/env bash
# Round 7: borrow ideas from OPSD (arXiv:2601.18734).
#
# Key OPSD idea: the teacher conditions on the dataset's ground-truth
# answer (privileged info) instead of a self-generated successful demo.
# Every sample gets a strong teacher signal (vs 50-70% in pure SDPO),
# and the teacher's predictions are anchored by known-correct answers.
#
# Current round-6 status at step ~80:
#   GRPO baseline (target):    peak=0.648  final=0.606
#   r6_d_grpoheavy_lr1e5 (leader): peak=0.582  (gap -0.066)
#   r5_paper (stable SDPO):         peak=0.512  final=0.508
#
# OPSD paper reference config for Qwen3-1.7B (our exact model!):
#   learning_rate=5e-6      (between our 1e-6 and 1e-5)
#   max_grad_norm=0.1       (10x tighter than our 1.0)
#   temperature=1.1         (more diverse rollouts -> richer distillation)
#   beta=0 (forward KL)
#   jsd_token_clip=0.05
#   fixed_teacher (we use EMA instead)
#
# We run on the 4 GPUs freed by killing the non-promising round-6 runs
# (a_pure, b_pgfill, e_grpoheavy_lr1e6, f_grpodomlight_lr1e6).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/sdpo_r7

# Shared env for all 4 runs. 200 steps matches round 6 and GRPO.
BASE_ENV=(
    N_GRPO_STEPS=200
    EVAL_EVERY=10
    TEACHER_UPDATE_RATE=0.05      # EMA teacher, proven in r5
    SUCCESS_THRESHOLD=0.5         # paper default, not used when GT_TEACHER=1 but kept for safety
    REMOVE_THINKING_FROM_DEMO=1
    MIN_DEMO_THINKING_CHARS=0
    GPU_MEM_UTIL=0.35
    GT_TEACHER=1                  # all 4 runs use OPSD-style GT teacher
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

# a: clean test of "does GT teacher alone beat self-demo?" at safe mid LR.
#    No PG mixing. If this alone reaches ~0.55+, GT teacher is the main unlock.
launch 0 a_gt_pure_lr5e6  LR=5e-6  PG_LOSS_WEIGHT=0.0  PG_APPLY_TO_ALL_SAMPLES=0

# b: drop GT teacher into the current leader's recipe (r6_d_grpoheavy_lr1e5
#    peak 0.582). Hypothesis: strong teacher + GRPO-heavy PG pushes past GRPO.
launch 1 b_gt_grpoheavy_lr1e5  LR=1e-5  PG_LOSS_WEIGHT=10.0  PG_APPLY_TO_ALL_SAMPLES=1

# c: faithful OPSD borrow on top of our stack: lr=5e-6, grad_clip=0.1,
#    sampling temp=1.1 (their reference config for Qwen3-1.7B).
launch 4 c_gt_opsd_faithful  LR=5e-6  PG_LOSS_WEIGHT=0.0  PG_APPLY_TO_ALL_SAMPLES=0 \
    GRAD_CLIP=0.1  SAMPLING_TEMPERATURE=1.1

# d: light PG mix on all samples at LR=1e-5. Tests if a modest GRPO term
#    closes the gap when we already have 100% teacher coverage.
launch 5 d_gt_pglight_lr1e5  LR=1e-5  PG_LOSS_WEIGHT=1.0  PG_APPLY_TO_ALL_SAMPLES=1

echo
echo "4 OPSD-inspired runs launched on GPUs 0,1,4,5."
