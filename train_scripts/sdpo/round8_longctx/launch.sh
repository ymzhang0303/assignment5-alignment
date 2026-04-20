#!/usr/bin/env bash
# Round 8: long-context sweep.
#
# Hypothesis: SAMPLING_MAX_TOKENS=1536 was truncating ~60% of Qwen3-1.7B
# rollouts at init. Format reward at val step 0 sits at ~0.38-0.62 because
# many completions never close <answer>...</answer>. Bumping the budget to
# 4096 should:
#   (a) raise the starting format / answer reward,
#   (b) expand the SDPO demo pool (fewer rollouts that can't contribute),
#   (c) reduce the short-bias selection pressure from the teacher.
#
# Five experiments at N=200 steps, SAMPLING_MAX_TOKENS=4096, LR=1e-5:
#   exp0 GRPO only                              -> GRPO baseline
#   exp1 pure SDPO, EMA=0.02                    -> SDPO baseline (d_strongreg)
#   exp2 + pg_w=1,  PG on no-demo only
#   exp3 + pg_w=1,  PG on all samples
#   exp4 + pg_w=10, PG on all samples

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/sdpo_r8

# Shared shape for all 5 runs.
COMMON_ENV=(
    N_GRPO_STEPS=200
    EVAL_EVERY=10
    EVAL_EXAMPLES=256
    SAMPLING_MAX_TOKENS=4096
    # 4096 ctx => ~2.7x KV footprint of 1536; bump vLLM budget accordingly.
    GPU_MEM_UTIL=0.50
)

SDPO_COMMON_ENV=(
    "${COMMON_ENV[@]}"
    LR=1e-5
    TEACHER_UPDATE_RATE=0.02
    SUCCESS_THRESHOLD=0.5
    REMOVE_THINKING_FROM_DEMO=1
    MIN_DEMO_THINKING_CHARS=0
)

launch_sdpo() {
    local gpu=$1 name=$2; shift 2
    local log=logs/sdpo_r8/${name}.log
    nohup env "${SDPO_COMMON_ENV[@]}" "$@" \
        DEVICE="cuda:$gpu" \
        OUTPUT_DIR="runs/sdpo_r8_${name}" \
        WANDB_RUN_NAME="sdpo_r8_${name}" \
        bash train_scripts/sdpo/run_sdpo.sh > "$log" 2>&1 < /dev/null &
    echo "[gpu $gpu] sdpo_${name}  pid=$!  log=$log"
    sleep 3
}

launch_grpo() {
    local gpu=$1 name=$2; shift 2
    local log=logs/sdpo_r8/${name}.log
    nohup env "${COMMON_ENV[@]}" "$@" \
        LR=1e-5 \
        DEVICE="cuda:$gpu" \
        OUTPUT_DIR="runs/sdpo_r8_${name}" \
        WANDB_RUN_NAME="sdpo_r8_${name}" \
        WANDB_PROJECT="cs336-sdpo" \
        bash train_scripts/loss_type/run_reinforce_baseline.sh \
            > "$log" 2>&1 < /dev/null &
    echo "[gpu $gpu] grpo_${name}  pid=$!  log=$log"
    sleep 3
}

# exp0: GRPO baseline at long context.
launch_grpo  1  exp0_grpo_baseline

# exp1: pure SDPO baseline (= round-4 "d_strongreg": LR=1e-5, EMA=0.02).
launch_sdpo  2  exp1_sdpo_baseline  PG_LOSS_WEIGHT=0.0  PG_APPLY_TO_ALL_SAMPLES=0

# exp2: SDPO + PG on samples without a teacher demo.
launch_sdpo  3  exp2_pgfill_w1      PG_LOSS_WEIGHT=1.0  PG_APPLY_TO_ALL_SAMPLES=0

# exp3: SDPO + PG on every sample, light weight.
launch_sdpo  5  exp3_pgall_w1       PG_LOSS_WEIGHT=1.0  PG_APPLY_TO_ALL_SAMPLES=1

# exp4: SDPO + PG on every sample, GRPO-dominant weight (r6_d recipe at EMA=0.02).
launch_sdpo  6  exp4_pgall_w10      PG_LOSS_WEIGHT=10.0 PG_APPLY_TO_ALL_SAMPLES=1

echo
echo "5 runs launched on cuda:{1,2,3,5,6}. Monitor with:"
echo "  python train_scripts/sdpo/round8_longctx/status.py"
