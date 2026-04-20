#!/usr/bin/env bash
# 100-step runs of the top round-3 recipes, in parallel on 4 GPUs.
# Based on round-3 findings:
#   * EMA teacher (teacher_update_rate < 1.0) is the critical fix.
#   * LR=1e-5 + EMA 0.05 gave the fastest climb (peak val=0.496 at step 35).
#   * LR=1e-6 + EMA 0.05 was slower but still climbing at step 49.
# We run 4 hedges to see which best survives to step 100.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/sdpo_r4

BASE_ENV=(
    N_GRPO_STEPS=100
    EVAL_EVERY=5
    SUCCESS_THRESHOLD=0.5
    REMOVE_THINKING_FROM_DEMO=1
    MIN_DEMO_THINKING_CHARS=0
    PG_LOSS_WEIGHT=0.0
    PG_APPLY_TO_ALL_SAMPLES=0
    GPU_MEM_UTIL=0.35
)

# a: winner from round-3 (LR=1e-5, EMA decay=0.05)
nohup env "${BASE_ENV[@]}" \
    DEVICE=cuda:0 LR=1e-5 TEACHER_UPDATE_RATE=0.05 \
    OUTPUT_DIR=runs/sdpo_r4_a_winner WANDB_RUN_NAME=sdpo_r4_a_winner \
    bash train_scripts/sdpo/run_sdpo.sh > logs/sdpo_r4/a_winner.log 2>&1 < /dev/null &
echo "a_winner (LR=1e-5 EMA=0.05) launched on cuda:0 pid=$!"
sleep 3

# b: paper recipe (LR=1e-6, EMA 0.05)
nohup env "${BASE_ENV[@]}" \
    DEVICE=cuda:1 LR=1e-6 TEACHER_UPDATE_RATE=0.05 \
    OUTPUT_DIR=runs/sdpo_r4_b_paper WANDB_RUN_NAME=sdpo_r4_b_paper \
    bash train_scripts/sdpo/run_sdpo.sh > logs/sdpo_r4/b_paper.log 2>&1 < /dev/null &
echo "b_paper (LR=1e-6 EMA=0.05) launched on cuda:1 pid=$!"
sleep 3

# c: mid LR hedge (LR=3e-6, EMA 0.05)
nohup env "${BASE_ENV[@]}" \
    DEVICE=cuda:2 LR=3e-6 TEACHER_UPDATE_RATE=0.05 \
    OUTPUT_DIR=runs/sdpo_r4_c_midlr WANDB_RUN_NAME=sdpo_r4_c_midlr \
    bash train_scripts/sdpo/run_sdpo.sh > logs/sdpo_r4/c_midlr.log 2>&1 < /dev/null &
echo "c_midlr (LR=3e-6 EMA=0.05) launched on cuda:2 pid=$!"
sleep 3

# d: high LR + stronger teacher reg (LR=1e-5, EMA 0.02)
nohup env "${BASE_ENV[@]}" \
    DEVICE=cuda:3 LR=1e-5 TEACHER_UPDATE_RATE=0.02 \
    OUTPUT_DIR=runs/sdpo_r4_d_strongreg WANDB_RUN_NAME=sdpo_r4_d_strongreg \
    bash train_scripts/sdpo/run_sdpo.sh > logs/sdpo_r4/d_strongreg.log 2>&1 < /dev/null &
echo "d_strongreg (LR=1e-5 EMA=0.02) launched on cuda:3 pid=$!"

echo
echo "4 runs launched. monitor with: python train_scripts/sdpo/round4_100steps/status.py"
