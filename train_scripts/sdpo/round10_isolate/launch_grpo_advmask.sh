#!/usr/bin/env bash
# Round 10 ablation: isolate the contribution of SDPO's distillation term.
#
# The winning recipe `sdpo_r9_strict_demo_advmask` combines three things:
#   (i)   GRPO-clip policy-gradient at PG_LOSS_WEIGHT=10 and cliprange=0.2,
#   (ii)  a positive-advantage gate on the distillation mask (advmask),
#   (iii) a strict "only fully-correct rollouts count as demos" threshold.
#
# The question is whether the self-distillation term is actually pulling
# its weight, or whether all of the lift comes from (i) + (ii) (which are
# GRPO-native concepts that don't need a teacher at all).
#
# This launch runs pure GRPO with the positive-advantage gate applied to
# the policy-gradient loss itself (rollouts with advantage <= 0 are
# excluded from both numerator and denominator of masked_mean). Every
# other knob matches `strict_demo_advmask`. No distillation, no teacher.
#
# Interpretation:
#   grpo_advmask >> strict_demo_advmask  -> distillation was hurting.
#   grpo_advmask ~= strict_demo_advmask  -> distillation is vestigial.
#   grpo_advmask <<  strict_demo_advmask -> distillation does real work.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/sdpo_r10

NAME=grpo_advmask
LOG=logs/sdpo_r10/${NAME}.log

nohup env \
    N_GRPO_STEPS=200 \
    ROLLOUT_BATCH_SIZE=128 \
    GROUP_SIZE=8 \
    TRAIN_BATCH_SIZE=128 \
    GRAD_ACCUM_STEPS=64 \
    SAMPLING_MAX_TOKENS=1536 \
    EPOCHS_PER_ROLLOUT=1 \
    LOSS_TYPE=grpo_clip \
    CLIPRANGE=0.2 \
    USE_STD_NORMALIZATION=1 \
    LENGTH_NORMALIZATION=masked_mean \
    ADV_MASK=1 \
    GPU_MEM_UTIL=0.45 \
    EVAL_EVERY=10 \
    EVAL_EXAMPLES=256 \
    LR=1e-5 \
    DEVICE=cuda:2 \
    OUTPUT_DIR="runs/sdpo_r10_${NAME}" \
    WANDB_PROJECT=cs336-sdpo \
    WANDB_RUN_NAME="sdpo_r10_${NAME}" \
    bash train_scripts/loss_type/run_reinforce_baseline.sh \
    > "$LOG" 2>&1 < /dev/null &

echo "[gpu 2] sdpo_r10_${NAME}  pid=$!  log=$LOG"
