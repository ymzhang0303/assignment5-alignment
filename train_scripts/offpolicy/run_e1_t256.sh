#!/usr/bin/env bash
# On-policy reference: 1 epoch, train_batch=rollout_batch=256, so 1 optimizer
# update per rollout batch (= the standard GRPO recipe). loss_type=grpo_clip
# is harmless here -- with 1 epoch and 1 update, the ratio is exactly 1.
set -euo pipefail
export TAG="e1_t256"
export EPOCHS_PER_ROLLOUT=1
export TRAIN_BATCH_SIZE=256
export DEVICE="${DEVICE:-cuda:0}"
export N_GRPO_STEPS="${N_GRPO_STEPS:-40}"
source "$(dirname "$0")/_common.sh" "$@"
