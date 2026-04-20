#!/usr/bin/env bash
# Off-policy via re-iteration only: 2 epochs, train_batch=rollout_batch=256.
# Updates per rollout = 2 * (256/256) = 2. Same chunking as e1_t256 (none),
# only E differs -- isolates the "stale-rollout" axis from the "small-batch
# chunking" axis.
set -euo pipefail
export TAG="e2_t256"
export EPOCHS_PER_ROLLOUT=2
export TRAIN_BATCH_SIZE=256
export DEVICE="${DEVICE:-cuda:4}"
export N_GRPO_STEPS="${N_GRPO_STEPS:-40}"
source "$(dirname "$0")/_common.sh" "$@"
