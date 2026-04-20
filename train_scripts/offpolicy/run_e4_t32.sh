#!/usr/bin/env bash
# Most off-policy in the broad sweep: 4 epochs, train_batch=32.
# Updates per rollout = 4 * (256/32) = 32.
set -euo pipefail
export TAG="e4_t32"
export EPOCHS_PER_ROLLOUT=4
export TRAIN_BATCH_SIZE=32
export DEVICE="${DEVICE:-cuda:3}"
export N_GRPO_STEPS="${N_GRPO_STEPS:-40}"
source "$(dirname "$0")/_common.sh" "$@"
