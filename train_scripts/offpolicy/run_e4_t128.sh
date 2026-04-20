#!/usr/bin/env bash
# Off-policy via re-iteration + chunking: 4 epochs, train_batch=128.
# Updates per rollout = 4 * (256/128) = 8. Fills the gap between e2_t128
# (4 updates) and e4_t64 (16 updates) on the broad sweep.
set -euo pipefail
export TAG="e4_t128"
export EPOCHS_PER_ROLLOUT=4
export TRAIN_BATCH_SIZE=128
export DEVICE="${DEVICE:-cuda:6}"
export N_GRPO_STEPS="${N_GRPO_STEPS:-40}"
source "$(dirname "$0")/_common.sh" "$@"
