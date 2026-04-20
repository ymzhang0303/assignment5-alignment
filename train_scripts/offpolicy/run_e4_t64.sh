#!/usr/bin/env bash
# Off-policy: 4 epochs over rollout batch, train_batch=64.
# Updates per rollout = 4 * (256/64) = 16.
set -euo pipefail
export TAG="e4_t64"
export EPOCHS_PER_ROLLOUT=4
export TRAIN_BATCH_SIZE=64
export DEVICE="${DEVICE:-cuda:2}"
export N_GRPO_STEPS="${N_GRPO_STEPS:-40}"
source "$(dirname "$0")/_common.sh" "$@"
