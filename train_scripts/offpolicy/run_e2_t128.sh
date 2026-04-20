#!/usr/bin/env bash
# Mild off-policy: 2 epochs over the rollout batch, train_batch=128.
# Updates per rollout = 2 * (256/128) = 4.
set -euo pipefail
export TAG="e2_t128"
export EPOCHS_PER_ROLLOUT=2
export TRAIN_BATCH_SIZE=128
export DEVICE="${DEVICE:-cuda:1}"
export N_GRPO_STEPS="${N_GRPO_STEPS:-40}"
source "$(dirname "$0")/_common.sh" "$@"
