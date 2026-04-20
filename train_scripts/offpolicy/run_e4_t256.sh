#!/usr/bin/env bash
# Off-policy via re-iteration only: 4 epochs, train_batch=rollout_batch=256.
# Updates per rollout = 4 * (256/256) = 4. Pairs with e2_t128 (also 4 updates
# but via E=2, T=128) to disambiguate "stale-rollout" (E) vs "small-batch
# chunking" (T) at matched updates/rollout.
set -euo pipefail
export TAG="e4_t256"
export EPOCHS_PER_ROLLOUT=4
export TRAIN_BATCH_SIZE=256
export DEVICE="${DEVICE:-cuda:5}"
export N_GRPO_STEPS="${N_GRPO_STEPS:-40}"
source "$(dirname "$0")/_common.sh" "$@"
