# Shared off-policy GRPO knobs sourced by every wrapper in this directory.
# All wrappers fix:
#   - rollout_batch_size = 256
#   - loss_type = grpo_clip          (PPO-style ratio clip with cliprange=0.2)
#   - lr = 1e-5                      (best from the LR sweep)
#   - micro_train_batch_size = 2     (keep memory constant; gradient_accum varies)
#
# Off-policyness is controlled per-wrapper by EPOCHS_PER_ROLLOUT and
# TRAIN_BATCH_SIZE. Number of optimizer updates per rollout batch is:
#   updates_per_rollout = EPOCHS_PER_ROLLOUT * (256 / TRAIN_BATCH_SIZE)
#
# Wrappers MUST export TAG, EPOCHS_PER_ROLLOUT, TRAIN_BATCH_SIZE, DEVICE,
# and N_GRPO_STEPS before sourcing this file.
: "${TAG:?_common.sh requires TAG (e.g. e1_t256) to be set}"
: "${EPOCHS_PER_ROLLOUT:?_common.sh requires EPOCHS_PER_ROLLOUT}"
: "${TRAIN_BATCH_SIZE:?_common.sh requires TRAIN_BATCH_SIZE}"
: "${DEVICE:?_common.sh requires DEVICE (e.g. cuda:0)}"
: "${N_GRPO_STEPS:?_common.sh requires N_GRPO_STEPS}"

# Memory-constant: micro batch = 2, so grad_accum = T / 2.
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-$((TRAIN_BATCH_SIZE / 2))}"

export LR="${LR:-1e-5}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
export LOSS_TYPE="${LOSS_TYPE:-grpo_clip}"
export EPOCHS_PER_ROLLOUT
export TRAIN_BATCH_SIZE
export GRAD_ACCUM_STEPS
export N_GRPO_STEPS

export OUTPUT_DIR="${OUTPUT_DIR:-runs/grpo_qwen3_bigmath_offpolicy_${TAG}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-grpo_qwen3_bigmath_offpolicy_${TAG}}"
export WANDB_PROJECT="${WANDB_PROJECT:-cs336-grpo-offpolicy}"

source "$(dirname "${BASH_SOURCE[0]}")/../lr_sweep/_base.sh" "$@"
