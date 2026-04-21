#!/usr/bin/env bash
# Shared GRPO training entrypoint for the Big-Math LR sweep on Qwen3-1.7B.
#
# Per-LR wrapper scripts (e.g. run_qwen3_bigMath_1e-5.sh) export LR / DEVICE /
# OUTPUT_DIR / WANDB_RUN_NAME and then `source` this file. All other knobs are
# overridable via env so we keep one source of truth for the hyperparameters.
set -euo pipefail

# Resolve repo root relative to this file (so wrappers can be invoked from anywhere).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_PROJECT="${WANDB_PROJECT:-cs336-grpo}"
export WANDB_MODE="${WANDB_MODE:-online}"

: "${LR:?_base.sh requires LR to be set by the wrapper}"
: "${DEVICE:?_base.sh requires DEVICE (e.g. cuda:0) to be set by the wrapper}"
: "${OUTPUT_DIR:?_base.sh requires OUTPUT_DIR to be set by the wrapper}"
: "${WANDB_RUN_NAME:?_base.sh requires WANDB_RUN_NAME to be set by the wrapper}"
export WANDB_RUN_NAME

MODEL_ID="${MODEL_ID:-/root/assignment5-alignment/models/Qwen3-1.7B}"
TRAIN_PATH="${TRAIN_PATH:-data/big_math/train.jsonl}"
VAL_PATH="${VAL_PATH:-data/big_math/validation.jsonl}"

# Qwen3 thinking on by default (matches eval_qwen3_bigmath.py).
ENABLE_THINKING="${ENABLE_THINKING:-1}"
if [[ "$ENABLE_THINKING" == "1" ]]; then
    THINKING_FLAG="--enable-thinking"
    PROMPT_NAME_DEFAULT="qwen3_thinking"
else
    THINKING_FLAG="--no-enable-thinking"
    PROMPT_NAME_DEFAULT="qwen3_no_thinking"
fi
PROMPT_NAME="${PROMPT_NAME:-$PROMPT_NAME_DEFAULT}"

# GRPO hyperparameters -- shared across the sweep so only LR varies.
N_GRPO_STEPS="${N_GRPO_STEPS:-200}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
GROUP_SIZE="${GROUP_SIZE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-64}"
SAMPLING_MAX_TOKENS="${SAMPLING_MAX_TOKENS:-1536}"
EPOCHS_PER_ROLLOUT="${EPOCHS_PER_ROLLOUT:-1}"
LOSS_TYPE="${LOSS_TYPE:-reinforce_with_baseline}"
EVAL_EVERY="${EVAL_EVERY:-5}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-256}"

# Optional Dr-GRPO-style length normalization. Leave LENGTH_NORMALIZATION unset
# to use the default 'masked_mean'. Set to 'masked_normalize' to switch to
# sum / NORMALIZE_CONSTANT (defaults to sampling_max_tokens if unset).
LENGTH_NORMALIZATION="${LENGTH_NORMALIZATION:-masked_mean}"
LENGTH_NORM_FLAGS=(--length-normalization "$LENGTH_NORMALIZATION")
if [[ -n "${NORMALIZE_CONSTANT:-}" ]]; then
    LENGTH_NORM_FLAGS+=(--normalize-constant "$NORMALIZE_CONSTANT")
fi

# Optional: turn off std normalization in the group-relative advantage
# (Dr-GRPO recipe). Default is "1" (use std). Set USE_STD_NORMALIZATION=0
# to pass --no-use-std-normalization, which keeps advantages = r - mean(r).
USE_STD_NORMALIZATION="${USE_STD_NORMALIZATION:-1}"
case "$USE_STD_NORMALIZATION" in
    1|true|True|TRUE|yes|on)  STD_NORM_FLAG="--use-std-normalization" ;;
    0|false|False|FALSE|no|off) STD_NORM_FLAG="--no-use-std-normalization" ;;
    *) echo "USE_STD_NORMALIZATION must be 0/1, got: $USE_STD_NORMALIZATION" >&2; exit 1 ;;
esac

# Optional: AND (advantage > 0) into the PG response-mask so rollouts that
# under-perform their group mean don't contribute to the policy-gradient
# loss (mirror of SDPO's --adv-mask-distill, used to disentangle the
# advantage-gate from the self-distillation term).
ADV_MASK="${ADV_MASK:-0}"
case "$ADV_MASK" in
    1|true|True|TRUE|yes|on)  ADV_MASK_FLAG="--adv-mask" ;;
    0|false|False|FALSE|no|off) ADV_MASK_FLAG="--no-adv-mask" ;;
    *) echo "ADV_MASK must be 0/1, got: $ADV_MASK" >&2; exit 1 ;;
esac

# Optional cliprange override (default 0.2 in grpo_train). Mostly useful
# when LOSS_TYPE=grpo_clip.
CLIPRANGE="${CLIPRANGE:-0.2}"

mkdir -p "$OUTPUT_DIR"

uv run python -m cs336_alignment.grpo_train \
    --model-id "$MODEL_ID" \
    --train-path "$TRAIN_PATH" \
    --val-path  "$VAL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-format chat \
    --prompt-name "$PROMPT_NAME" \
    $THINKING_FLAG \
    --reward-fn-name r1_zero_thinking \
    --device "$DEVICE" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.45}" \
    --n-grpo-steps "$N_GRPO_STEPS" \
    --learning-rate "$LR" \
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
    --group-size "$GROUP_SIZE" \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM_STEPS" \
    --sampling-max-tokens "$SAMPLING_MAX_TOKENS" \
    --epochs-per-rollout-batch "$EPOCHS_PER_ROLLOUT" \
    --loss-type "$LOSS_TYPE" \
    --cliprange "$CLIPRANGE" \
    "${LENGTH_NORM_FLAGS[@]}" \
    "$STD_NORM_FLAG" \
    "$ADV_MASK_FLAG" \
    --eval-every "$EVAL_EVERY" \
    --eval-examples "$EVAL_EXAMPLES" \
    --log-all-rollouts \
    --log-examples-every 1 \
    --n-log-examples 8 \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_RUN_NAME" \
    --wandb-mode "$WANDB_MODE" \
    "$@"
