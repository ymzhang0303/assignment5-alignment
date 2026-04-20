#!/usr/bin/env bash
# SDPO (self-distillation) training entrypoint, modelled on
# train_scripts/lr_sweep/_base.sh. The defaults reproduce the SDPO recipe
# from the paper / lasgroup/SDPO repo on our Big-Math / Qwen3-1.7B setup:
#   - rollout_batch_size = 256, group_size = 8 (32 prompts per batch)
#   - lr = 1e-5, alpha = 0.5 (JSD), top-k = 100 with tail bucket
#   - is_clip = 2.0, success_reward_threshold = 1.0
#   - teacher = current policy (teacher_update_rate = 1.0 -> no extra mem)
#   - pure SDPO: no PG fallback term (set PG_LOSS_WEIGHT > 0 to mix in GRPO)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-cs336-sdpo}"
export WANDB_MODE="${WANDB_MODE:-online}"

LR="${LR:-1e-5}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/sdpo_qwen3_bigmath}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-sdpo_qwen3_bigmath}"
export WANDB_RUN_NAME

MODEL_ID="${MODEL_ID:-/root/assignment5-alignment/models/Qwen3-1.7B}"
TRAIN_PATH="${TRAIN_PATH:-data/big_math/train.jsonl}"
VAL_PATH="${VAL_PATH:-data/big_math/validation.jsonl}"

ENABLE_THINKING="${ENABLE_THINKING:-1}"
if [[ "$ENABLE_THINKING" == "1" ]]; then
    THINKING_FLAG="--enable-thinking"
    PROMPT_NAME_DEFAULT="qwen3_thinking"
else
    THINKING_FLAG="--no-enable-thinking"
    PROMPT_NAME_DEFAULT="qwen3_no_thinking"
fi
PROMPT_NAME="${PROMPT_NAME:-$PROMPT_NAME_DEFAULT}"

# Same rollout/training shape as the GRPO baseline.
N_GRPO_STEPS="${N_GRPO_STEPS:-200}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
GROUP_SIZE="${GROUP_SIZE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-64}"
SAMPLING_MAX_TOKENS="${SAMPLING_MAX_TOKENS:-1536}"
EPOCHS_PER_ROLLOUT="${EPOCHS_PER_ROLLOUT:-1}"
EVAL_EVERY="${EVAL_EVERY:-5}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-256}"

# SDPO knobs.
SDPO_ALPHA="${SDPO_ALPHA:-0.5}"
DISTILLATION_TOPK="${DISTILLATION_TOPK:-100}"
IS_CLIP="${IS_CLIP:-2.0}"
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-1.0}"
TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-1.0}"
PG_LOSS_TYPE="${PG_LOSS_TYPE:-grpo_clip}"
PG_LOSS_WEIGHT="${PG_LOSS_WEIGHT:-0.0}"

USE_STD_NORMALIZATION="${USE_STD_NORMALIZATION:-1}"
case "$USE_STD_NORMALIZATION" in
    1|true|True|TRUE|yes|on)  STD_NORM_FLAG="--use-std-normalization" ;;
    0|false|False|FALSE|no|off) STD_NORM_FLAG="--no-use-std-normalization" ;;
    *) echo "USE_STD_NORMALIZATION must be 0/1, got: $USE_STD_NORMALIZATION" >&2; exit 1 ;;
esac

DONT_REPROMPT_ON_SELF_SUCCESS="${DONT_REPROMPT_ON_SELF_SUCCESS:-1}"
case "$DONT_REPROMPT_ON_SELF_SUCCESS" in
    1|true|True|TRUE|yes|on)  DROSS_FLAG="--dont-reprompt-on-self-success" ;;
    0|false|False|FALSE|no|off) DROSS_FLAG="--no-dont-reprompt-on-self-success" ;;
    *) echo "DONT_REPROMPT_ON_SELF_SUCCESS must be 0/1" >&2; exit 1 ;;
esac

mkdir -p "$OUTPUT_DIR"

uv run python -m cs336_alignment.sdpo_train \
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
    --pg-loss-type "$PG_LOSS_TYPE" \
    --pg-loss-weight "$PG_LOSS_WEIGHT" \
    --sdpo-alpha "$SDPO_ALPHA" \
    --distillation-topk "$DISTILLATION_TOPK" \
    --is-clip "$IS_CLIP" \
    --success-reward-threshold "$SUCCESS_THRESHOLD" \
    --teacher-update-rate "$TEACHER_UPDATE_RATE" \
    $DROSS_FLAG \
    "$STD_NORM_FLAG" \
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
