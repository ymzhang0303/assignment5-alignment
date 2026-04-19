#!/usr/bin/env bash
# GRPO training of Qwen3-1.7B (a thinking model) on GSM8K.
#
# Key Qwen3-specific bits:
#   * --prompt-format chat        -> render prompts via tokenizer.apply_chat_template
#   * --prompt-name qwen3_thinking -> our system prompt asking for <answer>...</answer>
#   * --reward-fn-name r1_zero_thinking -> permissive </think> ... <answer> grader
#   * --log-all-rollouts          -> dump every rollout to runs/.../rollouts.jsonl
#
# Override via env, e.g.:
#   WANDB_MODE=offline OUTPUT_DIR=runs/grpo_qwen3_smoke ./train_scripts/run_qwen3.sh
set -euo pipefail

cd "$(dirname "$0")/.."

export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_PROJECT="${WANDB_PROJECT:-cs336-grpo}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-grpo_qwen3_gsm8k}"
export WANDB_MODE="${WANDB_MODE:-online}"

MODEL_ID="${MODEL_ID:-/root/assignment5-alignment/models/Qwen3-1.7B}"
TRAIN_PATH="${TRAIN_PATH:-data/gsm8k_grpo/train.jsonl}"
VAL_PATH="${VAL_PATH:-data/gsm8k_grpo/validation.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/grpo_qwen3_gsm8k}"

# Enable / disable Qwen3's native <think>...</think> reasoning.
#   ENABLE_THINKING=1 (default): model thinks freely, very long rollouts.
#   ENABLE_THINKING=0          : chat template injects an empty <think></think>
#                                so the model emits only <answer>...</answer>.
ENABLE_THINKING="${ENABLE_THINKING:-1}"
if [[ "$ENABLE_THINKING" == "1" ]]; then
    THINKING_FLAG="--enable-thinking"
    PROMPT_NAME_DEFAULT="qwen3_thinking"
else
    THINKING_FLAG="--no-enable-thinking"
    PROMPT_NAME_DEFAULT="qwen3_no_thinking"
fi
PROMPT_NAME="${PROMPT_NAME:-$PROMPT_NAME_DEFAULT}"

# GRPO hyperparameters (writeup defaults, but a touch smaller rollout batch
# since Qwen3 thinks => longer responses => more KV-cache).
N_GRPO_STEPS="${N_GRPO_STEPS:-200}"
LR="${LR:-1e-5}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
GROUP_SIZE="${GROUP_SIZE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-64}"
SAMPLING_MAX_TOKENS="${SAMPLING_MAX_TOKENS:-1536}"
EPOCHS_PER_ROLLOUT="${EPOCHS_PER_ROLLOUT:-1}"
LOSS_TYPE="${LOSS_TYPE:-reinforce_with_baseline}"
EVAL_EVERY="${EVAL_EVERY:-5}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-256}"

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
    --device "${DEVICE:-cuda:0}" \
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
