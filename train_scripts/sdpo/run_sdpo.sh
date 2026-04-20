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

export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_PROJECT="${WANDB_PROJECT:-cs336-sdpo}"
export WANDB_MODE="${WANDB_MODE:-online}"

LR="${LR:-1e-5}"
DEVICE="${DEVICE:-cuda:7}"
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

REMOVE_THINKING_FROM_DEMO="${REMOVE_THINKING_FROM_DEMO:-1}"
case "$REMOVE_THINKING_FROM_DEMO" in
    1|true|True|TRUE|yes|on)  RTFD_FLAG="--remove-thinking-from-demonstration" ;;
    0|false|False|FALSE|no|off) RTFD_FLAG="--no-remove-thinking-from-demonstration" ;;
    *) echo "REMOVE_THINKING_FROM_DEMO must be 0/1" >&2; exit 1 ;;
esac

MIN_DEMO_THINKING_CHARS="${MIN_DEMO_THINKING_CHARS:-0}"

PG_APPLY_TO_ALL_SAMPLES="${PG_APPLY_TO_ALL_SAMPLES:-0}"
case "$PG_APPLY_TO_ALL_SAMPLES" in
    1|true|True|TRUE|yes|on)  PG_ALL_FLAG="--pg-apply-to-all-samples" ;;
    0|false|False|FALSE|no|off) PG_ALL_FLAG="--no-pg-apply-to-all-samples" ;;
    *) echo "PG_APPLY_TO_ALL_SAMPLES must be 0/1" >&2; exit 1 ;;
esac

# OPSD-style ground-truth teacher (arXiv:2601.18734). When enabled, the
# teacher conditions on the dataset's ground-truth answer instead of a
# self-generated demo, giving 100% teacher-signal coverage.
GT_TEACHER="${GT_TEACHER:-0}"
case "$GT_TEACHER" in
    1|true|True|TRUE|yes|on)  GT_TEACHER_FLAG="--gt-teacher" ;;
    0|false|False|FALSE|no|off) GT_TEACHER_FLAG="--no-gt-teacher" ;;
    *) echo "GT_TEACHER must be 0/1" >&2; exit 1 ;;
esac

# OPSD reference uses max_grad_norm=0.1 (much tighter than our 1.0 default)
# and temperature=1.1 during rollouts (more diversity -> richer distillation).
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SAMPLING_TEMPERATURE="${SAMPLING_TEMPERATURE:-1.0}"

# OPSD per-token divergence clip. Their reference value is 0.05 for 1.7B/4B.
# Empty or 'none' disables (our prior behaviour).
TOKEN_CLIP="${TOKEN_CLIP:-}"
TOKEN_CLIP_FLAGS=()
if [[ -n "$TOKEN_CLIP" && "$TOKEN_CLIP" != "none" && "$TOKEN_CLIP" != "0" ]]; then
    TOKEN_CLIP_FLAGS=(--token-clip "$TOKEN_CLIP")
fi

# Optional reprompt-template overrides (set REPROMPT_TEMPLATE / SOLUTION_TEMPLATE
# to non-empty strings to override the SDPO defaults). The defaults live in
# cs336_alignment/sdpo.py.
EXTRA_REPROMPT_FLAGS=()
if [[ -n "${REPROMPT_TEMPLATE:-}" ]]; then
    EXTRA_REPROMPT_FLAGS+=(--reprompt-template "$REPROMPT_TEMPLATE")
fi
if [[ -n "${SOLUTION_TEMPLATE:-}" ]]; then
    EXTRA_REPROMPT_FLAGS+=(--solution-template "$SOLUTION_TEMPLATE")
fi

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
    $RTFD_FLAG \
    --min-demo-thinking-chars "$MIN_DEMO_THINKING_CHARS" \
    $PG_ALL_FLAG \
    $GT_TEACHER_FLAG \
    --grad-clip "$GRAD_CLIP" \
    --sampling-temperature "$SAMPLING_TEMPERATURE" \
    "${TOKEN_CLIP_FLAGS[@]}" \
    "${EXTRA_REPROMPT_FLAGS[@]}" \
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
