#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY="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"
# ---- W&B config (override via env if you want) ----
# Online mode requires `wandb login` (or WANDB_API_KEY) once.
# Offline mode logs to disk -- sync later with `wandb sync runs/grpo_dapo/wandb/offline-run-*`.
export WANDB_PROJECT="${WANDB_PROJECT:-cs336-grpo}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-grpo_dapo}"
export WANDB_MODE="${WANDB_MODE:-online}"

uv run python -m cs336_alignment.grpo_train \
    --model-id /root/assignment5-alignment/models/Qwen2.5-Math-1.5B \
    --train-path data/dapo_math/train.jsonl \
    --val-path  data/dapo_math/validation.jsonl \
    --output-dir runs/grpo_dapo \
    --device "${DEVICE:-cuda:0}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.45}" \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_RUN_NAME" \
    --wandb-mode "$WANDB_MODE"
