#!/usr/bin/env bash
# Focused 200-step off-policy sweep on cuda:0..3. Reuses the same 4 configs
# as the broad sweep so we can extend each trajectory to 200 GRPO steps.
# Logs land in logs/offpolicy/<tag>_focused.log.
set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="logs/offpolicy"
mkdir -p "$LOG_DIR"
: > "$LOG_DIR/pids_focused.txt"

JOBS=(
    "e1_t256|cuda:0|train_scripts/offpolicy/run_e1_t256.sh"
    "e2_t128|cuda:1|train_scripts/offpolicy/run_e2_t128.sh"
    "e4_t64|cuda:2|train_scripts/offpolicy/run_e4_t64.sh"
    "e4_t32|cuda:3|train_scripts/offpolicy/run_e4_t32.sh"
)
for spec in "${JOBS[@]}"; do
    IFS="|" read -r tag dev script <<< "$spec"
    log="$LOG_DIR/${tag}_focused.log"
    echo "[offpolicy-focused] launching ${tag} on ${dev} -> $log"
    DEVICE="$dev" N_GRPO_STEPS=200 \
        OUTPUT_DIR="runs/grpo_qwen3_bigmath_offpolicy_${tag}_focused" \
        WANDB_RUN_NAME="grpo_qwen3_bigmath_offpolicy_${tag}_focused" \
        bash "$script" > "$log" 2>&1 &
    pid=$!
    echo "$pid" >> "$LOG_DIR/pids_focused.txt"
    echo "[offpolicy-focused]   pid=$pid"
done

echo
echo "[offpolicy-focused] launched. PIDs:"
cat "$LOG_DIR/pids_focused.txt"
