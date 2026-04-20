#!/usr/bin/env bash
# Launch the off-policy broad sweep on cuda:0..3 (40 GRPO steps each).
#
# Spans 1x, 4x, 16x, 32x optimizer-updates-per-rollout, all with the same
# rollout_batch_size=256 and the same micro_train_batch_size=2 (memory
# constant). Logs land in logs/offpolicy/<tag>.log; PIDs in pids_broad.txt.
set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="logs/offpolicy"
mkdir -p "$LOG_DIR"
: > "$LOG_DIR/pids_broad.txt"

# tag|device|script
JOBS=(
    "e1_t256|cuda:0|train_scripts/offpolicy/run_e1_t256.sh"
    "e2_t128|cuda:1|train_scripts/offpolicy/run_e2_t128.sh"
    "e4_t64|cuda:2|train_scripts/offpolicy/run_e4_t64.sh"
    "e4_t32|cuda:3|train_scripts/offpolicy/run_e4_t32.sh"
)
for spec in "${JOBS[@]}"; do
    IFS="|" read -r tag dev script <<< "$spec"
    log="$LOG_DIR/${tag}_broad.log"
    echo "[offpolicy-broad] launching ${tag} on ${dev} -> $log"
    DEVICE="$dev" N_GRPO_STEPS=40 \
        OUTPUT_DIR="runs/grpo_qwen3_bigmath_offpolicy_${tag}_broad" \
        WANDB_RUN_NAME="grpo_qwen3_bigmath_offpolicy_${tag}_broad" \
        bash "$script" > "$log" 2>&1 &
    pid=$!
    echo "$pid" >> "$LOG_DIR/pids_broad.txt"
    echo "[offpolicy-broad]   pid=$pid"
done

echo
echo "[offpolicy-broad] launched. PIDs:"
cat "$LOG_DIR/pids_broad.txt"
echo "[offpolicy-broad] tail with: tail -f $LOG_DIR/<tag>_broad.log"
echo "[offpolicy-broad] kill with: xargs -a $LOG_DIR/pids_broad.txt kill"
