#!/usr/bin/env bash
# Launch the 2-way loss-type ablation (reinforce_with_baseline vs no_baseline)
# in parallel on cuda:4 / cuda:5. Both runs use LR=1e-5 (best from the LR
# sweep) so the only knob that varies is loss_type. The grader was patched
# to fold Unicode math glyphs before this launch, so Unicode answers like
# "1.656×10⁶" no longer get spuriously zero reward.
#
# Logs: logs/loss_type/{reinforce_bl,no_baseline}.log
# PIDs: logs/loss_type/pids.txt
#   xargs -a logs/loss_type/pids.txt kill   # to stop the whole ablation
set -euo pipefail

cd "$(dirname "$0")/../.."

LOG_DIR="logs/loss_type"
mkdir -p "$LOG_DIR"
: > "$LOG_DIR/pids.txt"

# name|device|script
JOBS=(
    "reinforce_bl|cuda:4|train_scripts/loss_type/run_reinforce_baseline.sh"
    "no_baseline|cuda:5|train_scripts/loss_type/run_no_baseline.sh"
)

for spec in "${JOBS[@]}"; do
    IFS="|" read -r name dev script <<< "$spec"
    log="$LOG_DIR/${name}.log"
    echo "[loss_type] launching ${name} on ${dev} -> $log"
    DEVICE="$dev" bash "$script" > "$log" 2>&1 &
    pid=$!
    echo "$pid" >> "$LOG_DIR/pids.txt"
    echo "[loss_type]   pid=$pid"
done

echo
echo "[loss_type] launched. PIDs:"
cat "$LOG_DIR/pids.txt"
echo "[loss_type] tail any log with: tail -f $LOG_DIR/<name>.log"
echo "[loss_type] kill the ablation with: xargs -a $LOG_DIR/pids.txt kill"
