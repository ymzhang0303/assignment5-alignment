#!/usr/bin/env bash
# Launch the 3 SDPO collapse-fix ablations in parallel on cuda:2/3/7.
# Logs go to logs/sdpo_ablations/<tag>.log; PIDs to pids.txt in the same dir.
set -euo pipefail
cd "$(dirname "$0")/../../.."

LOG_DIR="logs/sdpo_ablations"
mkdir -p "$LOG_DIR"
: > "$LOG_DIR/pids.txt"

# tag|device|script
JOBS=(
    "fix_a_keep_thinking|cuda:2|train_scripts/sdpo/ablations/fix_a_keep_thinking.sh"
    "fix_b_force_reason|cuda:3|train_scripts/sdpo/ablations/fix_b_force_reason.sh"
    "fix_c_both|cuda:7|train_scripts/sdpo/ablations/fix_c_both.sh"
)
for spec in "${JOBS[@]}"; do
    IFS="|" read -r tag dev script <<< "$spec"
    log="$LOG_DIR/${tag}.log"
    echo "[sdpo-abl] launching ${tag} on ${dev} -> $log"
    DEVICE="$dev" TAG="$tag" bash "$script" > "$log" 2>&1 &
    pid=$!
    echo "$pid $tag $dev" >> "$LOG_DIR/pids.txt"
    echo "[sdpo-abl]   pid=$pid"
done

echo
echo "[sdpo-abl] launched. PIDs:"
cat "$LOG_DIR/pids.txt"
echo "[sdpo-abl] tail with: tail -f $LOG_DIR/<tag>.log"
echo "[sdpo-abl] kill with: awk '{print \$1}' $LOG_DIR/pids.txt | xargs kill"
