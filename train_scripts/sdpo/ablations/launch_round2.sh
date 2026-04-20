#!/usr/bin/env bash
# Round 2 of SDPO collapse-fix ablations -- structural fixes (PG mix +
# demo-quality filter) on top of round 1's best (fix_b force-reason
# template). Logs land in logs/sdpo_ablations/<tag>.log; PIDs in
# pids_round2.txt in the same directory.
set -euo pipefail
cd "$(dirname "$0")/../../.."

LOG_DIR="logs/sdpo_ablations"
mkdir -p "$LOG_DIR"
: > "$LOG_DIR/pids_round2.txt"

# tag|device|script
JOBS=(
    "fix_d_pg_mix|cuda:2|train_scripts/sdpo/ablations/fix_d_pg_mix.sh"
    "fix_e_pg_all|cuda:3|train_scripts/sdpo/ablations/fix_e_pg_all.sh"
    "fix_f_min_demo|cuda:7|train_scripts/sdpo/ablations/fix_f_min_demo.sh"
)
for spec in "${JOBS[@]}"; do
    IFS="|" read -r tag dev script <<< "$spec"
    log="$LOG_DIR/${tag}.log"
    echo "[sdpo-abl-r2] launching ${tag} on ${dev} -> $log"
    DEVICE="$dev" TAG="$tag" bash "$script" > "$log" 2>&1 &
    pid=$!
    echo "$pid $tag $dev" >> "$LOG_DIR/pids_round2.txt"
    echo "[sdpo-abl-r2]   pid=$pid"
done

echo
echo "[sdpo-abl-r2] launched. PIDs:"
cat "$LOG_DIR/pids_round2.txt"
echo "[sdpo-abl-r2] tail with: tail -f $LOG_DIR/<tag>.log"
echo "[sdpo-abl-r2] kill with: awk '{print \$1}' $LOG_DIR/pids_round2.txt | xargs kill"
