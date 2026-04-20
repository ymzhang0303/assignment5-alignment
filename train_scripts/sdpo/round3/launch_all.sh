#!/usr/bin/env bash
# Launch the 8 round-3 ablations, one per GPU, in the background.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"
LOG_DIR="$REPO_ROOT/logs/sdpo_round3"
mkdir -p "$LOG_DIR"

experiments=(
    "a_paper_default:0"
    "b_ema_strong:1"
    "c_lr_only:2"
    "d_ema_only:3"
    "e_grpo_mix:4"
    "f_grpo_dominant:5"
    "g_succ_1p0:6"
    "h_keep_thinking:7"
)

echo "launching ${#experiments[@]} round-3 experiments..."
for entry in "${experiments[@]}"; do
    name="${entry%%:*}"
    gpu="${entry##*:}"
    log="$LOG_DIR/$name.log"
    echo "  [gpu=$gpu] $name -> $log"
    nohup env DEVICE="cuda:$gpu" \
        bash "$DIR/$name.sh" \
        > "$log" 2>&1 &
    sleep 3  # small stagger so HF cache/tokenizer init doesn't thrash
done

echo
echo "all 8 launched. pids: $(jobs -p | tr '\n' ' ')"
echo "tail with:  tail -F $LOG_DIR/*.log"
wait
