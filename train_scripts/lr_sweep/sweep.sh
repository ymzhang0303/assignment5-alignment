#!/usr/bin/env bash
# Launch the 4-way Big-Math LR sweep on cuda:0..3.
#
# Each LR runs as a separate background `uv run` process. Logs land in
# logs/lr_sweep/<lr>.log so you can `tail -f` any one of them. PIDs are saved to
# logs/lr_sweep/pids.txt so it's easy to kill the whole sweep:
#
#   xargs -a logs/lr_sweep/pids.txt kill
#
# Override the GPU range or LR list via env, e.g.:
#   LRS="3e-6 1e-5" DEVICES="cuda:4 cuda:5" ./train_scripts/lr_sweep/sweep.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

LRS=(${LRS:-3e-6 1e-5 3e-5 1e-4})
DEVICES=(${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3})

if [[ ${#LRS[@]} -ne ${#DEVICES[@]} ]]; then
    echo "error: LRS (${#LRS[@]}) and DEVICES (${#DEVICES[@]}) must have the same length" >&2
    exit 1
fi

LOG_DIR="logs/lr_sweep"
mkdir -p "$LOG_DIR"
: > "$LOG_DIR/pids.txt"

for i in "${!LRS[@]}"; do
    lr="${LRS[$i]}"
    dev="${DEVICES[$i]}"
    script="train_scripts/lr_sweep/run_qwen3_bigMath_${lr}.sh"
    if [[ ! -x "$script" && ! -f "$script" ]]; then
        echo "error: missing wrapper $script" >&2
        exit 1
    fi
    log="$LOG_DIR/${lr}.log"
    echo "[sweep] launching LR=$lr on $dev -> $log"
    DEVICE="$dev" bash "$script" > "$log" 2>&1 &
    pid=$!
    echo "$pid" >> "$LOG_DIR/pids.txt"
    echo "[sweep]   pid=$pid"
done

echo
echo "[sweep] all 4 jobs launched. PIDs:"
cat "$LOG_DIR/pids.txt"
echo "[sweep] tail any log with: tail -f $LOG_DIR/<lr>.log"
echo "[sweep] kill the sweep with: xargs -a $LOG_DIR/pids.txt kill"
