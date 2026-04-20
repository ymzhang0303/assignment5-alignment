#!/usr/bin/env bash
# Round 6: close the 10-point gap to the GRPO baseline.
#
# GRPO baseline (runs/grpo_qwen3_bigmath_reinforce_bl):
#   init  val=0.348 chars=4016
#   peak  val=0.648 @ step 165
#   final val=0.606 @ step 199
#
# Our best SDPO so far (sdpo_r5_paper, pure SDPO, LR=1e-6, EMA=0.05):
#   init  val=0.285 chars=4099
#   peak  val=0.512 @ step 160
#   final val=0.508 @ step 199
#
# Key hypotheses for why SDPO underperforms GRPO on Qwen3-1.7B:
#
# (i)  Qwen3-1.7B is on the "weak" side of the paper's scaling study.
#      Paper Fig.11: Qwen3-0.6B benefits from SDPO+GRPO (lambda=0.9 GRPO,
#      0.1 SDPO). Our model sits between 0.6B and 8B so likely wants a
#      similar mix.
#
# (ii) With pure SDPO only ~70% of samples have a teacher signal; the
#      other 30% contribute zero gradient. Turning on a modest PG term
#      for those samples recovers the full effective batch.
#
# (iii) GRPO uses LR=1e-5. Our SDPO collapsed at LR=1e-5 without EMA,
#       but now WITH EMA teacher it may tolerate LR=1e-5 long enough to
#       match GRPO's trajectory.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/sdpo_r6

BASE_ENV=(
    N_GRPO_STEPS=200
    EVAL_EVERY=10
    SUCCESS_THRESHOLD=0.5
    REMOVE_THINKING_FROM_DEMO=1
    MIN_DEMO_THINKING_CHARS=0
    TEACHER_UPDATE_RATE=0.05
    GPU_MEM_UTIL=0.35
)

launch() {
    local gpu=$1 name=$2; shift 2
    local log=logs/sdpo_r6/${name}.log
    nohup env "${BASE_ENV[@]}" "$@" \
        DEVICE="cuda:$gpu" \
        OUTPUT_DIR="runs/sdpo_r6_${name}" \
        WANDB_RUN_NAME="sdpo_r6_${name}" \
        bash train_scripts/sdpo/run_sdpo.sh > "$log" 2>&1 < /dev/null &
    echo "[gpu $gpu] $name  pid=$!"
    sleep 3
}

# a: pure SDPO, paper recipe, LR=1e-5 + EMA=0.05. This combo collapsed in
#    round-3 at step 35 *without* EMA but might survive now. Best-case:
#    matches d_ema_only's peak 0.496 at much longer horizon.
launch 0 a_pure_lr1e5  LR=1e-5  PG_LOSS_WEIGHT=0.0  PG_APPLY_TO_ALL_SAMPLES=0

# b: SDPO + PG fill-in on samples without demos (default PG_APPLY_TO_ALL=0).
#    LR=1e-5, pg_weight=1.0. Recovers the 30% of samples pure SDPO drops.
launch 1 b_pgfill_lr1e5  LR=1e-5  PG_LOSS_WEIGHT=1.0  PG_APPLY_TO_ALL_SAMPLES=0

# c: SDPO + PG on ALL samples at GRPO LR. pg_weight=1 means distill and pg
#    get equal nominal scale, but distill_loss tends to dwarf pg numerically.
launch 2 c_pgall_lr1e5  LR=1e-5  PG_LOSS_WEIGHT=1.0  PG_APPLY_TO_ALL_SAMPLES=1

# d: GRPO-dominant blend (paper's lambda~=0.9 for weak models) at LR=1e-5.
#    pg_weight=10 roughly makes the PG term comparable to distill in magnitude.
launch 3 d_grpoheavy_lr1e5  LR=1e-5  PG_LOSS_WEIGHT=10.0  PG_APPLY_TO_ALL_SAMPLES=1

# e: same GRPO-dominant blend but at the stable LR=1e-6 -- slow but safe.
#    Tests whether adding heavy GRPO to the already-stable paper recipe
#    adds the missing signal.
launch 4 e_grpoheavy_lr1e6  LR=1e-6  PG_LOSS_WEIGHT=10.0  PG_APPLY_TO_ALL_SAMPLES=1

# f: extreme GRPO weight at LR=1e-6 (almost pure GRPO with a sprinkle of
#    SDPO as a regularizer). If this wins, the takeaway is "GRPO does the
#    work, SDPO is a cheap stabiliser".
launch 5 f_grpodomlight_lr1e6  LR=1e-6  PG_LOSS_WEIGHT=50.0  PG_APPLY_TO_ALL_SAMPLES=1

# g: extend winning recipe longer. If the 0.508 plateau is a true ceiling,
#    more steps won't help; if it's still climbing, we should see gains.
launch 6 g_longer_lr1e6  N_GRPO_STEPS=300  LR=1e-6  PG_LOSS_WEIGHT=0.0  PG_APPLY_TO_ALL_SAMPLES=0

# h: GRPO blend with stronger teacher regularisation. If the d/e runs drift,
#    a tighter EMA (0.02) should anchor longer.
launch 7 h_grpoheavy_strongema  LR=1e-5  TEACHER_UPDATE_RATE=0.02  PG_LOSS_WEIGHT=10.0  PG_APPLY_TO_ALL_SAMPLES=1

echo
echo "8 runs launched. Use status.py to monitor."
