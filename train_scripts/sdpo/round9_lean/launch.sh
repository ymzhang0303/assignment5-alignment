#!/usr/bin/env bash
# Round 9: lean ablations on top of the r6_d recipe.
#
# Baseline to beat (smoothed last-10 evals, steps ~100-199):
#   GRPO reinforce_bl:            0.622 (peak 0.648 / final 0.606)
#   sdpo_r6_d_grpoheavy_lr1e5:    0.607 (gap -0.015 vs GRPO)
#   sdpo_r6_h_grpoheavy_strongema:0.588 (gap -0.034 vs GRPO)
#
# r6_h does NOT beat GRPO when smoothed; pg_w=10 means PG is ~94% of the
# total loss magnitude and train/grad_norm is 7-9x GRPO's. The distill
# loss is effectively a vestigial regulariser.
#
# Four single-knob changes vs the r6_d recipe, 200 steps each, comparable
# to r6_d / GRPO (SAMPLING_MAX_TOKENS=1536, LR=1e-5, tau=0.05, pg_w=10,
# pg_all=1):
#   A fwdkl      SDPO_ALPHA=0.0  (forward KL instead of JSD)
#   B nostd      USE_STD_NORMALIZATION=0  (Dr.GRPO advantage fix)
#   C advmask    ADV_MASK_DISTILL=1  (distill only on positive-advantage samples)
#   D lenpenalty LENGTH_PENALTY=5e-5  (Dr.GRPO-style length regulariser)
#
# Reserved GPUs: 0, 3, 6, 7 (GPU 4 is busy with r7_f_tokenclip, still climbing).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/sdpo_r9

# Shared recipe = r6_d (GRPO-heavy SDPO blend at LR=1e-5, EMA=0.05).
R6D_COMMON_ENV=(
    N_GRPO_STEPS=200
    EVAL_EVERY=10
    EVAL_EXAMPLES=256
    SAMPLING_MAX_TOKENS=1536
    GPU_MEM_UTIL=0.45
    LR=1e-5
    TEACHER_UPDATE_RATE=0.05
    SUCCESS_THRESHOLD=0.5
    REMOVE_THINKING_FROM_DEMO=1
    MIN_DEMO_THINKING_CHARS=0
    PG_LOSS_WEIGHT=10.0
    PG_APPLY_TO_ALL_SAMPLES=1
)

launch() {
    local gpu=$1 name=$2; shift 2
    local log=logs/sdpo_r9/${name}.log
    nohup env "${R6D_COMMON_ENV[@]}" "$@" \
        DEVICE="cuda:$gpu" \
        OUTPUT_DIR="runs/sdpo_r9_${name}" \
        WANDB_RUN_NAME="sdpo_r9_${name}" \
        bash train_scripts/sdpo/run_sdpo.sh > "$log" 2>&1 < /dev/null &
    echo "[gpu $gpu] sdpo_r9_${name}  pid=$!  log=$log"
    sleep 3
}

# A: forward-KL.
launch  0  lean_A_fwdkl       SDPO_ALPHA=0.0

# B: Dr.GRPO advantage fix (no group-std normalisation).
launch  3  lean_B_nostd       USE_STD_NORMALIZATION=0

# C: advantage-masked distillation.
launch  6  lean_C_advmask     ADV_MASK_DISTILL=1

# D: length-penalty on PG reward (5e-5 per token).
launch  7  lean_D_lenpenalty  LENGTH_PENALTY=5e-5

echo
echo "4 runs launched on cuda:{0,3,6,7}. Monitor with:"
echo "  python train_scripts/sdpo/round9_lean/status.py"
