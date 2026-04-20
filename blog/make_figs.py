"""Generate figures for the LR-sweep blog post.

All figures are written to blog/figs/ as PNGs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "1e-4": REPO_ROOT / "runs/grpo_qwen3_bigmath_lr1e-4/metrics.jsonl",
    "3e-5": REPO_ROOT / "runs/grpo_qwen3_bigmath_lr3e-5/metrics.jsonl",
    "1e-5": REPO_ROOT / "runs/grpo_qwen3_bigmath_lr1e-5/metrics.jsonl",
    "3e-6": REPO_ROOT / "runs/grpo_qwen3_bigmath_lr3e-6/metrics.jsonl",
}
ORDER = ["3e-6", "1e-5", "3e-5", "1e-4"]
COLORS = {
    "3e-6": "#4c72b0",
    "1e-5": "#55a868",
    "3e-5": "#c44e52",
    "1e-4": "#8172b2",
}
OUT = REPO_ROOT / "blog/figs"
OUT.mkdir(parents=True, exist_ok=True)


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open()]


def col(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] is not None and not (
            isinstance(r[key], float) and np.isnan(r[key])
        ):
            xs.append(r["step"])
            ys.append(r[key])
    return xs, ys


def smooth(ys: list[float], k: int = 5) -> list[float]:
    if k <= 1 or len(ys) < k:
        return list(ys)
    a = np.asarray(ys, dtype=float)
    out = np.convolve(a, np.ones(k) / k, mode="valid")
    pad = [a[: i + 1].mean() for i in range(k - 1)]
    return pad + out.tolist()


DATA = {lr: load(p) for lr, p in RUNS.items()}

# Style
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---- Fig 1: val/answer_reward over steps -------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.0))
for lr in ORDER:
    xs, ys = col(DATA[lr], "val/answer_reward")
    if not xs:
        continue
    ax.plot(xs, ys, marker="o", ms=3, lw=1.6, color=COLORS[lr], label=f"lr={lr}")
ax.set_xlabel("GRPO step")
ax.set_ylabel("val / answer_reward")
ax.set_title("Validation accuracy on Big-Math (256 problems, eval every 5 steps)")
ax.set_ylim(-0.02, 0.75)
ax.legend(loc="lower right", frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "val_reward.png")
plt.close(fig)

# ---- Fig 2: val/response_chars -----------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.0))
for lr in ORDER:
    xs, ys = col(DATA[lr], "val/response_chars")
    if not xs:
        continue
    ax.plot(xs, ys, marker="o", ms=3, lw=1.6, color=COLORS[lr], label=f"lr={lr}")
ax.axhline(1536, color="gray", ls=":", lw=1, label="max-token cap (1536)")
ax.set_xlabel("GRPO step")
ax.set_ylabel("avg val response length (chars)")
ax.set_title("Length collapse vs LR")
ax.legend(loc="upper right", frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "val_response_chars.png")
plt.close(fig)

# ---- Fig 3: 2x2 train dynamics -----------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
panels = [
    ("train/token_entropy",   "policy entropy (avg over response tokens)"),
    ("train/grad_norm",       "grad norm (post-clip)"),
    ("train/format_reward_mean", "train format reward"),
    ("train/reward_mean",     "train reward (= answer reward, binary)"),
]
for ax, (key, title) in zip(axes.flat, panels):
    for lr in ORDER:
        xs, ys = col(DATA[lr], key)
        if not xs:
            continue
        ax.plot(xs, smooth(ys, 5), lw=1.4, color=COLORS[lr], label=f"lr={lr}")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if key == "train/grad_norm":
        ax.axhline(1.0, color="gray", ls=":", lw=1)
        ax.set_ylim(0, 2.0)
for ax in axes[-1]:
    ax.set_xlabel("GRPO step")
axes[0, 0].legend(loc="upper right", frameon=False)
fig.suptitle("Training dynamics across the LR sweep", fontsize=12, y=1.0)
fig.tight_layout()
fig.savefig(OUT / "train_dynamics.png")
plt.close(fig)

# ---- Fig 4: format vs answer divergence at val -------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.0))
for lr in ORDER:
    xs_a, ys_a = col(DATA[lr], "val/answer_reward")
    xs_f, ys_f = col(DATA[lr], "val/format_reward")
    if not xs_a:
        continue
    ax.plot(xs_f, ys_f, lw=1.5, color=COLORS[lr], label=f"format  lr={lr}")
    ax.plot(xs_a, ys_a, lw=1.5, color=COLORS[lr], ls="--",
            label=f"answer  lr={lr}")
ax.set_xlabel("GRPO step")
ax.set_ylabel("val reward")
ax.set_title("Format reward (solid) is learned faster than answer reward (dashed)")
ax.set_ylim(-0.02, 1.0)
ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "format_vs_answer.png")
plt.close(fig)

# ---- Fig 5: 1e-4 collapse zoom (steps 0..15) ---------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.0))
rows = DATA["1e-4"]
zoom_keys = [
    ("train/token_entropy", "entropy",     "#c44e52"),
    ("train/grad_norm",     "grad_norm",   "#4c72b0"),
    ("train/reward_mean",   "reward_mean", "#55a868"),
    ("train/format_reward_mean", "format_reward", "#8172b2"),
]
for key, label, color in zoom_keys:
    xs, ys = col(rows, key)
    xs_ys = [(x, y) for x, y in zip(xs, ys) if x <= 15]
    if not xs_ys:
        continue
    xs2, ys2 = zip(*xs_ys)
    ax.plot(xs2, ys2, marker="o", ms=4, lw=1.5, color=color, label=label)
ax.axvline(5, color="gray", ls=":", lw=1)
ax.text(5.2, 0.85, "rewards collapse →", color="gray", fontsize=8)
ax.axvline(10, color="gray", ls=":", lw=1)
ax.text(10.2, 0.85, "NaN", color="gray", fontsize=8)
ax.set_xlabel("GRPO step")
ax.set_ylabel("value")
ax.set_title("1e-4 collapse (entropy spike → reward → 0 → NaN)")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "lr1e4_collapse.png")
plt.close(fig)

# ---- Fig 6: baseline vs no-baseline (loss_type ablation, on-policy) ----------
BL_RUNS = {
    "reinforce_with_baseline": REPO_ROOT / "runs/grpo_qwen3_bigmath_reinforce_bl/metrics.jsonl",
    "no_baseline":             REPO_ROOT / "runs/grpo_qwen3_bigmath_no_baseline/metrics.jsonl",
}
BL_COLORS = {
    "reinforce_with_baseline": "#55a868",
    "no_baseline":             "#c44e52",
}
BL_DATA = {k: load(p) for k, p in BL_RUNS.items()}

fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
bl_panels = [
    ("val/answer_reward",          "val answer reward",            False, None),
    ("val/format_reward",          "val format reward",            False, None),
    ("val/response_chars",         "val response length (chars)",  False, 1536),
    ("train/grad_norm",            "train grad_norm (post-clip)",  True,  1.0),
    ("train/group_reward_std_mean","group reward std (signal)",    True,  None),
    ("train/token_entropy",        "policy entropy",               True,  None),
]
for ax, (key, title, do_smooth, hline) in zip(axes.flat, bl_panels):
    for name, rows in BL_DATA.items():
        xs, ys = col(rows, key)
        if not xs:
            continue
        ys_plot = smooth(ys, 5) if do_smooth else ys
        ax.plot(xs, ys_plot, lw=1.6, color=BL_COLORS[name], label=name)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if hline is not None:
        ax.axhline(hline, color="gray", ls=":", lw=1)
    if key == "train/grad_norm":
        ax.set_yscale("log")
for ax in axes[-1]:
    ax.set_xlabel("GRPO step")
axes[0, 0].legend(loc="lower right", frameon=False, fontsize=8)
fig.suptitle("Baseline vs no-baseline at lr=1e-5 (everything else identical)",
             fontsize=12, y=1.0)
fig.tight_layout()
fig.savefig(OUT / "baseline_compare.png")
plt.close(fig)


# ---- Fig 7: length normalization (masked_mean vs Dr-GRPO masked_normalize) ---
LN_RUNS = {
    "masked_mean (vanilla GRPO)":       REPO_ROOT / "runs/grpo_qwen3_bigmath_reinforce_bl/metrics.jsonl",
    "masked_normalize (Dr-GRPO, L=1536)": REPO_ROOT / "runs/grpo_qwen3_bigmath_reinforce_bl_drnorm/metrics.jsonl",
}
LN_COLORS = {
    "masked_mean (vanilla GRPO)":         "#55a868",
    "masked_normalize (Dr-GRPO, L=1536)": "#4c72b0",
}
LN_DATA = {k: load(p) for k, p in LN_RUNS.items()}

fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
ln_panels = [
    ("val/answer_reward",          "val answer reward",            False, None),
    ("val/response_chars",         "val response length (chars)",  False, 1536),
    ("train/grad_norm",            "train grad_norm (post-clip)",  True,  1.0),
    ("train/loss",                 "train loss (per micro-batch)", True,  0.0),
    ("train/token_entropy",        "policy entropy",               True,  None),
    ("train/group_reward_std_mean","group reward std (signal)",    True,  None),
]
for ax, (key, title, do_smooth, hline) in zip(axes.flat, ln_panels):
    for name, rows in LN_DATA.items():
        xs, ys = col(rows, key)
        if not xs:
            continue
        ys_plot = smooth(ys, 5) if do_smooth else ys
        ax.plot(xs, ys_plot, lw=1.6, color=LN_COLORS[name], label=name)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if hline is not None:
        ax.axhline(hline, color="gray", ls=":", lw=1)
for ax in axes[-1]:
    ax.set_xlabel("GRPO step")
axes[0, 0].legend(loc="lower right", frameon=False, fontsize=8)
fig.suptitle(
    "Length normalization at lr=1e-5 + reinforce_with_baseline (everything else identical)",
    fontsize=12, y=1.0,
)
fig.tight_layout()
fig.savefig(OUT / "length_norm_compare.png")
plt.close(fig)


# ---- Fig 8: std-norm ON vs OFF (advantage normalization ablation) ------------
SN_RUNS = {
    "use_std_normalization=True":  REPO_ROOT / "runs/grpo_qwen3_bigmath_reinforce_bl/metrics.jsonl",
    "use_std_normalization=False": REPO_ROOT / "runs/grpo_qwen3_bigmath_reinforce_bl_no_std/metrics.jsonl",
}
SN_COLORS = {
    "use_std_normalization=True":  "#55a868",
    "use_std_normalization=False": "#dd8452",
}
SN_DATA = {k: load(p) for k, p in SN_RUNS.items()}

fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
sn_panels = [
    ("val/answer_reward",          "val answer reward",            False, None),
    ("val/response_chars",         "val response length (chars)",  False, 1536),
    ("train/grad_norm",            "train grad_norm (post-clip)",  True,  1.0),
    ("train/token_entropy",        "policy entropy",               True,  None),
    ("train/group_reward_std_mean","group reward std (signal)",    True,  None),
    ("train/loss",                 "train loss (per micro-batch)", True,  0.0),
]
for ax, (key, title, do_smooth, hline) in zip(axes.flat, sn_panels):
    for name, rows in SN_DATA.items():
        xs, ys = col(rows, key)
        if not xs:
            continue
        ys_plot = smooth(ys, 5) if do_smooth else ys
        ax.plot(xs, ys_plot, lw=1.6, color=SN_COLORS[name], label=name)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if hline is not None:
        ax.axhline(hline, color="gray", ls=":", lw=1)
for ax in axes[-1]:
    ax.set_xlabel("GRPO step")
axes[0, 0].legend(loc="lower right", frameon=False, fontsize=8)
fig.suptitle(
    "Group-std normalization on vs off at lr=1e-5 + reinforce_with_baseline",
    fontsize=12, y=1.0,
)
fig.tight_layout()
fig.savefig(OUT / "std_norm_compare.png")
plt.close(fig)


print("wrote:")
for f in sorted(OUT.glob("*.png")):
    print(" ", f.relative_to(REPO_ROOT))
