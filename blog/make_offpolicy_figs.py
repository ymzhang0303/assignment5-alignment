"""Generate figures for the off-policy GRPO sweep blog post.

Loads metrics.jsonl from runs/grpo_qwen3_bigmath_offpolicy_<tag>_<phase>/
and produces per-phase figures comparing val reward, clip fraction,
entropy, and response length -- both vs GRPO step and vs wall-clock time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "blog/figs"

# (tag, color, label, updates_per_rollout)
CONFIGS_BROAD = [
    ("e1_t256", "#4c72b0", "E=1, T=256  (1 upd/rollout, on-policy)", 1),
    ("e2_t128", "#55a868", "E=2, T=128  (4 upd/rollout)",            4),
    ("e4_t64",  "#dd8452", "E=4, T=64   (16 upd/rollout)",           16),
    ("e4_t32",  "#c44e52", "E=4, T=32   (32 upd/rollout)",           32),
]


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open()]


def col(rows: list[dict], key: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] is not None and not (
            isinstance(r[key], float) and np.isnan(r[key])
        ):
            xs.append(r["step"])
            ys.append(r[key])
    return xs, ys


def cumtime(rows: list[dict]) -> list[float]:
    """Cumulative wall-clock seconds per logged step."""
    out, t = [], 0.0
    for r in rows:
        # time/step_s is the per-step rollout+train time; time/eval_s is added
        # only on eval steps.
        dt = float(r.get("time/step_s", 0.0)) + float(r.get("time/eval_s", 0.0))
        t += dt
        out.append(t / 60.0)  # minutes
    return out


def smooth(ys: list[float], k: int = 3) -> list[float]:
    if k <= 1 or len(ys) < k:
        return list(ys)
    a = np.asarray(ys, dtype=float)
    out = np.convolve(a, np.ones(k) / k, mode="valid")
    pad = [a[: i + 1].mean() for i in range(k - 1)]
    return pad + out.tolist()


def make_figs(phase: str, configs: list[tuple]) -> None:
    out = OUT_ROOT / f"offpolicy_{phase}"
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 160,
        "font.size": 10, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    runs = {}
    for tag, *_ in configs:
        p = REPO_ROOT / f"runs/grpo_qwen3_bigmath_offpolicy_{tag}_{phase}/metrics.jsonl"
        runs[tag] = load(p)
    print(f"[{phase}] row counts:", {t: len(r) for t, r in runs.items()})

    # ---- Fig 1: val/answer_reward vs step ----
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for tag, color, label, _ in configs:
        rows = runs[tag]
        xs, ys = col(rows, "val/answer_reward")
        if not xs: continue
        ax.plot(xs, ys, marker="o", ms=3, lw=1.6, color=color, label=label)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("val / answer_reward")
    ax.set_title(f"Off-policy sweep ({phase}): validation accuracy vs step")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "val_reward_vs_step.png"); plt.close(fig)

    # ---- Fig 2: val/answer_reward vs wall-clock ----
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for tag, color, label, _ in configs:
        rows = runs[tag]
        if not rows: continue
        t_min = cumtime(rows)
        # Pair eval values with their cumulative time
        t_eval = [t_min[i] for i, r in enumerate(rows) if "val/answer_reward" in r]
        ys = [r["val/answer_reward"] for r in rows if "val/answer_reward" in r]
        if not t_eval: continue
        ax.plot(t_eval, ys, marker="o", ms=3, lw=1.6, color=color, label=label)
    ax.set_xlabel("wall-clock time (min)")
    ax.set_ylabel("val / answer_reward")
    ax.set_title(f"Off-policy sweep ({phase}): validation accuracy vs wall-clock")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "val_reward_vs_walltime.png"); plt.close(fig)

    # ---- Fig 3: 2x2 dynamics ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [
        ("train/clip_fraction",  "GRPO-clip fraction (per microbatch)"),
        ("train/token_entropy",  "policy entropy (avg over response tokens)"),
        ("train/grad_norm",      "grad norm (post-clip)"),
        ("val/response_chars",   "val avg response length (chars)"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        for tag, color, label, _ in configs:
            xs, ys = col(runs[tag], key)
            if not xs: continue
            ax.plot(xs, smooth(ys, 3), lw=1.4, color=color, label=label)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if key == "train/grad_norm": ax.axhline(1.0, color="gray", ls=":", lw=1)
        if key == "val/response_chars": ax.axhline(1536, color="gray", ls=":", lw=1)
    for ax in axes[-1]:
        ax.set_xlabel("GRPO step")
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle(f"Off-policy sweep ({phase}): training dynamics", fontsize=12, y=1.0)
    fig.tight_layout(); fig.savefig(out / "dynamics.png"); plt.close(fig)

    # ---- Fig 4: optimizer updates vs val accuracy (efficiency view) ----
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for tag, color, label, upr in configs:
        rows = runs[tag]
        if not rows: continue
        xs_eval, ys = [], []
        for r in rows:
            if "val/answer_reward" in r:
                xs_eval.append(r["step"] * upr)  # cumulative optimizer updates
                ys.append(r["val/answer_reward"])
        if not xs_eval: continue
        ax.plot(xs_eval, ys, marker="o", ms=3, lw=1.6, color=color, label=label)
    ax.set_xlabel("cumulative optimizer updates")
    ax.set_ylabel("val / answer_reward")
    ax.set_title(f"Off-policy sweep ({phase}): val acc vs total optimizer updates")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "val_reward_vs_updates.png"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["broad", "focused", "both"], default="both")
    args = ap.parse_args()

    if args.phase in ("broad", "both"):
        make_figs("broad", CONFIGS_BROAD)
    if args.phase in ("focused", "both"):
        # Focused configs are decided after the broad sweep; populated below.
        configs_focused_path = REPO_ROOT / "blog/_focused_configs.json"
        if configs_focused_path.exists():
            cfg = json.loads(configs_focused_path.read_text())
            CONFIGS_FOCUSED = [(c["tag"], c["color"], c["label"], c["upr"]) for c in cfg]
            make_figs("focused", CONFIGS_FOCUSED)
        else:
            print("focused configs not yet picked; skipping focused phase")


if __name__ == "__main__":
    main()
