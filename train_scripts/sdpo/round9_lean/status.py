#!/usr/bin/env python
"""Round-9 lean-ablation status board.

Compares four single-knob SDPO ablations against the GRPO baseline and
the r6_d recipe they build on. Scores by the smoothed tail-mean of the
last 10 val evaluations (matches how we compared GRPO in FINDINGS).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

GRPO_BASELINE = REPO / "runs" / "grpo_qwen3_bigmath_reinforce_bl" / "metrics.jsonl"
GRPO_TAIL_MEAN_TARGET = 0.622  # smoothed last-10 evals for GRPO reinforce_bl
GRPO_PEAK = 0.648

REFERENCE_RUNS = [
    ("grpo_baseline (target)", REPO / "runs" / "grpo_qwen3_bigmath_reinforce_bl"),
    ("r6_d (prev SDPO best)",  REPO / "runs" / "sdpo_r6_d_grpoheavy_lr1e5"),
    ("r6_h (strong EMA)",      REPO / "runs" / "sdpo_r6_h_grpoheavy_strongema"),
]

R9_RUNS = [
    "lean_A_fwdkl",
    "lean_B_nostd",
    "lean_C_advmask",
    "combo_all3",
    "pgw5_advmask",
    "lean_AC_fwdkl_advmask",
    "drgrpo_proper",
    "combo_BC_nostd_advmask",
    "lean_C_tight_ema",
    "lean_C_tokenclip",
    "pgw3_advmask",
    "advmask_2ep",
    "strict_demo_advmask",
    "pgw5_tight_ema",
    "pgw5_nostd_advmask",
]


def load_val_rows(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "val/answer_reward" in r:
            out.append(r)
    return out


def summarize(rows: list[dict], tail: int = 10) -> dict | None:
    if not rows:
        return None
    ar = [r["val/answer_reward"] for r in rows]
    peak = max(ar)
    peak_step = rows[ar.index(peak)]["step"]
    last = ar[-1]
    last_step = rows[-1]["step"]
    tail_ar = ar[-tail:]
    tail_mean = sum(tail_ar) / len(tail_ar)

    chars = [r.get("val/response_chars") for r in rows if "val/response_chars" in r]
    fmt = [r.get("val/format_reward") for r in rows if "val/format_reward" in r]
    last_chars = chars[-1] if chars else None
    last_fmt = fmt[-1] if fmt else None
    tail_chars = sum(chars[-tail:]) / max(1, len(chars[-tail:])) if chars else None
    tail_fmt = sum(fmt[-tail:]) / max(1, len(fmt[-tail:])) if fmt else None

    return {
        "n_evals": len(rows),
        "last_step": int(last_step),
        "last": last,
        "peak": peak,
        "peak_step": int(peak_step),
        "tail_mean": tail_mean,
        "tail_chars": tail_chars,
        "tail_fmt": tail_fmt,
        "last_chars": last_chars,
        "last_fmt": last_fmt,
    }


def fmt_row(name: str, s: dict | None, baseline: float | None) -> str:
    if s is None:
        return f"  {name:32s}  (no val yet)"
    delta = s["tail_mean"] - baseline if baseline is not None else None
    delta_s = f"{delta:+.3f}" if delta is not None else "  -  "
    tc = f"{int(s['tail_chars']):5d}" if s["tail_chars"] is not None else "  -  "
    tf = f"{s['tail_fmt']:.3f}" if s["tail_fmt"] is not None else "  -  "
    return (
        f"  {name:32s}  step {int(s['last_step']):4d}  "
        f"peak {s['peak']:.3f}@{int(s['peak_step']):3d}  "
        f"last {s['last']:.3f}  "
        f"tail10 {s['tail_mean']:.3f}  "
        f"d_vs_grpo {delta_s}  "
        f"chars {tc}  fmt {tf}"
    )


def main() -> None:
    grpo_rows = load_val_rows(GRPO_BASELINE)
    grpo_s = summarize(grpo_rows)
    baseline_tail = grpo_s["tail_mean"] if grpo_s else GRPO_TAIL_MEAN_TARGET

    print(
        f"GRPO target (smoothed tail-10 of last 10 evals): {baseline_tail:.3f}  "
        f"(peak {GRPO_PEAK})"
    )
    print()
    print("=== reference runs ===")
    for name, path in REFERENCE_RUNS:
        s = summarize(load_val_rows(path / "metrics.jsonl"))
        print(fmt_row(name, s, baseline_tail))

    print()
    print("=== Round 9 ablations ===")
    any_winning = False
    any_promising = False
    for name in R9_RUNS:
        s = summarize(load_val_rows(REPO / "runs" / f"sdpo_r9_{name}" / "metrics.jsonl"))
        print(fmt_row(name, s, baseline_tail))
        if s is None:
            continue
        # Winning: beats smoothed GRPO with chars <= 2500 and fmt >= 0.83.
        chars_ok = s["tail_chars"] is None or s["tail_chars"] <= 2500
        fmt_ok = s["tail_fmt"] is None or s["tail_fmt"] >= 0.83
        if s["tail_mean"] >= baseline_tail and chars_ok and fmt_ok:
            any_winning = True
        # Promising: beats r6_d's 0.607 smoothed.
        if s["tail_mean"] >= 0.607:
            any_promising = True

    print()
    print(
        f"  winning (>= {baseline_tail:.3f} smoothed, chars<=2500, fmt>=0.83): "
        f"{'YES' if any_winning else 'no'}"
    )
    print(f"  promising (>= 0.607 smoothed, beats r6_d): {'YES' if any_promising else 'no'}")


if __name__ == "__main__":
    main()
