#!/usr/bin/env python
"""Dashboard for round-6 vs GRPO baseline comparison."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS_GLOB = str(REPO / "runs" / "sdpo_r6_*")
GRPO_BASELINE = 0.6055  # final val/answer_reward of grpo_qwen3_bigmath_reinforce_bl
GRPO_PEAK = 0.6484


def _fmt(v, spec=".3f"):
    if v is None:
        return "-"
    try:
        return format(v, spec)
    except Exception:
        return str(v)


def load(run_dir):
    mjson = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(mjson):
        return None
    lines = Path(mjson).read_text().strip().split("\n")
    last = None
    peak = None
    peak_step = None
    init_val = None
    history = []
    for ln in lines:
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        last = r
        if "val/answer_reward" in r:
            v = r["val/answer_reward"]
            s = r.get("step")
            history.append((s, v, r.get("val/response_chars")))
            if init_val is None:
                init_val = v
            if peak is None or v > peak:
                peak = v
                peak_step = s
    return {
        "step": last.get("step") if last else None,
        "init": init_val,
        "peak": peak,
        "peak_step": peak_step,
        "last_val": history[-1][1] if history else None,
        "last_chars": history[-1][2] if history else None,
        "history": history,
    }


def main() -> int:
    print(f"GRPO baseline target:  peak={GRPO_PEAK}  final={GRPO_BASELINE}\n")
    hdr = (
        f"{'run':<28} {'step':>4}  {'init':>6}  {'peak':>6}  "
        f"{'pkStp':>5}  {'last':>6}  {'chars':>6}  {'vsGRPO':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for run_dir in sorted(glob.glob(RUNS_GLOB)):
        name = os.path.basename(run_dir).replace("sdpo_r6_", "")
        d = load(run_dir)
        if d is None:
            print(f"{name:<28} (no metrics yet)")
            continue
        rows.append((name, d))
        vs = None
        if d["peak"] is not None:
            vs = d["peak"] - GRPO_BASELINE
        print(
            f"{name:<28} {d['step'] if d['step'] is not None else '-':>4}  "
            f"{_fmt(d['init']):>6}  {_fmt(d['peak']):>6}  "
            f"{d['peak_step'] if d['peak_step'] is not None else '-':>5}  "
            f"{_fmt(d['last_val']):>6}  {_fmt(d['last_chars'], '.0f'):>6}  "
            f"{_fmt(vs, '+.3f'):>8}"
        )
    # sort by peak
    rows.sort(key=lambda x: (x[1]["peak"] or 0), reverse=True)
    print("\n=== leader ==="  )
    if rows:
        lead_name, lead = rows[0]
        print(f"  {lead_name}  peak={_fmt(lead['peak'])}  "
              f"(delta vs GRPO peak {GRPO_PEAK}: "
              f"{_fmt((lead['peak'] or 0)-GRPO_PEAK, '+.3f')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
