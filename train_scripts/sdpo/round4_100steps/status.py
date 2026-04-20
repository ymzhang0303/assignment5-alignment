#!/usr/bin/env python
"""Dashboard for round-4 SDPO 100-step runs."""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS_GLOB = str(REPO / "runs" / "sdpo_r4_*")


def _fmt(v, spec=".3f"):
    if v is None:
        return "-"
    try:
        return format(v, spec)
    except Exception:
        return str(v)


def main() -> int:
    rows = []
    for run_dir in sorted(glob.glob(RUNS_GLOB)):
        name = os.path.basename(run_dir)
        mjson = os.path.join(run_dir, "metrics.jsonl")
        if not os.path.exists(mjson):
            rows.append((name, 0, None, None, None, None, None, None))
            continue
        lines = Path(mjson).read_text().strip().split("\n")
        last = None
        last_val = None
        peak_val = None
        peak_step = None
        init_val = None
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
                last_val = r
                if init_val is None:
                    init_val = v
                if peak_val is None or v > peak_val:
                    peak_val = v
                    peak_step = r.get("step")
        step = last.get("step") if last else None
        last_va = last_val.get("val/answer_reward") if last_val else None
        last_vc = last_val.get("val/response_chars") if last_val else None
        rows.append((name, step, init_val, last_va, peak_val, peak_step, last_vc, None))

    hdr = f"{'run':<28} {'step':>4}  {'init':>6}  {'last':>6}  {'peak':>6}  {'pkStp':>5}  {'chars':>6}"
    print(hdr)
    print("-" * len(hdr))
    for (name, step, iv, va, pv, ps, vc, _) in rows:
        print(
            f"{name:<28} {step if step is not None else '-':>4}  "
            f"{_fmt(iv):>6}  {_fmt(va):>6}  {_fmt(pv):>6}  "
            f"{ps if ps is not None else '-':>5}  {_fmt(vc, '.0f'):>6}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
