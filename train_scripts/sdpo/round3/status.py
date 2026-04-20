#!/usr/bin/env python
"""Quick status dashboard for round-3 SDPO ablations."""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS_GLOB = str(REPO / "runs" / "sdpo_r3_*")


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
            rows.append((name, 0, None, None, None, None, None, None, None))
            continue
        lines = Path(mjson).read_text().strip().split("\n")
        last = None
        last_val = None
        peak_val = None
        peak_val_step = None
        for ln in lines:
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            last = r
            if "val/answer_reward" in r:
                last_val = r
                v = r["val/answer_reward"]
                if peak_val is None or v > peak_val:
                    peak_val = v
                    peak_val_step = r.get("step")
        step = last.get("step") if last else None
        last_va = last_val.get("val/answer_reward") if last_val else None
        last_vc = last_val.get("val/response_chars") if last_val else None
        last_pg = last.get("sdpo/pg_loss") if last else None
        last_dl = last.get("sdpo/distill_loss") if last else None
        demo_frac = last.get("sdpo/sample_with_demo_fraction") if last else None
        rows.append(
            (name, step, last_va, last_vc, peak_val, peak_val_step, demo_frac, last_pg, last_dl)
        )

    # header
    hdr = (
        f"{'run':<26} {'step':>4}  "
        f"{'lastVal':>8}  {'lastCh':>7}  "
        f"{'peakVal':>8}  {'pkStp':>5}  {'demo%':>6}  "
        f"{'pgL':>8}  {'dlL':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for (name, step, va, vc, pv, pvs, demo, pg, dl) in rows:
        print(
            f"{name:<26} {step if step is not None else '-':>4}  "
            f"{_fmt(va):>8}  {_fmt(vc, '.0f'):>7}  "
            f"{_fmt(pv):>8}  {pvs if pvs is not None else '-':>5}  "
            f"{_fmt(demo*100 if demo is not None else None, '.1f'):>6}  "
            f"{_fmt(pg, '.4f'):>8}  {_fmt(dl, '.4f'):>8}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
