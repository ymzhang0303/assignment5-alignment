#!/usr/bin/env python
"""Dashboard for round-5 SDPO 200-step runs."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS_GLOB = str(REPO / "runs" / "sdpo_r5_*")


def _fmt(v, spec=".3f"):
    if v is None:
        return "-"
    try:
        return format(v, spec)
    except Exception:
        return str(v)


def main() -> int:
    for run_dir in sorted(glob.glob(RUNS_GLOB)):
        name = os.path.basename(run_dir)
        mjson = os.path.join(run_dir, "metrics.jsonl")
        if not os.path.exists(mjson):
            print(f"{name}: no metrics yet")
            continue
        lines = Path(mjson).read_text().strip().split("\n")
        last = None
        peak = None
        peak_step = None
        init_val = None
        val_history = []
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
                val_history.append((s, v))
                if init_val is None:
                    init_val = v
                if peak is None or v > peak:
                    peak = v
                    peak_step = s
        step = last.get("step") if last else None
        chars = last.get("val/response_chars") if last else None
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        print(
            f"step={step}  init_val={_fmt(init_val)}  "
            f"peak_val={_fmt(peak)}@{peak_step}  chars={_fmt(chars, '.0f')}"
        )
        if val_history:
            # Show last 8 eval points
            print("  last 8 eval points:")
            for s, v in val_history[-8:]:
                bar = "#" * int(v * 50)
                print(f"    step {s:>4}: {v:.4f}  {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
