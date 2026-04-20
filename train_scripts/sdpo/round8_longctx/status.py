#!/usr/bin/env python
"""Monitor round-8 long-context SDPO vs GRPO baseline sweep."""
from __future__ import annotations
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = ROOT / "runs"
# 1536-ctx reference baselines (for context, not direct comparison).
GRPO_1536_PEAK = 0.6484
GRPO_1536_LAST = 0.6055

R8_RUNS = [
    "exp0_grpo_baseline",
    "exp1_sdpo_baseline",
    "exp2_pgfill_w1",
    "exp3_pgall_w1",
    "exp4_pgall_w10",
]


def _load_val(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "val/answer_reward" in r:
            out.append(r)
    return out


def _summarise(run_dir: Path) -> dict:
    vals = _load_val(run_dir / "metrics.jsonl")
    if not vals:
        return {}
    first = vals[0]
    last = vals[-1]
    peak = max(vals, key=lambda r: r["val/answer_reward"])
    return {
        "step": last.get("step", -1),
        "init": first["val/answer_reward"],
        "init_fmt": first.get("val/format_reward", 0.0),
        "init_chars": first.get("val/response_chars", 0),
        "peak": peak["val/answer_reward"],
        "pkStp": peak.get("step", -1),
        "last": last["val/answer_reward"],
        "last_fmt": last.get("val/format_reward", 0.0),
        "chars": last.get("val/response_chars", 0),
    }


def main() -> None:
    print(f"Reference: GRPO@1536  peak={GRPO_1536_PEAK:.4f}  final={GRPO_1536_LAST:.4f}")
    print()
    hdr = (
        f"{'run':28s}  {'step':>4s}  {'init':>5s}  {'iFmt':>4s}  {'iChr':>5s}  "
        f"{'peak':>5s}  {'pkStp':>5s}  {'last':>5s}  {'lFmt':>4s}  {'chars':>5s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for n in R8_RUNS:
        s = _summarise(RUNS_DIR / f"sdpo_r8_{n}")
        if not s:
            print(f"{n:28s}  (no metrics yet)")
            continue
        print(
            f"{n:28s}  {int(s['step']):4d}  "
            f"{s['init']:5.3f}  {s['init_fmt']:4.2f}  {int(s['init_chars']):5d}  "
            f"{s['peak']:5.3f}  {int(s['pkStp']):5d}  "
            f"{s['last']:5.3f}  {s['last_fmt']:4.2f}  {int(s['chars']):5d}"
        )


if __name__ == "__main__":
    main()
