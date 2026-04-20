#!/usr/bin/env python
"""Monitor round-7 OPSD-inspired SDPO runs vs the GRPO baseline."""
from __future__ import annotations
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = ROOT / "runs"
GRPO_BASELINE = RUNS_DIR / "grpo_qwen3_bigmath_reinforce_bl"

R7_RUNS = [
    "a_gt_pure_lr5e6",
    "b_gt_grpoheavy_lr1e5",
    "c_gt_opsd_faithful",
    "d_gt_pglight_lr1e5",
    "e_tokenclip_on_r6d",
    "f_tokenclip_gt_grpoheavy",
]

# Also track the still-running round-6 runs for a full picture.
R6_RUNS = [
    "c_pgall_lr1e5",
    "d_grpoheavy_lr1e5",
    "g_longer_lr1e6",
    "h_grpoheavy_strongema",
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
        "peak": peak["val/answer_reward"],
        "pkStp": peak.get("step", -1),
        "last": last["val/answer_reward"],
        "chars": last.get("val/response_chars", 0),
    }


def main() -> None:
    grpo = _summarise(GRPO_BASELINE)
    grpo_peak = grpo.get("peak", 0.6484)
    grpo_last = grpo.get("last", 0.6055)
    print(f"GRPO baseline:  peak={grpo_peak:.4f}  final={grpo_last:.4f}")
    print()
    hdr = f"{'run':32s}  {'step':>4s}  {'init':>5s}  {'peak':>5s}  {'pkStp':>5s}  {'last':>5s}  {'chars':>6s}  {'vsGRPO':>6s}"
    print(hdr)
    print("-" * len(hdr))

    def _print(prefix: str, names: list[str]) -> None:
        for n in names:
            s = _summarise(RUNS_DIR / f"{prefix}_{n}")
            if not s:
                print(f"{n:32s}  (no metrics yet)")
                continue
            vs = s["last"] - grpo_last
            print(
                f"{n:32s}  {int(s['step']):4d}  {s['init']:5.3f}  {s['peak']:5.3f}  "
                f"{int(s['pkStp']):5d}  {s['last']:5.3f}  {int(s['chars']):6d}  {vs:+.3f}"
            )

    print("# round 7 (OPSD-inspired, GT teacher):")
    _print("sdpo_r7", R7_RUNS)
    print()
    print("# round 6 survivors:")
    _print("sdpo_r6", R6_RUNS)


if __name__ == "__main__":
    main()
