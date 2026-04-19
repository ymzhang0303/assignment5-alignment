"""Convert ``open-r1/DAPO-Math-17k-Processed`` parquet files into JSONL in
the schema expected by ``cs336_alignment.grpo_train``.

The HF dataset ships these columns:
    prompt          str   the math question
    solution        str   the verified numeric answer
    data_source     str   always "math_dapo"
    source_prompt   list  the original DAPO chat-style prompt
    ability         str   always "MATH"
    reward_model    dict  {"ground_truth": str, "style": "rule-lighteval/MATH_v2"}
    extra_info      dict  {"index": uuid}

We need ``problem`` (question) + ``solution`` (answer) per line. Upstream
only ships a ``train`` split, so we hold out ``--val-size`` rows for
validation here.

Usage::

    uv run python scripts/prepare_dapo_math.py \
        --input-dir data/DAPO-Math-17k \
        --output-dir data/dapo_math \
        --subset all \
        --val-size 1024
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer


def _row_to_record(row: dict) -> dict:
    """Pick out the fields we actually need."""
    answer = row.get("solution")
    if answer is None:
        rm = row.get("reward_model") or {}
        answer = rm.get("ground_truth")
    extra = row.get("extra_info") or {}
    return {
        "problem": row["prompt"],
        "solution": str(answer),
        "data_source": row.get("data_source", "math_dapo"),
        "index": extra.get("index"),
    }


def _read_parquet(path: Path) -> list[dict]:
    """Read a parquet file into a list of plain Python dicts."""
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(str(path))
        return table.to_pylist()
    except ImportError:
        import pandas as pd

        df = pd.read_parquet(str(path))
        return df.to_dict(orient="records")


def main(
    input_dir: str = typer.Option(
        "data/DAPO-Math-17k",
        help="Directory containing the HF parquet snapshot (with all/ en/ cn/ subdirs).",
    ),
    output_dir: str = typer.Option("data/dapo_math", help="Where to write JSONL."),
    subset: str = typer.Option(
        "all",
        help="Dataset config to load: one of {all, en, cn}.",
    ),
    val_size: int = typer.Option(
        1024, help="Number of examples to hold out as validation."
    ),
    seed: int = 0,
):
    """Convert local DAPO-Math-17k parquet to train/validation JSONL."""
    in_dir = Path(input_dir) / subset
    parquets = sorted(in_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"no .parquet files under {in_dir}")
    print(f"reading {len(parquets)} parquet file(s) from {in_dir} ...")
    rows: list[dict] = []
    for p in parquets:
        rows.extend(_read_parquet(p))
    print(f"  -> {len(rows)} rows")

    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    if val_size >= len(indices):
        raise ValueError(
            f"val_size={val_size} must be smaller than dataset size {len(indices)}"
        )
    val_idx = set(indices[:val_size])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.jsonl"
    val_path = out / "validation.jsonl"
    n_train = n_val = 0
    with open(train_path, "w") as ftrain, open(val_path, "w") as fval:
        for i, row in enumerate(rows):
            rec = _row_to_record(row)
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            if i in val_idx:
                fval.write(line)
                n_val += 1
            else:
                ftrain.write(line)
                n_train += 1

    print(f"wrote {n_train} -> {train_path}")
    print(f"wrote {n_val} -> {val_path}")


if __name__ == "__main__":
    typer.run(main)
