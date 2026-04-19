"""Convert GSM8K JSONL into the schema expected by ``cs336_alignment.grpo_train``.

GSM8K rows look like::

    {"question": "Natalia sold clips ...",
     "answer":   "Natalia sold 48/2 = <<48/2=24>>24 clips ...\n#### 72"}

The ``answer`` field is the full reasoning trace; the verified final answer is
the substring after the ``#### `` marker (a single integer/decimal). For
GRPO with ``r1_zero_reward_fn`` we only want that final number as the ground
truth -- the model is supposed to *learn* to produce its own reasoning.

Usage::

    uv run python scripts/prepare_gsm8k.py \
        --input-dir data/gsm8k \
        --output-dir data/gsm8k_grpo \
        --val-size 0    # gsm8k already ships a test.jsonl; we just rename it
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer


def _extract_final_answer(answer: str) -> str:
    """Return everything after the last ``#### `` marker, stripped of commas
    and surrounding whitespace. Falls back to the full ``answer`` if the
    marker is missing.
    """
    if "####" in answer:
        tail = answer.rsplit("####", 1)[1]
    else:
        tail = answer
    return tail.strip().replace(",", "")


def _row_to_record(row: dict) -> dict:
    return {
        "problem": row["question"].strip(),
        "solution": _extract_final_answer(row["answer"]),
        "data_source": "gsm8k",
    }


def _convert(path: Path, out_path: Path) -> int:
    n = 0
    with open(path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            fout.write(json.dumps(_row_to_record(row), ensure_ascii=False) + "\n")
            n += 1
    return n


def main(
    input_dir: str = typer.Option(
        "data/gsm8k", help="Directory holding GSM8K's train.jsonl / test.jsonl."
    ),
    output_dir: str = typer.Option(
        "data/gsm8k_grpo", help="Where to write the converted JSONL."
    ),
    val_size: int = typer.Option(
        0,
        help=(
            "If > 0, hold out this many examples from train.jsonl as a "
            "separate validation.jsonl. Otherwise use test.jsonl as validation."
        ),
    ),
    seed: int = 0,
):
    """Convert GSM8K JSONL to {problem, solution} JSONL."""
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_in = in_dir / "train.jsonl"
    test_in = in_dir / "test.jsonl"
    if not train_in.exists():
        raise FileNotFoundError(f"missing {train_in}")

    if val_size and val_size > 0:
        # Hold out from train; keep test as a separate file too.
        with open(train_in) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        if val_size >= len(rows):
            raise ValueError(f"val_size={val_size} >= dataset size {len(rows)}")
        idx = list(range(len(rows)))
        random.Random(seed).shuffle(idx)
        val_idx = set(idx[:val_size])
        n_train = n_val = 0
        with (
            open(out_dir / "train.jsonl", "w") as ftrain,
            open(out_dir / "validation.jsonl", "w") as fval,
        ):
            for i, row in enumerate(rows):
                line = json.dumps(_row_to_record(row), ensure_ascii=False) + "\n"
                if i in val_idx:
                    fval.write(line)
                    n_val += 1
                else:
                    ftrain.write(line)
                    n_train += 1
        print(f"wrote {n_train} -> {out_dir / 'train.jsonl'}")
        print(f"wrote {n_val} -> {out_dir / 'validation.jsonl'}")
        if test_in.exists():
            n_test = _convert(test_in, out_dir / "test.jsonl")
            print(f"wrote {n_test} -> {out_dir / 'test.jsonl'}")
    else:
        # No held-out split from train; use test.jsonl as the validation file.
        n_train = _convert(train_in, out_dir / "train.jsonl")
        print(f"wrote {n_train} -> {out_dir / 'train.jsonl'}")
        if not test_in.exists():
            raise FileNotFoundError(
                f"--val-size=0 was given but {test_in} does not exist; "
                "either pass --val-size N or provide test.jsonl."
            )
        n_val = _convert(test_in, out_dir / "validation.jsonl")
        print(f"wrote {n_val} -> {out_dir / 'validation.jsonl'}")


if __name__ == "__main__":
    typer.run(main)
