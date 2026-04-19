"""Convert ``SynthLabsAI/Big-Math-RL-Verified`` into JSONL in the schema
expected by ``cs336_alignment.grpo_train``.

Big-Math ships ~251K rigorously filtered, RL-friendly math problems. Each
row has these columns::

    problem            str    the question (always open-ended, never MC)
    answer             str    closed-form verifiable answer
    source             str    upstream dataset (Orca-Math, MATH, olympiads, ...)
    domain             str    math domain (algebra, number theory, ...)
    llama8b_solve_rate float  fraction of 64 Llama-3.1-8B rollouts that pass

We need ``problem`` + ``solution`` (the answer string) per line.

The big win for GRPO is the ``llama8b_solve_rate`` field: for group-relative
advantages to be informative, the per-prompt solve rate of the rollout policy
should sit roughly in (0, 1) -- otherwise every advantage in the group is 0.
We therefore filter to a difficulty band (default ``0.1 <= rate <= 0.6``),
which gives Qwen3-1.7B-class models a large pool of "learnable but not
trivial" problems.

The dataset is gated on the HuggingFace Hub -- you must accept the terms at
https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified once and be
logged in (``huggingface-cli login`` or ``HF_TOKEN=... uv run ...``).
Alternatively, point ``--input-dir`` at a local parquet snapshot.

Usage::

    uv run python scripts/prepare_big_math.py \\
        --output-dir data/big_math \\
        --min-solve-rate 0.1 \\
        --max-solve-rate 0.6 \\
        --val-size 1024
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import typer


SOURCE_TAG = "big_math"


def _row_to_record(row: dict) -> dict:
    """Pick out the fields we actually need."""
    return {
        "problem": str(row["problem"]).strip(),
        "solution": str(row["answer"]).strip(),
        "data_source": SOURCE_TAG,
        "source": row.get("source"),
        "domain": row.get("domain"),
        "llama8b_solve_rate": row.get("llama8b_solve_rate"),
    }


def _load_rows_from_hub(repo_id: str, cache_dir: Optional[str]) -> list[dict]:
    """Stream the dataset from the Hub via ``datasets.load_dataset``."""
    from datasets import load_dataset

    print(f"loading {repo_id} from HuggingFace Hub ...")
    ds = load_dataset(repo_id, split="train", cache_dir=cache_dir)
    print(f"  -> {len(ds)} rows")
    return ds.to_list()


def _load_rows_from_parquet(input_dir: Path) -> list[dict]:
    """Load any ``*.parquet`` files under ``input_dir`` into dicts."""
    import pyarrow.parquet as pq

    parquets = sorted(input_dir.rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"no .parquet files under {input_dir}")
    print(f"reading {len(parquets)} parquet file(s) from {input_dir} ...")
    rows: list[dict] = []
    for p in parquets:
        rows.extend(pq.read_table(str(p)).to_pylist())
    print(f"  -> {len(rows)} rows")
    return rows


def main(
    output_dir: str = typer.Option(
        "data/big_math", help="Where to write the converted JSONL."
    ),
    input_dir: Optional[str] = typer.Option(
        None,
        help=(
            "If set, load parquet files from this local directory instead of "
            "downloading from the Hub. Useful when the dataset is mirrored on "
            "disk."
        ),
    ),
    repo_id: str = typer.Option(
        "SynthLabsAI/Big-Math-RL-Verified",
        help="HuggingFace dataset repo id (only used when --input-dir is unset).",
    ),
    cache_dir: Optional[str] = typer.Option(
        None,
        help="HF datasets cache dir (defaults to ~/.cache/huggingface).",
    ),
    min_solve_rate: float = typer.Option(
        0.1,
        help=(
            "Drop problems with llama8b_solve_rate strictly below this value "
            "(too hard -> all-zero rollouts -> no GRPO signal)."
        ),
    ),
    max_solve_rate: float = typer.Option(
        0.6,
        help=(
            "Drop problems with llama8b_solve_rate strictly above this value "
            "(too easy -> all-one rollouts -> no GRPO signal)."
        ),
    ),
    keep_unrated: bool = typer.Option(
        False,
        help=(
            "If True, also keep rows whose llama8b_solve_rate is null "
            "(some upstream subsets like HARP/Omni-MATH skip the filter)."
        ),
    ),
    sources: Optional[str] = typer.Option(
        None,
        help=(
            "Comma-separated list of upstream sources to keep "
            "(e.g. 'MATH,olympiads,Big-Math-Reformulated'). "
            "Default: keep all sources."
        ),
    ),
    exclude_sources: Optional[str] = typer.Option(
        "GSM8k",
        help=(
            "Comma-separated list of upstream sources to drop. "
            "Defaults to dropping GSM8k since you likely already trained on it."
        ),
    ),
    max_problem_chars: int = typer.Option(
        2000,
        help="Drop problems whose text exceeds this many characters.",
    ),
    max_answer_chars: int = typer.Option(
        200,
        help="Drop problems whose answer exceeds this many characters.",
    ),
    limit: int = typer.Option(
        0,
        help="If > 0, randomly subsample to at most this many rows after filtering.",
    ),
    val_size: int = typer.Option(
        1024, help="Number of examples to hold out as validation."
    ),
    seed: int = typer.Option(0, help="Random seed for shuffling / sampling."),
):
    """Convert Big-Math-RL-Verified into train/validation JSONL."""
    if input_dir:
        rows = _load_rows_from_parquet(Path(input_dir))
    else:
        rows = _load_rows_from_hub(repo_id, cache_dir)

    keep_set = (
        {s.strip() for s in sources.split(",") if s.strip()} if sources else None
    )
    drop_set = (
        {s.strip() for s in exclude_sources.split(",") if s.strip()}
        if exclude_sources
        else set()
    )

    filtered: list[dict] = []
    n_dropped_diff = n_dropped_src = n_dropped_len = 0
    for row in rows:
        rate = row.get("llama8b_solve_rate")
        if rate is None:
            if not keep_unrated:
                n_dropped_diff += 1
                continue
        else:
            if rate < min_solve_rate or rate > max_solve_rate:
                n_dropped_diff += 1
                continue
        src = row.get("source")
        if src in drop_set:
            n_dropped_src += 1
            continue
        if keep_set is not None and src not in keep_set:
            n_dropped_src += 1
            continue
        problem = (row.get("problem") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not problem or not answer:
            n_dropped_len += 1
            continue
        if len(problem) > max_problem_chars or len(answer) > max_answer_chars:
            n_dropped_len += 1
            continue
        filtered.append(row)

    print(
        f"filtered: kept {len(filtered)} / {len(rows)} "
        f"(diff={n_dropped_diff}, src={n_dropped_src}, len={n_dropped_len})"
    )
    if not filtered:
        raise RuntimeError(
            "no rows survived filtering -- relax --min/--max-solve-rate or "
            "--exclude-sources"
        )

    rng = random.Random(seed)
    rng.shuffle(filtered)
    if limit and limit > 0 and limit < len(filtered):
        filtered = filtered[:limit]
        print(f"  subsampled to {len(filtered)} rows (limit={limit})")

    if val_size >= len(filtered):
        raise ValueError(
            f"val_size={val_size} must be smaller than dataset size {len(filtered)}"
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.jsonl"
    val_path = out / "validation.jsonl"

    n_train = n_val = 0
    with open(train_path, "w") as ftrain, open(val_path, "w") as fval:
        for i, row in enumerate(filtered):
            line = json.dumps(_row_to_record(row), ensure_ascii=False) + "\n"
            if i < val_size:
                fval.write(line)
                n_val += 1
            else:
                ftrain.write(line)
                n_train += 1

    print(f"wrote {n_train} -> {train_path}")
    print(f"wrote {n_val} -> {val_path}")

    if filtered:
        rates = [r["llama8b_solve_rate"] for r in filtered if r.get("llama8b_solve_rate") is not None]
        if rates:
            rates_sorted = sorted(rates)
            n = len(rates_sorted)
            print(
                "llama8b_solve_rate stats: "
                f"min={rates_sorted[0]:.3f} "
                f"p25={rates_sorted[n // 4]:.3f} "
                f"median={rates_sorted[n // 2]:.3f} "
                f"p75={rates_sorted[3 * n // 4]:.3f} "
                f"max={rates_sorted[-1]:.3f}"
            )
        from collections import Counter

        src_counts = Counter(r.get("source") for r in filtered)
        print("source breakdown:")
        for src, cnt in sorted(src_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {src}: {cnt}")


if __name__ == "__main__":
    typer.run(main)
