"""Evaluate Qwen3-8B (thinking on) on a random sample of Big-Math validation.

Samples ``--n-samples`` problems uniformly at random (with a reproducible
``--seed``), generates one response each with vLLM, and reports statistics on
the reasoning (``<think>...</think>``) tokens as well as the answer portion.

Example::

    CUDA_VISIBLE_DEVICES=7 uv run python scripts/eval_qwen3_8b_bigmath_sample.py \\
        --model-id /root/assignment5-alignment/models/Qwen3-8B \\
        --val-path data/big_math/validation.jsonl \\
        --output-dir runs/eval_qwen3_8b_bigmath_sample125 \\
        --device cuda:0 \\
        --n-samples 125 \\
        --seed 0 \\
        --max-tokens 32768
"""

from __future__ import annotations

import json
import random
import re
import statistics
import time
from pathlib import Path
from typing import Optional

import typer
from transformers import AutoTokenizer

from cs336_alignment.drgrpo_grader import r1_zero_thinking_reward_fn
from cs336_alignment.grpo_train import (
    format_chat_prompts,
    load_jsonl,
    load_system_prompt,
)


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _split_thinking(text: str) -> tuple[str, str, bool]:
    """Return ``(thinking_text, post_think_text, closed)``.

    If the response contains a closed ``<think>...</think>`` block, split on
    it. If only an opening ``<think>`` is present (truncated), everything
    after it is treated as thinking. If there's no ``<think>`` tag at all,
    the whole response is treated as post-think.
    """
    m = _THINK_RE.search(text)
    if m is not None:
        return m.group(1), text[m.end():], True
    # Truncated case: opening tag only.
    if "<think>" in text:
        idx = text.index("<think>") + len("<think>")
        return text[idx:], "", False
    return "", text, False


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _describe(xs: list[int], label: str) -> dict:
    if not xs:
        return {f"{label}_count": 0}
    return {
        f"{label}_count": len(xs),
        f"{label}_mean": statistics.fmean(xs),
        f"{label}_stdev": statistics.pstdev(xs),
        f"{label}_min": min(xs),
        f"{label}_p25": _percentile(xs, 0.25),
        f"{label}_p50": _percentile(xs, 0.50),
        f"{label}_p75": _percentile(xs, 0.75),
        f"{label}_p90": _percentile(xs, 0.90),
        f"{label}_p95": _percentile(xs, 0.95),
        f"{label}_p99": _percentile(xs, 0.99),
        f"{label}_max": max(xs),
    }


def main(
    model_id: str = typer.Option(
        "/root/assignment5-alignment/models/Qwen3-8B", help="Local model path."
    ),
    val_path: str = typer.Option(
        "data/big_math/validation.jsonl", help="Validation JSONL."
    ),
    output_dir: str = typer.Option(
        "runs/eval_qwen3_8b_bigmath_sample125",
        help="Where to write results.jsonl + summary.json.",
    ),
    device: str = typer.Option(
        "cuda:0",
        help=(
            "Single GPU index for vLLM (as seen by the process). Pair with "
            "CUDA_VISIBLE_DEVICES to pin to a specific physical GPU."
        ),
    ),
    enable_thinking: bool = typer.Option(
        True, help="Pass enable_thinking flag through Qwen3 chat template."
    ),
    system_prompt_name: str = typer.Option(
        "qwen3_thinking",
        help="Basename of system prompt under cs336_alignment/prompts/<name>.system.",
    ),
    n_samples: int = typer.Option(
        125, help="Number of validation rows to sample uniformly at random."
    ),
    n: int = typer.Option(1, help="Number of generations per sampled problem."),
    temperature: float = typer.Option(
        0.6, help="Sampling temperature (Qwen3-recommended for thinking is 0.6)."
    ),
    top_p: float = typer.Option(0.95, help="Nucleus sampling cutoff."),
    top_k: int = typer.Option(
        20, help="Top-k (Qwen3 recommends 20 for thinking mode)."
    ),
    max_tokens: int = typer.Option(
        32768, help="Per-sample token cap. Set high for 8B thinking."
    ),
    min_tokens: int = typer.Option(4, help="Per-sample min tokens."),
    gpu_memory_utilization: float = typer.Option(0.90, help="vLLM KV-cache budget."),
    max_model_len: Optional[int] = typer.Option(
        40960, help="vLLM max_model_len; set to cover prompt + max_tokens."
    ),
    seed: int = typer.Option(0, help="Seed for both sampling + vLLM."),
):
    """Evaluate Qwen3-8B on a random sample of Big-Math with reasoning-token stats."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_examples = load_jsonl(val_path)
    rng = random.Random(seed)
    if n_samples > len(all_examples):
        raise ValueError(
            f"n_samples={n_samples} exceeds validation size {len(all_examples)}"
        )
    idxs = rng.sample(range(len(all_examples)), n_samples)
    examples = [all_examples[i] for i in idxs]
    print(
        f"sampled {len(examples)} / {len(all_examples)} validation rows "
        f"from {val_path} (seed={seed})"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    sys_prompt = load_system_prompt(system_prompt_name)
    prompts = format_chat_prompts(
        examples,
        tokenizer,
        system_prompt=sys_prompt,
        enable_thinking=enable_thinking,
    )
    prompt_token_lens = [len(tokenizer(p).input_ids) for p in prompts]
    avg_prompt_tokens = sum(prompt_token_lens) / max(len(prompts), 1)
    print(
        f"chat-template ready (system_prompt={system_prompt_name!r}, "
        f"enable_thinking={enable_thinking}, "
        f"avg_prompt_tokens={avg_prompt_tokens:.1f}, "
        f"max_prompt_tokens={max(prompt_token_lens)})"
    )

    from cs336_alignment.vllm_utils import init_vllm

    llm = init_vllm(
        model_id=model_id,
        device=device,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_sleep_mode=False,
    )

    from vllm import SamplingParams

    sp = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        # Let the model close its own answer tag; we do NOT stop on </think>
        # because we want to measure reasoning length.
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=seed,
    )
    print(
        f"generating with vLLM (n={n}, T={temperature}, top_p={top_p}, "
        f"top_k={top_k}, max_tokens={max_tokens}) ..."
    )
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params=sp, use_tqdm=True)
    gen_t = time.time() - t0

    by_prompt = {o.prompt: o.outputs for o in outputs}

    results_path = out_dir / "results.jsonl"
    total_token_lens: list[int] = []
    thinking_token_lens: list[int] = []
    post_think_token_lens: list[int] = []
    truncated_total = 0
    no_close_think = 0
    n_correct = 0
    n_format = 0
    n_pass_at_n = 0

    with open(results_path, "w") as f:
        for ex, p in zip(examples, prompts):
            outs = by_prompt[p]
            sample_records = []
            any_correct = False
            for o in outs:
                resp = o.text
                ntok = len(o.token_ids)
                total_token_lens.append(ntok)

                think_text, post_text, closed = _split_thinking(resp)
                if not closed:
                    no_close_think += 1
                # Token-level counts via the tokenizer (no chat template).
                think_toks = (
                    len(tokenizer(think_text, add_special_tokens=False).input_ids)
                    if think_text
                    else 0
                )
                post_toks = (
                    len(tokenizer(post_text, add_special_tokens=False).input_ids)
                    if post_text
                    else 0
                )
                thinking_token_lens.append(think_toks)
                post_think_token_lens.append(post_toks)

                finish_reason = getattr(o, "finish_reason", None)
                if finish_reason == "length":
                    truncated_total += 1

                rew = r1_zero_thinking_reward_fn(resp, ex["solution"])
                sample_records.append(
                    {
                        "response": resp,
                        "n_tokens_total": ntok,
                        "n_tokens_thinking": think_toks,
                        "n_tokens_post_think": post_toks,
                        "think_closed": closed,
                        "finish_reason": finish_reason,
                        "reward": float(rew["reward"]),
                        "format_reward": float(rew["format_reward"]),
                        "answer_reward": float(rew["answer_reward"]),
                    }
                )
                if rew["answer_reward"] > 0:
                    any_correct = True

            first = sample_records[0]
            n_correct += int(first["answer_reward"] > 0)
            n_format += int(first["format_reward"] > 0)
            n_pass_at_n += int(any_correct)
            f.write(
                json.dumps(
                    {
                        "problem": ex["problem"],
                        "solution": ex["solution"],
                        "data_source": ex.get("data_source"),
                        "llama8b_solve_rate": ex.get("llama8b_solve_rate"),
                        "samples": sample_records,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    n_examples = len(examples)
    summary = {
        "model_id": model_id,
        "val_path": val_path,
        "n_examples": n_examples,
        "n_samples_per_problem": n,
        "sampling_seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
        "pass_at_1": n_correct / max(n_examples, 1),
        "format_at_1": n_format / max(n_examples, 1),
        f"pass_at_{n}": n_pass_at_n / max(n_examples, 1),
        "truncated_rollouts_length": truncated_total,
        "rollouts_without_think_close": no_close_think,
        "generation_seconds": gen_t,
        "prompt_tokens_avg": avg_prompt_tokens,
        "prompt_tokens_max": max(prompt_token_lens),
    }
    summary.update(_describe(total_token_lens, "response_tokens"))
    summary.update(_describe(thinking_token_lens, "thinking_tokens"))
    summary.update(_describe(post_think_token_lens, "post_think_tokens"))

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Big-Math random-sample eval summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nresults -> {results_path}")
    print(f"summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    typer.run(main)
