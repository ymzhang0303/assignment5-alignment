"""Standalone evaluation of Qwen3-1.7B (thinking on by default) on Big-Math.

Loads a vLLM instance, formats prompts with the model's chat template, samples
``--n`` responses per problem, and computes pass@1 / pass@n with the
``r1_zero_thinking_reward_fn`` grader. Per-problem results are written to a
JSONL so you can audit individual rollouts.

Example::

    uv run python scripts/eval_qwen3_bigmath.py \\
        --model-id /root/assignment5-alignment/models/Qwen3-1.7B \\
        --val-path data/big_math/validation.jsonl \\
        --output-dir runs/eval_qwen3_bigmath_thinking \\
        --device cuda:4 \\
        --max-tokens 16384 \\
        --max-examples 1024 \\
        --n 1 \\
        --temperature 0.6
"""

from __future__ import annotations

import json
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
    vllm_generate,
)


def main(
    model_id: str = typer.Option(
        "/root/assignment5-alignment/models/Qwen3-1.7B", help="Local model path."
    ),
    val_path: str = typer.Option(
        "data/big_math/validation.jsonl", help="Validation JSONL."
    ),
    output_dir: str = typer.Option(
        "runs/eval_qwen3_bigmath_thinking",
        help="Where to write results.jsonl + summary.json.",
    ),
    device: str = typer.Option("cuda:4", help="Single GPU for vLLM."),
    enable_thinking: bool = typer.Option(
        True, help="Pass enable_thinking flag through Qwen3 chat template."
    ),
    system_prompt_name: str = typer.Option(
        "qwen3_thinking",
        help=(
            "Basename of system prompt under cs336_alignment/prompts/<name>.system "
            "(use 'qwen3_no_thinking' when --no-enable-thinking)."
        ),
    ),
    n: int = typer.Option(1, help="Number of samples per problem."),
    temperature: float = typer.Option(
        0.6, help="Sampling temperature (Qwen3-recommended for thinking is 0.6)."
    ),
    top_p: float = typer.Option(0.95, help="Nucleus sampling cutoff."),
    max_tokens: int = typer.Option(
        16384, help="Per-sample token cap. Set high for thinking mode."
    ),
    min_tokens: int = typer.Option(4, help="Per-sample min tokens."),
    max_examples: Optional[int] = typer.Option(
        None, help="Cap on # validation rows (None = all)."
    ),
    gpu_memory_utilization: float = typer.Option(0.85, help="vLLM KV-cache budget."),
    seed: int = typer.Option(0, help="vLLM sampling seed."),
):
    """Evaluate Qwen3-1.7B on Big-Math."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_jsonl(val_path)
    if max_examples is not None:
        examples = examples[:max_examples]
    print(f"loaded {len(examples)} validation rows from {val_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    sys_prompt = load_system_prompt(system_prompt_name)
    prompts = format_chat_prompts(
        examples, tokenizer, system_prompt=sys_prompt, enable_thinking=enable_thinking
    )
    avg_prompt_tokens = sum(len(tokenizer(p).input_ids) for p in prompts) / max(
        len(prompts), 1
    )
    print(
        f"chat-template ready (system_prompt={system_prompt_name!r}, "
        f"enable_thinking={enable_thinking}, "
        f"avg_prompt_tokens={avg_prompt_tokens:.1f})"
    )

    from cs336_alignment.vllm_utils import init_vllm

    llm = init_vllm(
        model_id=model_id,
        device=device,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # Custom vllm_generate-like call that also passes top_p.
    from vllm import SamplingParams

    sp = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=seed,
    )
    print(f"generating with vLLM (n={n}, T={temperature}, max_tokens={max_tokens}) ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params=sp, use_tqdm=True)
    gen_t = time.time() - t0

    by_prompt = {o.prompt: o.outputs for o in outputs}

    results_path = out_dir / "results.jsonl"
    n_correct = 0
    n_format = 0
    n_pass_at_n = 0
    total_tokens = 0
    truncated = 0
    with open(results_path, "w") as f:
        for ex, p in zip(examples, prompts):
            outs = by_prompt[p]
            sample_records = []
            any_correct = False
            for o in outs:
                resp = o.text
                ntok = len(o.token_ids)
                total_tokens += ntok
                if "</answer>" not in resp and "</think>" not in resp:
                    truncated += 1
                rew = r1_zero_thinking_reward_fn(resp, ex["solution"])
                sample_records.append(
                    {
                        "response": resp,
                        "n_tokens": ntok,
                        "reward": float(rew["reward"]),
                        "format_reward": float(rew["format_reward"]),
                        "answer_reward": float(rew["answer_reward"]),
                    }
                )
                if rew["answer_reward"] > 0:
                    any_correct = True
            # Score the *first* sample as pass@1 (deterministic w/ seed for fair report).
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
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
        "pass_at_1": n_correct / max(n_examples, 1),
        "format_at_1": n_format / max(n_examples, 1),
        f"pass_at_{n}": n_pass_at_n / max(n_examples, 1),
        "avg_response_tokens": total_tokens / max(n_examples * n, 1),
        "truncated_rollouts": truncated,
        "generation_seconds": gen_t,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Big-Math eval summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"results -> {results_path}")
    print(f"summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    typer.run(main)
