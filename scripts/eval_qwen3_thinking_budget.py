"""Evaluate Qwen3 with a *thinking-token budget* via two-stage generation.

Stage 1 generates the `<think>...` chain under a hard budget, stopping on
`</think>`. If stage 1 hits the budget without closing `</think>`, we append a
forced closer (``--reasoning-end-str``) containing ``</think>`` to jump the
model out of the thinking phase. Stage 2 then continues from the (forced)
``</think>`` boundary and produces ``<answer>...</answer>`` under a small
post-think budget.

This mirrors the "budget forcing" trick from the s1 paper
(Muennighoff et al. 2025) and is the standard way to cap Qwen3's runaway CoT
without mutating the tokenizer or the chat template.

Per-sample rollouts are written to ``results.jsonl`` with the two stages
clearly separated so you can audit the forced closures.

Example::

    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python \\
        scripts/eval_qwen3_thinking_budget.py \\
        --model-id /root/assignment5-alignment/models/Qwen3-1.7B \\
        --system-prompt-name qwen3_thinking_concise \\
        --output-dir runs/eval_qwen3_1p7b_budget1024_n8 \\
        --device cuda:0 --n-samples 8 --seed 0 \\
        --thinking-token-budget 1024 \\
        --reasoning-end-str "I have to give the solution based on the reasoning directly now.</think>" \\
        --post-think-max-tokens 256
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
    m = _THINK_RE.search(text)
    if m is not None:
        return m.group(1), text[m.end():], True
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
        "/root/assignment5-alignment/models/Qwen3-1.7B", help="Local model path."
    ),
    val_path: str = typer.Option(
        "data/big_math/validation.jsonl", help="Validation JSONL."
    ),
    output_dir: str = typer.Option(
        "runs/eval_qwen3_1p7b_budget1024_n8",
        help="Where to write results.jsonl + summary.json.",
    ),
    device: str = typer.Option("cuda:0", help="Single GPU for vLLM."),
    enable_thinking: bool = typer.Option(True, help="Qwen3 chat-template thinking flag."),
    system_prompt_name: str = typer.Option(
        "qwen3_thinking_concise",
        help="System-prompt basename under cs336_alignment/prompts/<name>.system.",
    ),
    n_samples: int = typer.Option(8, help="Number of validation rows (random sample)."),
    n: int = typer.Option(1, help="Number of generations per sampled problem."),
    temperature: float = typer.Option(0.6, help="Sampling temperature."),
    top_p: float = typer.Option(0.95, help="Nucleus sampling cutoff."),
    top_k: int = typer.Option(20, help="Top-k."),
    thinking_token_budget: int = typer.Option(
        1024, help="Hard cap on stage-1 (thinking) tokens."
    ),
    reasoning_end_str: str = typer.Option(
        "I have to give the solution based on the reasoning directly now.</think>",
        help=(
            "String appended when stage-1 hits the budget without emitting "
            "</think>. Must contain </think> to transition to the answer phase."
        ),
    ),
    post_think_max_tokens: int = typer.Option(
        256, help="Max tokens for stage-2 (answer) generation."
    ),
    min_tokens: int = typer.Option(4, help="Per-sample min tokens in stage 1."),
    gpu_memory_utilization: float = typer.Option(0.6, help="vLLM KV-cache budget."),
    max_model_len: Optional[int] = typer.Option(
        40960, help="vLLM max_model_len; must cover prompt + budgets."
    ),
    seed: int = typer.Option(0, help="Seed for sampling + vLLM."),
):
    """Two-stage eval with a hard thinking-token budget + forced closure."""
    if "</think>" not in reasoning_end_str:
        raise ValueError(
            "reasoning_end_str must contain '</think>' to exit the thinking phase."
        )

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
        f"(seed={seed})"
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
    print(
        f"chat-template ready (system_prompt={system_prompt_name!r}, "
        f"enable_thinking={enable_thinking}, "
        f"avg_prompt_tokens={sum(prompt_token_lens)/len(prompts):.1f}, "
        f"max_prompt_tokens={max(prompt_token_lens)})"
    )
    print(
        f"budgets: thinking={thinking_token_budget}, post_think={post_think_max_tokens}"
    )
    print(f"reasoning_end_str={reasoning_end_str!r}")

    from cs336_alignment.vllm_utils import init_vllm
    from vllm import SamplingParams

    llm = init_vllm(
        model_id=model_id,
        device=device,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_sleep_mode=False,
    )

    # --- Stage 1: thinking under a hard budget, stopping on </think>.
    sp_think = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=thinking_token_budget,
        min_tokens=min_tokens,
        stop=["</think>"],
        include_stop_str_in_output=True,
        seed=seed,
    )
    print(
        f"[stage1] generating thinking (n={n}, T={temperature}, "
        f"top_p={top_p}, top_k={top_k}, max_tokens={thinking_token_budget}) ..."
    )
    t0 = time.time()
    stage1_outputs = llm.generate(prompts, sampling_params=sp_think, use_tqdm=True)
    t_stage1 = time.time() - t0

    # Build stage-2 prompts by appending the stage-1 text (possibly with forced closer).
    # Keep per-rollout bookkeeping aligned across (prompt, n) pairs.
    stage2_prompts: list[str] = []
    stage1_texts: list[str] = []
    stage1_token_counts: list[int] = []
    forced_flags: list[bool] = []
    sample_index: list[tuple[int, int]] = []  # (prompt_idx, n_idx)

    by_prompt = {o.prompt: o.outputs for o in stage1_outputs}
    for p_idx, (ex, p) in enumerate(zip(examples, prompts)):
        for n_idx, o in enumerate(by_prompt[p]):
            s1_text = o.text
            n_toks = len(o.token_ids)
            finished_by_stop = "</think>" in s1_text
            if not finished_by_stop:
                # Budget-forcing: append the closer to jump to answer phase.
                s1_text = s1_text + reasoning_end_str
                forced = True
            else:
                forced = False
            stage1_texts.append(s1_text)
            stage1_token_counts.append(n_toks)
            forced_flags.append(forced)
            stage2_prompts.append(p + s1_text)
            sample_index.append((p_idx, n_idx))

    # --- Stage 2: produce the answer up to post_think_max_tokens, stop on </answer>.
    sp_answer = SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=post_think_max_tokens,
        min_tokens=1,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=seed,
    )
    print(
        f"[stage2] generating answers for {len(stage2_prompts)} rollouts "
        f"(max_tokens={post_think_max_tokens}, stop=</answer>) ..."
    )
    t0 = time.time()
    stage2_outputs = llm.generate(stage2_prompts, sampling_params=sp_answer, use_tqdm=True)
    t_stage2 = time.time() - t0

    # vLLM preserves order of inputs in outputs; we built stage2_prompts in the
    # same order as sample_index, so we can zip directly.
    assert len(stage2_outputs) == len(stage2_prompts)

    # Group back by (prompt_idx, n_idx).
    results_path = out_dir / "results.jsonl"
    rollouts_path = out_dir / "rollouts.txt"
    total_token_lens: list[int] = []
    thinking_token_lens: list[int] = []
    post_think_token_lens: list[int] = []
    n_correct = 0
    n_format = 0
    n_pass_at_n = 0
    n_forced = 0
    truncated_total = 0

    per_problem: list[list[dict]] = [[] for _ in range(len(examples))]
    for i, (p_idx, n_idx) in enumerate(sample_index):
        s1_text = stage1_texts[i]
        s1_tokens = stage1_token_counts[i]
        forced = forced_flags[i]
        s2 = stage2_outputs[i].outputs[0]
        s2_text = s2.text
        s2_tokens = len(s2.token_ids)
        full_response = s1_text + s2_text
        think_text, post_text, closed = _split_thinking(full_response)
        think_toks = (
            len(tokenizer(think_text, add_special_tokens=False).input_ids)
            if think_text else 0
        )
        post_toks = (
            len(tokenizer(post_text, add_special_tokens=False).input_ids)
            if post_text else 0
        )
        total_toks = s1_tokens + s2_tokens
        total_token_lens.append(total_toks)
        thinking_token_lens.append(think_toks)
        post_think_token_lens.append(post_toks)

        if forced:
            n_forced += 1
        # Truncation: no </answer> emitted in stage 2.
        if "</answer>" not in s2_text:
            truncated_total += 1

        rew = r1_zero_thinking_reward_fn(full_response, examples[p_idx]["solution"])
        per_problem[p_idx].append({
            "response": full_response,
            "stage1_text": s1_text,
            "stage2_text": s2_text,
            "stage1_tokens": s1_tokens,
            "stage2_tokens": s2_tokens,
            "total_tokens": total_toks,
            "thinking_tokens": think_toks,
            "post_think_tokens": post_toks,
            "think_closed": closed,
            "forced_closure": forced,
            "stage2_finish_reason": getattr(s2, "finish_reason", None),
            "reward": float(rew["reward"]),
            "format_reward": float(rew["format_reward"]),
            "answer_reward": float(rew["answer_reward"]),
        })

    with open(results_path, "w") as rf, open(rollouts_path, "w") as rtf:
        for p_idx, ex in enumerate(examples):
            recs = per_problem[p_idx]
            first = recs[0]
            n_correct += int(first["answer_reward"] > 0)
            n_format += int(first["format_reward"] > 0)
            n_pass_at_n += int(any(r["answer_reward"] > 0 for r in recs))
            rf.write(
                json.dumps(
                    {
                        "problem": ex["problem"],
                        "solution": ex["solution"],
                        "data_source": ex.get("data_source"),
                        "llama8b_solve_rate": ex.get("llama8b_solve_rate"),
                        "samples": recs,
                    },
                    ensure_ascii=False,
                ) + "\n"
            )
            # Human-readable dump for inspection.
            rtf.write("=" * 100 + "\n")
            rtf.write(f"[problem #{p_idx}] {ex['problem'][:200]}\n")
            rtf.write(f"[ground truth]  {str(ex['solution'])[:200]}\n")
            for i, r in enumerate(recs):
                rtf.write(
                    f"\n--- sample {i} "
                    f"(s1={r['stage1_tokens']}, s2={r['stage2_tokens']}, "
                    f"forced={r['forced_closure']}, "
                    f"reward={r['reward']}, "
                    f"format={r['format_reward']}, "
                    f"answer={r['answer_reward']}) ---\n"
                )
                rtf.write("[STAGE1]\n" + r["stage1_text"] + "\n")
                rtf.write("[STAGE2]\n" + r["stage2_text"] + "\n")
            rtf.write("\n")

    n_rollouts = len(total_token_lens)
    n_examples = len(examples)
    summary = {
        "model_id": model_id,
        "val_path": val_path,
        "system_prompt_name": system_prompt_name,
        "n_examples": n_examples,
        "n_samples_per_problem": n,
        "sampling_seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "enable_thinking": enable_thinking,
        "thinking_token_budget": thinking_token_budget,
        "post_think_max_tokens": post_think_max_tokens,
        "reasoning_end_str": reasoning_end_str,
        "pass_at_1": n_correct / max(n_examples, 1),
        "format_at_1": n_format / max(n_examples, 1),
        f"pass_at_{n}": n_pass_at_n / max(n_examples, 1),
        "forced_closure_count": n_forced,
        "forced_closure_rate": n_forced / max(n_rollouts, 1),
        "truncated_rollouts_length": truncated_total,
        "generation_seconds_stage1": t_stage1,
        "generation_seconds_stage2": t_stage2,
        "generation_seconds_total": t_stage1 + t_stage2,
        "prompt_tokens_avg": sum(prompt_token_lens) / len(prompts),
        "prompt_tokens_max": max(prompt_token_lens),
    }
    summary.update(_describe(total_token_lens, "response_tokens"))
    summary.update(_describe(thinking_token_lens, "thinking_tokens"))
    summary.update(_describe(post_think_token_lens, "post_think_tokens"))

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== thinking-budget eval summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nresults   -> {results_path}")
    print(f"rollouts  -> {rollouts_path}")
    print(f"summary   -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    typer.run(main)
