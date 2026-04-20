"""End-to-end GRPO training loop on the MATH dataset.

Implements the algorithm from Section 7.1 of the assignment writeup:

    for step in range(n_grpo_steps):
        sample n_prompts_per_rollout_batch prompts from D
        roll out group_size responses per prompt with vLLM (= old policy)
        compute group-normalized advantages
        (off-policy) compute old-policy log-probs once per rollout batch
        for epoch in range(epochs_per_rollout_batch):
            shuffle rollout batch into train batches of train_batch_size
            for microbatch in train batch:
                forward train policy, compute log-probs (and entropy)
                grpo_microbatch_train_step(...)
            optimizer.step(); zero_grad()
        sync trained policy weights into vLLM
        every eval_every steps -> validation rollouts + reward

Run with::

    uv run python -m cs336_alignment.grpo_train \
        --model-id /data/a5-alignment/models/Qwen2.5-Math-1.5B \
        --train-path /data/a5-alignment/MATH/train.jsonl \
        --val-path  /data/a5-alignment/MATH/validation.jsonl \
        --output-dir runs/grpo_baseline
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import torch
import typer
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.drgrpo_grader import (
    question_only_reward_fn,
    r1_zero_reward_fn,
    r1_zero_thinking_reward_fn,
)
from cs336_alignment.grpo import (
    compute_group_normalized_rewards,
    get_response_log_probs,
    grpo_microbatch_train_step,
    masked_mean,
    tokenize_prompt_and_output,
)
from cs336_alignment.vllm_utils import (
    init_vllm,
    load_policy_into_vllm_instance,
    sleep_engine,
    wake_engine,
)


REWARD_FNS = {
    "r1_zero": r1_zero_reward_fn,
    "r1_zero_thinking": r1_zero_thinking_reward_fn,
    "question_only": question_only_reward_fn,
}

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt_template(name: str) -> str:
    """Read one of the bundled prompt templates by short name.

    ``name`` is the basename without the ``.prompt`` suffix (e.g.
    ``r1_zero``).
    """
    return (PROMPTS_DIR / f"{name}.prompt").read_text()


def load_system_prompt(name: str) -> str:
    """Read a ``.system`` prompt template by short name."""
    return (PROMPTS_DIR / f"{name}.system").read_text().strip()


def load_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    """Load a ``.jsonl`` file into a list of dicts."""
    out: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def format_prompts(
    examples: list[dict[str, Any]], template: str, key: str = "problem"
) -> list[str]:
    """Render the prompt template for each example.

    The template uses ``{question}`` as the placeholder; ``key`` selects the
    field of the example that holds the question text.
    """
    return [template.format(question=ex[key]) for ex in examples]


def format_chat_prompts(
    examples: list[dict[str, Any]],
    tokenizer,
    system_prompt: Optional[str] = None,
    key: str = "problem",
    enable_thinking: bool = True,
) -> list[str]:
    """Render prompts using the model's native chat template.

    For Qwen3 (a thinking model) this is the right thing to do: the chat
    template puts the model into the assistant turn at the point where it
    is expected to start emitting ``<think>...``. We append nothing else --
    the model will produce its own ``<think>``/``</think>`` tags as part of
    its sampled response.

    If ``enable_thinking`` is False, Qwen3's chat template injects an empty
    ``<think>\\n\\n</think>\\n\\n`` before the assistant generation point, so
    the model skips the thinking phase and emits its final answer directly.
    """
    out = []
    for ex in examples:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": ex[key]})
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        out.append(rendered)
    return out


def _ground_truth_of(ex: dict[str, Any]) -> str:
    for k in ("answer", "ground_truth", "solution"):
        if k in ex:
            return ex[k]
    raise KeyError(
        f"example {list(ex.keys())} has no answer/ground_truth/solution field"
    )


def vllm_generate(
    llm,
    prompts: list[str],
    *,
    n: int,
    temperature: float,
    min_tokens: int,
    max_tokens: int,
    stop: Optional[list[str]] = None,
    seed: Optional[int] = None,
) -> list[list[str]]:
    """Sample ``n`` completions per prompt with vLLM. Returns
    ``len(prompts)`` lists of length ``n``.
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        stop=stop,
        include_stop_str_in_output=True,
        seed=seed,
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    out: list[list[str]] = []
    prompt_to_outs = {o.prompt: o.outputs for o in outputs}
    for p in prompts:
        outs = prompt_to_outs[p]
        out.append([o.text for o in outs])
    return out


@torch.inference_mode()
def evaluate(
    llm,
    val_examples: list[dict[str, Any]],
    prompt_renderer,
    reward_fn,
    *,
    temperature: float,
    max_tokens: int,
    min_tokens: int,
    stop: Optional[list[str]],
    max_examples: Optional[int] = None,
) -> dict[str, float]:
    """Generate one sample per validation prompt and average the reward
    components. ``prompt_renderer`` takes a ``list[example]`` and returns a
    ``list[str]`` of formatted prompts.
    """
    if max_examples is not None:
        val_examples = val_examples[:max_examples]
    prompts = prompt_renderer(val_examples)
    samples = vllm_generate(
        llm,
        prompts,
        n=1,
        temperature=temperature,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        stop=stop,
    )
    rewards = []
    fmt_rewards = []
    ans_rewards = []
    lengths = []
    for ex, [resp] in zip(val_examples, samples):
        r = reward_fn(resp, _ground_truth_of(ex))
        rewards.append(float(r["reward"]))
        fmt_rewards.append(float(r.get("format_reward", 0.0)))
        ans_rewards.append(float(r.get("answer_reward", 0.0)))
        lengths.append(len(resp))
    n = max(len(rewards), 1)
    return {
        "val/reward": sum(rewards) / n,
        "val/format_reward": sum(fmt_rewards) / n,
        "val/answer_reward": sum(ans_rewards) / n,
        "val/response_chars": sum(lengths) / n,
        "val/n": n,
    }


def build_microbatches(
    tokenized: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    raw_rewards: torch.Tensor,
    old_log_probs: Optional[torch.Tensor],
    micro_batch_size: int,
) -> Iterable[dict[str, torch.Tensor]]:
    """Yield contiguous micro-batches of the given training arrays."""
    n = tokenized["input_ids"].shape[0]
    for start in range(0, n, micro_batch_size):
        end = min(start + micro_batch_size, n)
        batch = {
            "input_ids": tokenized["input_ids"][start:end],
            "labels": tokenized["labels"][start:end],
            "response_mask": tokenized["response_mask"][start:end],
            "advantages": advantages[start:end],
            "raw_rewards": raw_rewards[start:end],
        }
        if old_log_probs is not None:
            batch["old_log_probs"] = old_log_probs[start:end]
        yield batch


@torch.inference_mode()
def compute_old_log_probs(
    policy: nn.Module,
    tokenized: dict[str, torch.Tensor],
    micro_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Run the *current* policy on the rollout batch in eval-mode and stash
    the resulting per-token log-probs. These are the ``old_log_probs`` used by
    GRPO-Clip (and frozen across off-policy epochs).
    """
    out = []
    n = tokenized["input_ids"].shape[0]
    for start in range(0, n, micro_batch_size):
        end = min(start + micro_batch_size, n)
        ids = tokenized["input_ids"][start:end].to(device)
        labels = tokenized["labels"][start:end].to(device)
        result = get_response_log_probs(policy, ids, labels, return_token_entropy=False)
        out.append(result["log_probs"].cpu())
    return torch.cat(out, dim=0)


def main(
    model_id: str = typer.Option(
        "/data/a5-alignment/models/Qwen2.5-Math-1.5B",
        help="HF model id or local path for the policy.",
    ),
    train_path: str = typer.Option(
        "/data/a5-alignment/MATH/train.jsonl",
        help="Path to the train .jsonl (each line has problem/solution).",
    ),
    val_path: str = typer.Option(
        "/data/a5-alignment/MATH/validation.jsonl",
        help="Path to the validation .jsonl.",
    ),
    output_dir: str = typer.Option("runs/grpo", help="Where to save checkpoints/logs."),
    prompt_name: str = typer.Option(
        "r1_zero",
        help=(
            "Prompt template name. For ``prompt_format='raw'`` this is a "
            "``.prompt`` file under cs336_alignment/prompts/. For "
            "``prompt_format='chat'`` this is the basename of a ``.system`` "
            "file (or 'none' for no system prompt)."
        ),
    ),
    prompt_format: str = typer.Option(
        "raw",
        help=(
            "How to render prompts. 'raw' uses a {question}-templated text "
            "file; 'chat' uses tokenizer.apply_chat_template (recommended "
            "for instruction/thinking models like Qwen3)."
        ),
    ),
    enable_thinking: bool = typer.Option(
        True,
        help=(
            "Only used when prompt_format='chat' on a thinking model. "
            "If False, passes enable_thinking=False to the chat template "
            "(Qwen3 will emit an empty <think></think> block in the prompt "
            "and answer directly -- much faster, much shorter rollouts)."
        ),
    ),
    reward_fn_name: str = typer.Option("r1_zero", help="Reward fn name."),
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.45,
    loss_type: str = typer.Option(
        "reinforce_with_baseline",
        help="One of {no_baseline, reinforce_with_baseline, grpo_clip}.",
    ),
    use_std_normalization: bool = True,
    length_normalization: str = typer.Option(
        "masked_mean",
        help=(
            "Sequence-dim reducer for the per-token policy-gradient loss. "
            "'masked_mean' (default) averages over response tokens "
            "(standard GRPO; biased toward shorter sequences). "
            "'masked_normalize' uses Dr-GRPO style sum / normalize_constant "
            "to remove the length bias."
        ),
    ),
    normalize_constant: Optional[float] = typer.Option(
        None,
        help=(
            "Denominator for length_normalization='masked_normalize'. "
            "Defaults to sampling_max_tokens if unset, so per-token weights "
            "match those of a fully-extended rollout."
        ),
    ),
    cliprange: float = 0.2,
    grad_clip: float = 1.0,
    seed: int = 0,
    eval_every: int = 5,
    eval_examples: int = 1024,
    log_all_rollouts: bool = typer.Option(
        True,
        help=(
            "If True, write *every* rollout (prompt, response, reward, "
            "advantage) to rollouts.jsonl every step. Disable to save disk."
        ),
    ),
    log_examples_every: int = typer.Option(
        1,
        help="Step cadence for adding rollout examples to the wandb table.",
    ),
    n_log_examples: int = typer.Option(
        8,
        help="How many rollouts to surface in the wandb Table per logged step.",
    ),
    device: str = typer.Option(
        "cuda:0",
        help=(
            "GPU to colocate the trainer and the vLLM rollout engine on. "
            "vLLM uses sleep/wake to release its GPU memory while the "
            "trainer is doing fwd/bwd, so a single H200-class GPU is "
            "enough for Qwen3-1.7B."
        ),
    ),
    save_every: Optional[int] = None,
    use_wandb: bool = True,
    wandb_project: str = "cs336-grpo",
    wandb_run_name: Optional[str] = None,
    wandb_mode: str = typer.Option(
        "online", help="wandb mode: 'online', 'offline', or 'disabled'."
    ),
):
    """Run GRPO training. See module docstring for an example invocation."""

    # ---------- sanity asserts (from writeup) ---------- #
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    if loss_type == "grpo_clip" and epochs_per_rollout_batch == 1:
        # Allowed but pointless: with one on-policy step per rollout, ratio = 1.
        pass
    valid_loss_types = {"no_baseline", "reinforce_with_baseline", "grpo_clip"}
    if loss_type not in valid_loss_types:
        raise ValueError(f"loss_type must be one of {valid_loss_types}")
    valid_length_norms = {"masked_mean", "masked_normalize"}
    if length_normalization not in valid_length_norms:
        raise ValueError(
            f"length_normalization must be one of {valid_length_norms}"
        )
    if length_normalization == "masked_normalize" and normalize_constant is None:
        normalize_constant = float(sampling_max_tokens)
        print(
            f"length_normalization='masked_normalize' but no --normalize-constant; "
            f"defaulting to sampling_max_tokens={sampling_max_tokens}."
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in locals().items() if not k.startswith("_")}
    with open(Path(output_dir) / "config.json", "w") as f:
        json.dump(
            {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
             for k, v in config.items()},
            f,
            indent=2,
        )

    random.seed(seed)
    torch.manual_seed(seed)

    if use_wandb:
        import wandb

        run_name = wandb_run_name or Path(output_dir).name
        wandb.init(
            project=wandb_project,
            name=run_name,
            mode=wandb_mode,
            dir=output_dir,
            config={
                k: v
                for k, v in config.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            },
        )
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("val/*", step_metric="step")
        wandb.define_metric("time/*", step_metric="step")

    # ---------- data, reward fn ---------- #
    if reward_fn_name not in REWARD_FNS:
        raise ValueError(f"unknown reward_fn {reward_fn_name!r}")
    reward_fn = REWARD_FNS[reward_fn_name]

    train_examples = load_jsonl(train_path)
    val_examples = load_jsonl(val_path)
    print(f"loaded {len(train_examples)} train / {len(val_examples)} val examples")

    # ---------- tokenizer & policy ---------- #
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- prompt renderer ---------- #
    if prompt_format == "raw":
        template = load_prompt_template(prompt_name)
        prompt_renderer = lambda exs: format_prompts(exs, template)
    elif prompt_format == "chat":
        sys_prompt = (
            None if prompt_name in ("", "none") else load_system_prompt(prompt_name)
        )
        prompt_renderer = lambda exs: format_chat_prompts(
            exs,
            tokenizer,
            system_prompt=sys_prompt,
            enable_thinking=enable_thinking,
        )
        print(
            f"using chat-template prompts (system_prompt={prompt_name!r}, "
            f"len={len(sys_prompt) if sys_prompt else 0} chars, "
            f"enable_thinking={enable_thinking})"
        )
    else:
        raise ValueError(
            f"prompt_format must be 'raw' or 'chat', got {prompt_format!r}"
        )

    policy = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    policy.to(device)
    policy.train()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    # ---------- vLLM rollout engine (colocated on `device`) ---------- #
    # ``enable_sleep_mode`` lets us release vLLM's GPU memory between
    # rollouts so the trainer has the full GPU for fwd/bwd.
    llm = init_vllm(
        model_id=model_id,
        device=device,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_sleep_mode=True,
    )
    load_policy_into_vllm_instance(policy, llm)
    # Park vLLM right after init -- the first step will wake it before rollout.
    sleep_engine(llm, level=1)

    # Stop on the closing answer tag for both raw r1_zero and the chat-mode
    # thinking models, so we never burn tokens past the answer.
    if reward_fn_name in ("r1_zero", "r1_zero_thinking"):
        stop = ["</answer>"]
    else:
        stop = None

    # ---------- main loop ---------- #
    rollout_log_path = Path(output_dir) / "rollouts.jsonl"
    rollout_log = open(rollout_log_path, "w")
    metrics_log_path = Path(output_dir) / "metrics.jsonl"
    metrics_log = open(metrics_log_path, "w")

    rng = random.Random(seed)

    def log_metrics(step: int, metrics: dict[str, Any]) -> None:
        record = {"step": step, **metrics}
        metrics_log.write(json.dumps(record) + "\n")
        metrics_log.flush()
        if use_wandb:
            import wandb

            wandb.log({"step": step, **metrics}, step=step)
        compact = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}
        print(f"[step {step}] {compact}")

    for step in range(n_grpo_steps):
        step_t0 = time.time()
        # ---- 1. sample prompts ---- #
        batch_examples = rng.sample(
            train_examples, k=min(n_prompts_per_rollout_batch, len(train_examples))
        )
        prompt_strs = prompt_renderer(batch_examples)
        ground_truths = [_ground_truth_of(ex) for ex in batch_examples]

        # ---- 2. roll out with vLLM ---- #
        # Wake the colocated vLLM engine, push the trainer's current weights
        # into it, generate, then put the engine back to sleep so the
        # trainer has exclusive GPU access for fwd/bwd.
        rollout_t0 = time.time()
        wake_engine(llm)
        load_policy_into_vllm_instance(policy, llm)
        sampled = vllm_generate(
            llm,
            prompt_strs,
            n=group_size,
            temperature=sampling_temperature,
            min_tokens=sampling_min_tokens,
            max_tokens=sampling_max_tokens,
            stop=stop,
            seed=seed + step,
        )
        sleep_engine(llm, level=1)
        rollout_time = time.time() - rollout_t0

        # Flatten so contiguous chunks of `group_size` come from the same prompt.
        rollout_responses: list[str] = []
        repeated_prompts: list[str] = []
        repeated_ground_truths: list[str] = []
        for prompt, gt, group in zip(prompt_strs, ground_truths, sampled):
            for resp in group:
                rollout_responses.append(resp)
                repeated_prompts.append(prompt)
                repeated_ground_truths.append(gt)

        # ---- 3. compute advantages ---- #
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )

        # ---- 4. tokenize the rollout batch ---- #
        tokenized = tokenize_prompt_and_output(
            repeated_prompts, rollout_responses, tokenizer
        )

        # ---- 5. (off-policy) compute frozen old log probs ---- #
        need_old = (loss_type == "grpo_clip") or (epochs_per_rollout_batch > 1)
        if need_old:
            old_log_probs = compute_old_log_probs(
                policy, tokenized, micro_train_batch_size, torch.device(device)
            )
        else:
            old_log_probs = None

        # ---- 6. train ---- #
        train_t0 = time.time()
        n = rollout_batch_size
        loss_running = 0.0
        entropy_running = 0.0
        clip_frac_running = 0.0
        n_micro = 0

        for epoch in range(epochs_per_rollout_batch):
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + step * 1000 + epoch))

            for tb_start in range(0, n, train_batch_size):
                tb_end = min(tb_start + train_batch_size, n)
                tb_idx = perm[tb_start:tb_end]

                tokenized_tb = {k: v[tb_idx] for k, v in tokenized.items()}
                adv_tb = advantages[tb_idx]
                raw_tb = raw_rewards[tb_idx]
                old_tb = old_log_probs[tb_idx] if old_log_probs is not None else None

                optimizer.zero_grad(set_to_none=True)
                for micro in build_microbatches(
                    tokenized_tb, adv_tb, raw_tb, old_tb, micro_train_batch_size
                ):
                    input_ids = micro["input_ids"].to(device)
                    labels = micro["labels"].to(device)
                    response_mask = micro["response_mask"].to(device)

                    fwd = get_response_log_probs(
                        policy, input_ids, labels, return_token_entropy=True
                    )
                    log_probs = fwd["log_probs"]
                    token_entropy = fwd["token_entropy"]

                    raw_kw = micro["raw_rewards"].to(device).unsqueeze(-1)
                    adv_kw = micro["advantages"].to(device).unsqueeze(-1)
                    old_kw = (
                        micro["old_log_probs"].to(device)
                        if "old_log_probs" in micro
                        else None
                    )

                    loss, meta = grpo_microbatch_train_step(
                        policy_log_probs=log_probs,
                        response_mask=response_mask,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        loss_type=loss_type,
                        raw_rewards=raw_kw,
                        advantages=adv_kw,
                        old_log_probs=old_kw,
                        cliprange=cliprange if loss_type == "grpo_clip" else None,
                        length_normalization=length_normalization,
                        normalize_constant=normalize_constant,
                    )

                    loss_running += float(loss.detach().item()) * gradient_accumulation_steps
                    with torch.no_grad():
                        entropy_running += float(
                            masked_mean(token_entropy, response_mask).item()
                        )
                        if "clip_fraction" in meta:
                            clip_frac_running += float(meta["clip_fraction"].item())
                    n_micro += 1

                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
                )
                optimizer.step()

        train_time = time.time() - train_t0

        # ---- 7. (no separate weight sync here) ---- #
        # vLLM is asleep. The next iteration's rollout phase will wake it
        # and call ``load_policy_into_vllm_instance`` with the freshly
        # updated policy weights.

        avg_loss = loss_running / max(n_micro, 1)
        avg_entropy = entropy_running / max(n_micro, 1)
        avg_clip_frac = clip_frac_running / max(n_micro, 1) if loss_type == "grpo_clip" else 0.0
        metrics = {
            "train/loss": avg_loss,
            "train/grad_norm": grad_norm,
            "train/token_entropy": avg_entropy,
            "train/clip_fraction": avg_clip_frac,
            "train/reward_mean": reward_meta["reward_mean"],
            "train/reward_std": reward_meta["reward_std"],
            "train/format_reward_mean": reward_meta["format_reward_mean"],
            "train/answer_reward_mean": reward_meta["answer_reward_mean"],
            "train/group_reward_std_mean": reward_meta["group_reward_std_mean"],
            "time/rollout_s": rollout_time,
            "time/train_s": train_time,
            "time/step_s": time.time() - step_t0,
        }

        # ---- 8. rollout dump ---- #
        # Always write *every* rollout in this batch to the JSONL log so we
        # can audit training trajectories offline. wandb tables get a small
        # subset to stay manageable.
        n_dump = len(rollout_responses) if log_all_rollouts else min(
            n_log_examples, len(rollout_responses)
        )
        all_records = []
        for i in range(n_dump):
            rec = {
                "step": step,
                "prompt_idx": i // group_size,
                "rollout_idx": i % group_size,
                "prompt": repeated_prompts[i],
                "response": rollout_responses[i],
                "ground_truth": repeated_ground_truths[i],
                "reward": float(raw_rewards[i].item()),
                "advantage": float(advantages[i].item()),
            }
            rollout_log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            all_records.append(rec)
        rollout_log.flush()

        if (
            use_wandb
            and log_examples_every > 0
            and step % log_examples_every == 0
            and all_records
        ):
            import wandb

            sample = all_records[: min(n_log_examples, len(all_records))]
            table = wandb.Table(
                columns=[
                    "step",
                    "prompt",
                    "response",
                    "ground_truth",
                    "reward",
                    "advantage",
                ],
                data=[
                    [
                        r["step"],
                        r["prompt"],
                        r["response"],
                        r["ground_truth"],
                        r["reward"],
                        r["advantage"],
                    ]
                    for r in sample
                ],
            )
            wandb.log({"rollouts": table}, step=step)

        # ---- 9. validation ---- #
        if eval_every > 0 and (step % eval_every == 0 or step == n_grpo_steps - 1):
            eval_t0 = time.time()
            # vLLM is asleep after the rollout phase; wake it for eval
            # generation, then put it back so training memory is reclaimed.
            wake_engine(llm)
            load_policy_into_vllm_instance(policy, llm)
            val_metrics = evaluate(
                llm,
                val_examples,
                prompt_renderer,
                reward_fn,
                temperature=sampling_temperature,
                max_tokens=sampling_max_tokens,
                min_tokens=sampling_min_tokens,
                stop=stop,
                max_examples=eval_examples,
            )
            sleep_engine(llm, level=1)
            val_metrics["time/eval_s"] = time.time() - eval_t0
            metrics.update(val_metrics)

        log_metrics(step, metrics)

        if save_every is not None and save_every > 0 and step > 0 and step % save_every == 0:
            ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
            policy.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    # ---- final save ---- #
    final_dir = Path(output_dir) / "final"
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    rollout_log.close()
    metrics_log.close()
    if use_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    typer.run(main)
