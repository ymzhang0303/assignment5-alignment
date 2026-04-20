"""End-to-end SDPO training loop on the MATH/BigMath dataset.

Mirrors :mod:`cs336_alignment.grpo_train` but swaps the GRPO policy-gradient
update for the SDPO self-distillation update from
"Reinforcement Learning via Self-Distillation" (Hubotter et al., 2026,
arXiv:2601.20802).

Per-step pipeline (changes vs. GRPO marked with ★)::

    sample n_prompts_per_rollout_batch prompts from D
    roll out group_size responses per prompt with vLLM (= old policy)
    compute group-normalized advantages          (still useful for the optional PG
                                                  fallback term + for logging)
  ★ for each rollout, pick a successful sibling demo and build a reprompted
    user message; tokenize the *teacher prompt + same response* in parallel
    to the student prompt + response so KL aligns on the response tokens
  ★ for each microbatch:
        forward student on student-prompt, gather top-k log-probs + indices
        forward student (under no_grad) on teacher-prompt, gather log-probs at
            student's top-k indices  (= self-teacher under reprompted context)
        compute SDPO loss (alpha-JSD on top-k+tail, IS-clipped) over the
            response mask, masked further by ``self_distillation_mask``
        + (optional) GRPO-clip PG loss on the *non-distilled* samples
        backward, accumulate
    optimizer step
  ★ (optional) update the EMA teacher shadow weights
    every eval_every steps -> validation rollouts + reward

Run with::

    uv run python -m cs336_alignment.sdpo_train \
        --model-id /root/assignment5-alignment/models/Qwen3-1.7B \
        --train-path data/big_math/train.jsonl \
        --val-path  data/big_math/validation.jsonl \
        --output-dir runs/sdpo_qwen3_bigmath
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable, Optional

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
    compute_policy_gradient_loss,
    masked_mean,
)
from cs336_alignment.grpo_train import (
    REWARD_FNS,
    _ground_truth_of,
    compute_old_log_probs,
    evaluate,
    format_chat_prompts,
    format_prompts,
    load_jsonl,
    load_prompt_template,
    load_system_prompt,
    vllm_generate,
)
from cs336_alignment.sdpo import (
    DEFAULT_FEEDBACK_TEMPLATE,
    DEFAULT_REPROMPT_TEMPLATE,
    DEFAULT_SOLUTION_TEMPLATE,
    OPSD_REPROMPT_TEMPLATE,
    OPSD_SOLUTION_TEMPLATE,
    EmaTeacher,
    SelfDistillationConfig,
    build_reprompts,
    get_response_topk_log_probs,
    get_teacher_log_probs_at_indices,
    pick_successful_demo,
    sdpo_microbatch_train_step,
    tokenize_prompt_response_pair,
)
from cs336_alignment.vllm_utils import (
    init_vllm,
    load_policy_into_vllm_instance,
    sleep_engine,
    wake_engine,
)


def _format_user_questions(
    examples: list[dict[str, Any]], key: str = "problem"
) -> list[str]:
    """Return just the raw user-question strings (used inside reprompts)."""
    return [ex[key] for ex in examples]


def _render_chat(
    user_msg: str,
    tokenizer,
    *,
    system_prompt: Optional[str],
    enable_thinking: bool,
) -> str:
    """Render a single ``user_msg`` through the model's chat template,
    putting it into the assistant's turn (so the next tokens are the
    response). Used for both the original student prompt and the SDPO
    teacher reprompt.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_msg})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def build_microbatches_sdpo(
    student: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    raw_rewards: torch.Tensor,
    old_log_probs: Optional[torch.Tensor],
    self_distillation_mask: torch.Tensor,
    micro_batch_size: int,
) -> Iterable[dict[str, torch.Tensor]]:
    """Iterate microbatches over the rollout batch.

    Both ``student`` and ``teacher`` dicts use the keys
    ``input_ids/labels/response_mask`` returned by
    :func:`tokenize_prompt_response_pair`. They have *different* sequence
    lengths (because the teacher prompt embeds a demonstration), but their
    response masks, when active, point at the same response tokens.
    """
    n = student["input_ids"].shape[0]
    for start in range(0, n, micro_batch_size):
        end = min(start + micro_batch_size, n)
        batch = {
            "s_input_ids": student["input_ids"][start:end],
            "s_labels": student["labels"][start:end],
            "s_response_mask": student["response_mask"][start:end],
            "t_input_ids": teacher["input_ids"][start:end],
            "t_labels": teacher["labels"][start:end],
            "t_response_mask": teacher["response_mask"][start:end],
            "advantages": advantages[start:end],
            "raw_rewards": raw_rewards[start:end],
            "self_distillation_mask": self_distillation_mask[start:end],
        }
        if old_log_probs is not None:
            batch["old_log_probs"] = old_log_probs[start:end]
        yield batch


def _gather_response_only(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    response_len: int,
) -> torch.Tensor:
    """Pack the per-position log-probs at response positions into a fixed
    ``(B, response_len)`` tensor, padding shorter sequences with 0.

    The student and teacher prefixes have different lengths, so their
    per-position log_prob tensors are not directly aligned. To compare
    them we extract just the response slots into a common ``response_len``
    layout (the longest response in this microbatch).
    """
    bsz = log_probs.shape[0]
    out = log_probs.new_zeros((bsz, response_len))
    for i in range(bsz):
        idxs = response_mask[i].nonzero(as_tuple=False).squeeze(-1)
        n = idxs.numel()
        if n > 0:
            out[i, :n] = log_probs[i, idxs]
    return out


def _gather_response_only_extra(
    tensor: torch.Tensor,
    response_mask: torch.Tensor,
    response_len: int,
) -> torch.Tensor:
    """Same as :func:`_gather_response_only` but for tensors with an
    additional trailing dim (e.g., top-k log-probs of shape
    ``(B, T, K)``).
    """
    bsz, _, k = tensor.shape
    out = tensor.new_zeros((bsz, response_len, k))
    for i in range(bsz):
        idxs = response_mask[i].nonzero(as_tuple=False).squeeze(-1)
        n = idxs.numel()
        if n > 0:
            out[i, :n] = tensor[i, idxs]
    return out


def _aligned_response_mask(
    response_mask: torch.Tensor, response_len: int
) -> torch.Tensor:
    """Build a left-aligned response mask of width ``response_len``.

    ``response_mask`` has 1s scattered at the response positions inside the
    full (prompt+response) sequence. After packing the response log-probs
    into a left-aligned tensor we want a mask whose 1s are simply at
    columns ``[0 .. n_response_tokens)`` for each row.
    """
    counts = response_mask.sum(dim=-1).long()
    arange = torch.arange(response_len, device=response_mask.device).unsqueeze(0)
    return (arange < counts.unsqueeze(1)).to(response_mask.dtype)


def main(
    model_id: str = typer.Option(
        "/root/assignment5-alignment/models/Qwen3-1.7B",
        help="HF model id or local path for the policy (= self-teacher).",
    ),
    train_path: str = typer.Option(
        "data/big_math/train.jsonl",
        help="Path to the train .jsonl (each line has problem/solution).",
    ),
    val_path: str = typer.Option(
        "data/big_math/validation.jsonl",
        help="Path to the validation .jsonl.",
    ),
    output_dir: str = typer.Option("runs/sdpo", help="Where to save checkpoints/logs."),
    prompt_name: str = typer.Option(
        "qwen3_thinking",
        help=(
            "Prompt template name. For ``prompt_format='raw'`` this is a "
            "``.prompt`` file under cs336_alignment/prompts/. For "
            "``prompt_format='chat'`` this is the basename of a ``.system`` "
            "file (or 'none' for no system prompt)."
        ),
    ),
    prompt_format: str = typer.Option(
        "chat",
        help="'raw' or 'chat'. SDPO is much more natural with 'chat'.",
    ),
    enable_thinking: bool = typer.Option(
        True,
        help=(
            "Only used when prompt_format='chat'. If True, Qwen3-style "
            "thinking is enabled in both the student prompt and the "
            "teacher reprompt."
        ),
    ),
    reward_fn_name: str = typer.Option("r1_zero_thinking", help="Reward fn name."),
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1536,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.45,
    enforce_eager: bool = typer.Option(
        False,
        help=(
            "Disable vLLM's torch.compile graphs and run the rollout model "
            "in eager mode. Required when running under debugpy (debugpy's "
            "PEP 669 sys.monitoring tracer triggers a 'generator' "
            "Unsupported error inside dynamo's frame eval) and useful when "
            "iterating on the trainer code itself."
        ),
    ),
    use_std_normalization: bool = True,
    pg_loss_type: str = typer.Option(
        "grpo_clip",
        help=(
            "PG loss applied to non-distilled rollouts (those without a "
            "successful sibling demonstration). One of {no_baseline, "
            "reinforce_with_baseline, grpo_clip}."
        ),
    ),
    pg_loss_weight: float = typer.Option(
        0.0,
        help=(
            "Coefficient on the optional GRPO-style PG term applied to "
            "non-distilled samples. Set to 0 (default) for pure SDPO -- "
            "matches the paper but means rollouts without a sibling demo "
            "contribute zero gradient. Set ~1.0 to fall back to GRPO on "
            "those samples."
        ),
    ),
    cliprange: float = 0.2,
    grad_clip: float = 1.0,
    # ----- SDPO knobs ----- #
    full_logit_distillation: bool = typer.Option(
        True, help="Whether to use logit-level KL distillation."
    ),
    distillation_topk: Optional[int] = typer.Option(
        100,
        help=(
            "If set, distill on top-k logits + tail bucket instead of the "
            "full vocab (saves a *lot* of memory, near-lossless)."
        ),
    ),
    distillation_add_tail: bool = typer.Option(
        True, help="Add a tail bucket to the top-k distribution."
    ),
    sdpo_alpha: float = typer.Option(
        0.5,
        help="0=forward KL(s||t), 1=reverse KL(t||s), 0.5=JSD (default).",
    ),
    token_clip: Optional[float] = typer.Option(
        None,
        help=(
            "OPSD (arXiv:2601.18734) per-token pointwise divergence clip: "
            "cap each token's KL/JSD contribution at this max before "
            "averaging. Prevents stylistic tokens ('<think>', 'wait', "
            "newlines) from dominating the gradient. OPSD's reference "
            "value for Qwen3-1.7B is 0.05. None (default) = no clipping."
        ),
    ),
    is_clip: Optional[float] = typer.Option(
        2.0,
        help="Importance-sampling clip on the distillation loss; null disables.",
    ),
    success_reward_threshold: float = typer.Option(
        1.0, help="Sequence reward threshold to count a rollout as 'successful'."
    ),
    dont_reprompt_on_self_success: bool = typer.Option(
        True,
        help=(
            "If True, a sample never uses its own success as its own demo. "
            "(So lone-success groups don't get a teacher signal.)"
        ),
    ),
    remove_thinking_from_demonstration: bool = typer.Option(
        True, help="Strip <think>...</think> from demonstrations."
    ),
    min_demo_thinking_chars: int = typer.Option(
        0,
        help=(
            "Reject successful demonstrations whose <think>...</think> "
            "content has fewer than this many non-whitespace chars. "
            "Setting >0 (e.g. 64) prevents the SDPO collapse mode where a "
            "rollout that emits '<think></think><answer>X</answer>' and "
            "happens to guess the right answer becomes the demo for its "
            "siblings, training the teacher to also skip thinking."
        ),
    ),
    pg_apply_to_all_samples: bool = typer.Option(
        False,
        help=(
            "If True, apply the optional GRPO-style PG term to every "
            "rollout (not just the ones without a teacher signal). "
            "Useful for 'GRPO with KL distillation regulariser' setups; "
            "increases the effective scalar reward signal that anchors "
            "the policy to correct answers, which empirically prevents "
            "the SDPO collapse-to-no-thinking failure."
        ),
    ),
    reprompt_template: Optional[str] = typer.Option(
        None,
        help=(
            "Override the reprompt template fed to the teacher. Must contain "
            "the literal placeholders '{prompt}', '{solution}', '{feedback}'. "
            "Defaults to the SDPO paper's template "
            "('{prompt}{solution}{feedback}\\n\\nCorrectly solve the original "
            "question.'). Use this to e.g. force the teacher to explicitly "
            "produce a thinking trace before answering."
        ),
    ),
    solution_template: Optional[str] = typer.Option(
        None,
        help=(
            "Override the solution sub-template (rendered into '{solution}'). "
            "Must contain '{successful_previous_attempt}'."
        ),
    ),
    teacher_regularization: str = typer.Option(
        "ema",
        help=(
            "'ema' = self-teacher is current policy (or EMA shadow with "
            "teacher_update_rate < 1.0). 'none' = forces teacher = policy "
            "every step."
        ),
    ),
    teacher_update_rate: float = typer.Option(
        1.0,
        help=(
            "EMA decay for the teacher shadow weights. 1.0 (default) = "
            "teacher always equals current policy and we don't materialise "
            "a separate copy. <1.0 keeps a bf16 EMA shadow on GPU."
        ),
    ),
    gt_teacher: bool = typer.Option(
        False,
        help=(
            "OPSD-style teacher (arXiv:2601.18734): instead of conditioning "
            "the teacher on a self-generated successful demo, condition it "
            "on the ground-truth answer from the dataset. Every sample "
            "(whose ground truth is non-empty and not 'Omitted') then gets "
            "a strong, reliable teacher signal, vs ~50-70% coverage from "
            "self-generated demos. Implemented by injecting the ground "
            "truth as the 'demo' into build_reprompts; we also swap in an "
            "OPSD-style reprompt template that phrases it as privileged "
            "information the teacher should rationalise."
        ),
    ),
    gt_teacher_min_chars: int = typer.Option(
        1,
        help=(
            "Minimum non-whitespace chars in a ground-truth string for it "
            "to be considered a usable teacher signal under --gt-teacher. "
            "Filters out 'Omitted', empty strings, etc. BigMath answers "
            "are often single digits ('2', '5') so keep this at 1."
        ),
    ),
    seed: int = 0,
    eval_every: int = 5,
    eval_examples: int = 1024,
    log_all_rollouts: bool = True,
    log_examples_every: int = 1,
    n_log_examples: int = 8,
    device: str = "cuda:0",
    save_every: Optional[int] = None,
    use_wandb: bool = True,
    wandb_project: str = "cs336-sdpo",
    wandb_run_name: Optional[str] = None,
    wandb_mode: str = "online",
):
    """Run SDPO training. See module docstring for an example invocation."""

    # ---------- sanity asserts (mirrors grpo_train) ---------- #
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    valid_pg_losses = {"no_baseline", "reinforce_with_baseline", "grpo_clip"}
    if pg_loss_type not in valid_pg_losses:
        raise ValueError(f"pg_loss_type must be one of {valid_pg_losses}")
    if teacher_regularization not in {"ema", "none"}:
        raise ValueError("teacher_regularization must be one of {'ema', 'none'}")
    if not 0.0 < teacher_update_rate <= 1.0:
        raise ValueError("teacher_update_rate must be in (0, 1].")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in locals().items() if not k.startswith("_")}
    with open(Path(output_dir) / "config.json", "w") as f:
        json.dump(
            {
                k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
                for k, v in config.items()
            },
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
        wandb.define_metric("sdpo/*", step_metric="step")

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

    # ---------- prompt renderers (student-side and per-rollout teacher-side) ---------- #
    if prompt_format == "raw":
        template = load_prompt_template(prompt_name)

        def student_prompt_renderer(exs):
            return format_prompts(exs, template)

        def teacher_prompt_renderer(reprompted_user_strs):
            return [template.format(question=q) for q in reprompted_user_strs]

        sys_prompt = None
    elif prompt_format == "chat":
        sys_prompt = (
            None if prompt_name in ("", "none") else load_system_prompt(prompt_name)
        )

        def student_prompt_renderer(exs):
            return format_chat_prompts(
                exs, tokenizer, system_prompt=sys_prompt, enable_thinking=enable_thinking
            )

        def teacher_prompt_renderer(reprompted_user_strs):
            return [
                _render_chat(
                    q,
                    tokenizer,
                    system_prompt=sys_prompt,
                    enable_thinking=enable_thinking,
                )
                for q in reprompted_user_strs
            ]

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

    # Optional EMA shadow (lazy-initialized so we don't burn extra memory
    # in the common ``teacher_update_rate=1.0`` case).
    ema_teacher: Optional[EmaTeacher] = None
    if teacher_regularization == "ema" and teacher_update_rate < 1.0:
        ema_teacher = EmaTeacher(policy, decay=teacher_update_rate)
        print(
            f"using EMA teacher with decay={teacher_update_rate} "
            f"(adds ~1x policy param memory)"
        )

    # ---------- vLLM rollout engine (colocated on `device`) ---------- #
    llm = init_vllm(
        model_id=model_id,
        device=device,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        enable_sleep_mode=True,
    )
    load_policy_into_vllm_instance(policy, llm)
    sleep_engine(llm, level=1)

    if reward_fn_name in ("r1_zero", "r1_zero_thinking"):
        stop = ["</answer>"]
    else:
        stop = None

    sd_config = SelfDistillationConfig(
        full_logit_distillation=full_logit_distillation,
        distillation_topk=distillation_topk,
        distillation_add_tail=distillation_add_tail,
        alpha=sdpo_alpha,
        is_clip=is_clip,
        token_clip=token_clip,
    )

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
        compact = {
            k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()
        }
        print(f"[step {step}] {compact}")

    for step in range(n_grpo_steps):
        step_t0 = time.time()
        # ---- 1. sample prompts ---- #
        batch_examples = rng.sample(
            train_examples, k=min(n_prompts_per_rollout_batch, len(train_examples))
        )
        student_prompt_strs = student_prompt_renderer(batch_examples)
        user_questions = _format_user_questions(batch_examples)
        ground_truths = [_ground_truth_of(ex) for ex in batch_examples]

        # ---- 2. roll out with vLLM ---- #
        rollout_t0 = time.time()
        wake_engine(llm)
        load_policy_into_vllm_instance(policy, llm)
        sampled = vllm_generate(
            llm,
            student_prompt_strs,
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
        repeated_student_prompts: list[str] = []
        repeated_user_questions: list[str] = []
        repeated_ground_truths: list[str] = []
        for s_prompt, q, gt, group in zip(
            student_prompt_strs, user_questions, ground_truths, sampled
        ):
            for resp in group:
                rollout_responses.append(resp)
                repeated_student_prompts.append(s_prompt)
                repeated_user_questions.append(q)
                repeated_ground_truths.append(gt)

        # ---- 3. compute advantages (kept for optional PG fallback + logging) ---- #
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )

        # ---- 3b. SDPO: pick demos and build reprompted prompts ---- #
        # Reshape rewards/responses to (n_prompts, group_size) for demo picking.
        n_prompts = len(student_prompt_strs)
        rewards_2d = raw_rewards.view(n_prompts, group_size).tolist()
        # rollout_responses is already grouped; reshape into n_prompts x group_size.
        responses_2d = [
            rollout_responses[i * group_size : (i + 1) * group_size]
            for i in range(n_prompts)
        ]

        demos: list[Optional[str]] = []
        if gt_teacher:
            # OPSD mode: teacher conditions on dataset ground-truth answer.
            # Every sample in a prompt's group shares the same gt demo.
            for p in range(n_prompts):
                gt = (ground_truths[p] or "").strip()
                gt_usable = (
                    len(gt.replace(" ", "")) >= gt_teacher_min_chars
                    and "Omitted" not in gt
                )
                demo = gt if gt_usable else None
                for _ in range(group_size):
                    demos.append(demo)
        else:
            for p in range(n_prompts):
                for g in range(group_size):
                    demos.append(
                        pick_successful_demo(
                            rewards_2d[p],
                            responses_2d[p],
                            self_idx=g,
                            success_reward_threshold=success_reward_threshold,
                            dont_reprompt_on_self_success=dont_reprompt_on_self_success,
                            remove_thinking_from_demonstration=remove_thinking_from_demonstration,
                            min_demo_thinking_chars=min_demo_thinking_chars,
                        )
                    )

        if gt_teacher:
            active_reprompt_tmpl = reprompt_template or OPSD_REPROMPT_TEMPLATE
            active_solution_tmpl = solution_template or OPSD_SOLUTION_TEMPLATE
        else:
            active_reprompt_tmpl = reprompt_template or DEFAULT_REPROMPT_TEMPLATE
            active_solution_tmpl = solution_template or DEFAULT_SOLUTION_TEMPLATE
        reprompted_user_strs, has_signal = build_reprompts(
            repeated_user_questions,
            demos,
            feedbacks=None,  # math task has no rich environment feedback
            reprompt_template=active_reprompt_tmpl,
            solution_template=active_solution_tmpl,
            feedback_template=DEFAULT_FEEDBACK_TEMPLATE,
        )
        repeated_teacher_prompts = teacher_prompt_renderer(reprompted_user_strs)
        self_distillation_mask = torch.tensor(
            [1.0 if h else 0.0 for h in has_signal], dtype=torch.float32
        )

        # ---- 4. tokenize student-side and teacher-side ---- #
        student_tok = tokenize_prompt_response_pair(
            repeated_student_prompts, rollout_responses, tokenizer
        )
        teacher_tok = tokenize_prompt_response_pair(
            repeated_teacher_prompts, rollout_responses, tokenizer
        )

        # ---- 5. (off-policy) compute frozen old log probs on student side ---- #
        need_old = (
            (pg_loss_type == "grpo_clip" and pg_loss_weight != 0.0)
            or epochs_per_rollout_batch > 1
            or sd_config.is_clip is not None
        )
        if need_old:
            old_log_probs = compute_old_log_probs(
                policy, student_tok, micro_train_batch_size, torch.device(device)
            )
        else:
            old_log_probs = None

        # ---- 6. train ---- #
        train_t0 = time.time()
        n = rollout_batch_size
        loss_running = 0.0
        distill_loss_running = 0.0
        pg_loss_running = 0.0
        sd_token_running = 0.0
        is_ratio_running = 0.0
        token_clip_frac_running = 0.0
        token_pre_clip_running = 0.0
        token_post_clip_running = 0.0
        n_micro = 0

        for epoch in range(epochs_per_rollout_batch):
            perm = torch.randperm(
                n, generator=torch.Generator().manual_seed(seed + step * 1000 + epoch)
            )

            for tb_start in range(0, n, train_batch_size):
                tb_end = min(tb_start + train_batch_size, n)
                tb_idx = perm[tb_start:tb_end]

                student_tb = {k: v[tb_idx] for k, v in student_tok.items()}
                teacher_tb = {k: v[tb_idx] for k, v in teacher_tok.items()}
                adv_tb = advantages[tb_idx]
                raw_tb = raw_rewards[tb_idx]
                old_tb = old_log_probs[tb_idx] if old_log_probs is not None else None
                sdm_tb = self_distillation_mask[tb_idx]

                optimizer.zero_grad(set_to_none=True)
                for micro in build_microbatches_sdpo(
                    student_tb,
                    teacher_tb,
                    adv_tb,
                    raw_tb,
                    old_tb,
                    sdm_tb,
                    micro_train_batch_size,
                ):
                    s_input_ids = micro["s_input_ids"].to(device)
                    s_labels = micro["s_labels"].to(device)
                    s_response_mask = micro["s_response_mask"].to(device)
                    t_input_ids = micro["t_input_ids"].to(device)
                    t_labels = micro["t_labels"].to(device)
                    t_response_mask = micro["t_response_mask"].to(device)
                    sd_mask = micro["self_distillation_mask"].to(device)

                    # Length of the longest response in this microbatch
                    # (may be slightly smaller than sampling_max_tokens after
                    # the tokenizer reattaches stop-strings).
                    response_len = int(
                        max(
                            s_response_mask.sum(dim=-1).max().item(),
                            t_response_mask.sum(dim=-1).max().item(),
                            1,
                        )
                    )

                    # ---- student forward (with grad) ---- #
                    s_out = get_response_topk_log_probs(
                        policy,
                        s_input_ids,
                        s_labels,
                        topk=sd_config.distillation_topk
                        if sd_config.full_logit_distillation
                        else None,
                    )
                    s_log_probs_full = s_out["log_probs"]
                    s_topk_log_probs_full = s_out.get("topk_log_probs")
                    s_topk_indices_full = s_out.get("topk_indices")
                    s_all_log_probs_full = s_out.get("all_log_probs")

                    # ---- teacher forward (no grad) ---- #
                    # Project student's top-k indices into the teacher's
                    # response slots so we can gather teacher log-probs at
                    # the same vocab positions per response token.
                    if s_topk_indices_full is not None:
                        # First collapse student top-k to response-aligned slots.
                        s_topk_aligned = _gather_response_only_extra(
                            s_topk_indices_full, s_response_mask, response_len
                        )
                        # Then re-scatter into teacher's frame: build
                        # (B, T_t, K) tensor with the indices placed at
                        # teacher response positions and zeros elsewhere.
                        t_T = t_input_ids.shape[1]
                        t_topk_idx_full = s_topk_aligned.new_zeros(
                            (s_topk_aligned.shape[0], t_T, s_topk_aligned.shape[2]),
                            dtype=torch.long,
                        )
                        for i in range(t_topk_idx_full.shape[0]):
                            idxs = (
                                t_response_mask[i]
                                .nonzero(as_tuple=False)
                                .squeeze(-1)
                            )
                            n_resp = idxs.numel()
                            if n_resp > 0:
                                t_topk_idx_full[i, idxs] = s_topk_aligned[i, :n_resp]
                    else:
                        t_topk_idx_full = None

                    teacher_module = policy  # self-teacher = current policy
                    teacher_was_training = teacher_module.training
                    teacher_module.eval()
                    with torch.no_grad():
                        if ema_teacher is not None:
                            # Swap in EMA weights for the teacher pass, then
                            # restore. Cheap but needs *two* policy-shaped
                            # buffers in memory; fine for 1.7B-ish models.
                            backup = {
                                name: p.detach().clone()
                                for name, p in teacher_module.named_parameters()
                            }
                            ema_teacher.copy_into(teacher_module)
                            try:
                                t_out = get_teacher_log_probs_at_indices(
                                    teacher_module,
                                    t_input_ids,
                                    t_labels,
                                    topk_indices=t_topk_idx_full,
                                )
                            finally:
                                # Restore live weights so backward sees the
                                # right params.
                                target_state = dict(teacher_module.named_parameters())
                                for name, p in backup.items():
                                    target_state[name].data.copy_(p)
                                del backup
                        else:
                            t_out = get_teacher_log_probs_at_indices(
                                teacher_module,
                                t_input_ids,
                                t_labels,
                                topk_indices=t_topk_idx_full,
                            )
                    if teacher_was_training:
                        teacher_module.train()

                    t_log_probs_full = t_out["log_probs"]
                    t_topk_log_probs_full = t_out.get("topk_log_probs")
                    t_all_log_probs_full = t_out.get("all_log_probs")

                    # ---- align student/teacher tensors on response slots ---- #
                    s_log_probs = _gather_response_only(
                        s_log_probs_full, s_response_mask, response_len
                    )
                    t_log_probs = _gather_response_only(
                        t_log_probs_full, t_response_mask, response_len
                    )
                    if s_topk_log_probs_full is not None:
                        s_topk = _gather_response_only_extra(
                            s_topk_log_probs_full, s_response_mask, response_len
                        )
                        t_topk = _gather_response_only_extra(
                            t_topk_log_probs_full, t_response_mask, response_len
                        )
                    else:
                        s_topk = None
                        t_topk = None

                    if s_all_log_probs_full is not None:
                        s_all = _gather_response_only_extra(
                            s_all_log_probs_full, s_response_mask, response_len
                        )
                        t_all = _gather_response_only_extra(
                            t_all_log_probs_full, t_response_mask, response_len
                        )
                    else:
                        s_all = None
                        t_all = None

                    aligned_response_mask = _aligned_response_mask(
                        s_response_mask, response_len
                    )

                    if old_tb is not None:
                        old_full = micro["old_log_probs"].to(device)
                        old_log_probs_aligned = _gather_response_only(
                            old_full, s_response_mask, response_len
                        )
                    else:
                        old_log_probs_aligned = None

                    # ---- (optional) per-token PG loss for non-distilled samples ---- #
                    pg_loss_per_token = None
                    if pg_loss_weight != 0.0:
                        adv_kw = micro["advantages"].to(device).unsqueeze(-1)
                        raw_kw = micro["raw_rewards"].to(device).unsqueeze(-1)
                        pg_loss_per_token, _pg_meta = compute_policy_gradient_loss(
                            policy_log_probs=s_log_probs,
                            loss_type=pg_loss_type,
                            raw_rewards=raw_kw,
                            advantages=adv_kw,
                            old_log_probs=old_log_probs_aligned,
                            cliprange=cliprange if pg_loss_type == "grpo_clip" else None,
                        )

                    # ---- backward ---- #
                    loss, meta = sdpo_microbatch_train_step(
                        student_log_probs=s_log_probs,
                        teacher_log_probs=t_log_probs,
                        response_mask=aligned_response_mask,
                        self_distillation_mask=sd_mask,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        config=sd_config,
                        old_log_probs=old_log_probs_aligned,
                        student_topk_log_probs=s_topk,
                        teacher_topk_log_probs=t_topk,
                        student_all_log_probs=s_all,
                        teacher_all_log_probs=t_all,
                        pg_loss=pg_loss_per_token,
                        pg_loss_weight=pg_loss_weight,
                        pg_apply_to_all_samples=pg_apply_to_all_samples,
                    )

                    loss_running += float(loss.detach().item()) * gradient_accumulation_steps
                    distill_loss_running += float(meta["sdpo/distill_loss"].item())
                    pg_loss_running += float(meta["sdpo/pg_loss"].item())
                    sd_token_running += float(meta["sdpo/sd_token_count"].item())
                    if "sdpo/is_ratio_mean" in meta:
                        is_ratio_running += float(meta["sdpo/is_ratio_mean"].item())
                    if "sdpo/token_clip_fraction" in meta:
                        token_clip_frac_running += float(
                            meta["sdpo/token_clip_fraction"].item()
                        )
                        token_pre_clip_running += float(
                            meta["sdpo/token_loss_pre_clip_mean"].item()
                        )
                        token_post_clip_running += float(
                            meta["sdpo/token_loss_post_clip_mean"].item()
                        )
                    n_micro += 1

                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
                )
                optimizer.step()

                # ---- (optional) update EMA teacher shadow ---- #
                if ema_teacher is not None:
                    ema_teacher.update(policy)

        train_time = time.time() - train_t0

        avg_loss = loss_running / max(n_micro, 1)
        avg_distill = distill_loss_running / max(n_micro, 1)
        avg_pg = pg_loss_running / max(n_micro, 1)
        avg_sd_tokens = sd_token_running / max(n_micro, 1)
        avg_is_ratio = is_ratio_running / max(n_micro, 1)

        metrics: dict[str, Any] = {
            "train/loss": avg_loss,
            "train/grad_norm": grad_norm,
            "train/reward_mean": reward_meta["reward_mean"],
            "train/reward_std": reward_meta["reward_std"],
            "train/format_reward_mean": reward_meta["format_reward_mean"],
            "train/answer_reward_mean": reward_meta["answer_reward_mean"],
            "train/group_reward_std_mean": reward_meta["group_reward_std_mean"],
            "sdpo/distill_loss": avg_distill,
            "sdpo/pg_loss": avg_pg,
            "sdpo/sd_token_count_per_micro": avg_sd_tokens,
            "sdpo/sample_with_demo_fraction": float(
                self_distillation_mask.float().mean().item()
            ),
            "sdpo/group_with_success_fraction": float(
                sum(1 for r in rewards_2d if max(r) >= success_reward_threshold)
                / max(n_prompts, 1)
            ),
            "sdpo/is_ratio_mean": avg_is_ratio,
            "sdpo/token_clip_fraction": token_clip_frac_running / max(n_micro, 1),
            "sdpo/token_loss_pre_clip_mean": token_pre_clip_running / max(n_micro, 1),
            "sdpo/token_loss_post_clip_mean": token_post_clip_running / max(n_micro, 1),
            "time/rollout_s": rollout_time,
            "time/train_s": train_time,
            "time/step_s": time.time() - step_t0,
        }

        # ---- 7. rollout dump ---- #
        n_dump = (
            len(rollout_responses)
            if log_all_rollouts
            else min(n_log_examples, len(rollout_responses))
        )
        all_records = []
        for i in range(n_dump):
            rec = {
                "step": step,
                "prompt_idx": i // group_size,
                "rollout_idx": i % group_size,
                "prompt": repeated_student_prompts[i],
                "response": rollout_responses[i],
                "ground_truth": repeated_ground_truths[i],
                "reward": float(raw_rewards[i].item()),
                "advantage": float(advantages[i].item()),
                "has_demo": bool(has_signal[i]),
                "demo": demos[i],
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
                    "has_demo",
                    "demo",
                ],
                data=[
                    [
                        r["step"],
                        r["prompt"],
                        r["response"],
                        r["ground_truth"],
                        r["reward"],
                        r["advantage"],
                        r["has_demo"],
                        r["demo"],
                    ]
                    for r in sample
                ],
            )
            wandb.log({"rollouts": table}, step=step)

        # ---- 8. validation ---- #
        if eval_every > 0 and (step % eval_every == 0 or step == n_grpo_steps - 1):
            eval_t0 = time.time()
            wake_engine(llm)
            load_policy_into_vllm_instance(policy, llm)
            val_metrics = evaluate(
                llm,
                val_examples,
                student_prompt_renderer,
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
