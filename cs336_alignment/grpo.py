"""GRPO (Group Relative Policy Optimization) building blocks.

Implements the components described in Section 7 of the assignment writeup:
  - tokenize_prompt_and_output
  - compute_entropy
  - get_response_log_probs
  - compute_group_normalized_rewards
  - compute_naive_policy_gradient_loss
  - compute_grpo_clip_loss
  - compute_policy_gradient_loss (dispatch wrapper)
  - masked_mean / masked_normalize
  - grpo_microbatch_train_step
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, torch.Tensor]:
    """Tokenize prompts and outputs, returning padded ``input_ids``/``labels``
    suitable for next-token-prediction along with a response-only mask.

    ``input_ids`` and ``labels`` are shifted by one (i.e., ``labels`` is
    ``concat(prompt, output)`` without the first token, and ``input_ids`` is
    that same concatenation without the last token). The ``response_mask`` is
    aligned with ``labels`` and is 1 only on positions corresponding to the
    output (response) tokens, and 0 on prompt or padding positions.
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must be the same length")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    prompt_ids: list[list[int]] = [
        tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs
    ]
    output_ids: list[list[int]] = [
        tokenizer.encode(o, add_special_tokens=False) for o in output_strs
    ]
    full_ids: list[list[int]] = [p + o for p, o in zip(prompt_ids, output_ids)]
    full_lens = [len(x) for x in full_ids]
    max_len = max(full_lens)

    batch_size = len(full_ids)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    response_mask_full = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, ids in enumerate(full_ids):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        response_mask_full[i, len(prompt_ids[i]) : len(ids)] = 1

    labels = input_ids[:, 1:].contiguous()
    input_ids_in = input_ids[:, :-1].contiguous()
    response_mask = response_mask_full[:, 1:].contiguous()
    return {
        "input_ids": input_ids_in,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of the categorical distribution implied by ``logits``.

    Computed in a numerically stable way by working with log-softmax rather
    than softmax. Returns a tensor with the final dimension reduced.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def get_response_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Forward ``model`` on ``input_ids`` and gather the log-probability of
    each token in ``labels``. Optionally also return per-position entropy.
    """
    outputs = model(input_ids)
    logits = outputs.logits
    log_probs_full = F.log_softmax(logits, dim=-1)
    log_probs = log_probs_full.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    result: dict[str, torch.Tensor] = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float | int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """SFT loss = ``-log pi`` averaged over response tokens, using
    ``masked_normalize`` then averaging across the batch and dividing by
    ``gradient_accumulation_steps`` before backprop.
    """
    if normalize_constant is None:
        normalize_constant = 1.0
    nll = -policy_log_probs
    per_example = masked_normalize(
        nll, response_mask, dim=-1, normalize_constant=float(normalize_constant)
    )
    loss = per_example.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, {}


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute group-normalized advantages for a rollout batch.

    See Eq. (28) in the writeup. ``rollout_responses`` is laid out so that
    contiguous chunks of size ``group_size`` correspond to the same prompt.
    """
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError(
            "rollout_responses and repeated_ground_truths must have the same length"
        )
    if len(rollout_responses) % group_size != 0:
        raise ValueError(
            "len(rollout_responses) must be divisible by group_size"
        )

    raw_rewards_list: list[float] = []
    format_rewards_list: list[float] = []
    answer_rewards_list: list[float] = []
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        r = reward_fn(response, gt)
        raw_rewards_list.append(float(r["reward"]))
        format_rewards_list.append(float(r.get("format_reward", 0.0)))
        answer_rewards_list.append(float(r.get("answer_reward", 0.0)))

    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    grouped = raw_rewards.view(-1, group_size)

    group_means = grouped.mean(dim=1, keepdim=True)
    advantages = grouped - group_means
    if normalize_by_std:
        group_stds = grouped.std(dim=1, keepdim=True)
        advantages = advantages / (group_stds + advantage_eps)

    advantages = advantages.reshape(-1)

    metadata: dict[str, float] = {
        "reward_mean": float(raw_rewards.mean().item()),
        "reward_std": float(raw_rewards.std(unbiased=False).item()),
        "reward_max": float(raw_rewards.max().item()),
        "reward_min": float(raw_rewards.min().item()),
        "format_reward_mean": float(
            torch.tensor(format_rewards_list, dtype=torch.float32).mean().item()
        ),
        "answer_reward_mean": float(
            torch.tensor(answer_rewards_list, dtype=torch.float32).mean().item()
        ),
        "group_reward_std_mean": float(
            grouped.std(dim=1, unbiased=False).mean().item()
        ),
    }

    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Per-token naive policy-gradient loss: ``-A * log pi(a|s)``."""
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-token GRPO-Clip loss (Eq. 33 in the writeup)."""
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    unclipped_obj = ratio * advantages
    clipped_obj = clipped_ratio * advantages

    per_token_objective = torch.minimum(unclipped_obj, clipped_obj)
    loss = -per_token_objective

    was_clipped = clipped_obj < unclipped_obj
    metadata = {
        "clipped": was_clipped,
        "clip_fraction": was_clipped.float().mean(),
        "ratio": ratio.detach(),
    }
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Wrapper that selects between the supported policy-gradient loss types."""
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards is required for loss_type='no_baseline'")
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}
    if loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError(
                "advantages is required for loss_type='reinforce_with_baseline'"
            )
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}
    if loss_type == "grpo_clip":
        if advantages is None or old_log_probs is None or cliprange is None:
            raise ValueError(
                "advantages, old_log_probs, and cliprange are required for "
                "loss_type='grpo_clip'"
            )
        return compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
    raise ValueError(f"Unknown loss_type: {loss_type!r}")


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Mean of ``tensor`` over positions where ``mask`` is truthy."""
    mask_f = mask.to(tensor.dtype)
    masked = tensor * mask_f
    if dim is None:
        return masked.sum() / mask_f.sum()
    return masked.sum(dim=dim) / mask_f.sum(dim=dim)


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension, dividing by ``normalize_constant``."""
    mask_f = mask.to(tensor.dtype)
    masked = tensor * mask_f
    if dim is None:
        return masked.sum() / normalize_constant
    return masked.sum(dim=dim) / normalize_constant


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    length_normalization: Literal["masked_mean", "masked_normalize"] = "masked_mean",
    normalize_constant: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """One microbatch forward+backward pass for GRPO.

    Computes the per-token policy-gradient loss, reduces it along the sequence
    dimension, averages over the batch, and divides by
    ``gradient_accumulation_steps`` before calling ``backward``.

    The sequence-dim reducer is selectable:

    * ``length_normalization="masked_mean"`` (default): per-example loss is the
      *average* over response tokens (``Σ / num_response_tokens``). This is the
      standard GRPO recipe and weights every sequence equally regardless of
      length, which biases the policy toward shorter rollouts (see Dr-GRPO).
    * ``length_normalization="masked_normalize"``: per-example loss is the
      *sum* over response tokens divided by ``normalize_constant`` (Dr-GRPO
      style ``Σ / L_max``). This removes the per-sequence length bias --
      longer correct rollouts contribute proportionally more gradient. Pick
      ``normalize_constant`` to be the rollout cap (e.g., ``sampling_max_tokens``)
      so the per-token weight matches that of a fully-extended rollout.
    """
    per_token_loss, loss_metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    if length_normalization == "masked_mean":
        per_example_loss = masked_mean(per_token_loss, response_mask, dim=-1)
    elif length_normalization == "masked_normalize":
        if normalize_constant is None or normalize_constant <= 0:
            raise ValueError(
                "length_normalization='masked_normalize' requires a positive "
                "normalize_constant (e.g., sampling_max_tokens)."
            )
        per_example_loss = masked_normalize(
            per_token_loss,
            response_mask,
            dim=-1,
            normalize_constant=float(normalize_constant),
        )
    else:
        raise ValueError(
            f"Unknown length_normalization {length_normalization!r}; expected "
            f"'masked_mean' or 'masked_normalize'."
        )

    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()

    metadata: dict[str, Any] = {**loss_metadata}
    return loss, metadata
