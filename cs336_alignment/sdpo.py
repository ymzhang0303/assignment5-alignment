"""SDPO (Self-Distilled Policy Optimization) building blocks.

Implements the algorithm from "Reinforcement Learning via Self-Distillation"
(Hubotter et al., 2026, arXiv:2601.20802) and the reference verl
implementation at https://github.com/lasgroup/SDPO.

The big idea: given a group of rollouts for a prompt, find the ones that the
verifier marks as successful and reuse them as "demonstrations" inside a
*reprompted* user message. The same model conditioned on this reprompted
prompt becomes the **self-teacher**; we then distill that teacher's
next-token distribution back into the **student** (the policy conditioned on
the original prompt). The two distributions are matched on the *same*
response tokens, which gives a dense per-token KL signal to learn from --
much richer than the scalar GRPO advantage alone.

This module provides:

* :func:`remove_thinking_trace` and :func:`pick_successful_demo` -- demo
  selection utilities that match the verl reference.
* :func:`build_reprompts` -- build the reprompted user message for each
  rollout from configurable templates (matches ``actor.self_distillation``
  defaults in the SDPO repo).
* :func:`tokenize_prompt_response_pair` -- tokenize a (prompt, response)
  list into shifted ``input_ids``/``labels``/``response_mask`` (this mirrors
  ``tokenize_prompt_and_output`` in :mod:`cs336_alignment.grpo` and is used
  separately for the student and teacher prompt prefixes so they align on
  the response tokens).
* :func:`get_response_topk_log_probs` -- forward + return top-k log-probs
  (and indices) at every response position for the student.
* :func:`get_teacher_log_probs_at_indices` -- forward + gather log-probs
  at the *student's* top-k indices for the teacher (so KL is computed on the
  same support).
* :func:`compute_self_distillation_loss` -- the SDPO loss with full-logit
  or top-k+tail distillation, alpha-JSD interpolation, and optional IS
  clipping.
* :func:`sdpo_microbatch_train_step` -- one fwd/bwd microbatch, mixing the
  distillation loss with an optional GRPO-style policy-gradient term for
  samples without a teacher signal.

The implementation only requires a single model in memory: the teacher is
the *current policy* under the reprompted prompt (forwarded under
``torch.no_grad()``), exactly matching the ``teacher_regularization='ema'``
default with ``teacher_update_rate=1.0``. We expose hooks for an explicit
EMA teacher copy via :class:`EmaTeacher` for the trust-region-style recipe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------- #
# Demo / reprompt construction
# --------------------------------------------------------------------- #


_THINKING_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_THINKING_CONTENT_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)


def remove_thinking_trace(text: str) -> str:
    """Strip ``<think>...</think>`` blocks from a response.

    Mirrors ``RayPPOTrainer._remove_thinking_trace`` in the SDPO verl fork.
    Useful when feeding a successful rollout from a thinking model back as
    a demonstration to a (potentially non-thinking) reprompted message --
    we don't want to leak the chain-of-thought.
    """
    return _THINKING_RE.sub("", text)


def thinking_content_chars(text: str) -> int:
    """Return the total non-whitespace char count inside ``<think>...</think>``.

    Used by :func:`pick_successful_demo` to filter out demos whose chain of
    thought is empty or trivial (e.g. ``<think></think>``), which would
    otherwise teach the student to skip thinking entirely.
    """
    matches = _THINKING_CONTENT_RE.findall(text)
    return sum(len(m.strip()) for m in matches)


def pick_successful_demo(
    group_rewards: list[float],
    group_responses: list[str],
    self_idx: int,
    *,
    success_reward_threshold: float = 1.0,
    dont_reprompt_on_self_success: bool = True,
    remove_thinking_from_demonstration: bool = True,
    min_demo_thinking_chars: int = 0,
) -> Optional[str]:
    """Choose a successful demonstration from a rollout group.

    Args:
        group_rewards: rewards for every rollout in this prompt's group.
        group_responses: the rollout response strings (same order).
        self_idx: the position of the rollout we're constructing a teacher
            prompt for. When ``dont_reprompt_on_self_success`` is True, this
            rollout is excluded from the candidate pool so we don't reuse a
            sample's own success as its own demonstration.
        success_reward_threshold: minimum sequence-level reward to count as
            "successful". Defaults to 1.0 (a perfect rollout under
            r1_zero-style rewards).
        dont_reprompt_on_self_success: see above.
        remove_thinking_from_demonstration: strip ``<think>...</think>`` from
            the chosen demo (recommended for thinking models so the
            reprompted user message stays clean).

    Returns the demonstration string, or ``None`` if no eligible
    demonstration exists in this group.
    """
    if len(group_rewards) != len(group_responses):
        raise ValueError("group_rewards and group_responses must align")
    candidate_idxs = [
        j for j, r in enumerate(group_rewards) if r >= success_reward_threshold
    ]
    if dont_reprompt_on_self_success:
        candidate_idxs = [j for j in candidate_idxs if j != self_idx]
    if min_demo_thinking_chars > 0:
        # Reject demos whose <think>...</think> is shorter than the
        # threshold. Without this, a successful no-think rollout (e.g.
        # '<think></think><answer>X</answer>' that happened to guess the
        # right number) becomes a demo that pushes the teacher toward
        # "skip thinking" behaviour -- exactly the collapse mode we saw
        # in the baseline SDPO run.
        candidate_idxs = [
            j
            for j in candidate_idxs
            if thinking_content_chars(group_responses[j]) >= min_demo_thinking_chars
        ]
    if not candidate_idxs:
        return None
    # The verl reference picks index 0 of the success list. Because rollouts
    # within a group are sampled i.i.d., the "first" success is effectively a
    # uniform random draw; we keep that convention for byte-for-byte parity.
    demo = group_responses[candidate_idxs[0]]
    if remove_thinking_from_demonstration:
        demo = remove_thinking_trace(demo)
    return demo


# Templates copied verbatim from
# SDPO/verl/trainer/config/actor/actor.yaml (defaults block).
DEFAULT_REPROMPT_TEMPLATE = (
    "{prompt}{solution}{feedback}\n\nCorrectly solve the original question."
)
DEFAULT_SOLUTION_TEMPLATE = "\n\nCorrect solution:\n\n{successful_previous_attempt}"

# OPSD (arXiv:2601.18734) "privileged ground-truth" teacher prompt. The
# teacher sees the reference answer and is asked to rationalise it, yielding
# a strong next-token distribution to distill from, for 100% of rollouts.
OPSD_REPROMPT_TEMPLATE = (
    "{prompt}{solution}{feedback}\n\n"
    "Re-derive the answer above with a complete step-by-step argument. "
    "Show all work, then state the final answer in the required format."
)
OPSD_SOLUTION_TEMPLATE = (
    "\n\nPrivileged information: the correct final answer is "
    "{successful_previous_attempt}"
)
DEFAULT_FEEDBACK_TEMPLATE = (
    "\n\nThe following is feedback from your unsuccessful earlier attempt:"
    "\n\n{feedback_raw}"
)


def build_reprompts(
    user_questions: list[str],
    demos: list[Optional[str]],
    feedbacks: Optional[list[Optional[str]]] = None,
    *,
    reprompt_template: str = DEFAULT_REPROMPT_TEMPLATE,
    solution_template: str = DEFAULT_SOLUTION_TEMPLATE,
    feedback_template: str = DEFAULT_FEEDBACK_TEMPLATE,
    feedback_only_without_solution: bool = True,
) -> tuple[list[str], list[bool]]:
    """Render the reprompted user message for each rollout.

    Returns ``(reprompted_user_text, has_teacher_signal)`` where
    ``has_teacher_signal[i]`` is True iff the i-th rollout will get a
    *different* user message than its original (i.e., a successful demo or
    feedback was injected). Rollouts where ``has_teacher_signal`` is False
    have no teacher signal and are masked out of the distillation loss.

    The default templates exactly match the SDPO repo defaults; pass custom
    ones to experiment.
    """
    n = len(user_questions)
    if len(demos) != n:
        raise ValueError("demos must align with user_questions")
    if feedbacks is None:
        feedbacks = [None] * n
    if len(feedbacks) != n:
        raise ValueError("feedbacks must align with user_questions")

    reprompts: list[str] = []
    flags: list[bool] = []
    for q, demo, fb in zip(user_questions, demos, feedbacks):
        has_demo = demo is not None
        has_fb = bool(fb) and (not feedback_only_without_solution or not has_demo)
        if not (has_demo or has_fb):
            reprompts.append(q)
            flags.append(False)
            continue

        sol_section = (
            solution_template.format(successful_previous_attempt=demo) if has_demo else ""
        )
        fb_section = feedback_template.format(feedback_raw=fb) if has_fb else ""
        text = reprompt_template.format(
            prompt=q, solution=sol_section, feedback=fb_section
        )
        reprompts.append(text)
        flags.append(True)
    return reprompts, flags


# --------------------------------------------------------------------- #
# Tokenization (per-prefix variant of grpo.tokenize_prompt_and_output)
# --------------------------------------------------------------------- #


def tokenize_prompt_response_pair(
    prompt_strs: list[str],
    response_strs: list[str],
    tokenizer,
) -> dict[str, torch.Tensor]:
    """Tokenize ``(prompt, response)`` pairs, padded to the longest sequence.

    Returns shifted ``input_ids`` / ``labels`` and a ``response_mask``
    aligned to ``labels`` (1 only on response positions). Identical to
    :func:`cs336_alignment.grpo.tokenize_prompt_and_output` -- duplicated
    here so we can call it independently for the student-prompt and
    teacher-prompt prefixes that wrap the same response.
    """
    if len(prompt_strs) != len(response_strs):
        raise ValueError("prompt_strs and response_strs must be the same length")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    response_ids = [tokenizer.encode(o, add_special_tokens=False) for o in response_strs]
    full_ids = [p + r for p, r in zip(prompt_ids, response_ids)]
    full_lens = [len(x) for x in full_ids]
    max_len = max(full_lens) if full_lens else 1

    bsz = len(full_ids)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    response_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, ids in enumerate(full_ids):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        response_mask[i, len(prompt_ids[i]) : len(ids)] = 1

    labels = input_ids[:, 1:].contiguous()
    input_ids_in = input_ids[:, :-1].contiguous()
    response_mask = response_mask[:, 1:].contiguous()
    return {
        "input_ids": input_ids_in,
        "labels": labels,
        "response_mask": response_mask,
    }


# --------------------------------------------------------------------- #
# Forward helpers: top-k logits for student/teacher
# --------------------------------------------------------------------- #


def get_response_topk_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    topk: Optional[int],
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Run ``model``, gather per-position log-probs of ``labels``, and
    additionally return top-k log-probs + indices for distillation.

    When ``topk is None`` we return the full vocab log-probs as
    ``all_log_probs`` (heavy memory; only use for tiny vocabs/tests).
    """
    outputs = model(input_ids)
    logits = outputs.logits
    log_probs_full = F.log_softmax(logits, dim=-1)
    label_log_probs = log_probs_full.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    out: dict[str, torch.Tensor] = {"log_probs": label_log_probs}
    if return_token_entropy:
        probs = log_probs_full.exp()
        out["token_entropy"] = -(probs * log_probs_full).sum(dim=-1)

    if topk is None:
        out["all_log_probs"] = log_probs_full
    else:
        topk_lp, topk_idx = torch.topk(log_probs_full, k=topk, dim=-1)
        out["topk_log_probs"] = topk_lp
        out["topk_indices"] = topk_idx
    return out


def get_teacher_log_probs_at_indices(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    topk_indices: Optional[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Forward ``model`` and return:

    * ``log_probs``: per-position log-prob of ``labels`` (used for IS
      ratios / sequence-level reverse KL).
    * ``topk_log_probs``: log-probs at the *student's* ``topk_indices`` so
      KL is computed on the same support. ``None`` when ``topk_indices`` is
      ``None`` (full-logit mode).
    * ``all_log_probs``: full log-softmax (only when ``topk_indices`` is
      ``None``; mainly for the full-vocab path used in tiny tests).

    Always called under ``torch.no_grad()`` by the trainer.
    """
    outputs = model(input_ids)
    logits = outputs.logits
    log_probs_full = F.log_softmax(logits, dim=-1)
    label_log_probs = log_probs_full.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    out: dict[str, torch.Tensor] = {"log_probs": label_log_probs}
    if topk_indices is None:
        out["all_log_probs"] = log_probs_full
    else:
        out["topk_log_probs"] = log_probs_full.gather(dim=-1, index=topk_indices)
    return out


# --------------------------------------------------------------------- #
# SDPO loss
# --------------------------------------------------------------------- #


def _add_tail_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """Append a single tail bucket whose mass = ``1 - sum(top-k probs)``.

    Numerically safe via ``log1p``/``expm1``: lifts the top-k log-probs into
    a proper distribution over ``k + 1`` outcomes. Matches the ``add_tail``
    helper in the SDPO verl fork.
    """
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)


def _renorm_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """Renormalise top-k log-probs to sum to 1 over the top-k support."""
    return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)


@dataclass
class SelfDistillationConfig:
    """Knobs for :func:`compute_self_distillation_loss`.

    Mirrors the fields documented in the SDPO repo's README under the
    "Self-Distillation Configuration" section.
    """

    full_logit_distillation: bool = True
    distillation_topk: Optional[int] = 100
    distillation_add_tail: bool = True
    alpha: float = 0.5
    is_clip: Optional[float] = 2.0
    # Per-token pointwise divergence clipping (OPSD, arXiv:2601.18734).
    # Clips each per-token KL/JSD value at this max before averaging. The
    # OPSD README notes style tokens ("wait", "think", ...) can exhibit
    # 6-15x higher divergence than math tokens and dominate the gradient;
    # clipping stabilises training. Their reference value is 0.05 (1.7B/4B)
    # or 0.06 (8B). None = no clipping (our previous behaviour).
    token_clip: Optional[float] = None


def compute_self_distillation_loss(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    config: SelfDistillationConfig,
    old_log_probs: Optional[torch.Tensor] = None,
    student_all_log_probs: Optional[torch.Tensor] = None,
    teacher_all_log_probs: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    self_distillation_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-token SDPO loss.

    All ``*_log_probs`` are aligned on the *same* response positions. The
    student tensors are differentiable; the teacher tensors are not.

    The returned ``per_token_loss`` is the **JSD/KL contribution** before
    masking; the trainer is responsible for reducing it under
    ``response_mask`` (and applying any sample-level ``self_distillation_mask``).

    When ``full_logit_distillation`` is True we use either the full vocab
    log-probs (``*_all_log_probs``) or the top-k+tail distribution
    (``*_topk_log_probs``) -- the latter is the default and what the SDPO
    paper uses in practice. With ``alpha=0`` we get forward KL
    (``KL(student || teacher)``), ``alpha=1`` reverse KL
    (``KL(teacher || student)``), and intermediate values give the
    generalised Jensen-Shannon divergence (default ``alpha=0.5`` = JSD).

    When ``full_logit_distillation`` is False we fall back to a
    sequence-level reverse-KL surrogate ``(stop_grad(s - t)) * s`` on the
    gathered ``student_log_probs``; only ``alpha=1`` is supported in this
    mode, matching the verl reference.
    """
    metrics: dict[str, torch.Tensor] = {}
    sample_mask = (
        response_mask
        if self_distillation_mask is None
        else response_mask * self_distillation_mask.unsqueeze(1).to(response_mask.dtype)
    )

    if config.full_logit_distillation:
        use_topk = config.distillation_topk is not None
        if use_topk:
            if student_topk_log_probs is None or teacher_topk_log_probs is None:
                raise ValueError(
                    "top-k distillation requires student_topk_log_probs and "
                    "teacher_topk_log_probs"
                )
            s_lp = student_topk_log_probs
            t_lp = teacher_topk_log_probs
            if config.distillation_add_tail:
                s_lp = _add_tail_log_probs(s_lp)
                t_lp = _add_tail_log_probs(t_lp)
            else:
                s_lp = _renorm_log_probs(s_lp)
                t_lp = _renorm_log_probs(t_lp)
        else:
            if student_all_log_probs is None or teacher_all_log_probs is None:
                raise ValueError(
                    "full_logit_distillation=True with topk=None requires "
                    "student_all_log_probs and teacher_all_log_probs"
                )
            s_lp, t_lp = student_all_log_probs, teacher_all_log_probs

        if config.alpha == 0.0:
            # Forward KL: Σ_x p_s(x) (log p_s(x) - log p_t(x)).
            kl = F.kl_div(s_lp, t_lp, reduction="none", log_target=True)
        elif config.alpha == 1.0:
            # Reverse KL: Σ_x p_t(x) (log p_t(x) - log p_s(x)).
            kl = F.kl_div(t_lp, s_lp, reduction="none", log_target=True)
        else:
            alpha_t = torch.tensor(config.alpha, dtype=s_lp.dtype, device=s_lp.device)
            mixture = torch.logsumexp(
                torch.stack(
                    [s_lp + torch.log1p(-alpha_t), t_lp + torch.log(alpha_t)]
                ),
                dim=0,
            )
            kl_t = F.kl_div(mixture, t_lp, reduction="none", log_target=True)
            kl_s = F.kl_div(mixture, s_lp, reduction="none", log_target=True)
            kl = torch.lerp(kl_s, kl_t, alpha_t)
        per_token_loss = kl.sum(dim=-1)
    else:
        if config.alpha != 1.0:
            raise ValueError(
                "Only reverse KL (alpha=1.0) is supported when "
                "full_logit_distillation=False."
            )
        log_ratio = student_log_probs - teacher_log_probs
        # Stop-grad the ratio so this matches reverse KL via the score
        # function (∂/∂θ E_{p_s}[log p_s] = E_{p_s}[(stopgrad·∇ log p_s)]).
        per_token_loss = log_ratio.detach() * student_log_probs

    # OPSD-style per-token pointwise divergence clipping. Capping each
    # token's contribution prevents high-divergence stylistic tokens (e.g.
    # '<think>', 'wait', newlines) from dominating the gradient over the
    # math-content tokens we actually want to distill. Applied *before*
    # importance-sampling reweighting so that a large divergence paired
    # with a large IS ratio can't together blow up the loss.
    if config.token_clip is not None:
        raw_token_loss = per_token_loss
        per_token_loss = per_token_loss.clamp(max=config.token_clip)
        with torch.no_grad():
            live_mask = sample_mask.to(raw_token_loss.dtype)
            denom = live_mask.sum().clamp_min(1.0)
            clipped = (raw_token_loss > config.token_clip).to(raw_token_loss.dtype)
            metrics["token_clip_fraction"] = (clipped * live_mask).sum() / denom
            metrics["token_loss_pre_clip_mean"] = (
                raw_token_loss.detach() * live_mask
            ).sum() / denom
            metrics["token_loss_post_clip_mean"] = (
                per_token_loss.detach() * live_mask
            ).sum() / denom

    if config.is_clip is not None:
        if old_log_probs is None:
            raise ValueError("old_log_probs required when is_clip is set")
        approx_log_ratio = (student_log_probs - old_log_probs).detach()
        approx_log_ratio = torch.clamp(approx_log_ratio, min=-20.0, max=20.0)
        is_ratio = torch.exp(approx_log_ratio).clamp(max=config.is_clip)
        per_token_loss = per_token_loss * is_ratio
        metrics["is_ratio_mean"] = is_ratio.detach().mean()

    metrics["sd_token_count"] = sample_mask.sum().detach()
    return per_token_loss, metrics | {"sd_loss_mask": sample_mask.detach()}


# --------------------------------------------------------------------- #
# Microbatch step
# --------------------------------------------------------------------- #


def _masked_mean(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(t.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (t * mask_f).sum() / denom


def sdpo_microbatch_train_step(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    self_distillation_mask: Optional[torch.Tensor],
    gradient_accumulation_steps: int,
    config: SelfDistillationConfig,
    old_log_probs: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    student_all_log_probs: Optional[torch.Tensor] = None,
    teacher_all_log_probs: Optional[torch.Tensor] = None,
    pg_loss: Optional[torch.Tensor] = None,
    pg_loss_weight: float = 0.0,
    pg_apply_to_all_samples: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """One microbatch fwd+bwd for SDPO.

    Reduces the per-token distillation loss with a token-mean over the
    intersection of ``response_mask`` and ``self_distillation_mask`` (the
    latter zeroes out rollouts with no teacher signal). Optionally mixes in
    a precomputed per-token policy-gradient loss ``pg_loss`` weighted by
    ``pg_loss_weight``.

    By default the PG term is restricted to *non-distilled* samples (so the
    KL term and the PG term don't both push on the same tokens). Set
    ``pg_apply_to_all_samples=True`` to apply PG to every rollout -- useful
    when the goal is "GRPO with an auxiliary distillation regulariser"
    rather than "pure SDPO with a fallback gradient on no-demo samples".
    """
    per_token_distill, distill_meta = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        config=config,
        old_log_probs=old_log_probs,
        student_all_log_probs=student_all_log_probs,
        teacher_all_log_probs=teacher_all_log_probs,
        student_topk_log_probs=student_topk_log_probs,
        teacher_topk_log_probs=teacher_topk_log_probs,
        self_distillation_mask=self_distillation_mask,
    )
    distill_mask = distill_meta.pop("sd_loss_mask")
    distill_loss = _masked_mean(per_token_distill, distill_mask)

    total = distill_loss
    pg_term = torch.tensor(0.0, device=distill_loss.device, dtype=distill_loss.dtype)
    if pg_loss is not None and pg_loss_weight != 0.0:
        if self_distillation_mask is not None and not pg_apply_to_all_samples:
            # Default: PG only on samples without a teacher signal so KL and
            # PG don't fight on the same tokens.
            non_distill = (1.0 - self_distillation_mask.to(pg_loss.dtype)).unsqueeze(1)
            pg_mask = response_mask.to(pg_loss.dtype) * non_distill
        else:
            # Either no per-sample mask, or the user explicitly asked for
            # PG on every rollout (GRPO-with-KL-regulariser mode).
            pg_mask = response_mask.to(pg_loss.dtype)
        pg_term = _masked_mean(pg_loss, pg_mask)
        total = distill_loss + pg_loss_weight * pg_term

    loss = total / gradient_accumulation_steps
    loss.backward()

    metadata: dict[str, Any] = {
        "sdpo/distill_loss": distill_loss.detach(),
        "sdpo/pg_loss": pg_term.detach(),
        "sdpo/sd_token_count": distill_meta["sd_token_count"],
    }
    if "is_ratio_mean" in distill_meta:
        metadata["sdpo/is_ratio_mean"] = distill_meta["is_ratio_mean"]
    for k in ("token_clip_fraction", "token_loss_pre_clip_mean", "token_loss_post_clip_mean"):
        if k in distill_meta:
            metadata[f"sdpo/{k}"] = distill_meta[k]
    return loss, metadata


# --------------------------------------------------------------------- #
# EMA teacher
# --------------------------------------------------------------------- #


class EmaTeacher:
    """Exponential-moving-average shadow of a policy's parameters.

    Used when ``teacher_regularization='ema'`` with
    ``teacher_update_rate < 1.0`` so the teacher lags the student. With
    ``teacher_update_rate=1.0`` (the default in this codebase) the EMA
    teacher reduces to the current policy and we don't need this class --
    just forward the policy itself under no_grad.

    Stored on the same device(s) as the source model. We keep parameters in
    bfloat16 to halve memory; this matches how the SDPO repo treats the
    teacher.
    """

    def __init__(self, source: torch.nn.Module, decay: float = 0.95):
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in source.named_parameters()
        }

    @torch.no_grad()
    def update(self, source: torch.nn.Module) -> None:
        d = self.decay
        for name, param in source.named_parameters():
            self.shadow[name].mul_(1.0 - d).add_(param.detach(), alpha=d)

    @torch.no_grad()
    def copy_into(self, target: torch.nn.Module) -> None:
        """Copy the shadow weights into ``target`` (in-place)."""
        target_state = dict(target.named_parameters())
        for name, shadow in self.shadow.items():
            target_state[name].data.copy_(shadow)
