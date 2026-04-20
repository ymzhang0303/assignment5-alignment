# SDPO Experiments: Ablation Groups and Findings

Full picture of the SDPO sweep on **Qwen3-1.7B + Big-Math**, organized by round.
Numbers are `val/answer_reward` on the 256-example validation set unless noted.
GRPO baseline target is **peak 0.648 / final 0.606**
(`runs/grpo_qwen3_bigmath_reinforce_bl`).

## Round 0 — Baseline collapse (pre-ablation)

Single run `runs/sdpo_qwen3_bigmath` using the naive SDPO recipe
(no EMA teacher, `LR=1e-5`, strip thinking from demos, `success_threshold=1.0`).


| run                  | init  | peak    | last        | chars  |
| -------------------- | ----- | ------- | ----------- | ------ |
| `sdpo_qwen3_bigmath` | 0.473 | 0.473@0 | 0.121 @ s37 | **38** |


**Finding:** catastrophic collapse — response length dropped to ~38 chars by
step 10, teacher distribution collapsed to `<answer>X</answer>`. This
motivated every subsequent round.

## Round 1 — Collapse-fix ablations, demo-side only (30 steps, LR=1e-5, no EMA)

Vary what the teacher sees inside the demo; keep everything else at the broken
baseline.


| run                   | change                                        | peak    | last  | chars | verdict           |
| --------------------- | --------------------------------------------- | ------- | ----- | ----- | ----------------- |
| `fix_a_keep_thinking` | keep `<think>` in demo                        | 0.477@0 | 0.102 | 194   | collapses         |
| `fix_b_force_reason`  | reprompt instructs explicit think-then-answer | 0.496@5 | 0.164 | 123   | slowest collapse  |
| `fix_c_both`          | A + B combined                                | 0.465@0 | 0.242 | 184   | the two **fight** |


**Finding:** touching only the demo/instruction **delays but does not prevent**
collapse. The KL has no anchor to correctness, so once the self-teacher drifts,
the student follows.

## Round 2 — Collapse-fix ablations, loss-side (30 steps, built on `fix_b`)

Add a policy-gradient anchor and filter on demo quality.


| run              | change                                 | peak        | last  | chars | verdict                        |
| ---------------- | -------------------------------------- | ----------- | ----- | ----- | ------------------------------ |
| `fix_d_pg_mix`   | PG only on samples without teacher     | 0.512@5     | 0.215 | 145   | marginal                       |
| `fix_e_pg_all`   | PG on **every** rollout, `pg_w=1.0`    | **0.582@5** | 0.297 | 239   | round winner                   |
| `fix_f_min_demo` | reject demos with `<think>` < 64 chars | 0.527@5     | 0.008 | 864   | catastrophic — starves teacher |


**Finding:** a GRPO-style PG term on **all** samples is the first intervention
that keeps the policy anchored to correctness. Demo-quality filtering is worse
than useless because it removes the teacher signal entirely when rollouts
rarely succeed.

## Round 3 — Paper-faithful rewrite (50 steps, 8-way sweep)

After re-reading the SDPO paper and the `lasgroup/SDPO` reference YAML, we
realized we were 10x off on LR and running with an **unregularized (identity)
teacher** — which the paper explicitly calls divergent. Round 3 sweeps each
knob individually on top of `LR=1e-6 + EMA teacher (τ=0.05) + success_thresh=0.5 + remove_thinking_from_demo`.


| run               | lever                        | peak@step    | last  | chars    |
| ----------------- | ---------------------------- | ------------ | ----- | -------- |
| `a_paper_default` | paper recipe                 | 0.418@49     | 0.418 | 3771     |
| `b_ema_strong`    | EMA τ=0.01                   | 0.387@40     | 0.355 | 3813     |
| `c_lr_only`       | LR=1e-6, τ=1.0 (no EMA)      | 0.414@45     | 0.371 | 3765     |
| `**d_ema_only`**  | **LR=1e-5 + EMA τ=0.05**     | **0.496@35** | 0.457 | **3254** |
| `e_grpo_mix`      | + `pg_w=5`, PG on all        | 0.340@15     | 0.320 | 4048     |
| `f_grpo_dominant` | + `pg_w=20`, PG on all       | 0.340@20     | 0.316 | 4033     |
| `g_succ_1p0`      | `success_thresh` back to 1.0 | 0.387@45     | 0.371 | 3810     |
| `h_keep_thinking` | don't strip thinking in demo | 0.363@40     | 0.363 | 3746     |


**Findings:**

- **EMA teacher is the single critical fix.** `c_lr_only` (LR fix without EMA)
plateaus; `d_ema_only` (EMA without LR fix) wins the round.
- **LR=1e-5 is fine once EMA is on.** Our earlier collapse was not the LR —
it was the unregularized teacher.
- **PG mixing at weight 5–20 hurts at this horizon** — the PG grad dominates
and throws the student off the distillation target.
- `success_threshold`, stronger EMA, keeping thinking in demo: all
neutral-to-slightly-worse.

## Round 4 — 100-step hedge on top-3 round-3 recipes

Longevity test around `d_ema_only`.


| run               | (LR, τ)          | peak@step    | last  |
| ----------------- | ---------------- | ------------ | ----- |
| `a_winner`        | (1e-5, 0.05)     | 0.480@35     | 0.438 |
| `b_paper`         | (1e-6, 0.05)     | 0.449@90     | 0.449 |
| `c_midlr`         | (3e-6, 0.05)     | 0.504@80     | 0.457 |
| `**d_strongreg`** | **(1e-5, 0.02)** | **0.512@75** | 0.469 |


**Finding:** at longer horizons `a_winner`'s fast climb peters out; a tighter
EMA (`d_strongreg`, τ=0.02) or lower LR (`c_midlr`) keeps climbing. Pure SDPO
tops out around **0.50**, with the 0.10 gap to GRPO still open.

## Round 5 — 200-step durability


| run             | (LR, τ)      | peak@step | last      | chars |
| --------------- | ------------ | --------- | --------- | ----- |
| `sdpo_r5_midlr` | (3e-6, 0.05) | 0.551@150 | **0.160** | 180   |
| `sdpo_r5_paper` | (1e-6, 0.05) | 0.512@160 | 0.508     | 3026  |


**Finding:** `midlr` peaks highest but then catastrophically collapses
(val 0.55 → 0.16, chars → 180) around step 170 — late-stage teacher drift even
with EMA. `paper` (LR=1e-6) is stable but plateaus 0.10 below GRPO.
**Pure SDPO has a ceiling around 0.51 on this setup.**

## Round 6 — Close the gap to GRPO (200 steps, EMA τ=0.05)

Introduce SDPO + GRPO blending (paper's weak-model recipe) and sweep
LR × `pg_w` × "apply PG to all samples".


| run                         | config                                    | peak@step     | last  | vs GRPO peak (0.648) |
| --------------------------- | ----------------------------------------- | ------------- | ----- | -------------------- |
| `a_pure_lr1e5`              | LR=1e-5, pure SDPO                        | 0.504@70      | 0.504 | −0.144               |
| `b_pgfill_lr1e5`            | LR=1e-5, `pg_w=1`, PG on no-demo only     | 0.496@40      | 0.453 | −0.152               |
| `c_pgall_lr1e5`             | LR=1e-5, `pg_w=1`, PG on all              | 0.578@120     | 0.449 | −0.070               |
| `**d_grpoheavy_lr1e5`**     | **LR=1e-5, `pg_w=10`, PG on all**         | **0.625@150** | 0.598 | **−0.023**           |
| `e_grpoheavy_lr1e6`         | LR=1e-6, `pg_w=10`                        | 0.340@40      | 0.340 | −0.308               |
| `f_grpodomlight_lr1e6`      | LR=1e-6, `pg_w=50`                        | 0.336@60      | 0.293 | −0.312               |
| `g_longer_lr1e6`            | LR=1e-6, pure SDPO, 300 steps             | 0.508@170     | 0.508 | −0.140               |
| `**h_grpoheavy_strongema`** | **LR=1e-5, `pg_w=10`, PG on all, τ=0.02** | **0.609@150** | 0.605 | **−0.039**           |


**Findings:**

- **GRPO-dominant blend at LR=1e-5** (`d` and `h`) finally essentially
**matches GRPO** (peak 0.625 / final 0.598 vs GRPO 0.648 / 0.606).
- **Round-3's finding that PG mixing hurts was a short-horizon artefact.** At
200 steps the PG anchor is what lets SDPO reach GRPO-parity.
- **LR=1e-6 + heavy PG is a dead combo** (`e`, `f`) — the PG update is too
small to move the policy.
- Tighter EMA (`h`) stabilises the tail for a small final-reward gain but
costs ~0.02 peak.

## Round 7 — OPSD-inspired teacher & per-token KL clip (200 steps)

Borrowed from OPSD (arXiv:2601.18734). `GT_TEACHER=1` prompts the teacher with
the ground-truth answer instead of a self-generated demo; `TOKEN_CLIP=0.05`
clips per-token pointwise JSD so "style tokens" (`wait`, `think`) don't
dominate gradients.


| run                        | config                                      | peak@step | last  | verdict         |
| -------------------------- | ------------------------------------------- | --------- | ----- | --------------- |
| `a_gt_pure_lr5e6`          | GT teacher, LR=5e-6, pure SDPO              | 0.383@10  | 0.223 | collapses       |
| `b_gt_grpoheavy_lr1e5`     | GT teacher + r6_d recipe                    | 0.543@50  | 0.512 | worse than r6_d |
| `c_gt_opsd_faithful`       | GT teacher, LR=5e-6, `grad_clip=0.1`, T=1.1 | 0.344@10  | 0.184 | collapses       |
| `d_gt_pglight_lr1e5`       | GT teacher, LR=1e-5, `pg_w=1`, PG on all    | 0.441@70  | 0.426 | weak            |
| `e_tokenclip_on_r6d`       | r6_d recipe + `TOKEN_CLIP=0.05`             | —         | —     | (early)         |
| `f_tokenclip_gt_grpoheavy` | `b` + `TOKEN_CLIP=0.05`                     | —         | —     | (early)         |


**Findings:**

- **GT-teacher is a regression for us.** It over-concentrates teacher
probability on answer tokens, which pulls the student into "gamble the
answer" behavior — especially visible in `a` and `c` where val collapses
from 0.38 → 0.22 within 70 steps.
- The per-token KL clip (`e`, `f`) was the intended mitigation; both were
still warming up at last check.
- **The GT teacher idea does not transfer cleanly from OPSD to our
BigMath/Qwen3-1.7B setting.**

---

## Glossary

- `**pg_w` / `PG_LOSS_WEIGHT`** — scale of the GRPO-style policy-gradient term
added to the SDPO JSD distill loss:
`loss = distill_loss + PG_LOSS_WEIGHT * pg_loss`. `0.0` = pure SDPO,
`1.0` = equal nominal weight (distill dominates numerically),
`10.0` = GRPO-dominant blend (paper's λ≈0.9 weak-model recipe),
`50.0` = almost pure GRPO with SDPO as a regularizer.
- `**PG_APPLY_TO_ALL_SAMPLES**` — `0`: PG only fills rollouts that lack a
teacher demo. `1`: PG runs on every rollout alongside distillation.
- **EMA teacher (`TEACHER_UPDATE_RATE`, a.k.a. τ)** — decay of the shadow
teacher weights: `teacher ← (1-τ)*teacher + τ*student`. `τ=1.0` means
teacher = current policy (unregularized, divergent per paper Table 4).
Paper reference is `τ=0.05`; `0.02` is tighter.
- `**success_threshold`** — minimum reward for a sibling rollout to qualify
as a demo for the teacher. Paper default 0.5; 1.0 requires exact correctness.
- `**GT_TEACHER**` — prompt teacher with the ground-truth answer (OPSD-style)
instead of a self-generated successful sibling demo.
- `**TOKEN_CLIP**` — per-token pointwise JSD clip value. OPSD reports 0.05
for Qwen3-1.7B; suppresses style-token gradient dominance.

## Bottom line

1. **Why the original SDPO collapsed:** no teacher regularization
  (student == teacher) combined with demos stripped of `<think>` collapsed
   the teacher distribution to single-answer emissions.
2. **The single most important fix:** EMA teacher with `τ=0.05`. Everything
  else is secondary.
3. **Pure SDPO ceiling on Qwen3-1.7B / Big-Math:** ~0.50–0.51. Round 3/5
  established this solidly.
4. **To reach GRPO parity you need the SDPO+GRPO blend**
  (`PG_LOSS_WEIGHT≈10` applied to all samples, LR=1e-5, EMA τ=0.05 or 0.02).
   `sdpo_r6_d_grpoheavy_lr1e5` got to peak 0.625, within 0.02 of GRPO's
   0.648; `sdpo_r6_h_grpoheavy_strongema` has the best final at 0.605,
   effectively tying GRPO's 0.606.
5. **What did *not* help:** strict `success_threshold=1.0`, keeping thinking
  in demos, the min-thinking-char demo filter, stronger EMA (τ=0.01) at
   short horizons, LR=1e-6 with any PG weight, GT-teacher (OPSD-style) on
   Qwen3-1.7B, and the paper's "force reason" reprompt template once we had
   a regularized teacher.
6. **Canonical recipes to reproduce:**
  - `train_scripts/sdpo/run_sdpo_best.sh` — pure-SDPO paper-faithful ceiling.
  - `sdpo_r6_d_grpoheavy_lr1e5` (or `h_grpoheavy_strongema`) — GRPO-matching
  blend. Note: the existing `run_sdpo_final.sh` encodes the older
  round-2 `fix_e` recipe and is now superseded.

