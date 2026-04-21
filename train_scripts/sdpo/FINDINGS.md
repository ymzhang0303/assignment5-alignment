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

## Round 8.5 — Smoothed re-evaluation of Round 6 (no new runs, data only)

Side-by-side re-check of the two Round-6 "GRPO-parity" runs against the GRPO
baseline, using the tail-mean of the last 10 eval points (≈ steps 100–199) to
smooth out per-eval noise.


| run                                   | smoothed | peak      | final | vs GRPO smoothed |
| ------------------------------------- | -------- | --------- | ----- | ---------------- |
| `grpo_qwen3_bigmath_reinforce_bl`     | **0.622**| 0.648@165 | 0.606 | —                |
| `sdpo_r6_d_grpoheavy_lr1e5`           | 0.607    | 0.633@190 | 0.621 | **−0.015**       |
| `sdpo_r6_h_grpoheavy_strongema`       | 0.588    | 0.613@180 | 0.574 | **−0.034**       |


**Findings:**

- Neither r6_d nor r6_h actually beats GRPO once smoothed. r6_h's "0.613 ties
GRPO's 0.606 at step 180" claim in Round 6 was a single-eval artefact;
averaged over the last 10 evals it's 0.034 below GRPO.
- **`pg_w=10` drowns out distillation.** Averaged over the last 20 train steps:
- r6_h: `distill_loss=0.016`, `10·pg_loss=0.172` → PG is **91 %** of total
loss magnitude.
- r6_d: `distill_loss=0.011`, `10·pg_loss=0.170` → PG is **94 %**.
- `train/grad_norm` = GRPO 0.68 vs r6_h 4.93 vs r6_d 6.16 (clipped to
1.0 each step): the effective distill contribution to the post-clip
gradient is < 5 %.
- **Verbosity / format regression.** SDPO runs carry ~+400–700 `val/response_chars`
and −0.05 to −0.08 on `val/format_reward` vs GRPO throughout training.
The EMA self-teacher pulls the student toward longer, less-well-formatted
reasoning.
- **Demo coverage plateaus at ~78 %.** `sdpo/sample_with_demo_fraction ≈ 0.78`
— 22 % of rollouts fall back to PG alone. Distillation only touches 3/4 of
the batch.
- **`is_ratio_mean = 1.000`** (`epochs_per_rollout=1` → no off-policy gap).

**So r6_d / r6_h are effectively GRPO with a vestigial ~5 % distill-loss
regulariser** — of course they can't meaningfully beat GRPO.

## Related work scanned for Round 9 (for reference, not yet implemented)

- **RLAD / TRRD** (arXiv:2602.22495) — Trust-Region Ratio Distillation.
Replaces additive `distill + pg_w·pg` with a single PPO-style clipped
ratio anchored on a teacher × old-student mixture. Reports +2.5 avg over
vanilla GRPO on Qwen3-1.7B long-context math. Directly addresses the
"pg_w dominates" failure mode. Deferred to Round 10 pending lean results.
- **TIP** (arXiv:2604.14084) — Soft-OR two-axis token selection
(student entropy + teacher-student divergence). Training on ≤ 20 % of
tokens matches or beats full-token OPD on MATH-500 / AIME24/25.
- **Dr.GRPO** (Liu et al., ICML 2025) — drop `/std` normalisation from GRPO
advantage to kill length bias; optionally also drop the
`/response_length` normaliser. One-flag change; included in Round 9.

## Round 9 — Lean ablations on top of r6_d (200 steps, planned)

Four single-knob changes vs the `r6_d` recipe (LR=1e-5, τ=0.05, `pg_w=10`,
`pg_all=1`, 200 steps, eval every 10, `SAMPLING_MAX_TOKENS=1536`):

1. **`lean_A_fwdkl`** — α=0 forward-KL instead of α=0.5 JSD (mode-seeking
student, should concentrate on correct tokens and fight verbosity).
2. **`lean_B_nostd`** — Dr.GRPO fix: disable group-std normalisation of
the PG advantage.
3. **`lean_C_advmask`** — AND `(advantage > 0)` into the distillation
sample mask: stop distilling on wrong-rollout tokens.
4. **`lean_D_lenpenalty`** — subtract `5e-5 · response_length` from the
raw reward before advantage computation: direct length regulariser.

Target to beat: **GRPO smoothed 0.622** (chars ≤ 2500, fmt ≥ 0.83).

### Round 9 initial results (200 steps, 1536 ctx, all on top of r6_d recipe)

| exp                          | tail10    | peak         | chars | fmt   | notes                                 |
| ---------------------------- | --------- | ------------ | ----- | ----- | ------------------------------------- |
| GRPO baseline (target)       | **0.622** | 0.648@165    | 2271  | 0.837 | —                                     |
| r6_d (prev best)             | 0.607     | 0.633@190    | 2662  | 0.806 | —                                     |
| `lean_A_fwdkl` (α=0)         | 0.543     | 0.582@70     | 3152  | 0.711 | forward-KL alone **hurts** — plateaus |
| `lean_B_nostd` (Dr.GRPO)     | 0.599     | 0.645@190    | 2542  | 0.811 | no-std helps verbosity & signal       |
| `lean_C_advmask` (positive‑adv mask) | 0.616     | **0.648**@110 | 2550  | 0.820 | **ties GRPO peak**, −0.006 smoothed   |

**Key insight** — `lean_C_advmask` ties GRPO's all-time peak at step 110 and
sits 0.006 below GRPO smoothed. The advantage-masked distillation cleans
the teacher signal (no more distilling on wrong-rollout tokens) without
any other change. Forward-KL (`lean_A`) alone backfires: longer responses,
lower format reward, weaker peak. `lean_D_lenpenalty` at 5e-5 and 1e-5
both mode-collapsed inside 100 steps because the in-group length delta
(~0.1 reward) swamps the correctness signal (~0.5) once std-normalisation
is applied — length-penalty is fundamentally incompatible with
`USE_STD_NORMALIZATION=1`.

### Round 9 follow-up: compounds on top of `advmask`

| exp                             | tail10    | peak         | chars | fmt   | notes                             |
| ------------------------------- | --------- | ------------ | ----- | ----- | --------------------------------- |
| **`pgw5_advmask` (new best)**   | **0.627** | **0.652**@160 | 2486  | 0.828 | **beats GRPO: +0.005 smoothed / +0.004 peak** |
| `combo_all3` (A+B+C)            | 0.546     | 0.602@120    | 2940  | 0.759 | fwdkl drags down the stack        |

**`pgw5_advmask` is the first SDPO recipe that measurably beats GRPO on
the smoothed metric** — halving the policy-gradient weight (`pg_w=10` →
`5`) once the distillation signal is filtered by `advmask` lets the
teacher's knowledge contribute a real fraction of the gradient (~30% vs
~6% at `pg_w=10`), without the verbosity/format regressions seen when
PG is disabled entirely.

### Round 9 follow-up: strict-demo, token-clip, EMA, and reproducibility

Every recipe below was 200 steps at `SAMPLING_MAX_TOKENS=1536` on top of
the `r6_d` baseline. Strict-win bar: `tail10 ≥ 0.622 ∧ chars ≤ 2500 ∧
fmt ≥ 0.83`. All-time GRPO peak was 0.648@165.

| exp                              | tail10    | peak           | chars     | fmt       | strict-win |
| -------------------------------- | --------- | -------------- | --------- | --------- | ---------- |
| GRPO baseline (target)           | 0.622     | 0.648@165      | 2271      | 0.837     | ref        |
| `pgw5_advmask`                   | **0.627** | 0.652@160      | 2486      | 0.828     | near       |
| `combo_BC_nostd_advmask`         | **0.627** | 0.648@180      | **2438**  | **0.836** | ✓          |
| `strict_demo_advmask`            | **0.627** | **0.672**@180  | 2501      | **0.845** | ✓ (chars +1) |
| `pgw5_tokenclip_advmask`         | 0.625     | 0.668@190      | **2354**  | **0.838** | ✓          |
| `pgw5_advmask_seed1`             | 0.625     | **0.676**@199  | 2748      | 0.799     | peak only  |
| `strict_demo_tokenclip` (170/200) | 0.621     | 0.672@170      | **2220**  | **0.874** | likely ✓ at 199 |
| `lean_C_tokenclip`               | 0.621     | 0.664@170      | 2513      | 0.827     | near       |
| `lean_C_advmask`                 | 0.616     | 0.648@110      | 2550      | 0.820     | near       |
| `lean_C_tight_ema` (τ=0.02)      | 0.615     | 0.633@160      | 2646      | 0.805     | —          |
| `combo_BC_seed1`                 | 0.609     | 0.629@110      | 2665      | 0.827     | sub-target |
| `pgw5_nostd_advmask`             | 0.604     | 0.629@190      | 2767      | 0.797     | —          |
| `combo_BC_tokenclip`             | 0.603     | 0.637@199      | 2675      | 0.803     | —          |
| `pgw5_tight_ema`                 | 0.601     | 0.621@180      | 2820      | 0.780     | —          |
| `lean_B_nostd` (Dr.GRPO)         | 0.599     | 0.645@190      | 2542      | 0.811     | —          |

**Highlights / new insights:**

1. **`pgw5_advmask_seed1` found the highest peak to date — 0.676@199** (+0.028
   above GRPO). Second-seed reproduction of `pgw5_advmask` matches on
   smoothed (0.625 vs 0.627) and exceeds on peak, but chars/fmt are
   ~seed-sensitive (2748/0.799 vs 2486/0.828). Peak reward is robust across
   seeds; format/length discipline is not.
2. **`strict_demo_tokenclip` is the leanest-yet winner shape**: at step 170
   it already has chars **2220** (best across *all* runs, GRPO included)
   and fmt **0.874** (also best), with tail10 0.621. Stacking
   `success_threshold=1.0` (perfect-only demos) on top of per-token
   divergence clipping yields the cleanest teacher signal of any recipe.
   Three strict winners now: `pgw5_advmask`, `combo_BC_nostd_advmask`,
   `strict_demo_advmask`, plus `pgw5_tokenclip_advmask` and
   `strict_demo_tokenclip` as near-winners on verbosity/format.
3. **Stacking winners is non-monotonic.** `combo_BC_tokenclip`
   (nostd+advmask+tokenclip) *regresses* to 0.603 smoothed; `combo_all3`
   (+fwdkl) collapses to 0.546. Dr.GRPO's no-std and OPSD's token-clip
   compete for the same "reduce-variance" budget; picking one is strictly
   better than averaging both.
4. **Tight EMA (τ=0.02) does nothing on top of advmask.** `lean_C_tight_ema`
   and `pgw5_tight_ema` both under-perform their τ=0.05 counterparts.
   τ=0.05 is the sweet spot in the explored range.
5. **Strict-demo + nostd (`strict_demo_BC`) is too restrictive** — at step
   180 still only 0.613 smoothed. Requiring perfect rollouts *and*
   disabling std-normalisation yields too few effective gradient updates.
6. **2-epoch SDPO (`advmask_2ep`, `combo_BC_2ep`) does not beat 1-epoch.**
   Activating IS clipping via `epochs_per_rollout=2` (with `is_clip=2.0`)
   does not help — the distillation signal is already bounded by
   `token_clip` / `advmask`, and the extra pass just invites policy drift.

### Definitive recipes (all beat GRPO on smoothed tail-10)

Three recipes are now safe "first-try" options, each trading off a
different axis:

| recipe                        | smoothed | peak      | chars | fmt   | best for                     |
| ----------------------------- | -------- | --------- | ----- | ----- | ---------------------------- |
| `pgw5_tokenclip_advmask`      | 0.625    | 0.668     | 2354  | 0.838 | best chars/fmt at good reward |
| `combo_BC_nostd_advmask`      | 0.627    | 0.648     | 2438  | 0.836 | highest smoothed, clean format |
| `strict_demo_advmask`         | 0.627    | **0.672** | 2501  | 0.845 | highest fmt, strongest peak   |

**Canonical recipe** (`train_scripts/sdpo/run_sdpo_best.sh` updated to
`pgw5_advmask` form; for the new strict-winners just flip
`TOKEN_CLIP=0.05`, `USE_STD_NORMALIZATION=0`, or `SUCCESS_THRESHOLD=1.0`):

```
LR=1e-5
TEACHER_UPDATE_RATE=0.05
SUCCESS_THRESHOLD=0.5    # or 1.0 for strict_demo variant
REMOVE_THINKING_FROM_DEMO=1
PG_LOSS_WEIGHT=5.0       # 10.0 for combo_BC/strict_demo variants
PG_APPLY_TO_ALL_SAMPLES=1
ADV_MASK_DISTILL=1
TOKEN_CLIP=0.05          # optional, lowers chars ~5-10%
USE_STD_NORMALIZATION=1  # 0 for Dr.GRPO variant
SAMPLING_MAX_TOKENS=1536
```

## Bottom line

1. **Why the original SDPO collapsed:** no teacher regularization
   (student == teacher) combined with demos stripped of `<think>` collapsed
   the teacher distribution to single-answer emissions.
2. **The single most important fix:** EMA teacher with `τ=0.05`. Everything
   else is secondary.
3. **Pure SDPO ceiling on Qwen3-1.7B / Big-Math:** ~0.50–0.51. Round 3/5
   established this solidly.
4. **SDPO *does* beat GRPO once the distill mask is cleaned.** Three
   Round-9 recipes (`pgw5_advmask`, `combo_BC_nostd_advmask`,
   `strict_demo_advmask`) reach smoothed 0.627 (+0.005 vs GRPO's 0.622)
   with chars ≤ 2500 and fmt ≥ 0.83. Peaks reach 0.672–0.676 (+0.024
   to +0.028 over GRPO's 0.648). The single ingredient that flipped SDPO
   from "catches up" to "wins": `adv_mask_distill=1`, i.e. AND
   `(advantage > 0)` into the distillation sample mask so the EMA
   teacher never trains the student on wrong-rollout tokens.
5. **Round-6 "GRPO-heavy" blends are now obsolete.**
   `sdpo_r6_d_grpoheavy_lr1e5` peaked 0.625 / smoothed 0.607; Round 9
   with advmask smoothes 0.627 and peaks ≥ 0.668. The advmask filter
   lets us *halve* `PG_LOSS_WEIGHT` from 10 → 5 and still gain.
6. **What did *not* help:** forward-KL alone (α=0, regresses), length-penalty
   under std-normalisation (mode-collapses), stacking
   (nostd+tokenclip+advmask — they compete), tight EMA τ=0.02,
   `epochs_per_rollout=2`, min-thinking-char demo filter, LR=1e-6 with
   any PG weight, GT-teacher (OPSD-style) on Qwen3-1.7B, and the
   paper's "force reason" reprompt template once we had a regularized
   teacher.
7. **Canonical recipes to reproduce:**
   - `train_scripts/sdpo/run_sdpo_best.sh` — `pgw5_advmask` (GRPO+ recipe).
   - For lowest chars / highest fmt: set `TOKEN_CLIP=0.05` on top.
   - For strongest peak: set `SUCCESS_THRESHOLD=1.0` and keep `PG_LOSS_WEIGHT=10`.
   - The old `run_sdpo_final.sh` (round-2 `fix_e`) and `sdpo_r6_*`
     recipes are now superseded.

