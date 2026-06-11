# Pilot1 re-adjudication under ci-v2 — supersession record

Date: 2026-06-10 · Policy: ci-v2 (beads 27q3 / ag73) · Supersedes: the ci-v1
verdicts of `artifacts/adjudications/pilot1` (which remain in the ledger,
stamped `ci-v1` — append-only, nothing rewritten).

## What changed and why

The pilot1 campaign produced 5 REFUTED verdicts in a regime its own
campaign.md documented as degenerate: every arm — standard included —
collapsed to the best-constant-answer policy (answer format + majority
token), so every arm scored exactly the answer-prior of the eval sample.
An experiment in which the baseline never learned the task has **no power**
to surface a mechanism advantage; a null effect there is evidence of *no
power*, not evidence of *absence*. ci-v1's CI rule converted "everything
identical at the floor, tiny variance" into confident-looking refutations.

ci-v2 corrects this and three adjacent statistical defects (full policy in
`cli.py` at `_ADJ_POLICY_VERSION`):

1. **Floor validity gate** — REFUTED now requires the baseline to clear the
   answer-prior floor; below it the verdict is INCONCLUSIVE with
   `floor_effect: true`. The floor is the artifact-recorded `answer_prior`
   (mgr.evaltasks.v2: the best constant-answer score on the exact docs
   scored), falling back to the registered `validity.baseline_floor`.
2. **Honest observation units** — one observation per *trained model*
   (ci-v1 pooled eval seeds as i.i.d.: 3 training runs masqueraded as n=9).
3. **Welch-t CIs** (Satterthwaite df) instead of z=1.96 at small n.
4. **Budget cohorts + lineage dedupe** — larger-budget evidence supersedes
   smaller automatically; re-evals of the same checkpoint never double-count.

This is a post-hoc policy amendment, made transparently: the floors are
objective properties of the task distributions (not tuned to outcomes), the
amendment moves verdicts only in the conservative direction
(refuted -> inconclusive), every verdict is stamped with its policy version,
and the ci-v1 entries stay readable in `verdict_history` forever.

## Evidence

The 33 pilot1 checkpoints were re-evaluated (`pilot1r-*`,
mgr.evaltasks.v2, commit 351bd3b, clean tree) — identical scores to the
pilot1 evals, now with recorded answer-priors and per-example receipts
(`generations.jsonl`, committed). The receipts make the degenerate-policy
claim auditable from the repo alone, e.g. every arith model answers `lt` on
every prompt; per-arm baseline-vs-floor:

| task  | baseline EM (held-out) | recorded answer-prior | at/below floor |
|-------|------------------------|-----------------------|----------------|
| hier  | 0.0208                 | 0.0625                | yes            |
| arith | 0.5000                 | 0.5208                | yes            |
| rot   | 0.0313                 | 0.1354                | yes            |
| rel   | 0.0208                 | 0.1667                | yes            |
| dyck  | 0.3750                 | 0.6250                | yes            |

## Verdict deltas (all six EM hypotheses, ci-v1 -> ci-v2)

| hypothesis                          | ci-v1        | ci-v2                       |
|-------------------------------------|--------------|-----------------------------|
| hyp-ultrametric-hier-heldout-depth  | refuted      | inconclusive (floor_effect) |
| hyp-fractal-hier-heldout-depth      | refuted      | inconclusive (floor_effect) |
| hyp-quaternion-rotation-composition | refuted      | inconclusive (floor_effect) |
| hyp-simplicial-two-hop-composition  | refuted      | inconclusive (floor_effect) |
| hyp-surreal-wide-dynamic-range      | refuted      | inconclusive (floor_effect) |
| hyp-braid-dyck-depth-extrapolation  | inconclusive | inconclusive (floor_effect) |

The 18 BLOCKED refusals are unchanged (same machine-readable reasons,
including the tainted-evidence refusal).

## What decides these hypotheses for real

The E1 rung: a model+budget where the standard baseline itself clears the
answer-prior floor (sizing probe first), 3 training seeds per arm, same
preregistered thresholds. Under ci-v2 budget cohorts, E1 evidence will
supersede this pilot cohort automatically at the next `mgr adjudicate`.
