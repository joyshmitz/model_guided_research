# hqwi — the rescue confirmations: three SUPPORTED verdicts (policy ci-v3)

Date: 2026-06-11 · 42 runs + 8 taint re-runs (frozen-worktree provenance) ·
n=25/arm dyck, n=7/6 placebo (3 placebo runs lost to dirty-window taint;
both arms still clear the registered min_seeds=5).

## Verdicts

| hypothesis | registered | result | verdict |
|---|---|---|---|
| braid-dyck-depth-extrapolation | ≥ +0.05 EM, min 3 seeds | **+0.144 [+0.093, +0.195]**, n=25/25 | **SUPPORTED** |
| braid-dyck-directional | ≥ +0.02 EM (post-sm47 registration) | same evidence | **SUPPORTED** |
| surreal-optimization-rate | placebo ppl ratio ≤ 0.85 | **0.539 [0.438, 0.659]**, n=7/6 | **SUPPORTED** |

## What this means

1. **The program's first confirmed mechanism advantage, at the ORIGINAL
   preregistered effect size.** Braid attention beats standard attention on
   held-out Dyck nesting depths by +0.14 EM — the CI lower bound nearly
   doubles the +0.05 the claim was registered at on 2026-06-10, before any
   E1 data existed. The verdict trajectory tells the discipline's story:
   pilot1 floor-gated (no power) → rrs5 inconclusive at n=3 (CI ±0.28) →
   sm47 effect-real-size-unresolved at n=9 → **supported at n=25**.
2. **The directional sister claim** (registered after sm47, decided by
   post-registration evidence as promised) confirms trivially alongside.
3. **Surreal's rescue**: the held-out-generalization claim died (refuted,
   sm47) but the optimization-rate phenomenon its placebo control exposed is
   real — surreal HALVES structure-free perplexity at equal FLOPs. Per the
   registered protocol this trips hyp-placebo-no-winner BY DESIGN when that
   for-all becomes adjudicable: the mandatory 2x-budget two-cause run (bead
   to file at E2) must separate optimization-rate from harness unfairness
   before any cross-mechanism claim cites placebo-bearing comparisons.

## Provenance notes (the honest ledger of a noisy day)

- 8 dyck runs (s21–24) + 3 placebo runs (std s3/s4, sur s4) were tainted by
  concurrent-agent dirty windows; the dyck eight were re-run from a frozen
  worktree (`ed2664a`, clean) preserving the n=25 design — the placebo three
  were not needed (min_seeds already cleared) and remain excluded.
- 8 dyck evals (s17–20) re-issued as `-r2` after eval-time taint; ci-v3
  drops tainted artifacts before lineage dedupe, so the clean twins govern.
- Verdicts stamped `ci-v3` (the engine gained budget-exempt schemas,
  single-arm predictions, and evaltasks variant resolution from the parallel
  agent's session; all 27 engine tests green at adjudication time).

## Pending from this campaign family

- hyp-maslov-anneal-loss-retention: 9qk3's worktree arms finishing (~1–2h);
  seed-0 preview ratio 1.0002 vs registered ≤ 1.053.
- E2 dyck rung probe (83r6) is now about MARGIN, not existence: how much
  bigger does braid's edge get when the baseline can actually learn?
