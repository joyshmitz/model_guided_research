# sm47 — E1 seed expansion: the program's first powered refutation

Date: 2026-06-11 · Policy: ci-v2 · 24 runs (dyck-{standard,braid} +
arith-{standard,surreal} × seeds 3–8) at the E1 rung, pooled with rrs5's
seeds 0–2 into the 1e14 cohort: **n = 9 training runs per arm**. All runs
exit 0, clean provenance; evals mgr.evaltasks.v2 with receipts, committed.

## Verdicts (ledger appends — fourth entry for each)

| hypothesis | n=3 (rrs5) | n=9 (sm47) | verdict |
|---|---|---|---|
| surreal-wide-dynamic-range | −0.031 [−0.143, +0.080] | **−0.009 [−0.047, +0.029]** | **REFUTED** |
| braid-dyck-depth-extrapolation | +0.125 [−0.155, +0.405] | **+0.101 [+0.012, +0.190]** | inconclusive |

## What the verdicts mean

1. **The first genuine refutation.** Unlike pilot1's floor-effect artifacts
   (superseded for exactly that reason), this one is real: both arith arms
   sit far above the answer-prior floor (0.83 vs 0.52 — the floor gate
   checked and stood aside), nine independent training runs per arm, and the
   CI lies entirely below the registered +0.05. The surreal scale-direction
   parameterization confers **no held-out advantage on wide-dynamic-range
   comparison at the E1 rung** — if anything a small cost. The claim dies at
   this scale honestly, supersedable by bigger rungs through the cohort
   mechanism as always.
2. **braid-dyck: a real effect, an unresolved size.** The CI now excludes
   zero — braid attention genuinely beats standard on held-out Dyck depths
   (+0.10 mean over nine seeds each) — but the registered claim is ≥ +0.05
   and +0.05 sits inside [+0.012, +0.190]. Honest status: inconclusive.
   At sd ≈ 0.10 (braid arm), resolving ±0.04 needs ~25 seeds/arm — at that
   point a bigger rung (where the standard baseline escapes its constant
   policy more often than 1-in-9 runs) is likely cheaper than more seeds:
   the variance is largely the baseline's bimodality at the floor.
3. **Variance behaved as the t-CI predicted**: the n=3 → n=9 CI shrink
   (±0.28 → ±0.089 for braid) tracks 1/√3 with the df gain — the ci-v2
   clustering produced honest small-n intervals that converged instead of
   flip-flopping.

## Next decisions this enables

- Retire seed-expansion as the braid strategy; the next braid evidence
  should come from a rung where the dyck baseline learns (E2 sizing probe).
- The arith task is now a **validated discriminator** (powered, off-floor,
  low variance) — the cheapest place to test any mechanism's claimed
  advantage, and the natural first target for the 9qk3 annealing
  experiments (rgyl's variant selectors landed in the same push).
