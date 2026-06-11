# E1 Campaign — the first powered rung (bead rrs5)

Date: 2026-06-11 · Policy: ci-v2 · Training commits: 97f587c/a201a10/acc9814
(all clean) · Eval commit: 741c8f0 (clean)

## Design (preregistered shape, sized by probe)

- **Rung**: n_layer=4, n_head=4, n_kv_head=2, n_embd=128 (~13.6M params),
  seq 256, batch 8, equal `--target-flops 1e14` (1097 steps), CPU.
  Chosen by a sizing probe (kept out of the evidence pool under
  `/data/tmp/e1_probe`): at this rung the standard baseline decisively
  clears the arith answer-prior floor (probe EM held-out 0.875 vs prior
  0.521); hier remains floored — the next rung's problem.
- **Corpora**: `artifacts/diagnostics_e1` — 20k docs/task, seed 42, so the
  baseline sees mostly unique data instead of re-epoching the pilot's 3.2k.
- **Arms**: the 11 pilot combos (hier: standard/ultrametric/fractal;
  arith: standard/surreal; rot: standard/quaternion; rel:
  standard/simplicial; dyck: standard/braid) + 7 placebo fairness arms
  (standard + the 6 tested mechanisms) × 3 training seeds = **54 runs**,
  all exit 0. Evals: seeds 0,1,2 × 32 examples, mgr.evaltasks.v2 (recorded
  answer-priors + per-example receipts), run-ids `e1-*`.
- **Octonion** deferred (7b0.6 vectorization); needle/copyops/bag tasks not
  yet in the trained battery (their hypotheses stay blocked).

## Verdicts (ledger appends, policy ci-v2 — third entry for each)

| hypothesis | effect (held-out EM Δ) | CI95 | verdict |
|---|---|---|---|
| braid-dyck-depth-extrapolation | **+0.125** | [−0.155, +0.405] | inconclusive |
| surreal-wide-dynamic-range | −0.031 | [−0.143, +0.080] | inconclusive |
| quaternion-rotation-composition | +0.007 | [−0.063, +0.077] | inconclusive |
| simplicial-two-hop-composition | −0.024 | [−0.129, +0.080] | inconclusive |
| ultrametric-hier-heldout-depth | −0.007 | [−0.022, +0.008] | inconclusive (floor_effect) |
| fractal-hier-heldout-depth | 0.000 | [−0.025, +0.025] | inconclusive (floor_effect) |

18 BLOCKED refusals unchanged (incl. the standing tainted-evidence refusal
of the pre-provenance reversible artifact). Budget cohorts worked as
designed: every verdict above adjudicated on the 1e14 cohort, superseding
the pilot 1e13 evidence without a single budget_mismatch block.

## Findings

1. **The floor design validated end-to-end.** arith left the degenerate
   regime exactly as the probe predicted (standard 0.837 mean held-out EM
   vs 0.521 prior, real seed variance 0.81–0.87); hier/rel/rot stayed at
   their floors and the gate correctly converted would-be refutations into
   no-power verdicts at 10× the pilot budget.
2. **braid-dyck is the program's most promising signal**: braid mean 0.573
   vs standard 0.448 (which mostly sat at the constant-"invalid" policy;
   seed 2 escaped to 0.594). Effect +0.125 — 2.5× the registered +0.05 —
   but seed variance swamps n=3. The decisive experiment is more training
   seeds at this exact rung, not more budget.
3. **surreal trends negative on arith** (−0.031): with both arms genuinely
   off-floor, the scale-direction parameterization shows no advantage and
   possibly a small cost. The Welch-t CI (n=3) refuses the refutation the
   pilot's pseudo-replicated z-CI would have stamped — more seeds decide.
4. **Placebo fairness observation** (not yet adjudicable — the for-all
   needs all 10 mechanisms): held-out perplexity on structure-free data
   varies wildly at equal FLOPs (surreal ≈48, ultrametric ≈66, standard
   ≈94, fractal ≈136 across-seed means). Mechanisms differ strongly in
   optimization rate alone — any future "X beats standard" claim must be
   read against this confound, which is precisely what the placebo
   hypothesis exists to police.
5. **Receipts discipline held**: per-example generations.jsonl for all 54
   evals; behavioral claims (constant policies, braid's real attempts) are
   auditable from the artifacts alone. The seed-0 arith arms produced
   byte-identical outputs (different weights, same competence boundary —
   verified per-example; failures concentrate in the extrapolation
   region); seeds 1–2 broke the tie.

## What the next rung needs

- **Seeds, not FLOPs, for braid-dyck and surreal-arith**: at ±0.13 CI
  half-widths, ~6–8 training seeds per arm would resolve both registered
  +0.05 thresholds at the observed variances.
- **A bigger rung (or curriculum) for hier/rel/rot**: 1e14 with 13.6M params
  leaves retrieval/relational/rotation semantics unlearned by every
  mechanism; their floor-gated verdicts will keep superseding honestly.
- **7b0.6** unblocks the octonion arm; the needle/copyops/bag tasks unblock
  three more registered hypotheses.
