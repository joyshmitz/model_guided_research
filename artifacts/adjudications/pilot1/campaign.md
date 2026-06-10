# Pilot Campaign 1 — the first preregistered run of the receipts loop

Bead: model_guided_research-kbj2 · Date: 2026-06-10 · Code: commits `951e755` (training) / scorer fix (evals)

## Design

- 11 arm×task combos × 3 training seeds = 33 runs, all from a CLEAN committed tree
  (untainted provenance), equal budget `--target-flops 1e13` (≈242 steps for standard).
- Config: n_layer=2, n_head=4, n_kv_head=2 (GQA), n_embd=64, seq 256, batch 8, CPU.
- Corpora: `mgr gen-tasks` (seed 42, size 4000/task, manifests hashed).
- Eval: `mgr eval-tasks --task <train task> --seeds 0,1,2 --examples 32` per checkpoint.
- Out of scope this round (registry rows stay BLOCKED): needle, bag, placebo, copyops,
  octonion (per-query loop too slow at seq 256), ordinal/hoss arms.

## Results

|---|---|---|---|---|---|---|
| hier | standard | 3 | 1.138 | 0.042 | 0.021 | 3.041 |
| hier | ultrametric | 3 | 1.139 | 0.042 | 0.021 | 3.052 |
| hier | fractal | 3 | 1.140 | 0.042 | 0.021 | 3.052 |
| arith | standard | 3 | 0.927 | 0.448 | 0.500 | 2.771 |
| arith | surreal | 3 | 0.923 | 0.448 | 0.500 | 2.751 |
| dyck | standard | 3 | 0.985 | 0.438 | 0.375 | 2.839 |
| dyck | braid | 3 | 0.984 | 0.438 | 0.375 | 2.823 |
| rot | standard | 3 | 0.794 | 0.073 | 0.031 | 2.414 |
| rot | quaternion | 3 | 0.794 | 0.073 | 0.031 | 2.417 |
| rel | standard | 3 | 1.067 | 0.135 | 0.021 | 2.869 |
| rel | simplicial | 3 | 1.066 | 0.135 | 0.021 | 2.865 |

## Findings

1. **Harness bug found and fixed before adjudication** (the cross-arm sanity check
   did its job): models trained on BOS-packed docs emit the answer immediately
   followed by `<|endoftext|>`, which `decode().split()` glued onto the answer —
   correct answers scored 0. Fixed (scorer stops at the separator id), evals re-run
   on the fixed commit. Identical-EM-across-arms was the tell.

2. **The degenerate-policy regime**: at this budget every mechanism converges to
   the same answer policy — surface format + the majority answer token (verified:
   all arith models answer "lt" on every prompt; per-seed EM vectors are identical
   across arms because EM then measures the eval set's label distribution, not the
   model). Train CE separates from init (10.8 → 0.79–1.14) but task SEMANTICS are
   unlearned by every mechanism equally.

3. **Verdicts** (policy ci-v1): 5 REFUTED (ultrametric-hier, fractal-hier,
   surreal-arith, quaternion-rot, simplicial-rel: effect exactly 0.0, CI95 well
   below the registered +0.05 threshold), 1 INCONCLUSIVE (braid-dyck: wider
   variance, CI straddles), 18 BLOCKED (correct machine-readable reasons,
   including one tainted-evidence refusal of a pre-provenance artifact).

## Interpretation & scope

These refutations are real and scoped: at 1e13 FLOPs on 2-layer/64-dim models, NO
mechanism's claimed advantage manifests, because no mechanism (including standard)
learns the task semantics — the registered effect-size claims are false at this
budget. The registry's scale caveats anticipated exactly this; the ledger is
append-only, so E1-scale runs (bigger models, longer budgets, GPU) will append new
verdicts and update statuses. What this campaign establishes: the loop runs
end-to-end with receipts, the equal-FLOPs comparison machinery works, the harness
artifacts get caught by design, and the bar for "mechanism X helps" now has a
concrete, committed baseline to beat.

## Next steps

- E1-scale rung: larger model + budget where standard attention itself clears the
  degenerate regime (in-range EM well above answer-prior), then re-adjudicate.
- Include the out-of-scope arms (needle/bag/placebo/copyops, octonion after 7b0.6).
- hij.4: power-gated verdicts would have flagged the braid-dyck comparison as
  underpowered rather than merely inconclusive.
