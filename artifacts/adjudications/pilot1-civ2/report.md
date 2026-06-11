# Adjudication — 2026-06-10

- policy: `ci-v2`
- artifacts indexed: 123 from ['artifacts']
- ledger entries appended: 6
- verdicts: blocked: 18 · inconclusive: 6

| hypothesis | verdict | detail |
|---|---|---|
| hyp-braid-dyck-depth-extrapolation | inconclusive | braid: effect=0 ci95=[0,0] (n=3/3) FLOOR(base 0.375 <= 0.625, no power) |
| hyp-fractal-hier-heldout-depth | inconclusive | fractal: effect=0 ci95=[0,0] (n=3/3) FLOOR(base 0.02083 <= 0.0625, no power) |
| hyp-quaternion-rotation-composition | inconclusive | quaternion: effect=0 ci95=[0,0] (n=3/3) FLOOR(base 0.03125 <= 0.1354, no power) |
| hyp-simplicial-two-hop-composition | inconclusive | simplicial: effect=0 ci95=[0,0] (n=3/3) FLOOR(base 0.02083 <= 0.1667, no power) |
| hyp-surreal-wide-dynamic-range | inconclusive | surreal: effect=0 ci95=[0,0] (n=3/3) FLOOR(base 0.5 <= 0.5208, no power) |
| hyp-ultrametric-hier-heldout-depth | inconclusive | ultrametric: effect=0 ci95=[0,0] (n=3/3) FLOOR(base 0.02083 <= 0.0625, no power) |
| hyp-braid-length-generalization | blocked | metric_missing [braid] |
| hyp-gauge-grad-stability-demo | blocked | prediction_not_operationalized |
| hyp-gauge-gradient-stability | blocked | prediction_not_operationalized |
| hyp-gauge-bag-conservation | blocked | no_candidate_artifacts [gauge] |
| hyp-hoss-stiff-lr-robustness | blocked | prediction_not_operationalized |
| hyp-octonion-rotation-composition | blocked | no_candidate_artifacts [octonion] |
| hyp-octonion-norm-stability | blocked | prediction_not_operationalized |
| hyp-ordinal-cosine-parity-simple | blocked | no_candidate_artifacts [ordinal] |
| hyp-ordinal-regime-shift-recovery | blocked | no_candidate_artifacts [ordinal] |
| hyp-reversible-activation-memory | blocked | tainted_evidence [reversible] |
| hyp-reversible-gradient-stability | blocked | prediction_not_operationalized |
| hyp-group-nonsolvable-barrier | blocked | prediction_not_operationalized |
| hyp-surreal-scaling-axis-prediction | blocked | prediction_not_operationalized |
| hyp-tropical-certified-robustness | blocked | prediction_not_operationalized |
| hyp-tropical-needle-no-dilution | blocked | no_candidate_artifacts [tropical] |
| hyp-placebo-no-winner | blocked | no_candidate_artifacts [tropical] |
| hyp-ultrametric-needle-long-context | blocked | metric_missing [ultrametric] |
| hyp-ultrametric-trie-decode-speedup | blocked | prediction_not_operationalized |

BLOCKED rows are refusals, not adjudications: the engine declines to rule on weak, mismatched, or tainted evidence. See `verdicts.json` for machine-readable reasons.
