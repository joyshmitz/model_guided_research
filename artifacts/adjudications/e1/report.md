# Adjudication — 2026-06-11

- policy: `ci-v2`
- artifacts indexed: 231 from ['artifacts']
- ledger entries appended: 6
- verdicts: blocked: 18 · inconclusive: 6

| hypothesis | verdict | detail |
|---|---|---|
| hyp-braid-dyck-depth-extrapolation | inconclusive | braid: effect=0.125 ci95=[-0.1547,0.4047] (n=3/3) |
| hyp-fractal-hier-heldout-depth | inconclusive | fractal: effect=0 ci95=[-0.02499,0.02499] (n=3/3) FLOOR(base 0.006944 <= 0.0625, no power) |
| hyp-quaternion-rotation-composition | inconclusive | quaternion: effect=0.006944 ci95=[-0.06311,0.077] (n=3/3) |
| hyp-simplicial-two-hop-composition | inconclusive | simplicial: effect=-0.02431 ci95=[-0.1289,0.08027] (n=3/3) |
| hyp-surreal-wide-dynamic-range | inconclusive | surreal: effect=-0.03125 ci95=[-0.1426,0.08009] (n=3/3) |
| hyp-ultrametric-hier-heldout-depth | inconclusive | ultrametric: effect=-0.006944 ci95=[-0.02188,0.007995] (n=3/3) FLOOR(base 0.006944 <= 0.0625, no power) |
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
