# Adjudication — 2026-06-10

- policy: `ci-v1`
- artifacts indexed: 90 from ['artifacts']
- ledger entries appended: 6
- verdicts: blocked: 18 · inconclusive: 1 · refuted: 5

| hypothesis | verdict | detail |
|---|---|---|
| hyp-fractal-hier-heldout-depth | refuted | fractal: effect=0 ci95=[-0.01444,0.01444] (n=9/9) |
| hyp-quaternion-rotation-composition | refuted | quaternion: effect=0 ci95=[-0.02501,0.02501] (n=9/9) |
| hyp-simplicial-two-hop-composition | refuted | simplicial: effect=0 ci95=[-0.02887,0.02887] (n=9/9) |
| hyp-surreal-wide-dynamic-range | refuted | surreal: effect=0 ci95=[-0.02501,0.02501] (n=9/9) |
| hyp-ultrametric-hier-heldout-depth | refuted | ultrametric: effect=0 ci95=[-0.01444,0.01444] (n=9/9) |
| hyp-braid-dyck-depth-extrapolation | inconclusive | braid: effect=0 ci95=[-0.07502,0.07502] (n=9/9) |
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
