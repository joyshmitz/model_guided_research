# Adjudication — 2026-06-12

- policy: `ci-v5`
- artifacts indexed: 641 from ['artifacts']
- ledger entries appended: 1
- verdicts: refuted: 1
- **0 supported, of which 0 survive FDR at q=0.1 (family: 1 adjudicated - PARTIAL run, not the whole ledger)**
- ledger note: latest recorded verdicts span policies ['ci-v2', 'ci-v3', 'ci-v4']; q-values here are computed fresh under ci-v5 for this run's family only.

| hypothesis | verdict | q | detail |
|---|---|---|---|
| hyp-braid-dyck-depth-extrapolation | refuted-underpowered | 1 | braid: effect=-0.2448 ci95=[-0.3258,-0.1638] (n=8/8) power=29% UNDERPOWERED(need n≈32) |

BLOCKED rows are refusals, not adjudications: the engine declines to rule on weak, mismatched, or tainted evidence. UNDERPOWERED verdicts cleared their threshold at a test with under 50% power to detect the registered effect - an asterisk, not a clean verdict. See `verdicts.json` for machine-readable reasons, p-values, and q-values.
