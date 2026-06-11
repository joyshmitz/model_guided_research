# eval-tasks — pilot1r-hier-ultrametric-s0

- checkpoint: `artifacts/campaigns/pilot1/hier-ultrametric-s0/checkpoints` @ step 241
- attention_type: ultrametric
- n_params: 6,529,536
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | curve |
|---|---|---|---|---|
| hier | 0.042 | 0.021 | 3.1/16.9 | [curve](curve_hier.png) |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
