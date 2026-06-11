# eval-tasks — e1-hier-ultrametric-s0

- checkpoint: `artifacts/campaigns/e1/hier-ultrametric-s0/checkpoints` @ step 1096
- attention_type: ultrametric
- n_params: 13,600,768
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | curve |
|---|---|---|---|---|
| hier | 0.000 | 0.000 | 1.6/48.4 | [curve](curve_hier.png) |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
