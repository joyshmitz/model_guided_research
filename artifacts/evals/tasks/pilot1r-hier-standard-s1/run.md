# eval-tasks — pilot1r-hier-standard-s1

- checkpoint: `artifacts/campaigns/pilot1/hier-standard-s1/checkpoints` @ step 241
- attention_type: standard
- n_params: 6,529,024
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | curve |
|---|---|---|---|---|
| hier | 0.042 | 0.021 | 3.0/18.6 | [curve](curve_hier.png) |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
