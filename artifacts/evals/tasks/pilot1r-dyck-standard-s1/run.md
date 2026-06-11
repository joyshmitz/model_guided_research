# eval-tasks — pilot1r-dyck-standard-s1

- checkpoint: `artifacts/campaigns/pilot1/dyck-standard-s1/checkpoints` @ step 241
- attention_type: standard
- n_params: 6,529,024
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | curve |
|---|---|---|---|---|
| dyck | 0.438 | 0.375 | 2.8/3.2 | [curve](curve_dyck.png) |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
