# eval-tasks — pilot1r-dyck-braid-s0

- checkpoint: `artifacts/campaigns/pilot1/dyck-braid-s0/checkpoints` @ step 241
- attention_type: braid
- n_params: 6,529,056
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | curve |
|---|---|---|---|---|
| dyck | 0.438 | 0.375 | 2.8/3.2 | [curve](curve_dyck.png) |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
