# eval-tasks — pilot1r-arith-standard-s0

- checkpoint: `artifacts/campaigns/pilot1/arith-standard-s0/checkpoints` @ step 241
- attention_type: standard
- n_params: 6,529,024
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | curve |
|---|---|---|---|---|
| arith | 0.448 | 0.500 | 2.8/7.2 | [curve](curve_arith.png) |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
