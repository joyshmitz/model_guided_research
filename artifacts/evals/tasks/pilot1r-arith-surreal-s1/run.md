# eval-tasks — pilot1r-arith-surreal-s1

- checkpoint: `artifacts/campaigns/pilot1/arith-surreal-s1/checkpoints` @ step 241
- attention_type: surreal
- n_params: 6,529,408
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | curve |
|---|---|---|---|---|
| arith | 0.448 | 0.500 | 2.7/6.8 | [curve](curve_arith.png) |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
