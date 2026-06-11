# eval-tasks — e1-arith-standard-s2

- checkpoint: `artifacts/campaigns/e1/arith-standard-s2/checkpoints` @ step 1096
- attention_type: standard
- n_params: 13,598,720
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | curve |
|---|---|---|---|---|
| arith | 1.000 | 0.833 | 2.5/11.2 | [curve](curve_arith.png) |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
