# eval-tasks — e1-rel-standard-s0

- checkpoint: `artifacts/campaigns/e1/rel-standard-s0/checkpoints` @ step 1096
- attention_type: standard
- n_params: 13,598,720
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | curve |
|---|---|---|---|---|
| rel | 0.052 | 0.031 | 2.1/29.8 | [curve](curve_rel.png) |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
