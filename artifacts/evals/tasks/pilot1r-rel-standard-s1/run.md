# eval-tasks — pilot1r-rel-standard-s1

- checkpoint: `artifacts/campaigns/pilot1/rel-standard-s1/checkpoints` @ step 241
- attention_type: standard
- n_params: 6,529,024
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | curve |
|---|---|---|---|---|
| rel | 0.135 | 0.021 | 2.9/21.0 | [curve](curve_rel.png) |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
