# eval-tasks — pilot1r-rel-simplicial-s2

- checkpoint: `artifacts/campaigns/pilot1/rel-simplicial-s2/checkpoints` @ step 241
- attention_type: simplicial
- n_params: 6,529,028
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | curve |
|---|---|---|---|---|
| rel | 0.135 | 0.021 | 2.9/22.1 | [curve](curve_rel.png) |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
