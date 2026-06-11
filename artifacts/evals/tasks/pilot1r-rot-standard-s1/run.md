# eval-tasks — pilot1r-rot-standard-s1

- checkpoint: `artifacts/campaigns/pilot1/rot-standard-s1/checkpoints` @ step 241
- attention_type: standard
- n_params: 6,529,024
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | curve |
|---|---|---|---|---|
| rot | 0.073 | 0.031 | 2.4/2.8 | [curve](curve_rot.png) |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
