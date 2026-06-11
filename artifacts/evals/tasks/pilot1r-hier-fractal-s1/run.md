# eval-tasks — pilot1r-hier-fractal-s1

- checkpoint: `artifacts/campaigns/pilot1/hier-fractal-s1/checkpoints` @ step 241
- attention_type: fractal
- n_params: 6,529,536
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | curve |
|---|---|---|---|---|
| hier | 0.042 | 0.021 | 3.1/25.0 | [curve](curve_hier.png) |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
