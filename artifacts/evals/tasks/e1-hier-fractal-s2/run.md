# eval-tasks — e1-hier-fractal-s2

- checkpoint: `artifacts/campaigns/e1/hier-fractal-s2/checkpoints` @ step 1096
- attention_type: fractal
- n_params: 13,600,768
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | curve |
|---|---|---|---|---|
| hier | 0.021 | 0.000 | 1.7/30.5 | [curve](curve_hier.png) |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
