# eval-tasks — e1-rot-quaternion-s2

- checkpoint: `artifacts/campaigns/e1/rot-quaternion-s2/checkpoints` @ step 1096
- attention_type: quaternion
- n_params: 13,598,720
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | curve |
|---|---|---|---|---|
| rot | 0.083 | 0.021 | 2.3/5.5 | [curve](curve_rot.png) |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
