# eval-tasks — e2-dyck-braid-s35

- checkpoint: `/data/projects/model_guided_research/artifacts/campaigns/e2-dyck/dyck-braid-s35/checkpoints` @ step 3289
- attention_type: braid
- n_params: 13,598,848
- seeds: [0, 1, 2] · examples/seed: 24 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| dyck | 0.750 | 0.569 | 4.5/11.7 | -0.0761 [-0.1400,-0.0122] | [curve](curve_dyck.png) |

See `summary.json` (schema mgr.evaltasks.v3) for the full contract output and `generations.jsonl` for per-example receipts.
