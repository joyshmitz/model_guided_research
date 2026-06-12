# eval-tasks — e2-dyck-braid-s30

- checkpoint: `/data/projects/model_guided_research/artifacts/campaigns/e2-dyck/dyck-braid-s30/checkpoints` @ step 3289
- attention_type: braid
- n_params: 13,598,848
- seeds: [0, 1, 2] · examples/seed: 24 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| dyck | 0.917 | 0.736 | 4.2/13.2 | -0.1525 [-0.1995,-0.1055] | [curve](curve_dyck.png) |

See `summary.json` (schema mgr.evaltasks.v3) for the full contract output and `generations.jsonl` for per-example receipts.
