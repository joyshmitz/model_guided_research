# eval-tasks — e2-dyck-braid-s37

- checkpoint: `/data/projects/model_guided_research/artifacts/campaigns/e2-dyck/dyck-braid-s37/checkpoints` @ step 3289
- attention_type: braid
- n_params: 13,598,848
- seeds: [0, 1, 2] · examples/seed: 24 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| dyck | 0.819 | 0.681 | 4.2/6.9 | -0.1361 [-0.1898,-0.0824] | [curve](curve_dyck.png) |

See `summary.json` (schema mgr.evaltasks.v3) for the full contract output and `generations.jsonl` for per-example receipts.
