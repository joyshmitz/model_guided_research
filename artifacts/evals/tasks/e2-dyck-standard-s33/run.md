# eval-tasks — e2-dyck-standard-s33

- checkpoint: `/data/projects/model_guided_research/artifacts/campaigns/e2-dyck/dyck-standard-s33/checkpoints` @ step 3289
- attention_type: standard
- n_params: 13,598,720
- seeds: [0, 1, 2] · examples/seed: 24 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| dyck | 1.000 | 0.861 | 2.2/3.1 | -0.0975 [-0.1378,-0.0572] | [curve](curve_dyck.png) |

See `summary.json` (schema mgr.evaltasks.v3) for the full contract output and `generations.jsonl` for per-example receipts.
