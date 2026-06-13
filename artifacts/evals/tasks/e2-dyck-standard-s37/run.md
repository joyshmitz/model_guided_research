# eval-tasks — e2-dyck-standard-s37

- checkpoint: `/data/projects/model_guided_research/artifacts/campaigns/e2-dyck/dyck-standard-s37/checkpoints` @ step 3289
- attention_type: standard
- n_params: 13,598,720
- seeds: [0, 1, 2] · examples/seed: 24 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| dyck | 0.986 | 0.903 | 2.2/3.0 | -0.0882 [-0.1221,-0.0543] | [curve](curve_dyck.png) |

See `summary.json` (schema mgr.evaltasks.v3) for the full contract output and `generations.jsonl` for per-example receipts.
