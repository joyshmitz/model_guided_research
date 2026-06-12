# eval-tasks — e1-dyck-standard-s20-r2

- checkpoint: `/data/projects/model_guided_research/artifacts/campaigns/e1/dyck-standard-s20/checkpoints` @ step 1096
- attention_type: standard
- n_params: 13,598,720
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | curve |
|---|---|---|---|---|
| dyck | 0.625 | 0.604 | 2.4/2.9 | [curve](curve_dyck.png) |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
