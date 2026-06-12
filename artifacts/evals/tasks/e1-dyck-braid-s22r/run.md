# eval-tasks — e1-dyck-braid-s22r

- checkpoint: `/data/projects/model_guided_research/artifacts/campaigns/e1/dyck-braid-s22r/checkpoints` @ step 1096
- attention_type: braid
- n_params: 13,598,848
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| dyck | 0.552 | 0.625 | 3.0/4.6 | +0.0347 [-0.0187,+0.0880] | [curve](curve_dyck.png) |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
