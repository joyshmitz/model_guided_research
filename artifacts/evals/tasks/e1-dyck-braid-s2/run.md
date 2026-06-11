# eval-tasks — e1-dyck-braid-s2

- checkpoint: `artifacts/campaigns/e1/dyck-braid-s2/checkpoints` @ step 1096
- attention_type: braid
- n_params: 13,598,848
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | curve |
|---|---|---|---|---|
| dyck | 0.562 | 0.458 | 2.8/4.1 | [curve](curve_dyck.png) |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
