# eval-tasks — e1-dyck-braid-s21

- checkpoint: `artifacts/campaigns/e1/dyck-braid-s21/checkpoints` @ step 1096
- attention_type: braid
- n_params: 13,598,848
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| dyck | 0.615 | 0.531 | 3.0/4.5 | -0.0804 [-0.1334,-0.0273] | [curve](curve_dyck.png) |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
