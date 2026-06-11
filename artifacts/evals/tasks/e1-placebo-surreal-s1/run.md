# eval-tasks — e1-placebo-surreal-s1

- checkpoint: `artifacts/campaigns/e1/placebo-surreal-s1/checkpoints` @ step 1096
- attention_type: surreal
- n_params: 13,600,256
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | curve |
|---|---|---|---|---|
| placebo | - | - | 4.1/49.9 | - |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
