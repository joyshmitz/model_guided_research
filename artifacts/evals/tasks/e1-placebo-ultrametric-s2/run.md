# eval-tasks — e1-placebo-ultrametric-s2

- checkpoint: `artifacts/campaigns/e1/placebo-ultrametric-s2/checkpoints` @ step 1096
- attention_type: ultrametric
- n_params: 13,600,768
- seeds: [0, 1, 2] · examples/seed: 32 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | curve |
|---|---|---|---|---|
| placebo | - | - | 4.2/61.8 | - |

See `summary.json` (schema mgr.evaltasks.v2) for the full contract output and `generations.jsonl` for per-example receipts.
