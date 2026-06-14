# Per-op Microbenchmarks (attention / FFN)

**Bead:** `model_guided_research-2cg`
**Harness:** `scripts/microbench_ops.py` (reusable; tracked)
**Artifacts:** `artifacts/microbench/ops_*.json` (gitignored — reproducible
output of the harness, regenerate with the command below)

Isolated timing of a **single** attention op and a **single** FFN op (one
transformer block's sub-modules — not the whole model or training loop), math
mechanism vs vanilla, across shapes/dtypes. This localizes the model-level
hardware-efficiency finding of `docs/flops_validation.md` (bead `bks`) to the
specific op a Triton kernel (bead `c6h`) should target.

## Reproduce

```bash
uv run python scripts/microbench_ops.py                       # CPU, fp32, T∈{256,512}
uv run python scripts/microbench_ops.py --device cuda --dtype bf16
uv run python scripts/microbench_ops.py --seqs 256,512,1024 --out artifacts/microbench/ops_gpu.json
```

## Methodology

- **Per-op isolation:** build a 1-layer `GPT` (for correct RoPE `cos_sin` and
  module construction), extract block 0, time `block.attn(norm(x), cos_sin,
  None)` and `block.mlp(norm(x))` directly. No dataloader, optimizer, lm_head,
  or logging — only the op.
- **Timing:** median of N iters after warmup; `torch.cuda.synchronize()` around
  each iter on CUDA. `fwd` under `no_grad`; `fwd_bwd` backprops `out.pow(2).mean()`.
- **`peak_intermediate_elems`:** analytical size of the attention **score-path**
  intermediate — `(B,H,T,T)` standard vs `(B,H,T,T,head_dim)` tropical (the
  `tropical_attention_torch.py:21` broadcast-then-max). Models the attention
  path only; the tropical FFN's own broadcast is large too but is not in this
  column — read the FFN row's latency.

## Headline finding (CPU, fp32, B=8, D=128, H=4) — `artifacts/microbench/ops_cpu.json`

| op | mech | T | ms_fwd | ms_fwd+bwd | vs standard | peak interm elems |
|---|---|---|---|---|---|---|
| attn | standard | 256 | 3.58 | 9.98 | 1.0× | 2.1M |
| attn | tropical | 256 | 128.1 | 284.6 | **28.5×** | 67.1M |
| attn | tropical | 512 | 502.0 | 1084.3 | **67.8×** | 268.4M |
| ffn | standard | 256 | 1.65 | 6.08 | 1.0× | — |
| ffn | tropical | 256 | 235.8 | 1199.9 | **197×** | — |
| ffn | tropical | 512 | 460.9 | 2428.9 | **102×** | — |

**Read:**

1. **Per-op tropical slowdown (28–197×) ≫ the model-level ~2–6×** seen in `bks`.
   At the model level the vocab-50304 lm_head matmul dominates total time and
   dilutes the tropical block cost; isolating the op exposes the true
   inefficiency. *This is precisely the value of per-op benchmarking* — it makes
   the kernel target unambiguous and quantifies it without lm_head dilution.
2. **The attention-op intermediate grows as T²·d** (67M elems at T=256 → 268M at
   T=512) — confirming the O(T²·d) materialization is the bandwidth bottleneck a
   FlashAttention-style fused max-plus kernel removes (`c6h` Phase-1).
3. **The tropical FFN is the single worst op** (up to 197×) — so `c6h` Phase-2
   (fused max-plus FFN) is at least as important as Phase-1 (attention).

A future Triton kernel's before/after belongs in the same harness
(`--device cuda`), giving a like-for-like speedup record in
`artifacts/microbench/`.
</content>
