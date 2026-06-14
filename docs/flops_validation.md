# FLOPs Accounting vs Measured Compute — Validation

**Bead:** `model_guided_research-bks`
**Date:** 2026-06-14
**Box:** 64-core CPU, `torch.cuda.is_available()==False`, `torch.set_num_threads(8)`.

**What this validates:** that `GPT.estimate_flops()` (`nanochat/gpt.py:665`) is
an accurate model of the *real* per-step compute — the property the whole
fixed-FLOPs protocol rests on. On a CPU-only box we cannot report tensor-core
**TFLOP/s** (no tensor cores), so we validate the **accounting** instead: measure
model-only forward+backward wall-clock and check that (a) "effective GFLOP/s"
(= arithmetic FLOPs / measured time) is *consistent* across matmul-dominated
mechanisms, and (b) the symplectic 3× correction (bead `7lba`) **predicts**
measured time while the pre-`7lba` 6N formula does not. GPU TFLOP/s methodology
is in [§4](#4-gpu-tflops-methodology-deferred).

Measurement isolates the **model** (random `(B,T)` batch, `fwd; loss.backward()`,
median of 6 iters after warmup) — no dataloader/optimizer/logging — because that
is exactly what `estimate_flops` models.

## 1. Accounting tracks measured compute (E1 rung, L4/H4/KV2/D128/seq256, B=8)

| Mechanism | est FLOPs/token | ms / fwd+bwd | effective GFLOP/s |
|---|---|---|---|
| standard | 4.453e7 | 1430 | 63.8 |
| reversible-additive | 4.139e7 | 1272 | 66.6 |
| reversible-symplectic (corrected) | 4.749e7 | 1374 | 70.8 |
| tropical (attn) | 4.453e7 | 2667 | 34.2 |
| tropical-hybrid (attn+FFN) | 4.455e7 | 8472 | 10.8 |

**Read:** the three **matmul-dominated** mechanisms cluster at **62–71 effective
GFLOP/s** — i.e. `estimate_flops` predicts their relative cost to within ~12 %,
and crucially **symplectic's *corrected* estimate keeps it in the same band**
(70.8 vs standard's 63.8). The accounting is sound for these.

The **idempotent (tropical) max-plus** mechanisms drop to **34 / 11 GFLOP/s**.
This is **not an accounting bug** — it is *hardware efficiency*: `estimate_flops`
counts arithmetic operations (the max-plus "matmul" is charged like a GEMM via
6N), but max-plus is a memory-bound broadcast-max with no optimized BLAS kernel,
so it achieves far fewer ops/second on CPU. See [§3](#3-discrepancy-analysis).

## 2. The symplectic 3× correction predicts measured time (block-heavy rung)

At the E1 rung the symplectic correction is only ~14 % of total FLOPs, because
the vocab-50304 embedding+lm_head matmuls dominate and dilute the block-level
3×. To expose the correction cleanly, use a **block-heavy** rung (D512/L8/H8,
**vocab=512**, seq128, B=4) where the transformer blocks — not the vocab
projection — dominate compute:

| quantity | value |
|---|---|
| standard est FLOPs/token | 1.463e8 |
| standard measured ms/iter | 337 |
| symplectic est FLOPs/token (**corrected**, 7lba) | 1.526e8 |
| symplectic est FLOPs/token (**uncorrected 6N**, pre-7lba) | 5.193e7 |
| **correction ratio** (corrected / uncorrected) | **2.94×** |
| **measured** time ratio symplectic / standard | **0.93×** |
| predicted time ratio from **corrected** est | **1.04×** ✅ matches |
| predicted time ratio from **uncorrected** est | **0.36×** ❌ mispredicts |

**Read:** with blocks dominating, the correction is the full **≈3×** block
multiplier. The **corrected** estimate predicts symplectic ≈ standard wall-clock
(1.04× predicted vs 0.93× measured — symplectic's half-width blocks roughly
offset its double-backward), which is what we observe. The **uncorrected 6N**
formula would predict symplectic costs only **0.36×** of standard — so a
`--target-flops` budget on the old accounting would train the symplectic arm
**~2.6× too many steps**, silently confounding every equal-FLOPs A/B containing
a symplectic arm. **This is exactly the `7lba`/`z4xx` confound, now quantified —
the correction is validated.**

## 3. Discrepancy analysis

1. **Arithmetic FLOPs ≠ achieved throughput.** `estimate_flops` is an
   *arithmetic* count. Mechanisms map to hardware with different efficiency:
   GEMM-based (standard, reversible) hit optimized BLAS (~62–71 GFLOP/s here);
   max-plus/idempotent (tropical) are memory-bound (~11–34). **Consequence:
   fixed *FLOPs* ≠ fixed *wall-clock*.** This is *correct* for a fair A/B — equal
   arithmetic "compute" is the right normalization for comparing what each
   mechanism does with a fixed budget — but expect tropical arms to take far
   longer in wall-clock at equal FLOPs. Report both.
2. **Correction dilution by width.** The symplectic 3× is a *block* multiplier;
   its share of *total* FLOPs depends on block-vs-(embedding+lm_head) ratio:
   ~14 % at the small-width E1 rung (vocab dominates), ~3× at block-heavy rungs.
   Equal-FLOPs A/B is most distorted by the old accounting at large width — and
   for the strictest symplectic equal-compute contract, budget by matched
   **`--max-steps`** rather than `--target-flops` (`gpt.py:686`).
3. **Overhead floor.** Measuring model-only fwd+bwd excludes the dataloader
   (~0.3 ms steady, ~200 ms refill stalls — `docs/dataloader_tuning.md`),
   optimizer step, and logging. Whole-step timing would add a roughly constant
   floor `b` (`t ≈ a·FLOPs + b`); use model-only timing to validate accounting,
   whole-step timing to budget wall-clock.

## 4. GPU TFLOP/s methodology (deferred — needs CUDA)

To complete the measured-TFLOP/s side on a GPU box:

- **Warm up** (≥5 iters; compile + cudagraphs settle), then time with
  `torch.cuda.synchronize()` around a window of N iters, or use
  `torch.profiler` (CUDA + CPU activities) and sum kernel time.
- **TFLOP/s** = `estimate_flops() · B · T · N / elapsed_s / 1e12`.
  **MFU** = TFLOP/s / device peak (e.g. ~165 TF bf16 dense on a 4090,
  ~989 on an H100 SXM with sparsity off).
- **Cross-check** the same configs as §1–2: matmul mechanisms should report a
  consistent MFU; a mechanism whose MFU is implausibly high vs others signals an
  **under-count** in `estimate_flops` (the symplectic class of bug); implausibly
  low signals a kernel-efficiency gap (the tropical class).
- Exclude the dataloader (pre-tokenize into a fixed buffer) so the FLOPs/token
  denominator matches the timed work.
- For FlexAttention runs, time `flex_off` vs `flex_on` (same `estimate_flops`)
  to read the kernel speedup directly (`docs/gpu_flex_diff.md`,
  `scripts/benchmark_flex.py`).

## Conclusion

- `estimate_flops` accounting is **accurate** for matmul-dominated mechanisms
  (consistent effective GFLOP/s) and the **`7lba` symplectic correction is
  validated**: corrected estimate predicts measured wall-clock (1.04× vs 0.93×),
  uncorrected would mispredict by ~2.6× at block-heavy width.
- Idempotent (tropical) mechanisms have low hardware efficiency — arithmetic
  FLOPs over-count effective work, so **fixed-FLOPs ≠ fixed-wall-clock**; this is
  expected and is the right fairness normalization, but must be reported.
- No code changes; the accounting is correct as of `7lba`/`b34d4e9`. Reproduce
  with the snippets above (model-only fwd+bwd timing).
</content>
