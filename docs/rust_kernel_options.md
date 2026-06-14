# Rust / PyO3 Kernel Options — Study

**Bead:** `model_guided_research-2xa`
**Date:** 2026-06-14
**Status:** recommendation note, no code (per bead). Companion to
`docs/triton_hotspot_plan.md` (c6h).

## TL;DR

**Do not put the GPU attention hotpath in Rust/PyO3.** For high-frequency
on-device ops the host↔device copy dominates and kills any Rust speedup —
`bio_inspired_nanochat`'s own optimization plan reached exactly this conclusion
and pivoted to Triton ("*moving data between GPU and CPU for high-frequency
kernels is a performance killer … we must execute on-device using Triton*",
`PLAN_TO_OPTIMIZE_KEY_HOTSPOTS_USING_RUST_PYO3.md`). Reserve Rust/PyO3 for
**CPU-bound, pointer-/bit-heavy, host-side** work that never touches the GPU
hotpath. There is exactly one strong such candidate here.

## Evidence

- **bio abandoned PyO3 for the hotpath.** Its `rust_src/` (`lib.rs`, `moe.rs`,
  `presyn.rs`) is the earlier Rust attempt; the current plan supersedes it with
  Triton for the synaptic kernels.
- **Our measured hotspot is on-GPU.** `docs/flops_validation.md` shows the
  bandwidth-bound mechanism is tropical max-plus — an *attention* op that lives
  on-device every step. That is a Triton target (c6h), **not** a Rust one:
  shipping Q/K/V to the CPU per layer per step would cost orders of magnitude
  more than the compute saved.
- **Rust already pays off where it belongs.** The tokenizer optionally uses
  **`rustbpe`** (`nanochat/tokenizer.py`), a CPU-side, throughput-bound BPE —
  the textbook good fit. This is the existing proof that "Rust for CPU/host
  work" is the right rule, and "Rust for GPU tensor ops" is not.

## Where Rust/PyO3 *could* help in this repo (CPU-side only)

| Candidate | Why Rust fits | Why not Triton/Torch | Tracking |
|---|---|---|---|
| **Ultrametric trie retrieval** | bit-prefix LSH signatures + packed-trie LCP lookup = pointer-chasing / bitset rank-select, branchy, **CPU-only** in the current impl (`ultrametric_attention_torch.py:55`) | GPUs are bad at branchy pointer-chasing; this path is explicitly CPU-only | bead `a1o` (packed trie) — Rust/PyO3 is a viable *implementation* for it |
| Tokenizer BPE | already `rustbpe` | n/a — done | (precedent) |
| Dataloader refill | CPU-bound encode | the bottleneck is *overlap*, not language — fix with a prefetch thread | bead `atkp` (prefer prefetch over a Rust rewrite) |

The ultrametric trie is the only candidate where a Rust/PyO3 kernel would add
something Torch/Triton cannot, **and** it composes with the existing `a1o`
structural bead. Everything else is either already Rust (tokenizer) or better
solved another way (prefetch thread, Triton).

## Toolchain / cost notes (if `a1o` goes Rust)

- **Build:** PyO3 + `maturin`; add a Rust workspace under `rust_src/` (bio has a
  `Cargo.toml`/`Cargo.lock` template to copy). `uv` can drive `maturin build`;
  keep the Python reference path so CI without a Rust toolchain still passes.
- **Boundary:** keep the Rust↔Python boundary **coarse** — pass whole query
  batches and return indices/weights once per layer, never per-token, so the
  FFI/copy cost is amortized (the same lesson, inverted: Rust wins only when the
  CPU work per crossing is large).
- **Parity:** gate against the Python reference trie (and the attention goldens
  harness) exactly as Triton kernels would be (`new_mechanism_checklist.md`).
- **FLOPs:** CPU-side retrieval is not in `estimate_flops` (which models GPU
  matmul/attention); a Rust trie changes wall-clock, not the FLOPs budget.

## Recommendation

1. **No general Rust/PyO3 kernel effort** for the attention/MLP hotpath — use
   Triton (c6h).
2. **If/when `a1o` (ultrametric packed trie) is implemented**, Rust/PyO3 is a
   reasonable backend choice for the CPU-only sublinear-retrieval path; keep a
   Python reference and a coarse FFI boundary.
3. Keep `rustbpe` as the model: Rust earns its place on CPU-bound, branchy,
   throughput work — not on GPU tensor math.

No code changes; this records the decision so no one re-litigates "rewrite the
kernels in Rust" for the GPU hotpath.
</content>
