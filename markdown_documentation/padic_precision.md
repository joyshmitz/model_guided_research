# p-adic Precision: Quantization with Non-Accumulating Error and Hensel-Lift Curricula

*Bead: model_guided_research-8gk.4 (EPIC THEORY-I, 8gk). Implementation:
`nanochat/ultrametric_attention_torch.py` (`ultrametric_digits_k`,
`mgr eval-tasks --model-override`); Hensel prototype:
`ultrametric_worlds_and_p_adic_computation.py`
(`run_hensel_curriculum_section`); exact tests:
`tests/test_algebraic_properties.py` (flat-error section); theorems:
`thm-flat-error`, `thm-hensel-residue-preservation`; registry:
`hyp-padic-*`. Depends on the valuation dictionary (8gk.2): the digit
machinery contract and the semantics of valuation truncation.*

Post-training quantization is handled industry-wide by calibration heuristics
with empirical cliffs. In p-adic arithmetic, truncating to k digits is
metrically exact and *structurally stable* in a way float rounding is not:
the precision axis carries theorems. This note states them, draws the honest
boundary where they end, and turns each into a measurable, preregistered
observable.

## 1) The flat-error lemma

**Lemma (flat error).** In a field with non-archimedean valuation v, for
**k ≥ 0**: if v(e_i) ≥ k for every member of a finite family, then

    v(Σ_i e_i) ≥ k        and        v(Π_i (1 + e_i) − 1) ≥ k.

*Proof.* The sum part is the strong triangle inequality iterated:
v(a + b) ≥ min(v(a), v(b)). For the product part, expand
Π(1+e_i) − 1 = Σ_{∅≠S} Π_{i∈S} e_i; each subset term has valuation
Σ_{i∈S} v(e_i) ≥ |S|·k ≥ k — the last step **requires k ≥ 0** — and the sum
part finishes. ∎

N truncation errors of size p^{−k} sum to an error *still of size p^{−k}*,
versus √N growth (or worse) in archimedean floating point: quantization error
in the ultrametric world is **flat in depth and width**. Exact-integer
property tests (`test_flat_error_lemma_sum_exact` / `_product_exact`,
Hypothesis over p ∈ {2,3,5} with adversarial signs) assert both parts with no
tolerances.

**The k ≥ 0 hypothesis is sharp** (the bead's precision note, kept aligned
with vnl.2's formalization): with v(e₁) = v(e₂) = −1 the cross term e₁e₂ has
valuation −2 < −1 — `test_flat_error_product_part_requires_nonnegative_k`
exhibits exactly this rational counterexample. In the digit-truncation
setting k is a digit depth ≥ 1, so the hypothesis is automatic; the prose and
the formal statement both carry it explicitly anyway.

## 2) Quantization as valuation truncation

For network components whose arithmetic is p-adic-native — the ultrametric
digit machinery, IFS addresses, integer-coefficient tropical scores —
low-precision inference is **digit truncation**: keep the first k of K digit
channels. `ultrametric_digits_k` implements this for all three ultrametric
paths (kernel, balltree, trie) at eval time:

- the slice is the identity at k = K (bitwise parity with full precision,
  asserted), and kernel/balltree agree exactly under truncation (they compute
  the same function; verified);
- the trie path truncates structurally (a depth-k trie), which is the
  KV-cache-compression form: cached digit *keys* at k digits, with the lemma
  guaranteeing per-entry truncation errors cannot compound across thousands
  of cached entries — the long-context story float-quantized caches cannot
  tell;
- `mgr eval-tasks --model-override ultrametric_digits_k=k` evaluates the SAME
  checkpoint at any k, records the override into
  `meta.checkpoint.model_config` (so per-k arms are variant-selectable
  evidence for the verdict engine), and folds it into the provenance hash.

**The honest boundary — the archimedean interface.** The lemma governs the
digit-keyed path: which keys land in which balls, the LCP depths, the α^lcp
weight *pattern*. It does NOT govern the float-valued V aggregation: once
ball membership changes, the output moves by a float-valued difference of
value vectors, and that leak is archimedean. Precisely: truncation at digit k
changes out(q) only through reassignment of keys between the depth-≥k shells;
the leaked error is bounded by the α-weighted mass of exactly those
reassigned shells — measurable per query, and the quantity the interface-leak
instrumentation should log (valuations of everything crossing the boundary).
Characterizing which fraction of the network can live on the p-adic side *is*
the scorecard, not a footnote.

## 3) Hensel lifting as curriculum

**Theorem (residue preservation, classical).** Hensel lifting — the p-adic
Newton method — refines a solution mod p^j to one mod p^{j+1} while
preserving the previous residue *exactly*.

**Training translation.** Learn digit-0..j−1 structure first (a K=j model),
then lift: the stage-j trie nodes become **frozen** (their (S, R) data
immutable — the lift fixes the residue) and only deeper or newly-created
structure trains. Float curricula have no such invariant: early learning is
routinely overwritten. The prototype
(`run_hensel_curriculum_section` + `hensel_lift_model`):

- `HeadTrie.frozen_mark` marks every node existing at lift time; VOLF's
  ancestor scan skips frozen nodes (and a fully-frozen path is a loud no-op,
  never a silent overwrite); nodes created after the lift stay writable even
  at shallow depths — new coarse structure on unseen inputs refines the
  function without touching the lifted solution;
- the invariant is asserted bit-exactly: `_assert_residues_preserved`
  compares every lift-time node's (residue, S, R) after all stage-2 training
  against the lift-time snapshot — a single violated residue fails the run;
- the comparison arm is end-to-end full-depth training at the SAME total
  epoch budget, on the same Task A data.

## 4) Mahler-basis heads (design)

Continuous functions ℤ_p → ℚ_p have canonical expansions
f = Σ_n a_n·C(x, n) (Mahler's theorem; a_n → 0), and truncating the series is
*canonical compression with certified ultrametric error* ‖f − f_N‖ =
max_{n>N} |a_n|. Trie addresses ARE elements of ℤ_p, so heads operating on
hierarchical addresses can be Mahler polynomials instead of MLPs: the
function class with a principled truncation knob whose error certificate is
again non-archimedean (and so composes with the flat-error lemma rather than
fighting it). Design notes: binomial features C(x, n) for n ≤ N computed from
digit prefixes; coefficients learned; N is the precision axis. Left as the
implementation follow-on — the bead's deliverable here is the construction
and its certificate, which this section records.

## 5) Preregistered observables

1. **Graceful-vs-cliff degradation** (`hyp-padic-truncation-graceful`):
   quality-vs-k digit-truncation curves for hard-digit ultrametric
   checkpoints degrade more gracefully than float-quantization
   quality-vs-bits curves for the standard baseline (area-under-quality-curve
   as the scalar). Blocked on: the float-quantization baseline harness and
   hard-digit trained checkpoints.
2. **Hensel curriculum parity with monotone refinement**
   (`hyp-hensel-curriculum-parity`): digit-wise curricula match end-to-end
   final quality at equal budget, with the residue invariant exact at every
   stage (the demo asserts it; the prediction adds the parity claim).
3. **Depth-independence of truncation error**
   (`hyp-padic-truncation-depth-independent`): the flat-error signature
   visible end-to-end — quality at digit-k truncation independent of model
   DEPTH for the p-adic-native path, because per-layer truncation errors do
   not accumulate through depth (each layer's digit machinery re-quantizes;
   the lemma kills the compounding term). The experimentally striking
   signature, and the one float quantization cannot reproduce.

The perplexity-vs-k sweep harness runs against existing checkpoints today
(soft-digit e1 checkpoints: exploratory, since soft digits make truncation a
channel-drop rather than a valuation truncation); the preregistered claims
bind to hard-digit arms, where the dictionary's exactness contract holds.
