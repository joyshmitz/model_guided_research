# Maslov Dequantization Annealing: Softmax-to-Tropical as an Exact Semiring Homotopy

*Bead: model_guided_research-8gk.1 (EPIC THEORY-I, 8gk). Implementation:
`nanochat/tropical_attention_torch.py` (`--semiring-beta`); tests:
`tests/test_algebraic_properties.py` (Maslov section); formalization target:
vnl.2 / thm-route-stability.*

## 1) The semiring family

For β > 0 define on ℝ ∪ {−∞}:

    x ⊕_β y = (1/β) · log(e^{βx} + e^{βy}),        x ⊗ y = x + y.

**Fact (Maslov quantization).** For every finite β, (ℝ ∪ {−∞}, ⊕_β, ⊗) is a
commutative semiring: ⊕_β is associative and commutative with identity −∞,
and ⊗ distributes over it — `c + (a ⊕_β b) = (c+a) ⊕_β (c+b)` *exactly* (the
log-domain shift identity). The map x ↦ e^{βx} is a semiring isomorphism onto
(ℝ₊, +, ×): the family is ordinary arithmetic seen through a logarithmic
microscope of magnification β.

**Fact (dequantization, the LSE–max sandwich).** For any x₁..xₙ:

    max_i x_i  ≤  ⊕_β,i x_i  ≤  max_i x_i + log(n)/β.

Both inequalities are exact (any violation in code is an implementation bug —
asserted in `test_maslov_lse_max_sandwich_exact_inequality`). As β → ∞ the
family converges *uniformly* to the tropical semiring (max, +), with explicit
rate log(n)/β. The inflation is **one-sided**: ⊕_β never underestimates max.

## 2) Log-domain attention and its two endpoints

Define attention entirely inside the semiring (scores s_k for one query, value
coordinates v_{kd}; semiring division = subtraction):

    y_d = [⊕_β,k (s_k + v_{kd})] − [⊕_β,k s_k].

Expanding: y_d = (1/β)·log Σ_k softmax_β(s)_k · e^{β v_{kd}} — the log-domain
(tilted/geometric) softmax average of the values.

**Endpoint β → ∞.** Both LSEs collapse to maxes; the normalizer becomes
max_k s_k, and y_d → max_k (s_k + v_{kd}) − max_k s_k: this is *exactly* the
codebase's tropical attention **with score centering** —
`tropical_score_center` is the β = ∞ shadow of softmax normalization. The
code contained both endpoints of the family before knowing they were
connected.

**Relation to softmax temperature.** β = 1/τ, but the standard transformer
aggregates *linearly*: y_d = Σ_k softmax(s/τ)_k · v_{kd}. The log-domain form
differs by Jensen's inequality (log of an average vs average of logs) — they
share routing weights but not aggregation. The log-domain form is the
**certificate-carrying** one: it is a semiring polynomial at every β, so the
sandwich and the route-stability lemma below apply verbatim, and the β = ∞
limit is *exactly* the 1-Lipschitz tropical layer rather than a limit taken
through a different algebra.

**1-Lipschitz at every β.** LSE and max are both nonexpansive in sup-norm,
and the normalized difference of two nonexpansive aggregations of inputs
shifted by the same scores stays nonexpansive in v: the tropical robustness
certificate extends to every finite β, not just the endpoint.

## 3) The route-stability lemma (two-family form)

The precision-note form (this is what vnl.2 formalizes; the arity is the
**inner aggregation arity**, never the number of routes):

**Lemma.** Let x ∈ ℝ^m be tropical scores and y ∈ ℝ^m smoothed scores with
x_i ≤ y_i ≤ x_i + log(m)/β (the one-sided sandwich). If the runner-up margin
γ(x) = x_(1) − x_(2) satisfies **γ(x) > log(m)/β**, then argmax y = argmax x.

*Proof.* One-sided inflation can close a gap by at most its width: for any
challenger j, y_j ≤ x_j + log(m)/β ≤ x_(2) + log(m)/β < x_(1) ≤ y_(1). ∎

(`test_route_stability_lemma_two_family` checks this adversarially: every
non-winner inflated by the full budget.)

**Composition for the attention layer.** Two smoothings act in sequence:
the score stage (⊕_β over the head dim, inner arity D = head_dim) inflates
each score by at most log(D)/β, and the value aggregation (inner arity m =
keys visible to the query) carries the log(m)/β sandwich. The conservative
compositional certificate used by the coverage telemetry
(`route_stability_threshold`) is therefore

    γ > (log D + log m)/β   ⇒   the tropical route survives smoothing,

with γ computed from the *tropical* scores of the same q/k/v (the margins the
codebase already records via `tropical_record_margins`). **Certificate
coverage** = the fraction of (token, head) routes clearing this threshold —
logged per step as `route_coverage` alongside `semiring_beta`.

## 4) Dequantization annealing

Train with a schedule β: β₀ → β₁ (`--semiring-beta linear:B0:B1 | exp:B0:B1`),
i.e. a continuation method *with semiring semantics*: early training gets
smooth gradients everywhere; the late-time limit is certified piecewise-linear
routing. Unlike Gumbel-softmax, straight-through estimators, or REINFORCE —
which approximate gradients *of* a discrete object — here the discrete object
is the exact algebraic limit of the smooth family, with a convergence rate
(log n)/β and a per-token certificate (γ vs the threshold) observable DURING
training. The soft-attention-vs-hard-routing dichotomy dissolves into one
scalar with algebraic meaning.

Preregistered predictions (G1; experiment bead spun off from 8gk.1): annealed
models retain ≥ 95% of their β₀ quality at matched budget; certificate
coverage of annealed models far exceeds post-hoc snap (train soft, discretize
at the end — the baseline everyone implicitly uses).

## 5) Positioning

**(a) Modern Hopfield networks.** The Ramsauer et al. update rule *is*
softmax attention, and its β → ∞ limit performs exact nearest-pattern
retrieval: modern Hopfield theory already lives on the two endpoints of the
⊕_β homotopy. Dequantization annealing is zero-temperature annealing of an
attention–Hopfield energy, and the route-stability lemma bounds when
retrieval basins freeze: a stored pattern's basin is frozen at level β once
its margin exceeds log(m)/β — the lemma is a quantitative freezing criterion
for associative memory.

**(b) Linear attention as a third semiring point.** Kernelized linear
attention computes in (ℝ₊, +, ×) directly — no log, no exp. The semiring
viewpoint places softmax (Maslov image at finite β), linear attention
(untransformed semiring), and tropical attention (β = ∞) as three choices of
semiring on ONE design axis. "Which attention?" becomes "which semiring, at
which magnification?".

**(c) Entropic optimal transport.** Sinkhorn/entropic-OT attention lives
inside the family too: Sinkhorn iteration is alternating row/column
⊕_β-normalization, the entropic regularizer ε is 1/β, and the β → ∞ limit of
doubly-normalized attention is exact *assignment* (a permutation, by
Birkhoff–von Neumann). The tropical limit of Sinkhorn attention is learned
matching — connecting this pillar's semiring homotopy to the
braid/permutation machinery of Pillar II (u55.3). No implementation
obligation here; a doubly-normalized variant gets its own bead if ever built.

## 6) Schedules and their clients

Implemented now: constant, `linear`, `exp` (β interpolated over the run;
per-step hook `set_semiring_beta`, telemetry `semiring_beta` +
`route_coverage` in metrics.jsonl). Designed, spun off as follow-up beads:

- **Ordinal-orchestrated** (T3.3 client): the transfinite scheduler drives β
  transitions at its limit ordinals.
- **Closed-loop coverage feedback**: raise β only while measured coverage
  stays above a target floor, back off otherwise — certificate-driven
  control, consuming the 5ki.5 gamma telemetry; preregistration: reaches a
  target (β, coverage) operating point in fewer steps than the best open-loop
  schedule from the same grid, never violating the floor (checkable from
  logs alone).

## 7) Test inventory (the math is the spec)

| law | test |
|---|---|
| ⊕_β associative/commutative, + distributes | `test_maslov_oplus_associative_commutative`, `test_maslov_plus_distributes_over_oplus` |
| LSE–max sandwich (exact inequality) | `test_maslov_lse_max_sandwich_exact_inequality` |
| route-stability lemma (adversarial one-sided inflation) | `test_route_stability_lemma_two_family` |
| attention converges to the tropical endpoint at rate (log D + log m)/β, monotonically | `test_maslov_attention_converges_to_tropical_endpoint` |
| β = None is the untouched tropical path (bit-identical) | `test_maslov_beta_none_is_the_untouched_tropical_path` + the attention goldens (trajectories unchanged at recapture) |
| schedule hook + coverage telemetry lifecycle | `test_set_semiring_beta_and_coverage_telemetry` |
