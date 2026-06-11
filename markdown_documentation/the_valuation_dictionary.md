# The Valuation Dictionary: p-adic = Tropical = Dominance

*Bead: model_guided_research-8gk.2 (EPIC THEORY-I, 8gk). Reference mechanism +
ball-tree prototype: `ultrametric_worlds_and_p_adic_computation.py`
(valued-attention section); exact integer property tests:
`tests/test_algebraic_properties.py` (valuation-dictionary section);
registry: `hypotheses/registry.yaml` (`hyp-balltree-*`, `hyp-tie-locus-*`);
theorems: `thm-valuation-arithmetic`, `thm-lcp-is-padic-valuation`,
`thm-tropicalization-of-attention`, `thm-balltree-exact-attention`.*

Three of this project's frameworks — ultrametric/p-adic routing, tropical
(min,+)/(max,+) algebra, and surreal/transseries dominance — are not parallel
experiments. They are three shadows of one backbone: **geometry over a valued
field**. This note defines the common formal object first, then proves each
instance, then derives the two payoffs: the tropicalization theorem for
attention (with the exact measure of its failure locus) and an exact
sub-quadratic attention algorithm whose correctness is a theorem rather than
an approximation bound.

## 0) The formal object: valued attention structures

**Definition (valued attention structure).** A tuple (V, Γ, v, sim) where

- **V** is the representation space (whatever objects queries/keys live in);
- **Γ** is a totally ordered abelian group (the *value group*), extended by a
  top element ∞;
- **v : V × V → Γ ∪ {∞}** is a *dominance valuation*: writing
  d(x, y) = exp(−v(x, y)) formally, v must satisfy the **strong triangle
  axiom** v(x, z) ≥ min(v(x, y), v(y, z)) and v(x, x) = ∞ — equivalently, d
  is an ultrametric (every "ball" {y : v(x, y) ≥ γ} is either nested in or
  disjoint from every other);
- **sim : Γ ∪ {∞} → ℝ** is a monotone score transform (deeper agreement ⇒
  larger score);
- **compatibility (the almost-homomorphism axiom)**: for whatever algebraic
  operations ★ the mechanism applies to representations, v interacts with ★
  by exact laws on a generic set — multiplicativity v(x★y) = v(x) + v(y) for
  products, and super-additivity with generic equality
  v(x + y) ≥ min(v(x), v(y)) for sums.

Attention in such a structure is: score(q, k) = sim(v-data of (q, k)); routing
moves between valuation cells; hierarchy is the ultrametric ball tree that the
strong triangle axiom forces into existence. The instances below differ only
in (V, Γ): the mechanism-level consequences (§3–§4) are theorems of the
axioms, which is what makes the unification a definition with content rather
than rhetoric.

## 1) The instances (each proof short; the assembly is the theorem)

**(a) Q_p digits — Γ = ℤ.** V = ℤ_p as K-digit base-p expansions (LSB first),
v_p(n) = the exponent of the largest power of p dividing n, v(x, y) :=
v_p(x − y).

**Theorem (LCP IS VALUATION).** For x, y ∈ ℤ_p, LCP(x, y) = v_p(x − y),
exactly — and at finite precision, with both sides computed mod p^K and
capped at K.

*Proof.* x − y ≡ 0 (mod p^d) iff the first d base-p digits of x and y agree
(induction on d: digit i of x − y mod p^{i+1} is determined by digits ≤ i of
x and y with the standard borrow, and vanishes iff digit i agrees given all
earlier ones do). So the largest d with p^d | (x − y) is the longest common
prefix. ∎

The strong triangle axiom holds because if x, y agree to depth d and y, z
agree to depth d′, then x, z agree to depth ≥ min(d, d′). The ultrametric
mechanism's trie is exactly the ball tree of this valuation — a finite
subtree of the Bruhat–Tits tree for PGL₂(ℚ_p) (whose vertices are balls and
whose adjacency is one-step refinement). Tests:
`test_lcp_equals_padic_valuation_*` (exact integers, p ∈ {2, 3, 5}).

**(b) Tropical — Γ = ℤ (or ℝ), the valuation shadow.** v_p is a homomorphism
(ℚ_p^×, ×) → (ℤ, +): v(xy) = v(x) + v(y) **always** (count powers of p in a
product). For sums, v(x + y) ≥ min(v(x), v(y)) with **equality whenever
v(x) ≠ v(y)**: the lower-valuation leading digit cannot be cancelled by a
term divisible by a higher power of p. So v maps (×, +) to (+, min) exactly
off the cancellation locus: **tropical arithmetic is the image of valued-field
arithmetic under v**. Tests: `test_valuation_homomorphism_*`,
`test_valuation_min_rule_*` (including adversarially constructed
cancellations where the inequality must be strict).

**(c) Hahn/surreal — Γ = the exponent group, v = dominance.** A Hahn series
x = Σ_γ c_γ t^γ (well-ordered support) has v(x) = its leading exponent;
Conway normal form ω^{γ₁} r₁ + ω^{γ₂} r₂ + … **is** a Hahn series
representation of the surreal numbers, so the dominance order probed by the
surreal demo (which axis's leading term wins) is valuation by leading term.
Multiplicativity: leading terms multiply; super-additivity: the smaller
leading exponent survives a sum unless coefficients cancel — the same two
laws as (b) with Γ enlarged. Dominant-balance analysis *is* tropicalization
over a series field.

**(d) IFS cylinders — Γ = ℝ via radius products.** For an IFS with
contraction ratios r_a, the address space carries
d(σ, τ) = Π_{i<LCP(σ,τ)} r_{σ_i} — i.e. v(σ, τ) = Σ_{i<LCP} (−log r_{σ_i}),
a sum of per-level weights along the common prefix. Strong triangle: common
prefixes nest. The uniform-ratio case v = LCP·(−log r) recovers (a) with a
rescaled Γ: FractalKV and the ultrametric trie are the same structure with
different radius conventions (the 8gk.5 unification).

**(e) Tree boundary / Gromov product — the geometric instance.** For points
ξ, η on the boundary of a tree, the Gromov product (ξ|η) based at the root is
the depth of the branch point = LCP depth, and d(ξ, η) = exp(−(ξ|η)) is the
standard boundary ultrametric. Hyperbolic attention's similarity degenerates
to this as curvature → −∞ (the 8gk.6 correspondence); v = the Gromov product.

One backbone — digits (a), piecewise-linear geometry (b), asymptotics (c),
self-similar memory (d), negative curvature (e) — five shadows.

## 2) The reference mechanism: valued attention

Keys and queries are d-vectors of finite-precision ℤ_p elements (length-K
digit vectors with genuine valuation arithmetic). Scores are monotone
transforms of valuations, in two flavors:

- **difference form** (the ultrametric projection): sim = α^{v_p(q − k)} —
  digit similarity, LCP routing, the existing ultrametric mechanism. Uses
  only axiom (strong triangle); never touches products.
- **bilinear form** (the tropical projection): sim = α^{−v_p(⟨q, k⟩ − c)} —
  valuation arithmetic of products and sums. Uses only the compatibility
  axiom; never touches the trie.

The existing ultrametric mechanism (digit similarity only) and tropical
mechanism (valuation arithmetic only) are **the two projections of the one
structure** — which is the precise sense in which "tropical structure appears
wherever hierarchy does": they are the same valuation seen through two
different score maps. The demo's three-shadow table renders one bilinear
attention computation simultaneously as digits, as (min,+) arithmetic, and as
leading terms (`run_valued_attention_section`).

## 3) The tropicalization theorem for attention

**Theorem.** Let q, k ∈ (ℤ_p)^d with entries q_j, k_j. Then

    v_p(⟨q, k⟩) ≥ min_j (v_p(q_j) + v_p(k_j)),

with equality — i.e. **tropical attention computes exactly the valuation of
p-adic bilinear attention** — whenever the inputs are *valuation-generic*:
the leading terms of the minimal-valuation products do not cancel mod p,

    Σ_{j ∈ argmin} lead(q_j)·lead(k_j) ≢ 0 (mod p),

where lead(x) = (x/p^{v(x)}) mod p. In particular equality holds whenever the
minimum is attained by a single j.

*Proof.* Each product has v(q_j k_j) = v(q_j) + v(k_j) (homomorphism). Write
m for the min over j. Every term is divisible by p^m, so the sum is —
giving ≥. The coefficient of p^m in the sum is Σ_{argmin} lead(q_j) lead(k_j)
mod p; the valuation exceeds m iff that coefficient vanishes. ∎

**The cancellation locus is the tropical variety.** The set where the
inequality is strict is exactly where the tropical polynomial
min_j (v(q_j) + v(k_j)) achieves its min twice *with cancelling leading
digits* — the corner locus where routes switch. This is not a defect of the
correspondence; it is the geometry of the decision boundaries.

**Theorem (measure of the cancellation locus).** For x, y independent
Haar-uniform on ℤ_p:

    P[v(x + y) > min(v(x), v(y))] = 1/(p + 1).

*Proof.* Digits are i.i.d. uniform. Condition on v(x) = v(y) = m (probability
((1−1/p) p^{−m})² for each m): cancellation requires the leading digits
(uniform on the p−1 nonzero values, independent) to sum to 0 mod p, which
happens with probability (p−1)/(p−1)² = 1/(p−1). Summing the geometric
series: Σ_m (1−1/p)² p^{−2m} / (p−1) = (p−1)/p² · p²/(p²−1) = 1/(p+1). ∎

Verified empirically in the demo (p = 2, 3, 5: 0.332/0.250/0.166 vs
1/3, 1/4, 1/6) and exactly in
`test_cancellation_locus_measure_binary_sums`. For d-term inner products the
genericity event is the non-vanishing of one mod-p linear form in the leading
digits of the argmin set — measure ≥ 1 − 1/(p−1)-ish per tie configuration,
with the empirical rates reported by the demo; the binary case above is the
sharp closed form the registry's tie-locus observable normalizes against.

## 4) The algorithmic payoff: exact ball-tree attention

For the difference kernel, the strong triangle axiom makes balls trie nodes:
the set {k : v_p(q − k) ≥ d} is exactly the keys sharing q's depth-d residue
mod p^d, and these sets are nested in d. Hence the **shell decomposition**

    Σ_j α^{lcp(q, k_j)} v_j = Σ_{d=0}^{K} α^d ( S_{≥d}(q) − S_{≥d+1}(q) ),

where S_{≥d}(q) is the value-sum stored at q's depth-d trie node.

**Theorem (exactness and complexity).** With per-node sums and counts
maintained at insertion (O(K) per key), the right-hand side — numerator and
normalizer alike — is computed in **O(K) per query** and equals the
brute-force attention **exactly**: this is a partition of the key set, not an
approximation. Streaming insertion (query token i before inserting it) gives
an exactly-causal attention pass in O(NK) total, versus O(N²K) brute force.

*Proof.* The shells {lcp = d} = {lcp ≥ d} \ {lcp ≥ d+1} partition the keys by
nesting; each shell's weight is the constant α^d. Sums of a partition are
exact. Early exit is sound: if q's depth-d node is empty, all deeper balls
are empty (nesting). ∎

`BallTreeValuedAttention` implements this with α = 2 and integer values so
every arithmetic step is exact dyadic: the demo and
`test_balltree_attention_equals_bruteforce_exact` assert equality with `==`,
no tolerance. The demo's timing table (N up to 4096) is the measured E3 hook:
the torch ultrametric kernel mode is quadratic-with-better-constants, and
this is the theorem-backed sub-quadratic path it should converge to
(per-query cost flat in N for the tree vs linear in N for brute force). The
preregistered claims (`hyp-balltree-valued-attention-speedup`,
`hyp-tie-locus-density-decreases`) name the exact metrics; the second one —
the tie-locus density is small and DECREASES under training, i.e. the model
learns to avoid its own decision boundaries — is the novel observable this
dictionary makes measurable at all.

## 5) Relation to the rest of the program

- **8gk.1 (Maslov annealing)** is the β-smoothing of shadow (b): softmax is
  Maslov quantization of the tropical endpoint; this note supplies the
  underlying field whose valuation the tropical endpoint reads off.
- **T1.4 / 8gk.4 (p-adic precision)** depends on (a): digit truncation is
  ball-rounding, and the strong triangle inequality is why truncation error
  does not accumulate.
- **8gk.5 / 8gk.6** are instances (d) and (e) of §1, registered as their own
  beads; the definition in §0 is the umbrella their proofs land under.
- **E3 (sub-quadratic ultrametric training path)**: §4 is the concrete
  content of its anticipated discovered-from bead — the exact algorithm,
  complexity analysis, and benchmark hook now exist at demo level; the torch
  port is the follow-on engineering item.
- **The capstone (cbm)**: Pillar I's section opens with §0's definition, per
  the referee-round-3 contract.
