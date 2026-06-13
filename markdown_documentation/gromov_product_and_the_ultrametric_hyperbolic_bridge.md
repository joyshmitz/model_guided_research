# The Gromov Product and the Ultrametric–Hyperbolic Bridge

**Bead:** 8gk.6 (EPIC THEORY-I, 8gk — the non-archimedean unification) ·
**Companion design docs:** mnn.4 (hyperbolic mechanism), the ultrametric note
and `ultrametric_attention_torch.py` (the discrete endpoint) · **Status of
claims:** every proposition is verified computationally before it is stated
(receipts in Appendix A; scratch `/data/tmp/gromov_lcp_scratch.py` and
`/data/tmp/gromov_energy_scratch.py`, pure math, no repo dependencies).

This note closes the geometric side of the valuation dictionary: it proves
that the production **ultrametric** mechanism is *boundary-of-a-tree*
attention exactly, that **hyperbolic** attention is its continuous
completion, and that one mechanism with a learnable per-head curvature
traverses Euclidean ↔ hyperbolic ↔ tree as a **homotopy family**. The
load-bearing subtlety — which the design docs must get right or their c → 0
acceptance test tests the wrong thing — is that the *distance* Gromov product
and the *energy* (squared-distance) Gromov product have **different**
Euclidean limits. That distinction is §3 and it is the heart of the note.

Curvature convention: the space has curvature −c with **c > 0 a magnitude**;
"c → ∞" is the tree limit (the metric-geometry literature writes it as
curvature → −∞).

## 1. The Gromov product

For points x, y and a basepoint o in any metric space,

    (x | y)_o  :=  ½ · ( d(o,x) + d(o,y) − d(x,y) ).

Geometrically it is how long geodesics o→x and o→y travel together before
diverging. It is non-negative in trees and in δ-hyperbolic spaces (up to δ),
and it is the single object that unifies the two hierarchy mechanisms.

## 2. Proposition 1 (exact): the Gromov product at the root IS the LCP depth

**Statement.** In a rooted metric tree with unit edge lengths, for leaves x, y
at depth K with address strings (the digit sequences labelling the path from
the root),

    (x | y)_root  =  depth of the deepest common ancestor of x and y
                  =  LCP( address(x), address(y) ).

**Proof (elementary, exact).** Let ℓ = LCP(x, y). Both leaves sit at depth K,
so d(root, x) = d(root, y) = K. The unique path x → y climbs from x to the
deepest common ancestor (K − ℓ edges) and descends to y (another K − ℓ
edges), so d(x, y) = 2(K − ℓ). Therefore

    (x|y)_root = ½(K + K − 2(K − ℓ)) = ℓ.  ∎

This is an *integer identity with no tolerance*: any deviation in code is a
bug. Verified exhaustively over every leaf pair of depth-4 binary, depth-5
ternary, and depth-3 quaternary trees — 31 539 pairs, zero mismatches
(Appendix A.1).

**Corollary (the ultrametric mechanism is a Gromov-product kernel).** The
boundary of a rooted tree carries the ultrametric d(ξ, η) = exp(−(ξ|η)).
The production kernel (`ultrametric_attention_torch.py`) forms weights
w(q, k) ∝ α^{LCP(q,k)} = exp( (log α) · (q|k) ): an exponential of the Gromov
product. **Ultrametric attention is attention on the boundary of a tree,
exactly** — the discrete endpoint of the homotopy (A.2). The implemented
`lcp` tensor (expected longest-common-prefix depth from prefix-match
products) is the continuous relaxation of (q|k); α^{lcp} is the relaxed
Gromov kernel.

## 3. Proposition 2 (the correction terms ARE semantics) — and the distance/energy fork

**The identity.** By the definition of the Gromov product, for any q, k, o,

    − d(q, k)  =  2 (q|k)_o  −  d(q, o)  −  d(k, o).        (verified, A.3)

So a plain distance score −d(q,k) silently sums a **hierarchy-overlap** term
2(q|k)_o and two **depth-in-hierarchy** (radial) terms d(·, o). The current
hyperbolic-attention literature scores by −d and thereby conflates "how much
do our hierarchies overlap" with "how deep/specific are we." Separating them
with **learnable radial gates** is the design contribution:

    s(q, k)  =  [ 2 (q|k)_o  −  λ_q · r(q)  −  λ_k · r(k) ] / τ,

where r(·) is the radial term (distance or energy — see below), λ = 1 is the
plain distance/energy score, λ = 0 is pure overlap, and the learned λ per
head is an observable (generality-vs-specificity routing).

**The fork that the design docs must get right.** There are two Gromov
products, and they do **not** have the same Euclidean limit:

| object | definition | c → 0 limit | c → ∞ (tree) limit |
|---|---|---|---|
| **distance** 2(q\|k)_o | d(o,x)+d(o,y)−d(x,y) | **\|u\|+\|w\|−\|u−w\|** (Laplacian/distance-kernel attention) | **integer LCP** (Prop 1, exact) |
| **energy** 2·G_c | R_c(x)+R_c(y)−D_c(x,y) | **u·w** (dot-product / STANDARD attention) | LCP via the arsinh readout |

with x = exp_o(u), y = exp_o(w), the **chordal/radial energies**

    D_c(x,y) := −(1/c)(1 + c⟨x,y⟩_L) = (2/c)·sinh²(√c·d(x,y)/2),   R_c(x) := D_c(x,o),

(pure bilinear forms — no `arcosh` in the score path), and G_c :=
½(R_c(q)+R_c(k)−D_c(q,k)). The two columns are the proof obligations of the
two degeneration anchors and they pull in opposite directions:

- The **distance** Gromov product is the object for which Proposition 1 is
  *exact and elementary* — the tree/ultrametric anchor wants distances.
- The **energy** Gromov product is the object whose c → 0 limit is *exactly
  the dot product* u·w — the standard-attention anchor wants energies.

**This is a correction to the naive reduction claim.** It is *not* true that
the raw-distance score −d_H/τ converges to dot-product attention as c → 0 at
fixed temperature: it converges to **negative-Euclidean-distance** attention,
a Laplacian-kernel softmax. Numerically (A.4), as c sweeps 1 → 10⁻⁵ the
distance Gromov product locks onto |u|+|w|−|u−w| (error 3.3e−7) while the
energy Gromov product locks onto u·w (error 1.4e−6) — different limits, each
matched to the predicted target, neither matching the other's target. A
mechanism that wants the clean "c → 0 ⇒ standard attention" anchor (so the
learned-curvature readout is interpretable against a familiar baseline)
**must score with energies**, not raw distances; an implementation that
scores with raw distances and then tests "weights match dot-product attention
at c = 1e−3" will fail that test by ~0.5 in the logits.

**The bridge that reconciles them.** The two are exactly connected at every c
by the stable inverse (no `1+ε` cancellation, finite gradient at coincident
points):

    d(x, y)  =  (2/√c) · arsinh( √( c · D_c(x,y) / 2 ) ).        (A.5)

So the recommended computational path is: **score with energies** (cheap,
arcosh-free, correct standard-attention anchor) and **recover the distance
Gromov product via arsinh** for the tree-limit anchor and for telemetry. One
mechanism, both anchors, no transcendental in the training-gradient path.

## 4. Proposition 3 (the homotopy): one curvature dial, three geometries

**Statement.** A single attention mechanism with a learnable per-head
curvature magnitude c traverses, as c moves over (0, ∞):

    c → 0           :  Euclidean attention  (dot product, via the energy form §3)
    finite c        :  hyperbolic attention (continuous hierarchy bias)
    rescaled c → ∞  :  tree / ultrametric LCP attention  (Prop 1 boundary kernel)

**c → 0 (Euclidean).** Established in §3 and A.4: the energy Gromov product
→ u·w; the value aggregation (tangent-space / Lorentzian-centroid mean) → the
ordinary weighted mean. The mechanism *starts* near this anchor under the §
"z ≈ 1 at init" scale policy and earns curvature.

**Rescaled c → ∞ (tree).** Rescale the metric by √c. The rescaled space is
δ-hyperbolic with δ = δ₀/√c → 0 (δ₀ = ln(1+√2) for curvature −1; CAT(−c)
asymptotics), and rescaled pointed δ-hyperbolic spaces converge in the
Gromov–Hausdorff sense to **ℝ-trees** (Gromov; Bridson–Haefliger III.H).
On the limit tree the Gromov product at the root is the integer
deepest-common-ancestor depth = LCP (Prop 1). Numerically (A.6): embed a
depth-4 binary tree in the Poincaré disk with hyperbolic edge length s; the
rescaled Gromov product (x|y)_o / s converges to the integer LCP with max
error halving on every doubling of s (0.877 → 0.666 → 0.422 → 0.212 at
s = 1, 2, 4, 8) — the O(δ/s) = O(1/s) rate the theory predicts. (The s = 16
rung saturates the Poincaré model in fp64 — a live illustration of why the
*implementation* uses the Lorentz model; mnn.4 §1.)

**Consequence — the three-way comparison collapses into a readout.** The
planned hyperbolic-vs-ultrametric-vs-fractal bake-off becomes a single model
trained with learnable per-head c whose *learned curvature is the answer*:
which heads go hierarchical, how deep, on which data. c collapsing toward 0
is the honest "not hierarchical here" verdict and is explicitly welcomed — it
would refute the hierarchy hypothesis from inside the mechanism.

## 5. Requirements emitted into the implementation (F4/F5/F6 = mnn.4/5/6)

These are the binding consequences of §§2–4; mnn.4 (the committed design doc)
already adopts the Gromov-form scores, radial gates, learnable curvature, and
both degeneration anchors. This note tightens three points:

1. **Score with energies, not raw distances** (§3). The dot-product c → 0
   anchor is *only* exact for the energy Gromov product; the raw-distance
   form's c → 0 limit is distance-kernel attention. The c → 0 acceptance
   test must compare against the matching reference: energy-form scores vs
   standard dot-product attention (the clean, interpretable choice), or
   raw-distance scores vs a Laplacian/distance kernel (and then "Euclidean
   attention" must be labelled as the distance kernel, not dot product).
   **Filed to mnn.4/mnn.6 as a correction.**
2. **arcosh-free forward, arsinh readout** (§3, A.5). Energies are bilinear
   (no transcendental in the gradient path); exact distances for the tree
   anchor and telemetry use the stable arsinh identity. This is also the
   already-filed mnn.6 numerics suggestion — the same change serves both the
   correctness anchor and the numerical-stability goal.
3. **The tree anchor is an acceptance test, not narrative** (§4, A.6).
   F5/F6 certify: plant an exact tree embedding, assert the rescaled distance
   Gromov product equals integer LCP within tolerance at √c·edge = 8, and the
   weights match the α^{LCP} ultrametric kernel row-for-row.

## 6. Registry predictions (registry-ready)

Floor/rung discipline per the braid–dyck arc (E1 supported at a floored
baseline, E2 inverted at the off-floor rung): every EM claim names a floor
and a probe-selected rung in `scale_caveats`.

1. `hyp-hyperbolic-heads-go-hierarchical` — *trained on hierarchical
   retrieval, ≥ 25% of heads learn curvature above the hierarchy threshold,
   while a structure-free placebo control collapses to c ≈ 0.* Single-arm on
   the trained-model telemetry: `train:results.hyperbolic_frac_heads_hier
   >= 0.25` on hier vs `<= 0.05` on placebo at equal FLOPs (the paired claim
   is the mechanism-internal hierarchy detector; both can fail honestly, and
   that failure kills the homotopy-readout thesis — its falsifiable content).
2. `hyp-radial-gate-beats-pure-distance` — *radial-gated scores beat
   pure-distance (λ ≡ 1) scores on depth-sensitive hierarchical queries by
   ≥ +0.03 held-out EM at equal FLOPs*, baseline = the same mechanism with
   gates frozen at 1, floor = the hier answer prior.
3. `hyp-curvature-trained-matches-ultrametric` — *a hyperbolic model trained
   into the rescaled-large-c regime matches the dedicated ultrametric
   mechanism on hierarchical retrieval within ±0.02 EM* (the homotopy
   endpoint reproduces the specialist — the cleanest possible confirmation
   that the two mechanisms are one family). Registered `prediction: null`
   with an operationalization_note until F6 and an off-floor sizing probe
   land — visible debt, never silent omission.

## 7. Relation to the rest of the program

This closes the geometric half of the valuation dictionary: the *p-adic /
valuation* picture (the ultrametric note, `the_valuation_dictionary.md`) and
the *metric-tree-boundary* picture are the same object seen two ways, with the
Gromov product the translation. T1.2 (boundary ultrametric = the valuation
picture) is its algebraic mirror. The tropical/Maslov homotopy
(`maslov_dequantization_annealing.md`) is the analogous β-dial for the
max-plus semiring; together they say the project's three "hierarchy/idempotent"
mechanisms — ultrametric, hyperbolic, tropical — are each one endpoint of a
continuous family with an interpretable interior, not isolated tricks.

## Appendix A: verification receipts (8gk.6 scratch, 2026-06-13)

Pure-python, first-principles; reproduced in the F5/F6 degeneration suite.

1. **Prop 1 (Gromov = LCP), exact integer**: depth-4 binary (120 pairs),
   depth-5 ternary (29 403), depth-3 quaternary (2 016) — 0 mismatches, no
   tolerance. PASS.
2. **Ultrametric kernel identity**: α^{LCP} = exp((q|k)·log α) and boundary
   d = exp(−(q|k)) to 1e−12. PASS.
3. **Prop 2 identity**: −d(q,k) = 2(q|k)_o − d(q,o) − d(k,o) on 500 random
   Lorentz-model pairs, max err 4.4e−16. PASS.
4. **The distance/energy fork (§3)**: as c: 1 → 10⁻⁵, the *distance* Gromov
   product → |u|+|w|−|u−w| (err 3.3e−7) and the *energy* Gromov product →
   u·w (err 1.4e−6) — different limits, each matched to its predicted target;
   the raw-distance form does NOT converge to the dot product. PASS (the
   correction of §3 / requirement 5.1).
5. **arsinh bridge**: d = (2/√c)·arsinh(√(c·D_c/2)) reproduces the Lorentz
   distance exactly (same scratch, used to compute distances from energies).
6. **Prop 3b (rescaled tree limit)**: depth-4 binary tree in the Poincaré
   disk, (x|y)_o/s → integer LCP with max error 0.877 → 0.666 → 0.422 →
   0.212 at s = 1, 2, 4, 8 (halving per doubling = O(1/s) = O(δ/s)). The
   s = 16 rung saturates the Poincaré model (fp64) — the Lorentz-motivation
   illustration. PASS (monotone convergence).
