# Hyperbolic Geometry and Negative-Curvature Attention (Framework #13)

**Bead:** mnn.4 (EPIC mnn, frameworks #12–13) · **Downstream:** 8gk.6 (Gromov = LCP), mnn.5 (JAX demo), mnn.6 (nanochat torch implementation) · **Status of claims:** every numerical claim in this document was validated before it was written (validation receipts inline; script pattern: tiny fp32/fp64 torch, no repo dependencies).

---

## 0. Why this framework, in one paragraph

Hyperbolic space is the **continuous** hierarchy bias: trees embed in H^n with
arbitrarily low distortion (impossible in any fixed Euclidean dimension), so
negative curvature is to continuous hierarchy what the p-adic ultrametric is
to discrete hierarchy and fractal self-similarity is to scale hierarchy.
Adding it completes the set and enables the three-way "which hierarchy bias
wins" study on hierarchical retrieval — and, by the Gromov-product
reformulation (§3), it is not merely a 13th sibling: it is the **continuous
completion of the ultrametric mechanism**, with exact degeneration limits to
both Euclidean attention (c → 0) and ultrametric LCP attention (rescaled
c → ∞). Both limits are acceptance tests, not metaphors.

## 1. Model choice: Lorentz, not Poincaré

H^n has several isometric coordinate models; the implementation-relevant
candidates are the Poincaré ball (points in the open unit ball, conformal
metric blowing up at the boundary) and the **Lorentz / hyperboloid model**
(points on the upper sheet of ⟨x,x⟩_L = −1/c inside Minkowski space R^{1,n},
signature (−,+,⋯,+)).

**RECOMMENDATION: Lorentz.** The reason is not taste but fp32 arithmetic
(Nickel & Kiela 2018, *Learning Continuous Hierarchies in the Lorentz Model
of Hyperbolic Geometry*, reach the same conclusion for embeddings):

| hyperbolic distance from origin | Poincaré radius (fp32) | Lorentz x₀ (fp32) | distance recoverable? |
|---|---|---|---|
| 5  | 0.986614287 | 7.42e+01 | both |
| 10 | 0.999909103 | 1.10e+04 | both (Poincaré marginal) |
| 20 | **0.999999940 = saturated** | 2.43e+08 | **Lorentz only** (recovers 20.0000) |
| 40 | **1.000000000 = boundary** | 1.18e+17 | **Lorentz only** (recovers 40.0000) |

(Validated in fp32; "saturated" = within one ulp of the boundary, where the
conformal factor and therefore every gradient is numerically destroyed.) The
Poincaré ball is unusable past d ≈ 16–20 in fp32. The Lorentz model pushes
the usable radius to d ≈ 80 (cosh(80) ≈ 5.5e34 < fp32 max 3.4e38), a 4×
deeper hierarchy budget, and its distance

```
d_H(x, y) = (1/√c) · acosh(−c · ⟨x, y⟩_L)
```

is computed from a Minkowski inner product — one fused multiply-add chain,
no division by a vanishing conformal factor anywhere.

**Basepoint, exp/log, transport.** The attention mechanism only needs the
exp/log maps at the basepoint o = (1/√c, 0, …, 0):

```
exp_o(v) = (cosh(√c‖v‖)/√c,  sinh(√c‖v‖)·v/(√c‖v‖))      v ∈ R^n (spatial)
log_o(x) = (d_H(o,x)/‖x_s‖) · x_s                            x_s = spatial part
```

Parallel transport from o to x is closed-form in the Lorentz model
(τ_{o→x}(v) = v + ⟨x, v⟩_L/(1/c − ⟨o, x⟩_L) · (o + x)) but is **not needed
for v1** of the mechanism: all tangent computations happen at o (§2). It is
recorded here because the Fréchet-mean upgrade (§2.3, deferred) needs it.

## 2. Mechanism design

### 2.1 Scores: the Gromov-product form (NOT raw −d_H)

The naive score is s(q,k) = −d_H(x_q, x_k)/τ. It works, but it **conflates
two different quantities**: how much of q's and k's paths from the root
overlap (hierarchy overlap), and how deep each sits in the hierarchy. The
Gromov product separates them. Define, with the basepoint o as "root",

```
(x|y)_o = ½ · ( d(o,x) + d(o,y) − d(x,y) )
```

— in a tree this is EXACTLY the depth of the lowest common ancestor of x and
y (validated on a worked tree: leaves under a shared depth-1 node give
(x|y)_o = 1.0; leaves under different root children give 0.0). The score is

```
s(q, k) = [ 2·(x_q | x_k)_o  −  λ_q·d(o, x_q)  −  λ_k·d(o, x_k) ] / τ
```

with **learnable per-head radial gates λ_q, λ_k ≥ 0** (init 0). At
λ_q = λ_k = 0 the score is pure hierarchy-overlap (2(q|k)_o); at
λ_q = λ_k = 1 it reduces to −d(x_q, x_k) (the naive form) because the radial
terms cancel against the Gromov product's own d(o,·) terms. The gates let
each head interpolate between "attend to my relatives" and "attend to my
neighbors", and their learned values are a first-class observable (§5).

Queries and keys reach H^n via exp_o of the (linearly projected) head
vectors: x_q = exp_o(W_q h / s_q), with a learnable input scale s_q clamped
to keep ‖v‖ inside the saturation budget (§4).

### 2.2 Degeneration limit I: c → 0 recovers Euclidean attention

As c → 0, d_H(exp_o(u), exp_o(v)) → ‖u − v‖ pointwise, so the naive-score
softmax converges to softmax(−‖u−v‖²/2τ′) under the standard
small-distance/temperature pairing — and that equals **dot-product attention
exactly**, because −‖u−v‖²/2 = u·v − ‖u‖²/2 − ‖v‖²/2 and the −‖u‖²/2 term
is row-constant (softmax-invariant) while −‖v‖²/2 is a per-key bias
absorbed into the key projection's bias capacity. Validated:

| c | max softmax deviation from Euclidean |
|---|---|
| 1e−1 | 8.5e−03 |
| 1e−3 | 9.0e−05 |
| 1e−6 | 9.0e−08 |

Convergence is **linear in c** (each ×100 in c gives ×100 in error). The
F5/F6 acceptance test: at c = 1e−3, attention-weight rows must match
Euclidean attention to 1e−4 sup-norm (the Clifford framework's quaternion
reduction test is the precedent for this pattern). The Gromov form inherits
the same limit with λ_q = λ_k = 1 (it equals the naive form there); the
gated forms acquire the corresponding radial biases.

### 2.3 Value aggregation: tangent-space approximation of the Fréchet mean

True hyperbolic averaging (the Fréchet mean) is an iterative fixed point —
wrong cost profile for attention. v1 uses the standard tangent-space
approximation, which is exact in the c → 0 limit and first-order accurate
in the spread of the attended set:

```
out_i = exp_o( Σ_j  a_ij · log_o(x_vj) )        a_ij = softmax_j(s(q_i, k_j))
```

i.e. log-map the values at the basepoint, average with the attention
weights in the tangent space (an ordinary weighted sum — the existing
attention kernel), exp-map back. Values use their own projection + exp map
(x_vj = exp_o(W_v h_j / s_v)). The output is log-mapped again before the
output projection, so the surrounding transformer sees ordinary R^n vectors:
**the hyperbolic structure is entirely inside the head**, exactly like the
tropical/ultrametric mechanisms. (Upgrade path, explicitly deferred beyond
v1: one Karcher step from the tangent mean using parallel transport — §1
records the transport formula for that day.)

### 2.4 Degeneration limit II: rescaled c → ∞ recovers ultrametric LCP attention

This is the 8gk.6 anchor and what makes hyperbolic the *continuous
completion* of the ultrametric mechanism. δ-hyperbolicity of H^n means the
Gromov product is within δ = δ₀/√c of its tree value, where δ₀ = ln(1+√2)
at c = 1; rescaling distances by √c (equivalently, measuring depth in units
of 1/√c) sends δ → 0: **the rescaled geometry converges to a real tree**,
and the Gromov product converges to the LCP depth that the ultrametric
mechanism computes on hard digits. Quantitative witnesses (validated):

- Two geodesic rays from o at angle θ = 60°: (x|y)_o converges as the rays
  extend (0.6921, 0.6931, 0.6931, 0.6931 at t = 4, 8, 16, 32) to the exact
  closed form **ln(1/sin(θ/2)) = ln 2 ≈ 0.6931** — the branch point of the
  limiting tree sits at depth ln 2, finite and angle-determined.
- The same configuration at c = 4 and c = 16 gives ln2/√c (0.3466, 0.1733)
  — confirming the 1/√c rescaling law on the nose.

The F5/F6 acceptance test (designed in 8gk.6, restated here): embed a known
tree's nodes along geodesic rays (one direction per branch), sweep c upward
with depth measured in rescaled units, and require the score matrix
2(q|k)_o to converge to 2·LCP-depth with error ≤ δ₀/√c — and therefore the
attention weights to converge to the ultrametric mechanism's weights on the
same tree at matched temperature.

## 3. Positioning

Hyperbolic representation learning (Nickel & Kiela 2017/2018), HNN and
HNN++ (Ganea et al.; Shimizu et al.), and hyperbolic transformers/attention
(Gulcehre et al. 2019 and successors) establish that hyperbolic embeddings
and attention are trainable and sometimes win on hierarchical data. What
this framework adds, and what to our knowledge is unrun anywhere:

1. **The certificate-first methodology**: B1-certified invariants (on-manifold
   constraint, exp/log round-trip, both degeneration limits) as runtime
   checks, not paper claims — the same harness that certifies the other 12
   mechanisms.
2. **The three-way hierarchy-bias comparison** on the same task battery at
   equal FLOPs with preregistered predictions: continuous (hyperbolic) vs
   p-adic discrete (ultrametric) vs self-similar (fractal) — §6.
3. **The Gromov-product score with radial gates** as the bridge form whose
   two limits are existing mechanisms in this codebase, with both limits as
   acceptance tests. The naive −d form appears in prior art; the gated
   Gromov form with the LCP completion does not, to our knowledge.

## 4. Numerics (the section this mechanism lives or dies on)

**N1. Saturation budget.** fp32 viability requires d(o, x) ≤ D_max with
D_max ≈ 80 in the Lorentz model (cosh overflow at ~89; one order of margin).
Policy: clamp tangent norms at exp_o input: ‖v‖ ≤ D_clamp = 32 (per-head,
pre-curvature; in rescaled units this is depth 32√c — far beyond any
realistic hierarchy while leaving 2.5× headroom to overflow). The learnable
input scales s_q, s_k, s_v are parameterized as softplus(raw) + 0.1 so they
cannot collapse to 0 (which would freeze all points at o).

**N2. acosh near 1.** d_H's acosh(−c⟨x,y⟩_L) has argument → 1⁺ for nearby
points, where acosh'(z) = 1/√(z²−1) → ∞: raw fp32 gradients explode for
near-coincident pairs. Policy: clamp the argument to ≥ 1 + 1e−7 (validated:
this recovers d = 5..40 to 4 decimals in fp32 while bounding the gradient).
Equivalently the implementation may use the stable form
acosh(1 + 2u) with u = c·‖x−y‖²_L-based expression when u is small; v1 uses
the clamp (simpler, certified by the round-trip check).

**N3. Constraint maintenance.** Optimizer steps push points off the
hyperboloid. Policy: after every update of any manifold-valued tensor (none
exist in v1 — all manifold points are produced by exp_o in the forward pass;
this policy matters for the deferred Karcher upgrade and for any cached
value points), re-project by recomputing the time coordinate
x₀ = √(1/c + ‖x_s‖²). Validated: 200 drift+project cycles hold
max |⟨x,x⟩_L + 1/c| = 3.6e−07 (fp32 eps-level, no accumulation).

**N4. bf16 policy.** Under autocast, Minkowski inner products and acosh run
in **fp32** (the products mix +/− terms of magnitude up to cosh(D_clamp) —
catastrophic cancellation in bf16's 8-bit mantissa); the tangent-space
weighted sum (the bulk of FLOPs, an ordinary matmul) stays in bf16. This
mirrors the house pattern (rmatrix charge telemetry computes fp64 on CPU;
SDPA math-backend pinning in the symplectic kick).

**N5. Curvature parameterization.** c per head, c = softplus(c_raw)/H_scale,
init c ≈ 1.0, floored at 1e−4: below the floor the mechanism IS Euclidean
attention to 1e−4 (§2.2 table) so smaller values are unidentifiable — the
floor prevents the optimizer from wandering in a flat direction. No upper
clamp: large c is the tree limit and well-behaved after rescaling (§2.4);
the saturation budget already constrains depth × √c through D_clamp.

## 5. Telemetry (first-class deliverable, not an afterthought)

Per the u55.5/nyp pattern: when `hyperbolic_record_geometry` is on, each
step record carries

- `hyperbolic_c_per_head` — the learned curvatures: **the
  which-geometry-does-language-want observable.** c → floor means the head
  wants flat (Euclidean) geometry; c growing means it wants tree-like
  geometry. This single readout is the framework's headline diagnostic.
- `hyperbolic_lambda_q_mean` / `_lambda_k_mean` — the radial gates:
  relatives-vs-neighbors attention style per head.
- `hyperbolic_depth_mean` / `_depth_max` — d(o, x) statistics vs the
  D_clamp budget (saturation early-warning, renders in the nyp dashboard's
  invariant panel automatically).

## 6. Preregistered predictions (registry-ready; register BEFORE any training evidence)

To be registered with `candidate_variant` selectors (the engine reads
`candidate_variant`, not `variant` — z4xx lesson) and `--val-interval > 0`
on any campaign (run_campaign val-cadence lesson):

1. **hyp-hyperbolic-hier-heldout-depth** (the depth ≥ 4 claim):
   `evaltasks:tasks.hier.exact_match.greedy.held_out.mean`, comparator ≥,
   absolute_delta +0.05 vs `{mechanism: standard, equal_flops: true}`,
   min_seeds 3, with the dial pinned so held-out depth ≥ 4 (the regime where
   continuous hierarchy should pay; validity.baseline_floor from the
   recorded answer prior, as house policy).
2. **hyp-hierarchy-bias-three-way** (the headline comparative): on the same
   hier battery at equal FLOPs, the ORDERING hyperbolic ≥ ultrametric ≥
   fractal on held-out-depth EM, stated as an explicit open hypothesis.
   Operationalization note (this is honest): the current engine adjudicates
   one candidate arm against one baseline — a three-way ordering needs
   either three pairwise registrations (hyperbolic>ultrametric,
   ultrametric>fractal, hyperbolic>fractal, each absolute_delta ≥ 0 with
   min_seeds 3) or an ordering-aware extension; register the three pairwise
   forms, and treat "all three supported" as the ordering verdict. The
   prediction itself (which geometry wins) is registered as: hyperbolic
   wins at held-out depth ≥ 4; ultrametric wins when the dial makes the
   hierarchy EXACTLY tree-discrete (integer depths, no geometry between
   levels); the gap between them shrinks as c grows (the §2.4 completion).
3. **hyp-hyperbolic-c-tracks-task-geometry** (the telemetry claim): on hier
   the per-head learned c (mean over heads, `train:` metric from §5
   telemetry) ends ABOVE its init; on a flat-structure control task (rel or
   needle) it ends at/below init — a two-task dissociation registered as
   two single-arm predictions on the recorded telemetry.

Wave-0 sizing probe (NOT evidence; `artifacts/probes/sizing/` per the dzor
policy): 2 seeds of hyperbolic on hier at the E1 rung to read variance,
then size min_seeds with `mgr hypotheses power` before registering exact
thresholds for (2).

## 7. Implementation plan (F5/F6-ready; zero further design decisions)

**nanochat (mnn.6), in-place in the house pattern:**

1. `nanochat/hyperbolic_attention_torch.py`: `HyperbolicCausalSelfAttention
   (AttentionCore)` — new file justified as a genuinely new mechanism
   (precedent: every other mechanism module). Internals: linear q/k/v
   projections (scaffold-standard names), per-head `c_raw`, `lambda_q_raw`,
   `lambda_k_raw`, `s_{q,k,v}_raw`; forward = exp_o → Gromov scores (§2.1)
   with causal mask → softmax → tangent aggregation (§2.3) → log_o → output
   projection. Minkowski ops in fp32 under autocast (N4); clamps per N1/N2.
2. `gpt.py`: `attention_type: "hyperbolic"`; config fields
   `hyperbolic_record_geometry: bool = False` only (everything else is
   learnable, not configured — fewer knobs, fewer goldens recaptures).
   GOLDENS: adding the config field requires the standard recapture with
   config-only diff verification.
3. `train.py`: telemetry collection per §5 (mirror
   `_collect_symplectic_energy_stats`).
4. **Certify (B1)**: `hyperbolic.on_manifold` (exp_o output satisfies the
   constraint to 1e−6), `hyperbolic.exp_log_roundtrip` (log_o(exp_o(v)) = v
   to 1e−6 inside the clamp budget), `hyperbolic.euclidean_limit`
   (§2.2 test at c = 1e−3, tol 1e−4), `hyperbolic.lcp_limit` (§2.4 tree
   test, tol δ₀/√c at c = 64), `hyperbolic.causality_no_future_grad`
   (shared harness).
5. **Tests**: the four certify properties as pytest (fp64), plus the
   ln(1/sin(θ/2)) closed-form witness (θ = 60° → ln 2, validated above) and
   the fp32 saturation table as regression pins.

**JAX demo (mnn.5):** standalone
`hyperbolic_negative_curvature_attention.py` demo in the demo-file pattern:
tree-embedding visual, the two degeneration limits as printed property
checks, the three-way comparison teaser on a toy hierarchy.

**Order:** mnn.5 and mnn.6 are independent given this doc; 8gk.6 needs only
§2.4 + the certify designs and can start immediately.

## 8. Validation receipts

All numbers in this document were produced by a standalone validation run
(fp32/fp64 torch, no repo dependencies) before the document was written:
Poincaré-vs-Lorentz saturation table (§1), the c → 0 convergence table
(§2.2), the tree Gromov-product identities and the ln 2 ray witness with
its 1/√c scaling (§2.4), and the 200-cycle constraint-maintenance bound
(N3). The script lives at the validation-scratch path and its assertions
are the seeds of the §7 test suite.
