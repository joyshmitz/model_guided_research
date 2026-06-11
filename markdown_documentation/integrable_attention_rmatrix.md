# Integrable Attention: Yang–Baxter R-Matrices, Conserved Charges, and the Group Word-Problem Barrier

*Bead: model_guided_research-u55.3 (EPIC THEORY-II, u55). Implementation:
`nanochat/braid_attention_torch.py` (`--braid-crossing-law rmatrix`); tests:
`tests/test_mathematical_properties.py::TestIntegrableRMatrix`; certificates:
`mgr certify -m braid` (the five `braid.rmatrix_*` checks); probe:
`mgr probe-charges`; preregistration: `hypotheses/registry.yaml`
(`hyp-rmatrix-*`).*

The braid mechanism's crossing updates were heuristic with an optional YBE
check. This note goes all in on integrability: the crossing law becomes the
trigonometric R-matrix of U_q(sl₂) with spectral parameter, and the structure
this buys — commuting transfer matrices, a conserved-charge tower, exact
invertibility, a principled relative-position theory — is proved for our exact
configuration and instrumented live in training.

## 1) The R-matrix, self-contained

On C² ⊗ C² with basis |00⟩, |01⟩, |10⟩, |11⟩ define the **a-normalized
trigonometric six-vertex R-matrix** with deformation η > 0 (q = e^η) and
spectral parameter w:

    R(w) |00⟩ = |00⟩                      R(w) |11⟩ = |11⟩
    R(w) |01⟩ = b(w) |01⟩ + c(w) |10⟩     R(w) |10⟩ = c(w) |01⟩ + b(w) |10⟩

    b(w) = sinh(w) / sinh(w + η),         c(w) = sinh(η) / sinh(w + η).

Three exact identities (all asserted in fp64 by tests/certificates):

- **Yang–Baxter equation** (braid form, with Ř = P·R and Ř₁₂ = Ř ⊗ I,
  Ř₂₃ = I ⊗ Ř on (C²)^⊗3):

      Ř₁₂(u) Ř₂₃(u+v) Ř₁₂(v) = Ř₂₃(v) Ř₁₂(u+v) Ř₂₃(u).

  Verified to ~1e-14 over randomized (u, v, η)
  (`braid.rmatrix_braid_relation_holds`).
- **Regularity**: b(0) = 0, c(0) = 1, so Ř(0) = I — crossing two strands of
  equal rapidity does nothing. Mixing strength is a function of rapidity
  *difference* only.
- **Inversion (unitarity)**: using sinh(η+w)·sinh(η−w) = sinh²η − sinh²w,

      Ř(w) Ř(−w) = I    exactly.

  The mixing is exactly invertible by the same law at negated argument: this
  is the algebraic content of "topologically protected information" — nothing
  the layer does can destroy information, because its inverse is another
  member of the same family (`braid.rmatrix_inversion_relation_holds`).

### 1.1) The one-particle braid form (what the code applies to strand pairs)

R conserves particle number (U(1) symmetry: the four nonzero amplitudes never
change total spin), so the one-particle sector of (C²)^⊗N — span of
|0…1ᵢ…0⟩ — is invariant. Restricted to that sector, Ř₁₂(w) acts on the two
amplitudes at positions (1,2) as

    N(w) = [ c(w)  b(w) ]
           [ b(w)  c(w) ]      (and identity on every other position).

Because the sector is invariant, **N inherits the braid relation verbatim**:
with rapidities riding on strands (they swap at each crossing; the crossing
argument is the rapidity difference of the two positions read before the
crossing), σ₁σ₂σ₁ = σ₂σ₁σ₂ exactly. This is the crossing law of
`rmatrix_crossing`: strand states are the (x, y) pairs of the braid program
model, and N(w) ⊗ I₂ acts on the pair identically per coordinate. The
*restricted* heuristic law fails this relation with residual O(1) — the
separation witness proving the test has teeth
(`test_restricted_law_fails_braid_relation`).

### 1.2) A subtlety that cost a debugging session: gauges

Per-crossing stochastic normalization N̂(w) = N(w)/(b+c) — attractive because
it makes rows sum to 1 — **breaks the lifted braid relation**: the scalar does
not multiply the identity slots of N ⊕ 1, so it does not factor out of the
composition (`test_stochastic_gauge_does_not_satisfy_lifted_braid_relation`
asserts the failure). Consequently:

- the **law** (everything YBE-, transfer-, and inversion-certified) is the
  a-normalized N(w);
- the **value kernel** (§2) uses the stochastic gauge (b̂, ĉ) = (b, c)/(b+c)
  per crossing — a diagonal gauge of the same data, chosen so the transported
  mass is an exact partition of unity. The gauged law still satisfies
  inversion: (b+c)(w)·(b+c)(−w) = 1, so N̂(w)N̂(−w) = I too.

## 2) The attention kernel: a causal monodromy sweep

Per head, learn a deformation η = softplus(η_raw) + 0.01 and a **strictly
monotone rapidity profile** u₁ < u₂ < … (cumulative softplus increments; the
table is allocated at 10× sequence_len, matching the rotary cache's
over-allocation, so held-out 2×–8× lengths extrapolate by uniform
continuation of the default increment — the integrable analogue of RoPE's
by-formula extrapolation).

Query i's output is the **monodromy of a fresh auxiliary strand** of rapidity
u_i scattering through its prefix j = 1..i in order, with crossing
N̂(u_i − u_j). Unrolling the 2×2 recursion m_j = ĉ m_{j−1} + b̂ v_j gives the
closed form

    out_i = Σ_{j≤i} A_ij v_j,
    A_ij  = b̂(u_i − u_j) · Π_{j<j'≤i} ĉ(u_i − u_{j'}),

computed in parallel with one cumulative log-sum over log ĉ (O(T²), same
class as standard attention; decode recomputes one row per step, giving exact
KV-cache parity — the house decode contract,
`test_nanochat_braid_rmatrix_kv_cache_parity_and_charges`).

Properties, all consequences of monotone rapidities (every causal argument
w = u_i − u_j ≥ 0, and the only pole of the weights sits at w = −η < 0):

- b̂ ∈ [0, 1), ĉ ∈ (0, 1]: weights bounded, products decay — **stability
  without normalization layers**; the kernel is an exponential-decay causal
  kernel with learned per-position decay rates (the integrable cousin of
  RetNet/SSM decay kernels, but derived rather than ansatz'd).
- **Difference property = relative position.** A_ij depends on positions only
  through rapidity differences; learned profiles are learned position warps.
  RoPE is a particular phase ansatz; rapidities are the integrable
  generalization, and monotonicity ("time flows forward") is exactly the
  condition placing every causal crossing in the stable region.
- The self-crossing has w = 0, so A_ii = 0 (Ř(0) = I): a token does not
  attend to itself; the residual stream carries identity.

## 3) Commuting transfer matrices: the proof for our configuration

**Setup (our exact boundary conventions).** Finite chain of length T, sites
carrying C², *inhomogeneities* u₁..u_T = the layer's learned rapidities,
auxiliary space a ≅ C², periodic closure via the partial trace over a. Define

    L_{a,i}(θ) = R_{a,i}(θ − u_i),      T_a(θ) = L_{a,T}(θ) ⋯ L_{a,1}(θ),
    t(θ) = tr_a T_a(θ).

**Theorem ([t(θ), t(θ')] = 0 for all θ, θ').** Proof, in full:

1. *RTT exchange relation.* On V_a ⊗ V_b ⊗ (C²)^⊗T,

       R_{ab}(θ−θ') T_a(θ) T_b(θ') = T_b(θ') T_a(θ) R_{ab}(θ−θ').

   Induction on T. Base T = 1: this is the YBE
   R_{ab}(θ−θ') R_{a,1}(θ−u₁) R_{b,1}(θ'−u₁) =
   R_{b,1}(θ'−u₁) R_{a,1}(θ−u₁) R_{ab}(θ−θ') — the standard form with the
   *same* shift u₁ subtracted from both site arguments, which preserves the
   difference (θ−u₁) − (θ'−u₁) = θ−θ'. Inhomogeneities are therefore free.
   Step: T_a(θ) = L_{a,T}(θ) T'_a(θ) with T' the length-(T−1) monodromy;
   operators on site T commute with operators on sites < T, so

       R_{ab} T_a T_b = R_{ab} L_{a,T} L_{b,T} T'_a T'_b
                      = L_{b,T} L_{a,T} R_{ab} T'_a T'_b          (YBE at site T)
                      = L_{b,T} L_{a,T} T'_b T'_a R_{ab}          (induction)
                      = T_b T_a R_{ab}.                            ∎(1)

2. *Trace.* R_{ab}(θ−θ') is invertible whenever sinh(θ−θ'+η) ≠ 0 and
   (θ−θ') ≠ ±η (the middle 2×2 block has determinant
   (sinh²(θ−θ') − sinh²η)/sinh²(θ−θ'+η)). For such generic (θ, θ'):

       t(θ) t(θ') = tr_{ab}[T_a T_b]
                  = tr_{ab}[R_{ab}^{-1} T_b T_a R_{ab}]   (by (1))
                  = tr_{ab}[T_b T_a]                       (cyclicity in ab)
                  = t(θ') t(θ).

   Both sides are entire functions of (θ, θ'), so the identity extends from
   the generic set to all values. ∎

Tests verify this two ways: the **dense 2^T tensor construction** on small
chains (T = 5, residual ~1e-17, fp64), and the **closed-form one-particle
restriction** (next section) at T = 24
(`test_transfer_matrices_commute_and_perturbation_breaks_it`,
`braid.rmatrix_transfer_matrices_commute`). Commutativity is *not* generic:
perturbing one Boltzmann weight by ε breaks it detectably at ε = 1e-6 already
(`braid.rmatrix_perturbed_transfer_separates`). A finding worth recording:
the *existing heuristic laws are too algebraically degenerate to even fail
this test* — their accumulator structure is abelian (the soft law's effective
pair map is I + p·E with one nilpotent generator), so transfer-like products
built from them commute for a trivial reason. Their failure is at the braid
relation (Q2), not at commutativity; the two-charge fingerprint of §5
separates everything.

### 3.1) The one-particle transfer matrix in closed form

Restricted to the one-particle sector, t(θ) is the T×T matrix (b_i = b(θ−u_i),
c_i = c(θ−u_i)):

    A-part (aux returns in |0⟩):  A_jj = b_j ;  A_kj = c_j c_k Π_{j<l<k} b_l   (k > j)
    D-part (aux returns in |1⟩):  D_jj = Π_{l≠j} b_l ;
                                  D_kj = c_k c_j Π_{l<k} b_l Π_{l>j} b_l       (k < j)
    t(θ) = A + D.

Derivation: in the A-element the auxiliary spin stays |0⟩ at both ends, so it
either passes the particle's site diagonally (amplitude b_j) or picks the
particle up at j (amplitude c_j, aux flips to |1⟩), carries it (amplitude b_l
per empty site), and deposits it at k > j (amplitude c_k); the D-element is
the time-reverse with deposit before pickup, wrapping around the trace. This
closed form (`one_particle_transfer`) equals the dense construction to ~1e-18
(`test_transfer_closed_form_matches_dense_tensor_construction`) and makes the
commuting family computable at real sizes in O(T²) — which is what the charge
probe uses.

## 4) The conserved charges Q₁, Q₂

**Canonical charges (homogeneous limit, u_i ≡ u₀).** At θ = u₀ regularity
gives R(0) = P, so t(u₀) = tr_a[P_{aT}⋯P_{a1}] = the cyclic shift U: the
**momentum charge** Q₁ = t(u₀), with U = e^{iP̂}. The log-derivative at the
same point is local:

    Q₂ = d/dθ log t(θ)|_{θ=u₀} = Σ_i h_{i,i+1},
    h = (1/sinh η) [ ½(σ^x σ^x + σ^y σ^y) + (cosh η)/2 (σ^z σ^z − 1) ]   (up to additive constant)

— the XXZ Hamiltonian with anisotropy Δ = cosh η. Higher log-derivatives
continue the tower; all commute by §3. For the inhomogeneous chain the family
{t(θ)} itself is the tower: any probe values t(θ₁), t(θ₂), … generate the
same commutative algebra, which is what the implementation uses (no
homogeneous limit needed at runtime).

**Operational charges (what the code measures every logged step).** The bead
demands drift ~0 for rmatrix and measurably nonzero for the heuristic laws.
Two observables do this with *exact* theorems behind them:

- **Q₁ — mass partition.** Because b̂ + ĉ = 1 per crossing, the sweep
  telescopes: Σ_j A_ij + Π_{j≤i} ĉ(u_i − u_j) = 1 **exactly** for every
  query — transported mass plus untransported auxiliary mass is a partition
  of unity. Logged as `braid_q1_mass_defect_max` (fp32 forward: ~1e-6;
  asserted < 1e-5). The heuristic modes are additive accumulations with no
  partition — their defect is O(1)
  (`braid.rmatrix_mass_partition_charge_conserved`, drift test).
- **Q₂ — path independence of transport.** The live layer's crossing law,
  composed along the two braid-equivalent schedules on a probe triple with
  its *actual learned* (η, rapidities): residual ~1e-15 for rmatrix (the
  braid relation), O(1) for the restricted law. Logged as
  `braid_q2_braid_residual_max`.

The **two-charge fingerprint** separates all modes
(`test_charge_drift_through_layer_stack_separates_laws`): rmatrix passes both;
restricted/soft fail both; the constant swap-output "ybe" law passes Q₂ (it
does satisfy R3) but fails Q₁ — protected *paths* without conserved *mass*.

**Charge decodability (the missing success criterion, referee round 3).**
Conservation of a useless quantity is free; the protected-memory story needs
charges that *encode the task state*. `mgr probe-charges` trains linear and
small-MLP probes from the per-head charge observables
q_{hk} = ⟨v_h, t_h(θ_k) v_h⟩ / ⟨v_h, v_h⟩ (final braid layer, v = its value
sequences, t_h from its learned parameters) to the ground-truth composed group
element. The representation-theoretic compatibility condition, checked on
paper before training anything: our R is U(1)-symmetric, so the charge tower
lives in a *commutative* algebra and its observables are (rapidity-weighted)
symmetric statistics of the token sequence. A task state that is a function
of symmetric statistics — **Z60**: the product is the generator-count sum mod
60 — is in-span; a state that is order-sensitive in an essential way —
**S5/A5** products — is not. The preregistered prediction
(`hyp-rmatrix-charge-decodability`) is therefore a *dissociation*: decodable
on the abelian control, at/near chance on the non-solvable groups. If it
fails everywhere, the honest conclusion is "integrable structure conserves
the wrong quantities for this task," and the redesign direction is a
higher-rank R (U_q(sl_n) fundamental ⊗ fundamental) or a non-abelian charge
tower whose span meets the group algebra — recorded here as the design
consideration the result would activate.

## 5) Temperley–Lieb quotient and Markov-trace readouts (design)

Baxterization runs in reverse for our family: Ř(w) = (sinh(η−w)·I +
sinh(w)·Ě)/sinh(η+w)... more usefully, in the TL idiom, Ř(w) is a linear
combination of I and the TL generator E with E² = (q + q^{-1})E = 2cosh(η)·E
and E_i E_{i±1} E_i = E_i. The TL algebra carries the **Markov trace** tr_M
(the trace with the q^{σ^z}-weighted closure), whose values on braid words
are Jones-polynomial evaluations — *invariant under Markov moves*, i.e.
under exactly the rewrites that change a braid word without changing its
closure. Design for sequence-level invariant pooling: evaluate tr_M on the
layer's crossing word at K spectral points and concatenate — a pooled
representation invariant to schedule rewrites by construction.

**The causal realization (bead 0i1v).** A global pooled trace cannot enter
the residual stream of a causal LM (it reads the future), and the Markov
trace needs a *closed* word while every causal prefix is open. The open-word
analogue of a trace evaluation is the **monodromy element itself at a fixed
spectral point**: for each query position i, sweep the prefix with K extra
probe rapidities θ_k = u_i + δ_k (learned δ_k > 0 keeps every causal argument
in the stable region) and gate the resulting transported values into the
output with zero-init per-(head, probe) gates (`--braid-rmatrix-probes K`,
parameters `rmatrix_probe_delta_raw` / `rmatrix_probe_gate`). At init the
mechanism is bitwise the base sweep (asserted exactly in
`test_nanochat_braid_rmatrix_spectral_probes_zero_init_and_kv_parity`);
training opens the spectral views it finds useful. Each view satisfies its
own mass-partition identity; Q1 telemetry tracks the base view, and the
gated views are explicit, learned departures from it. The closed-word Markov
trace remains available off-line: the charge observables ⟨v, t(θ_k) v⟩ that
`mgr probe-charges` computes are precisely its quadratic shadows.

## 6) Circuit class: precisely scoped claims

What we claim and do not claim. The rmatrix mixing is **linear in the values
with input-independent (per-position) weights**. A fixed-depth stack of such
layers with MLPs remains inside the TC⁰-type upper bounds known for
fixed-depth transformers (Merrill–Sabharwal line); composing S₅ words is
NC¹-complete under AC⁰ reductions; therefore **we claim no expressivity
separation** — an R-matrix transformer does not escape the state-tracking
barrier in the worst case. The honest scientific question, and the only one
we preregister, is whether structured mixing changes **learnability and the
length-generalization slope** at fixed depth and equal FLOPs: the integrable
kernel composes associatively along the sequence by construction (the
monodromy IS a sequential composition), so the inductive bias matches the
task's algebraic structure even though the ceiling does not move. The
preregistered observables (`hyp-rmatrix-s5-length-slope`,
`hyp-rmatrix-solvable-control-specificity`) are doc-level OLS slopes of
held-out exact-match against word length (`length_slope` in
mgr.evaltasks.v2), with the solvable controls required to *shrink* the gap —
if the win does not track non-solvability, the story is wrong and the
registry will say so.

## 7) Engineering notes

- **Purely positional law**: the scaffold-mandated q/k projections are dead in
  rmatrix mode; they are frozen (`requires_grad_(False)`) so Muon never sees
  parameters without gradients. The rapidity table is 2D but is a per-position
  scalar field, not a matmul weight — `setup_optimizers` routes it to AdamW
  (Newton–Schulz orthogonalization of a rapidity profile is meaningless).
- **Goldens**: the default braid path constructs the same modules in the same
  RNG order as before (rmatrix parameters are deterministic-init and created
  only when the law is selected); the braid trajectory fixture is bitwise
  unchanged.
- **Telemetry**: `braid_q1_mass_defect_max`, `braid_q2_braid_residual_max`,
  per-layer η and rapidity spans stream to metrics.jsonl per logged step
  (D2 schema), giving the integrable-vs-heuristic diagnostic live during any
  braid run.
- **e2e**: `scripts/e2e_pipeline.py --scenario word-problem` composes
  gen-tasks(group) → train(rmatrix + standard) → eval-tasks with `--dial
  length=4`, asserting the charge telemetry and the slope tables land in the
  artifacts (rz8.8 maintenance contract).

## 8) Positioning

Quantum-group/representation-theoretic sequence models exist in several
flavors — unitary/orthogonal RNNs (uRNN line) constrain the *spectrum* of
recurrence for gradient stability; quaternion/Clifford networks constrain the
*algebra* of features; recent integrable-spin-chain × ML work (R-matrix
parameterizations of normalizing flows, Bethe-ansatz-inspired layers) uses
Yang–Baxter structure generatively. What is different here: (1) **spectral
parameters with the difference property as the position theory** — relative
position is derived from integrability rather than bolted on; (2) **the
charge tower as a live training diagnostic** — conservation is measured per
step against an exact partition-of-unity theorem, with a separation
fingerprint against the non-integrable laws sharing the same scaffold; (3)
**the word-problem protocol with preregistered mechanism-specificity
controls** — the solvable groups must close the gap, which makes the claim
falsifiable in the registry rather than rhetorical; (4) **the charge-decoding
probe with its representation-theoretic prediction written down before
training** — including the predicted abelian/non-abelian dissociation that
follows from the U(1) symmetry of the six-vertex family.
