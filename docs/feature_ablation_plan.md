# Math-Feature Ablation Matrix (plan)

**Bead:** `model_guided_research-qq7`
**Date:** 2026-06-14
**Status:** plan only (no code changes). Operationalizes *within-mechanism*
feature isolation on top of the existing fixed-FLOPs A/B infrastructure.

## Goal & the gap it fills

`mgr bench-fixed-flops` (bead `z7r`) already does multi-seed, fixed-FLOPs A/B
**across attention types** (standard vs tropical vs reversible …) with
Welch-delta-vs-baseline, CI95, NaN/Inf, tokens/s (`cli.py:1192,1778`). That
answers "*does mechanism X beat standard?*"

It does **not** answer "*which sub-feature of X is doing the work?*" — because
its arm selector (`-a <attention_type>`) only varies `attention_type`, holding
every other knob at default. This plan defines the **within-mechanism** matrix:
toggle one math feature at a time (control = behavior-preserving default,
treatment = feature on/off) at matched FLOPs, so each feature's marginal
contribution is isolated.

Every ablation here is a mechanism-vs-baseline comparison and therefore goes
through the standard methodology — fill `docs/campaign_preregistration_template.md`
into a bead **before** evidence (rung-finding probe, sample-efficiency-vs-
asymptotic split, power-derived seed count, one stopping rule), and pass the
`docs/new_mechanism_checklist.md` certify/placebo gates. This doc does **not**
restate that methodology; it enumerates *what to toggle*.

## The matrix (toggleable sub-features)

All flags are `GPTConfig` fields with matching `--…` train flags; control is the
behavior-preserving default (see `docs/integration_matrix.md` for the full knob
list). Each row is one preregistered ablation.

### Tropical (`--attention-type tropical`)
| Feature | Control | Treatment | Isolates |
|---|---|---|---|
| `tropical_gauge_fix` | `True` | `False` | gauge-fixing (centering/anchoring) contribution |
| `tropical_score_center` | `True` | `False` | score centering's effect on routing |
| `semiring_beta` | `None` (exact max) | finite β (+ anneal) | Maslov soft-vs-hard max (the dequantization axis, 8gk.1) |
| `ffn_type` | `standard` | `tropical` / `tropical-rational` | the 1-Lipschitz FFN's marginal effect (8gk.8) |

### Ultrametric (`--attention-type ultrametric`)
| Feature | Control | Treatment | Isolates |
|---|---|---|---|
| `ultrametric_mode` | `kernel` | `trie` (CPU) / `balltree` | exactness/scaling path vs O(T²) kernel (33dd) |
| `ultrametric_digits_k` | `None` (full) | k < K | digit-precision = valuation truncation (8gk.4/tcuy) |
| `ultrametric_hard_digits` | `False` | `True` | soft vs hard digit routing |
| `ultrametric_alpha` | `2.0` | sweep | LCP decay strength |

### Reversible (`--attention-type reversible`)
| Feature | Control | Treatment | Isolates |
|---|---|---|---|
| `reversible_mode` | `additive` | `symplectic` | symplectic structure (⚠ 3× FLOPs — budget by `--max-steps`, 7lba) |
| `reversible_tied` | `False` | `True` (symplectic) | layer-tied shadow-conservation regime (u55.5) |
| `reversible_lambda_min` | `0.05` | `0.0` | confinement vs unconfined falsification control |

### Braid (`--attention-type braid`)
| Feature | Control | Treatment | Isolates |
|---|---|---|---|
| `braid_mode` | `soft` | `discrete` | soft approx vs discrete braid-word decoder |
| `braid_crossing_law` | `restricted` | `ybe` / `rmatrix` | YBE/integrability contribution (k2y/u55.3) |
| `braid_tau` | `0.0` | sweep | crossing temperature |

### Standard / cross-cutting
| Feature | Control | Treatment | Isolates |
|---|---|---|---|
| `disable_block_norms` | `False` | `True` (standard only) | the norm layers (z4xx no-norm control) |
| `ffn_type` (any attn) | `standard` | `tropical`/`tropical-rational` | FFN semiring axis, independent of attention |
| `use_flex_attention` | `False` | `True` | **placebo** — kernel only, identical math; expect Δloss≈0, Δthroughput≠0 |

### Monolithic mechanisms (ablate mechanism-vs-standard only)
Gauge (parallel-transport wrapper — no sub-flags in the torch block),
simplicial (2-hop), fractal (router depth/m internal), quaternion/octonion
(rotor normalization is now always-on), surreal attention (the NSA *width*
parameterization is a separate axis, `lab.1`). For these, the `z7r`
across-type suite is already the right tool.

## Metrics & logging

Reuse the `bench-fixed-flops` schema (`cli.py:1778`): per-arm `mean/std`,
`delta_vs_base`, `ci95_lo/hi`, `welch_p`, `nan_inf`, `tokens_s_mean`, over
power-derived seeds. Score metric = the registered one (typically held-out
`val_ce` exact-match accuracy on the relevant `evaltasks` family — note raw
`val_ce` is **not** comparable across norm/no-norm arms, the z4xx lesson, so
prefer exact-match). Add the toggled feature flag(s) as extra columns so each
row is self-describing. Pair every quantitative arm with its **certificate**
observable via `mgr certify` (the mechanism invariant: tropical margins,
reversible det≈1 / shadow-energy drift, etc.) as a leading-indicator channel.

## Fixed-FLOPs discipline (must-match across arms)

- `--grad-clip-norm 1.0` on every arm (`docs/config_parity.md`).
- Match `(B, T, world_size)` and `--target-flops`; remember matched dims ≠
  matched FLOPs/token, so the budget auto-adjusts steps — **except** symplectic
  (3× charge): budget by `--max-steps` for the strict equal-compute contract.
- Use the `configs/` templates as arm bases so dims/seed/fairness flags are
  fixed; vary only the one ablated flag.

## Harness gap & how to run today

`bench-fixed-flops` varies only `attention_type`, so within-mechanism toggles
need one of:

1. **(recommended) thin harness extension** — let the arm selector accept
   `attention_type + extra-flags` arm specs (e.g. `tropical@gauge_fix=False`)
   so two same-`attention_type` arms can be compared with the existing Welch
   path. Small, localized change to `cli.py:1192`.
2. **today, no code** — drive arms with `scripts/run_campaign.py
   --combo <task>:<mech> --extra-args "--<flag> <val>"` (one topic per arm),
   then compare with `mgr regressions` or the `_bench_welch_delta` logic
   (`cli.py:1144`). Frozen-worktree provenance comes for free.

Statistics across the full matrix: Welch per row + Benjamini-Hochberg FDR over
the family (the ci-v6 engine already power-gates and FDR-corrects). Include a
**placebo row** (`use_flex_attention`) per suite as a negative control — a
significant placebo delta means the harness or seeds are mis-specified.

## Suggested first slice (highest-signal, cheapest)

1. Tropical `semiring_beta` None→finite (the dequantization axis; infra exists).
2. Tropical `ffn_type` standard→tropical (closes the certified-chain hole, 8gk.8).
3. Standard `disable_block_norms` False→True (the z4xx control, already wired).
4. Reversible `reversible_lambda_min` 0.05→0.0 (confinement, `--max-steps`).

Each as its own preregistered bead (`--deps discovered-from:model_guided_research-qq7`).

## Follow-ups

- **Extend `bench-fixed-flops` arm specs** to carry per-arm extra flags
  (harness gap above). → file when the first within-mechanism ablation is
  scheduled.
</content>
