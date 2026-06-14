# Rich Console Styling Audit

**Bead:** `model_guided_research-iso`
**Date:** 2026-06-14
**Scope:** console output across `cli.py`, `nanochat/`, the root JAX demos, and
infra utilities. Audit only — no code changes (per bead). Aligns with the
CLAUDE.md directive: *"all console output should be informative, detailed,
stylish, colorful … by fully leveraging the rich library."*

**Bottom line:** the **user-facing surfaces are already well-styled** —
`cli.py` (176 `console.print` vs 4 plain `print`), `nanochat/train.py` (31
`console.print` + rich tables/panels), `report.py`, `serve.py`. The consistent
gaps are: (1) the **JAX demos are mixed** — rich panels for headlines but plain
`print()` for the actual *results* (timing/accuracy lines) and warnings; (2)
the **dataset download UX is plain** (no progress bar); (3) scattered infra
diagnostics print plain. A shared `console` already exists (`utils.py:20`), so
the fix is convention, not new infrastructure.

## Well-styled (no action)

| File | Evidence |
|---|---|
| `cli.py` | 176 `console.print`, rich `Table`/`Panel`/`Progress`/`box` (`cli.py:21–34`); only 4 plain prints |
| `nanochat/train.py` | 31 `console.print` + rich report tables |
| `nanochat/report.py`, `nanochat/serve.py` | rich-based |

## Gaps (ranked by user impact)

### 1. Dataset download — plain, should be a Progress bar  ★ highest UX impact
`nanochat/dataset.py` prints download lifecycle plainly (10 calls,
`dataset.py:120–153`): `Downloading …`, `Successfully downloaded …`,
`Attempt n/5 failed …`, `Waiting … seconds`. This is a **long-running,
user-facing** operation and the prime candidate for `rich.progress.Progress`
(per-shard bar + total) with `[green]`/`[yellow]`/`[red]` status. Highest
return because shard downloads are slow and currently give a wall of text.

### 2. JAX demos — rich panels, plain results  ★ consistency
The demos open with rich panels but print their **substantive output** plainly:

- **Result lines** — e.g. `ultrametric_worlds_and_p_adic_computation.py:791`
  `Timing(s): eval_pre=… train=… eval_test=…`, `:865`
  `Small A: pre=… post=… test=… created=…`. These should be `rich.Table`
  rows (metric / value), matching the rich panels around them.
- **Warnings/fallbacks** — e.g. `:602` `[ultrametric] Could not cache head
  sims: {err}`, `:904` `… sanity check failed: {err}`. Should be
  `console.print("[yellow]…[/]")` / `[red]` so failures are visible.

Plain-`print()` counts (results+warnings interleaved with rich panels):

| Demo | plain `print()` |
|---|---|
| `ultrametric_worlds_and_p_adic_computation.py` | 37 |
| `matrix_exponential_gauge_learning.py` | 26 |
| `reversible_computation_and_measure_preserving_learning.py` | 25 |
| `ordinal_schedules_and_well_founded_optimization.py` | 10 |
| `iterated_function_systems_and_fractal_memory.py` | 10 |
| `nonstandard_analysis_and_hyperreal_training.py` | 9 |
| `knot_theoretic_programs_and_braid_based_attention.py` | 8 |
| `surreal_numbers_transseries_and_scaling.py` | 6 |
| `octonionic_quaternionic_signal_flow.py` | 6 |
| `tropical_geometry_and_idempotent_algebra.py` | 3 |
| `simplicial_complexes_and_higher_order_attention.py` | 3 |

Recommended pattern per demo: a `Console()` (or `from utils import console`),
results → one `Table` per result block, warnings/errors → styled
`console.print`. Keep the existing rich panels.

### 3. Infra diagnostics — plain, route through the shared console
Lower priority (some are legitimately library-internal), but for a uniform
look: `nanochat/engine.py` (8), `nanochat/tokenizer.py` (4),
`nanochat/common.py` (3), `utils.py` (4), `nanochat/synaptic_splitmerge.py`
(5), `nanochat/gpt.py` (2). Prefer styled `console.print` for warnings
(`[yellow]`/`[red]`) over bare `print`, or a `logging` channel for truly
internal diagnostics. `nanochat/configurator.py`'s `print0` is DDP-rank-aware
and acceptable as-is.

## Out of scope (plain `print` is fine)

Tests (`tests/test_demos.py`, `nanochat/test_tropical.py`), debug scripts
(`debug_gauge.py`), and the experimental `nanochat/train_jax.py` (14) — these
are developer-facing; styling them is not worth the churn.

## Recommendation (follow-up, not this bead)

1. **One shared console + a 3-line convention.** Reuse `utils.py:20`'s
   `console`; standardize: warnings `[yellow]`, errors `[red] bold`, results in
   `rich.Table`, long ops in `rich.progress.Progress`. Document it once.
2. **Do (1) for `dataset.py` first** (highest UX), then sweep the demos
   (results→tables, warnings→styled) — these are independent, parallelizable
   edits (good for several subagents per CLAUDE.md), one demo per commit.
3. Leave infra diagnostics for a later pass; convert only the user-visible ones.

These are deferred to a styling-sweep bead (filed); no code is changed here.
</content>
