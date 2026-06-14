# Campaign 8h0e — E2 braid–Dyck decisive verdict (the off-floor rung)

**Bead:** model_guided_research-8h0e (discovered-from 83r6) · **Hypothesis:**
`hyp-braid-dyck-depth-extrapolation` · **Verdict:** **REFUTED** (ci-v5,
2026-06-12), superseding the E1 SUPPORTED · **Provenance:** frozen worktree
`043db2c`, all 16 train runs + 16 evals clean.

## Result

| arm | n (checkpoints) | dyck held-out EM | answer prior |
|---|---|---|---|
| standard | 8 | **0.908 ± 0.027** | 0.625 |
| braid | 8 | **0.663 ± 0.096** | 0.625 |

Engine (ci-v5): **effect −0.245, CI95 [−0.326, −0.164], REFUTED-UNDERPOWERED**
(power 29% for the registered +0.05, n≈32 for 80%). The registered claim was
braid ≥ standard **+0.05**; the measured effect is **−0.245** with a CI that
excludes the threshold by more than 3× the threshold's own size and excludes
zero entirely. At the rung where the standard baseline actually learns Dyck,
braid is decisively *worse*.

## Why this supersedes the E1 SUPPORTED (and what the pair means)

`verdict_history` now carries the full arc, append-only:

- 2026-06-11 **SUPPORTED** (ci-v3, budget 1e14): effect +0.144 — but against a
  **floored** baseline. At 1e14 no standard seed clears the 0.625 prior; the
  result proves *braid learns Dyck where standard cannot get off the floor*.
- 2026-06-12 **REFUTED** (ci-v5, budget 3e14): effect −0.245 — at the
  off-floor rung the 83r6 probe found (standard EM ~0.90). The budget-cohort
  rule promotes the larger-budget cohort automatically; this is the ledger's
  **first supersession with a status flip**.

The two are consistent and together tell a sharper story than either alone:
**braid's Dyck advantage is a low-budget / sample-efficiency effect, not an
asymptotic one.** Braid reaches the stack-structure inductive bias at budgets
where pairwise-similarity standard attention is still stuck at the
constant-answer floor; once standard has budget enough to learn the task,
it learns it *better*. Three wave-2 braid seeds (s34–36) even land at 0.569,
**below the 0.625 answer prior** — braid underperforms a constant-answer
baseline at this rung on those seeds, which is why the braid arm's variance is
large.

## On the UNDERPOWERED qualifier (read this before citing the verdict)

> **Update (ci-v6, bead 4b82 landed):** the follow-on flagged below was
> implemented. The ledger entry here stays stamped ci-v5 (append-only history);
> but under the current engine a re-adjudication of this case produces
> `refuted, refutation_margin=3.6×, no underpowered` — the UNDERPOWERED
> qualifier now fires on SUPPORTED arms only, and a refutation records how
> decisively its CI excludes the threshold instead. The reasoning below is why.

The verdict was stamped **REFUTED-UNDERPOWERED** (under ci-v5). This was the
power gate's *conservatism for an opposite-sign effect*, not a weakness of the
refutation:

- The gate computes power to detect the **registered** effect size (+0.05) at
  the observed variance: 29%. That answers "could this test have *confirmed* a
  small +0.05 braid advantage?" — and the honest answer is "not reliably,"
  because braid's variance is large.
- But the result is not a failure-to-confirm; it is a strong effect of the
  **opposite sign**. The adequacy criterion that matters for a *refutation* is
  "does the CI exclude the registered threshold?" — and CI95 [−0.326, −0.164]
  excludes +0.05 overwhelmingly. By that criterion the test is fully adequate.
- The power instrument, as designed (bead hij.4), measures power against the
  registered effect regardless of the observed sign, so it flags this
  decisively-negative result UNDERPOWERED. A follow-on methodology bead is
  filed to consider sign-aware power adequacy for refutations (a ci-v6
  candidate, to be reviewed independently — **not** changed to make this
  verdict look cleaner).

## Protocol conformance (registered-report discipline)

- **Preregistered** in the bead description at `043db2c`, before any E2
  evidence existed: arms, rung (d128/L4 @ 3e14 from 83r6), seeds, and the
  adaptive rule.
- **Adaptive rule, one extension only:** wave-1 (seeds 30–33) read 20% power
  at n=4; the rule authorized training seeds 34–37 once, then "adjudicate
  exactly once." Done. The n=8 power reading (29%) did **not** trigger a
  second extension — the rule named seeds 34–37 as the single extension, not
  an iterate-to-80% loop. Re-extending would have been optional-stopping for
  zero scientific gain (no number of seeds yields power to detect a +0.05
  effect of the wrong sign).
- **One ledger append**, `-H` only (never `--all`): the rmatrix drift movers
  stay untouched for the o85g audit. Ledger write performed on a clean,
  in-sync tree under the committed ci-v5 engine.
- **Sizing probes excluded:** the 83r6 probe runs that selected this rung live
  in `artifacts/probes/sizing/` (dzor quarantine), out of the evidence pool.

## Artifacts

- Train: `artifacts/campaigns/e2-dyck/dyck-{braid,standard}-s{30..37}/` (16
  runs, frozen `043db2c`).
- Eval: `artifacts/evals/tasks/e2-dyck-{braid,standard}-s{30..37}/` (16, clean
  provenance, eval seeds 0,1,2).
- Verdict: `hypotheses/registry.yaml` verdict_history (ci-v5 entry);
  per-run report under `artifacts/adjudications/2026-06-12/`.
