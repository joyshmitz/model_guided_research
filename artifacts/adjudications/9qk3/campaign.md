# 9qk3 — dequantization annealing: retention is FREE (policy ci-v3)

Date: 2026-06-11 · 6 runs (3 seeds × {fixed beta=1, linear:1:32 anneal}), E1 rung, arith corpus, --val-interval 100. First adjudication through the rgyl variant selectors (semiring_beta_spec).

**hyp-maslov-anneal-loss-retention SUPPORTED**: val-CE ratio 0.9997 [0.9991, 1.0000], n=3/3 — against a registered budget of ≤1.053. Annealing β: 1→32 over the run loses NOTHING vs the smooth-start model; the 8gk.1 theory note's continuation-method claim holds at its first test.

Caveat recorded honestly: certificate route_coverage at β=32 is ~0.2% (margins mostly under the conservative (log D + log m)/β threshold) — retention is free but certified-route coverage at this β needs larger margins; the coverage-vs-snap comparison and closed-loop schedules (9jzb) are the follow-on instruments.

Provenance: pair-1 from clean main, seeds 1–2 from the frozen worktree after the dirty-window driver kill — all six untainted.
