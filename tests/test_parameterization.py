"""Tests for the NSA width-scaling parameterization (bead lab.1): the
extreme-value primitives, the per-mechanism scaling table, and the
coordinate-check harness. The EVT claims are validated independently of any
network (the Gumbel ingredient); the coordinate check is validated on the
CLT control (standard must be flat in width)."""

import math

import pytest

from nanochat import parameterization as P


def test_exact_expected_max_matches_monte_carlo():
    """E[max of n N(0,1)] via quadrature matches a Monte-Carlo estimate within
    sampling noise across the width ladder -- the finite-n location the table
    carries."""
    import torch

    torch.manual_seed(0)
    for n in (16, 64, 256, 1024):
        mc = float(torch.randn(60000, n).max(dim=1).values.mean())
        exact = P.exact_expected_max(n)
        assert abs(exact - mc) / mc < 0.015, f"n={n}: exact {exact:.4f} vs MC {mc:.4f}"


def test_asymptotic_location_is_off_at_small_n():
    """The 2nd-order asymptote captures the LAW but not the small-n CONSTANT:
    it must be visibly off at n=16 (the finite-n lesson) and converge toward
    the exact value as n grows."""
    err16 = abs(P.gumbel_asymptotic_location(16) - P.exact_expected_max(16)) / P.exact_expected_max(16)
    err4096 = abs(P.gumbel_asymptotic_location(4096) - P.exact_expected_max(4096)) / P.exact_expected_max(4096)
    assert err16 > 0.05, f"asymptote should be >5% off at n=16, got {err16:.3f}"
    assert err4096 < err16, "asymptote must converge to exact as n grows"


def test_gumbel_scaling_law_slope():
    """E[max] grows ~sqrt(2 ln N): on a log scale vs ln N the exact location
    tracks sqrt(2 ln N) -- the SCALING LAW that makes max-plus an EVT class."""
    import numpy as np

    ns = [16, 64, 256, 1024, 4096]
    exact = np.array([P.exact_expected_max(n) for n in ns])
    asymp = np.array([math.sqrt(2 * math.log(n)) for n in ns])
    # ratio exact/asymptote approaches 1 monotonically from below
    ratios = exact / asymp
    assert all(0.7 < r < 1.0 for r in ratios)
    assert ratios[-1] > ratios[0], "exact/asymptote ratio must rise toward 1"


def test_scaling_table_covers_core_mechanisms():
    for mech in ("standard", "tropical", "ultrametric", "quaternion", "octonion", "reversible"):
        rule = P.scaling_rule(mech)
        assert rule.mechanism == mech
        assert rule.concentration_class
        assert rule.notes
    # tropical is the EVT showcase; quaternion/octonion split the LR exponent
    assert "EVT" in P.scaling_rule("tropical").concentration_class
    assert P.scaling_rule("quaternion").lr_exponent == -1.0
    # unclassified falls back to CLT conservatively, never KeyErrors
    assert "CLT" in P.scaling_rule("does-not-exist").concentration_class


def test_coordinate_check_standard_is_flat():
    """The CLT control: standard attention's activation RMS is flat in width
    (|log-log slope| small). If this drifts, the harness is broken, not the
    theory -- so it gates everything else."""
    import warnings

    warnings.filterwarnings("ignore")
    res = P.coordinate_check("standard", [64, 128, 256, 512], seed=0)
    assert abs(res["loglog_slope"]) < 0.05, f"standard not flat: slope {res['loglog_slope']:.4f}"
    assert all(v > 0 for v in res["activation_rms"].values())


@pytest.mark.parametrize("mech", ["tropical", "quaternion", "reversible"])
def test_coordinate_check_runs_for_mechanisms(mech):
    """The harness is generic over mechanisms: each produces finite per-width
    scales and a finite slope (the per-mechanism interpretation lives in the
    theory note; here we assert the apparatus runs and is well-formed)."""
    import warnings

    warnings.filterwarnings("ignore")
    res = P.coordinate_check(mech, [64, 128, 256], seed=0)
    assert res["concentration_class"]
    assert len(res["activation_rms"]) == 3
    assert all(math.isfinite(v) and v > 0 for v in res["activation_rms"].values())
    assert math.isfinite(res["loglog_slope"])
