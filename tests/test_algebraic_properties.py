"""
Property-based tests (Hypothesis) for the algebraic kernels (br: model_guided_research-5ki.2).

These tests attack the LAWS the mechanism kernels are built on, over a generated
input space — the complement of `mgr certify` (cli.py), which checks invariants on
instantiated modules. Boundary: if a check needs a trained/instantiated module it
belongs in certify; if it is a pure mathematical identity it belongs here.

Profiles: the "ci" profile (default) is tuned to keep the whole module under ~60s;
set HYPOTHESIS_PROFILE=thorough for deep local runs. On failure, Hypothesis prints
the falsifying example and the reproduction seed (`@reproduce_failure`), which is
the logging contract for this suite: every red is reproducible from the output.

Exactness policy:
- EXACT laws use exact equality: tropical max/plus identities (max is exactly
  associative/commutative on floats; float addition is monotone, which makes
  distributivity exact), integer LCP ultrametricity.
- Approximate-in-fp laws use magnitude-scaled tolerances: quaternion/octonion
  product identities accumulate rounding proportional to the operand magnitudes.

This file is the designated home for the theory-program identity tests
(see the bead's notes: Maslov LSE semiring, valuation dictionary, Gromov=LCP,
YBE residuals, symplecticity, Gumbel asymptotics, CNF ordinal arithmetic land
here as their beads are implemented — maintenance contract).
"""

import math
import os

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nanochat.octonion_attention_torch import oconj, omul
from nanochat.quaternion_attention_torch import qconj, qmul

settings.register_profile(
    "ci",
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("thorough", max_examples=300, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Bounded finite floats: large enough to stress scaling, small enough that
# fp64 product identities stay verifiable with scaled tolerances.
finite = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False, width=64)
# Strictly finite values for EXACT tropical laws (no -inf except where tested explicitly).
trop_val = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, width=64)


def _vec(n: int):
    return st.lists(finite, min_size=n, max_size=n).map(lambda v: torch.tensor(v, dtype=torch.float64))


quat = _vec(4)
octo = _vec(8)


def _scaled_atol(*tensors: torch.Tensor, base: float = 1e-12) -> float:
    """Magnitude-scaled absolute tolerance for fp64 product identities."""
    scale = 1.0
    for t in tensors:
        scale *= float(t.abs().max()) + 1.0
    return base * scale


# ---------------------------------------------------------------------------
# Tropical semiring (max, +): EXACT laws on floats
# ---------------------------------------------------------------------------


@given(a=trop_val, b=trop_val, c=trop_val)
def test_tropical_max_associative_commutative_exact(a, b, c):
    assert max(max(a, b), c) == max(a, max(b, c))
    assert max(a, b) == max(b, a)


@given(a=trop_val)
def test_tropical_idempotence_and_identity_exact(a):
    assert max(a, a) == a
    assert max(a, float("-inf")) == a  # -inf is the additive identity
    assert a + float("-inf") == float("-inf")  # ... and annihilates under tropical multiplication


@given(a=trop_val, b=trop_val, c=trop_val)
def test_tropical_distributivity_exact(a, b, c):
    # c + max(a, b) == max(c + a, c + b): exact because float addition is monotone,
    # so the max is attained at the same argument on both sides.
    assert c + max(a, b) == max(c + a, c + b)


@given(
    q=st.lists(trop_val, min_size=4, max_size=4),
    k=st.lists(trop_val, min_size=4, max_size=4),
)
def test_tropical_score_shift_covariance(q, k):
    # score(q + s, k) = score(q, k) + s for a scalar broadcast shift s:
    # the algebraic fact behind tropical gauge-fixing / score-centering.
    # Exact in REAL arithmetic; in floats the two sides associate additions
    # differently ((q_d + s) + k_d vs (q_d + k_d) + s), so allow a few ulp.
    # (The thorough Hypothesis profile found the falsifying 1-ulp example for
    # the exact-equality version of this test — kept as a worked reminder that
    # float addition is not associative even when every term is benign.)
    qt = torch.tensor(q, dtype=torch.float64)
    kt = torch.tensor(k, dtype=torch.float64)
    s = 7.25
    score = torch.max(qt + kt)
    score_shifted = torch.max((qt + s) + kt)
    scale = max(abs(float(score)), abs(s), 1.0)
    assert abs(float(score_shifted) - float(score + s)) <= 4.0 * torch.finfo(torch.float64).eps * scale


# ---------------------------------------------------------------------------
# Quaternions: associativity, norm multiplicativity, conjugation
# ---------------------------------------------------------------------------


@given(a=quat, b=quat, c=quat)
def test_quaternion_associativity(a, b, c):
    lhs = qmul(qmul(a, b), c)
    rhs = qmul(a, qmul(b, c))
    assert float((lhs - rhs).abs().max()) <= _scaled_atol(a, b, c)


@given(a=quat, b=quat)
def test_quaternion_norm_multiplicative(a, b):
    lhs = float(qmul(a, b).norm())
    rhs = float(a.norm() * b.norm())
    assert abs(lhs - rhs) <= _scaled_atol(a, b)


@given(a=quat, b=quat)
def test_quaternion_conj_antihomomorphism(a, b):
    lhs = qconj(qmul(a, b))
    rhs = qmul(qconj(b), qconj(a))
    assert float((lhs - rhs).abs().max()) <= _scaled_atol(a, b)


@given(a=quat)
def test_quaternion_conj_involution_and_norm(a):
    assert float((qconj(qconj(a)) - a).abs().max()) == 0.0  # conjugation is an exact sign flip
    prod = qmul(a, qconj(a))
    # a * conj(a) = |a|^2 (scalar part), zero imaginary part
    assert abs(float(prod[0]) - float(a.norm()) ** 2) <= _scaled_atol(a, a)
    assert float(prod[1:].abs().max()) <= _scaled_atol(a, a)


# ---------------------------------------------------------------------------
# Octonions: alternativity, Moufang, norm multiplicativity, NON-associativity
# ---------------------------------------------------------------------------


@given(a=octo, b=octo)
def test_octonion_norm_multiplicative(a, b):
    lhs = float(omul(a, b).norm())
    rhs = float(a.norm() * b.norm())
    assert abs(lhs - rhs) <= _scaled_atol(a, b)


@given(a=octo, b=octo)
def test_octonion_alternativity(a, b):
    left = omul(a, omul(a, b)) - omul(omul(a, a), b)
    right = omul(omul(b, a), a) - omul(b, omul(a, a))
    tol = _scaled_atol(a, a, b)
    assert float(left.abs().max()) <= tol
    assert float(right.abs().max()) <= tol


@given(x=octo, y=octo, z=octo)
def test_octonion_moufang_identity(x, y, z):
    # Middle Moufang identity: (z x)(y z) = (z (x y)) z
    lhs = omul(omul(z, x), omul(y, z))
    rhs = omul(omul(z, omul(x, y)), z)
    assert float((lhs - rhs).abs().max()) <= _scaled_atol(x, y, z, base=1e-11)


@given(a=octo)
def test_octonion_conj_gives_norm_squared(a):
    prod = omul(a, oconj(a))
    assert abs(float(prod[0]) - float(a.norm()) ** 2) <= _scaled_atol(a, a)
    assert float(prod[1:].abs().max()) <= _scaled_atol(a, a)


def test_octonion_nonassociativity_exists():
    # EXISTENCE witness on canonical basis elements (deterministic, not @given):
    # (e1 e2) e4 = -e1 (e2 e4) in the Cayley-Dickson basis — associator is nonzero.
    e1 = torch.zeros(8, dtype=torch.float64)
    e2 = torch.zeros(8, dtype=torch.float64)
    e4 = torch.zeros(8, dtype=torch.float64)
    e1[1] = 1.0
    e2[2] = 1.0
    e4[4] = 1.0
    associator = omul(omul(e1, e2), e4) - omul(e1, omul(e2, e4))
    assert float(associator.abs().max()) > 0.5, "octonion multiplication must NOT be associative"


# ---------------------------------------------------------------------------
# Ultrametric LCP: strong triangle inequality (EXACT, integer)
# ---------------------------------------------------------------------------


def _lcp(x: list[int], y: list[int]) -> int:
    n = 0
    for a, b in zip(x, y, strict=True):
        if a != b:
            break
        n += 1
    return n


digit_seq = st.lists(st.integers(min_value=0, max_value=4), min_size=6, max_size=6)


@given(x=digit_seq, y=digit_seq, z=digit_seq)
def test_lcp_strong_triangle_inequality_exact(x, y, z):
    # lcp(x, z) >= min(lcp(x, y), lcp(y, z)): the ultrametric law, exact on integers.
    assert _lcp(x, z) >= min(_lcp(x, y), _lcp(y, z))


@given(x=digit_seq, y=digit_seq)
def test_lcp_symmetry_and_self_exact(x, y):
    assert _lcp(x, y) == _lcp(y, x)
    assert _lcp(x, x) == len(x)


# ---------------------------------------------------------------------------
# Givens rotations (gauge transport): orthogonality, additivity
# ---------------------------------------------------------------------------

angle = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False, width=64)


@given(theta=angle, x0=finite, x1=finite)
def test_givens_rotation_norm_and_det(theta, x0, x1):
    c, s = math.cos(theta), math.sin(theta)
    # det = cos^2 + sin^2 = 1
    assert abs((c * c + s * s) - 1.0) <= 1e-12
    # norm preservation of the pair rotation
    y0, y1 = c * x0 - s * x1, s * x0 + c * x1
    n_in = math.hypot(x0, x1)
    n_out = math.hypot(y0, y1)
    assert abs(n_in - n_out) <= 1e-9 * (1.0 + n_in)


@given(t1=angle, t2=angle, x0=finite, x1=finite)
def test_givens_rotation_additivity(t1, t2, x0, x1):
    # R(t2) R(t1) = R(t1 + t2): the law that justifies cumsum-as-transport in the gauge block.
    def rot(t, a, b):
        c, s = math.cos(t), math.sin(t)
        return c * a - s * b, s * a + c * b

    seq = rot(t2, *rot(t1, x0, x1))
    direct = rot(t1 + t2, x0, x1)
    scale = 1.0 + math.hypot(x0, x1)
    assert abs(seq[0] - direct[0]) <= 1e-9 * scale
    assert abs(seq[1] - direct[1]) <= 1e-9 * scale


# ---------------------------------------------------------------------------
# RoPE: per-pair norm preservation
# ---------------------------------------------------------------------------


@given(
    angles=st.lists(angle, min_size=4, max_size=4),
    vals=st.lists(finite, min_size=8, max_size=8),
)
def test_rope_pairwise_norm_preservation(angles, vals):
    from nanochat.model_utils import apply_rotary_emb

    d = 4
    x = torch.tensor(vals, dtype=torch.float64).view(1, 1, 1, 2 * d)
    th = torch.tensor(angles, dtype=torch.float64).view(1, 1, 1, d)
    y = apply_rotary_emb(x, th.cos(), th.sin())
    pn_in = torch.sqrt(x[..., :d] ** 2 + x[..., d:] ** 2)
    pn_out = torch.sqrt(y[..., :d] ** 2 + y[..., d:] ** 2)
    assert float((pn_in - pn_out).abs().max()) <= 1e-9 * (1.0 + float(pn_in.max()))


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main(sys.argv[1:] or [__file__]))
