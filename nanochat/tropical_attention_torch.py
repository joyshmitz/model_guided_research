"""
Tropical Attention Module (PyTorch)
Implements attention using Max-Plus algebra for similarity.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from nanochat.model_utils import apply_rotary_emb, causal_attn_mask, norm, repeat_kv_heads


def _tropical_center(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Gauge-fix by subtracting the max along `dim` so the max becomes 0."""
    return x - x.amax(dim=dim, keepdim=True)


def tropical_inner(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Max-plus inner product: out[..., i, j] = max_d (q[..., i, d] + k[..., j, d]).

    This is the score stage of tropical attention, exposed as a pure function.
    It is the torch twin of the JAX demo's tropical matrix product
    (tropical_geometry_and_idempotent_algebra.tmm): tropical_inner(Q, K)
    equals tmm(Q, K^T) entrywise, exactly (max and + are exact float ops).
    Kept separable so the cross-framework parity suite
    (tests/test_cross_framework_parity.py, bead model_guided_research-5ki.6)
    can compare the two implementations directly.
    """
    return torch.max(q.unsqueeze(-2) + k.unsqueeze(-3), dim=-1).values


def tropical_maxplus_layer(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, beta: float | None = None
) -> torch.Tensor:
    """One max-plus affine layer: out_j = max_d (W_jd + x_d) (+ b_j).

    `weight` has shape (out_features, in_features); the broadcast intermediate
    is (..., out_features, in_features) - fine at research scale, a Triton
    tile kernel is the production path (see bead 7b0.7).

    With finite `beta`, max is replaced by the Maslov-smoothed
    (1/beta) * logsumexp(beta * .) - the SAME (+)_beta semiring family as
    dequantization annealing (bead 8gk.1); beta=None is the exact tropical
    endpoint. The LSE-max sandwich guarantees
    max <= lse_beta <= max + log(in_features)/beta (thm-lse-max-sandwich).
    """
    z = x.unsqueeze(-2) + weight  # (..., out, in)
    if beta is None:
        out = z.amax(dim=-1)
    else:
        out = torch.logsumexp(beta * z, dim=-1) / beta
    if bias is not None:
        out = out + bias
    return out


class TropicalMLP(nn.Module):
    """Max-plus FFN: the semiring design axis extended past attention.

    Bead: model_guided_research-8gk.8 (theory: markdown_documentation/
    tropical_geometry_and_idempotent_algebra.md, section "The tropical FFN").

    MODES (config.ffn_type):
      "tropical"          - PURE two-stage max-plus stack. 1-LIPSCHITZ in
          sup-norm by construction (max of unit-slope affines is
          nonexpansive; composition stays nonexpansive) - this is the layer
          that closes the certified chain's MLP hole (bead 8gk.7). The two
          stages COLLAPSE mathematically to one tropical-affine map of
          Barvinok rank <= d_ff (thm-maxplus-ffn-collapse): hidden width is a
          tropical RANK budget, not depth expressivity.
      "tropical-rational" - difference of two pure stacks (param-matched at
          d_ff/2 hidden each): reaches all piecewise-linear maps but is
          2-LIPSCHITZ per layer (difference of two 1-Lipschitz maps); the
          certificate chain must compose the declared constant 2.

    FINITE-BETA SMOOTHING (config.ffn_beta): replaces max by the (+)_beta
    semiring addition, enabling NETWORK-WIDE dequantization annealing with
    the attention path (8gk.1). beta=None is the exact tropical endpoint.

    EVT-AWARE INIT (lab.1 owns the full derivation; the rule used here):
    max over m iid terms of scale s concentrates at ~s*sqrt(2 ln m) (Gumbel
    location), NOT at 0 - so an additive max-plus stage drifts its output
    upward at init. The correction is baked into BIAS init (constant shift:
    Lipschitz constants untouched, no runtime renormalization), and applies
    to STAGE 1 ONLY: its input is the unit-RMS residual stream (norm(x)
    precedes the FFN in Block), so the asymptotic location -sqrt(2 ln d) is
    the right offset there. Stage 2's input is the POST-MAX distribution -
    centered and concentrated (Gumbel fluctuations ~1/sqrt(2 ln d), not unit
    scale) - so its drift is second-order and its bias inits to ZERO; using
    the unit-scale correction there overshoots by ~sqrt(2 ln d_ff) (caught
    by the init-centering test). The exact finite-n constants belong to
    lab.1's width-scaling table. Weights init N(0, 0.02^2) per repo
    convention.

    MARGINS: with config.tropical_record_margins (the existing flag, reused
    by design - one switch for all tropical diagnostics), registers
    ffn_gamma_min / ffn_gamma_mean buffers with the runner-up margin of the
    OUTPUT stage maxes (the certificate-bearing quantity, top1 - top2).
    Hard-max mode only (margins of a smoothed max are not route margins).
    """

    # buffer annotations (registered conditionally in __init__): these give
    # mypy the Tensor type that nn.Module.__getattr__'s union would otherwise hide
    ffn_gamma_min: torch.Tensor
    ffn_gamma_mean: torch.Tensor

    def __init__(self, config) -> None:
        super().__init__()
        d = int(config.n_embd)
        d_ff = 4 * d
        self.mode = str(getattr(config, "ffn_type", "tropical"))
        beta = getattr(config, "ffn_beta", None)
        self.beta: float | None = None if beta is None else float(beta)
        if self.beta is not None and not (self.beta > 0):
            raise ValueError(f"ffn_beta must be None or > 0, got {self.beta}")
        self.record_margins = bool(getattr(config, "tropical_record_margins", False))

        def w(out_f: int, in_f: int) -> nn.Parameter:
            return nn.Parameter(torch.randn(out_f, in_f) * 0.02)

        def evt_bias(out_f: int, in_f: int) -> nn.Parameter:
            # Gumbel location correction: center the max over in_f terms
            return nn.Parameter(torch.full((out_f,), -math.sqrt(2.0 * math.log(max(in_f, 1)))))

        def zero_bias(out_f: int) -> nn.Parameter:
            # stage-2 bias: post-max inputs are concentrated, drift is
            # second-order - start at zero (see EVT-AWARE INIT above)
            return nn.Parameter(torch.zeros(out_f))

        if self.mode == "tropical":
            self.w1, self.b1 = w(d_ff, d), evt_bias(d_ff, d)
            self.w2, self.b2 = w(d, d_ff), zero_bias(d)
        elif self.mode == "tropical-rational":
            h = d_ff // 2
            self.w1a, self.b1a = w(h, d), evt_bias(h, d)
            self.w2a, self.b2a = w(d, h), zero_bias(d)
            self.w1b, self.b1b = w(h, d), evt_bias(h, d)
            self.w2b, self.b2b = w(d, h), zero_bias(d)
        else:
            raise ValueError(f"TropicalMLP got unknown ffn_type {self.mode!r} (tropical | tropical-rational)")

        if self.record_margins:
            self.register_buffer("ffn_gamma_min", torch.zeros(()), persistent=False)
            self.register_buffer("ffn_gamma_mean", torch.zeros(()), persistent=False)

    def _stage_pair(self, x: torch.Tensor, w1, b1, w2, b2, *, record: bool) -> torch.Tensor:
        h = tropical_maxplus_layer(x, w1, b1, beta=self.beta)
        if record and self.beta is None:
            z = h.unsqueeze(-2) + w2  # (..., out, in): margins of the output stage
            top2 = torch.topk(z, k=2, dim=-1).values
            gamma = top2[..., 0] - top2[..., 1]
            self.ffn_gamma_min.copy_(gamma.detach().min())
            self.ffn_gamma_mean.copy_(gamma.detach().mean())
            out = top2[..., 0]
            return out + b2 if b2 is not None else out
        return tropical_maxplus_layer(h, w2, b2, beta=self.beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        record = self.record_margins and not torch.jit.is_scripting()
        if self.mode == "tropical":
            return self._stage_pair(x, self.w1, self.b1, self.w2, self.b2, record=record)
        pos = self._stage_pair(x, self.w1a, self.b1a, self.w2a, self.b2a, record=record)
        neg = self._stage_pair(x, self.w1b, self.b1b, self.w2b, self.b2b, record=False)
        return pos - neg

    @torch.no_grad()
    def collapsed_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The single tropical-affine map the PURE stack collapses to:
        M[j, d] = max_h (W2[j, h] + b1[h] + W1[h, d]), bias b2 (exact algebra;
        floating-point regrouping costs ulps - thm-maxplus-ffn-collapse).
        Raises for the rational mode (the difference does not collapse)."""
        if self.mode != "tropical":
            raise ValueError("collapsed_weight() is defined for the pure max-plus mode only")
        m = (self.w2.unsqueeze(-1) + self.b1.unsqueeze(0).unsqueeze(-1) + self.w1.unsqueeze(0)).amax(dim=1)
        return m, self.b2.clone()


def tropical_max_plus_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    gauge_fix: bool,
    score_center: bool,
    return_margins: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Pure max-plus (tropical) attention with optional gauge-fixing and margin certificates.

    Shapes:
      q: (B, H, Tq, D)
      k: (B, H, Tk, D)
      v: (B, H, Tk, D)

    Returns:
      y: (B, H, Tq, D)
      gamma: optional per-token, per-head runner-up margin (B, H, Tq) (>=0), or None if disabled.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("tropical_max_plus_attention expects q/k/v of shape (B, H, T, D)")
    if k.shape != v.shape:
        raise ValueError(
            f"tropical_max_plus_attention expects k and v to have the same shape, got k={k.shape}, v={v.shape}"
        )
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[3] != k.shape[3]:
        raise ValueError(f"tropical_max_plus_attention head shapes mismatch: q={q.shape}, k={k.shape}, v={v.shape}")

    B, H, Tq, D = q.shape
    Tk = k.size(2)

    if gauge_fix:
        q = _tropical_center(q, dim=-1)
        k = _tropical_center(k, dim=-1)
        v = _tropical_center(v, dim=-1)

    # Similarity/logits: score(q,k) = max_d (q_d + k_d)  (max-plus dot product)
    attn_scores = tropical_inner(q, k)  # (B, H, Tq, Tk)

    mask = causal_attn_mask(Tq, Tk, device=q.device)
    attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

    # Score-centering is a pure gauge change (per-query constant shift): it preserves argmax structure.
    if score_center:
        attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True)

    # Value aggregation: y_d = max_k (score(q,k) + v_{k,d})  (tropical matmul)
    logits = attn_scores.unsqueeze(-1) + v.unsqueeze(2)  # (B, H, Tq, Tk, D)
    gamma: torch.Tensor | None
    if return_margins:
        if Tk >= 2:
            top2 = torch.topk(logits, k=2, dim=3).values  # (B, H, Tq, 2, D)
            y = top2[..., 0, :]
            margin_d = top2[..., 0, :] - top2[..., 1, :]  # (B, H, Tq, D)
            gamma = margin_d.amin(dim=-1)  # (B, H, Tq)
        else:
            y = torch.max(logits, dim=3).values
            gamma = torch.full((B, H, Tq), float("inf"), device=y.device, dtype=y.dtype)
    else:
        y = torch.max(logits, dim=3).values
        gamma = None

    if gauge_fix:
        y = _tropical_center(y, dim=-1)
    return y, gamma


class TropicalCausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.tropical_gauge_fix = bool(getattr(config, "tropical_gauge_fix", True))
        self.tropical_score_center = bool(getattr(config, "tropical_score_center", True))
        self.tropical_record_margins = bool(getattr(config, "tropical_record_margins", False))
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.register_buffer(
            "tropical_gamma_head_mean",
            torch.full((self.n_head,), float("nan"), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "tropical_gamma_head_min",
            torch.full((self.n_head,), float("nan"), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "tropical_gamma_min",
            torch.tensor(float("nan"), dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        if self.n_kv_head != self.n_head:
            k, v = repeat_kv_heads(k, v, n_head=self.n_head)

        y, gamma = tropical_max_plus_attention(
            q,
            k,
            v,
            gauge_fix=self.tropical_gauge_fix,
            score_center=self.tropical_score_center,
            return_margins=self.tropical_record_margins,
        )
        if gamma is not None:
            gamma_f = gamma.to(dtype=torch.float32)
            with torch.no_grad():
                finite = torch.isfinite(gamma_f)
                summed = gamma_f.masked_fill(~finite, 0.0).sum(dim=(0, 2))
                count = finite.sum(dim=(0, 2)).to(dtype=summed.dtype)
                mean = summed / count.clamp_min(1.0)
                mean = mean.masked_fill(count == 0, float("nan"))
                self.tropical_gamma_head_mean.copy_(mean)
                self.tropical_gamma_head_min.copy_(gamma_f.amin(dim=(0, 2)))
                self.tropical_gamma_min.copy_(gamma_f.amin())

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        if self.tropical_gauge_fix:
            y = _tropical_center(y, dim=-1)
        return y
