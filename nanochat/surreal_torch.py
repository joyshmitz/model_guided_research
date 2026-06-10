"""
Surreal Regularization / Probe (PyTorch)
Implements a dynamic scaling probe based on "Surreal Numbers and Transseries".
This acts as a "Meta-Optimizer" hook that logs dominance metrics and uses "Surreal Layers"
where weights are parameterized as `w = exp(s) * v` (Scale * Direction) to separate
magnitude (exponent) from direction (coefficient), mimicking Transseries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.model_utils import AttentionCore, sdpa_causal_attend


class SurrealProbe:
    def __init__(self, model, enabled=False):
        self.model = model
        self.enabled = enabled

    def step(self, loss, inputs, targets):
        """
        Compute dominance metrics:
        T_D: Data scaling benefit (simulated via split?)
        T_H: Depth scaling benefit (simulated via skipping layers)
        T_W: Width scaling benefit (simulated via masking channels)

        Returns: extra_loss, metrics
        """
        if not self.enabled:
            return 0.0, {}

        # Placeholder for dominance check
        # In a full implementation, this would run the forward pass with:
        # 1. Half depth (skip layers)
        # 2. Half width (mask channels)
        # 3. Log the ratios E_half / E_full

        return 0.0, {"surreal_balance": 1.0}


class SurrealLayer(nn.Module):
    """
    A Linear layer with "Surreal" weight parameterization.
    Weights are represented as `w = s * v` where s is a learnable scale (exponent)
    and v is the direction.
    This mimics "transseries" where we separate magnitude (scale) from direction.

    w = exp(s) * normalize(v)
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Direction v
        self.weight_v = nn.Parameter(torch.randn(out_features, in_features))
        # Scale s (log-magnitude)
        self.weight_s = nn.Parameter(torch.zeros(out_features, 1))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        # w = exp(s) * normalize(v)
        w = torch.exp(self.weight_s) * F.normalize(self.weight_v, dim=1)
        return F.linear(input, w, self.bias)


class SurrealCausalSelfAttention(AttentionCore):
    # GQA is handled inside SDPA via enable_gqa; no materialized repeat.
    gqa_via_repeat = False

    def __init__(self, config, layer_idx):
        # Surreal Linear Layers (w = exp(s) * normalize(v)) for all projections;
        # attribute names and state-dict keys match the canonical scaffold.
        super().__init__(config, layer_idx, linear_cls=SurrealLayer)

    def attend(self, q, k, v, *, kv_cache, pos0):
        enable_gqa = self.n_head != self.n_kv_head
        return sdpa_causal_attend(q, k, v, kv_cache=kv_cache, enable_gqa=enable_gqa)
