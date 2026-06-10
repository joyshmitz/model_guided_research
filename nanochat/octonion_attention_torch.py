"""
Octonion Attention Module (PyTorch)
Implements Octonion-based attention using the Cayley-Dickson construction over Quaternions.
Octonions are 8D hypercomplex numbers. Multiplication is non-associative.
"""

import torch
import torch.nn.functional as F

from nanochat.model_utils import AttentionCore
from nanochat.quaternion_attention_torch import qconj, qmul

# Octonion Multiplication via Cayley-Dickson
# O1 = (a, b), O2 = (c, d) where a,b,c,d are Quaternions.
# O1 * O2 = (a*c - d_conj*b, d*a + b*c_conj)
# Note: Order matters! Octonions are non-associative.


def omul(o1, o2):
    """
    Multiply octonion tensors o1 and o2.
    Shape: (..., 8)
    Splits into two quaternions (..., 4).
    """
    a, b = torch.split(o1, 4, dim=-1)
    c, d = torch.split(o2, 4, dim=-1)

    # a*c
    ac = qmul(a, c)
    # d_conj * b
    db = qmul(qconj(d), b)
    # d*a
    da = qmul(d, a)
    # b*c_conj
    bc = qmul(b, qconj(c))

    first = ac - db
    second = da + bc

    return torch.cat([first, second], dim=-1)


def oconj(o):
    """
    Conjugate of octonion o = (a, b) is (a_conj, -b).
    """
    a, b = torch.split(o, 4, dim=-1)
    return torch.cat([qconj(a), -b], dim=-1)


def onorm(o):
    return torch.norm(o, dim=-1, keepdim=True)


def onormalize(o):
    return F.normalize(o, p=2, dim=-1)


class OctonionCausalSelfAttention(AttentionCore):
    """Octonionic signal flow: standard scalar scores, non-associative mixing.

    The value update is Y_i = sum_j probs_ij * ((Q_i * conj(K_j)) * V_j) with
    the EXPLICIT parenthesization (rotor first, then value): octonions are
    non-associative, so unlike the quaternion rotor-gate the query CANNOT be
    factored out of the sum - the pairwise products are intrinsic to the
    mechanism and cost O(T^2 * D). Q/K are per-channel normalized so the
    octonion multiplications act as norm-preserving "rotors".
    """

    def __init__(self, config, layer_idx):
        # Standard linear projections; the output is interpreted as
        # head_dim/8 octonions per head.
        super().__init__(config, layer_idx)
        if self.n_embd % 8 != 0:
            raise ValueError("n_embd must be divisible by 8 for Octonion attention")
        if self.head_dim % 8 != 0:
            raise ValueError("head_dim must be divisible by 8 for Octonion attention")

    def score(self, q, k):
        return (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))

    def aggregate(self, weights, v, *, q, k, kv_cache, pos0):
        B = q.size(0)
        Tq = q.size(2)

        # Interpret as octonions: (..., D) -> (..., D/8, 8).
        q_o = onormalize(q.view(B, self.n_head, -1, self.head_dim // 8, 8))
        k_o = onormalize(k.view(B, self.n_head, -1, self.head_dim // 8, 8))
        v_o = v.view(B, self.n_head, -1, self.head_dim // 8, 8)

        # Per-query Python loop: broadcasting all of (B, H, Tq, Tk, N, 8) at
        # once would blow memory, so each query's rotor row is materialized
        # in turn. Vectorizing this loop is bead model_guided_research-7b0.6.
        k_conj = oconj(k_o)
        y_list = []
        for i in range(Tq):
            q_i = q_o[:, :, i : i + 1, :, :]  # (B, H, 1, N, 8)
            r_i = omul(q_i, k_conj)  # rotors for query i: (B, H, Tk, N, 8)
            term = omul(r_i, v_o)  # (Q*conj(K))*V, parenthesized: (B, H, Tk, N, 8)
            p_i = weights[:, :, i, :].unsqueeze(-1).unsqueeze(-1)  # (B, H, Tk, 1, 1)
            y_i = (term * p_i).sum(dim=2).unsqueeze(2)  # (B, H, 1, N, 8)
            y_list.append(y_i)

        y_o = torch.cat(y_list, dim=2)  # (B, H, Tq, N, 8)
        return y_o.view(B, self.n_head, Tq, self.head_dim)
