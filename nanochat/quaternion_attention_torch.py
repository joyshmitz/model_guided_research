"""
Quaternion Attention Module (PyTorch)
Implements "Rotor-Gate" style attention where features are treated as quaternions.
"""

import torch
import torch.nn.functional as F

from nanochat.model_utils import AttentionCore


def qmul(a, b):
    """
    Multiply quaternion tensors a and b.
    Shape: (..., 4)
    """
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)

    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw

    return torch.stack((ow, ox, oy, oz), dim=-1)


def qconj(q):
    """
    Conjugate of quaternion q.
    Shape: (..., 4)
    """
    w, x, y, z = q.unbind(-1)
    return torch.stack((w, -x, -y, -z), dim=-1)


def qnorm(q):
    """
    Norm of quaternion q.
    """
    return torch.norm(q, dim=-1, keepdim=True)


def qnormalize(q):
    return F.normalize(q, p=2, dim=-1)


class QuaternionCausalSelfAttention(AttentionCore):
    """Rotor-gate attention: standard scalar scores, quaternion value mixing.

    Scores stay the real dot product (the R^4 dot product IS the scalar part
    of q1 * conj(q2), so flat vectors give the same scalar affinity without
    materializing the (Tq, Tk, N, 4) rotor tensor). The value update uses the
    FULL quaternion product via relative rotors R_ij = Q_i * conj(K_j):

        y_i = sum_j probs_ij * (Q_i * conj(K_j) * V_j)

    which quaternion associativity factors into three cheap steps (see
    aggregate()). Q/K are per-channel normalized before any quaternion
    product so the rotor multiplications are norm-preserving.
    """

    def __init__(self, config, layer_idx):
        # Standard linear projections; the output is interpreted as
        # head_dim/4 quaternions per head.
        super().__init__(config, layer_idx)
        if self.n_embd % 4 != 0:
            raise ValueError("n_embd must be divisible by 4 for Quaternion attention")
        if self.head_dim % 4 != 0:
            raise ValueError("head_dim must be divisible by 4 for Quaternion attention")

    def score(self, q, k):
        return (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))

    def aggregate(self, weights, v, *, q, k, kv_cache, pos0):
        B = q.size(0)
        Tk = v.size(2)

        # Interpret as quaternions: (..., D) -> (..., D/4, 4), and normalize
        # per-channel so rotor multiplications are norm-preserving.
        q_q = qnormalize(q.view(B, self.n_head, -1, self.head_dim // 4, 4))
        k_q = qnormalize(k.view(B, self.n_head, -1, self.head_dim // 4, 4))
        v_q = v.view(B, self.n_head, -1, self.head_dim // 4, 4)

        # y_i = sum_j probs_ij * (q_i * k_j_conj * v_j) is associative, so:
        # 1. T_j = k_j_conj * v_j      (elementwise quaternion mul)
        t_q = qmul(qconj(k_q), v_q)  # (B, H, Tk, N, 4)
        t_flat = t_q.view(B, self.n_head, Tk, self.head_dim)
        # 2. A_i = sum_j probs_ij T_j  (standard attention aggregation)
        agg = weights @ t_flat  # (B, H, Tq, D)
        # 3. y_i = q_i * A_i           (rotate by the query rotor)
        agg_q = agg.view(B, self.n_head, -1, self.head_dim // 4, 4)
        y_q = qmul(q_q, agg_q)

        return y_q.view(B, self.n_head, -1, self.head_dim)
