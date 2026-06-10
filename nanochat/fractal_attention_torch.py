"""
Fractal Memory / Attention Module (PyTorch)
Implements attention over a "Fractal" memory structure (IFS).

Mathematical core (from JAX reference):
  - Values live as fixed points x*_w of composed contractions F_w.
  - Keys are depth-k paths w = (i_1, ..., i_k).
  - c_w = sum_{j=1..k} A^{k-j} t_{j, i_j}.
  - Read v_hat = x*_w = (c_w + u_w) / (1 - s^k).

PyTorch Implementation:
  - We simulate the "Path Matching" aspect via Hierarchical Soft-Routing.
  - Q defines a target path. K defines a storage path.
  - Attention weight ~ Probability(Path(Q) == Path(K)).
"""

import torch.nn as nn
import torch.nn.functional as F

from nanochat.model_utils import AttentionCore


class FractalCausalSelfAttention(AttentionCore):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)

        # IFS Router
        # "A learned router (k independent m-way classifiers) maps query q->w"
        self.m = 4  # Branching factor
        self.depth = 4  # Depth of IFS
        self.router = nn.Linear(self.head_dim, self.depth * self.m, bias=False)

    def score(self, q, k):
        # Fractal Addressing / Similarity
        # We compute the "Soft Path" for Q and K.
        # Path is a sequence of categorical distributions over m branches at each depth d.
        # q_route: (B, H, Tq, Depth, m)
        B = q.size(0)
        q_route = self.router(q).view(B, self.n_head, -1, self.depth, self.m)
        k_route = self.router(k).view(B, self.n_head, -1, self.depth, self.m)

        # Softmax over m to get branch probabilities
        q_prob = F.softmax(q_route, dim=-1)
        k_prob = F.softmax(k_route, dim=-1)

        # Similarity = Probability that Q and K took the SAME path.
        # P(overlap) = prod_{d=1..Depth} (sum_{i=1..m} q_{d,i} * k_{d,i})
        # The exact per-depth product is replaced by the dot product of the
        # flattened probability vectors - a proxy for "total path overlap
        # mass" - scaled by 1/sqrt(Depth) (the structure dimension, not the
        # per-feature dimension).
        q_flat = q_prob.view(B, self.n_head, -1, self.depth * self.m)
        k_flat = k_prob.view(B, self.n_head, -1, self.depth * self.m)

        return (q_flat @ k_flat.transpose(-2, -1)) * (1.0 / (self.depth**0.5))
