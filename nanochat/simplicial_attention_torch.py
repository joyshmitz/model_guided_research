"""
Simplicial Attention Module (PyTorch)
Implements Higher-Order Attention via multi-hop diffusion, mimicking random walks on the simplicial complex 1-skeleton.
"""

import torch
import torch.nn as nn

from nanochat.model_utils import AttentionCore


class SimplicialCausalSelfAttention(AttentionCore):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)

        # Mixing weights for 1-hop (Edge) and 2-hop (Triangle/Path) attention
        self.mix_1 = nn.Parameter(torch.tensor(1.0))
        self.mix_2 = nn.Parameter(torch.tensor(0.5))

    def score(self, q, k):
        # Standard Attention Weights
        return (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))

    def aggregate(self, weights, v, *, q, k, kv_cache, pos0):
        # 1-hop Aggregation (Edges)
        y1 = weights @ v

        # 2-hop Aggregation (Simplicial/Paths)
        #
        # Cache-backed 2-hop: y2 = A @ y1_all, where y1_all stores the 1-hop outputs
        # for every past token (and is updated for the current chunk starting at the
        # pre-insert cache position pos0).
        if kv_cache is None:
            y2 = weights @ y1  # A @ (A @ v)
        else:
            if pos0 is None:
                raise RuntimeError("Expected pos0 to be set when kv_cache is provided")
            kv_cache.ensure_simplicial_y1_cache(
                num_heads=self.n_head,
                head_dim=self.head_dim,
                dtype=y1.dtype,
                device=y1.device,
            )
            y1_all = kv_cache.insert_simplicial_y1(self.layer_idx, pos0, y1)  # (B, H, Tk, D)
            y2 = weights @ y1_all

        return self.mix_1 * y1 + self.mix_2 * y2
