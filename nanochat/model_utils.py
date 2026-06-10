import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def causal_attn_mask(Tq: int, Tk: int, *, device: torch.device) -> torch.Tensor:
    """
    Build a boolean attention mask of shape (Tq, Tk) where True means "keep".

    Matches the KV-cache semantics used in `nanochat.gpt.CausalSelfAttention`:
    - Training / prefill (Tk == Tq): standard causal (lower-triangular) mask
    - Decode (Tq == 1): allow attending to all cached keys (no masking)
    - Chunked decode (Tk > Tq > 1): allow full prefix + causal within the chunk
    """
    if Tq <= 0 or Tk <= 0:
        raise ValueError("Tq and Tk must be positive")
    if Tq == 1:
        return torch.ones((Tq, Tk), dtype=torch.bool, device=device)
    if Tk == Tq:
        return torch.tril(torch.ones((Tq, Tk), dtype=torch.bool, device=device))
    if Tk < Tq:
        raise ValueError(f"Expected Tk >= Tq for causal attention, got Tk={Tk}, Tq={Tq}")
    prefix_len = Tk - Tq
    mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=device)
    if prefix_len > 0:
        mask[:, :prefix_len] = True
    mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=device))
    return mask


def repeat_kv_heads(k: torch.Tensor, v: torch.Tensor, *, n_head: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat KV heads for Group-Query Attention (GQA) to match `n_head` query heads."""
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("repeat_kv_heads expects k/v of shape (B, H, T, D)")
    if k.shape != v.shape:
        raise ValueError(f"repeat_kv_heads expects k and v to have the same shape, got k={k.shape}, v={v.shape}")
    n_kv_head = k.size(1)
    if n_kv_head == n_head:
        return k, v
    if n_kv_head <= 0 or n_head <= 0:
        raise ValueError("n_head and n_kv_head must be positive")
    if n_head % n_kv_head != 0:
        raise ValueError(f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})")
    repeat = n_head // n_kv_head
    return k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)


def apply_rotary_emb(x, cos, sin):
    if x.ndim != 4:
        raise ValueError("apply_rotary_emb expects tensor of shape (B, T, H, D)")
    if x.shape[3] % 2 != 0:
        raise ValueError("apply_rotary_emb requires an even head dimension D (pairs of channels)")
    d = x.shape[3] // 2
    if cos.shape[-1] != d or sin.shape[-1] != d:
        raise ValueError(
            f"apply_rotary_emb expects cos/sin last dim == D/2 ({d}), got cos={cos.shape}, sin={sin.shape}"
        )
    x1, x2 = x[..., :d], x[..., d:]  # split up last time into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)  # re-assemble
    out = out.to(x.dtype)  # ensure input/output dtypes match
    return out


def sdpa_causal_attend(q, k, v, *, kv_cache, enable_gqa: bool) -> torch.Tensor:
    """Fused scaled-dot-product attention across the three causal cases.

    - Training / prefill (no cache, or Tq == Tk): SDPA with is_causal=True.
    - Single-token decode (Tq == 1): the query attends to every cached key.
    - Chunked decode (Tk > Tq > 1): full prefix + causal-within-chunk mask.

    k/v may carry n_kv_head heads; SDPA broadcasts them to the query heads
    when enable_gqa is True (no materialized repeat).
    """
    Tq = q.size(2)
    Tk = k.size(2)
    if kv_cache is None or Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
    if Tq == 1:
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
    attn_mask = causal_attn_mask(Tq, Tk, device=q.device)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)


class AttentionCore(nn.Module):
    """Shared causal self-attention scaffolding (bead model_guided_research-7b0.1).

    Every attention mechanism historically duplicated the same scaffold
    around its math: QKV projection, RoPE + QK-norm, head transposition,
    KV-cache insertion, GQA head repetition, and output reassembly +
    projection - and a fix in one copy did not propagate to the other ten.
    This base class owns that scaffold once; a mechanism contributes only
    its distinguishing operations.

    Override points, coarse to fine:

    - ``attend(q, k, v, *, kv_cache, pos0) -> y``: everything between the
      prepared (q, k, v) and the per-head output (B, n_head, Tq, head_dim).
      Override when the mechanism is fused (SDPA / flex), stateful
      (ultrametric trie), or does not factor through a (Tq, Tk) affinity
      matrix. The default composes the finer hooks below:
      score -> causal mask -> softmax -> aggregate.
    - ``score(q, k) -> scores``: raw affinities (B, n_head, Tq, Tk), BEFORE
      causal masking and normalization.
    - ``aggregate(weights, v, *, q, k, kv_cache, pos0) -> y``: combine
      normalized weights with values (default: ``weights @ v``). q/k are the
      post-RoPE, post-norm tensors, for mechanisms whose aggregation
      re-reads them (quaternion/octonion rotors); pos0 serves
      cache-position-stateful aggregation (simplicial y1 history).
    - ``finalize(y) -> y``: post-projection hook on the (B, T, n_embd)
      output (default: identity; tropical re-centers here).

    Tensor contract at the attend() boundary:
    - q is (B, n_head, Tq, head_dim), RoPE-rotated and RMS-normed.
    - k, v are (B, n_head, Tk, head_dim) when ``gqa_via_repeat`` is True
      (the default; KV heads are materialized via repeat_kv_heads after the
      cache insert). Mechanisms that handle GQA themselves (e.g. SDPA's
      enable_gqa) set ``gqa_via_repeat = False`` and receive
      (B, n_kv_head, Tk, head_dim) instead.
    - pos0 is the KV-cache write position BEFORE this forward's insert
      (None when kv_cache is None).

    ``linear_cls`` swaps the projection layer class while keeping the
    canonical attribute names c_q/c_k/c_v/c_proj (surreal passes its
    exp(s)*normalize(v) SurrealLayer); state-dict keys are unchanged.

    Boundary (A1 decision, bead 7b0.1): reversible and gauge are BLOCK-level
    specializations - reversible halves channels and wraps a whole attention
    module as its coupling F-function; gauge replaces the entire block
    including the MLP - so neither uses this scaffold. The scaffold's
    responsibility ends at "a causal self-attention layer with the standard
    residual-stream interface": forward(x, cos_sin, kv_cache) -> (B, T, n_embd).
    """

    gqa_via_repeat: bool = True

    def __init__(self, config, layer_idx, *, linear_cls: type[nn.Module] = nn.Linear):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not (self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0):
            raise ValueError("n_kv_head must divide n_head and be <= n_head")
        self.head_dim = self.n_embd // self.n_head
        self.c_q = linear_cls(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = linear_cls(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = linear_cls(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = linear_cls(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Rotary embeddings (relative positions), then QK norm
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        # make head be the batch dim: (B, T, H, D) -> (B, H, T, D)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # KV cache: capture the pre-insert write position, insert current
        # k/v, and get back the full cached view so far
        pos0 = None
        if kv_cache is not None:
            pos0 = kv_cache.get_pos()
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        # GQA: materialize repeated KV heads unless the mechanism opted out
        if self.gqa_via_repeat and self.n_kv_head != self.n_head:
            k, v = repeat_kv_heads(k, v, n_head=self.n_head)

        y = self.attend(q, k, v, kv_cache=kv_cache, pos0=pos0)

        # Re-assemble the heads side by side and project back to the residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return self.finalize(y)

    def score(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} must implement score(q, k) or override attend()")

    def aggregate(self, weights, v, *, q, k, kv_cache, pos0):
        return weights @ v

    def attend(self, q, k, v, *, kv_cache, pos0):
        scores = self.score(q, k)
        Tq = q.size(2)
        Tk = k.size(2)
        # Single-token decode (Tq == 1 with a cache) attends to the whole
        # prefix - no mask needed; every other case masks in place.
        if kv_cache is None or Tq > 1:
            mask = causal_attn_mask(Tq, Tk, device=q.device)
            scores.masked_fill_(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return self.aggregate(weights, v, q=q, k=k, kv_cache=kv_cache, pos0=pos0)

    def finalize(self, y: torch.Tensor) -> torch.Tensor:
        return y
