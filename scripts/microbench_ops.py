"""Per-op microbenchmark for nanochat attention and FFN ops (bead 2cg).

Isolates a SINGLE attention op and a SINGLE FFN op (one transformer block's
sub-modules, not the whole model / training loop) and times forward and
forward+backward across mechanisms, shapes, and dtypes. This is the foundation
the Triton hotspot work (bead c6h) plugs into: it is where a fused max-plus
kernel's before/after numbers live (artifacts/microbench/).

Why per-op (not per-model): docs/flops_validation.md (bead bks) showed tropical
max-plus is 3-6x less hardware-efficient than matmul at the *model* level. This
harness localizes that to the attention op vs the FFN op so a kernel target is
unambiguous.

Usage:
    uv run python scripts/microbench_ops.py                 # default sweep, CPU
    uv run python scripts/microbench_ops.py --device cuda --dtype bf16
    uv run python scripts/microbench_ops.py --out artifacts/microbench/ops.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from nanochat.gpt import GPT, GPTConfig
from nanochat.model_utils import norm

console = Console()
MAIN = Path(__file__).resolve().parents[1]

# (label, attention_type, ffn_type) — mechanisms with a normal .attn slot so we
# can time the attention op in isolation. standard = the matmul baseline,
# tropical = the measured bandwidth-bound max-plus target.
MECHS = [
    ("standard", "standard", "standard"),
    ("tropical", "tropical", "tropical"),
]


def _build_block(attention_type: str, ffn_type: str, n_embd: int, n_head: int, n_kv_head: int, seq: int):
    cfg = GPTConfig(
        sequence_len=seq,
        vocab_size=50304,
        n_layer=1,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
        attention_type=attention_type,
        ffn_type=ffn_type,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        model = GPT(cfg)
    block = model.transformer.h[0]
    cos_sin = (model.cos[:, :seq], model.sin[:, :seq])
    return block, cos_sin


def _time(fn, *, warmup: int, iters: int, device: str) -> float:
    sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] * 1000.0  # median ms


def bench_op(
    kind: str, label: str, attn_t: str, ffn_t: str, *, B, T, n_embd, n_head, n_kv_head, dtype, device, warmup, iters
):
    torch.manual_seed(0)
    block, cos_sin = _build_block(attn_t, ffn_t, n_embd, n_head, n_kv_head, T)
    block = block.to(device=device, dtype=dtype)
    cos_sin = tuple(c.to(device=device) for c in cos_sin)  # rotary stays fp32-ish; module handles cast
    x = torch.randn(B, T, n_embd, device=device, dtype=dtype, requires_grad=True)
    op = (lambda: block.attn(norm(x), cos_sin, None)) if kind == "attn" else (lambda: block.mlp(norm(x)))

    def fwd():
        with torch.no_grad():
            op()

    def fwdbwd():
        if x.grad is not None:
            x.grad = None
        block.zero_grad(set_to_none=True)
        op().pow(2).mean().backward()

    ms_fwd = _time(fwd, warmup=warmup, iters=iters, device=device)
    ms_fb = _time(fwdbwd, warmup=warmup, iters=iters, device=device)
    head_dim = n_embd // n_head
    # analytical peak intermediate for the attention score path: standard keeps a
    # (B,H,T,T) score matrix; tropical max-plus materializes (B,H,T,T,head_dim)
    # before the reduce (tropical_attention_torch.py:21) - the bandwidth cost.
    score_elems = B * n_head * T * T
    interm = score_elems * (head_dim if (kind == "attn" and attn_t == "tropical") else 1)
    return {
        "op": kind,
        "mech": label,
        "attn_type": attn_t,
        "ffn_type": ffn_t,
        "B": B,
        "T": T,
        "n_embd": n_embd,
        "n_head": n_head,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "ms_fwd": round(ms_fwd, 4),
        "ms_fwd_bwd": round(ms_fb, 4),
        "peak_intermediate_elems": interm,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--out", default=str(MAIN / "artifacts" / "microbench" / "ops_cpu.json"))
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-kv-head", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seqs", default="256,512", help="comma-separated T values to sweep")
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    seqs = [int(s) for s in args.seqs.split(",") if s.strip()]
    if args.device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]--device cuda requested but CUDA unavailable; falling back to cpu[/]")
        args.device = "cpu"

    rows = []
    for T in seqs:
        for kind in ("attn", "ffn"):
            for label, attn_t, ffn_t in MECHS:
                rows.append(
                    bench_op(
                        kind,
                        label,
                        attn_t,
                        ffn_t,
                        B=args.batch_size,
                        T=T,
                        n_embd=args.n_embd,
                        n_head=args.n_head,
                        n_kv_head=args.n_kv_head,
                        dtype=dtype,
                        device=args.device,
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                )

    table = Table(
        title=f"per-op microbench — {args.device}/{args.dtype} (B={args.batch_size}, D={args.n_embd}, H={args.n_head})",
        border_style="cyan",
    )
    for col in ("op", "mech", "T", "ms_fwd", "ms_fwd+bwd", "peak_interm_elems", "vs_standard"):
        table.add_column(col, justify="right" if col not in ("op", "mech") else "left")
    # relative slowdown vs the standard arm of the same (op,T)
    base = {(r["op"], r["T"]): r["ms_fwd_bwd"] for r in rows if r["mech"] == "standard"}
    for r in rows:
        b = base.get((r["op"], r["T"]))
        rel = f"{r['ms_fwd_bwd'] / b:.2f}x" if b else "-"
        style = "yellow" if (b and r["ms_fwd_bwd"] / b > 1.5) else ""
        table.add_row(
            r["op"],
            r["mech"],
            str(r["T"]),
            f"{r['ms_fwd']:.2f}",
            f"{r['ms_fwd_bwd']:.2f}",
            f"{r['peak_intermediate_elems']:,}",
            f"[{style}]{rel}[/{style}]" if style else rel,
        )
    console.print(table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "device": args.device,
            "dtype": args.dtype,
            "B": args.batch_size,
            "n_embd": args.n_embd,
            "n_head": args.n_head,
            "n_kv_head": args.n_kv_head,
            "seqs": seqs,
            "warmup": args.warmup,
            "iters": args.iters,
        },
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2))
    console.print(f"[green]wrote[/] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
