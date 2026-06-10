#!/usr/bin/env python3
"""
Model Guided Research CLI - Run experimental mathematical models for ML research
"""

import importlib
import json
import math
import platform
import shlex
import statistics
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

app = typer.Typer(
    name="model-guided-research",
    help="Run experimental mathematical models for machine learning research",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()

_ALLOWED_CMDS = {"git"}


def _run_command(cmd: str) -> str | None:
    try:
        parts = shlex.split(cmd)
        if not parts or parts[0] not in _ALLOWED_CMDS:
            return None
        result = subprocess.run(parts, shell=False, capture_output=True, text=True, timeout=5)  # nosec B603
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except Exception:
        return None


def _get_git_info() -> dict[str, Any]:
    commit = _run_command("git rev-parse --short HEAD") or "unknown"
    commit_full = _run_command("git rev-parse HEAD") or "unknown"
    branch = _run_command("git rev-parse --abbrev-ref HEAD") or "unknown"
    status = _run_command("git status --porcelain")
    dirty = bool(status) if status is not None else False
    return {"commit": commit, "commit_full": commit_full, "branch": branch, "dirty": dirty}


def _default_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _write_artifacts(run_dir: Path, *, summary: dict[str, Any], report_md: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "run.md").write_text(report_md, encoding="utf-8")


def _sparkline(values: list[float], *, width: int = 20) -> str:
    bars = "▁▂▃▄▅▆▇█"
    if not values or width <= 0:
        return ""
    if len(values) > width:
        # Take the tail: most useful for training loss curves.
        values = values[-width:]
    lo = min(values)
    hi = max(values)
    if not math.isfinite(lo) or not math.isfinite(hi):
        return ""
    if hi - lo < 1e-12:
        return bars[0] * len(values)
    idxs = [int((v - lo) / (hi - lo) * (len(bars) - 1)) for v in values]
    return "".join(bars[i] for i in idxs)


_T_CRIT_975: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _summary_stats(values: list[float]) -> dict[str, Any]:
    vals = [
        float(v) for v in values if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(float(v))
    ]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": None}
    mean = float(statistics.fmean(vals))
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci95": None}
    var = float(sum((x - mean) ** 2 for x in vals) / (n - 1))
    std = math.sqrt(var)
    t = _T_CRIT_975.get(n - 1, 1.96)
    ci = float(t * std / math.sqrt(n))
    return {"n": n, "mean": mean, "std": std, "ci95": ci}


def _aggregate_per_head(samples: list[tuple[int, list[float]]]) -> dict[str, Any] | None:
    valid = [
        (seed, vec)
        for seed, vec in samples
        if isinstance(vec, list) and vec and all(isinstance(x, int | float) and not isinstance(x, bool) for x in vec)
    ]
    if not valid:
        return None
    n_head = min(len(vec) for _, vec in valid)
    means: list[float] = []
    stds: list[float] = []
    ci95s: list[float | None] = []
    ns: list[int] = []
    for i in range(n_head):
        vals = [float(vec[i]) for _, vec in valid if i < len(vec) and math.isfinite(float(vec[i]))]
        stats = _summary_stats(vals)
        ns.append(int(stats["n"]))
        means.append(float(stats["mean"]) if stats["mean"] is not None else float("nan"))
        stds.append(float(stats["std"]) if stats["std"] is not None else float("nan"))
        ci95s.append(float(stats["ci95"]) if stats["ci95"] is not None else None)
    return {
        "n_head": n_head,
        "seeds": [int(seed) for seed, _ in valid],
        "samples": {str(seed): vec[:n_head] for seed, vec in valid},
        "n": ns,
        "mean": means,
        "std": stds,
        "ci95": ci95s,
    }


def _resolve_summary_path(path: Path, *, artifacts_dir: Path) -> Path:
    candidates: list[Path] = [path]
    if not path.exists():
        candidates.append(artifacts_dir / path)
    for cand in candidates:
        if cand.is_dir():
            summary_path = cand / "summary.json"
            if summary_path.is_file():
                return summary_path
        if cand.is_file():
            return cand
    raise typer.BadParameter(f"Could not find summary.json for {path} (tried: {', '.join(str(c) for c in candidates)})")


def _get_nested(obj: dict[str, Any], keys: tuple[str, ...]) -> Any:
    cur: Any = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _as_float(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    return None


def _extract_metric(summary: dict[str, Any], *, metric: str, variant: str | None) -> float | None:
    """Extract a canonical metric from heterogeneous summary.json shapes."""
    # Suite summaries: choose by attention_type.
    if isinstance(summary.get("runs"), list) and variant:
        for run in summary["runs"]:
            if isinstance(run, dict) and run.get("attention_type") == variant:
                if metric == "final_loss":
                    return _as_float(run.get("final_loss")) or _as_float(run.get("score"))
                if metric == "tokens_per_second":
                    return _as_float(run.get("tokens_per_second")) or _as_float(run.get("tokens_per_s"))
                if metric == "tflops_per_second_est":
                    return _as_float(run.get("tflops_per_second_est"))
                if metric == "peak_memory_allocated_gb":
                    return _as_float(run.get("peak_memory_allocated_gb"))

    results = summary.get("results")
    if not isinstance(results, dict):
        results = {}

    if metric == "final_loss":
        losses = results.get("losses")
        if isinstance(losses, list) and losses:
            vals = [_as_float(v) for v in losses]
            vals2 = [v for v in vals if v is not None]
            return vals2[-1] if vals2 else None
        for k in ("final_loss", "loss", "score"):
            v = _as_float(results.get(k))
            if v is not None:
                return v
        return _as_float(summary.get("final_loss")) or _as_float(summary.get("score"))

    if metric == "tokens_per_second":
        v = results.get("tokens_per_second")
        if isinstance(v, dict):
            return _as_float(v.get(variant)) if variant else None
        v2 = _as_float(v)
        if v2 is not None:
            return v2
        v = results.get("tokens_per_s")
        if isinstance(v, dict):
            return _as_float(v.get(variant)) if variant else None
        return _as_float(v) or _as_float(summary.get("tokens_per_second")) or _as_float(summary.get("tokens_per_s"))

    if metric == "tflops_per_second_est":
        v = results.get("tflops_per_second_est")
        if isinstance(v, dict):
            return _as_float(v.get(variant)) if variant else None
        return _as_float(v) or _as_float(summary.get("tflops_per_second_est"))

    if metric == "peak_memory_allocated_gb":
        v = results.get("peak_memory_allocated_gb")
        if isinstance(v, dict):
            return _as_float(v.get(variant)) if variant else None
        v2 = _as_float(v)
        if v2 is not None:
            return v2
        peak_mb = results.get("peak_mem_mb")
        if isinstance(peak_mb, dict):
            mb = _as_float(peak_mb.get(variant)) if variant else None
            return (mb / 1024.0) if mb is not None else None
        mb = _as_float(peak_mb)
        return (mb / 1024.0) if mb is not None else None

    raise ValueError(f"Unknown metric {metric!r}")


def _extract_loss_series(summary: dict[str, Any]) -> list[float]:
    results = summary.get("results")
    if not isinstance(results, dict):
        return []
    losses = results.get("losses")
    if not isinstance(losses, list):
        return []
    out: list[float] = []
    for v in losses:
        fv = _as_float(v)
        if fv is not None:
            out.append(fv)
    return out


def _summarize_provenance(summary: dict[str, Any]) -> dict[str, Any]:
    meta = summary.get("meta") if isinstance(summary.get("meta"), dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    git = summary.get("git") if isinstance(summary.get("git"), dict) else meta.get("git", {})
    if not isinstance(git, dict):
        git = {}
    cfg = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    if not isinstance(cfg, dict):
        cfg = {}
    return {
        "run_id": summary.get("run_id") or meta.get("run_id"),
        "kind": meta.get("kind") or summary.get("kind"),
        "device": meta.get("device") or summary.get("device"),
        "commit": git.get("commit") or git.get("commit_full"),
        "dirty": git.get("dirty"),
        "attention_type": cfg.get("attention_type"),
        "use_flex_attention": cfg.get("use_flex_attention"),
        "compile": _get_nested(summary, ("compile", "enabled")) or cfg.get("compile"),
        "command": meta.get("command") or summary.get("command"),
    }


# Map of available demos
DEMOS = {
    "ifs-fractal": {
        "module": "iterated_function_systems_and_fractal_memory",
        "description": "Iterated Function Systems and Fractal Memory structures",
        "func": "demo",
    },
    "knot-braid": {
        "module": "knot_theoretic_programs_and_braid_based_attention",
        "description": "Knot-theoretic programs and braid-based attention mechanisms",
        "func": "demo",
    },
    "matrix-gauge": {
        "module": "matrix_exponential_gauge_learning",
        "description": "Matrix exponential gauge learning with Lie groups",
        "func": "demo",
    },
    "nonstandard": {
        "module": "nonstandard_analysis_and_hyperreal_training",
        "description": "Nonstandard analysis and hyperreal training methods",
        "func": "demo",
    },
    "octonion": {
        "module": "octonionic_quaternionic_signal_flow",
        "description": "Octonionic and quaternionic signal flow processing",
        "func": "demo",
    },
    "ordinal": {
        "module": "ordinal_schedules_and_well_founded_optimization",
        "description": "Ordinal schedules and well-founded optimization",
        "func": "demo",
    },
    "reversible": {
        "module": "reversible_computation_and_measure_preserving_learning",
        "description": "Reversible computation and measure-preserving learning",
        "func": "demo",
    },
    "simplicial": {
        "module": "simplicial_complexes_and_higher_order_attention",
        "description": "Simplicial complexes and higher-order attention",
        "func": "demo",
    },
    "surreal": {
        "module": "surreal_numbers_transseries_and_scaling",
        "description": "Surreal numbers, transseries and scaling methods",
        "func": "demo",
    },
    "tropical": {
        "module": "tropical_geometry_and_idempotent_algebra",
        "description": "Tropical geometry and idempotent algebra",
        "func": "demo",
    },
    "ultrametric": {
        "module": "ultrametric_worlds_and_p_adic_computation",
        "description": "Ultrametric worlds and p-adic computation",
        "func": "demo",
    },
}


@app.command("list")
def list_demos():
    """List all available demos with descriptions"""
    table = Table(
        title="[bold cyan]Available Model Demos[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Demo Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Module", style="dim white")

    for name, info in DEMOS.items():
        table.add_row(name, info["description"], info["module"])

    console.print(table)
    console.print("\n[dim]Run a demo with:[/dim] [bold green]mgr run <demo-name>[/bold green]")
    console.print("[dim]Get info about a demo:[/dim] [bold green]mgr info <demo-name>[/bold green]")


@app.command()
def run(
    demo_name: Annotated[
        str,
        typer.Argument(
            help="Name of the demo to run",
            autocompletion=lambda: DEMOS.keys(),  # type: ignore[call-arg]
        ),
    ],
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to JSON config file (see config.example.json)")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show verbose output")] = False,
    verbose_level: Annotated[
        int | None,
        typer.Option("--verbose-level", min=0, max=3, help="Verbosity level: 0=silent, 1=normal, 2=detailed, 3=debug"),
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", "-s", help="Random seed for reproducibility")] = None,
    max_iterations: Annotated[
        int | None,
        typer.Option(
            "--max-iterations", min=1, help="Override ProjectConfig.max_iterations (for demos that respect it)"
        ),
    ] = None,
    no_rich: Annotated[bool, typer.Option("--no-rich", help="Disable rich formatting for plain text output")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Enable debug mode with numerical checking")] = False,
    ultra_packed: Annotated[
        bool,
        typer.Option(
            "--ultra-packed",
            help="Use packed bit-trie implementation in ultrametric demo (and set ULTRA_PACKED for tests)",
        ),
    ] = False,
    tropical_cert: Annotated[
        bool, typer.Option("--tropical-cert", help="Compute a tropical attention robustness margin certificate")
    ] = False,
    simplicial_hodge: Annotated[
        bool, typer.Option("--simplicial-hodge", help="Demonstrate Hodge-based readout coefficients on a tiny graph")
    ] = False,
    simplicial_signed: Annotated[
        bool, typer.Option("--simplicial-signed", help="Demonstrate signed (orientation-aware) diffusion vs unsigned")
    ] = False,
    rev_cayley: Annotated[
        bool, typer.Option("--rev-cayley", help="Demonstrate Cayley orthogonal property check (skew → orthogonal)")
    ] = False,
    rev_cayley_o1: Annotated[
        bool,
        typer.Option(
            "--rev-cayley-o1/--no-rev-cayley-o1", help="Use O(1)-memory custom gradient for Cayley step (default on)"
        ),
    ] = True,
    rev_cayley_iters: Annotated[
        int,
        typer.Option("--rev-cayley-iters", help="Cayley fixed-point iterations (trade compute for accuracy)", min=1),
    ] = 1,
    rev_symplectic: Annotated[
        bool, typer.Option("--rev-symplectic", help="Demonstrate symplectic Cayley property check (S^T J S ≈ J)")
    ] = False,
    rev_inv_iters: Annotated[
        int, typer.Option("--rev-inv-iters", help="Inverse fixed-point iteration count for Cayley inverse", min=1)
    ] = 1,
    rev_pareto: Annotated[
        bool, typer.Option("--rev-pareto", help="Run a small Cayley-iterations Pareto sweep (time vs memory)")
    ] = False,
    rev_symp_hybrid: Annotated[
        bool, typer.Option("--rev-symplectic-hybrid", help="Enable a symplectic leapfrog step inside coupling (hybrid)")
    ] = False,
    rev_givens: Annotated[
        bool, typer.Option("--rev-givens", help="Use strict Givens mixing (exact inverse; det=1)")
    ] = False,
    rev_generating: Annotated[
        bool, typer.Option("--rev-generating", help="Enable generating-function symplectic step (exact inverse)")
    ] = False,
    rev_gen_vjp: Annotated[
        bool, typer.Option("--rev-gen-vjp", help="Use custom VJP for generating step (O(1) grads; ignores ∂/∂(a,b,c))")
    ] = False,
    gauge_structured: Annotated[
        bool, typer.Option("--gauge-structured", help="Enable structured SO/SPD/Sp channel blocks in matrix-gauge demo")
    ] = False,
    gauge_bch_compact: Annotated[
        bool, typer.Option("--gauge-bch-compact", help="Print only compact BCH summary table (skip heatmap)")
    ] = False,
    gauge_alt_struct: Annotated[
        bool,
        typer.Option("--gauge-alt-struct", help="Alternate structured/unstructured on odd blocks in matrix-gauge demo"),
    ] = False,
    export_json: Annotated[
        Path | None, typer.Option("--export-json", help="Write a JSON artifact with any computed certificates/readouts")
    ] = None,
    artifacts_dir: Annotated[
        Path | None,
        typer.Option(
            "--artifacts-dir",
            help="Write run artifacts under this directory using the standard layout (see artifacts/README.md).",
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id", help="Run identifier (directory name) when writing artifacts. Defaults to YYYYMMDD_HHMMSS."
        ),
    ] = None,
):
    """Run a specific demo by name"""

    # Configure settings
    from config import ProjectConfig, set_config
    from utils import seed_everything

    # Load config from file if provided
    if config_file and config_file.exists():
        config = ProjectConfig.from_file(config_file)
        if verbose:
            console.print(f"[dim]Loaded config from {config_file}[/dim]")
    else:
        config = ProjectConfig()

    # Override with command-line arguments
    if verbose:
        config.verbose = True
    if verbose_level is not None:
        config.verbose_level = verbose_level
    if seed is not None:
        config.random_seed = seed
    if max_iterations is not None:
        config.max_iterations = max_iterations
    if no_rich:
        config.use_rich_output = False
    if debug:
        config.debug_mode = True
        config.check_numerics = True
        config.jax_debug_nans = True
        config.jax_debug_infs = True

    # Set the global config
    set_config(config)
    seed_everything(config.random_seed)

    # Optional environment knobs for tests/internals
    if ultra_packed:
        import os as _os

        _os.environ["ULTRA_PACKED"] = "1"
    if gauge_structured:
        import os as _os

        _os.environ["GAUGE_STRUCTURED"] = "1"
    if gauge_bch_compact:
        import os as _os

        _os.environ["GAUGE_BCH_COMPACT"] = "1"
    if gauge_alt_struct:
        import os as _os

        _os.environ["GAUGE_ALT_STRUCT"] = "1"
    if rev_givens:
        import os as _os

        _os.environ["REV_GIVENS"] = "1"
    if rev_generating:
        import os as _os

        _os.environ["REV_GENERATING"] = "1"
    if rev_gen_vjp:
        import os as _os

        _os.environ["REV_GEN_VJP"] = "1"

    if demo_name not in DEMOS:
        console.print(f"[bold red]Error:[/bold red] Demo '{demo_name}' not found")
        console.print("\nAvailable demos:")
        for name in DEMOS:
            console.print(f"  • {name}")
        raise typer.Exit(1)

    demo_info = DEMOS[demo_name]

    # Display what we're running
    panel = Panel(
        f"[bold cyan]{demo_info['description']}[/bold cyan]\n[dim]Module: {demo_info['module']}.py[/dim]",
        title=f"Running Demo: {demo_name}",
        box=box.ROUNDED,
    )
    console.print(panel)
    console.print()

    try:
        artifacts: dict = {"demo": demo_name, "certificates": {}}
        # Import the module dynamically
        if verbose:
            console.print(f"[dim]Importing module: {demo_info['module']}[/dim]")

        module = importlib.import_module(demo_info["module"])

        # Get the demo function
        func_name = demo_info["func"]
        if hasattr(module, func_name):
            demo_func = getattr(module, func_name)

            if verbose:
                console.print(f"[dim]Running function: {func_name}()[/dim]\n")

            # Pre-demo feature showcases
            if demo_name == "tropical" and tropical_cert:
                import numpy as _np

                from tropical_geometry_and_idempotent_algebra import TropicalAttention

                Q_np = _np.random.randn(32, 16)
                K_np = _np.random.randn(32, 16)
                V_np = _np.random.randn(32, 16)
                attn = TropicalAttention(16)
                _ = attn(Q_np, K_np, V_np)
                table = Table(title="Tropical Robustness Certificate", show_header=True, header_style="bold magenta")
                table.add_column("Min (best−second) margin", justify="center")
                margin = float(getattr(attn, "last_min_margin", 0.0))
                table.add_row(f"{margin:.4f}")
                console.print(table)
                artifacts["certificates"]["tropical_min_margin"] = margin
                # Toggle ASCII summary of K if matrix-gauge demo is run too
                # (No-op here; matrix-gauge prints uniformization K when demo runs.)

            if demo_name == "simplicial" and simplicial_hodge:
                import numpy as _np

                from simplicial_complexes_and_higher_order_attention import hodge_readout

                n = 8
                A = _np.zeros((n, n))
                for _ in range(12):
                    i, j = _np.random.randint(0, n, 2)
                    if i != j:
                        A[i, j] = A[j, i] = 1
                flow = _np.random.randn(n)
                coeff = hodge_readout(flow, A, k_small=3)
                t = Table(title="Hodge Readout Coefficients (k=3)", show_header=True, header_style="bold magenta")
                t.add_column("Mode", justify="center")
                t.add_column("Coeff", justify="right")
                for i, c in enumerate(coeff):
                    t.add_row(str(i), f"{float(c):.4f}")
                console.print(t)
                artifacts["certificates"]["simplicial_hodge_coeffs"] = [float(c) for c in coeff]

            if (demo_name == "reversible") and (
                rev_cayley or rev_symplectic or rev_pareto or rev_symp_hybrid or (rev_inv_iters != 1)
            ):
                import numpy as _np

                from matrix_exponential_gauge_learning import cayley_orthogonal_from_skew, symplectic_cayley

                # Cayley orthogonal check
                if rev_cayley:
                    try:
                        from reversible_computation_and_measure_preserving_learning import (
                            set_reversible_cayley,
                            set_reversible_cayley_iters,
                            set_reversible_cayley_o1,
                        )

                        set_reversible_cayley(True)
                        set_reversible_cayley_o1(bool(rev_cayley_o1))
                        set_reversible_cayley_iters(int(rev_cayley_iters))
                        import os as _os

                        _os.environ["REV_LAYER_CERT"] = "1"
                    except Exception:
                        pass
                    M = _np.random.randn(16, 16)
                    A = 0.1 * (M - M.T)  # skew
                    import jax.numpy as _jnp

                    Q = cayley_orthogonal_from_skew(_jnp.array(A))
                    eye_q = _jnp.eye(Q.shape[-1])
                    err = float(_jnp.linalg.norm(Q.T @ Q - eye_q))
                    table = Table(title="Cayley Orthogonality Check", show_header=True, header_style="bold magenta")
                    table.add_column("||Q^T Q − I||_F", justify="right")
                    table.add_row(f"{err:.2e}")
                    console.print(table)
                    artifacts["certificates"]["reversible_cayley_orth_err"] = err
                # Symplectic check
                if rev_symplectic:
                    n = 8
                    H = _np.random.randn(2 * n, 2 * n)
                    H = 0.1 * (H + H.T)
                    import jax.numpy as _jnp

                    S = symplectic_cayley(_jnp.array(H))
                    Z = _jnp.zeros((n, n))
                    eye_n = _jnp.eye(n)
                    J = _jnp.block([[Z, eye_n], [-eye_n, Z]])
                    err = float(_jnp.linalg.norm(S.T @ J @ S - J))
                    t2 = Table(title="Symplectic Cayley Check", show_header=True, header_style="bold magenta")
                    t2.add_column("||S^T J S − J||_F", justify="right")
                    t2.add_row(f"{err:.2e}")
                    console.print(t2)
                    artifacts["certificates"]["reversible_symplectic_err"] = err
                if rev_pareto:
                    import os as _os

                    _os.environ["REV_PARETO"] = "1"
                if rev_inv_iters and rev_inv_iters != 1:
                    try:
                        import os as _os

                        _os.environ["REV_INV_ITERS"] = str(int(rev_inv_iters))
                    except Exception:
                        pass
                if rev_symp_hybrid:
                    try:
                        from reversible_computation_and_measure_preserving_learning import set_reversible_symplectic

                        set_reversible_symplectic(True)
                    except Exception:
                        pass

            # Run the demo
            with console.status("[bold green]Running demo...[/bold green]"):
                demo_func()

            # Collect module-level diagnostics if present
            try:
                diag = getattr(module, "last_diagnostics", None)
                if diag is not None:
                    artifacts.setdefault("diagnostics", {})[demo_name] = diag
            except Exception:
                pass

        # Write artifacts if requested (legacy path)
        if export_json is not None:
            try:
                export_json.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            with export_json.open("w", encoding="utf-8") as f:
                json.dump(artifacts, f, indent=2)
            if verbose:
                console.print(f"[dim]Wrote JSON artifact to {export_json}[/dim]")

        # Write artifacts using the unified artifacts layout
        if artifacts_dir is not None:
            resolved_run_id = run_id or _default_run_id()
            run_dir = artifacts_dir / "certs" / "demos" / demo_name / resolved_run_id

            meta = {
                "demo": demo_name,
                "run_id": resolved_run_id,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "git": _get_git_info(),
                "python": {
                    "executable": sys.executable,
                    "version": platform.python_version(),
                },
                "argv": sys.argv,
                "seed": int(config.random_seed),
                "config_file": str(config_file) if config_file is not None else None,
                "flags": {
                    "max_iterations": int(config.max_iterations),
                    "ultra_packed": bool(ultra_packed),
                    "tropical_cert": bool(tropical_cert),
                    "simplicial_hodge": bool(simplicial_hodge),
                    "rev_givens": bool(rev_givens),
                    "rev_generating": bool(rev_generating),
                    "rev_gen_vjp": bool(rev_gen_vjp),
                    "rev_pareto": bool(rev_pareto),
                    "rev_inv_iters": rev_inv_iters,
                    "rev_symp_hybrid": bool(rev_symp_hybrid),
                    "gauge_structured": bool(gauge_structured),
                    "gauge_bch_compact": bool(gauge_bch_compact),
                    "gauge_alt_struct": bool(gauge_alt_struct),
                },
            }

            summary = {"meta": meta, "artifacts": artifacts}
            report_md = f"""# Demo certificate run: `{demo_name}`

- Run ID: `{resolved_run_id}`
- Generated: {meta["generated_at"]}
- Commit: {meta["git"]["commit_full"]}{" (dirty)" if meta["git"]["dirty"] else " (clean)"}

## Command

```bash
{shlex.join(sys.argv)}
```

## Certificates

This run writes `summary.json` (machine-readable) and `run.md` (human-readable) under:

`{run_dir}`
"""
            _write_artifacts(run_dir, summary=summary, report_md=report_md)
            console.print(f"[dim]Wrote artifacts → {run_dir}[/dim]")

    except ImportError as e:
        console.print(f"[bold red]Import Error:[/bold red] {e}")
        console.print("\n[dim]Make sure all dependencies are installed:[/dim]")
        console.print("[bold]uv sync --extra dev[/bold]")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
        raise typer.Exit(0) from None
    except Exception as e:
        console.print(f"[bold red]Error running demo:[/bold red] {e}")
        if verbose:
            import traceback

            console.print("[dim]Traceback:[/dim]")
            traceback.print_exc()
        raise typer.Exit(1) from e


@app.command()
def info(
    demo_name: str = typer.Argument(
        ...,
        help="Name of the demo to get info about",
        autocompletion=lambda: DEMOS.keys(),  # type: ignore[call-arg]
    ),
):
    """Show detailed information about a specific demo"""

    if demo_name not in DEMOS:
        console.print(f"[bold red]Error:[/bold red] Demo '{demo_name}' not found")
        console.print("\nAvailable demos:")
        for name in DEMOS:
            console.print(f"  • {name}")
        raise typer.Exit(1)

    demo_info = DEMOS[demo_name]
    module_file = Path(f"{demo_info['module']}.py")

    # Display demo information
    panel = Panel(
        f"[bold cyan]{demo_info['description']}[/bold cyan]\n\n"
        f"[bold]Module:[/bold] {demo_info['module']}.py\n"
        f"[bold]Function:[/bold] {demo_info['func']}()\n"
        f"[bold]File exists:[/bold] {'✓' if module_file.exists() else '✗'}",
        title=f"Demo: {demo_name}",
        box=box.ROUNDED,
    )
    console.print(panel)

    # Try to extract and display the module docstring
    if module_file.exists():
        try:
            with open(module_file, encoding="utf-8") as f:
                lines = f.readlines()

            # Find module docstring
            in_docstring = False
            docstring_lines = []
            for _i, line in enumerate(lines[:50]):  # Check first 50 lines
                if '"""' in line:
                    if not in_docstring:
                        in_docstring = True
                        # Check if it's a one-liner
                        if line.count('"""') == 2:
                            docstring_lines.append(line.strip().replace('"""', ""))
                            break
                    else:
                        in_docstring = False
                        break
                elif in_docstring:
                    docstring_lines.append(line.rstrip())

            if docstring_lines:
                console.print("\n[bold]Module Documentation:[/bold]")
                console.print(
                    Panel(
                        "\n".join(docstring_lines),
                        box=box.ROUNDED,
                        padding=(1, 2),
                    )
                )

        except Exception as e:
            if str(e):  # Only show error if it has a message
                console.print(f"[dim]Could not read module documentation: {e}[/dim]")

    console.print(f"\n[dim]Run this demo with:[/dim] [bold green]mgr run {demo_name}[/bold green]")


@app.command()
def config(
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output path for config file")] = None,
    show: Annotated[bool, typer.Option("--show", help="Show current configuration")] = False,
):
    """Generate example config file or show current configuration"""

    import json

    from config import get_config

    if show:
        # Show current configuration
        current = get_config()
        console.print("[bold cyan]Current Configuration:[/bold cyan]\n")

        config_dict = {}
        for field in current.__dataclass_fields__:
            value = getattr(current, field)
            if isinstance(value, Path):
                value = str(value)
            config_dict[field] = value

        console.print(json.dumps(config_dict, indent=2))
        return

    # Generate example config
    output_path = output or Path("config.json")

    if output_path.exists():
        if not typer.confirm(f"File {output_path} exists. Overwrite?"):
            raise typer.Exit(0)

    example_config = {
        "use_gpu": False,
        "jax_precision": "float32",
        "random_seed": 42,
        "jax_debug_nans": False,
        "jax_debug_infs": False,
        "jax_disable_jit": False,
        "verbose": True,
        "verbose_level": 1,
        "save_outputs": False,
        "output_dir": "outputs",
        "save_checkpoints": False,
        "checkpoint_dir": "checkpoints",
        "log_metrics": True,
        "log_interval": 100,
        "use_rich_output": True,
        "show_progress_bars": True,
        "debug_mode": False,
        "check_numerics": False,
        "profile_performance": False,
        "max_iterations": 1000,
        "convergence_threshold": 1e-6,
        "early_stopping_patience": 10,
        "default_learning_rate": 0.001,
        "default_batch_size": 32,
        "gradient_clip_norm": None,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(example_config, f, indent=2)

    console.print(f"[green]✓ Example config written to {output_path}[/green]")
    console.print(f"\n[dim]Use it with:[/dim] [bold]mgr run <demo> --config {output_path}[/bold]")


@app.command()
def run_all(
    delay: Annotated[int, typer.Option("--delay", "-d", help="Delay in seconds between demos")] = 2,
    seed: Annotated[int | None, typer.Option("--seed", "-s", help="Random seed for reproducibility")] = None,
    skip_errors: Annotated[
        bool, typer.Option("--skip-errors/--stop-on-error", help="Continue running demos even if one fails")
    ] = True,
):
    """Run all available demos in sequence"""

    from config import ProjectConfig, set_config
    from utils import seed_everything

    config = ProjectConfig()
    if seed is not None:
        config.random_seed = seed
    set_config(config)
    seed_everything(config.random_seed)

    console.print("[bold cyan]Running all demos...[/bold cyan]\n")

    success_count = 0
    error_count = 0

    for i, (name, info) in enumerate(DEMOS.items(), 1):
        console.rule(f"[bold]Demo {i}/{len(DEMOS)}: {name}[/bold]")

        try:
            # Import and run the demo
            module = importlib.import_module(info["module"])
            func_name = info["func"]

            if hasattr(module, func_name):
                console.print(f"[cyan]{info['description']}[/cyan]\n")

                demo_func = getattr(module, func_name)
                demo_func()

                success_count += 1
                console.print(f"\n[green]✓ Demo '{name}' completed successfully[/green]")
            else:
                raise AttributeError(f"Function '{func_name}' not found")

        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped by user[/yellow]")
            break
        except Exception as e:
            error_count += 1
            console.print(f"\n[red]✗ Demo '{name}' failed: {e}[/red]")

            if not skip_errors:
                console.print("[red]Stopping due to error (use --skip-errors to continue)[/red]")
                break

        # Add delay between demos (except after the last one)
        if i < len(DEMOS) and delay > 0:
            import time

            console.print(f"\n[dim]Waiting {delay} seconds before next demo...[/dim]")
            time.sleep(delay)

    # Summary
    console.rule("[bold]Summary[/bold]")
    console.print(f"[green]Successful:[/green] {success_count}")
    console.print(f"[red]Failed:[/red] {error_count}")
    console.print(f"[dim]Total:[/dim] {len(DEMOS)}")


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-V", help="Show version information"),
):
    """
    Model Guided Research CLI - Run experimental mathematical models for ML research

    This CLI provides easy access to various experimental mathematical models
    and algorithms for machine learning research, including fractal memories,
    knot-theoretic attention, gauge learning, and more.
    """
    if version:
        console.print("[bold]Model Guided Research[/bold] v0.1.0")
        raise typer.Exit()


@app.command("eval")
def evaluate(
    ultra_packed: Annotated[
        bool,
        typer.Option(
            "--ultra-packed", help="Use packed bit-trie implementation for ultrametric tests (sets ULTRA_PACKED=1)"
        ),
    ] = False,
    seed: Annotated[int | None, typer.Option("--seed", "-s", help="Random seed for reproducibility")] = None,
    export_json: Annotated[
        Path | None, typer.Option("--export-json", help="Write a combined JSON artifact of the practical utility suite")
    ] = None,
    artifacts_dir: Annotated[
        Path | None,
        typer.Option(
            "--artifacts-dir",
            help="Write run artifacts under this directory using the standard layout (see artifacts/README.md).",
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id", help="Run identifier (directory name) when writing artifacts. Defaults to YYYYMMDD_HHMMSS."
        ),
    ] = None,
    print_ultra_table: Annotated[
        bool, typer.Option("--print-ultra-table", help="Print ultrametric exponent table")
    ] = False,
    print_trop_table: Annotated[
        bool, typer.Option("--print-trop-table", help="Print tropical Lipschitz table")
    ] = False,
):
    """Run the practical utility test suite and optionally export a JSON artifact."""
    from config import ProjectConfig, set_config
    from utils import seed_everything

    config = ProjectConfig()
    if seed is not None:
        config.random_seed = seed
    set_config(config)
    seed_everything(config.random_seed)

    import os as _os

    if ultra_packed:
        _os.environ["ULTRA_PACKED"] = "1"
    if print_ultra_table:
        _os.environ["PRINT_ULTRA_TABLE"] = "1"
    if print_trop_table:
        _os.environ["PRINT_TROP_TABLE"] = "1"

    from tests.test_practical_utility import run_all_utility_tests

    results = run_all_utility_tests()

    if export_json is not None:
        payload = []
        for r in results:
            payload.append(
                {
                    "approach": r.approach_name,
                    "claim": r.claim,
                    "baseline": float(r.baseline_metric),
                    "proposed": float(r.proposed_metric),
                    "improvement": float(r.improvement_ratio),
                    "is_better": bool(r.is_better),
                    "verdict": r.verdict,
                    "details": r.details,
                }
            )
        try:
            export_json.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        with export_json.open("w", encoding="utf-8") as f:
            json.dump({"results": payload}, f, indent=2)
        console.print(f"[dim]Wrote suite JSON to {export_json}[/dim]")

    if artifacts_dir is not None:
        resolved_run_id = run_id or _default_run_id()
        run_dir = artifacts_dir / "bench" / "practical_utility" / resolved_run_id

        summary = {
            "meta": {
                "suite": "practical_utility",
                "run_id": resolved_run_id,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "git": _get_git_info(),
                "python": {
                    "executable": sys.executable,
                    "version": platform.python_version(),
                },
                "argv": sys.argv,
                "seed": int(config.random_seed),
                "flags": {
                    "ultra_packed": bool(ultra_packed),
                    "print_ultra_table": bool(print_ultra_table),
                    "print_trop_table": bool(print_trop_table),
                },
            },
            "results": payload,
        }

        report_md = f"""# Practical Utility Suite

- Run ID: `{resolved_run_id}`
- Generated: {summary["meta"]["generated_at"]}
- Commit: {summary["meta"]["git"]["commit_full"]}{" (dirty)" if summary["meta"]["git"]["dirty"] else " (clean)"}

## Command

```bash
{shlex.join(sys.argv)}
```

See `summary.json` for full details.
"""
        _write_artifacts(run_dir, summary=summary, report_md=report_md)
        console.print(f"[dim]Wrote artifacts → {run_dir}[/dim]")


@app.command("bench-fixed-flops")
def bench_fixed_flops(
    attention_types: Annotated[
        list[str],
        typer.Option(
            "--attention-type",
            "-a",
            help="Nanochat attention types to benchmark (repeatable).",
        ),
    ] = ("standard", "tropical", "ultrametric", "simplicial", "reversible", "gauge"),
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Device for nanochat training runs (passed through to nanochat.train).",
        ),
    ] = "cpu",
    target_flops: Annotated[
        float,
        typer.Option(
            "--target-flops",
            help="Target total FLOPs budget (est) per run.",
            min=1e6,
        ),
    ] = 2e9,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Training seed (same seed used for each attention type).",
        ),
    ] = 0,
    score_tail: Annotated[
        int,
        typer.Option(
            "--score-tail",
            help="Score is mean of last N losses from nanochat summary.json.",
            min=1,
        ),
    ] = 3,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Batch size for nanochat training.",
            min=1,
        ),
    ] = 8,
    sequence_len: Annotated[
        int,
        typer.Option(
            "--sequence-len",
            help="Sequence length for nanochat training.",
            min=8,
        ),
    ] = 256,
    n_layer: Annotated[
        int,
        typer.Option(
            "--n-layer",
            help="Number of transformer layers.",
            min=1,
        ),
    ] = 4,
    n_head: Annotated[
        int,
        typer.Option(
            "--n-head",
            help="Number of attention heads.",
            min=1,
        ),
    ] = 4,
    n_kv_head: Annotated[
        int,
        typer.Option(
            "--n-kv-head",
            help="Number of KV heads (GQA).",
            min=1,
        ),
    ] = 4,
    n_embd: Annotated[
        int,
        typer.Option(
            "--n-embd",
            help="Embedding dimension.",
            min=16,
        ),
    ] = 128,
    optimizer_type: Annotated[
        str,
        typer.Option(
            "--optimizer-type",
            help="nanochat optimizer type (passed through).",
        ),
    ] = "adamw",
    learning_rate: Annotated[
        float,
        typer.Option(
            "--learning-rate",
            help="Base learning rate for nanochat.train.",
            min=1e-8,
        ),
    ] = 6e-4,
    warmup_steps: Annotated[
        int,
        typer.Option(
            "--warmup-steps",
            help="Warmup steps excluded from throughput measurement.",
            min=0,
        ),
    ] = 0,
    log_interval: Annotated[
        int,
        typer.Option(
            "--log-interval",
            help="Train logging interval (steps).",
            min=1,
        ),
    ] = 1,
    check_numerics: Annotated[
        bool,
        typer.Option(
            "--check-numerics",
            help="Enable NaN/Inf watchpoints inside nanochat.train.",
        ),
    ] = False,
    compile: Annotated[
        bool,
        typer.Option(
            "--compile/--no-compile",
            help="Enable torch.compile in nanochat.train (optional).",
        ),
    ] = False,
    auto_download_data: Annotated[
        bool,
        typer.Option(
            "--auto-download-data/--no-auto-download-data",
            help="Auto-download minimal dataset shards if missing.",
        ),
    ] = True,
    min_parquet_files: Annotated[
        int,
        typer.Option(
            "--min-parquet-files",
            help="Minimum number of parquet shards required (>=2 recommended).",
            min=2,
        ),
    ] = 2,
    include_demo_certs: Annotated[
        bool,
        typer.Option(
            "--include-demo-certs/--no-include-demo-certs",
            help="Also run a few demo certificate runs (math-specific diagnostics) and link them in the suite report.",
        ),
    ] = False,
    artifacts_dir: Annotated[
        Path,
        typer.Option(
            "--artifacts-dir",
            help="Base directory for artifacts (default: artifacts/).",
        ),
    ] = Path("artifacts"),
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help="Suite run identifier (directory name). Defaults to YYYYMMDD_HHMMSS.",
        ),
    ] = None,
    timeout_s: Annotated[
        float,
        typer.Option(
            "--timeout-s",
            help="Per-run timeout (seconds) for subprocess invocations.",
            min=1.0,
        ),
    ] = 1800.0,
):
    attention_types = list(attention_types)
    """Benchmark nanochat attention variants under a fixed FLOPs budget.

    Per-run nanochat artifacts:
    - `artifacts/bench/fixed_flops/nanochat/<suite_run_id>/<attention_type>/seed_<seed>/`

    Suite aggregation:
    - `artifacts/bench/fixed_flops/nanochat/<suite_run_id>/summary.json`
    - `artifacts/bench/fixed_flops/nanochat/<suite_run_id>/run.md`
    """
    if not attention_types:
        raise typer.BadParameter("--attention-type must be provided at least once")

    suite_run_id = run_id or _default_run_id()
    suite_dir = artifacts_dir / "bench" / "fixed_flops" / "nanochat" / suite_run_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = suite_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    bench_meta = {
        "suite": "bench_fixed_flops",
        "run_id": suite_run_id,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git": _get_git_info(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "argv": sys.argv,
        "device": device,
        "target_flops": float(target_flops),
        "seed": int(seed),
        "score_tail": int(score_tail),
        "train_config": {
            "batch_size": int(batch_size),
            "sequence_len": int(sequence_len),
            "n_layer": int(n_layer),
            "n_head": int(n_head),
            "n_kv_head": int(n_kv_head),
            "n_embd": int(n_embd),
            "optimizer_type": str(optimizer_type),
            "learning_rate": float(learning_rate),
            "warmup_steps": int(warmup_steps),
            "log_interval": int(log_interval),
            "check_numerics": bool(check_numerics),
            "compile": bool(compile),
        },
        "attention_types": list(attention_types),
    }

    def _run_train(attn: str) -> dict[str, Any]:
        run_topic = f"fixed_flops/nanochat/{suite_run_id}/{attn}"
        run_id_local = f"seed_{seed}"
        train_cmd = [
            sys.executable,
            "-m",
            "nanochat.train",
            "--device",
            device,
            "--seed",
            str(seed),
            "--batch-size",
            str(batch_size),
            "--sequence-len",
            str(sequence_len),
            "--n-layer",
            str(n_layer),
            "--n-head",
            str(n_head),
            "--n-kv-head",
            str(n_kv_head),
            "--n-embd",
            str(n_embd),
            "--learning-rate",
            str(learning_rate),
            "--optimizer-type",
            str(optimizer_type),
            "--attention-type",
            attn,
            "--target-flops",
            str(float(target_flops)),
            "--warmup-steps",
            str(int(warmup_steps)),
            "--log-interval",
            str(int(log_interval)),
            "--artifacts-dir",
            str(artifacts_dir),
            "--artifacts-kind",
            "bench",
            "--artifacts-topic",
            run_topic,
            "--run-id",
            run_id_local,
        ]
        if compile:
            train_cmd.append("--compile")
        if check_numerics:
            train_cmd.append("--check-numerics")
        if auto_download_data:
            train_cmd.extend(["--auto-download-data", "--min-parquet-files", str(min_parquet_files)])

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(  # nosec B603
                train_cmd,
                capture_output=True,
                text=True,
                timeout=float(timeout_s),
                check=False,
            )
            stdout = proc.stdout
            stderr = proc.stderr
            returncode = int(proc.returncode)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            returncode = 124
        t1 = time.perf_counter()

        stdout_path = logs_dir / f"nanochat_{attn}.stdout.txt"
        stderr_path = logs_dir / f"nanochat_{attn}.stderr.txt"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

        summary_path = artifacts_dir / "bench" / run_topic / run_id_local / "summary.json"
        status = "ok" if returncode == 0 and summary_path.exists() else ("timeout" if returncode == 124 else "error")

        losses: list[float] = []
        final_loss: float | None = None
        score: float | None = None
        ppl: float | None = None
        tokens_s: float | None = None
        tflops_s: float | None = None
        peak_mem_gb: float | None = None
        if status == "ok":
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            res = payload.get("results", {})
            losses = [float(x) for x in res.get("losses", [])]
            if losses:
                final_loss = float(losses[-1])
                tail = losses[-min(len(losses), int(score_tail)) :]
                score = float(sum(tail) / len(tail))
                ppl = float(math.exp(score))
            if isinstance(res.get("tokens_per_second"), int | float):
                tokens_s = float(res["tokens_per_second"])
            if isinstance(res.get("tflops_per_second_est"), int | float):
                tflops_s = float(res["tflops_per_second_est"])
            if isinstance(res.get("peak_memory_allocated_gb"), int | float):
                peak_mem_gb = float(res["peak_memory_allocated_gb"])

        return {
            "attention_type": attn,
            "status": status,
            "returncode": int(returncode),
            "duration_s": float(t1 - t0),
            "command": shlex.join(train_cmd),
            "stdout_path": str(stdout_path.relative_to(artifacts_dir)),
            "stderr_path": str(stderr_path.relative_to(artifacts_dir)),
            "summary_path": str(summary_path.relative_to(artifacts_dir)) if summary_path.exists() else None,
            "score": score,
            "final_loss": final_loss,
            "perplexity_est": ppl,
            "tokens_per_second": tokens_s,
            "tflops_per_second_est": tflops_s,
            "peak_memory_allocated_gb": peak_mem_gb,
        }

    results: list[dict[str, Any]] = []
    total = len(attention_types)
    with Progress(
        TextColumn("[bold cyan]fixed-FLOPs bench[/bold cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("runs", total=total)
        for attn in attention_types:
            console.print(Panel(f"[bold]nanochat[/bold] attention_type={attn!r}", box=box.ROUNDED))
            results.append(_run_train(attn))
            prog.advance(task)

    demo_certs: list[dict[str, Any]] = []
    if include_demo_certs:
        console.print(Panel("[bold]Demo certificates[/bold] (math-specific diagnostics)", box=box.ROUNDED))
        cert_runs: list[tuple[str, list[str]]] = [
            ("tropical", ["--tropical-cert"]),
            ("reversible", ["--rev-cayley", "--rev-symplectic"]),
            ("matrix-gauge", ["--gauge-structured"]),
            ("simplicial", ["--simplicial-hodge"]),
            ("ultrametric", ["--ultra-packed"]),
        ]
        for demo_name, extra_flags in cert_runs:
            cert_run_id = f"{suite_run_id}_{demo_name}"
            cmd = [
                sys.executable,
                "-m",
                "cli",
                "run",
                demo_name,
                "--max-iterations",
                "50",
                "--seed",
                str(seed),
                "--artifacts-dir",
                str(artifacts_dir),
                "--run-id",
                cert_run_id,
                *extra_flags,
            ]
            try:
                proc = subprocess.run(  # nosec B603
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=float(timeout_s),
                    check=False,
                )
                stdout = proc.stdout
                stderr = proc.stderr
                returncode = int(proc.returncode)
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                returncode = 124

            (logs_dir / f"demo_{demo_name}.stdout.txt").write_text(stdout, encoding="utf-8")
            (logs_dir / f"demo_{demo_name}.stderr.txt").write_text(stderr, encoding="utf-8")

            cert_dir = artifacts_dir / "certs" / "demos" / demo_name / cert_run_id
            cert_summary = cert_dir / "summary.json"
            demo_certs.append(
                {
                    "demo": demo_name,
                    "status": "ok"
                    if returncode == 0 and cert_summary.exists()
                    else ("timeout" if returncode == 124 else "error"),
                    "returncode": int(returncode),
                    "command": shlex.join(cmd),
                    "summary_path": str(cert_summary.relative_to(artifacts_dir)) if cert_summary.exists() else None,
                }
            )

    baseline_attn = "standard" if "standard" in attention_types else attention_types[0]
    baseline = next((r for r in results if r["attention_type"] == baseline_attn), None)

    table = Table(title="Fixed-FLOPs nanochat benchmark (train loss)", box=box.ROUNDED)
    table.add_column("attention_type", style="bold")
    table.add_column("status")
    table.add_column("score", justify="right")
    table.add_column("Δ vs baseline", justify="right")
    table.add_column("tokens/s", justify="right")
    table.add_column("TFLOP/s(est)", justify="right")
    table.add_column("peak_mem_gb", justify="right")

    for r in results:
        score = r.get("score")
        delta = None
        if baseline and baseline.get("score") is not None and score is not None:
            base = float(baseline["score"])
            delta = (float(score) - base) / base if base != 0 else None

        table.add_row(
            str(r["attention_type"]),
            str(r["status"]),
            f"{float(score):.6f}" if isinstance(score, int | float) else "n/a",
            f"{float(delta):+.2%}" if isinstance(delta, int | float) else "n/a",
            f"{float(r['tokens_per_second']):,.0f}" if isinstance(r.get("tokens_per_second"), int | float) else "n/a",
            f"{float(r['tflops_per_second_est']):.2f}"
            if isinstance(r.get("tflops_per_second_est"), int | float)
            else "n/a",
            f"{float(r['peak_memory_allocated_gb']):.2f}"
            if isinstance(r.get("peak_memory_allocated_gb"), int | float)
            else "n/a",
        )

    console.print(table)

    summary = {
        "schema_version": "mgr.bench.fixed_flops.v1",
        "meta": bench_meta,
        "baseline_attention_type": baseline_attn,
        "runs": results,
        "demo_certs": demo_certs,
    }

    ok_runs = [r for r in results if r.get("status") == "ok" and isinstance(r.get("score"), (int, float))]
    ok_runs_sorted = sorted(ok_runs, key=lambda r: float(r["score"]))
    best = ok_runs_sorted[0] if ok_runs_sorted else None

    def _md_row(values: list[str]) -> str:
        return "| " + " | ".join(values) + " |"

    md_lines: list[str] = []
    md_lines.append("# Fixed-FLOPs nanochat benchmark")
    md_lines.append("")
    md_lines.append(f"- Run ID: `{suite_run_id}`")
    md_lines.append(f"- Baseline: `{baseline_attn}`")
    md_lines.append(f"- Device: `{device}`")
    md_lines.append(f"- Target FLOPs/run (est): `{float(target_flops):.3e}`")
    md_lines.append(f"- Seed: `{seed}`")
    md_lines.append("")
    md_lines.append("## Results")
    md_lines.append("")
    md_lines.append(
        _md_row(["attention_type", "status", "score", "Δ vs baseline", "tokens/s", "TFLOP/s(est)", "peak_mem_gb"])
    )
    md_lines.append(_md_row(["---"] * 7))

    for r in results:
        score = r.get("score")
        delta = None
        if baseline and baseline.get("score") is not None and score is not None:
            base = float(baseline["score"])
            delta = (float(score) - base) / base if base != 0 else None
        md_lines.append(
            _md_row(
                [
                    str(r["attention_type"]),
                    str(r["status"]),
                    f"{float(score):.6f}" if isinstance(score, (int, float)) else "n/a",
                    f"{float(delta):+.2%}" if isinstance(delta, (int, float)) else "n/a",
                    f"{float(r['tokens_per_second']):,.0f}"
                    if isinstance(r.get("tokens_per_second"), (int, float))
                    else "n/a",
                    f"{float(r['tflops_per_second_est']):.2f}"
                    if isinstance(r.get("tflops_per_second_est"), (int, float))
                    else "n/a",
                    f"{float(r['peak_memory_allocated_gb']):.2f}"
                    if isinstance(r.get("peak_memory_allocated_gb"), (int, float))
                    else "n/a",
                ]
            )
        )

    md_lines.append("")
    md_lines.append("## Conclusions")
    md_lines.append("")
    if best is None:
        md_lines.append("- No successful runs; see `logs/` for stdout/stderr.")
    else:
        md_lines.append(f"- Best (lowest score): `{best['attention_type']}` score=`{float(best['score']):.6f}`")
        if baseline and baseline.get("score") is not None:
            base = float(baseline["score"])
            md_lines.append(
                f"- Baseline `{baseline_attn}` score=`{base:.6f}`; best Δ=`{(float(best['score']) - base) / base:+.2%}`"
            )
        better = []
        worse = []
        if baseline and baseline.get("score") is not None:
            base = float(baseline["score"])
            for r in ok_runs_sorted:
                if r["attention_type"] == baseline_attn:
                    continue
                d = (float(r["score"]) - base) / base if base != 0 else 0.0
                (better if d < 0 else worse).append((r["attention_type"], d))
        if better:
            md_lines.append("- Better than baseline: " + ", ".join(f"`{a}` ({d:+.2%})" for a, d in better))
        if worse:
            md_lines.append("- Worse than baseline: " + ", ".join(f"`{a}` ({d:+.2%})" for a, d in worse))
        failed = [r for r in results if r.get("status") != "ok"]
        if failed:
            md_lines.append("- Failures: " + ", ".join(f"`{f['attention_type']}`" for f in failed))

    if demo_certs:
        md_lines.append("")
        md_lines.append("## Demo Certificates (Diagnostics)")
        md_lines.append("")
        for d in demo_certs:
            md_lines.append(f"- `{d['demo']}` status={d['status']} summary={d.get('summary_path')}")

    md_lines.append("")
    md_lines.append("## Command")
    md_lines.append("")
    md_lines.append("```bash")
    md_lines.append(shlex.join(sys.argv))
    md_lines.append("```")
    report_md = "\n".join(md_lines) + "\n"

    _write_artifacts(suite_dir, summary=summary, report_md=report_md)
    console.print(f"[dim]Wrote suite artifacts → {suite_dir}[/dim]")


@app.command("per-head-metrics")
def per_head_metrics(
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Device for nanochat training runs (passed through to nanochat.train).",
        ),
    ] = "cpu",
    seeds: Annotated[
        list[int],
        typer.Option(
            "--seed",
            "-s",
            help="Training seeds (repeatable).",
        ),
    ] = (0, 1, 2),
    target_flops: Annotated[
        float,
        typer.Option(
            "--target-flops",
            help="Target total FLOPs budget (est) per run.",
            min=1e6,
        ),
    ] = 2e8,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Batch size for nanochat training.",
            min=1,
        ),
    ] = 8,
    sequence_len: Annotated[
        int,
        typer.Option(
            "--sequence-len",
            help="Sequence length for nanochat training.",
            min=8,
        ),
    ] = 256,
    n_layer: Annotated[
        int,
        typer.Option(
            "--n-layer",
            help="Number of transformer layers.",
            min=1,
        ),
    ] = 4,
    n_head: Annotated[
        int,
        typer.Option(
            "--n-head",
            help="Number of attention heads.",
            min=1,
        ),
    ] = 4,
    n_kv_head: Annotated[
        int,
        typer.Option(
            "--n-kv-head",
            help="Number of KV heads (GQA).",
            min=1,
        ),
    ] = 4,
    n_embd: Annotated[
        int,
        typer.Option(
            "--n-embd",
            help="Embedding dimension.",
            min=16,
        ),
    ] = 128,
    optimizer_type: Annotated[
        str,
        typer.Option(
            "--optimizer-type",
            help="nanochat optimizer type (passed through).",
        ),
    ] = "adamw",
    learning_rate: Annotated[
        float,
        typer.Option(
            "--learning-rate",
            help="Base learning rate for nanochat.train.",
            min=1e-8,
        ),
    ] = 6e-4,
    warmup_steps: Annotated[
        int,
        typer.Option(
            "--warmup-steps",
            help="Warmup steps excluded from throughput measurement.",
            min=0,
        ),
    ] = 0,
    log_interval: Annotated[
        int,
        typer.Option(
            "--log-interval",
            help="Train logging interval (steps).",
            min=1,
        ),
    ] = 1,
    auto_download_data: Annotated[
        bool,
        typer.Option(
            "--auto-download-data/--no-auto-download-data",
            help="Auto-download minimal dataset shards if missing.",
        ),
    ] = True,
    min_parquet_files: Annotated[
        int,
        typer.Option(
            "--min-parquet-files",
            help="Minimum number of parquet shards required (>=2 recommended).",
            min=2,
        ),
    ] = 2,
    artifacts_dir: Annotated[
        Path,
        typer.Option(
            "--artifacts-dir",
            help="Base directory for artifacts (default: artifacts/).",
        ),
    ] = Path("artifacts"),
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help="Suite run identifier (directory name). Defaults to YYYYMMDD_HHMMSS.",
        ),
    ] = None,
    timeout_s: Annotated[
        float,
        typer.Option(
            "--timeout-s",
            help="Per-run timeout (seconds) for subprocess invocations.",
            min=1.0,
        ),
    ] = 1800.0,
):
    """Compute per-head stability/error bars across multiple seeds for a few small nanochat configs.

    Note: the FlexAttention variant requires CUDA and is skipped otherwise.
    """
    seeds = list(seeds)
    if not seeds:
        raise typer.BadParameter("--seed must be provided at least once")

    suite_run_id = run_id or _default_run_id()
    suite_dir = artifacts_dir / "bench" / "feature_ablate" / "per_head_metrics" / suite_run_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = suite_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "suite": "per_head_metrics",
        "run_id": suite_run_id,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git": _get_git_info(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "argv": sys.argv,
        "device": device,
        "seeds": [int(s) for s in seeds],
        "train_config": {
            "batch_size": int(batch_size),
            "sequence_len": int(sequence_len),
            "n_layer": int(n_layer),
            "n_head": int(n_head),
            "n_kv_head": int(n_kv_head),
            "n_embd": int(n_embd),
            "optimizer_type": str(optimizer_type),
            "learning_rate": float(learning_rate),
            "warmup_steps": int(warmup_steps),
            "log_interval": int(log_interval),
            "target_flops": float(target_flops),
        },
    }

    variants: list[dict[str, Any]] = [
        {
            "key": "standard",
            "label": "standard (SDPA)",
            "attention_type": "standard",
            "use_flex_attention": False,
            "extra_flags": ["--standard-record-attn-entropy"],
        },
        {
            "key": "standard_flex",
            "label": "standard (FlexAttention)",
            "attention_type": "standard",
            "use_flex_attention": True,
            "extra_flags": ["--standard-record-attn-entropy"],
        },
        {
            "key": "tropical",
            "label": "tropical (margins)",
            "attention_type": "tropical",
            "use_flex_attention": False,
            "extra_flags": ["--tropical-record-margins"],
        },
    ]

    def _run_train(variant: dict[str, Any], *, seed: int) -> dict[str, Any]:
        run_topic = f"feature_ablate/per_head_metrics/{suite_run_id}/{variant['key']}"
        run_id_local = f"seed_{seed}"
        train_cmd = [
            sys.executable,
            "-m",
            "nanochat.train",
            "--device",
            device,
            "--seed",
            str(seed),
            "--batch-size",
            str(batch_size),
            "--sequence-len",
            str(sequence_len),
            "--n-layer",
            str(n_layer),
            "--n-head",
            str(n_head),
            "--n-kv-head",
            str(n_kv_head),
            "--n-embd",
            str(n_embd),
            "--learning-rate",
            str(learning_rate),
            "--optimizer-type",
            str(optimizer_type),
            "--attention-type",
            str(variant["attention_type"]),
            "--target-flops",
            str(float(target_flops)),
            "--warmup-steps",
            str(int(warmup_steps)),
            "--log-interval",
            str(int(log_interval)),
            "--artifacts-dir",
            str(artifacts_dir),
            "--artifacts-kind",
            "bench",
            "--artifacts-topic",
            run_topic,
            "--run-id",
            run_id_local,
            *list(variant.get("extra_flags", [])),
        ]
        if bool(variant.get("use_flex_attention", False)):
            train_cmd.append("--use-flex-attention")
        if auto_download_data:
            train_cmd.extend(["--auto-download-data", "--min-parquet-files", str(min_parquet_files)])

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(  # nosec B603
                train_cmd,
                capture_output=True,
                text=True,
                timeout=float(timeout_s),
                check=False,
            )
            stdout = proc.stdout
            stderr = proc.stderr
            returncode = int(proc.returncode)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            returncode = 124
        t1 = time.perf_counter()

        stdout_path = logs_dir / f"nanochat_{variant['key']}_seed_{seed}.stdout.txt"
        stderr_path = logs_dir / f"nanochat_{variant['key']}_seed_{seed}.stderr.txt"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

        summary_path = artifacts_dir / "bench" / run_topic / run_id_local / "summary.json"
        status = "ok" if returncode == 0 and summary_path.exists() else ("timeout" if returncode == 124 else "error")

        final_loss: float | None = None
        tokens_s: float | None = None
        tflops_s: float | None = None
        peak_mem_gb: float | None = None
        use_flex_actual: bool | None = None
        entropy_head_mean: list[float] | None = None
        tropical_gamma_head_mean: list[float] | None = None
        if status == "ok":
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            res = payload.get("results", {}) if isinstance(payload.get("results"), dict) else {}
            cfg = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
            use_flex_actual = bool(cfg.get("use_flex_attention", False))

            losses = res.get("losses")
            if isinstance(losses, list) and losses:
                try:
                    final_loss = float(losses[-1])
                except Exception:
                    final_loss = None
            if isinstance(res.get("tokens_per_second"), int | float):
                tokens_s = float(res["tokens_per_second"])
            if isinstance(res.get("tflops_per_second_est"), int | float):
                tflops_s = float(res["tflops_per_second_est"])
            if isinstance(res.get("peak_memory_allocated_gb"), int | float):
                peak_mem_gb = float(res["peak_memory_allocated_gb"])

            entropy = res.get("attention_entropy")
            if isinstance(entropy, dict) and isinstance(entropy.get("head_mean"), list):
                try:
                    entropy_head_mean = [float(x) for x in entropy["head_mean"]]
                except Exception:
                    entropy_head_mean = None
            tropical = res.get("tropical_margins")
            if isinstance(tropical, dict) and isinstance(tropical.get("head_mean"), list):
                try:
                    tropical_gamma_head_mean = [float(x) for x in tropical["head_mean"]]
                except Exception:
                    tropical_gamma_head_mean = None

            expect_flex = bool(variant.get("use_flex_attention", False))
            if expect_flex != bool(use_flex_actual):
                status = "mismatch"

        return {
            "variant": str(variant["key"]),
            "seed": int(seed),
            "status": status,
            "returncode": int(returncode),
            "duration_s": float(t1 - t0),
            "command": shlex.join(train_cmd),
            "stdout_path": str(stdout_path.relative_to(artifacts_dir)),
            "stderr_path": str(stderr_path.relative_to(artifacts_dir)),
            "summary_path": str(summary_path.relative_to(artifacts_dir)) if summary_path.exists() else None,
            "final_loss": final_loss,
            "tokens_per_second": tokens_s,
            "tflops_per_second_est": tflops_s,
            "peak_memory_allocated_gb": peak_mem_gb,
            "use_flex_attention": use_flex_actual,
            "attention_entropy_head_mean": entropy_head_mean,
            "tropical_gamma_head_mean": tropical_gamma_head_mean,
        }

    results: list[dict[str, Any]] = []
    cuda_available = False
    if device == "auto":
        try:
            import torch
        except Exception:
            cuda_available = False
        else:
            cuda_available = bool(torch.cuda.is_available())
    flex_capable_device = device == "cuda" or (device == "auto" and cuda_available)
    runnable_variants = [v for v in variants if not bool(v.get("use_flex_attention", False)) or flex_capable_device]
    skipped_variants = [v for v in variants if v not in runnable_variants]
    for v in skipped_variants:
        for seed in seeds:
            results.append(
                {
                    "variant": str(v["key"]),
                    "seed": int(seed),
                    "status": "skipped",
                    "returncode": None,
                    "duration_s": 0.0,
                    "command": None,
                    "stdout_path": None,
                    "stderr_path": None,
                    "summary_path": None,
                    "final_loss": None,
                    "tokens_per_second": None,
                    "tflops_per_second_est": None,
                    "peak_memory_allocated_gb": None,
                    "use_flex_attention": None,
                    "attention_entropy_head_mean": None,
                    "tropical_gamma_head_mean": None,
                    "note": f"variant requires CUDA (device={device!r})",
                }
            )

    total = len(runnable_variants) * len(seeds)
    with Progress(
        TextColumn("[bold cyan]per-head metrics[/bold cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("runs", total=total)
        for variant in runnable_variants:
            console.print(Panel(f"[bold]nanochat[/bold] variant={variant['key']!r}", box=box.ROUNDED))
            for seed in seeds:
                results.append(_run_train(variant, seed=int(seed)))
                prog.advance(task)

    variants_out: list[dict[str, Any]] = []
    for variant in variants:
        key = str(variant["key"])
        runs = [r for r in results if r.get("variant") == key]
        loss_vals = [float(r["final_loss"]) for r in runs if isinstance(r.get("final_loss"), int | float)]
        loss_stats = _summary_stats(loss_vals)

        entropy_agg = _aggregate_per_head(
            [
                (int(r["seed"]), r["attention_entropy_head_mean"])
                for r in runs
                if isinstance(r.get("seed"), int) and isinstance(r.get("attention_entropy_head_mean"), list)
            ]
        )
        gamma_agg = _aggregate_per_head(
            [
                (int(r["seed"]), r["tropical_gamma_head_mean"])
                for r in runs
                if isinstance(r.get("seed"), int) and isinstance(r.get("tropical_gamma_head_mean"), list)
            ]
        )
        metrics: dict[str, Any] = {}
        if entropy_agg is not None:
            metrics["attention_entropy"] = entropy_agg
        if gamma_agg is not None:
            metrics["tropical_margin_gamma"] = gamma_agg

        variants_out.append(
            {
                "variant": key,
                "label": str(variant.get("label", key)),
                "attention_type": str(variant.get("attention_type")),
                "expected_use_flex_attention": bool(variant.get("use_flex_attention", False)),
                "runs": runs,
                "final_loss": loss_stats,
                "metrics": metrics,
            }
        )

    def _md_row(values: list[str]) -> str:
        return "| " + " | ".join(values) + " |"

    def _md_per_head_table(metric: dict[str, Any], *, value_name: str) -> list[str]:
        n_head = int(metric.get("n_head", 0))
        means = metric.get("mean", [])
        stds = metric.get("std", [])
        ci95s = metric.get("ci95", [])
        lines: list[str] = []
        lines.append(_md_row(["head", value_name, "std", "ci95"]))
        lines.append(_md_row(["---"] * 4))
        for i in range(n_head):
            mean = means[i] if i < len(means) else float("nan")
            std = stds[i] if i < len(stds) else float("nan")
            ci = ci95s[i] if i < len(ci95s) else None
            lines.append(
                _md_row(
                    [
                        str(i),
                        f"{float(mean):.6f}" if isinstance(mean, int | float) and math.isfinite(float(mean)) else "n/a",
                        f"{float(std):.6f}" if isinstance(std, int | float) and math.isfinite(float(std)) else "n/a",
                        f"{float(ci):.6f}" if isinstance(ci, int | float) and math.isfinite(float(ci)) else "n/a",
                    ]
                )
            )
        return lines

    md_lines: list[str] = []
    md_lines.append("# Per-head metrics suite")
    md_lines.append("")
    md_lines.append(f"- Run ID: `{suite_run_id}`")
    md_lines.append(f"- Device: `{device}`")
    md_lines.append(f"- Seeds: `{', '.join(str(s) for s in seeds)}`")
    md_lines.append(f"- Target FLOPs/run (est): `{float(target_flops):.3e}`")
    md_lines.append("")
    md_lines.append("## Variants")
    md_lines.append("")

    for v in variants_out:
        md_lines.append(f"### {v['variant']}")
        md_lines.append("")
        md_lines.append(f"- label: `{v['label']}`")
        md_lines.append(f"- attention_type: `{v['attention_type']}`")
        md_lines.append(f"- expected_use_flex_attention: `{v['expected_use_flex_attention']}`")
        md_lines.append(
            f"- final_loss: mean=`{v['final_loss']['mean']}` std=`{v['final_loss']['std']}` ci95=`{v['final_loss']['ci95']}` n=`{v['final_loss']['n']}`"
        )
        md_lines.append("")
        metrics = v.get("metrics", {})
        if isinstance(metrics, dict) and metrics.get("attention_entropy") is not None:
            md_lines.append("#### attention_entropy (per head)")
            md_lines.append("")
            md_lines.extend(_md_per_head_table(metrics["attention_entropy"], value_name="entropy_mean"))
            md_lines.append("")
        if isinstance(metrics, dict) and metrics.get("tropical_margin_gamma") is not None:
            md_lines.append("#### tropical margin gamma (per head)")
            md_lines.append("")
            md_lines.extend(_md_per_head_table(metrics["tropical_margin_gamma"], value_name="gamma_mean"))
            md_lines.append("")

        md_lines.append("#### Runs")
        md_lines.append("")
        md_lines.append(_md_row(["seed", "status", "final_loss", "summary"]))
        md_lines.append(_md_row(["---"] * 4))
        for r in v.get("runs", []):
            md_lines.append(
                _md_row(
                    [
                        str(r.get("seed")),
                        str(r.get("status")),
                        f"{float(r['final_loss']):.6f}" if isinstance(r.get("final_loss"), int | float) else "n/a",
                        str(r.get("summary_path") or "n/a"),
                    ]
                )
            )
        md_lines.append("")

    md_lines.append("## Command")
    md_lines.append("")
    md_lines.append("```bash")
    md_lines.append(shlex.join(sys.argv))
    md_lines.append("```")
    report_md = "\n".join(md_lines) + "\n"

    summary = {
        "schema_version": "mgr.bench.per_head_metrics.v1",
        "meta": meta,
        "variants": variants_out,
    }

    for v in variants_out:
        metrics = v.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        panel_title = f"[bold]{v['variant']}[/bold] ({v['label']})"
        console.print(Panel(panel_title, box=box.ROUNDED))

        if metrics.get("attention_entropy") is not None:
            ent = metrics["attention_entropy"]
            table = Table(title="attention entropy per head (mean ± ci95)", box=box.ROUNDED)
            table.add_column("head", justify="right", style="cyan")
            table.add_column("mean", justify="right")
            table.add_column("std", justify="right")
            table.add_column("ci95", justify="right")
            for i in range(int(ent.get("n_head", 0))):
                mean = ent["mean"][i]
                std = ent["std"][i]
                ci = ent["ci95"][i]
                table.add_row(
                    str(i),
                    f"{float(mean):.6f}" if math.isfinite(float(mean)) else "n/a",
                    f"{float(std):.6f}" if math.isfinite(float(std)) else "n/a",
                    f"{float(ci):.6f}" if isinstance(ci, int | float) and math.isfinite(float(ci)) else "n/a",
                )
            console.print(table)

        if metrics.get("tropical_margin_gamma") is not None:
            gam = metrics["tropical_margin_gamma"]
            table = Table(title="tropical gamma per head (mean ± ci95)", box=box.ROUNDED)
            table.add_column("head", justify="right", style="cyan")
            table.add_column("mean", justify="right")
            table.add_column("std", justify="right")
            table.add_column("ci95", justify="right")
            for i in range(int(gam.get("n_head", 0))):
                mean = gam["mean"][i]
                std = gam["std"][i]
                ci = gam["ci95"][i]
                table.add_row(
                    str(i),
                    f"{float(mean):.6f}" if math.isfinite(float(mean)) else "n/a",
                    f"{float(std):.6f}" if math.isfinite(float(std)) else "n/a",
                    f"{float(ci):.6f}" if isinstance(ci, int | float) and math.isfinite(float(ci)) else "n/a",
                )
            console.print(table)

    _write_artifacts(suite_dir, summary=summary, report_md=report_md)
    console.print(f"[dim]Wrote suite artifacts → {suite_dir}[/dim]")


@app.command()
def regressions(
    baseline: Annotated[
        Path,
        typer.Option(
            "--baseline",
            "-b",
            help="Baseline run directory or summary.json (relative paths also searched under --artifacts-dir).",
        ),
    ],
    candidate: Annotated[
        Path,
        typer.Option(
            "--candidate",
            "-c",
            help="Candidate run directory or summary.json (relative paths also searched under --artifacts-dir).",
        ),
    ],
    baseline_variant: Annotated[
        str | None,
        typer.Option(
            "--baseline-variant",
            help="Optional sub-run selector inside baseline (e.g. attention_type in suite summaries; or 'sdpa'/'flex' in flex perf summaries).",
        ),
    ] = None,
    candidate_variant: Annotated[
        str | None,
        typer.Option(
            "--candidate-variant",
            help="Optional sub-run selector inside candidate (e.g. attention_type in suite summaries; or 'sdpa'/'flex' in flex perf summaries).",
        ),
    ] = None,
    loss_abs: Annotated[
        float,
        typer.Option(
            "--loss-abs",
            help="Absolute loss regression threshold (candidate > baseline + loss_abs).",
            min=0.0,
        ),
    ] = 0.01,
    loss_rel: Annotated[
        float,
        typer.Option(
            "--loss-rel",
            help="Relative loss regression threshold (candidate > baseline*(1+loss_rel)).",
            min=0.0,
        ),
    ] = 0.01,
    throughput_rel: Annotated[
        float,
        typer.Option(
            "--throughput-rel",
            help="Relative throughput regression threshold (candidate < baseline*(1-throughput_rel)).",
            min=0.0,
            max=1.0,
        ),
    ] = 0.05,
    tflops_rel: Annotated[
        float,
        typer.Option(
            "--tflops-rel",
            help="Relative TFLOP/s regression threshold (candidate < baseline*(1-tflops_rel)).",
            min=0.0,
            max=1.0,
        ),
    ] = 0.05,
    memory_rel: Annotated[
        float,
        typer.Option(
            "--memory-rel",
            help="Relative memory regression threshold (candidate > baseline*(1+memory_rel)).",
            min=0.0,
            max=10.0,
        ),
    ] = 0.05,
    artifacts_dir: Annotated[
        Path,
        typer.Option(
            "--artifacts-dir",
            help="Artifacts base directory (default: artifacts/).",
        ),
    ] = Path("artifacts"),
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help="Regression report run id (directory name). Defaults to YYYYMMDD_HHMMSS.",
        ),
    ] = None,
    write_artifacts: Annotated[
        bool,
        typer.Option(
            "--write-artifacts/--no-write-artifacts",
            help="Write summary.json + run.md under artifacts/regressions/<run_id>/.",
        ),
    ] = True,
    html: Annotated[
        bool,
        typer.Option(
            "--html/--no-html",
            help="Also write a minimal HTML report alongside run.md (when --write-artifacts is on).",
        ),
    ] = True,
    fail_on_regression: Annotated[
        bool,
        typer.Option(
            "--fail-on-regression/--no-fail-on-regression",
            help="Exit with code 1 if any metric is flagged as a regression (useful for guardrails/CI).",
        ),
    ] = False,
    fail_on_missing: Annotated[
        bool,
        typer.Option(
            "--fail-on-missing/--no-fail-on-missing",
            help="Also treat missing metrics as failures when --fail-on-regression is enabled.",
        ),
    ] = False,
):
    """Compare two artifact snapshots and highlight regressions/improvements.

    Supports common `summary.json` shapes used in this repo:
    - nanochat.train run summaries (results.losses, tokens_per_second, tflops_per_second_est, peak_memory_allocated_gb)
    - bench suite summaries (top-level runs[]; select a sub-run via --*-variant=attention_type)
    - flex perf summaries (results.tokens_per_s / results.peak_mem_mb; select via --*-variant=sdpa|flex)
    """
    baseline_path = _resolve_summary_path(baseline, artifacts_dir=artifacts_dir)
    candidate_path = _resolve_summary_path(candidate, artifacts_dir=artifacts_dir)

    base_obj = json.loads(baseline_path.read_text(encoding="utf-8"))
    cand_obj = json.loads(candidate_path.read_text(encoding="utf-8"))
    if not isinstance(base_obj, dict) or not isinstance(cand_obj, dict):
        raise typer.BadParameter("Expected both summaries to be JSON objects (dicts).")

    base_meta = _summarize_provenance(base_obj)
    cand_meta = _summarize_provenance(cand_obj)

    metrics = [
        ("final_loss", "Final loss", "lower"),
        ("tokens_per_second", "Tokens/s", "higher"),
        ("tflops_per_second_est", "TFLOP/s (est)", "higher"),
        ("peak_memory_allocated_gb", "Peak mem (GB)", "lower"),
    ]

    comparisons: list[dict[str, Any]] = []
    for key, label, direction in metrics:
        b = _extract_metric(base_obj, metric=key, variant=baseline_variant)
        c = _extract_metric(cand_obj, metric=key, variant=candidate_variant)
        if b is None or c is None:
            comparisons.append(
                {
                    "metric": key,
                    "label": label,
                    "direction": direction,
                    "baseline": b,
                    "candidate": c,
                    "delta": None,
                    "delta_rel": None,
                    "status": "missing",
                }
            )
            continue

        delta = float(c - b)
        delta_rel = float(delta / b) if b != 0 else None

        status = "ok"
        if direction == "lower":
            if c > b + loss_abs or (delta_rel is not None and delta_rel > loss_rel):
                status = "regression"
        else:
            thr = tflops_rel if key.startswith("tflops") else throughput_rel
            if c < b * (1.0 - thr):
                status = "regression"
        if direction == "lower" and key.startswith("peak_memory"):
            if c > b * (1.0 + memory_rel):
                status = "regression"

        comparisons.append(
            {
                "metric": key,
                "label": label,
                "direction": direction,
                "baseline": b,
                "candidate": c,
                "delta": delta,
                "delta_rel": delta_rel,
                "status": status,
            }
        )

    # Rich output
    header = Table.grid(padding=(0, 2))
    header.add_column(justify="left")
    header.add_column(justify="left")
    header.add_row("[bold]Baseline[/bold]", str(baseline_path))
    header.add_row("[bold]Candidate[/bold]", str(candidate_path))
    if baseline_variant or candidate_variant:
        header.add_row("[bold]Variants[/bold]", f"baseline={baseline_variant!r} candidate={candidate_variant!r}")
    console.print(Panel(header, title="Regression Diff", border_style="cyan"))

    meta_t = Table(title="Run provenance", box=box.SIMPLE_HEAVY)
    meta_t.add_column("field")
    meta_t.add_column("baseline", overflow="fold")
    meta_t.add_column("candidate", overflow="fold")
    for k in ("run_id", "device", "commit", "dirty", "attention_type", "use_flex_attention", "compile"):
        meta_t.add_row(str(k), str(base_meta.get(k)), str(cand_meta.get(k)))
    console.print(meta_t)

    loss_spark_base = _sparkline(_extract_loss_series(base_obj), width=24)
    loss_spark_cand = _sparkline(_extract_loss_series(cand_obj), width=24)

    t = Table(title="Metrics", box=box.SIMPLE_HEAVY)
    t.add_column("metric")
    t.add_column("baseline", justify="right")
    t.add_column("candidate", justify="right")
    t.add_column("Δ", justify="right")
    t.add_column("Δ%", justify="right")
    t.add_column("status", justify="right")
    for row in comparisons:
        b = row["baseline"]
        c = row["candidate"]
        d = row["delta"]
        dr = row["delta_rel"]
        status = row["status"]
        style = {"regression": "bold red", "ok": "green", "missing": "dim"}.get(status, "")
        t.add_row(
            row["label"],
            "-" if b is None else f"{b:.6g}",
            "-" if c is None else f"{c:.6g}",
            "-" if d is None else f"{d:+.6g}",
            "-" if dr is None else f"{(100.0 * dr):+.2f}%",
            f"[{style}]{status}[/{style}]" if style else status,
        )
    console.print(t)
    if loss_spark_base or loss_spark_cand:
        spark_t = Table(title="Loss sparklines (tail)", box=box.SIMPLE_HEAVY)
        spark_t.add_column("baseline")
        spark_t.add_column("candidate")
        spark_t.add_row(loss_spark_base or "-", loss_spark_cand or "-")
        console.print(spark_t)

    regressions_found = [row for row in comparisons if row.get("status") == "regression"]
    missing_found = [row for row in comparisons if row.get("status") == "missing"]

    thresholds = {
        "loss_abs": float(loss_abs),
        "loss_rel": float(loss_rel),
        "throughput_rel": float(throughput_rel),
        "tflops_rel": float(tflops_rel),
        "memory_rel": float(memory_rel),
    }

    report_run_id = run_id or _default_run_id()
    report_dir = artifacts_dir / "regressions" / report_run_id

    md_lines: list[str] = []
    md_lines.append("# Regression Report")
    md_lines.append("")
    md_lines.append(f"- generated_at: `{time.strftime('%Y-%m-%d %H:%M:%S %Z')}`")
    md_lines.append(f"- baseline: `{baseline_path}`")
    md_lines.append(f"- candidate: `{candidate_path}`")
    if baseline_variant or candidate_variant:
        md_lines.append(f"- variants: baseline=`{baseline_variant}` candidate=`{candidate_variant}`")
    md_lines.append("")
    md_lines.append("## Thresholds")
    md_lines.append("")
    md_lines.append("```json")
    md_lines.append(json.dumps(thresholds, indent=2, sort_keys=True))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## Metrics")
    md_lines.append("")
    md_lines.append("| metric | baseline | candidate | delta | delta% | status |")
    md_lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for row in comparisons:
        b = row["baseline"]
        c = row["candidate"]
        d = row["delta"]
        dr = row["delta_rel"]
        md_lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    "-" if b is None else f"{b:.6g}",
                    "-" if c is None else f"{c:.6g}",
                    "-" if d is None else f"{d:+.6g}",
                    "-" if dr is None else f"{(100.0 * dr):+.2f}%",
                    row["status"],
                ]
            )
            + " |"
        )
    if loss_spark_base or loss_spark_cand:
        md_lines.append("")
        md_lines.append("## Loss sparklines (tail)")
        md_lines.append("")
        md_lines.append(f"- baseline: `{loss_spark_base}`")
        md_lines.append(f"- candidate: `{loss_spark_cand}`")

    md_lines.append("")
    md_lines.append("## Command")
    md_lines.append("")
    md_lines.append("```bash")
    md_lines.append(shlex.join(sys.argv))
    md_lines.append("```")
    report_md = "\n".join(md_lines) + "\n"

    summary = {
        "meta": {
            "run_id": report_run_id,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "git": _get_git_info(),
            "argv": sys.argv,
            "baseline_path": str(baseline_path),
            "candidate_path": str(candidate_path),
            "baseline_variant": baseline_variant,
            "candidate_variant": candidate_variant,
        },
        "thresholds": thresholds,
        "baseline": base_meta,
        "candidate": cand_meta,
        "comparisons": comparisons,
    }

    if write_artifacts:
        _write_artifacts(report_dir, summary=summary, report_md=report_md)
        if html:
            html_table = [
                "<table><thead><tr><th>metric</th><th>baseline</th><th>candidate</th><th>delta</th><th>delta%</th><th>status</th></tr></thead><tbody>"
            ]
            for row in comparisons:
                b = row["baseline"]
                c = row["candidate"]
                d = row["delta"]
                dr = row["delta_rel"]
                status = row["status"]
                cls = "regression" if status == "regression" else ("ok" if status == "ok" else "missing")
                html_table.append(
                    "<tr class='"
                    + cls
                    + "'>"
                    + "".join(
                        f"<td>{cell}</td>"
                        for cell in [
                            row["label"],
                            "-" if b is None else f"{b:.6g}",
                            "-" if c is None else f"{c:.6g}",
                            "-" if d is None else f"{d:+.6g}",
                            "-" if dr is None else f"{(100.0 * dr):+.2f}%",
                            status,
                        ]
                    )
                    + "</tr>"
                )
            html_table.append("</tbody></table>")
            html_doc = "\n".join(
                [
                    "<!doctype html>",
                    "<meta charset='utf-8'/>",
                    "<title>Regression Report</title>",
                    "<style>",
                    "body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:24px;}",
                    "table{border-collapse:collapse;width:100%;}",
                    "th,td{border:1px solid #ddd;padding:8px;}",
                    "th{background:#f6f6f6;text-align:left;}",
                    "tr.ok td{background:#e9f7ef;}",
                    "tr.regression td{background:#fdecea;}",
                    "tr.missing td{color:#666;}",
                    "</style>",
                    "<h1>Regression Report</h1>",
                    f"<p><b>Baseline</b>: <code>{baseline_path}</code></p>",
                    f"<p><b>Candidate</b>: <code>{candidate_path}</code></p>",
                    "\n".join(html_table),
                ]
            )
            (report_dir / "report.html").write_text(html_doc + "\n", encoding="utf-8")
        console.print(f"[bold green]Wrote regression report[/bold green] → {report_dir}")

    if fail_on_regression:
        failing_rows = list(regressions_found)
        if fail_on_missing:
            failing_rows.extend(missing_found)

        if failing_rows:
            failing_metrics = ", ".join(row.get("metric", "?") for row in failing_rows)
            console.print(
                Panel(
                    f"[bold red]Regressions detected[/bold red]: {failing_metrics}",
                    title="Guardrail",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# mgr certify — per-mechanism mathematical invariant checks (br: model_guided_research-5ki.1)
#
# Verifies each nanochat attention mechanism's DEFINING mathematical invariant
# directly against the production torch modules, with dtype-aware tolerances.
# The shared causality certificate (no gradient flow from the future) runs for
# every mechanism. Exit code is nonzero if any check fails.
# ---------------------------------------------------------------------------

_CERTIFY_MECHANISMS: list[str] = [
    "standard",
    "tropical",
    "ultrametric",
    "simplicial",
    "quaternion",
    "braid",
    "fractal",
    "octonion",
    "surreal",
    "reversible",
    "gauge",
]


def _certify_tiny_config(mechanism: str):
    """Build a tiny GPTConfig for `mechanism` satisfying its structural constraints."""
    from nanochat.gpt import GPTConfig

    kwargs: dict[str, Any] = dict(
        sequence_len=64,
        vocab_size=128,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=64,  # head_dim = 32: divisible by 4 (quaternion) and 8 (octonion), even (RoPE/gauge)
        attention_type=mechanism,
    )
    if mechanism == "reversible":
        # Sub-attention gets n_head//2 = 1 query head; n_kv_head must divide it.
        kwargs["n_kv_head"] = 1
    if mechanism == "tropical":
        kwargs["tropical_record_margins"] = True
    return GPTConfig(**kwargs)


def _certify_cos_sin(T: int, head_dim: int, device: Any, dtype: Any) -> tuple[Any, Any]:
    """Rotary cos/sin matching GPT._precompute_rotary_embeddings, in the requested dtype."""
    import torch

    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000 ** (channel_range / head_dim))
    t = torch.arange(T, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos().to(dtype), freqs.sin().to(dtype)
    return cos[None, :, None, :], sin[None, :, None, :]


def _run_certify_checks(
    mechanisms: list[str],
    *,
    device_str: str = "cpu",
    dtype_str: str = "fp32",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Run the invariant checks for the requested mechanisms and return one record per
    check: {mechanism, check, family, status, measured, tolerance, comparator,
    duration_ms, detail}. status is "pass" | "fail" | "error".

    Pass criterion: measured <= tolerance (comparator "le", default) or
    measured >= tolerance (comparator "ge", used for separation witnesses that
    must be bounded AWAY from zero, e.g. octonion non-associativity).
    """
    import copy

    import torch

    from nanochat.gpt import Block
    from nanochat.model_utils import apply_rotary_emb, causal_attn_mask
    from nanochat.model_utils import norm as rmsnorm

    device = torch.device(device_str)
    if dtype_str not in {"fp32", "bf16"}:
        raise ValueError(f"dtype must be 'fp32' or 'bf16', got {dtype_str!r}")
    dtype = torch.float32 if dtype_str == "fp32" else torch.bfloat16

    def fp_tol(base: float) -> float:
        # bf16 has ~8 bits of mantissa; loosen float-comparison tolerances accordingly.
        return base if dtype_str == "fp32" else max(base * 512.0, 5e-2)

    checks: list[dict[str, Any]] = []

    def add_check(
        mechanism: str,
        name: str,
        family: str,
        fn: Any,
        *,
        tolerance: float,
        comparator: str = "le",
        detail: str = "",
    ) -> None:
        t0 = time.perf_counter()
        try:
            measured = float(fn())
            if comparator == "le":
                status = "pass" if measured <= tolerance else "fail"
            else:
                status = "pass" if measured >= tolerance else "fail"
            error = None
        except Exception as exc:  # noqa: BLE001 - report, don't crash the suite
            measured = float("nan")
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        duration_ms = (time.perf_counter() - t0) * 1000.0
        checks.append(
            {
                "mechanism": mechanism,
                "check": name,
                "family": family,
                "status": status,
                "measured": measured,
                "tolerance": tolerance,
                "comparator": comparator,
                "duration_ms": round(duration_ms, 3),
                "detail": detail if error is None else f"{detail} [{error}]".strip(),
            }
        )

    def make_block(mechanism: str) -> Any:
        torch.manual_seed(seed)
        cfg = _certify_tiny_config(mechanism)
        block = Block(cfg, 0).to(device=device, dtype=dtype)
        block.eval()
        return block, cfg

    # ----- shared causality certificate (the single most important check) -----
    def causality_measure(mechanism: str) -> float:
        block, cfg = make_block(mechanism)
        T = 8
        head_dim = cfg.n_embd // cfg.n_head
        cos_sin = _certify_cos_sin(T, head_dim, device, dtype)
        x = torch.randn(1, T, cfg.n_embd, device=device, dtype=dtype, requires_grad=True)
        y = block(x, cos_sin, None)
        worst = 0.0
        past_signal = 0.0
        for t in (0, T // 2, T - 2):
            (g,) = torch.autograd.grad(y[0, t].sum(), x, retain_graph=True, allow_unused=True)
            if g is None:
                continue  # no dependence on x at all: past_signal stays 0 -> vacuity guard trips below
            if t + 1 < T:
                worst = max(worst, float(g[0, t + 1 :].abs().max()))
            past_signal = max(past_signal, float(g[0, : t + 1].abs().max()))
        if past_signal == 0.0:
            # Vacuity guard: an output that depends on NO input would trivially satisfy
            # the future-gradient condition. That is a broken mechanism, not a causal one.
            return float("inf")
        return worst

    for mech in mechanisms:
        add_check(
            mech,
            "causality_no_future_grad",
            "causality",
            lambda m=mech: causality_measure(m),
            tolerance=1e-12,
            detail=(
                "max |d y[t] / d x[t+k]| over t in {0, T/2, T-2}, k >= 1 (must be exactly 0); "
                "inf = vacuity guard tripped (output has no input dependence at all)"
            ),
        )

    # ----- standard: scaffolding invariants -----
    if "standard" in mechanisms:

        def rope_norm_measure() -> float:
            torch.manual_seed(seed)
            T, H, D = 8, 2, 32
            cos, sin = _certify_cos_sin(T, D, device, torch.float32)
            x = torch.randn(1, T, H, D, device=device)
            y = apply_rotary_emb(x, cos, sin)
            d = D // 2
            pn_in = torch.sqrt(x[..., :d] ** 2 + x[..., d:] ** 2)
            pn_out = torch.sqrt(y[..., :d] ** 2 + y[..., d:] ** 2)
            return float((pn_in - pn_out).abs().max())

        def mask_structure_measure() -> float:
            mismatches = 0
            m = causal_attn_mask(8, 8, device=device)
            mismatches += int((m != torch.tril(torch.ones(8, 8, dtype=torch.bool, device=device))).sum())
            m = causal_attn_mask(1, 9, device=device)
            mismatches += int((~m).sum())
            m = causal_attn_mask(4, 12, device=device)
            expect = torch.zeros(4, 12, dtype=torch.bool, device=device)
            expect[:, :8] = True
            expect[:, 8:] = torch.tril(torch.ones(4, 4, dtype=torch.bool, device=device))
            mismatches += int((m != expect).sum())
            return float(mismatches)

        def rmsnorm_measure() -> float:
            torch.manual_seed(seed)
            x = torch.randn(4, 8, 64, device=device)
            y = rmsnorm(x)
            rms = torch.sqrt((y**2).mean(dim=-1))
            return float((rms - 1.0).abs().max())

        def softmax_rows_measure() -> float:
            torch.manual_seed(seed)
            scores = torch.randn(1, 2, 8, 8, device=device)
            scores = scores.masked_fill(~causal_attn_mask(8, 8, device=device), float("-inf"))
            p = torch.softmax(scores, dim=-1)
            return float((p.sum(dim=-1) - 1.0).abs().max())

        add_check("standard", "rope_pairwise_norm_preservation", "classical", rope_norm_measure, tolerance=1e-5)
        add_check(
            "standard",
            "causal_mask_structure",
            "classical",
            mask_structure_measure,
            tolerance=0.0,
            detail="mismatch count vs spec across train/decode/chunk regimes",
        )
        add_check("standard", "rmsnorm_unit_rms", "classical", rmsnorm_measure, tolerance=1e-3)
        add_check("standard", "softmax_row_stochastic", "classical", softmax_rows_measure, tolerance=1e-6)

    # ----- tropical: 1-Lipschitz, gauge invariance, margin consistency -----
    if "tropical" in mechanisms:
        from nanochat.tropical_attention_torch import tropical_max_plus_attention

        def _trop_inputs():
            torch.manual_seed(seed)
            q = torch.randn(1, 2, 8, 32, device=device)
            k = torch.randn(1, 2, 8, 32, device=device)
            v = torch.randn(1, 2, 8, 32, device=device)
            return q, k, v

        def trop_lipschitz_measure(which: str) -> float:
            q, k, v = _trop_inputs()
            eps = 0.123
            y0, _ = tropical_max_plus_attention(q, k, v, gauge_fix=False, score_center=False, return_margins=False)
            delta = torch.empty_like(q if which == "q" else v).uniform_(-eps, eps)
            if which == "q":
                y1, _ = tropical_max_plus_attention(
                    q + delta, k, v, gauge_fix=False, score_center=False, return_margins=False
                )
            else:
                y1, _ = tropical_max_plus_attention(
                    q, k, v + delta, gauge_fix=False, score_center=False, return_margins=False
                )
            return float((y1 - y0).abs().max() / eps)

        def trop_center_invariance_measure() -> float:
            q, k, v = _trop_inputs()
            y0, _ = tropical_max_plus_attention(q, k, v, gauge_fix=False, score_center=False, return_margins=False)
            y1, _ = tropical_max_plus_attention(q, k, v, gauge_fix=False, score_center=True, return_margins=False)
            diff = y1 - y0  # must be constant across the feature dim (a pure per-query gauge shift)
            spread = diff.amax(dim=-1) - diff.amin(dim=-1)
            return float(spread.abs().max())

        def trop_margin_measure() -> float:
            q, k, v = _trop_inputs()
            _, gamma = tropical_max_plus_attention(q, k, v, gauge_fix=False, score_center=False, return_margins=True)
            if gamma is None:
                raise RuntimeError("tropical_max_plus_attention returned no margins despite return_margins=True")
            scores = torch.max(q.unsqueeze(3) + k.unsqueeze(2), dim=-1).values
            scores = scores.masked_fill(~causal_attn_mask(8, 8, device=device), float("-inf"))
            logits = scores.unsqueeze(-1) + v.unsqueeze(2)
            srt, _ = torch.sort(logits, dim=3, descending=True)
            gamma_bf = (srt[..., 0, :] - srt[..., 1, :]).amin(dim=-1)
            both_finite = torch.isfinite(gamma) & torch.isfinite(gamma_bf)
            inf_mismatch = float((torch.isfinite(gamma) != torch.isfinite(gamma_bf)).sum())
            max_diff = float((gamma - gamma_bf).abs().masked_fill(~both_finite, 0.0).max())
            return max_diff + inf_mismatch

        add_check(
            "tropical",
            "lipschitz_1_sup_norm_q",
            "classical",
            lambda: trop_lipschitz_measure("q"),
            tolerance=1.0 + 1e-6,
            detail="sup-norm output change / eps under q-perturbation",
        )
        add_check(
            "tropical",
            "lipschitz_1_sup_norm_v",
            "classical",
            lambda: trop_lipschitz_measure("v"),
            tolerance=1.0 + 1e-6,
            detail="sup-norm output change / eps under v-perturbation",
        )
        add_check(
            "tropical",
            "score_center_pure_gauge_shift",
            "classical",
            trop_center_invariance_measure,
            tolerance=1e-5,
            detail="centering must shift outputs by a per-query constant only",
        )
        add_check(
            "tropical",
            "margin_matches_bruteforce",
            "classical",
            trop_margin_measure,
            tolerance=1e-6,
            detail="recorded gamma vs sort-based runner-up gap (+ count of inf mismatches)",
        )

    # ----- quaternion: algebra laws + rotor norm preservation (fp64) -----
    if "quaternion" in mechanisms:
        from nanochat.quaternion_attention_torch import qconj, qmul, qnormalize

        def _quats() -> tuple[Any, Any, Any]:
            torch.manual_seed(seed)
            a = torch.randn(512, 4, dtype=torch.float64, device=device)
            b = torch.randn(512, 4, dtype=torch.float64, device=device)
            c = torch.randn(512, 4, dtype=torch.float64, device=device)
            return a, b, c

        def q_assoc_measure() -> float:
            a, b, c = _quats()
            return float((qmul(qmul(a, b), c) - qmul(a, qmul(b, c))).abs().max())

        def q_norm_mult_measure() -> float:
            a, b, _ = _quats()
            lhs = qmul(a, b).norm(dim=-1)
            return float((lhs - a.norm(dim=-1) * b.norm(dim=-1)).abs().max())

        def q_conj_antihom_measure() -> float:
            a, b, _ = _quats()
            return float((qconj(qmul(a, b)) - qmul(qconj(b), qconj(a))).abs().max())

        def q_rotor_norm_measure() -> float:
            a, b, _ = _quats()
            rotor = qnormalize(a)
            return float((qmul(rotor, b).norm(dim=-1) - b.norm(dim=-1)).abs().max())

        add_check("quaternion", "qmul_associativity", "classical", q_assoc_measure, tolerance=1e-10)
        add_check("quaternion", "qmul_norm_multiplicative", "classical", q_norm_mult_measure, tolerance=1e-10)
        add_check("quaternion", "qconj_antihomomorphism", "classical", q_conj_antihom_measure, tolerance=1e-10)
        add_check("quaternion", "rotor_norm_preservation", "classical", q_rotor_norm_measure, tolerance=1e-10)

    # ----- octonion: division-algebra laws + non-associativity witness (fp64) -----
    if "octonion" in mechanisms:
        from nanochat.octonion_attention_torch import oconj, omul

        def _octs() -> tuple[Any, Any, Any]:
            torch.manual_seed(seed)
            a = torch.randn(512, 8, dtype=torch.float64, device=device)
            b = torch.randn(512, 8, dtype=torch.float64, device=device)
            c = torch.randn(512, 8, dtype=torch.float64, device=device)
            return a, b, c

        def o_norm_mult_measure() -> float:
            a, b, _ = _octs()
            return float((omul(a, b).norm(dim=-1) - a.norm(dim=-1) * b.norm(dim=-1)).abs().max())

        def o_alternativity_measure() -> float:
            a, b, _ = _octs()
            left = (omul(a, omul(a, b)) - omul(omul(a, a), b)).abs().max()
            right = (omul(omul(b, a), a) - omul(b, omul(a, a))).abs().max()
            return float(torch.maximum(left, right))

        def o_nonassoc_witness_measure() -> float:
            a, b, c = _octs()
            return float((omul(omul(a, b), c) - omul(a, omul(b, c))).abs().max())

        def o_conj_scalar_measure() -> float:
            a, _, _ = _octs()
            prod = omul(a, oconj(a))
            scalar_err = (prod[..., 0] - a.norm(dim=-1) ** 2).abs().max()
            imag_err = prod[..., 1:].abs().max()
            return float(torch.maximum(scalar_err, imag_err))

        add_check("octonion", "omul_norm_multiplicative", "classical", o_norm_mult_measure, tolerance=1e-9)
        add_check("octonion", "omul_alternativity", "classical", o_alternativity_measure, tolerance=1e-9)
        add_check(
            "octonion",
            "omul_nonassociativity_witness",
            "classical",
            o_nonassoc_witness_measure,
            tolerance=1e-2,
            comparator="ge",
            detail="associator must be bounded AWAY from zero (guards against an associative shortcut)",
        )
        add_check("octonion", "o_times_conj_is_norm_squared", "classical", o_conj_scalar_measure, tolerance=1e-9)

    # ----- reversible: round-trip + custom-autograd gradient parity -----
    if "reversible" in mechanisms:

        def rev_roundtrip_measure() -> float:
            block, cfg = make_block("reversible")
            sb = block.special_block
            T = 8
            cos_sin = _certify_cos_sin(T, 32, device, dtype)
            x = torch.randn(1, T, cfg.n_embd, device=device, dtype=dtype)
            with torch.no_grad():
                y = sb(x, cos_sin, None)
                xr = sb.inverse(y, cos_sin, None)
            return float((x - xr).abs().max())

        def rev_grad_parity_measure() -> float:
            from nanochat.reversible_block_torch import ReversibleFunction

            block, cfg = make_block("reversible")
            sb = block.special_block.float()
            sb_ref = copy.deepcopy(sb)
            T = 8
            cos_sin = _certify_cos_sin(T, 32, device, torch.float32)
            torch.manual_seed(seed + 1)
            x = torch.randn(1, T, cfg.n_embd, device=device)
            w = torch.randn_like(x)

            xa = x.clone().requires_grad_(True)
            loss_a = (sb_ref(xa, cos_sin, None) * w).sum()
            grads_a = torch.autograd.grad(loss_a, [xa, *sb_ref.parameters()])

            xb = x.clone().requires_grad_(True)
            yb = ReversibleFunction.apply(xb, cos_sin, None, sb.f_block, sb.g_block)
            (yb * w).sum().backward()
            grads_b = [xb.grad, *(p.grad for p in sb.parameters())]

            worst = 0.0
            for ga, gb in zip(grads_a, grads_b, strict=True):
                if gb is None:
                    return float("inf")
                denom = float(ga.abs().max()) + 1e-8
                worst = max(worst, float((ga - gb).abs().max()) / denom)
            return worst

        add_check(
            "reversible",
            "forward_inverse_roundtrip",
            "classical",
            rev_roundtrip_measure,
            tolerance=fp_tol(1e-5),
            detail="max |x - inverse(forward(x))|",
        )
        add_check(
            "reversible",
            "custom_autograd_grad_parity",
            "classical",
            rev_grad_parity_measure,
            tolerance=1e-4,
            detail="max relative grad diff: ReversibleFunction vs naive autograd",
        )

    # ----- gauge: rotation orthogonality, inverse consistency, additivity -----
    if "gauge" in mechanisms:

        def _gauge_setup() -> tuple[Any, Any, Any]:
            block, cfg = make_block("gauge")
            gb = block.special_block.float()
            torch.manual_seed(seed + 2)
            x = torch.randn(1, 8, cfg.n_embd, device=device)
            th = torch.randn(1, 8, cfg.n_embd // 2, device=device)
            return gb, x, th

        def gauge_roundtrip_measure() -> float:
            gb, x, th = _gauge_setup()
            xr = gb._apply_rotations(gb._apply_rotations(x, th), th, inverse=True)
            return float((x - xr).abs().max())

        def gauge_norm_measure() -> float:
            gb, x, th = _gauge_setup()
            y = gb._apply_rotations(x, th)
            pn_in = torch.sqrt(x[..., 0::2] ** 2 + x[..., 1::2] ** 2)
            pn_out = torch.sqrt(y[..., 0::2] ** 2 + y[..., 1::2] ** 2)
            return float((pn_in - pn_out).abs().max())

        def gauge_additivity_measure() -> float:
            gb, x, th = _gauge_setup()
            torch.manual_seed(seed + 12)
            th2 = torch.randn_like(th)
            y_seq = gb._apply_rotations(gb._apply_rotations(x, th), th2)
            y_sum = gb._apply_rotations(x, th + th2)
            return float((y_seq - y_sum).abs().max())

        add_check(
            "gauge",
            "rotation_inverse_roundtrip",
            "classical",
            gauge_roundtrip_measure,
            tolerance=1e-5,
            detail="R(-theta) R(theta) = I (orthogonality of the Givens transport)",
        )
        add_check("gauge", "rotation_pairwise_norm_preservation", "classical", gauge_norm_measure, tolerance=1e-5)
        add_check(
            "gauge",
            "rotation_additivity_cumsum_law",
            "classical",
            gauge_additivity_measure,
            tolerance=1e-5,
            detail="R(t2) R(t1) = R(t1+t2): the property that justifies cumsum-as-transport",
        )

    # ----- ultrametric: strong triangle inequality on hard digits (exact) -----
    if "ultrametric" in mechanisms:

        def ultra_triangle_measure() -> float:
            block, _cfg = make_block("ultrametric")
            attn = block.attn.float()
            torch.manual_seed(seed + 3)
            feats = torch.randn(1, 2, 12, 32, device=device)
            digits = attn._digits_hard_int(attn.to_digits_q(feats))  # (1, 2, 12, K) integer digits
            d = digits[0, 0]  # (12, K)
            n = d.size(0)
            matches = (d.unsqueeze(0) == d.unsqueeze(1)).to(torch.int64)  # (n, n, K)
            lcp = matches.cumprod(dim=-1).sum(dim=-1)  # (n, n) exact LCP depths
            worst = 0
            for i in range(n):
                for j in range(n):
                    for kk in range(n):
                        lower = min(int(lcp[i, j]), int(lcp[j, kk]))
                        worst = max(worst, lower - int(lcp[i, kk]))
            return float(worst)

        add_check(
            "ultrametric",
            "strong_triangle_inequality_lcp",
            "classical",
            ultra_triangle_measure,
            tolerance=0.0,
            detail="lcp(x,z) >= min(lcp(x,y), lcp(y,z)) over all triples (exact, integer)",
        )

    # ----- braid: YBE law + restricted-law separation + payload invariance -----
    if "braid" in mechanisms:
        from nanochat.braid_attention_torch import BraidCausalSelfAttention as _Braid

        def _braid_strands() -> list[Any]:
            torch.manual_seed(seed + 4)
            return [torch.randn(64, 8, dtype=torch.float64, device=device) for _ in range(6)]

        def _ybe_residual(update: Any) -> float:
            ax, ay, bx, by, cx, cy = _braid_strands()

            def apply12(ax, ay, bx, by, cx, cy):
                nax, nay, nbx, nby = update(ax, ay, bx, by)
                return nax, nay, nbx, nby, cx, cy

            def apply23(ax, ay, bx, by, cx, cy):
                nbx, nby, ncx, ncy = update(bx, by, cx, cy)
                return ax, ay, nbx, nby, ncx, ncy

            lhs = apply12(*apply23(*apply12(ax, ay, bx, by, cx, cy)))
            rhs = apply23(*apply12(*apply23(ax, ay, bx, by, cx, cy)))
            return float(torch.max(torch.abs(torch.stack(lhs, dim=-1) - torch.stack(rhs, dim=-1))))

        def braid_payload_measure() -> float:
            ax, ay, bx, by, _, _ = _braid_strands()
            worst = 0.0
            for update in (_Braid._crossing_update_restricted, _Braid._crossing_update_ybe):
                _, nay, _, nby = update(ax, ay, bx, by)
                inp = torch.sort(torch.stack([ay, by], dim=0), dim=0).values
                out = torch.sort(torch.stack([nay, nby], dim=0), dim=0).values
                worst = max(worst, float((inp - out).abs().max()))
            return worst

        add_check(
            "braid",
            "ybe_law_holds",
            "classical",
            lambda: _ybe_residual(_Braid._crossing_update_ybe),
            tolerance=1e-10,
            detail="R3 residual for the swap-output crossing law",
        )
        add_check(
            "braid",
            "restricted_law_violates_ybe",
            "classical",
            lambda: _ybe_residual(_Braid._crossing_update_restricted),
            tolerance=1e-3,
            comparator="ge",
            detail="separation witness: the heuristic law must NOT satisfy YBE (proves the test has teeth)",
        )
        add_check(
            "braid",
            "payload_multiset_invariance",
            "classical",
            braid_payload_measure,
            tolerance=0.0,
            detail="crossings must preserve the payload multiset {y} exactly",
        )

    # ----- simplicial: mass conservation through 1-hop and 2-hop paths -----
    if "simplicial" in mechanisms:

        def simplicial_mass_measure() -> float:
            torch.manual_seed(seed + 5)
            scores = torch.randn(1, 2, 8, 8, device=device)
            scores = scores.masked_fill(~causal_attn_mask(8, 8, device=device), float("-inf"))
            att = torch.softmax(scores, dim=-1)
            c = 0.731
            v = torch.full((1, 2, 8, 32), c, device=device)
            y1 = att @ v
            y2 = att @ y1
            return float(torch.maximum((y1 - c).abs().max(), (y2 - c).abs().max()))

        add_check(
            "simplicial",
            "mass_conservation_two_hop",
            "classical",
            simplicial_mass_measure,
            tolerance=1e-5,
            detail="row-stochastic 1-hop and 2-hop aggregation must preserve constants",
        )

    # ----- fractal: router branch distributions are simplex-valued -----
    if "fractal" in mechanisms:

        def fractal_router_measure() -> float:
            import torch.nn.functional as t_func

            block, _cfg = make_block("fractal")
            attn = block.attn.float()
            torch.manual_seed(seed + 6)
            feats = torch.randn(1, 2, 8, 32, device=device)
            route = attn.router(feats).view(1, 2, 8, attn.depth, attn.m)
            probs = t_func.softmax(route, dim=-1)
            sums_err = (probs.sum(dim=-1) - 1.0).abs().max()
            neg_err = (-probs).clamp_min(0.0).max()
            return float(torch.maximum(sums_err, neg_err))

        add_check(
            "fractal",
            "router_branch_simplex",
            "classical",
            fractal_router_measure,
            tolerance=1e-6,
            detail="per-depth branch distributions must lie on the m-simplex",
        )

    # ----- surreal: scale/direction decomposition laws -----
    if "surreal" in mechanisms:
        from nanochat.surreal_torch import SurrealLayer

        def _surreal_layer() -> Any:
            torch.manual_seed(seed + 7)
            return SurrealLayer(64, 64, bias=False).to(device=device).float()

        def surreal_norm_measure() -> float:
            import torch.nn.functional as t_func

            layer = _surreal_layer()
            with torch.no_grad():
                layer.weight_s.uniform_(-1.0, 1.0)
                w = torch.exp(layer.weight_s) * t_func.normalize(layer.weight_v, dim=1)
                return float((w.norm(dim=1, keepdim=True) - torch.exp(layer.weight_s)).abs().max())

        def surreal_linearity_measure() -> float:
            layer = _surreal_layer()
            torch.manual_seed(seed + 8)
            x = torch.randn(4, 64, device=device)
            c = 3.7
            with torch.no_grad():
                return float((layer(c * x) - c * layer(x)).abs().max())

        def surreal_s_shift_measure() -> float:
            layer = _surreal_layer()
            torch.manual_seed(seed + 9)
            x = torch.randn(4, 64, device=device)
            delta = 0.9
            with torch.no_grad():
                y0 = layer(x)
                layer.weight_s.add_(delta)
                y1 = layer(x)
                return float((y1 - math.exp(delta) * y0).abs().max())

        add_check(
            "surreal",
            "row_norm_equals_exp_scale",
            "classical",
            surreal_norm_measure,
            tolerance=1e-5,
            detail="||w_row|| = exp(s_row): the scale/direction decomposition",
        )
        add_check("surreal", "layer_linearity", "classical", surreal_linearity_measure, tolerance=1e-4)
        add_check(
            "surreal",
            "scale_shift_equivariance",
            "classical",
            surreal_s_shift_measure,
            tolerance=1e-4,
            detail="s -> s + d must scale outputs by exp(d) exactly",
        )

    return checks


@app.command()
def certify(
    mechanism: Annotated[
        list[str] | None,
        typer.Option("--mechanism", "-m", help="Mechanism(s) to certify (default: all)"),
    ] = None,
    device: Annotated[str, typer.Option(help="Device: cpu | cuda")] = "cpu",
    dtype: Annotated[str, typer.Option(help="Dtype for module-level checks: fp32 | bf16")] = "fp32",
    seed: Annotated[int, typer.Option(help="Seed for all randomized checks")] = 42,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show per-check detail column")] = False,
    artifacts_dir: Annotated[Path | None, typer.Option(help="Write summary.json/run.md under this dir")] = None,
    run_id: Annotated[str | None, typer.Option(help="Run id (default: timestamp)")] = None,
):
    """
    Certify the mathematical invariants of the nanochat attention mechanisms.

    Runs the shared causality certificate for every mechanism plus per-mechanism
    invariant checks (Lipschitz bounds, algebra laws, round-trips, conservation).
    Exits nonzero if any check fails.
    """
    mechanisms = list(mechanism) if mechanism else list(_CERTIFY_MECHANISMS)
    unknown = [m for m in mechanisms if m not in _CERTIFY_MECHANISMS]
    if unknown:
        console.print(f"[bold red]Unknown mechanism(s):[/bold red] {unknown}. Valid: {_CERTIFY_MECHANISMS}")
        raise typer.Exit(code=2)

    git_info = _get_git_info()
    rid = run_id or _default_run_id()
    console.print(
        Panel(
            f"mechanisms: [cyan]{', '.join(mechanisms)}[/cyan]\n"
            f"device: [cyan]{device}[/cyan] · dtype: [cyan]{dtype}[/cyan] · seed: [cyan]{seed}[/cyan]\n"
            f"git: [cyan]{git_info['commit']}[/cyan]{' (dirty)' if git_info['dirty'] else ''}",
            title="[bold]mgr certify — mathematical invariant checks[/bold]",
            border_style="blue",
        )
    )

    t_start = time.perf_counter()
    checks = _run_certify_checks(mechanisms, device_str=device, dtype_str=dtype, seed=seed)
    total_s = time.perf_counter() - t_start

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Mechanism", style="cyan")
    table.add_column("Check")
    table.add_column("Family", style="dim")
    table.add_column("Status")
    table.add_column("Measured", justify="right")
    table.add_column("Tolerance", justify="right")
    table.add_column("ms", justify="right", style="dim")
    if verbose:
        table.add_column("Detail", style="dim", max_width=60)
    n_pass = n_fail = n_error = 0
    for c in checks:
        if c["status"] == "pass":
            n_pass += 1
            status = "[green]PASS[/green]"
        elif c["status"] == "fail":
            n_fail += 1
            status = "[bold red]FAIL[/bold red]"
        else:
            n_error += 1
            status = "[bold red]ERROR[/bold red]"
        cmp_sym = "<=" if c["comparator"] == "le" else ">="
        row = [
            c["mechanism"],
            c["check"],
            c["family"],
            status,
            f"{c['measured']:.3e}",
            f"{cmp_sym} {c['tolerance']:.3e}",
            f"{c['duration_ms']:.1f}",
        ]
        if verbose:
            row.append(c["detail"])
        table.add_row(*row)
    console.print(table)

    color = "green" if (n_fail + n_error) == 0 else "red"
    console.print(
        Panel(
            f"[bold {color}]{n_pass} passed · {n_fail} failed · {n_error} errored[/bold {color}]"
            f" · {len(checks)} checks in {total_s:.1f}s",
            border_style=color,
        )
    )
    for c in checks:
        if c["status"] != "pass":
            cmp_sym = "<=" if c["comparator"] == "le" else ">="
            console.print(
                f"[red]✗ {c['mechanism']}.{c['check']}[/red] measured={c['measured']:.3e} "
                f"({cmp_sym} {c['tolerance']:.3e} required) — "
                f"repro: [bold]mgr certify -m {c['mechanism']} --seed {seed} --device {device} --dtype {dtype}[/bold]"
                + (f"\n  [dim]{c['detail']}[/dim]" if c["detail"] else "")
            )

    if artifacts_dir is not None:
        run_dir = artifacts_dir / "certs" / "nanochat" / rid
        summary = {
            "schema_version": 1,
            "kind": "certify",
            "run_id": rid,
            "seed": seed,
            "device": device,
            "dtype": dtype,
            "git": git_info,
            "mechanisms": mechanisms,
            "checks": checks,
            "counts": {"pass": n_pass, "fail": n_fail, "error": n_error},
            "duration_s": round(total_s, 3),
        }
        lines = [
            "# mgr certify",
            "",
            f"- run_id: `{rid}` · seed: {seed} · device: {device} · dtype: {dtype}",
            f"- git: `{git_info['commit']}`{' (dirty)' if git_info['dirty'] else ''}",
            f"- result: **{n_pass} passed / {n_fail} failed / {n_error} errored** in {total_s:.1f}s",
            "",
            "| Mechanism | Check | Family | Status | Measured | Tolerance | ms |",
            "|---|---|---|---|---:|---:|---:|",
        ]
        for c in checks:
            cmp_sym = "<=" if c["comparator"] == "le" else ">="
            lines.append(
                f"| {c['mechanism']} | {c['check']} | {c['family']} | {c['status']} "
                f"| {c['measured']:.3e} | {cmp_sym} {c['tolerance']:.3e} | {c['duration_ms']:.1f} |"
            )
        _write_artifacts(run_dir, summary=summary, report_md="\n".join(lines) + "\n")
        console.print(f"[bold green]Wrote certificate artifacts[/bold green] → {run_dir}")

    if n_fail + n_error > 0:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# mgr fuzz — numerical robustness sweep (br: model_guided_research-5ki.4)
#
# Stresses every mechanism across dtypes, input scales, sequence-length
# boundaries, and adversarial input patterns, recording NaN/Inf in outputs AND
# gradients. Complements `mgr certify` (invariants on benign inputs): fuzz asks
# what happens on the inputs nobody writes unit tests for.
# ---------------------------------------------------------------------------

_FUZZ_PATTERNS: list[str] = ["randn", "all_equal", "alternating", "zeros"]


def _fuzz_make_input(pattern: str, *, B: int, T: int, C: int, scale: float, device: Any, dtype: Any) -> Any:
    import torch

    if pattern == "randn":
        x = torch.randn(B, T, C, device=device, dtype=torch.float32) * scale
    elif pattern == "all_equal":
        # Every token identical and every channel equal: maximal ties in every argmax/softmax.
        x = torch.full((B, T, C), 1.0, device=device, dtype=torch.float32) * scale
    elif pattern == "alternating":
        # Extreme +/- cancellation pattern along both time and channels.
        t_sign = (-1.0) ** torch.arange(T, device=device, dtype=torch.float32)
        c_sign = (-1.0) ** torch.arange(C, device=device, dtype=torch.float32)
        x = (t_sign.view(1, T, 1) * c_sign.view(1, 1, C)).expand(B, T, C).clone() * scale
    elif pattern == "zeros":
        # Zero-norm everything: rmsnorm of zeros, zero-norm quaternions/octonions, etc.
        x = torch.zeros(B, T, C, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown fuzz pattern: {pattern!r}")
    return x.to(dtype=dtype)


def _run_fuzz_cells(
    mechanisms: list[str],
    *,
    device_str: str = "cpu",
    dtypes: list[str] | None = None,
    scales: list[float] | None = None,
    lengths: list[int] | None = None,
    patterns: list[str] | None = None,
    seed: int = 42,
    warn_out_abs: float = 1e4,
    warn_grad_abs: float = 1e6,
    progress: Any = None,
) -> list[dict[str, Any]]:
    """
    Run the robustness sweep and return one record per cell:
    {mechanism, dtype, scale, T, pattern, status, out_nan_inf, grad_nan_inf,
     max_abs_out, max_abs_grad, duration_ms, recipe}.

    status: "fail" if any NaN/Inf appears in the output or input-gradient;
    "warn" if magnitudes exceed the warn thresholds; "error" on exception;
    else "pass". The `recipe` string is a complete reproduction command.

    Note: the `zeros` pattern is scale-invariant, so it runs once per
    (mechanism, dtype, T) regardless of the scales list.
    """
    import torch

    from nanochat.gpt import Block

    device = torch.device(device_str)
    dtypes = dtypes if dtypes is not None else ["fp32", "bf16"]
    scales = scales if scales is not None else [1e-3, 1.0, 1e3]
    # Boundary lengths around the tiny-config block size (1, 2, just-below, at).
    lengths = lengths if lengths is not None else [1, 2, 63, 64]
    patterns = patterns if patterns is not None else list(_FUZZ_PATTERNS)

    records: list[dict[str, Any]] = []
    for mech in mechanisms:
        for dtype_str in dtypes:
            dtype = torch.float32 if dtype_str == "fp32" else torch.bfloat16
            for T in lengths:
                for pattern in patterns:
                    cell_scales = [1.0] if pattern == "zeros" else scales
                    for scale in cell_scales:
                        t0 = time.perf_counter()
                        recipe = (
                            f"mechanism={mech} dtype={dtype_str} scale={scale:g} T={T} pattern={pattern} seed={seed}"
                        )
                        rec: dict[str, Any] = {
                            "mechanism": mech,
                            "dtype": dtype_str,
                            "scale": scale,
                            "T": T,
                            "pattern": pattern,
                            "recipe": recipe,
                        }
                        try:
                            torch.manual_seed(seed)
                            cfg = _certify_tiny_config(mech)
                            block = Block(cfg, 0).to(device=device, dtype=dtype)
                            block.eval()
                            head_dim = cfg.n_embd // cfg.n_head
                            cos_sin = _certify_cos_sin(T, head_dim, device, dtype)
                            x = _fuzz_make_input(
                                pattern, B=2, T=T, C=cfg.n_embd, scale=scale, device=device, dtype=dtype
                            )
                            x.requires_grad_(True)
                            y = block(x, cos_sin, None)
                            (g,) = torch.autograd.grad(y.float().sum(), x, allow_unused=True)
                            y_f = y.detach().float()
                            out_bad = int((~torch.isfinite(y_f)).sum())
                            if g is None:
                                grad_bad = 0
                                max_grad = 0.0
                            else:
                                g_f = g.detach().float()
                                grad_bad = int((~torch.isfinite(g_f)).sum())
                                max_grad = float(g_f.abs().max()) if g_f.numel() else 0.0
                            max_out = float(y_f.abs().max()) if y_f.numel() else 0.0
                            rec.update(
                                out_nan_inf=out_bad,
                                grad_nan_inf=grad_bad,
                                max_abs_out=max_out,
                                max_abs_grad=max_grad,
                            )
                            if out_bad or grad_bad:
                                rec["status"] = "fail"
                            elif max_out > warn_out_abs or max_grad > warn_grad_abs:
                                rec["status"] = "warn"
                            else:
                                rec["status"] = "pass"
                        except Exception as exc:  # noqa: BLE001 - a crash is a finding, not a harness bug
                            rec.update(
                                status="error",
                                out_nan_inf=-1,
                                grad_nan_inf=-1,
                                max_abs_out=float("nan"),
                                max_abs_grad=float("nan"),
                                detail=f"{type(exc).__name__}: {exc}",
                            )
                        rec["duration_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
                        records.append(rec)
                        if progress is not None:
                            progress()
    return records


@app.command()
def fuzz(
    mechanism: Annotated[
        list[str] | None,
        typer.Option("--mechanism", "-m", help="Mechanism(s) to fuzz (default: all)"),
    ] = None,
    device: Annotated[str, typer.Option(help="Device: cpu | cuda")] = "cpu",
    seed: Annotated[int, typer.Option(help="Seed for randomized patterns")] = 42,
    artifacts_dir: Annotated[Path | None, typer.Option(help="Write summary.json/run.md under this dir")] = None,
    run_id: Annotated[str | None, typer.Option(help="Run id (default: timestamp)")] = None,
    fail_on_warn: Annotated[bool, typer.Option(help="Treat warns as failures for the exit code")] = False,
):
    """
    Numerical robustness fuzz: dtype x scale x boundary-length x adversarial-pattern
    sweep over the attention mechanisms, recording NaN/Inf in outputs and gradients.

    Every non-pass cell prints its full reproduction recipe. Exit code 1 if any
    cell fails (or warns, with --fail-on-warn).
    """
    mechanisms = list(mechanism) if mechanism else list(_CERTIFY_MECHANISMS)
    unknown = [m for m in mechanisms if m not in _CERTIFY_MECHANISMS]
    if unknown:
        console.print(f"[bold red]Unknown mechanism(s):[/bold red] {unknown}. Valid: {_CERTIFY_MECHANISMS}")
        raise typer.Exit(code=2)

    git_info = _get_git_info()
    rid = run_id or _default_run_id()
    n_cells_per_mech = 2 * len([1, 2, 63, 64]) * (3 * 3 + 1)  # dtypes * lengths * (3 patterns x 3 scales + zeros)
    total_cells = n_cells_per_mech * len(mechanisms)
    console.print(
        Panel(
            f"mechanisms: [cyan]{', '.join(mechanisms)}[/cyan]\n"
            f"grid: dtype {{fp32, bf16}} x scale {{1e-3, 1, 1e3}} x T {{1, 2, 63, 64}} x "
            f"pattern {{{', '.join(_FUZZ_PATTERNS)}}} = [cyan]{total_cells}[/cyan] cells\n"
            f"device: [cyan]{device}[/cyan] · seed: [cyan]{seed}[/cyan] · git: [cyan]{git_info['commit']}[/cyan]",
            title="[bold]mgr fuzz — numerical robustness sweep[/bold]",
            border_style="blue",
        )
    )

    t_start = time.perf_counter()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("fuzzing", total=total_cells)
        records = _run_fuzz_cells(
            mechanisms,
            device_str=device,
            seed=seed,
            progress=lambda: prog.advance(task),
        )
    total_s = time.perf_counter() - t_start

    by_mech: dict[str, dict[str, int]] = {}
    for r in records:
        agg = by_mech.setdefault(r["mechanism"], {"pass": 0, "warn": 0, "fail": 0, "error": 0, "cells": 0})
        agg[r["status"]] += 1
        agg["cells"] += 1

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Mechanism", style="cyan")
    table.add_column("Cells", justify="right")
    table.add_column("Pass", justify="right", style="green")
    table.add_column("Warn", justify="right", style="yellow")
    table.add_column("Fail", justify="right", style="red")
    table.add_column("Error", justify="right", style="red")
    table.add_column("Worst cell", style="dim", max_width=60)
    for mech, agg in by_mech.items():
        worst = ""
        bad = [r for r in records if r["mechanism"] == mech and r["status"] in ("fail", "error")]
        if not bad:
            bad = [r for r in records if r["mechanism"] == mech and r["status"] == "warn"]
        if bad:
            worst = bad[0]["recipe"]
        table.add_row(
            mech,
            str(agg["cells"]),
            str(agg["pass"]),
            str(agg["warn"]),
            str(agg["fail"]),
            str(agg["error"]),
            worst,
        )
    console.print(table)

    n_pass = sum(1 for r in records if r["status"] == "pass")
    n_warn = sum(1 for r in records if r["status"] == "warn")
    n_fail = sum(1 for r in records if r["status"] == "fail")
    n_error = sum(1 for r in records if r["status"] == "error")
    color = "green" if (n_fail + n_error) == 0 else "red"
    console.print(
        Panel(
            f"[bold {color}]{n_pass} pass · {n_warn} warn · {n_fail} fail · {n_error} error[/bold {color}]"
            f" · {len(records)} cells in {total_s:.1f}s",
            border_style=color,
        )
    )
    for r in records:
        if r["status"] in ("fail", "error"):
            console.print(
                f"[red]✗[/red] {r['recipe']} — out_nan_inf={r['out_nan_inf']} grad_nan_inf={r['grad_nan_inf']}"
                + (f" [{r.get('detail', '')}]" if r.get("detail") else "")
            )
        elif r["status"] == "warn":
            console.print(
                f"[yellow]△[/yellow] {r['recipe']} — max|out|={r['max_abs_out']:.3e} max|grad|={r['max_abs_grad']:.3e}"
            )

    if artifacts_dir is not None:
        run_dir = artifacts_dir / "certs" / "fuzz" / rid
        summary = {
            "schema_version": 1,
            "kind": "fuzz",
            "run_id": rid,
            "seed": seed,
            "device": device,
            "git": git_info,
            "mechanisms": mechanisms,
            "counts": {"pass": n_pass, "warn": n_warn, "fail": n_fail, "error": n_error},
            "cells": records,
            "duration_s": round(total_s, 3),
        }
        lines = [
            "# mgr fuzz",
            "",
            f"- run_id: `{rid}` · seed: {seed} · device: {device} · git: `{git_info['commit']}`",
            f"- result: **{n_pass} pass / {n_warn} warn / {n_fail} fail / {n_error} error** "
            f"({len(records)} cells, {total_s:.1f}s)",
            "",
            "| Mechanism | Cells | Pass | Warn | Fail | Error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for mech, agg in by_mech.items():
            lines.append(
                f"| {mech} | {agg['cells']} | {agg['pass']} | {agg['warn']} | {agg['fail']} | {agg['error']} |"
            )
        non_pass = [r for r in records if r["status"] != "pass"]
        if non_pass:
            lines += ["", "## Non-pass cells", ""]
            for r in non_pass:
                lines.append(
                    f"- `{r['status']}` {r['recipe']} (out_nan_inf={r['out_nan_inf']}, "
                    f"grad_nan_inf={r['grad_nan_inf']}, max|out|={r['max_abs_out']:.3e})"
                )
        _write_artifacts(run_dir, summary=summary, report_md="\n".join(lines) + "\n")
        console.print(f"[bold green]Wrote fuzz artifacts[/bold green] → {run_dir}")

    if n_fail + n_error > 0 or (fail_on_warn and n_warn > 0):
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# mgr doctor — environment diagnostics (br: model_guided_research-rz8.7)
#
# Preflight for humans and agents: checks the four-way version matrix
# (Python/torch/JAX/CUDA), data presence, tokenizer, Triton, disk space, and
# runs a tiny end-to-end forward pass. Every failing row carries an actionable
# fix-it hint. Exit codes: 0 = all ok, 1 = warnings, 2 = failures.
# ---------------------------------------------------------------------------


def _run_doctor_checks() -> list[dict[str, Any]]:
    """Return one record per check: {name, status: ok|warn|fail, detail, hint}."""
    import shutil
    import sys as _sys

    rows: list[dict[str, Any]] = []

    def add(name: str, status: str, detail: str, hint: str = "") -> None:
        rows.append({"name": name, "status": status, "detail": detail, "hint": hint})

    # Python + tooling
    py = _sys.version_info
    if (py.major, py.minor) >= (3, 13):
        add("python", "ok", f"{py.major}.{py.minor}.{py.micro}")
    else:
        add("python", "fail", f"{py.major}.{py.minor}.{py.micro}", "This project targets Python 3.13+ (see AGENTS.md)")
    add(
        "uv",
        "ok" if shutil.which("uv") else "warn",
        shutil.which("uv") or "not on PATH",
        "" if shutil.which("uv") else "Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh",
    )
    in_venv = _sys.prefix != _sys.base_prefix
    add(
        "venv",
        "ok" if in_venv else "warn",
        _sys.prefix if in_venv else "no virtualenv active",
        "" if in_venv else "Run: uv venv && source .venv/bin/activate && uv sync --extra dev",
    )

    # torch
    try:
        import torch

        detail = f"torch {torch.__version__}"
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            bf16 = torch.cuda.is_bf16_supported()
            detail += f" · CUDA {torch.version.cuda} · {props.name} ({props.total_memory // (1 << 20)} MiB)"
            detail += f" · bf16={'yes' if bf16 else 'no'}"
            add("torch", "ok", detail)
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            add("torch", "ok", detail + " · MPS available (no CUDA)")
        else:
            add("torch", "warn", detail + " · CPU only", "GPU optional; nanochat training is much faster on CUDA")
    except Exception as exc:  # noqa: BLE001
        add("torch", "fail", f"{type(exc).__name__}: {exc}", "uv sync --extra dev")

    # JAX
    try:
        import jax

        from utils import get_device_info

        info = get_device_info()
        add("jax", "ok", f"jax {jax.__version__} · backend={info['default_backend']} · devices={info['devices']}")
    except Exception as exc:  # noqa: BLE001
        add("jax", "fail", f"{type(exc).__name__}: {exc}", "uv sync --extra dev (JAX powers the mgr demos)")

    # Triton (A7 kernel prerequisite; optional)
    try:
        import triton  # noqa: F401

        add("triton", "ok", f"triton {getattr(triton, '__version__', '?')}")
    except Exception:  # noqa: BLE001
        add("triton", "warn", "not importable", "Optional: needed only for custom GPU kernels (tropical Triton path)")

    # Training data (parquet shards)
    try:
        from nanochat.dataset import DATA_DIR, list_parquet_files

        files = list_parquet_files()
        if len(files) >= 2:
            add("training data", "ok", f"{len(files)} parquet shard(s) in {DATA_DIR}")
        elif len(files) == 1:
            add(
                "training data",
                "warn",
                f"only 1 parquet shard in {DATA_DIR} (train/val split needs >= 2)",
                "python -m nanochat.train --auto-download-data, or stage shards into the data dir",
            )
        else:
            add(
                "training data",
                "warn",
                f"no parquet shards in {DATA_DIR}",
                "python -m nanochat.train --auto-download-data downloads FineWeb-Edu shards",
            )
    except Exception as exc:  # noqa: BLE001
        add("training data", "warn", f"{type(exc).__name__}: {exc}", "check NANOCHAT_BASE_DIR")

    # Tokenizer
    try:
        from nanochat.tokenizer import get_tokenizer

        tok = get_tokenizer()
        n_vocab = tok.get_vocab_size() if hasattr(tok, "get_vocab_size") else "?"
        add("tokenizer", "ok", f"{type(tok).__name__} (vocab={n_vocab})")
    except Exception as exc:  # noqa: BLE001
        add(
            "tokenizer",
            "warn",
            f"{type(exc).__name__}: {exc}",
            "tokenizer assets load on first training run; check network access if this persists",
        )

    # Disk space at the nanochat cache dir
    try:
        from nanochat.common import get_base_dir

        base = get_base_dir()
        usage = shutil.disk_usage(base)
        free_gb = usage.free / (1 << 30)
        status = "ok" if free_gb >= 5.0 else "warn"
        add(
            "disk space",
            status,
            f"{free_gb:.1f} GiB free at {base}",
            "" if status == "ok" else "training data + checkpoints need headroom; free space or move NANOCHAT_BASE_DIR",
        )
    except Exception as exc:  # noqa: BLE001
        add("disk space", "warn", f"{type(exc).__name__}: {exc}", "")

    # End-to-end smoke: tiny GPT forward on the best available device
    try:
        import torch

        from nanochat.gpt import GPT, GPTConfig

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = GPTConfig(sequence_len=32, vocab_size=128, n_layer=1, n_head=2, n_kv_head=2, n_embd=64)
        model = GPT(cfg).to(device)
        # GPT.forward requires bf16 rotary buffers (asserted in forward); init_weights would
        # recompute them, but for a smoke test the constructor buffers are already bf16.
        idx = torch.randint(0, 128, (1, 16), device=device)
        with torch.no_grad():
            logits = model(idx)
        if logits.shape == (1, 16, 128) and bool(torch.isfinite(logits).all()):
            add("model smoke", "ok", f"tiny GPT forward on {device}: logits {tuple(logits.shape)}, finite")
        else:
            add("model smoke", "fail", f"unexpected logits: shape={tuple(logits.shape)}", "investigate nanochat.gpt")
    except Exception as exc:  # noqa: BLE001
        add("model smoke", "fail", f"{type(exc).__name__}: {exc}", "uv sync --extra dev; then rerun mgr doctor")

    return rows


@app.command()
def doctor(
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON only")] = False,
):
    """
    Environment diagnostics: versions, devices, data, tokenizer, disk, and a tiny
    end-to-end forward pass — with an actionable fix-it hint on every failing row.

    Exit codes: 0 = all ok, 1 = warnings present, 2 = failures present.
    """
    rows = _run_doctor_checks()
    n_warn = sum(1 for r in rows if r["status"] == "warn")
    n_fail = sum(1 for r in rows if r["status"] == "fail")

    if json_output:
        payload = {
            "schema_version": 1,
            "kind": "doctor",
            "checks": rows,
            "counts": {"ok": len(rows) - n_warn - n_fail, "warn": n_warn, "fail": n_fail},
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        table = Table(box=box.SIMPLE_HEAVY, title="mgr doctor")
        table.add_column("Check", style="cyan")
        table.add_column("Status")
        table.add_column("Detail", max_width=70)
        table.add_column("Fix hint", style="dim", max_width=50)
        style = {"ok": "[green]OK[/green]", "warn": "[yellow]WARN[/yellow]", "fail": "[bold red]FAIL[/bold red]"}
        for r in rows:
            table.add_row(r["name"], style[r["status"]], r["detail"], r["hint"])
        console.print(table)
        color = "green" if n_fail == 0 and n_warn == 0 else ("yellow" if n_fail == 0 else "red")
        console.print(
            Panel(
                f"[bold {color}]{len(rows) - n_warn - n_fail} ok · {n_warn} warn · {n_fail} fail[/bold {color}]",
                border_style=color,
            )
        )

    if n_fail > 0:
        raise typer.Exit(code=2)
    if n_warn > 0:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
