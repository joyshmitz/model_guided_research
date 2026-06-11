"""Tests for the verdict engine, mgr adjudicate (bead hij.2).

The acceptance matrix: fixture artifacts drive one hypothesis to EACH verdict
state (supported / refuted / inconclusive / blocked-for-every-reason);
verdicts are deterministic (fixed bootstrap seed -> byte-identical
verdicts.json); the ledger append survives validation, preserves hand-written
registry comments, never rewrites prior history, and rolls back atomically
when validation fails. The integrity core - REFUSING weak or tainted
evidence - is tested explicitly per the bead.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import cli

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parent.parent

CLEAN_PROV = {"schema_version": "mgr.metrics.v1", "git_sha": "deadbeef", "git_dirty": False,
              "config_hash": "abc", "data_snapshot_hash": None, "tainted": False}
TAINTED_PROV = {**CLEAN_PROV, "git_dirty": True, "tainted": True}


def _evaltasks_artifact(
    root: Path,
    name: str,
    *,
    mechanism: str,
    per_seed: list[float],
    task: str = "hier",
    target_flops: float = 2e9,
    tainted: bool = False,
    run_id: str | None = None,
    step: int = 100,
    generated_at: str = "2026-06-10T20:00:00Z",
    schema_version: str = "mgr.evaltasks.v2",
    answer_prior: float | None = None,
) -> Path:
    """One eval artifact == one trained checkpoint (ci-v2 observation unit).
    per_seed are EVAL seeds inside the artifact; the engine consumes only the
    mean. run_id defaults to the artifact name (distinct training runs)."""
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    mean = sum(per_seed) / len(per_seed)
    task_rec: dict = {
        "exact_match": {"greedy": {"held_out": {"mean": mean, "per_seed": per_seed},
                                    "in_range": {"mean": mean, "per_seed": per_seed}}},
        "perplexity": {"in_range": mean, "held_out": mean},
    }
    if answer_prior is not None:
        task_rec["answer_prior"] = {
            "held_out": {"mean": answer_prior, "per_seed": [answer_prior] * len(per_seed),
                          "majority_answer": "x"},
            "in_range": {"mean": answer_prior, "per_seed": [answer_prior] * len(per_seed),
                          "majority_answer": "x"},
        }
    summary = {
        "schema_version": schema_version,
        "kind": "eval-tasks",
        "meta": {
            "generated_at": generated_at,
            "checkpoint": {
                "attention_type": mechanism,
                "step": step,
                "budget": {"max_steps": 100, "target_flops": target_flops, "flops_per_step_est": 1e7},
                "lineage": {"run_id": run_id or name, "parent_run_ids": []},
            },
            "seeds": list(range(len(per_seed))),
        },
        "provenance": TAINTED_PROV if tainted else CLEAN_PROV,
        "tasks": {task: task_rec},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    return run_dir / "summary.json"


def _arm_artifacts(root: Path, prefix: str, mechanism: str, means: list[float], **kw) -> None:
    """One artifact per training-run mean (the ci-v2 observation unit)."""
    for i, m in enumerate(means):
        _evaltasks_artifact(root, f"{prefix}{i}", mechanism=mechanism, per_seed=[m, m, m], **kw)


def _hyp(per_pred_overrides=None, **overrides):
    pred = {
        "metric_path": "evaltasks:tasks.hier.exact_match.greedy.held_out.mean",
        "comparator": ">=",
        "threshold_kind": "absolute_delta",
        "threshold": 0.05,
        "baseline": {"mechanism": "standard", "equal_flops": True},
        "min_seeds": 3,
    }
    pred.update(per_pred_overrides or {})
    base = {
        "id": "hyp-engine-test",
        "statement": "engine test claim",
        "mechanisms": ["ultrametric"],
        "source": {"kind": "human", "provenance": "test"},
        "date_registered": "2026-06-10",
        "prediction": pred,
        "status": "open",
        "evidence": [],
        "verdict_history": [],
    }
    base.update(overrides)
    return base


def _index(root: Path):
    return cli._adj_collect_artifacts([root])


# ---------------------------------------------------------------------------
# verdict states


def test_supported_when_ci_clears_threshold(tmp_path):
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.80, 0.82, 0.81])
    _arm_artifacts(tmp_path, "base", "standard", [0.50, 0.51, 0.52])
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    assert v["verdict"] == "supported", v
    arm = v["arms"]["ultrametric"]
    assert arm["ci95"][0] >= 0.05 and abs(arm["effect"] - 0.30) < 0.02
    assert arm["n_candidate"] == 3 and arm["n_baseline"] == 3  # one obs per trained model
    assert v["policy_version"] == "ci-v2"


def test_refuted_when_ci_clears_opposite_side(tmp_path):
    # no floor info in fixture or prediction -> the gate stays out of the way
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.50, 0.51, 0.50])
    _arm_artifacts(tmp_path, "base", "standard", [0.50, 0.50, 0.51])
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    assert v["verdict"] == "refuted", v  # effect ~0, CI well below the +0.05 claim


def test_inconclusive_when_ci_straddles(tmp_path):
    # high variance across training runs: CI spans the 0.05 threshold
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.40, 0.75, 0.55])
    _arm_artifacts(tmp_path, "base", "standard", [0.45, 0.55, 0.50])
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    assert v["verdict"] == "inconclusive", v


def test_welch_t_not_normal_critical_value(tmp_path):
    """ci-v2 fix #2: at n=3 the CI must use the t critical value, not 1.96 -
    the interval is strictly wider than the Welch-normal one."""
    import statistics as stats_mod

    cand_means, base_means = [0.50, 0.55, 0.60], [0.50, 0.51, 0.52]
    _arm_artifacts(tmp_path, "cand", "ultrametric", cand_means)
    _arm_artifacts(tmp_path, "base", "standard", base_means)
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    arm = v["arms"]["ultrametric"]
    se = (stats_mod.variance(cand_means) / 3 + stats_mod.variance(base_means) / 3) ** 0.5
    half = (arm["ci95"][1] - arm["ci95"][0]) / 2
    assert half > 1.96 * se * 1.05, f"t interval must be wider than normal: half={half}, z-half={1.96 * se}"


def test_ratio_threshold_kind_supported(tmp_path):
    hyp = _hyp({"comparator": "<=", "threshold_kind": "ratio", "threshold": 0.7,
                "metric_path": "evaltasks:tasks.hier.perplexity.held_out"})
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.30, 0.31, 0.29])
    _arm_artifacts(tmp_path, "base", "standard", [0.60, 0.61, 0.59])
    v = cli._adjudicate_hypothesis(hyp, _index(tmp_path))
    assert v["verdict"] == "supported", v
    assert abs(v["arms"]["ultrametric"]["effect"] - 0.5) < 0.05


# ---------------------------------------------------------------------------
# blocked reasons (the integrity core)


def test_blocked_reasons(tmp_path):
    # no artifacts at all
    v = cli._adjudicate_hypothesis(_hyp(), [])
    assert v["verdict"] == "blocked" and v["reason_code"] == "no_candidate_artifacts"

    # candidate present, baseline missing
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.8, 0.8, 0.8])
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    assert v["verdict"] == "blocked" and v["reason_code"] == "no_baseline_artifacts"

    # insufficient training runs (2 < min_seeds=3; eval seeds inside an
    # artifact cannot substitute - that is the ci-v1 pseudo-replication)
    root2 = tmp_path / "few"
    _arm_artifacts(root2, "cand", "ultrametric", [0.8, 0.8])
    _arm_artifacts(root2, "base", "standard", [0.5, 0.5])
    v = cli._adjudicate_hypothesis(_hyp(), _index(root2))
    assert v["verdict"] == "blocked" and v["reason_code"] == "insufficient_seeds"

    # budget mismatch beyond 5%: no cohort contains both arms
    root3 = tmp_path / "budget"
    _arm_artifacts(root3, "cand", "ultrametric", [0.8, 0.8, 0.8], target_flops=2e9)
    _arm_artifacts(root3, "base", "standard", [0.5, 0.5, 0.5], target_flops=4e9)
    v = cli._adjudicate_hypothesis(_hyp(), _index(root3))
    assert v["verdict"] == "blocked" and v["reason_code"] == "budget_mismatch"

    # metric missing at the registered path
    root4 = tmp_path / "metric"
    _arm_artifacts(root4, "cand", "ultrametric", [0.8, 0.8, 0.8], task="dyck")
    _arm_artifacts(root4, "base", "standard", [0.5, 0.5, 0.5], task="dyck")
    v = cli._adjudicate_hypothesis(_hyp(), _index(root4))  # hyp points at tasks.hier
    assert v["verdict"] == "blocked" and v["reason_code"] == "metric_missing"

    # not operationalized
    v = cli._adjudicate_hypothesis(_hyp() | {"prediction": None}, [])
    assert v["verdict"] == "blocked" and v["reason_code"] == "prediction_not_operationalized"


def test_tainted_evidence_refused_and_clean_twin_adjudicates(tmp_path):
    """The bead's explicit pair: tainted fixture -> BLOCKED with the correct
    reason; a clean twin of the same numbers adjudicates normally."""
    dirty = tmp_path / "dirty"
    _arm_artifacts(dirty, "cand", "ultrametric", [0.8, 0.82, 0.81], tainted=True)
    _arm_artifacts(dirty, "base", "standard", [0.5, 0.51, 0.52])
    v = cli._adjudicate_hypothesis(_hyp(), _index(dirty))
    assert v["verdict"] == "blocked" and v["reason_code"] == "tainted_evidence", v

    clean = tmp_path / "clean"
    _arm_artifacts(clean, "cand", "ultrametric", [0.8, 0.82, 0.81])
    _arm_artifacts(clean, "base", "standard", [0.5, 0.51, 0.52])
    v = cli._adjudicate_hypothesis(_hyp(), _index(clean))
    assert v["verdict"] == "supported"

    # missing provenance entirely == tainted (pre-rz8.2 artifacts)
    noprov = tmp_path / "noprov"
    _arm_artifacts(noprov, "cand", "ultrametric", [0.8, 0.82, 0.81])
    for sub in noprov.iterdir():
        p = sub / "summary.json"
        data = json.loads(p.read_text())
        del data["provenance"]
        p.write_text(json.dumps(data))
    _arm_artifacts(noprov, "base", "standard", [0.5, 0.51, 0.52])
    v = cli._adjudicate_hypothesis(_hyp(), _index(noprov))
    assert v["verdict"] == "blocked" and v["reason_code"] == "tainted_evidence"


def test_forall_multi_mechanism_worst_case(tmp_path):
    """A multi-mechanism entry is a FOR-ALL claim: worst arm decides."""
    _arm_artifacts(tmp_path, "c1-", "ultrametric", [0.80, 0.81, 0.82])
    _arm_artifacts(tmp_path, "c2-", "fractal", [0.50, 0.51, 0.50])
    _arm_artifacts(tmp_path, "base", "standard", [0.50, 0.51, 0.52])
    hyp = _hyp(mechanisms=["ultrametric", "fractal"])
    v = cli._adjudicate_hypothesis(hyp, _index(tmp_path))
    assert v["verdict"] == "refuted"  # fractal arm fails the +0.05 claim decisively
    assert v["arms"]["ultrametric"]["verdict"] == "supported"
    assert v["arms"]["fractal"]["verdict"] == "refuted"

    # any arm without evidence blocks the whole FOR-ALL claim
    hyp = _hyp(mechanisms=["ultrametric", "braid"])
    v = cli._adjudicate_hypothesis(hyp, _index(tmp_path))
    assert v["verdict"] == "blocked" and v["mechanism"] == "braid"


def test_train_schema_arm_detection(tmp_path):
    """ordinal/hoss arms resolve via hparams/config, not attention_type."""

    def train_artifact(name, *, scheduler="none", optimizer="adamw", val_ce=2.0):
        run = tmp_path / name
        run.mkdir(parents=True)
        (run / "summary.json").write_text(json.dumps({
            "schema_version": "mgr.telemetry.v1",
            "config": {"attention_type": "standard", "optimizer_type": optimizer},
            "hparams": {"scheduler_type": scheduler},
            "budget": {"max_steps": 100, "target_flops": 1e9, "flops_per_step_est": 1e7},
            "provenance": CLEAN_PROV,
            "results": {"val_ce_final": val_ce},
        }))

    for i, ce in enumerate([1.90, 1.92, 1.91]):
        train_artifact(f"ord{i}", scheduler="ordinal", val_ce=ce)
    for i, ce in enumerate([2.00, 2.02, 2.01]):
        train_artifact(f"std{i}", val_ce=ce)
    hyp = _hyp(
        {"metric_path": "train:results.val_ce_final", "comparator": "<=",
         "threshold_kind": "ratio", "threshold": 0.98},
        mechanisms=["ordinal"],
    )
    v = cli._adjudicate_hypothesis(hyp, _index(tmp_path))
    assert v["verdict"] == "supported", v
    assert v["arms"]["ordinal"]["n_candidate"] == 3


# ---------------------------------------------------------------------------
# ci-v2: floor validity gate, budget cohorts, lineage dedupe


def test_floor_gate_recorded_prior_downgrades_refuted(tmp_path):
    """The pilot1/kbj2 regime: every arm at the best-constant-answer score.
    A null effect there is no-power, not evidence of absence."""
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.0208, 0.0208, 0.0208], answer_prior=0.0208)
    _arm_artifacts(tmp_path, "base", "standard", [0.0208, 0.0208, 0.0208], answer_prior=0.0208)
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    assert v["verdict"] == "inconclusive", v
    arm = v["arms"]["ultrametric"]
    assert arm["floor_effect"] is True and arm["floor_source"] == "recorded_answer_prior"
    assert arm["baseline_mean"] == pytest.approx(0.0208)
    assert v.get("floor_effect") is True


def test_floor_gate_registered_fallback_for_v1_artifacts(tmp_path):
    """v1 artifacts record no answer_prior; the registered validity floor
    must gate instead."""
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.5, 0.5, 0.5], schema_version="mgr.evaltasks.v1")
    _arm_artifacts(tmp_path, "base", "standard", [0.5, 0.5, 0.5], schema_version="mgr.evaltasks.v1")
    hyp = _hyp({"validity": {"baseline_floor": 0.521, "floor_source": "population prior (test)"}})
    v = cli._adjudicate_hypothesis(hyp, _index(tmp_path))
    assert v["verdict"] == "inconclusive", v
    assert v["arms"]["ultrametric"]["floor_source"] == "registered_baseline_floor"


def test_floor_gate_leaves_real_refutations_alone(tmp_path):
    """Baseline far above its floor: a tight null effect is a REAL refutation."""
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.40, 0.41, 0.40], answer_prior=0.05)
    _arm_artifacts(tmp_path, "base", "standard", [0.40, 0.40, 0.41], answer_prior=0.05)
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    assert v["verdict"] == "refuted", v
    arm = v["arms"]["ultrametric"]
    assert "floor_effect" not in arm and arm["baseline_mean"] > arm["baseline_floor"]


def test_budget_cohorts_largest_qualifying_wins(tmp_path):
    """Pilot evidence at 1e13 (at floor) + E1 evidence at 1e14 (off floor):
    the verdict must come from the E1 cohort, never a cross-budget mix."""
    _arm_artifacts(tmp_path, "pc", "ultrametric", [0.02, 0.02, 0.02], target_flops=1e13, answer_prior=0.02)
    _arm_artifacts(tmp_path, "pb", "standard", [0.02, 0.02, 0.02], target_flops=1e13, answer_prior=0.02)
    _arm_artifacts(tmp_path, "ec", "ultrametric", [0.80, 0.81, 0.82], target_flops=1e14, answer_prior=0.02)
    _arm_artifacts(tmp_path, "eb", "standard", [0.50, 0.51, 0.52], target_flops=1e14, answer_prior=0.02)
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    assert v["verdict"] == "supported", v
    arm = v["arms"]["ultrametric"]
    assert arm["budget_flops"] == pytest.approx(1e14)
    assert arm["n_candidate"] == 3 and arm["n_baseline"] == 3
    assert all("/ec" in p or "/eb" in p for p in v["artifacts"]), v["artifacts"]

    # larger cohort lacking min_seeds falls back to the next one down
    root2 = tmp_path / "partial"
    _arm_artifacts(root2, "pc", "ultrametric", [0.50, 0.51, 0.50], target_flops=1e13)
    _arm_artifacts(root2, "pb", "standard", [0.50, 0.50, 0.51], target_flops=1e13)
    _arm_artifacts(root2, "ec", "ultrametric", [0.9], target_flops=1e14)  # 1 run only
    _arm_artifacts(root2, "eb", "standard", [0.5], target_flops=1e14)
    v = cli._adjudicate_hypothesis(_hyp(), _index(root2))
    assert v["verdict"] == "refuted", v
    assert v["arms"]["ultrametric"]["budget_flops"] == pytest.approx(1e13)


def test_lineage_dedupe_re_evals_never_double_count(tmp_path):
    """Re-evaluating the same checkpoints (e.g. after an eval upgrade) must
    not inflate n; the newest/richest artifact per training run wins."""
    for s in range(3):
        _evaltasks_artifact(tmp_path, f"old-c{s}", mechanism="ultrametric", per_seed=[0.0208] * 3,
                            run_id=f"u{s}", schema_version="mgr.evaltasks.v1",
                            generated_at="2026-06-10T18:00:00Z")
        _evaltasks_artifact(tmp_path, f"new-c{s}", mechanism="ultrametric", per_seed=[0.0208] * 3,
                            run_id=f"u{s}", answer_prior=0.0208, generated_at="2026-06-10T23:00:00Z")
        _evaltasks_artifact(tmp_path, f"old-b{s}", mechanism="standard", per_seed=[0.0208] * 3,
                            run_id=f"s{s}", schema_version="mgr.evaltasks.v1",
                            generated_at="2026-06-10T18:00:00Z")
        _evaltasks_artifact(tmp_path, f"new-b{s}", mechanism="standard", per_seed=[0.0208] * 3,
                            run_id=f"s{s}", answer_prior=0.0208, generated_at="2026-06-10T23:00:00Z")
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    arm = v["arms"]["ultrametric"]
    assert arm["n_candidate"] == 3 and arm["n_baseline"] == 3, arm
    assert all("/new-" in p for p in v["artifacts"]), v["artifacts"]
    # and the v2 twins' recorded priors drive the gate
    assert v["verdict"] == "inconclusive" and arm["floor_effect"] is True


# ---------------------------------------------------------------------------
# determinism + CLI + ledger


def test_determinism_byte_identical_verdicts(tmp_path, monkeypatch):
    _arm_artifacts(tmp_path / "a", "cand", "ultrametric", [0.6, 0.7, 0.65])
    _arm_artifacts(tmp_path / "a", "base", "standard", [0.5, 0.55, 0.52])
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "schema_version: 1\nhypotheses:\n"
        + _yaml_entry(_hyp({"threshold_kind": "ratio", "threshold": 1.05}))
    )
    monkeypatch.setattr(cli, "_hypotheses_registry_path", lambda: registry)
    monkeypatch.setattr(cli, "_load_parent_hypothesis_registry", lambda repo_root: None)

    outs = []
    for run in ("r1", "r2"):
        result = runner.invoke(cli.app, [
            "adjudicate", "--all", "--dry-run", "--artifacts", str(tmp_path / "a"),
            "--artifacts-dir", str(tmp_path / run), "--run-id", "x",
        ])
        assert result.exit_code == 0, result.output
        outs.append((tmp_path / run / "adjudications" / "x" / "verdicts.json").read_bytes())
    assert outs[0] == outs[1], "same artifacts must produce byte-identical verdicts (fixed bootstrap seed)"


def _yaml_entry(h):
    pred = h["prediction"]
    lines = [
        f"  - id: {h['id']}",
        f"    statement: {json.dumps(h['statement'])}",
        f"    mechanisms: [{', '.join(h['mechanisms'])}]",
        "    source: {kind: human, provenance: test}",
        f"    date_registered: \"{h['date_registered']}\"",
    ]
    if pred is None:
        lines += ["    prediction: null", "    operationalization_note: test", "    status: blocked"]
    else:
        lines += [
            "    prediction:",
            f"      metric_path: {json.dumps(pred['metric_path'])}",
            f"      comparator: \"{pred['comparator']}\"",
            f"      threshold_kind: {pred['threshold_kind']}",
            f"      threshold: {pred['threshold']}",
            f"      baseline: {{mechanism: {pred['baseline']['mechanism']}, equal_flops: true}}",
            f"      min_seeds: {pred['min_seeds']}",
            "    status: open",
        ]
    lines += ["    evidence: []", "    verdict_history: []", ""]
    return "\n".join(lines)


def test_ledger_append_preserves_comments_and_is_append_only(tmp_path, monkeypatch):
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "# HAND-WRITTEN HEADER COMMENT - must survive ledger surgery\n"
        "schema_version: 1\nhypotheses:\n" + _yaml_entry(_hyp())
    )
    monkeypatch.setattr(cli, "_hypotheses_registry_path", lambda: registry)
    monkeypatch.setattr(cli, "_load_parent_hypothesis_registry", lambda repo_root: None)
    _arm_artifacts(tmp_path / "a", "cand", "ultrametric", [0.8, 0.82, 0.81])
    _arm_artifacts(tmp_path / "a", "base", "standard", [0.5, 0.51, 0.52])

    result = runner.invoke(cli.app, [
        "adjudicate", "--all", "--artifacts", str(tmp_path / "a"),
        "--artifacts-dir", str(tmp_path / "out"), "--run-id", "x",
    ])
    assert result.exit_code == 0, result.output
    text = registry.read_text()
    assert "HAND-WRITTEN HEADER COMMENT" in text, "ledger surgery must preserve comments"
    data, _ = cli._load_hypothesis_registry(registry)
    entry = data["hypotheses"][0]
    assert entry["status"] == "supported"
    assert len(entry["verdict_history"]) == 1
    first = entry["verdict_history"][0]
    assert first["verdict"] == "supported" and first["adjudicator"] == "engine:ci-v2"
    assert first["policy_version"] == "ci-v2" and first["artifacts"]

    # second adjudication APPENDS; the first entry is untouched
    result = runner.invoke(cli.app, [
        "adjudicate", "--all", "--artifacts", str(tmp_path / "a"),
        "--artifacts-dir", str(tmp_path / "out2"), "--run-id", "x",
    ])
    assert result.exit_code == 0, result.output
    data, _ = cli._load_hypothesis_registry(registry)
    history = data["hypotheses"][0]["verdict_history"]
    assert len(history) == 2 and history[0] == first, "prior entries must never mutate"

    errors, _, _ = cli._validate_hypothesis_registry(data, [], REPO_ROOT, parent=None)
    assert errors == [], f"ledgered registry must stay valid: {errors}"


def test_dry_run_leaves_registry_untouched(tmp_path, monkeypatch):
    registry = tmp_path / "registry.yaml"
    registry.write_text("schema_version: 1\nhypotheses:\n" + _yaml_entry(_hyp()))
    before = registry.read_text()
    monkeypatch.setattr(cli, "_hypotheses_registry_path", lambda: registry)
    monkeypatch.setattr(cli, "_load_parent_hypothesis_registry", lambda repo_root: None)
    _arm_artifacts(tmp_path / "a", "cand", "ultrametric", [0.8, 0.82, 0.81])
    _arm_artifacts(tmp_path / "a", "base", "standard", [0.5, 0.51, 0.52])
    result = runner.invoke(cli.app, [
        "adjudicate", "--all", "--dry-run", "--artifacts", str(tmp_path / "a"),
        "--artifacts-dir", str(tmp_path / "out"), "--run-id", "x",
    ])
    assert result.exit_code == 0, result.output
    assert registry.read_text() == before


def test_blocked_refusals_do_not_touch_the_ledger(tmp_path, monkeypatch):
    registry = tmp_path / "registry.yaml"
    registry.write_text("schema_version: 1\nhypotheses:\n" + _yaml_entry(_hyp()))
    before = registry.read_text()
    monkeypatch.setattr(cli, "_hypotheses_registry_path", lambda: registry)
    monkeypatch.setattr(cli, "_load_parent_hypothesis_registry", lambda repo_root: None)
    result = runner.invoke(cli.app, [
        "adjudicate", "--all", "--artifacts", str(tmp_path / "empty"),
        "--artifacts-dir", str(tmp_path / "out"), "--run-id", "x",
    ])
    assert result.exit_code == 0, result.output
    assert registry.read_text() == before, "refusals are report-only"
    verdicts = json.loads((tmp_path / "out" / "adjudications" / "x" / "verdicts.json").read_text())
    assert verdicts["verdicts"][0]["verdict"] == "blocked"


def test_cli_argument_errors(tmp_path):
    result = runner.invoke(cli.app, ["adjudicate"])
    assert result.exit_code == 2
    result = runner.invoke(cli.app, ["adjudicate", "--hypothesis", "hyp-does-not-exist", "--dry-run"])
    assert result.exit_code == 2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))


def test_variant_selector_distinguishes_semiring_beta_arms(tmp_path):
    """rgyl: annealed vs fixed-beta vs exact-tropical runs share a mechanism;
    variant selectors split them into arms. A null variant value matches both
    recorded-null and knob-absent (pre-rgyl) artifacts."""

    def train_artifact(name, *, spec, val_ce, record_key=True):
        run = tmp_path / name
        run.mkdir(parents=True)
        hparams = {"scheduler_type": "none"}
        if record_key:
            hparams["semiring_beta_spec"] = spec
        (run / "summary.json").write_text(json.dumps({
            "schema_version": "mgr.telemetry.v1",
            "config": {"attention_type": "tropical", "optimizer_type": "adamw"},
            "hparams": hparams,
            "budget": {"max_steps": 100, "target_flops": 1e9, "flops_per_step_est": 1e7},
            "provenance": CLEAN_PROV,
            "results": {"val_ce_final": val_ce},
        }))

    for i, ce in enumerate([1.90, 1.92, 1.91]):
        train_artifact(f"anneal{i}", spec="linear:1:32", val_ce=ce)
    for i, ce in enumerate([2.00, 2.02, 2.01]):
        train_artifact(f"fixed{i}", spec="1.0", val_ce=ce)
    # pre-rgyl artifact: the knob is ABSENT entirely (legacy summary)
    for i, ce in enumerate([2.50, 2.52, 2.51]):
        train_artifact(f"legacy{i}", spec=None, val_ce=ce, record_key=(i == 0))

    hyp = _hyp(
        {
            "metric_path": "train:results.val_ce_final",
            "comparator": "<=", "threshold_kind": "ratio", "threshold": 0.98,
            "baseline": {"mechanism": "tropical", "equal_flops": True,
                          "variant": {"semiring_beta_spec": "1.0"}},
            "candidate_variant": {"semiring_beta_spec": "linear:1:32"},
        },
        mechanisms=["tropical"],
    )
    v = cli._adjudicate_hypothesis(hyp, _index(tmp_path))
    assert v["verdict"] == "supported", v
    arm = v["arms"]["tropical"]
    assert arm["n_candidate"] == 3 and arm["n_baseline"] == 3
    assert abs(arm["effect"] - 1.91 / 2.01) < 0.01
    assert all("anneal" in p or "fixed" in p for p in v["artifacts"]), v["artifacts"]

    # null variant = the exact-tropical arm: catches recorded-null AND absent
    hyp_null = _hyp(
        {
            "metric_path": "train:results.val_ce_final",
            "comparator": "<=", "threshold_kind": "ratio", "threshold": 0.98,
            "baseline": {"mechanism": "tropical", "equal_flops": True,
                          "variant": {"semiring_beta_spec": None}},
            "candidate_variant": {"semiring_beta_spec": "linear:1:32"},
        },
        mechanisms=["tropical"],
    )
    v = cli._adjudicate_hypothesis(hyp_null, _index(tmp_path))
    assert v["verdict"] == "supported", v
    assert v["arms"]["tropical"]["n_baseline"] == 3  # all three legacy/null artifacts
    assert all("anneal" in p or "legacy" in p for p in v["artifacts"]), v["artifacts"]

    # no selector = pre-rgyl behavior: every tropical run pools into one arm
    hyp_plain = _hyp(
        {"metric_path": "train:results.val_ce_final",
         "comparator": "<=", "threshold_kind": "ratio", "threshold": 0.98},
        mechanisms=["tropical"],
    )
    hyp_plain["prediction"]["baseline"] = {"mechanism": "tropical", "equal_flops": True}
    v = cli._adjudicate_hypothesis(hyp_plain, _index(tmp_path))
    assert v["arms"]["tropical"]["n_candidate"] == 9
