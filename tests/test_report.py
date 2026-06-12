"""Tests for mgr report (bead yhy, roll-up half): cross-run aggregation over
an artifacts tree. The arm tables must use the engine's evidence semantics -
clean artifacts only, lineage-deduped (one observation per trained
checkpoint) - so a hand-written campaign.md and the roll-up can never
disagree about what the evidence says.
"""

import json
from pathlib import Path

from typer.testing import CliRunner

import cli

runner = CliRunner()

CLEAN_PROV = {"schema_version": "mgr.metrics.v1", "git_sha": "deadbeef", "git_dirty": False,
              "config_hash": "abc", "data_snapshot_hash": None, "tainted": False}
TAINTED_PROV = {**CLEAN_PROV, "git_dirty": True, "tainted": True}


def _train_summary(root: Path, name: str, *, mechanism: str, task: str, seed: int,
                   ce: float, flops: float = 3e14, tainted: bool = False) -> None:
    run_dir = root / "campaigns" / "t" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps({
        "schema_version": "mgr.telemetry.v1",
        "meta": {"run_id": name, "argv": ["train.py", "--seed", str(seed)]},
        "config": {"attention_type": mechanism},
        "dataset": {"data_dir": f"artifacts/diagnostics_e1/{task}"},
        "budget": {"target_flops": flops},
        "provenance": TAINTED_PROV if tainted else CLEAN_PROV,
        "results": {"train_ce_final": ce, "val_ce_final": None, "measured_time_s": 600.0},
    }))


def _eval_summary(root: Path, name: str, *, mechanism: str, task: str, em: float,
                  run_id: str | None = None, step: int = 100, prior: float = 0.625,
                  flops: float = 3e14, tainted: bool = False,
                  generated_at: str = "2026-06-12T00:00:00Z") -> None:
    run_dir = root / "evals" / "tasks" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps({
        "schema_version": "mgr.evaltasks.v2",
        "kind": "eval-tasks",
        "meta": {
            "generated_at": generated_at,
            "checkpoint": {"attention_type": mechanism, "step": step,
                           "budget": {"target_flops": flops},
                           "lineage": {"run_id": run_id or name, "parent_run_ids": []}},
            "seeds": [0, 1, 2],
        },
        "provenance": TAINTED_PROV if tainted else CLEAN_PROV,
        "tasks": {task: {
            "exact_match": {"greedy": {"held_out": {"mean": em, "per_seed": [em] * 3}}},
            "answer_prior": {"held_out": {"mean": prior, "per_seed": [prior] * 3,
                                          "majority_answer": "x"}},
        }},
    }))


def test_report_rolls_up_arms_with_engine_semantics(tmp_path):
    """The arm tables aggregate over CLEAN, lineage-deduped evidence: a
    tainted eval is counted-but-excluded and a same-(run_id, step) re-eval
    never double-counts."""
    arts = tmp_path / "artifacts"
    _train_summary(arts, "dyck-braid-s30", mechanism="braid", task="dyck", seed=30, ce=0.80)
    _train_summary(arts, "dyck-braid-s31", mechanism="braid", task="dyck", seed=31, ce=0.90)
    _train_summary(arts, "dyck-standard-s30", mechanism="standard", task="dyck", seed=30,
                   ce=1.20, tainted=True)
    _eval_summary(arts, "e2-dyck-braid-s30", mechanism="braid", task="dyck", em=0.95,
                  run_id="dyck-braid-s30")
    _eval_summary(arts, "e2-dyck-braid-s31", mechanism="braid", task="dyck", em=0.97,
                  run_id="dyck-braid-s31")
    # re-eval of the SAME checkpoint: deduped, never a second observation
    _eval_summary(arts, "e2-dyck-braid-s30-r2", mechanism="braid", task="dyck", em=0.10,
                  run_id="dyck-braid-s30", generated_at="2026-06-11T00:00:00Z")
    # tainted eval: counted, excluded from stats
    _eval_summary(arts, "e2-dyck-standard-s30", mechanism="standard", task="dyck", em=0.91,
                  run_id="dyck-standard-s30", tainted=True)

    result = runner.invoke(cli.app, [
        "report", "--artifacts", str(arts), "--out", str(tmp_path / "reports"), "--run-id", "r1",
    ])
    assert result.exit_code == 0, result.output

    payload = json.loads((tmp_path / "reports" / "r1" / "report.json").read_text())
    assert payload["schema_version"] == "mgr.report.v1"
    assert payload["counts"]["train"] == {"total": 3, "tainted": 1}
    assert payload["counts"]["evaltasks"] == {"total": 4, "tainted": 1}
    assert payload["eval_tainted_excluded"] == 1

    # train arm grouping: braid clean pair averaged; tainted standard CE excluded
    tg = {(g["task"], g["mechanism"]): g for g in payload["train_groups"]}
    braid_tg = tg[("dyck", "braid")]
    assert braid_tg["n"] == 2 and braid_tg["n_tainted"] == 0
    assert braid_tg["train_ce_final"].startswith("0.85")
    std_tg = tg[("dyck", "standard")]
    assert std_tg["n"] == 1 and std_tg["n_tainted"] == 1
    assert std_tg["train_ce_final"] == "-"

    # eval arm grouping: dedupe keeps the NEWEST braid-s30 eval (em=0.95, not
    # the older 0.10 re-eval); tainted standard arm contributes no group
    eg = {(g["task"], g["mechanism"]): g for g in payload["eval_groups"]}
    braid_eg = eg[("dyck", "braid")]
    assert braid_eg["n_models"] == 2
    assert braid_eg["em_held_out"].startswith("0.96")  # mean(0.95, 0.97)
    assert braid_eg["answer_prior"] == "0.625"
    assert ("dyck", "standard") not in eg

    md = (tmp_path / "reports" / "r1" / "report.md").read_text()
    assert "| dyck | braid | 3e+14 | 2 | 0.96" in md
    assert "1 tainted eval artifact(s) excluded" in md
    assert payload["git"] is not None and payload["roots"] == [str(arts)]


def test_report_handles_empty_tree(tmp_path):
    """An empty root produces an empty (but valid) report, not a crash."""
    result = runner.invoke(cli.app, [
        "report", "--artifacts", str(tmp_path / "nothing"),
        "--out", str(tmp_path / "reports"), "--run-id", "empty",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "reports" / "empty" / "report.json").read_text())
    assert payload["counts"] == {} and payload["train_groups"] == [] and payload["eval_groups"] == []
