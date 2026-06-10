"""Checkpoint/resume acceptance suite for nanochat.train (bead rz8.1).

The centerpiece is the interruption test: a run checkpointed mid-flight and
resumed must reproduce the uninterrupted run's loss trajectory BITWISE on CPU
fp32. This is achievable because every state component round-trips exactly:
model weights, optimizer momenta (AdamW + Muon), ordinal-scheduler counters,
the data position (exact replay fast-forward), and all RNG streams.

The parquet dataloader is replaced by a deterministic synthetic token stream
(seeded independently of consumption count), so the suite runs in CI without
the fineweb shards while still exercising the trainer's full state machinery
including the with-state loader contract and the fast-forward path.
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch

import nanochat.train as train_mod

VOCAB_FAKE = 1000  # < GPTConfig default vocab_size, so targets are always valid


def _fake_loader_factory(record: list[str] | None = None):
    """A deterministic stand-in for tokenizing_distributed_data_loader_with_state.

    Yields the SAME batch sequence on every instantiation (generator seeded by
    a constant, independent of global RNG), mirroring the real loader's
    determinism. Honors the resume_state_dict contract approximately the way
    the real loader does: resumes from the NEXT index after the recorded one.
    """

    def fake_loader(B, T, split, device="cpu", resume_state_dict=None, **kwargs):
        gen = torch.Generator().manual_seed(1234)
        idx = 0
        skip_to = None
        if resume_state_dict is not None:
            skip_to = int(resume_state_dict["rg_idx"]) + 1
        while True:
            tokens = torch.randint(0, VOCAB_FAKE, (B, T + 1), generator=gen, dtype=torch.long)
            state = {"pq_idx": 0, "rg_idx": idx}
            if skip_to is None or idx >= skip_to:
                if record is not None:
                    record.append(hashlib.sha256(tokens.numpy().tobytes()).hexdigest())
                yield tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous(), state
            idx += 1

    return fake_loader


def _train_args(tmp_path: Path, run_id: str, **overrides):
    argv = [
        "--device",
        "cpu",
        "--max-steps",
        "12",
        "--batch-size",
        "2",
        "--sequence-len",
        "32",
        "--n-layer",
        "1",
        "--n-head",
        "2",
        "--n-kv-head",
        "2",
        "--n-embd",
        "32",
        "--seed",
        "7",
        "--warmup-steps",
        "0",
        "--artifacts-dir",
        str(tmp_path / "artifacts"),
        "--run-id",
        run_id,
    ]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if value is None:
            argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return train_mod.build_parser().parse_args(argv)


def _run_train(monkeypatch, tmp_path: Path, run_id: str, record: list[str] | None = None, **overrides) -> dict:
    monkeypatch.setattr(train_mod, "tokenizing_distributed_data_loader_with_state", _fake_loader_factory(record))
    monkeypatch.setattr(train_mod, "list_parquet_files", lambda: ["fake_train.parquet", "fake_val.parquet"])
    train_mod.train(_train_args(tmp_path, run_id, **overrides))
    summary_path = tmp_path / "artifacts" / "baseline" / "nanochat" / run_id / "summary.json"
    assert summary_path.exists(), f"missing summary at {summary_path}"
    summary: dict = json.loads(summary_path.read_text())
    return summary


def _ckpt_dir(tmp_path: Path, run_id: str) -> Path:
    return tmp_path / "artifacts" / "baseline" / "nanochat" / run_id / "checkpoints"


def _assert_bitwise_trajectory(resumed: list[float], reference: list[float], *, offset: int) -> None:
    """Bead requirement: on mismatch, pinpoint the FIRST divergent step with both values."""
    assert len(resumed) == len(reference), f"trajectory length {len(resumed)} != reference {len(reference)}"
    for i, (a, b) in enumerate(zip(resumed, reference)):
        if a != b:
            pytest.fail(
                f"resume diverged at step {offset + i}: resumed loss {a!r} != uninterrupted loss {b!r} "
                f"(first divergence; {sum(1 for x, y in zip(resumed, reference) if x != y)} steps differ in total)"
            )


def test_interrupted_run_resumes_bitwise(monkeypatch, tmp_path):
    """THE acceptance test: checkpoint at step 5, resume, steps 6-11 bitwise-match
    the uninterrupted run on CPU fp32 (AdamW + Muon both round-trip)."""
    hashes_a: list[str] = []
    summary_a = _run_train(monkeypatch, tmp_path, "uninterrupted", record=hashes_a)
    losses_a = summary_a["results"]["losses"]
    assert len(losses_a) == 12

    hashes_b1: list[str] = []
    summary_b1 = _run_train(
        monkeypatch, tmp_path, "parent", record=hashes_b1, checkpoint_interval=6
    )
    assert summary_b1["checkpointing"]["saved_steps"] == [5, 11]
    # Checkpointing must not perturb the trajectory itself.
    _assert_bitwise_trajectory(summary_b1["results"]["losses"], losses_a, offset=0)

    hashes_b2: list[str] = []
    summary_b2 = _run_train(
        monkeypatch,
        tmp_path,
        "resumed",
        record=hashes_b2,
        resume_from=str(_ckpt_dir(tmp_path, "parent")),
        resume_step=5,
    )
    losses_b2 = summary_b2["results"]["losses"]
    assert summary_b2["results"]["start_step"] == 6
    assert len(losses_b2) == 6
    _assert_bitwise_trajectory(losses_b2, losses_a[6:], offset=6)

    # Batch-hash assertion (no repeated/skipped batches): the resumed run's
    # full yielded stream (fast-forward replays 0..5, then trains 6..11)
    # must equal the uninterrupted run's stream batch-for-batch.
    assert hashes_b2 == hashes_a, "resumed run consumed a different batch sequence than the uninterrupted run"

    # Resume lineage recorded.
    assert summary_b2["resume"]["parent_run_ids"] == ["parent"]
    assert summary_b2["resume"]["resume_step"] == 6


def test_resume_restores_original_budget(monkeypatch, tmp_path):
    """A resumed run honors the ORIGINAL budget even if the resume command line disagrees."""
    _run_train(monkeypatch, tmp_path, "budget-parent", checkpoint_interval=4, max_steps=8)
    summary = _run_train(
        monkeypatch,
        tmp_path,
        "budget-resumed",
        resume_from=str(_ckpt_dir(tmp_path, "budget-parent")),
        resume_step=3,
        max_steps=999,  # must be overridden by the checkpoint's recorded budget
    )
    assert summary["budget"]["max_steps"] == 8
    assert summary["results"]["start_step"] == 4
    assert len(summary["results"]["losses"]) == 4


def test_resume_with_ordinal_scheduler_round_trips_counters(monkeypatch, tmp_path):
    """Scheduler counters survive the round trip and the trajectory stays bitwise.

    Also covers the orphan fix: the ordinal scheduler is now actually stepped
    (P_init patience consumes on non-improving steps), so its state is
    no longer frozen at construction values.
    """
    summary_a = _run_train(monkeypatch, tmp_path, "sched-uninterrupted", scheduler_type="ordinal")
    summary_b1 = _run_train(
        monkeypatch, tmp_path, "sched-parent", scheduler_type="ordinal", checkpoint_interval=6
    )
    state = torch.load(
        _ckpt_dir(tmp_path, "sched-parent") / "optim_000005_rank0.pt", weights_only=False
    )
    assert len(state["schedulers"]) == len(state["optimizers"]) > 0
    sched_state = state["schedulers"][0]
    assert sched_state["ema_loss"] is not None, "scheduler was never stepped - the orphan bug is back"
    summary_b2 = _run_train(
        monkeypatch,
        tmp_path,
        "sched-resumed",
        scheduler_type="ordinal",
        resume_from=str(_ckpt_dir(tmp_path, "sched-parent")),
        resume_step=5,
    )
    _assert_bitwise_trajectory(
        summary_b2["results"]["losses"], summary_a["results"]["losses"][6:], offset=6
    )
    assert summary_b1["checkpointing"]["saved_steps"] == [5, 11]


def test_checkpoint_retention_prunes_oldest(monkeypatch, tmp_path):
    summary = _run_train(
        monkeypatch, tmp_path, "retention", checkpoint_interval=2, checkpoint_keep=2, max_steps=8
    )
    ckpt = _ckpt_dir(tmp_path, "retention")
    # Saves at steps 1,3,5,7; keep=2 retains only the newest two.
    assert summary["checkpointing"]["saved_steps"] == [5, 7]
    present = sorted(p.name for p in ckpt.glob("model_*.pt"))
    assert present == ["model_000005.pt", "model_000007.pt"]
    for step in (5, 7):
        assert (ckpt / f"optim_{step:06d}_rank0.pt").exists()
        assert (ckpt / f"meta_{step:06d}.json").exists()
    for step in (1, 3):
        assert not (ckpt / f"meta_{step:06d}.json").exists()


def test_checkpoint_verify_happy_path(monkeypatch, tmp_path):
    summary = _run_train(
        monkeypatch, tmp_path, "verified", checkpoint_interval=6, checkpoint_verify=None
    )
    assert summary["checkpointing"]["verify"] is True
    assert summary["checkpointing"]["saved_steps"] == [5, 11]


def test_resume_config_mismatch_is_actionable(monkeypatch, tmp_path):
    _run_train(monkeypatch, tmp_path, "mismatch-parent", checkpoint_interval=6)
    with pytest.raises(ValueError, match="n_embd"):
        _run_train(
            monkeypatch,
            tmp_path,
            "mismatch-resumed",
            resume_from=str(_ckpt_dir(tmp_path, "mismatch-parent")),
            n_embd=64,  # checkpoint was trained with 32
        )


def test_resume_approximate_mode_passes_loader_state(monkeypatch, tmp_path):
    """Approximate mode hands the recorded (pq_idx, rg_idx) to the loader's
    native resume instead of fast-forwarding; trajectories need not be bitwise."""
    _run_train(monkeypatch, tmp_path, "approx-parent", checkpoint_interval=6)
    summary = _run_train(
        monkeypatch,
        tmp_path,
        "approx-resumed",
        resume_from=str(_ckpt_dir(tmp_path, "approx-parent")),
        resume_step=5,
        resume_data_mode="approximate",
    )
    assert summary["resume"]["data_mode"] == "approximate"
    assert summary["results"]["start_step"] == 6
    assert len(summary["results"]["losses"]) == 6


def test_resume_under_torch_compile_round_trips(monkeypatch, tmp_path):
    """torch.compile both directions: the checkpoint is saved from the raw module
    (clean keys) and resume loads before the compile wrap (aot_eager keeps eager
    numerics, so compiled-parent -> compiled-resume stays bitwise)."""
    summary_a = _run_train(
        monkeypatch, tmp_path, "compile-uninterrupted", compile=None, compile_backend="aot_eager", max_steps=6
    )
    _run_train(
        monkeypatch,
        tmp_path,
        "compile-parent",
        compile=None,
        compile_backend="aot_eager",
        checkpoint_interval=3,
        max_steps=6,
    )
    model_sd = torch.load(
        _ckpt_dir(tmp_path, "compile-parent") / "model_000002.pt", weights_only=True
    )
    assert not any(k.startswith("_orig_mod.") for k in model_sd), "checkpoint keys must be clean under compile"
    summary_b = _run_train(
        monkeypatch,
        tmp_path,
        "compile-resumed",
        compile=None,
        compile_backend="aot_eager",
        resume_from=str(_ckpt_dir(tmp_path, "compile-parent")),
        resume_step=2,
        max_steps=6,
    )
    _assert_bitwise_trajectory(
        summary_b["results"]["losses"], summary_a["results"]["losses"][3:], offset=3
    )


def test_ordinal_scheduler_state_dict_round_trip():
    """Unit: a restored scheduler continues the exact LR trajectory of the original."""
    from nanochat.ordinal_scheduler import OrdinalLRScheduler

    def make():
        params = [torch.nn.Parameter(torch.zeros(2))]
        opt = torch.optim.SGD(params, lr=1.0)
        return OrdinalLRScheduler(opt, A_init=1, B_init=2, P_init=3, eta_init=0.1, gamma=0.5)

    # Non-improving losses force C to count down and trigger anneals/restarts.
    losses = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    ref = make()
    for x in losses:
        ref.step(x)
    ref_lrs = ref.get_last_lr()

    half = make()
    for x in losses[:5]:
        half.step(x)
    snapshot = half.state_dict()

    restored = make()
    restored.load_state_dict(snapshot)
    # LR is optimizer state; carry it over the way train.py does (optimizer
    # state_dict restore happens alongside the scheduler restore).
    for group, lr in zip(restored.optimizer.param_groups, half.get_last_lr()):
        group["lr"] = lr
    for x in losses[5:]:
        restored.step(x)
    assert restored.get_last_lr() == ref_lrs
    assert (restored.A, restored.B, restored.C) == (ref.A, ref.B, ref.C)
    assert restored.best_loss == ref.best_loss and restored.ema_loss == ref.ema_loss
