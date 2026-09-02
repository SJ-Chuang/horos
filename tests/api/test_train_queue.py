"""Training queue: a start while a run is active queues instead of refusing,
the queue advances when the active run ends (worker chain + status-poll
heartbeat), and queued runs can be cancelled or deleted."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from helpers.data import write_sample_coco_dir

from horos.api.dataset import import_dataset
from horos.api.project import create_project
from horos.api.train import (
    TrainRunConfig,
    advance_queue,
    delete_run,
    list_runs,
    read_record,
    runs_root,
    start_training,
    stop_training,
    training_status,
)

TESTS_ROOT = Path(__file__).parent.parent
FAKE = "helpers.fake_backend:FakeBackend"


@pytest.fixture(autouse=True)
def worker_can_import_helpers(monkeypatch):
    existing = os.environ.get("PYTHONPATH", "")
    joined = str(TESTS_ROOT) + (os.pathsep + existing if existing else "")
    monkeypatch.setenv("PYTHONPATH", joined)


@pytest.fixture
def project(tmp_path):
    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    return proj


def _config(**overrides):
    defaults = dict(entrypoint_override=FAKE, epochs=2)
    defaults.update(overrides)
    return TrainRunConfig(**defaults)


def _wait_state(project, run_id, states, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = training_status(project, run_id).run
        if record.state in states:
            return record
        time.sleep(0.2)
    pytest.fail(f"run {run_id} never reached {states} (last: {record.state})")


def test_queued_run_is_created_but_not_spawned(project):
    first = start_training(project, _config(epochs=60, extra={"sleep_per_epoch": 0.5}))
    try:
        queued = start_training(project, _config())
        assert queued.state == "queued"
        assert queued.pid is None
        # the snapshot and plan are fixed at enqueue time
        run_dir = runs_root(project) / queued.run_id
        assert (run_dir / "dataset" / "train" / "_annotations.coco.json").is_file()
        assert (run_dir / "config.json").is_file()
        states = {r.run_id: r.state for r in list_runs(project)}
        assert states[queued.run_id] == "queued"
        assert states[first.run_id] == "running"
        stop_training(project, queued.run_id)
    finally:
        stop_training(project, first.run_id)
        _wait_state(project, first.run_id, ("stopped", "completed", "failed"))


def test_queue_advances_when_active_run_ends(project):
    first = start_training(project, _config())  # 2 fake epochs: finishes fast
    queued = start_training(project, _config())
    assert queued.state == "queued"
    # the exiting worker chains into the queued run; the training_status
    # polling inside _wait_state is the fallback heartbeat
    record = _wait_state(project, queued.run_id, ("completed",))
    assert record.checkpoint and Path(record.checkpoint).is_file()
    assert _wait_state(project, first.run_id, ("completed",)).state == "completed"


def test_queue_is_fifo(project):
    first = start_training(project, _config(epochs=60, extra={"sleep_per_epoch": 0.5}))
    q1 = start_training(project, _config())
    q2 = start_training(project, _config())
    try:
        stop_training(project, first.run_id)
        _wait_state(project, q1.run_id, ("completed",))
        # q2 must not have started before q1 finished
        record_q2 = read_record(runs_root(project) / q2.run_id)
        assert record_q2.state in ("queued", "pending", "running", "completed")
        _wait_state(project, q2.run_id, ("completed",))
    finally:
        for run in (first, q1, q2):
            stop_training(project, run.run_id)
            _wait_state(project, run.run_id, ("stopped", "completed", "failed"))


def test_stopping_a_queued_run_cancels_it(project):
    first = start_training(project, _config(epochs=60, extra={"sleep_per_epoch": 0.5}))
    queued = start_training(project, _config())
    try:
        assert stop_training(project, queued.run_id) is True
        assert training_status(project, queued.run_id).run.state == "stopped"
    finally:
        stop_training(project, first.run_id)
        _wait_state(project, first.run_id, ("stopped",))
    # the cancelled run is never promoted
    advance_queue(runs_root(project))
    assert training_status(project, queued.run_id).run.state == "stopped"


def test_deleting_a_queued_run_works(project):
    first = start_training(project, _config(epochs=60, extra={"sleep_per_epoch": 0.5}))
    queued = start_training(project, _config())
    try:
        assert delete_run(project, queued.run_id) is True
        assert not (runs_root(project) / queued.run_id).exists()
        assert queued.run_id not in [r.run_id for r in list_runs(project)]
    finally:
        stop_training(project, first.run_id)
        _wait_state(project, first.run_id, ("stopped",))


def test_foreign_claim_halts_promotion_instead_of_skipping_ahead(project):
    """When the oldest queued run is already claimed by a concurrent promoter,
    advance_queue must NOT start the next queued run — that would train two
    runs at once."""
    first = start_training(project, _config(epochs=60, extra={"sleep_per_epoch": 0.5}))
    q1 = start_training(project, _config())
    q2 = start_training(project, _config())
    # simulate a concurrent promoter that claimed q1 but has not spawned yet
    (runs_root(project) / q1.run_id / "spawn.claim").touch()
    stop_training(project, first.run_id)
    _wait_state(project, first.run_id, ("stopped",))
    assert advance_queue(runs_root(project)) is None  # halted by q1's claim
    assert read_record(runs_root(project) / q1.run_id).state == "queued"
    assert read_record(runs_root(project) / q2.run_id).state == "queued"
    for run_id in (q1.run_id, q2.run_id):
        stop_training(project, run_id)


def test_reconcile_leaves_queued_runs_alone(project):
    first = start_training(project, _config(epochs=60, extra={"sleep_per_epoch": 0.5}))
    queued = start_training(project, _config())
    try:
        # queued runs have no worker; reconciliation must not fail them
        for _ in range(3):
            assert training_status(project, queued.run_id).run.state == "queued"
            time.sleep(0.1)
        stop_training(project, queued.run_id)
    finally:
        stop_training(project, first.run_id)
        _wait_state(project, first.run_id, ("stopped",))


# ------------------------------------------------------------- queue editing


def _queued_pair(project):
    """A long-running active run plus one queued behind it."""
    active = start_training(
        project, _config(epochs=60, extra={"sleep_per_epoch": 0.5})
    )
    queued = start_training(project, _config(epochs=10))
    assert queued.state == "queued"
    return active, queued


def test_update_queued_run_edits_in_place(project):
    from horos.api.train import update_queued_run

    active, queued = _queued_pair(project)
    try:
        updated = update_queued_run(
            project, queued.run_id,
            {"epochs": 25, "lr": 5e-5, "checkpoint_criterion": "loss"},
        )
        assert updated.run_id == queued.run_id  # same run, same queue slot
        assert updated.state == "queued"
        assert updated.config["epochs"] == 25
        assert updated.config["extra"]["lr"] == pytest.approx(5e-5)
        assert updated.config["checkpoint_criterion"] == "loss"
        by_name = {h.name: h for h in updated.hparams}
        assert by_name["epochs"].overridden and by_name["epochs"].value == 25
        assert by_name["lr"].overridden
        assert by_name["checkpoint_criterion"].value == "loss"
        # config.json (what the worker will read) matches the record
        stored = TrainRunConfig.model_validate_json(
            (runs_root(project) / queued.run_id / "config.json").read_text("utf-8")
        )
        assert stored.epochs == 25 and stored.checkpoint_criterion == "loss"
    finally:
        stop_training(project, queued.run_id)
        stop_training(project, active.run_id)
        _wait_state(project, active.run_id, ("stopped",))


def test_update_queued_run_rederives_dependent_values(project):
    from horos.api.train import update_queued_run

    active, queued = _queued_pair(project)
    try:
        before = {h.name: h.value for h in queued.hparams}
        updated = update_queued_run(project, queued.run_id, {"batch_size": 2})
        after = {h.name: h.value for h in updated.hparams}
        assert after["batch_size"] == 2
        # grad accumulation follows the batch edit (effective batch stays 16)
        assert after["grad_accum_steps"] == 8
        assert before["grad_accum_steps"] != after["grad_accum_steps"] or (
            before["batch_size"] == 2
        )
        # clearing the override goes back to the derived value
        reverted = update_queued_run(project, queued.run_id, {"batch_size": None})
        by_name = {h.name: h for h in reverted.hparams}
        assert by_name["batch_size"].overridden is False
        assert by_name["batch_size"].value == before["batch_size"]
    finally:
        stop_training(project, queued.run_id)
        stop_training(project, active.run_id)
        _wait_state(project, active.run_id, ("stopped",))


def test_update_refuses_non_queued_runs_and_unknown_fields(project):
    from horos.api.train import update_queued_run
    from horos.errors import ProjectError

    active, queued = _queued_pair(project)
    try:
        with pytest.raises(ProjectError, match="only queued runs"):
            update_queued_run(project, active.run_id, {"epochs": 5})
        with pytest.raises(ProjectError, match="Not editable"):
            update_queued_run(project, queued.run_id, {"categories": ["x"]})
    finally:
        stop_training(project, queued.run_id)
        stop_training(project, active.run_id)
        _wait_state(project, active.run_id, ("stopped",))
