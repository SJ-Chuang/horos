"""E5-T6: OOM backoff — the worker halves the batch size and retries, keeps
the effective batch via gradient accumulation, and tells the user (E5-S7)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from helpers.data import write_sample_coco_dir

from horos.api.dataset import import_dataset
from horos.api.project import create_project
from horos.api.train import TrainRunConfig, start_training, training_status

TESTS_ROOT = Path(__file__).parent.parent
FAKE = "helpers.fake_backend:FakeBackend"


@pytest.fixture(autouse=True)
def worker_can_import_helpers(monkeypatch):
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", str(TESTS_ROOT) + (os.pathsep + existing if existing else "")
    )


@pytest.fixture
def project(tmp_path):
    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    return proj


def _wait_terminal(project, run_id, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = training_status(project, run_id)
        if status.run.state not in ("pending", "running"):
            return status
        time.sleep(0.2)
    pytest.fail(f"run {run_id} still active after {timeout}s")


def test_oom_halves_batch_until_it_fits(project):
    # simulated device fits batch <= 2; the run starts at 8 → 8, 4 fail, 2 fits
    record = start_training(
        project,
        TrainRunConfig(
            entrypoint_override=FAKE,
            epochs=2,
            batch_size=8,
            extra={"oom_above_batch": 2, "grad_accum_steps": 2},
        ),
    )
    status = _wait_terminal(project, record.run_id)
    assert status.run.state == "completed"

    warnings = [e for e in status.events if e["type"] == "warning"]
    assert len(warnings) == 2  # 8→4 and 4→2
    assert "Out of memory at batch 8" in warnings[0]["message"]
    assert "retrying with 4" in warnings[0]["message"]

    # the run record reflects what actually trained, not what was requested
    assert status.run.config["batch_size"] == 2
    # effective batch preserved: accumulation 2 doubled twice → 8
    assert status.run.config["extra"]["grad_accum_steps"] == 8
    # the successful attempt's completion carries the fitted batch
    completed = [e for e in status.events if e["type"] == "completed"]
    assert completed[-1]["result"]["batch_size"] == 2

    # each retry restarts the event stream: one started per attempt
    assert sum(1 for e in status.events if e["type"] == "started") == 3


def test_oom_at_batch_1_fails_honestly(project):
    record = start_training(
        project,
        TrainRunConfig(
            entrypoint_override=FAKE,
            epochs=2,
            batch_size=2,
            extra={"oom_above_batch": 0},  # nothing fits
        ),
    )
    status = _wait_terminal(project, record.run_id)
    assert status.run.state == "failed"
    assert "OOM" in (status.run.error or "")
    # it backed off to 1 before giving up, and never looped below 1
    assert status.run.config["batch_size"] == 1


def test_non_oom_failure_is_not_retried(project):
    record = start_training(
        project,
        TrainRunConfig(entrypoint_override=FAKE, epochs=2, extra={"fail": True}),
    )
    status = _wait_terminal(project, record.run_id)
    assert status.run.state == "failed"
    assert sum(1 for e in status.events if e["type"] == "started") == 1
