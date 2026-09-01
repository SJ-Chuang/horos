"""E5-T5: checkpoint management and resuming training (E5-S6)."""

from __future__ import annotations

import json
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


def _finish(project, run_id, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = training_status(project, run_id)
        if status.run.state not in ("pending", "running"):
            return status
        time.sleep(0.2)
    pytest.fail(f"run {run_id} still active after {timeout}s")


def test_resume_from_a_previous_runs_checkpoint(project):
    first = start_training(project, TrainRunConfig(entrypoint_override=FAKE, epochs=2))
    first_status = _finish(project, first.run_id)
    checkpoint = first_status.run.checkpoint
    assert checkpoint and Path(checkpoint).is_file()

    second = start_training(
        project,
        TrainRunConfig(entrypoint_override=FAKE, epochs=2, resume_from=checkpoint),
    )
    second_status = _finish(project, second.run_id)
    assert second_status.run.state == "completed"

    # the checkpoint path reached the backend's TrainSpec intact
    completed = [e for e in second_status.events if e["type"] == "completed"][-1]
    assert completed["result"]["resumed_from"] == checkpoint

    # and it is recorded in the run's own config for reproducibility
    config = json.loads(
        (project.root / "runs" / second.run_id / "config.json").read_text("utf-8")
    )
    assert config["resume_from"] == checkpoint
    assert second_status.run.config["resume_from"] == checkpoint


def test_each_run_keeps_its_own_checkpoints(project):
    first = start_training(project, TrainRunConfig(entrypoint_override=FAKE, epochs=1))
    _finish(project, first.run_id)
    second = start_training(project, TrainRunConfig(entrypoint_override=FAKE, epochs=1))
    _finish(project, second.run_id)

    first_ckpt = project.root / "runs" / first.run_id / "checkpoints" / "best.fake"
    second_ckpt = project.root / "runs" / second.run_id / "checkpoints" / "best.fake"
    assert first_ckpt.is_file() and second_ckpt.is_file()
    assert first_ckpt != second_ckpt  # resuming can never overwrite the source run
