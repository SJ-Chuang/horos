"""Shared builder: a completed training run on the fake backend (no ML deps)."""

from __future__ import annotations

import os
import time
from pathlib import Path

from helpers.data import write_sample_coco_dir
from horos.api.dataset import import_dataset
from horos.api.project import create_project
from horos.api.train import TrainRunConfig, start_training, training_status

TESTS_ROOT = Path(__file__).parent.parent
FAKE = "helpers.fake_backend:FakeBackend"


def ensure_worker_can_import_helpers() -> None:
    """The training worker is a subprocess: it needs tests/ on PYTHONPATH."""
    existing = os.environ.get("PYTHONPATH", "")
    if str(TESTS_ROOT) not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = str(TESTS_ROOT) + (
            os.pathsep + existing if existing else ""
        )


def completed_fake_run(tmp_path: Path, *, epochs: int = 3, **config):
    """Create a project from the sample COCO dir, train `epochs` epochs on the
    fake backend and wait for completion. Returns (project, record)."""
    ensure_worker_can_import_helpers()
    project = create_project(tmp_path / "proj")
    import_dataset(project, write_sample_coco_dir(tmp_path / "coco"))
    record = start_training(
        project, TrainRunConfig(entrypoint_override=FAKE, epochs=epochs, **config)
    )
    deadline = time.monotonic() + 60
    while training_status(project, record.run_id).run.state in ("pending", "running"):
        assert time.monotonic() < deadline, "fake training did not finish"
        time.sleep(0.2)
    status = training_status(project, record.run_id)
    assert status.run.state == "completed", status.run.error
    return project, status.run
