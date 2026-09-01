"""E5-T6b: the training entry point is spawn-safe (R7).

Windows and macOS default to `spawn`, which re-imports the child's main
module. The worker runs everything under an `if __name__ == "__main__"` guard,
so a DataLoader-style spawned grandchild must not re-execute training. The
probe backend starts a real spawn-context child mid-training and the test
asserts the run stays coherent: exactly one training pass, no duplicated
events, no corrupted state.
"""

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
SPAWN_PROBE = "helpers.fake_backend:SpawnProbeBackend"


@pytest.fixture(autouse=True)
def worker_can_import_helpers(monkeypatch):
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", str(TESTS_ROOT) + (os.pathsep + existing if existing else "")
    )


def test_training_survives_a_spawned_child_process(tmp_path):
    project = create_project(tmp_path / "proj")
    import_dataset(project, write_sample_coco_dir(tmp_path / "coco"))
    record = start_training(
        project, TrainRunConfig(entrypoint_override=SPAWN_PROBE, epochs=1)
    )
    deadline = time.monotonic() + 60  # spawn interpreter startup is slow
    while training_status(project, record.run_id).run.state in ("pending", "running"):
        assert time.monotonic() < deadline, "spawn-probe run never finished"
        time.sleep(0.2)

    status = training_status(project, record.run_id)
    assert status.run.state == "completed", status.run.error

    run_dir = project.root / "runs" / record.run_id
    # the spawned child really ran (the probe fails the run otherwise)
    marker = run_dir / "checkpoints" / "spawn_marker.txt"
    assert marker.read_text("utf-8") == "spawned-child-ran"

    # spawn re-import did not re-enter the worker: one run, one event stream
    assert sum(1 for e in status.events if e["type"] == "started") == 1
    assert sum(1 for e in status.events if e["type"] == "completed") == 1
    assert [r.name for r in (project.root / "runs").iterdir()] == [record.run_id]
    assert Path(status.run.checkpoint).is_file()
