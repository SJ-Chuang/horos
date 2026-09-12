"""Shared builder: several completed fake runs inside ONE project (E7)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from helpers.data import write_sample_coco_dir
from helpers.runs import FAKE, ensure_worker_can_import_helpers
from horos.api.dataset import import_dataset
from horos.api.project import create_project
from horos.api.train import TrainRunConfig, start_training, training_status


def train_fake(project, *, epochs: int, **config):
    """Train one fake run to completion and return its record."""
    ensure_worker_can_import_helpers()
    record = start_training(
        project, TrainRunConfig(entrypoint_override=FAKE, epochs=epochs, **config)
    )
    deadline = time.monotonic() + 60
    while training_status(project, record.run_id).run.state in ("queued", "pending", "running"):
        assert time.monotonic() < deadline, "fake training did not finish"
        time.sleep(0.2)
    status = training_status(project, record.run_id)
    assert status.run.state == "completed", status.run.error
    return status.run


def project_with_runs(tmp_path: Path, epochs=(1, 2, 4)):
    """A project trained `len(epochs)` times; the fake backend's final loss is
    1/epochs, so the runs have distinct, predictable scores. Runs are created
    a second apart so run ids (timestamp-prefixed) and created_at differ."""
    project = create_project(tmp_path / "proj")
    import_dataset(project, write_sample_coco_dir(tmp_path / "coco"))
    records = []
    for count in epochs:
        if records:
            time.sleep(1.05)  # run ids carry a per-second timestamp prefix
        records.append(train_fake(project, epochs=count))
    return project, records


def write_eval(project, run_id: str, split: str = "test", **metrics) -> None:
    """Persist an evaluation report the way E6 does, with the given headline
    metrics (defaults fill the rest)."""
    values = {"map_5095": 0.0, "map_50": 0.0, "map_75": 0.0, "mar_100": 0.0, **metrics}
    eval_dir = project.root / "runs" / run_id / "eval"
    eval_dir.mkdir(exist_ok=True)
    (eval_dir / f"{split}.json").write_text(
        json.dumps({
            "run_id": run_id, "split": split, "created_at": "2026-09-12T00:00:00+00:00",
            "num_images": 1, "num_instances": 1, "per_class": [], **values,
        }),
        "utf-8",
    )
