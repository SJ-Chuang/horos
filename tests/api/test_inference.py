"""E6-T1: single and batch inference bound to a trained run, plus the
evaluation event stream over the run's own dataset snapshot."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from helpers.data import write_sample_coco_dir

from horos.api.dataset import import_dataset
from horos.api.evaluate import (
    _reset_backend_cache,
    evaluation_events,
    get_eval_report,
    infer_image,
)
from horos.api.project import create_project
from horos.api.train import TrainRunConfig, start_training, training_status
from horos.errors import ProjectError

TESTS_ROOT = Path(__file__).parent.parent
FAKE = "helpers.fake_backend:FakeBackend"

pycocotools = pytest.importorskip("pycocotools", reason="training stack not installed")


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", str(TESTS_ROOT) + (os.pathsep + existing if existing else "")
    )
    _reset_backend_cache()


@pytest.fixture
def trained(tmp_path):
    """A project with one completed fake run."""
    project = create_project(tmp_path / "proj")
    import_dataset(project, write_sample_coco_dir(tmp_path / "coco"))
    record = start_training(
        project, TrainRunConfig(entrypoint_override=FAKE, epochs=1)
    )
    deadline = time.monotonic() + 30
    while training_status(project, record.run_id).run.state in ("pending", "running"):
        assert time.monotonic() < deadline
        time.sleep(0.2)
    assert training_status(project, record.run_id).run.state == "completed"
    return project, record.run_id


def test_infer_image_uses_the_runs_checkpoint(trained, tmp_path):
    project, run_id = trained
    image = next((project.root / "runs" / run_id / "dataset" / "train").glob("*.png"))
    prediction = infer_image(project, run_id, image)
    assert prediction.instances and prediction.instances[0].score == 0.9


def test_infer_rejects_missing_file_and_unknown_run(trained):
    project, run_id = trained
    with pytest.raises(ProjectError, match="No such image"):
        infer_image(project, run_id, "does-not-exist.png")
    with pytest.raises(ProjectError, match="No such training run"):
        infer_image(project, "nope", "also-irrelevant.png")


def test_infer_rejects_unfinished_runs(trained):
    project, run_id = trained
    slow = start_training(
        project,
        TrainRunConfig(entrypoint_override=FAKE, epochs=60,
                       extra={"sleep_per_epoch": 0.5}),
    )
    try:
        with pytest.raises(ProjectError, match="no usable checkpoint"):
            infer_image(project, slow.run_id, "x.png")
    finally:
        from horos.api.train import stop_training

        stop_training(project, slow.run_id)
        deadline = time.monotonic() + 30
        while training_status(project, slow.run_id).run.state in ("pending", "running"):
            assert time.monotonic() < deadline
            time.sleep(0.2)


def test_evaluation_streams_and_persists_a_report(trained):
    project, run_id = trained
    events = list(evaluation_events(project, run_id, split="valid"))
    assert events[0].type == "started" and events[0].total == 1  # 1 valid image
    assert events[-1].type == "completed"
    report = events[-1].result
    assert report["split"] == "valid" and report["num_images"] == 1
    # the fake predicts category 0 which is not in the gt → zero AP, but the
    # pipeline is intact and every gt class is reported
    assert report["map_50"] == 0.0
    assert {c["name"] for c in report["per_class"]} == {"forklift", "pallet"}

    persisted = get_eval_report(project, run_id, "valid")
    assert persisted.run_id == run_id and persisted.map_50 == 0.0


def test_evaluation_refuses_missing_split(trained):
    project, run_id = trained
    with pytest.raises(ProjectError, match="no 'nope' split"):
        list(evaluation_events(project, run_id, split="nope"))


def test_report_before_any_evaluation_is_a_clear_error(trained):
    project, run_id = trained
    with pytest.raises(ProjectError, match="no persisted evaluation"):
        get_eval_report(project, run_id, "test")
