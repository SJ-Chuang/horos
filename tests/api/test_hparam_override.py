"""E5-T2: user overrides replace single values without shifting the rest of
the plan, and the override is visible in the recorded derivation trail."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from helpers.data import write_sample_coco_dir

from horos.api.dataset import import_dataset
from horos.api.hparams import derive_plan
from horos.api.project import create_project
from horos.api.train import (
    TrainRunConfig,
    derive_hyperparameters,
    start_training,
    training_status,
)
from horos.backends.memory import MemoryInfo
from horos.core.registry import get_model_info
from horos.core.stats import ClassStats, DatasetStats, RelativeAreaStats

NANO = get_model_info("rfdetr-nano")
TESTS_ROOT = Path(__file__).parent.parent
FAKE = "helpers.fake_backend:FakeBackend"


def _stats() -> DatasetStats:
    return DatasetStats(
        num_images=300,
        num_annotations=900,
        num_categories=1,
        per_class=[ClassStats(category_id=1, name="block", instances=900, images=300)],
        split_counts={"train": 300},
        image_sizes=[],
        relative_area=RelativeAreaStats(
            minimum=0.01, maximum=0.4, mean=0.1, median=0.1, histogram=[0] * 10
        ),
    )


def _memory() -> MemoryInfo:
    return MemoryInfo(kind="cuda", total_gb=16.0, available_gb=16.0, source="test")


def _plan(overrides=None):
    return derive_plan(
        _stats(), model="rfdetr-nano", model_info=NANO, memory=_memory(),
        overrides=overrides,
    )


def test_override_replaces_value_and_is_marked():
    plan = _plan({"batch_size": 8})
    assert plan.values["batch_size"] == 8
    entry = next(d for d in plan.derivations if d.name == "batch_size")
    assert entry.overridden is True and entry.reason == "user override"


def test_partial_override_leaves_other_derivations_untouched():
    derived = _plan()
    overridden = _plan({"batch_size": 8})
    for name in ("epochs", "resolution", "warmup_epochs", "num_workers"):
        assert overridden.values[name] == derived.values[name]
    # dependent values still follow the override: effective batch stays 16
    assert overridden.values["grad_accum_steps"] == 2


def test_none_is_not_an_override():
    plan = _plan({"epochs": None, "batch_size": None})
    assert not any(d.overridden for d in plan.derivations)


def test_start_training_records_the_plan(tmp_path, monkeypatch):
    """Integration: run.json carries values, reasons, and override marks."""
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", str(TESTS_ROOT) + (os.pathsep + existing if existing else "")
    )
    project = create_project(tmp_path / "proj")
    import_dataset(project, write_sample_coco_dir(tmp_path / "coco"))

    plan = derive_hyperparameters(project, TrainRunConfig(epochs=2))
    assert plan.values["epochs"] == 2  # override honored in the preview too

    record = start_training(
        project, TrainRunConfig(entrypoint_override=FAKE, epochs=2)
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = training_status(project, record.run_id)
        if status.run.state not in ("pending", "running"):
            break
        time.sleep(0.2)
    assert status.run.state == "completed"

    epochs = next(d for d in status.run.hparams if d.name == "epochs")
    assert epochs.overridden is True and epochs.value == 2
    batch = next(d for d in status.run.hparams if d.name == "batch_size")
    assert batch.overridden is False and batch.reason.strip()

    # the worker received the fully resolved config
    config = json.loads(
        (project.root / "runs" / record.run_id / "config.json").read_text("utf-8")
    )
    assert config["epochs"] == 2
    assert isinstance(config["batch_size"], int)
    assert "grad_accum_steps" in config["extra"]


def test_user_extra_beats_derived_extra(tmp_path, monkeypatch):
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", str(TESTS_ROOT) + (os.pathsep + existing if existing else "")
    )
    project = create_project(tmp_path / "proj")
    import_dataset(project, write_sample_coco_dir(tmp_path / "coco"))
    record = start_training(
        project,
        TrainRunConfig(
            entrypoint_override=FAKE, epochs=1, extra={"num_workers": 7}
        ),
    )
    config = json.loads(
        (project.root / "runs" / record.run_id / "config.json").read_text("utf-8")
    )
    assert config["extra"]["num_workers"] == 7  # E5-S5: expert override wins
    deadline = time.monotonic() + 30
    while training_status(project, record.run_id).run.state in ("pending", "running"):
        if time.monotonic() > deadline:
            pytest.fail("run did not finish")
        time.sleep(0.2)
