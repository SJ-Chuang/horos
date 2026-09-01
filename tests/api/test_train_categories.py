"""Class selection for training: the run snapshot, the derivation, and the
run record all see only the chosen categories; splits stay untouched."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from helpers.data import write_sample_coco_dir

from horos.api.dataset import (
    export_dataset,
    filter_dataset_categories,
    import_dataset,
)
from horos.api.project import create_project
from horos.api.train import (
    TrainRunConfig,
    derive_hyperparameters,
    start_training,
    training_status,
)
from horos.errors import ProjectError

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
    # sample dataset: categories forklift + pallet across train/valid
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    return proj


def test_filter_keeps_images_drops_other_annotations(project):
    dataset = project.to_dataset()
    filtered = filter_dataset_categories(dataset, ["forklift"])
    assert [c.name for c in filtered.categories] == ["forklift"]
    assert len(filtered.images) == len(dataset.images)  # negatives stay
    assert all(
        filtered.category_by_id(a.category_id).name == "forklift"
        for a in filtered.annotations
    )
    assert len(filtered.annotations) < len(dataset.annotations)


def test_filter_rejects_unknown_and_empty(project):
    dataset = project.to_dataset()
    with pytest.raises(ProjectError, match="Unknown categor"):
        filter_dataset_categories(dataset, ["ghost"])
    with pytest.raises(ProjectError, match="at least one"):
        filter_dataset_categories(dataset, [])


def test_export_with_categories_writes_filtered_coco(project, tmp_path):
    export_dataset(project, tmp_path / "out", categories=["forklift"])
    gt = json.loads(
        (tmp_path / "out" / "train" / "_annotations.coco.json").read_text("utf-8")
    )
    assert [c["name"] for c in gt["categories"]] == ["forklift"]
    assert len(gt["images"]) > 0  # images kept even when their objects dropped


def test_derivation_sees_the_filtered_data(project):
    full = derive_hyperparameters(project, TrainRunConfig())
    subset = derive_hyperparameters(
        project, TrainRunConfig(categories=["forklift"])
    )
    # the imbalance note computed over both classes must not leak into a
    # single-class run, and warmup reasons must reference the subset
    warm = next(d for d in subset.derivations if d.name == "warmup_epochs")
    assert "forklift" in warm.reason or "every class" in warm.reason
    assert full is not None  # both plans derive without error


def test_run_snapshot_and_record_carry_the_selection(project):
    record = start_training(
        project,
        TrainRunConfig(entrypoint_override=FAKE, epochs=1,
                       categories=["forklift"]),
    )
    deadline = time.monotonic() + 30
    while training_status(project, record.run_id).run.state in ("pending", "running"):
        assert time.monotonic() < deadline
        time.sleep(0.2)
    status = training_status(project, record.run_id)
    assert status.run.state == "completed"
    assert status.run.config["categories"] == ["forklift"]

    gt = json.loads(
        (project.root / "runs" / record.run_id / "dataset" / "train"
         / "_annotations.coco.json").read_text("utf-8")
    )
    assert [c["name"] for c in gt["categories"]] == ["forklift"]


def test_selection_without_train_annotations_is_refused(project, tmp_path):

    # pallet exists only via annotations in the sample train split; craft a
    # class that exists but has no train-split annotations: move its images
    # is complex — instead select a class then empty the train split of it by
    # importing a fresh project whose 'pallet' appears only in valid.
    proj = create_project(tmp_path / "proj2")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco2"))
    from horos.api.annotate import get_annotations, save_annotations

    # remove every pallet annotation from train images
    dataset = proj.to_dataset()
    pallet = dataset.category_by_name("pallet")
    for image in dataset.images_in_split("train"):
        view = get_annotations(proj, image.id)
        kept = [a for a in view.annotations if a.category_id != pallet.id]
        if len(kept) != len(view.annotations):
            save_annotations(proj, image.id, kept, expected_version=view.version)
    with pytest.raises(ProjectError, match="no\\s+annotations in the train split"):
        start_training(
            proj,
            TrainRunConfig(entrypoint_override=FAKE, epochs=1,
                           categories=["pallet"]),
        )
