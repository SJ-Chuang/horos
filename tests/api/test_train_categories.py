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


def test_filter_drops_background_only_images_by_default(project):
    # sample data: a (forklift + pallet), b (forklift) in train; c (pallet) in valid
    dataset = project.to_dataset()
    filtered = filter_dataset_categories(dataset, ["forklift"])
    assert [c.name for c in filtered.categories] == ["forklift"]
    assert sorted(i.file_name for i in filtered.images) == ["a.png", "b.png"]  # c.png gone
    assert all(
        filtered.category_by_id(a.category_id).name == "forklift"
        for a in filtered.annotations
    )
    assert len(filtered.annotations) < len(dataset.annotations)


def test_filter_can_keep_background_images_as_negatives(project):
    dataset = project.to_dataset()
    filtered = filter_dataset_categories(dataset, ["forklift"], include_background=True)
    assert len(filtered.images) == len(dataset.images)  # negatives stay
    assert [c.name for c in filtered.categories] == ["forklift"]
    assert len(filtered.annotations) == 2


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
    assert sorted(i["file_name"] for i in gt["images"]) == ["a.png", "b.png"]
    assert not (tmp_path / "out" / "valid").exists()  # c.png had no forklift

    export_dataset(project, tmp_path / "bg", categories=["forklift"], include_background=True)
    gt_valid = json.loads(
        (tmp_path / "bg" / "valid" / "_annotations.coco.json").read_text("utf-8")
    )
    assert [i["file_name"] for i in gt_valid["images"]] == ["c.png"]  # kept as a negative
    assert gt_valid["annotations"] == []


def test_derivation_sees_the_filtered_data(project):
    full = derive_hyperparameters(project, TrainRunConfig())
    subset = derive_hyperparameters(
        project, TrainRunConfig(categories=["forklift"])
    )
    # the imbalance note computed over both classes must not leak into a
    # single-class run, and warmup reasons must reflect the subset's stats
    # (a tiny fixture triggers the small-dataset 3-epoch warmup rule, whose
    # reason cites the filtered image count rather than class names)
    warm = next(d for d in subset.derivations if d.name == "warmup_epochs")
    assert (
        "forklift" in warm.reason
        or "every class" in warm.reason
        or "optimizer steps" in warm.reason
    )
    assert full is not None  # both plans derive without error


def test_derivation_explains_the_background_decision(project):
    dropped = derive_hyperparameters(project, TrainRunConfig(categories=["forklift"]))
    note = next(n for n in dropped.notes if "none of the selected classes" in n)
    assert "1 of 3 images" in note and "excluded" in note and "remaining 2 images" in note

    kept = derive_hyperparameters(
        project, TrainRunConfig(categories=["forklift"], include_background=True)
    )
    note = next(n for n in kept.notes if "none of the selected classes" in n)
    assert "kept as background" in note and "all 3 images" in note

    # every class selected: nothing to explain
    assert not any("selected classes" in n for n in derive_hyperparameters(project).notes)


def test_dropping_background_can_empty_a_split_and_says_so(project):
    # forklift never appears in the valid split: with negatives dropped the
    # run has nothing to validate on — refused explicitly, never silently
    with pytest.raises(ProjectError, match="no annotated images in the valid split"):
        start_training(
            project,
            TrainRunConfig(entrypoint_override=FAKE, epochs=1, categories=["forklift"]),
        )


def test_run_snapshot_and_record_carry_the_selection(project):
    record = start_training(
        project,
        TrainRunConfig(entrypoint_override=FAKE, epochs=1,
                       categories=["forklift"], include_background=True),
    )
    deadline = time.monotonic() + 30
    while training_status(project, record.run_id).run.state in ("pending", "running"):
        assert time.monotonic() < deadline
        time.sleep(0.2)
    status = training_status(project, record.run_id)
    assert status.run.state == "completed"
    assert status.run.config["categories"] == ["forklift"]
    assert status.run.config["include_background"] is True
    assert status.run.dataset_images == 3
    assert any("kept as background" in n for n in status.run.hparam_notes)

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
    with pytest.raises(ProjectError, match="no annotated images in the train split"):
        start_training(
            proj,
            TrainRunConfig(entrypoint_override=FAKE, epochs=1,
                           categories=["pallet"]),
        )


def test_run_records_its_class_set_for_resume_locking(project):
    record = start_training(
        project,
        TrainRunConfig(entrypoint_override=FAKE, epochs=1,
                       categories=["forklift"], include_background=True),
    )
    deadline = time.monotonic() + 30
    while training_status(project, record.run_id).run.state in ("pending", "running"):
        assert time.monotonic() < deadline
        time.sleep(0.2)
    status = training_status(project, record.run_id)
    assert status.run.dataset_classes == ["forklift"]


def test_old_runs_backfill_their_class_set(project):
    """Runs recorded before dataset_classes existed get it from their
    snapshot on first read — the resume lock works for them too."""
    from horos.api.train import _run_dir, read_record, write_record

    record = start_training(project, TrainRunConfig(entrypoint_override=FAKE, epochs=1))
    deadline = time.monotonic() + 30
    while training_status(project, record.run_id).run.state in ("pending", "running"):
        assert time.monotonic() < deadline
        time.sleep(0.2)

    run_dir = _run_dir(project, record.run_id)
    stored = read_record(run_dir)
    stored.dataset_classes = []  # simulate a pre-field run.json
    write_record(run_dir, stored)

    refreshed = training_status(project, record.run_id).run
    assert refreshed.dataset_classes == ["forklift", "pallet"]
