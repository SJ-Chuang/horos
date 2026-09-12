"""E7-T2: the dataset fingerprint is a content hash that detects dataset changes."""

from __future__ import annotations

from helpers.data import sample_dataset, write_sample_coco_dir
from helpers.runs import completed_fake_run

from horos.core.dataset import Annotation, ImageRecord
from horos.core.fingerprint import (
    compare_fingerprints,
    fingerprint_dataset,
    fingerprint_snapshot,
)
from horos.core.formats.coco import read_coco


def test_same_content_same_digest_regardless_of_ids_and_order():
    base = fingerprint_dataset(sample_dataset())
    assert base.digest.startswith("sha256:")
    assert set(base.splits) == {"train", "valid"}
    assert base.classes == ["forklift", "pallet"]
    assert (base.num_images, base.num_annotations) == (3, 4)

    shuffled = sample_dataset()
    shuffled.images.reverse()
    shuffled.annotations.reverse()
    # renumber everything the way an export does
    remap = {img.id: img.id + 100 for img in shuffled.images}
    for img in shuffled.images:
        img.id = remap[img.id]
    for i, ann in enumerate(shuffled.annotations, start=500):
        ann.id = i
        ann.image_id = remap[ann.image_id]
    assert fingerprint_dataset(shuffled) == base


def test_a_moved_box_changes_only_its_split():
    base = fingerprint_dataset(sample_dataset())
    edited = sample_dataset()
    edited.annotations[0].bbox = (5.0, 4.0, 16.0, 12.0)  # image 1 is in train
    changed = fingerprint_dataset(edited)
    assert changed.digest != base.digest
    assert changed.splits["train"] != base.splits["train"]
    assert changed.splits["valid"] == base.splits["valid"]
    diff = compare_fingerprints(base, changed)
    assert not diff.identical
    assert diff.changed_splits == ["train"] and not diff.classes_changed
    assert diff.describe() == "train split(s) differ"


def test_sub_rounding_noise_is_ignored_but_class_rename_is_not():
    base = fingerprint_dataset(sample_dataset())
    noisy = sample_dataset()
    noisy.annotations[0].bbox = (4.0001, 4.0, 16.0, 12.0)
    assert fingerprint_dataset(noisy).digest == base.digest

    renamed = sample_dataset()
    renamed.categories[0].name = "lift-truck"
    diff = compare_fingerprints(base, fingerprint_dataset(renamed))
    assert diff.classes_changed and "class set differs" in diff.describe()


def test_added_image_and_annotation_change_the_split():
    base = fingerprint_dataset(sample_dataset())
    grown = sample_dataset()
    grown.images.append(ImageRecord(id=9, file_name="d.png", width=8, height=8, split="valid"))
    grown.annotations.append(
        Annotation(id=9, image_id=9, category_id=1, bbox=(0.0, 0.0, 4.0, 4.0))
    )
    diff = compare_fingerprints(base, fingerprint_dataset(grown))
    assert diff.changed_splits == ["valid"]


def test_mosaic_composites_do_not_count():
    base = fingerprint_dataset(sample_dataset())
    with_mosaic = sample_dataset()
    with_mosaic.images.append(
        ImageRecord(id=7, file_name="mosaic_0001.jpg", width=128, height=96, split="train")
    )
    with_mosaic.annotations.append(
        Annotation(id=7, image_id=7, category_id=1, bbox=(1.0, 1.0, 2.0, 2.0))
    )
    assert fingerprint_dataset(with_mosaic) == base


def test_snapshot_on_disk_matches_in_memory(tmp_path):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    dataset, _ = read_coco(coco_dir)
    assert fingerprint_snapshot(coco_dir) == fingerprint_dataset(dataset)
    assert fingerprint_snapshot(coco_dir).digest == fingerprint_dataset(sample_dataset()).digest
    assert fingerprint_snapshot(tmp_path / "nowhere") is None


def test_run_records_its_fingerprint_at_enqueue(tmp_path):
    project, record = completed_fake_run(tmp_path, epochs=1)
    assert record.dataset_fingerprint is not None
    # the recorded fingerprint is what the snapshot on disk hashes to ...
    snapshot = fingerprint_snapshot(project.root / "runs" / record.run_id / "dataset")
    assert snapshot == record.dataset_fingerprint
    # ... and what the live project hashes to today (nothing changed since)
    assert fingerprint_dataset(project.to_dataset()).digest == record.dataset_fingerprint.digest
