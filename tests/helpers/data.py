"""Shared builders for tiny real datasets used across E1 tests."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from horos.core.dataset import Annotation, Category, Dataset, ImageRecord


def make_image(path: Path, width: int = 64, height: int = 48, color=(120, 30, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color).save(path)
    return path


def sample_dataset() -> Dataset:
    """Two categories, three images across splits, boxes + one polygon."""
    return Dataset(
        categories=[
            Category(id=1, name="forklift", color="#e6194b"),
            Category(id=2, name="pallet", color="#3cb44b"),
        ],
        images=[
            ImageRecord(id=1, file_name="a.png", width=64, height=48, split="train"),
            ImageRecord(id=2, file_name="b.png", width=64, height=48, split="train"),
            ImageRecord(id=3, file_name="c.png", width=32, height=32, split="valid"),
        ],
        annotations=[
            Annotation(id=1, image_id=1, category_id=1, bbox=(4.0, 4.0, 16.0, 12.0)),
            Annotation(id=2, image_id=1, category_id=2, bbox=(20.0, 8.0, 8.0, 8.0)),
            Annotation(id=3, image_id=2, category_id=1, bbox=(0.0, 0.0, 32.0, 24.0)),
            Annotation(
                id=4,
                image_id=3,
                category_id=2,
                bbox=(2.0, 2.0, 12.0, 10.0),
                segmentation=[[2.0, 2.0, 14.0, 2.0, 14.0, 12.0, 2.0, 12.0]],
            ),
        ],
    )


def write_sample_coco_dir(root: Path, *, split_layout: bool = True) -> Path:
    """Materialize sample_dataset() as a real COCO directory with images."""
    from horos.core.formats.coco import write_coco

    dataset = sample_dataset()
    image_paths: dict[int, Path] = {}
    staging = root / "_staging"
    for record in dataset.images:
        image_paths[record.id] = make_image(
            staging / record.file_name, record.width, record.height
        )
    write_coco(
        dataset, root, image_paths=image_paths, split_layout=split_layout,
        copy_images=True,
    )
    import shutil

    shutil.rmtree(staging)
    return root


def write_sample_yolo_dir(root: Path) -> Path:
    from horos.core.formats.yolo import write_yolo

    dataset = sample_dataset()
    image_paths: dict[int, Path] = {}
    staging = root / "_staging"
    for record in dataset.images:
        image_paths[record.id] = make_image(
            staging / record.file_name, record.width, record.height
        )
    write_yolo(dataset, root, image_paths=image_paths)
    import shutil

    shutil.rmtree(staging)
    return root


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
