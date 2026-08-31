"""Darknet format reader (import-only by design decision — horos exports COCO/YOLO).

Darknet is YOLO's label-line format without a data.yaml: each image has a
same-stem .txt next to it, and class names live in a `_darknet.labels` file
(one name per line — Roboflow writes one per split directory, all identical).

When no labels file exists, callers may pass `class_names`; otherwise indices
("0", "1", …) are used as placeholder names and the import surfaces a warning.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from horos.core.dataset import Annotation, Category, Dataset, ImageRecord, default_color
from horos.errors import DatasetFormatError

from . import IMAGE_SUFFIXES, split_from_dir_name
from .yolo import _parse_label_line

LABELS_FILE = "_darknet.labels"


def find_labels_file(root: Path | str) -> Path | None:
    """Return one `_darknet.labels` under root, erroring if copies disagree."""
    matches = sorted(Path(root).rglob(LABELS_FILE))
    if not matches:
        return None
    contents = {m.read_text(encoding="utf-8").strip() for m in matches}
    if len(contents) > 1:
        raise DatasetFormatError(
            f"Multiple {LABELS_FILE} files under {root} disagree on class names"
        )
    return matches[0]


def _label_files(root: Path) -> list[tuple[Path, Path]]:
    """(image, label .txt) pairs: every image with a same-stem .txt sibling."""
    pairs = []
    for image_file in sorted(root.rglob("*")):
        if image_file.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = image_file.with_suffix(".txt")
        pairs.append((image_file, label if label.is_file() else None))
    return pairs


def read_darknet(
    source: Path | str, *, class_names: list[str] | None = None
) -> tuple[Dataset, dict[int, Path]]:
    """Read a Darknet dataset directory. Returns the Dataset plus {image_id: path}.

    Class names come from `class_names`, else `_darknet.labels`, else
    placeholder index names sized to the largest class index seen.
    """
    source = Path(source)
    pairs = _label_files(source)
    if not any(label for _, label in pairs):
        raise DatasetFormatError(f"No Darknet label .txt files found under {source}")

    names = list(class_names) if class_names else None
    if names is None:
        labels_file = find_labels_file(source)
        if labels_file is not None:
            names = [
                line.strip()
                for line in labels_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    parsed: list[tuple[Path, Path | None, list[tuple[int, list[float]]]]] = []
    max_class = -1
    for image_file, label in pairs:
        lines: list[tuple[int, list[float]]] = []
        if label is not None:
            for lineno, line in enumerate(
                label.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                cls, values = _parse_label_line(line.strip(), label, lineno)
                max_class = max(max_class, cls)
                lines.append((cls, values))
        parsed.append((image_file, label, lines))

    if names is None:
        names = [str(i) for i in range(max_class + 1)]
    if max_class >= len(names):
        raise DatasetFormatError(
            f"Class index {max_class} outside the class-name list (0..{len(names) - 1}) "
            f"— check {LABELS_FILE} or the class_names given"
        )

    dataset = Dataset(
        categories=[
            Category(id=i, name=name, color=default_color(i))
            for i, name in enumerate(names)
        ]
    )
    image_paths: dict[int, Path] = {}
    for image_file, _, lines in parsed:
        try:
            with Image.open(image_file) as im:
                width, height = im.size
        except OSError as exc:
            raise DatasetFormatError(f"Cannot read image {image_file}: {exc}") from exc
        record = ImageRecord(
            id=dataset.next_image_id(),
            file_name=image_file.name,
            width=width,
            height=height,
            split=split_from_dir_name(image_file.parent.name),
        )
        dataset.images.append(record)
        image_paths[record.id] = image_file.resolve()
        for cls, values in lines:
            if len(values) != 4:
                raise DatasetFormatError(
                    f"Darknet labels are boxes only; got {len(values)} values for "
                    f"{image_file.name}"
                )
            cx, cy, w, h = values
            dataset.annotations.append(
                Annotation(
                    id=dataset.next_annotation_id(),
                    image_id=record.id,
                    category_id=cls,
                    bbox=(
                        (cx - w / 2) * width,
                        (cy - h / 2) * height,
                        w * width,
                        h * height,
                    ),
                )
            )
    return dataset, image_paths
