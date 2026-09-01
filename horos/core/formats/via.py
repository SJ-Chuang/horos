"""VIA (VGG Image Annotator) region JSON reader — import-only.

Handles the classic layout (e.g. the Mask R-CNN balloon dataset): one
`via_region_data.json` (or VIA 2's `via_export_json.json`) per split
directory with the images sitting next to it. Region shapes map as:
polygon/polyline → polygon annotation (bbox derived), rect → bbox,
circle/ellipse → enclosing bbox with a warning. VIA stores no image
dimensions, so they are read from the image files.

Class names live in free-form `region_attributes`. When regions carry a
recognizable label attribute its values become the classes; when no region
has any attribute at all (the balloon dataset), everything lands in a single
class — named by `class_names[0]` when given, else "object" with a warning.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from horos.core.dataset import Annotation, Category, Dataset, ImageRecord, default_color
from horos.errors import DatasetFormatError

from . import split_from_dir_name

MARKER_NAMES = ("via_region_data.json", "via_export_json.json")

#: attribute keys commonly used for the class label, tried in this order
_LABEL_KEYS = ("class", "label", "name", "category", "type")

_FALLBACK_CLASS = "object"


def find_via_files(source: Path | str) -> list[Path]:
    source = Path(source)
    return sorted(
        path
        for name in MARKER_NAMES
        for path in source.rglob(name)
        if "__MACOSX" not in path.parts  # macOS zip metadata, not data
    )


def _entries(via_path: Path) -> list[dict]:
    try:
        data = json.loads(via_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetFormatError(f"Cannot parse VIA JSON {via_path}: {exc}") from exc
    if "_via_img_metadata" in data:  # a full VIA 2 project file
        data = data["_via_img_metadata"]
    entries = []
    for key, value in data.items():
        if key.startswith("_via") or not isinstance(value, dict):
            continue
        if "filename" in value:
            entries.append(value)
    if not entries:
        raise DatasetFormatError(f"{via_path}: no image entries found")
    return entries


def _regions(entry: dict) -> list[dict]:
    regions = entry.get("regions") or []
    if isinstance(regions, dict):  # VIA 1 keys them "0", "1", ...
        regions = [regions[k] for k in sorted(regions, key=str)]
    return [r for r in regions if isinstance(r, dict)]


def _label_of(region: dict) -> str | None:
    attrs = region.get("region_attributes") or {}
    for key in _LABEL_KEYS:
        value = attrs.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # a single free-form attribute with a string value counts as the label
    strings = [v.strip() for v in attrs.values() if isinstance(v, str) and v.strip()]
    if len(strings) == 1:
        return strings[0]
    return None


def has_class_attributes(source: Path | str) -> bool:
    """Does any region name a class? Drives the class-name prompt in the UI."""
    for via_path in find_via_files(source):
        for entry in _entries(via_path):
            for region in _regions(entry):
                if _label_of(region) is not None:
                    return True
    return False


def _shape_to_geometry(
    shape: dict, context: str, warnings: list[str]
) -> tuple[tuple[float, float, float, float], list[list[float]]]:
    """One VIA shape_attributes dict → (COCO xywh bbox, segmentation)."""
    kind = shape.get("name")
    if kind in ("polygon", "polyline"):
        xs = shape.get("all_points_x") or []
        ys = shape.get("all_points_y") or []
        if len(xs) != len(ys) or len(xs) < 3:
            raise DatasetFormatError(
                f"{context}: {kind} needs ≥3 matching x/y points, "
                f"got {len(xs)}/{len(ys)}"
            )
        flat: list[float] = []
        for x, y in zip(xs, ys, strict=True):
            flat.extend((float(x), float(y)))
        x0, y0 = min(map(float, xs)), min(map(float, ys))
        x1, y1 = max(map(float, xs)), max(map(float, ys))
        return (x0, y0, x1 - x0, y1 - y0), [flat]
    if kind == "rect":
        return (
            float(shape["x"]), float(shape["y"]),
            float(shape["width"]), float(shape["height"]),
        ), []
    if kind in ("circle", "ellipse"):
        cx, cy = float(shape["cx"]), float(shape["cy"])
        rx = float(shape.get("r", shape.get("rx", 0)))
        ry = float(shape.get("r", shape.get("ry", 0)))
        warnings.append(
            f"{context}: {kind} region imported as its enclosing bounding box"
        )
        return (cx - rx, cy - ry, 2 * rx, 2 * ry), []
    raise DatasetFormatError(
        f"{context}: unsupported VIA region shape '{kind}' "
        f"(supported: polygon, polyline, rect, circle, ellipse)"
    )


def read_via(
    source: Path | str, *, class_names: list[str] | None = None
) -> tuple[Dataset, dict[int, Path]]:
    """Read a VIA dataset directory. Returns the Dataset plus {image_id: path}."""
    source = Path(source)
    via_files = find_via_files(source)
    if not via_files:
        raise DatasetFormatError(
            f"No VIA region JSON ({' / '.join(MARKER_NAMES)}) found under {source}"
        )

    # pass 1: the class list. Labeled regions define it; a fully unlabeled
    # dataset (the balloon case) collapses into one class.
    parsed: list[tuple[Path, dict, list[dict]]] = []
    labels: set[str] = set()
    for via_path in via_files:
        for entry in _entries(via_path):
            regions = _regions(entry)
            parsed.append((via_path, entry, regions))
            for region in regions:
                label = _label_of(region)
                if label is not None:
                    labels.add(label)

    warnings: list[str] = []
    if labels:
        names = sorted(labels)
        fallback: str | None = None
    else:
        fallback = (class_names[0] if class_names else _FALLBACK_CLASS).strip()
        names = [fallback]
        if not class_names:
            warnings.append(
                f"VIA regions carry no class attribute — everything imported "
                f"as one class '{_FALLBACK_CLASS}'; rename it later or "
                f"re-import with class_names"
            )

    dataset = Dataset(
        categories=[
            Category(id=i, name=name, color=default_color(i))
            for i, name in enumerate(names)
        ]
    )
    id_by_name = {name: i for i, name in enumerate(names)}
    image_paths: dict[int, Path] = {}

    for via_path, entry, regions in parsed:
        image_file = via_path.parent / entry["filename"]
        if not image_file.is_file():
            raise DatasetFormatError(
                f"{via_path}: image '{entry['filename']}' not found next to "
                f"the annotation file"
            )
        try:
            with Image.open(image_file) as im:
                width, height = im.size
        except OSError as exc:
            raise DatasetFormatError(
                f"Cannot read image {image_file}: {exc}"
            ) from exc

        record = ImageRecord(
            id=dataset.next_image_id(),
            file_name=image_file.name,
            width=width,
            height=height,
            split=split_from_dir_name(via_path.parent.name),
        )
        dataset.images.append(record)
        image_paths[record.id] = image_file.resolve()

        for region in regions:
            context = f"{via_path.name}: {entry['filename']}"
            shape = region.get("shape_attributes")
            if not isinstance(shape, dict):
                raise DatasetFormatError(f"{context}: region without shape_attributes")
            bbox, segmentation = _shape_to_geometry(shape, context, warnings)
            label = _label_of(region)
            if label is None:
                if fallback is None:
                    raise DatasetFormatError(
                        f"{context}: region has no class attribute while other "
                        f"regions in this dataset are labeled"
                    )
                label = fallback
            dataset.annotations.append(
                Annotation(
                    id=dataset.next_annotation_id(),
                    image_id=record.id,
                    category_id=id_by_name[label],
                    bbox=bbox,
                    segmentation=segmentation,
                )
            )

    dataset.reader_warnings.extend(warnings)
    return dataset, image_paths
