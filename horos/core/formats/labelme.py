"""LabelMe JSON reader and writer (one `<image stem>.json` per image).

Read: every JSON under the source that has LabelMe's `shapes` + `imagePath`
keys is an annotation file; the image is resolved from `imagePath` relative to
the JSON (backslashes from Windows-made files are normalised), then from a
same-stem image next to the JSON, and finally restored from the embedded
`imageData` when the file itself is missing. Images with no JSON at all are
imported as unannotated (LabelMe never writes a file until you draw), so they
can be annotated in horos. Split comes from the containing directory name.

Shape mapping (E1-S4 — nothing is dropped silently):

  rectangle           → bbox (any two opposite corners, or four corners)
  polygon             → polygon annotation, bbox derived
  circle              → enclosing bbox, counted in a warning
  line/linestrip/
  point/points/mask   → skipped, counted per type in a warning

Polygons on the same image that share a non-empty `group_id` and a label are
one instance with several parts — they merge into one annotation with a
multi-polygon segmentation (COCO semantics). Rectangles never merge.

Write: the same layout, split directories like the COCO/YOLO writers, floats
at full precision so a write→read round trip is lossless. A multi-polygon
annotation is written as several polygon shapes sharing a `group_id` so it
re-merges on import.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
from collections import Counter
from pathlib import Path, PureWindowsPath

from PIL import Image

from horos.core.dataset import SPLITS, Annotation, Category, Dataset, ImageRecord, default_color
from horos.errors import DatasetFormatError

from . import IMAGE_SUFFIXES, split_from_dir_name

#: the LabelMe file-format version this writer emits
LABELME_VERSION = "5.3.1"

#: shapes horos cannot represent — skipped with a count, never an error
_SKIPPED_SHAPES = ("line", "linestrip", "point", "points", "mask")

#: how many JSON files format detection may open before giving up
_DETECT_LIMIT = 25


def _is_labelme(data: object) -> bool:
    return (
        isinstance(data, dict)
        and isinstance(data.get("shapes"), list)
        and isinstance(data.get("imagePath"), str)
    )


def _json_candidates(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.rglob("*.json")
        if "__MACOSX" not in path.parts  # macOS zip metadata, not data
    )


def _load(path: Path) -> object | None:
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def find_labelme_files(source: Path | str, *, limit: int | None = None) -> list[Path]:
    """JSON files under `source` that parse as LabelMe annotation files.

    With `limit`, stop after opening that many candidates — format detection
    only needs to know whether there is at least one."""
    source = Path(source)
    found: list[Path] = []
    for index, path in enumerate(_json_candidates(source)):
        if limit is not None and index >= limit:
            break
        if _is_labelme(_load(path)):
            found.append(path)
    return found


def looks_like_labelme(source: Path | str) -> bool:
    """Format detection: is there at least one LabelMe JSON among the first
    few candidates? Bounded so detection stays cheap on huge trees."""
    return bool(find_labelme_files(source, limit=_DETECT_LIMIT))


def _norm_group(value: object) -> str | None:
    """LabelMe writes group_id as int, str or null; '' and None mean ungrouped."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _points(shape: dict, context: str) -> list[tuple[float, float]]:
    raw = shape.get("points")
    if not isinstance(raw, list):
        raise DatasetFormatError(f"{context}: shape has no 'points' list")
    points: list[tuple[float, float]] = []
    for point in raw:
        if not (isinstance(point, list | tuple) and len(point) == 2):
            raise DatasetFormatError(f"{context}: malformed point {point!r}")
        points.append((float(point[0]), float(point[1])))
    return points


def _bbox_of(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return (x0, y0, x1 - x0, y1 - y0)


def _resolve_image(json_path: Path, data: dict, warnings: list[str]) -> Path:
    """Locate (or restore) the image a LabelMe JSON describes."""
    image_path_value = str(data["imagePath"]).strip()
    # LabelMe on Windows writes "..\\images\\x.jpg"; PureWindowsPath parses both
    relative = Path(*PureWindowsPath(image_path_value).parts) if image_path_value else None
    if relative is not None:
        candidate = json_path.parent / relative
        if candidate.is_file():
            return candidate
    for suffix in IMAGE_SUFFIXES:
        sibling = json_path.with_suffix(suffix)
        if sibling.is_file():
            return sibling
    encoded = data.get("imageData")
    if isinstance(encoded, str) and encoded.strip():
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise DatasetFormatError(
                f"{json_path.name}: image file '{image_path_value}' is missing and the "
                f"embedded imageData is not valid base64: {exc}"
            ) from exc
        suffix = (relative.suffix if relative is not None else "") or ".png"
        target = json_path.with_suffix(suffix)
        try:
            target.write_bytes(raw)
        except OSError as exc:
            raise DatasetFormatError(
                f"{json_path.name}: image file '{image_path_value}' is missing and the "
                f"embedded imageData could not be written to {target}: {exc}"
            ) from exc
        warnings.append(
            f"{json_path.name}: image file '{image_path_value}' was missing — "
            f"restored {target.name} from the embedded imageData"
        )
        return target
    raise DatasetFormatError(
        f"{json_path.name}: image file '{image_path_value}' not found (looked next to "
        f"the JSON, then for a same-stem image) and no imageData is embedded"
    )


def _image_size(image_file: Path, data: dict) -> tuple[int, int]:
    width, height = data.get("imageWidth"), data.get("imageHeight")
    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return width, height
    try:
        with Image.open(image_file) as im:
            return im.size
    except OSError as exc:
        raise DatasetFormatError(f"Cannot read image {image_file}: {exc}") from exc


def read_labelme(source: Path | str) -> tuple[Dataset, dict[int, Path]]:
    """Read a LabelMe directory tree. Returns the Dataset plus {image_id: path}."""
    source = Path(source)
    parsed: list[tuple[Path, dict]] = []
    skipped_json = 0
    for path in _json_candidates(source):
        data = _load(path)
        if _is_labelme(data):
            parsed.append((path, data))  # type: ignore[arg-type]
        else:
            skipped_json += 1
    if not parsed:
        raise DatasetFormatError(
            f"No LabelMe annotation files (JSON with 'shapes' and 'imagePath') found "
            f"under {source}"
        )

    warnings: list[str] = []
    if skipped_json:
        warnings.append(
            f"{skipped_json} .json file(s) under the source are not LabelMe "
            f"annotation files and were ignored"
        )

    # pass 1: the class list, sorted for a stable id assignment
    labels: set[str] = set()
    for _path, data in parsed:
        for shape in data["shapes"]:
            label = shape.get("label") if isinstance(shape, dict) else None
            if isinstance(label, str) and label.strip():
                labels.add(label.strip())
    names = sorted(labels)
    dataset = Dataset(
        categories=[
            Category(id=i, name=name, color=default_color(i)) for i, name in enumerate(names)
        ]
    )
    id_by_name = {name: i for i, name in enumerate(names)}
    image_paths: dict[int, Path] = {}
    referenced: set[Path] = set()
    circles = Counter()  # type: Counter[str]
    skipped_shapes = Counter()  # type: Counter[str]

    for json_path, data in parsed:
        image_file = _resolve_image(json_path, data, warnings)
        referenced.add(image_file.resolve())
        width, height = _image_size(image_file, data)
        record = ImageRecord(
            id=dataset.next_image_id(),
            file_name=image_file.name,
            width=width,
            height=height,
            split=split_from_dir_name(json_path.parent.name),
        )
        dataset.images.append(record)
        image_paths[record.id] = image_file.resolve()

        # polygons sharing (group_id, label) merge into one multi-part instance
        grouped: dict[tuple[str, str], Annotation] = {}
        for index, shape in enumerate(data["shapes"]):
            context = f"{json_path.name}: shape #{index}"
            if not isinstance(shape, dict):
                raise DatasetFormatError(f"{context}: not an object")
            label = shape.get("label")
            if not isinstance(label, str) or not label.strip():
                raise DatasetFormatError(f"{context}: shape has no label")
            label = label.strip()
            kind = shape.get("shape_type") or "polygon"  # old LabelMe omitted it
            if kind in _SKIPPED_SHAPES:
                skipped_shapes[kind] += 1
                continue
            points = _points(shape, context)
            if kind == "rectangle":
                if len(points) < 2:
                    raise DatasetFormatError(
                        f"{context}: rectangle needs 2 corner points, got {len(points)}"
                    )
                bbox, segmentation = _bbox_of(points), []
            elif kind == "polygon":
                if len(points) < 3:
                    raise DatasetFormatError(
                        f"{context}: polygon needs ≥3 points, got {len(points)}"
                    )
                flat: list[float] = []
                for x, y in points:
                    flat.extend((x, y))
                bbox, segmentation = _bbox_of(points), [flat]
            elif kind == "circle":
                if len(points) != 2:
                    raise DatasetFormatError(
                        f"{context}: circle needs [center, edge] points, got {len(points)}"
                    )
                (cx, cy), (ex, ey) = points
                r = math.hypot(ex - cx, ey - cy)
                bbox, segmentation = (cx - r, cy - r, 2 * r, 2 * r), []
                circles[json_path.name] += 1
            else:
                raise DatasetFormatError(
                    f"{context}: unsupported LabelMe shape_type '{kind}' (supported: "
                    f"rectangle, polygon, circle; skipped: {', '.join(_SKIPPED_SHAPES)})"
                )

            group = _norm_group(shape.get("group_id")) if segmentation else None
            if group is not None:
                existing = grouped.get((group, label))
                if existing is not None:
                    existing.segmentation.extend(segmentation)
                    x0 = min(existing.bbox[0], bbox[0])
                    y0 = min(existing.bbox[1], bbox[1])
                    x1 = max(existing.bbox[0] + existing.bbox[2], bbox[0] + bbox[2])
                    y1 = max(existing.bbox[1] + existing.bbox[3], bbox[1] + bbox[3])
                    existing.bbox = (x0, y0, x1 - x0, y1 - y0)
                    continue
            annotation = Annotation(
                id=dataset.next_annotation_id(),
                image_id=record.id,
                category_id=id_by_name[label],
                bbox=bbox,
                segmentation=segmentation,
            )
            dataset.annotations.append(annotation)
            if group is not None:
                grouped[(group, label)] = annotation

    # images LabelMe never wrote a JSON for: not yet annotated, still part of the set
    negatives = 0
    for image_file in sorted(source.rglob("*")):
        if image_file.suffix.lower() not in IMAGE_SUFFIXES or not image_file.is_file():
            continue
        if "__MACOSX" in image_file.parts or image_file.resolve() in referenced:
            continue
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
        negatives += 1

    if negatives:
        warnings.append(
            f"{negatives} image(s) have no LabelMe JSON — imported as unannotated images"
        )
    if circles:
        total = sum(circles.values())
        warnings.append(
            f"{total} circle shape(s) imported as their enclosing bounding box "
            f"(e.g. {next(iter(circles))})"
        )
    for kind, count in sorted(skipped_shapes.items()):
        warnings.append(
            f"{count} '{kind}' shape(s) skipped — horos imports rectangles, polygons "
            f"and circles only"
        )
    dataset.reader_warnings.extend(warnings)
    return dataset, image_paths


def _shapes_for(annotation: Annotation, label: str) -> list[dict]:
    if not annotation.segmentation:
        x, y, w, h = annotation.bbox
        return [
            {
                "label": label,
                "points": [[x, y], [x + w, y + h]],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
            }
        ]
    # several parts of one instance share a group_id so they re-merge on read
    group = annotation.id if len(annotation.segmentation) > 1 else None
    return [
        {
            "label": label,
            "points": [[x, y] for x, y in zip(polygon[0::2], polygon[1::2], strict=True)],
            "group_id": group,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
        }
        for polygon in annotation.segmentation
    ]


def write_labelme(
    dataset: Dataset,
    out_dir: Path,
    *,
    image_paths: dict[int, Path] | None = None,
    copy_images: bool = True,
) -> Path:
    """Write one LabelMe JSON per image under out/<split>/ (empty splits are
    skipped, as in the COCO/YOLO writers). Every image gets a JSON, even with
    no shapes, so unannotated images and their split survive a round trip.
    Returns `out_dir`."""
    import shutil

    out_dir = Path(out_dir)
    names = {c.id: c.name for c in dataset.categories}
    for split in SPLITS:
        images = dataset.images_in_split(split)
        if not images:
            continue
        target_dir = out_dir / split
        target_dir.mkdir(parents=True, exist_ok=True)
        for record in images:
            shapes: list[dict] = []
            for annotation in dataset.annotations_for(record.id):
                shapes.extend(_shapes_for(annotation, names[annotation.category_id]))
            image_name = Path(record.file_name).name
            payload = {
                "version": LABELME_VERSION,
                "flags": {},
                "shapes": shapes,
                "imagePath": image_name,
                "imageData": None,
                "imageHeight": record.height,
                "imageWidth": record.width,
            }
            json_path = target_dir / f"{Path(image_name).stem}.json"
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            if copy_images and image_paths:
                src = image_paths.get(record.id)
                if src and src.exists():
                    shutil.copy2(src, target_dir / image_name)
    return out_dir
