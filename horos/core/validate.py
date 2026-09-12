"""Dataset validator (E1-T6).

Six failure classes, each with a precise, human-actionable message — never a
silent pass (E1-S4):

  missing_image_file      annotation JSON references an image file that isn't there
  bbox_out_of_bounds      box extends past the image edge
  invalid_box_size        zero or negative width/height
  unknown_category        annotation points at a category id that doesn't exist
  invalid_polygon         odd coordinate count, fewer than 3 points, or no area
  polygon_bbox_mismatch   the polygons' extent is not the bbox they belong to

Plus one warning-level check: non_contiguous_category_ids (common after manual
COCO surgery; harmless to horos but breaks some external tools).

bbox_out_of_bounds is tiered: a box past the edge by at most FIXABLE_OVERSHOOT
pixels is annotation-tool jitter (normalized-coordinate round-trips, off-by-one
exports), semantically an object touching the frame — a warning, marked
`fixable`, repaired by clamping (`horos validate --fix` / the UI's Fix button).
A larger overshoot usually means genuinely broken labels and stays an error.

polygon_bbox_mismatch is tiered the same way. A polygon is the finer geometry,
so when the polygons span most of their bbox (at least POLYGON_COVERAGE of its
width and height) but the bbox drifted — a box widened by mask specks, a
polygon edited without its box — the fix is to recompute the bbox from the
polygons: a `fixable` warning. When the polygons cover far less than the box,
they are a fragment of the object (a stray blob traced instead of the mask)
and only redrawing or re-segmenting helps: an error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from horos.core.dataset import Annotation, Dataset, clamp_to_image

IssueKind = Literal[
    "missing_image_file",
    "bbox_out_of_bounds",
    "invalid_box_size",
    "unknown_category",
    "invalid_polygon",
    "polygon_bbox_mismatch",
    "non_contiguous_category_ids",
]

_EDGE_TOLERANCE = 1e-6

#: Largest out-of-bounds overshoot (pixels) still treated as auto-fixable
#: tool jitter rather than a broken label.
FIXABLE_OVERSHOOT = 2.0

#: Largest disagreement (pixels, per edge) between a bbox and the extent of
#: its polygons before it is reported. Mask-derived boxes count pixels while
#: polygon vertices sit on pixel centres, so a 1px difference is normal.
POLYGON_BBOX_TOLERANCE = 2.0

#: Polygons spanning at least this fraction of the bbox's width AND height are
#: the object itself (bbox drifted -> fixable); less means a fragment (error).
POLYGON_COVERAGE = 0.5


def bbox_overshoot(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> float:
    """How far (pixels) the box extends past the image bounds; 0.0 if inside."""
    x, y, w, h = bbox
    return max(0.0, -x, -y, x + w - width, y + h - height)


def clamp_fix(annotation: Annotation, width: int, height: int) -> Annotation | None:
    """The clamped annotation if this is an auto-fixable overshoot, else None.

    Fixable means: out of bounds, by at most FIXABLE_OVERSHOOT pixels, and
    still a positive-size box after clamping. The validator uses this to mark
    issues `fixable`; the fixer applies exactly the same decision (E1-S4:
    what the report promises is what the fix does).
    """
    overshoot = bbox_overshoot(annotation.bbox, width, height)
    if overshoot <= _EDGE_TOLERANCE or overshoot > FIXABLE_OVERSHOOT:
        return None
    clamped = clamp_to_image(annotation, width, height)
    if clamped.bbox[2] <= 0 or clamped.bbox[3] <= 0:
        return None
    return clamped


def polygon_area(polygon: list[float]) -> float:
    """Shoelace area of one flat [x1, y1, ...] polygon; 0.0 when degenerate."""
    n = len(polygon) // 2
    if n < 3:
        return 0.0
    xs, ys = polygon[0::2], polygon[1::2]
    twice = sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))
    return abs(twice) / 2.0


def polygons_extent(
    segmentation: list[list[float]],
) -> tuple[float, float, float, float] | None:
    """COCO xywh bounds of every well-formed polygon together; None if none."""
    xs = [v for poly in segmentation if len(poly) >= 6 and len(poly) % 2 == 0 for v in poly[0::2]]
    ys = [v for poly in segmentation if len(poly) >= 6 and len(poly) % 2 == 0 for v in poly[1::2]]
    if not xs:
        return None
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def polygon_bbox_gap(annotation: Annotation) -> float:
    """Largest per-edge distance (pixels) between the bbox and its polygons'
    extent; 0.0 when they agree or the annotation has no polygon."""
    extent = polygons_extent(annotation.segmentation)
    if extent is None:
        return 0.0
    x, y, w, h = annotation.bbox
    ex, ey, ew, eh = extent
    return max(abs(ex - x), abs(ey - y), abs(ex + ew - (x + w)), abs(ey + eh - (y + h)))


def polygon_bbox_fix(annotation: Annotation) -> Annotation | None:
    """The annotation with its bbox recomputed from its polygons if that is
    the auto-fixable kind of mismatch, else None.

    Fixable means: the polygons' extent disagrees with the bbox by more than
    POLYGON_BBOX_TOLERANCE, yet spans at least POLYGON_COVERAGE of the bbox's
    width and height — the polygons are the object and the box drifted. A
    polygon covering less is a fragment; recomputing the box from it would
    shrink a correct box onto a wrong polygon, so it is left for a human.
    """
    extent = polygons_extent(annotation.segmentation)
    if extent is None or polygon_bbox_gap(annotation) <= POLYGON_BBOX_TOLERANCE:
        return None
    _, _, w, h = annotation.bbox
    if extent[2] < POLYGON_COVERAGE * w or extent[3] < POLYGON_COVERAGE * h:
        return None
    if extent[2] <= 0 or extent[3] <= 0:
        return None
    return annotation.model_copy(update={"bbox": extent})


class ValidationIssue(BaseModel):
    kind: IssueKind
    level: Literal["error", "warning"]
    message: str
    image_id: int | None = None
    annotation_id: int | None = None
    #: the image's file name, so a report reader (and the UI's "open" link)
    #: can find the picture without resolving ids
    file_name: str | None = None
    #: True when `horos validate --fix` (or the UI's Fix button) repairs this
    fixable: bool = False


class ValidationReport(BaseModel):
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for issue in self.issues:
            out[issue.kind] = out.get(issue.kind, 0) + 1
        return out


def validate_dataset(
    dataset: Dataset,
    images_root: Path | None = None,
    *,
    image_paths: dict[int, Path] | None = None,
) -> ValidationReport:
    """Validate a dataset snapshot.

    File existence is checked when either `images_root` (paths resolved as
    root/file_name) or an explicit `image_paths` map is provided.
    """
    issues: list[ValidationIssue] = []
    images_by_id = {i.id: i for i in dataset.images}
    category_ids = {c.id for c in dataset.categories}

    if images_root is not None or image_paths is not None:
        for record in dataset.images:
            if image_paths is not None:
                path = image_paths.get(record.id)
            else:
                path = Path(images_root) / Path(record.file_name)  # type: ignore[arg-type]
            if path is None or not path.exists():
                issues.append(
                    ValidationIssue(
                        kind="missing_image_file",
                        level="error",
                        message=(
                            f"Image {record.id} ('{record.file_name}') is referenced "
                            f"by the dataset but missing from {images_root}"
                        ),
                        image_id=record.id,
                        file_name=record.file_name,
                    )
                )

    for ann in dataset.annotations:
        image = images_by_id.get(ann.image_id)
        file_name = image.file_name if image is not None else None
        where = (
            f"on '{file_name}' (image {ann.image_id})"
            if file_name
            else f"on image {ann.image_id}"
        )
        x, y, w, h = ann.bbox

        if w <= 0 or h <= 0:
            issues.append(
                ValidationIssue(
                    kind="invalid_box_size",
                    level="error",
                    message=(
                        f"Annotation {ann.id} {where} has non-positive size (w={w}, h={h})"
                    ),
                    image_id=ann.image_id,
                    annotation_id=ann.id,
                    file_name=file_name,
                )
            )
        elif (
            image is not None
            and bbox_overshoot(ann.bbox, image.width, image.height) > _EDGE_TOLERANCE
        ):
            overshoot = bbox_overshoot(ann.bbox, image.width, image.height)
            fixable = clamp_fix(ann, image.width, image.height) is not None
            issues.append(
                ValidationIssue(
                    kind="bbox_out_of_bounds",
                    level="warning" if fixable else "error",
                    message=(
                        f"Annotation {ann.id} {where}: bbox ({x:.1f}, {y:.1f}, {w:.1f}, "
                        f"{h:.1f}) exceeds the image bounds "
                        f"({image.width}x{image.height}) by {overshoot:.2f}px"
                        + (
                            " — auto-fixable: run 'horos validate --fix'"
                            if fixable
                            else ""
                        )
                    ),
                    image_id=ann.image_id,
                    annotation_id=ann.id,
                    file_name=file_name,
                    fixable=fixable,
                )
            )

        if ann.category_id not in category_ids:
            issues.append(
                ValidationIssue(
                    kind="unknown_category",
                    level="error",
                    message=(
                        f"Annotation {ann.id} {where} references category id "
                        f"{ann.category_id}, which is not defined "
                        f"(defined ids: {sorted(category_ids)})"
                    ),
                    image_id=ann.image_id,
                    annotation_id=ann.id,
                    file_name=file_name,
                )
            )

        for poly_index, poly in enumerate(ann.segmentation):
            if len(poly) % 2 != 0 or len(poly) < 6:
                issues.append(
                    ValidationIssue(
                        kind="invalid_polygon",
                        level="error",
                        message=(
                            f"Annotation {ann.id} {where} polygon #{poly_index} has "
                            f"{len(poly)} coordinates; polygons need an even count "
                            f"of at least 6 (3 points)"
                        ),
                        image_id=ann.image_id,
                        annotation_id=ann.id,
                        file_name=file_name,
                    )
                )
            elif polygon_area(poly) < _EDGE_TOLERANCE:
                issues.append(
                    ValidationIssue(
                        kind="invalid_polygon",
                        level="error",
                        message=(
                            f"Annotation {ann.id} {where} polygon #{poly_index} has no "
                            f"area (its {len(poly) // 2} points are collinear or "
                            f"coincide); redraw it or drop the polygon"
                        ),
                        image_id=ann.image_id,
                        annotation_id=ann.id,
                        file_name=file_name,
                    )
                )

        gap = polygon_bbox_gap(ann)
        if gap > POLYGON_BBOX_TOLERANCE:
            extent = polygons_extent(ann.segmentation)
            fixable = polygon_bbox_fix(ann) is not None
            ex, ey, ew, eh = extent  # type: ignore[misc]  # gap > 0 implies an extent
            issues.append(
                ValidationIssue(
                    kind="polygon_bbox_mismatch",
                    level="warning" if fixable else "error",
                    message=(
                        f"Annotation {ann.id} {where}: bbox ({x:.1f}, {y:.1f}, {w:.1f}, "
                        f"{h:.1f}) does not match its polygons' extent ({ex:.1f}, "
                        f"{ey:.1f}, {ew:.1f}, {eh:.1f}), off by {gap:.1f}px"
                        + (
                            " — auto-fixable: 'horos validate --fix' recomputes the "
                            "bbox from the polygon"
                            if fixable
                            else (
                                f"; the polygon covers only {ew / w:.0%} x {eh / h:.0%} "
                                "of the box, so it is a fragment of the object — "
                                "redraw it, or turn the box into a polygon again "
                                "(boxes-to-polygons) after removing the polygon"
                            )
                        )
                    ),
                    image_id=ann.image_id,
                    annotation_id=ann.id,
                    file_name=file_name,
                    fixable=fixable,
                )
            )

    sorted_ids = sorted(category_ids)
    if sorted_ids and sorted_ids != list(range(sorted_ids[0], sorted_ids[0] + len(sorted_ids))):
        issues.append(
            ValidationIssue(
                kind="non_contiguous_category_ids",
                level="warning",
                message=(
                    f"Category ids are not contiguous: {sorted_ids}. horos handles "
                    f"this, but some external tools assume contiguous ids."
                ),
            )
        )

    return ValidationReport(issues=issues)
