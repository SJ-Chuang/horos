"""Dataset validator (E1-T6).

Five common failure classes, each with a precise, human-actionable message —
never a silent pass (E1-S4):

  missing_image_file      annotation JSON references an image file that isn't there
  bbox_out_of_bounds      box extends past the image edge
  invalid_box_size        zero or negative width/height
  unknown_category        annotation points at a category id that doesn't exist
  invalid_polygon         odd coordinate count or fewer than 3 points

Plus one warning-level check: non_contiguous_category_ids (common after manual
COCO surgery; harmless to horos but breaks some external tools).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from horos.core.dataset import Dataset

IssueKind = Literal[
    "missing_image_file",
    "bbox_out_of_bounds",
    "invalid_box_size",
    "unknown_category",
    "invalid_polygon",
    "non_contiguous_category_ids",
]

_EDGE_TOLERANCE = 1e-6


class ValidationIssue(BaseModel):
    kind: IssueKind
    level: Literal["error", "warning"]
    message: str
    image_id: int | None = None
    annotation_id: int | None = None


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
                    )
                )

    for ann in dataset.annotations:
        image = images_by_id.get(ann.image_id)
        x, y, w, h = ann.bbox

        if w <= 0 or h <= 0:
            issues.append(
                ValidationIssue(
                    kind="invalid_box_size",
                    level="error",
                    message=(
                        f"Annotation {ann.id} on image {ann.image_id} has "
                        f"non-positive size (w={w}, h={h})"
                    ),
                    image_id=ann.image_id,
                    annotation_id=ann.id,
                )
            )
        elif image is not None and (
            x < -_EDGE_TOLERANCE
            or y < -_EDGE_TOLERANCE
            or x + w > image.width + _EDGE_TOLERANCE
            or y + h > image.height + _EDGE_TOLERANCE
        ):
            issues.append(
                ValidationIssue(
                    kind="bbox_out_of_bounds",
                    level="error",
                    message=(
                        f"Annotation {ann.id} bbox ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}) "
                        f"exceeds image {ann.image_id} bounds "
                        f"({image.width}x{image.height})"
                    ),
                    image_id=ann.image_id,
                    annotation_id=ann.id,
                )
            )

        if ann.category_id not in category_ids:
            issues.append(
                ValidationIssue(
                    kind="unknown_category",
                    level="error",
                    message=(
                        f"Annotation {ann.id} references category id "
                        f"{ann.category_id}, which is not defined "
                        f"(defined ids: {sorted(category_ids)})"
                    ),
                    image_id=ann.image_id,
                    annotation_id=ann.id,
                )
            )

        for poly_index, poly in enumerate(ann.segmentation):
            if len(poly) % 2 != 0 or len(poly) < 6:
                issues.append(
                    ValidationIssue(
                        kind="invalid_polygon",
                        level="error",
                        message=(
                            f"Annotation {ann.id} polygon #{poly_index} has "
                            f"{len(poly)} coordinates; polygons need an even count "
                            f"of at least 6 (3 points)"
                        ),
                        image_id=ann.image_id,
                        annotation_id=ann.id,
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
