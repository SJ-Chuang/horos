"""Dataset operations — the public Python API (R2: business logic lives here)."""

from __future__ import annotations

import hashlib
import logging
import tempfile
import zipfile
from pathlib import Path

from pydantic import BaseModel, Field

from horos.api.manifest import capability
from horos.core import formats
from horos.core.dataset import Category, Dataset, default_color
from horos.core.formats import coco as coco_format
from horos.core.formats import darknet as darknet_format
from horos.core.formats import voc as voc_format
from horos.core.formats import yolo as yolo_format
from horos.core.project import Project
from horos.core.stats import DatasetStats, compute_stats
from horos.core.validate import ValidationReport, validate_dataset
from horos.errors import (
    ClassNamesRequiredError,
    DatasetFormatError,
    ImportConflictError,
    ProjectError,
)

CONFLICT_POLICIES = ("ask", "overwrite", "skip", "rename")

logger = logging.getLogger(__name__)

__all__ = [
    "ImportSummary",
    "import_dataset",
    "import_zip",
    "export_dataset",
    "convert_dataset",
    "validate_project",
    "dataset_stats",
    "resplit",
    "list_images",
]


class ImportSummary(BaseModel):
    format: str
    num_images: int
    num_annotations: int
    num_categories: int
    instances_per_category: dict[str, int] = Field(default_factory=dict)
    split_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    #: same name AND same content as an existing image — skipped automatically
    duplicates_skipped: int = 0
    #: file names that already existed with different content
    conflict_files: list[str] = Field(default_factory=list)
    #: how the conflicts were resolved (per the on_conflict policy)
    overwritten: int = 0
    conflicts_skipped: int = 0
    renamed: int = 0


def _read_any(
    source: Path, format: str | None, *, class_names: list[str] | None = None
) -> tuple[str, Dataset, dict[int, Path]]:
    detected = format or formats.detect_format(source)
    if detected is None:
        raise DatasetFormatError(
            f"Could not detect a supported dataset format under {source}. Expected "
            f"a COCO '_annotations.coco.json', a YOLO 'data.yaml', Pascal VOC "
            f"<annotation> XML files, or Darknet label .txt files next to images."
        )
    if detected == "coco":
        dataset, image_paths = coco_format.read_coco(source)
    elif detected == "yolo":
        dataset, image_paths = yolo_format.read_yolo(source)
    elif detected == "voc":
        dataset, image_paths = voc_format.read_voc(source)
    elif detected == "darknet":
        dataset, image_paths = darknet_format.read_darknet(source, class_names=class_names)
    else:
        raise DatasetFormatError(f"Unsupported dataset format '{detected}'")
    return detected, dataset, image_paths


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


@capability(
    "dataset.import",
    summary="Import a COCO or YOLO dataset into a project (format auto-detected)",
    web_route="/api/v1/dataset/import",
    web_methods=("POST",),
    cli="import",
)
def import_dataset(
    project: Project,
    source: Path | str,
    *,
    format: str | None = None,
    copy_images: bool = True,
    boundary: Path | None = None,
    on_conflict: str = "ask",
    class_names: list[str] | None = None,
    require_class_names: bool = False,
) -> ImportSummary:
    """Import a dataset directory (or annotation file) into the project.

    Categories are merged by name with any the project already has. Images are
    copied into the project by default; copy_images=False stores absolute-path
    references instead. When `boundary` is given, any annotation-referenced
    file resolving outside that directory is refused — import_zip passes the
    extraction dir so an uploaded archive cannot pull in server-side files.

    File-name conflicts: an incoming image whose name already exists in the
    project with identical content is skipped automatically; with different
    content it is resolved per `on_conflict` — "ask" (default) raises
    ImportConflictError listing the names without touching the project,
    "overwrite" replaces the image and its annotations, "skip" keeps the
    existing one, "rename" imports under an auto-suffixed name.

    `class_names` supplies Darknet class names when no _darknet.labels exists
    (placeholder index names plus a warning otherwise); `require_class_names`
    makes that case raise ClassNamesRequiredError instead — the WebUI upload
    path uses it to show an editable name list.
    """
    if on_conflict not in CONFLICT_POLICIES:
        raise ProjectError(f"on_conflict must be one of {CONFLICT_POLICIES}")
    source = Path(source)
    if not source.exists():
        raise DatasetFormatError(f"Dataset source does not exist: {source}")
    detected, dataset, image_paths = _read_any(source, format, class_names=class_names)
    pre_warnings: list[str] = []
    if (
        detected == "darknet"
        and class_names is None
        and darknet_format.find_labels_file(source) is None
    ):
        if require_class_names:
            raise ClassNamesRequiredError(
                "This Darknet dataset has no _darknet.labels file — provide "
                "class names to import it.",
                default_names=[c.name for c in dataset.categories],
            )
        pre_warnings.append(
            f"No _darknet.labels found — class indices 0..{len(dataset.categories) - 1} "
            f"used as class names; rename them later or re-import with class_names"
        )
    if boundary is not None:
        bound = boundary.resolve()
        for path in image_paths.values():
            if not path.is_relative_to(bound):
                raise DatasetFormatError(
                    f"Dataset references a file outside the archive: {path}"
                )

    # merge categories by name
    categories = list(project.categories)
    id_by_name = {c.name: c.id for c in categories}
    category_map: dict[int, int] = {}
    for cat in dataset.categories:
        if cat.name not in id_by_name:
            new_id = max((c.id for c in categories), default=0) + 1
            categories.append(
                Category(id=new_id, name=cat.name, color=default_color(len(categories)))
            )
            id_by_name[cat.name] = new_id
        category_map[cat.id] = id_by_name[cat.name]
    project.set_categories(categories)

    # conflict prescan — nothing is written until every decision is known
    existing_by_name = {r.file_name: r for r in project.list_images()}
    actions: dict[int, str] = {}  # image.id -> duplicate | overwrite | skip | rename
    conflict_files: list[str] = []
    for image in dataset.images:
        src = image_paths.get(image.id)
        existing = existing_by_name.get(image.file_name)
        if src is None or existing is None or not src.exists():
            continue
        if _sha256(src) == _sha256(project.image_path(existing)):
            actions[image.id] = "duplicate"
        else:
            conflict_files.append(image.file_name)
            actions[image.id] = on_conflict
    if conflict_files and on_conflict == "ask":
        raise ImportConflictError(
            f"{len(conflict_files)} image(s) already exist with different content: "
            f"{', '.join(conflict_files[:10])}"
            f"{' …' if len(conflict_files) > 10 else ''}. Nothing was imported — "
            f"retry with on_conflict='overwrite', 'skip', or 'rename'.",
            conflicts=conflict_files,
        )

    warnings = pre_warnings
    index = project._load_image_index()
    image_map: dict[int, int] = {}
    overwritten_ids: set[int] = set()
    duplicates_skipped = conflicts_skipped = renamed = 0
    for image in dataset.images:
        src = image_paths.get(image.id)
        if src is None or not src.exists():
            warnings.append(
                f"Image file missing for '{image.file_name}' — record skipped"
            )
            continue
        action = actions.get(image.id)
        if action == "duplicate":
            duplicates_skipped += 1
            continue
        if action == "skip":
            conflicts_skipped += 1
            continue
        if action == "overwrite":
            record = project.replace_image(
                existing_by_name[image.file_name].id,
                src,
                width=image.width,
                height=image.height,
                split=image.split,
                copy=copy_images,
                _index=index,
            )
            overwritten_ids.add(record.id)
        else:  # new image, or conflict resolved by rename (add_image auto-suffixes)
            record = project.add_image(
                src,
                width=image.width,
                height=image.height,
                split=image.split,
                copy=copy_images,
                _index=index,
            )
            if action == "rename":
                renamed += 1
        image_map[image.id] = record.id
    project._save_image_index(index)

    imported_annotations = 0
    instances: dict[str, int] = {}
    for old_image_id, new_image_id in image_map.items():
        annotations = []
        current = project.load_annotations(new_image_id)
        next_id = 1
        for ann in dataset.annotations_for(old_image_id):
            new_cat = category_map[ann.category_id]
            annotations.append(
                ann.model_copy(
                    update={
                        "id": next_id,
                        "image_id": new_image_id,
                        "category_id": new_cat,
                    }
                )
            )
            next_id += 1
            name = next(c.name for c in categories if c.id == new_cat)
            instances[name] = instances.get(name, 0) + 1
            imported_annotations += 1
        if annotations or new_image_id in overwritten_ids:
            # an overwritten image must not keep its old annotations, so an
            # empty incoming set still saves
            project.save_annotations(
                new_image_id, annotations, expected_version=current.version
            )

    split_counts: dict[str, int] = {}
    for image in dataset.images:
        if image.id in image_map:
            split_counts[image.split] = split_counts.get(image.split, 0) + 1

    logger.info(
        "imported %s dataset from %s: %d images, %d annotations",
        detected, source, len(image_map), imported_annotations,
    )
    return ImportSummary(
        format=detected,
        num_images=len(image_map),
        num_annotations=imported_annotations,
        num_categories=len(categories),
        instances_per_category=instances,
        split_counts=split_counts,
        warnings=warnings,
        duplicates_skipped=duplicates_skipped,
        conflict_files=conflict_files,
        overwritten=len(overwritten_ids),
        conflicts_skipped=conflicts_skipped,
        renamed=renamed,
    )


def _safe_extract(zip_path: Path, target: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            member_path = Path(member)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise DatasetFormatError(
                    f"Zip contains an unsafe path: {member!r}"
                )
        zf.extractall(target)


@capability(
    "dataset.import_zip",
    summary="Import a zipped COCO/YOLO dataset (the WebUI upload path)",
    web_route="/api/v1/dataset/upload",
    web_methods=("POST",),
    cli=None,
    not_cli_because="The CLI imports directories directly via 'import'.",
)
def import_zip(
    project: Project,
    zip_path: Path | str,
    *,
    on_conflict: str = "ask",
    class_names: list[str] | None = None,
    require_class_names: bool = False,
) -> ImportSummary:
    """Extract a dataset zip to a temp dir and import it (always copies)."""
    zip_path = Path(zip_path)
    if not zipfile.is_zipfile(zip_path):
        raise DatasetFormatError(f"{zip_path} is not a valid zip archive")
    with tempfile.TemporaryDirectory(prefix="horos_upload_") as tmp:
        _safe_extract(zip_path, Path(tmp))
        return import_dataset(
            project,
            Path(tmp),
            copy_images=True,
            boundary=Path(tmp),
            on_conflict=on_conflict,
            class_names=class_names,
            require_class_names=require_class_names,
        )


@capability(
    "dataset.export",
    summary="Export the project's dataset as COCO or YOLO",
    web_route="/api/v1/dataset/export",
    web_methods=("POST",),
    cli="export",
)
def export_dataset(
    project: Project, out_dir: Path | str, *, format: str = "coco"
) -> Path:
    """Write the project dataset to `out_dir` in the requested format."""
    dataset = project.to_dataset()
    image_paths = {i.id: project.image_path(i) for i in dataset.images}
    out_dir = Path(out_dir)
    if format == "coco":
        written = coco_format.write_coco(
            dataset, out_dir, image_paths=image_paths, copy_images=True
        )
        return written[0].parent.parent if len(written) > 1 else written[0]
    if format == "yolo":
        return yolo_format.write_yolo(dataset, out_dir, image_paths=image_paths)
    raise DatasetFormatError(f"Unsupported export format '{format}' (coco|yolo)")


@capability(
    "dataset.convert",
    summary="Convert a dataset between COCO and YOLO without creating a project",
    not_web_because="Server-side path-to-path conversion is a CLI/scripting concern.",
    cli="convert",
)
def convert_dataset(
    source: Path | str,
    out_dir: Path | str,
    *,
    to_format: str,
    from_format: str | None = None,
) -> Path:
    """One-shot format conversion (E1-S2): read `source`, write to `out_dir`."""
    detected, dataset, image_paths = _read_any(Path(source), from_format)
    if to_format == detected:
        raise DatasetFormatError(f"Source is already in format '{to_format}'")
    if to_format == "coco":
        written = coco_format.write_coco(
            dataset, Path(out_dir), image_paths=image_paths, copy_images=True
        )
        return written[0].parent.parent if len(written) > 1 else written[0]
    if to_format == "yolo":
        return yolo_format.write_yolo(dataset, Path(out_dir), image_paths=image_paths)
    raise DatasetFormatError(f"Unsupported target format '{to_format}' (coco|yolo)")


@capability(
    "dataset.validate",
    summary="Validate the project dataset and report structured issues",
    web_route="/api/v1/dataset/validation",
    web_methods=("GET",),
    cli="validate",
)
def validate_project(project: Project) -> ValidationReport:
    """Run the dataset validator over the project's current data (E1-T6)."""
    dataset = project.to_dataset()
    image_paths = {i.id: project.image_path(i) for i in dataset.images}
    return validate_dataset(dataset, image_paths=image_paths)


@capability(
    "dataset.stats",
    summary="Compute dataset statistics (feeds hyperparameter derivation)",
    web_route="/api/v1/dataset/stats",
    web_methods=("GET",),
    cli="stats",
)
def dataset_stats(project: Project) -> DatasetStats:
    """Class distribution, relative object area, image sizes, splits (E1-T7)."""
    return compute_stats(project.to_dataset())


@capability(
    "dataset.resplit",
    summary="Re-split images into train/valid/test with a fixed seed",
    web_route="/api/v1/dataset/split",
    web_methods=("POST",),
    cli="split",
)
def resplit(
    project: Project,
    *,
    train: float = 0.8,
    valid: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Randomly reassign splits (deterministic under `seed`). No symlinks —
    the split is an attribute on the image record (R7)."""
    import random

    total = train + valid + test
    if abs(total - 1.0) > 1e-6:
        raise ProjectError(f"Split ratios must sum to 1.0, got {total}")
    images = project.list_images()
    if not images:
        raise ProjectError("Project has no images to split")
    rng = random.Random(seed)
    ids = [i.id for i in images]
    rng.shuffle(ids)
    n = len(ids)
    n_train = round(n * train)
    n_valid = round(n * valid)
    assignment: dict[int, str] = {}
    for pos, image_id in enumerate(ids):
        if pos < n_train:
            assignment[image_id] = "train"
        elif pos < n_train + n_valid:
            assignment[image_id] = "valid"
        else:
            assignment[image_id] = "test"
    project.update_image_splits(assignment)
    counts: dict[str, int] = {"train": 0, "valid": 0, "test": 0}
    for split in assignment.values():
        counts[split] += 1
    return counts


@capability(
    "dataset.images",
    summary="List the project's images with size and split",
    web_route="/api/v1/images",
    web_methods=("GET",),
    cli=None,
    not_cli_because="Covered by 'stats'; a raw image list is a UI need.",
)
def list_images(project: Project):
    """All image records in the project."""
    return project.list_images()
