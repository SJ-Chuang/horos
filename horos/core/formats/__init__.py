"""Dataset format codecs. Each format reads to / writes from `horos.core.dataset.Dataset`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

DatasetFormat = Literal["coco", "yolo"]


def detect_format(root: Path) -> DatasetFormat | None:
    """Best-effort format detection for a dataset directory (used by zip import)."""
    root = Path(root)
    if any(root.rglob("_annotations.coco.json")) or any(root.glob("*.coco.json")):
        return "coco"
    if any(root.rglob("data.yaml")) or any(root.rglob("data.yml")):
        return "yolo"
    # A bare COCO export: a single .json next to an images dir
    json_files = [p for p in root.glob("*.json")]
    if len(json_files) == 1:
        return "coco"
    return None
