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


_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def unsupported_format_hint(root: Path) -> str | None:
    """Name the format an unrecognized dataset looks like, for error messages (E1-S4).

    Only called after detect_format() returned None, so a data.yaml is known
    to be absent.
    """
    root = Path(root)
    if any(root.rglob("_darknet.labels")):
        return "Darknet"
    for xml_path in root.rglob("*.xml"):
        try:
            head = xml_path.read_text(encoding="utf-8", errors="ignore")[:2048]
        except OSError:
            continue
        if "<annotation" in head:
            return "Pascal VOC"
    for txt in root.rglob("*.txt"):
        if any(txt.with_suffix(ext).exists() for ext in _IMAGE_SUFFIXES):
            return "Darknet-style YOLO (label .txt files without a data.yaml)"
    return None
