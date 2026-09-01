"""Dataset format codecs. Each format reads to / writes from `horos.core.dataset.Dataset`.

COCO and YOLO support read and write; Pascal VOC, Darknet, and VIA (VGG Image
Annotator) are import-only (design decision: users bring legacy data in,
horos exports COCO/YOLO).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

DatasetFormat = Literal["coco", "yolo", "voc", "darknet", "via"]

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_SPLIT_BY_DIR = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}


def split_from_dir_name(name: str) -> str:
    """Map a containing-directory name to a split, defaulting to train."""
    return _SPLIT_BY_DIR.get(name.lower(), "train")


def _looks_like_voc(root: Path) -> bool:
    for xml_path in root.rglob("*.xml"):
        try:
            head = xml_path.read_text(encoding="utf-8", errors="ignore")[:2048]
        except OSError:
            continue
        if "<annotation" in head:
            return True
    return False


def _looks_like_darknet(root: Path) -> bool:
    if any(root.rglob("_darknet.labels")):
        return True
    # bare YOLO-style label lines next to same-stem images, without a data.yaml
    for txt in root.rglob("*.txt"):
        if not any(txt.with_suffix(ext).exists() for ext in IMAGE_SUFFIXES):
            continue
        for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            tokens = line.split()
            if not tokens:
                continue
            try:
                int(tokens[0])
                [float(t) for t in tokens[1:]]
            except ValueError:
                break
            if len(tokens) >= 5:
                return True
            break
    return False


def detect_format(root: Path) -> DatasetFormat | None:
    """Best-effort format detection for a dataset directory (used by zip import)."""
    root = Path(root)
    if any(root.rglob("_annotations.coco.json")) or any(root.glob("*.coco.json")):
        return "coco"
    if any(root.rglob("data.yaml")) or any(root.rglob("data.yml")):
        return "yolo"
    from . import via as via_format

    if via_format.find_via_files(root):
        return "via"
    if _looks_like_voc(root):
        return "voc"
    if _looks_like_darknet(root):
        return "darknet"
    # A bare COCO export: a single .json next to an images dir
    json_files = [p for p in root.glob("*.json")]
    if len(json_files) == 1:
        return "coco"
    return None
