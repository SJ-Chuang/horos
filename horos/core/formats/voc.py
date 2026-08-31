"""Pascal VOC XML reader (import-only by design decision — horos exports COCO/YOLO).

Conventions handled (Roboflow VOC layout and plain folders alike):
- one <annotation> XML per image, sitting next to the image file
- split inferred from the containing directory name (train/valid/val/test),
  defaulting to train
- bounding boxes only; VOC has no standard polygon representation
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from horos.core.dataset import Annotation, Category, Dataset, ImageRecord, default_color
from horos.errors import DatasetFormatError

from . import IMAGE_SUFFIXES, split_from_dir_name


def _parse_xml(xml_path: Path) -> ET.Element | None:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise DatasetFormatError(f"Cannot parse VOC XML {xml_path}: {exc}") from exc
    return root if root.tag == "annotation" else None


def _text(node: ET.Element, tag: str, xml_path: Path) -> str:
    child = node.find(tag)
    if child is None or child.text is None:
        raise DatasetFormatError(f"{xml_path}: missing <{tag}> element")
    return child.text.strip()


def _image_for(xml_path: Path, root: ET.Element) -> Path | None:
    filename = root.findtext("filename", "").strip()
    if filename and (xml_path.parent / filename).is_file():
        return xml_path.parent / filename
    for suffix in IMAGE_SUFFIXES:
        sibling = xml_path.with_suffix(suffix)
        if sibling.is_file():
            return sibling
    return None


def read_voc(source: Path | str) -> tuple[Dataset, dict[int, Path]]:
    """Read a VOC dataset directory. Returns the Dataset plus {image_id: path}."""
    source = Path(source)
    parsed: list[tuple[Path, ET.Element]] = []
    for xml_path in sorted(source.rglob("*.xml")):
        root = _parse_xml(xml_path)
        if root is not None:
            parsed.append((xml_path, root))
    if not parsed:
        raise DatasetFormatError(f"No VOC <annotation> XML files found under {source}")

    names = sorted(
        {
            obj.findtext("name", "").strip()
            for _, root in parsed
            for obj in root.iter("object")
        }
        - {""}
    )
    dataset = Dataset(
        categories=[
            Category(id=i, name=name, color=default_color(i))
            for i, name in enumerate(names)
        ]
    )
    id_by_name = {name: i for i, name in enumerate(names)}
    image_paths: dict[int, Path] = {}

    for xml_path, root in parsed:
        image_file = _image_for(xml_path, root)
        if image_file is None:
            raise DatasetFormatError(
                f"{xml_path}: no image file found for this annotation"
            )
        size = root.find("size")
        if size is not None and size.findtext("width") and size.findtext("height"):
            width = int(float(_text(size, "width", xml_path)))
            height = int(float(_text(size, "height", xml_path)))
        else:
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
            split=split_from_dir_name(xml_path.parent.name),
        )
        dataset.images.append(record)
        image_paths[record.id] = image_file.resolve()

        for obj in root.iter("object"):
            name = obj.findtext("name", "").strip()
            box = obj.find("bndbox")
            if not name or box is None:
                raise DatasetFormatError(
                    f"{xml_path}: <object> without <name> or <bndbox>"
                )
            xmin = float(_text(box, "xmin", xml_path))
            ymin = float(_text(box, "ymin", xml_path))
            xmax = float(_text(box, "xmax", xml_path))
            ymax = float(_text(box, "ymax", xml_path))
            if xmax <= xmin or ymax <= ymin:
                raise DatasetFormatError(
                    f"{xml_path}: degenerate box ({xmin}, {ymin}, {xmax}, {ymax})"
                )
            dataset.annotations.append(
                Annotation(
                    id=dataset.next_annotation_id(),
                    image_id=record.id,
                    category_id=id_by_name[name],
                    bbox=(xmin, ymin, xmax - xmin, ymax - ymin),
                )
            )
    return dataset, image_paths
