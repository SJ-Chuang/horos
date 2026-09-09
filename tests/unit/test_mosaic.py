"""Offline mosaic synthesis (E5): annotation transforms must be exact.

The whole point of composing mosaics offline from whole (uncropped) tiles is
that every bbox and polygon maps by pure scale+offset — these tests pin that
math down, because a silent error here corrupts every training snapshot."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from horos.core.mosaic import synthesize_mosaics

ANN = "_annotations.coco.json"


def _coco_dir(root: Path, sizes, boxes_per_image) -> Path:
    """A minimal COCO split dir: one solid image per size, given boxes."""
    root.mkdir(parents=True)
    images, annotations = [], []
    ann_id = 1
    for i, ((w, h), boxes) in enumerate(
        zip(sizes, boxes_per_image, strict=True), start=1
    ):
        name = f"img{i}.png"
        Image.new("RGB", (w, h), (10 * i, 20, 30)).save(root / name)
        images.append({"id": i, "file_name": name, "width": w, "height": h})
        for x, y, bw, bh in boxes:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": i,
                    "category_id": 1,
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "segmentation": [[x, y, x + bw, y, x + bw, y + bh]],
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    (root / ANN).write_text(
        json.dumps(
            {
                "categories": [{"id": 1, "name": "block"}],
                "images": images,
                "annotations": annotations,
            }
        ),
        encoding="utf-8",
    )
    return root


def _load(root: Path) -> dict:
    return json.loads((root / ANN).read_text(encoding="utf-8"))


def test_mosaics_are_appended_with_valid_geometry(tmp_path):
    root = _coco_dir(
        tmp_path / "train",
        sizes=[(100, 80)] * 4,
        boxes_per_image=[[(10, 10, 40, 30)]] * 4,
    )
    added = synthesize_mosaics(root, count=3, seed=7)
    assert added == 3

    data = _load(root)
    mosaics = [i for i in data["images"] if i["file_name"].startswith("mosaic_")]
    assert len(mosaics) == 3
    for img in mosaics:
        # canvas is the median source size
        assert (img["width"], img["height"]) == (100, 80)
        assert (root / img["file_name"]).is_file()
        anns = [a for a in data["annotations"] if a["image_id"] == img["id"]]
        # four whole tiles, one box each — nothing may be clipped or dropped
        assert len(anns) == 4
        for a in anns:
            x, y, w, h = a["bbox"]
            assert 0 <= x and 0 <= y
            assert x + w <= img["width"] + 1e-6
            assert y + h <= img["height"] + 1e-6
            assert a["area"] > 0
            # polygon points stay inside the canvas too
            xs, ys = a["segmentation"][0][0::2], a["segmentation"][0][1::2]
            assert min(xs) >= 0 and max(xs) <= img["width"] + 1e-6
            assert min(ys) >= 0 and max(ys) <= img["height"] + 1e-6
            # the polygon must land inside its own (scaled) bbox
            assert min(xs) >= x - 1e-6 and max(xs) <= x + w + 1e-6


def test_annotation_transform_is_exact_scale_and_offset(tmp_path):
    # one source image → all four tiles show it; with a fixed seed the tile
    # rects are recomputable, so every transformed box can be checked exactly
    root = _coco_dir(
        tmp_path / "train", sizes=[(200, 100)], boxes_per_image=[[(50, 20, 100, 60)]]
    )
    assert synthesize_mosaics(root, count=1, seed=3) == 1
    data = _load(root)
    mosaic = next(i for i in data["images"] if i["file_name"].startswith("mosaic_"))
    anns = [a for a in data["annotations"] if a["image_id"] == mosaic["id"]]
    assert len(anns) == 4
    # reconstruct the tile rects from the recorded boxes: each box's scale
    # factors recover its tile size, and tile origins tile the canvas exactly
    tiles = []
    for a in sorted(anns, key=lambda a: (a["bbox"][1], a["bbox"][0])):
        x, y, w, h = a["bbox"]
        sx, sy = w / 100, h / 60  # source box was 100×60
        tiles.append((round(x - 50 * sx), round(y - 20 * sy), sx, sy))
    origins = {(t[0], t[1]) for t in tiles}
    assert (0, 0) in origins and len(origins) == 4
    # the four tiles' widths/heights must sum to the canvas dimensions
    xs = sorted({t[0] for t in tiles})
    ys = sorted({t[1] for t in tiles})
    assert len(xs) == 2 and len(ys) == 2
    assert abs(xs[1] + tiles[-1][2] * 200 - mosaic["width"]) < 1.0
    assert abs(ys[1] + tiles[-1][3] * 100 - mosaic["height"]) < 1.0


def test_degenerate_boxes_are_dropped(tmp_path):
    # a 3px box shrunk into a small tile goes under the 2px floor and is cut
    root = _coco_dir(
        tmp_path / "train",
        sizes=[(400, 400)] * 2,
        boxes_per_image=[[(0, 0, 3, 3)], [(10, 10, 200, 200)]],
    )
    synthesize_mosaics(root, count=4, seed=1)
    data = _load(root)
    for a in data["annotations"]:
        assert a["bbox"][2] >= 2.0 or a["image_id"] <= 2  # sources untouched
        if a["image_id"] > 2:
            assert a["bbox"][2] >= 2.0 and a["bbox"][3] >= 2.0


def test_deterministic_under_seed(tmp_path):
    kwargs = dict(sizes=[(64, 48)] * 3, boxes_per_image=[[(4, 4, 16, 12)]] * 3)
    a = _coco_dir(tmp_path / "a", **kwargs)
    b = _coco_dir(tmp_path / "b", **kwargs)
    synthesize_mosaics(a, count=2, seed=42)
    synthesize_mosaics(b, count=2, seed=42)
    assert _load(a) == _load(b)


def test_second_call_does_not_overwrite_existing_mosaics(tmp_path):
    root = _coco_dir(
        tmp_path / "train", sizes=[(64, 48)] * 2, boxes_per_image=[[], []]
    )
    synthesize_mosaics(root, count=2, seed=1)
    synthesize_mosaics(root, count=1, seed=2)
    data = _load(root)
    names = [i["file_name"] for i in data["images"] if "mosaic" in i["file_name"]]
    assert sorted(names) == ["mosaic_0000.jpg", "mosaic_0001.jpg", "mosaic_0002.jpg"]


def test_missing_annotations_file_is_a_noop(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    assert synthesize_mosaics(empty, count=3) == 0
