"""Offline mosaic synthesis for training snapshots (E5).

Mosaic augmentation (four images composed into one canvas, as popularized by
the YOLO family) is not available as a knob in the current training backend,
so horos composes mosaics OFFLINE into the run's dataset snapshot instead of
inside the training dataloader.

Design constraints this deliberately satisfies:

- **No annotation clipping.** Each quadrant holds one WHOLE source image
  resized to fit — nothing is cropped, so every annotation (bbox and polygon
  alike) transforms by pure scale+offset and none needs clipping. The
  backend's own augmentation history of silent annotation corruption (R5) is
  exactly the class of bug this sidesteps.
- **Backend neutral (R1).** The composites are ordinary COCO images written
  into the snapshot; any backend trains on them without knowing they exist,
  and the user can open the snapshot and see them.
- **Deterministic.** Composition is driven by a seeded RNG; the same seed
  reproduces the same snapshot.

The trade-off versus dataloader-time mosaic: composites are fixed at enqueue
time rather than re-rolled per epoch, and there is no object-cutoff effect.
"""

from __future__ import annotations

import json
import logging
import random
import statistics
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

#: the mosaic crosshair is jittered within this central band of the canvas,
#: so tile (and object) scales vary between composites
_CENTER_JITTER = (0.35, 0.65)
#: composed boxes narrower/shorter than this are dropped (degenerate slivers)
_MIN_BOX_PX = 2.0
_ANNOTATIONS_NAME = "_annotations.coco.json"


def synthesize_mosaics(
    split_dir: Path | str, *, count: int, seed: int = 42, jpeg_quality: int = 90
) -> int:
    """Append `count` mosaic composites to the COCO split directory in place.

    Reads `<split_dir>/_annotations.coco.json`, composes 2×2 mosaics from the
    split's own images, writes them as `mosaic_<n>.jpg` next to the sources
    and appends matching image/annotation entries. Returns how many mosaics
    were actually written (0 when the split has no usable images).
    """
    split_dir = Path(split_dir)
    ann_path = split_dir / _ANNOTATIONS_NAME
    if count <= 0 or not ann_path.is_file():
        return 0
    data = json.loads(ann_path.read_text(encoding="utf-8"))

    sources = [
        img for img in data["images"] if (split_dir / img["file_name"]).is_file()
    ]
    if not sources:
        return 0
    anns_by_image: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # canvas = the split's median image size, so composites look like ordinary
    # images (each tile ends up around half resolution — the scale-variation
    # effect mosaic is used for)
    canvas_w = int(statistics.median(img["width"] for img in sources))
    canvas_h = int(statistics.median(img["height"] for img in sources))

    rng = random.Random(seed)
    next_image_id = max((img["id"] for img in data["images"]), default=0) + 1
    next_ann_id = max((a["id"] for a in data["annotations"]), default=0) + 1
    # a second call on the same snapshot must not overwrite earlier composites
    name_offset = sum(
        1 for img in data["images"] if str(img["file_name"]).startswith("mosaic_")
    )
    written = 0

    for index in range(count):
        picks = rng.choices(sources, k=4)
        cx = int(canvas_w * rng.uniform(*_CENTER_JITTER))
        cy = int(canvas_h * rng.uniform(*_CENTER_JITTER))
        # (x, y, width, height) of the four tiles around the jittered center
        tiles = (
            (0, 0, cx, cy),
            (cx, 0, canvas_w - cx, cy),
            (0, cy, cx, canvas_h - cy),
            (cx, cy, canvas_w - cx, canvas_h - cy),
        )

        canvas = Image.new("RGB", (canvas_w, canvas_h))
        annotations: list[dict] = []
        for src, (tx, ty, tw, th) in zip(picks, tiles, strict=True):
            with Image.open(split_dir / src["file_name"]) as im:
                canvas.paste(im.convert("RGB").resize((tw, th)), (tx, ty))
            sx, sy = tw / src["width"], th / src["height"]
            for ann in anns_by_image.get(src["id"], []):
                x, y, w, h = ann["bbox"]
                bw, bh = w * sx, h * sy
                if bw < _MIN_BOX_PX or bh < _MIN_BOX_PX:
                    continue
                annotations.append(
                    {
                        "id": next_ann_id,
                        "image_id": next_image_id,
                        "category_id": ann["category_id"],
                        "bbox": [x * sx + tx, y * sy + ty, bw, bh],
                        "area": bw * bh,
                        "segmentation": [
                            [
                                v * (sx if i % 2 == 0 else sy)
                                + (tx if i % 2 == 0 else ty)
                                for i, v in enumerate(poly)
                            ]
                            for poly in ann.get("segmentation") or []
                        ],
                        "iscrowd": ann.get("iscrowd", 0),
                    }
                )
                next_ann_id += 1

        file_name = f"mosaic_{name_offset + index:04d}.jpg"
        canvas.save(split_dir / file_name, quality=jpeg_quality)
        data["images"].append(
            {
                "id": next_image_id,
                "file_name": file_name,
                "width": canvas_w,
                "height": canvas_h,
            }
        )
        data["annotations"].extend(annotations)
        next_image_id += 1
        written += 1

    ann_path.write_text(json.dumps(data), encoding="utf-8")
    logger.info("synthesized %d mosaic composites into %s", written, split_dir)
    return written
