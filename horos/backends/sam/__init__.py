"""SAM backend — box-prompted segmentation for the autolabel polygon output.

transformers hosts SAM (`facebook/sam-vit-base`, Apache 2.0), so this adds no
new dependency; like every backend, the ML imports happen lazily on first use
(R1b). SAM here is a refiner, not a detector: OWLv2 finds the boxes, SAM turns
each box into a mask, `polygonize` turns the mask into an editable polygon.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from horos.backends import weights
from horos.backends.base import (
    BoxToMaskBackend,
    Event,
    ExportSpec,
    ImagePrediction,
    TrainSpec,
    translate_backend_errors,
)
from horos.backends.sam.polygonize import mask_to_polygon
from horos.errors import BackendError

if TYPE_CHECKING:
    from horos.core.registry import ModelInfo

_NOT_A_DETECTOR = (
    "SAM is a box-to-mask refiner used by autolabel's polygon output; "
    "it does not {op} on its own."
)


class SAMBackend(BoxToMaskBackend):
    family = "sam"

    def __init__(
        self,
        info: ModelInfo,
        *,
        device: str | None = None,
        checkpoint: Path | None = None,
    ):
        super().__init__(info, device=device, checkpoint=checkpoint)
        self._model = None
        self._processor = None

    def _ensure_model(self):
        if self._model is not None:
            return
        with translate_backend_errors(self.family):
            import torch  # noqa: F401 — resolved lazily on first use (R1b)
            from transformers import SamModel, SamProcessor

            from horos.backends.device import select_device

            self.device = select_device(self.device).torch_device
            cache = str(weights.hf_cache_dir())
            self._processor = SamProcessor.from_pretrained(
                self.info.hf_id, cache_dir=cache
            )
            self._model = SamModel.from_pretrained(
                self.info.hf_id, cache_dir=cache
            ).to(self.device)
            self._model.eval()

    def polygons_for_boxes(
        self, image: Path, boxes: list[tuple[float, float, float, float]]
    ) -> list[list[float] | None]:
        if not boxes:
            return []
        self._ensure_model()
        with translate_backend_errors(self.family):
            import torch
            from PIL import Image

            with Image.open(image) as im:
                rgb = im.convert("RGB")
                # COCO xywh -> SAM xyxy prompts
                xyxy = [[x, y, x + w, y + h] for x, y, w, h in boxes]
                inputs = self._processor(rgb, input_boxes=[xyxy], return_tensors="pt")
                # the processor emits float64 box prompts; MPS has no float64
                inputs["input_boxes"] = inputs["input_boxes"].float()
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    outputs = self._model(**inputs, multimask_output=False)
                masks = self._processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu(),
                )[0]  # (num_boxes, 1, H, W) bool
            polygons: list[list[float] | None] = []
            for i in range(masks.shape[0]):
                mask = masks[i, 0].numpy()
                polygons.append(mask_to_polygon(mask))
            return polygons

    # -- not a detector -------------------------------------------------------
    def train(self, spec: TrainSpec) -> Iterator[Event]:
        raise BackendError(_NOT_A_DETECTOR.format(op="train"), backend=self.family)

    def infer_one(self, image: Path, *, threshold: float = 0.5) -> ImagePrediction:
        raise BackendError(_NOT_A_DETECTOR.format(op="detect"), backend=self.family)

    def infer_batch(
        self, images: Iterable[Path], *, threshold: float = 0.5
    ) -> Iterator[Event]:
        raise BackendError(_NOT_A_DETECTOR.format(op="detect"), backend=self.family)

    def export(self, checkpoint: Path, spec: ExportSpec) -> Iterator[Event]:
        raise BackendError(_NOT_A_DETECTOR.format(op="export"), backend=self.family)
