"""Shared prompt-decoding flow for the transformers-hosted SAM family.

SAM (v1) and SAM 2.1 expose the same shape through transformers: a processor
that resizes the image and scales prompts, `get_image_embeddings` for the
encoder, and a forward pass that accepts `image_embeddings` plus points /
labels / boxes. The two differ only in what `post_process_masks` needs and in
the embedding's type (a tensor vs a list of feature maps), so one mixin
serves both backends. Imports of torch/transformers stay inside methods (R1b).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from horos.backends.base import ImageEmbedding, SegmentPrompt, SegmentResult
from horos.backends.sam.polygonize import mask_to_polygon


class TransformersPromptableMixin:
    """Requires: self._ensure_model() setting self._model / self._processor /
    self.device, self.info.key, self.family, and `_needs_reshaped_sizes`."""

    _needs_reshaped_sizes: bool = False  # SAM v1's post_process_masks wants them

    def embed(self, image: Path) -> ImageEmbedding:
        self._ensure_model()
        import torch
        from PIL import Image

        with Image.open(image) as im:
            rgb = im.convert("RGB")
            width, height = rgb.size
            inputs = self._processor(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            features = self._model.get_image_embeddings(pixel_values)
        data: dict[str, Any] = {
            "features": features,
            "original_sizes": inputs["original_sizes"],
            "reshaped_input_sizes": inputs.get("reshaped_input_sizes"),
        }
        return ImageEmbedding(width, height, data, self.info.key)

    def segment(self, embedding: ImageEmbedding, prompt: SegmentPrompt) -> SegmentResult:
        prompt = prompt.validated()
        self._ensure_model()
        import numpy as np
        import torch
        from PIL import Image

        # the processor scales prompts from original-image pixels into model
        # space; it needs an image of the right size to know the scale — a
        # blank stand-in of the original dimensions is enough (no encoder run)
        stand_in = Image.new("RGB", (embedding.width, embedding.height))
        kwargs: dict[str, Any] = {}
        if prompt.points:
            kwargs["input_points"] = [[[list(p) for p in prompt.points]]]
            kwargs["input_labels"] = [[list(prompt.labels)]]
        if prompt.box is not None:
            x, y, w, h = prompt.box
            kwargs["input_boxes"] = [[[x, y, x + w, y + h]]]
        encoded = self._processor(images=stand_in, return_tensors="pt", **kwargs)
        model_kwargs: dict[str, Any] = {"image_embeddings": embedding.data["features"]}
        for key in ("input_points", "input_labels", "input_boxes"):
            if key in encoded:
                tensor = encoded[key]
                if tensor.dtype == torch.float64:  # MPS has no float64
                    tensor = tensor.float()
                model_kwargs[key] = tensor.to(self.device)
        with torch.no_grad():
            outputs = self._model(**model_kwargs, multimask_output=False)
        post_args = [outputs.pred_masks.cpu(), embedding.data["original_sizes"]]
        if self._needs_reshaped_sizes:
            post_args.append(embedding.data["reshaped_input_sizes"])
        masks = self._processor.post_process_masks(*post_args)[0]
        mask = masks.reshape(-1, masks.shape[-2], masks.shape[-1])[0].numpy().astype(bool)
        score = float(outputs.iou_scores.flatten()[0])
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return SegmentResult(polygon=None, bbox=None, score=score, area=0)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        return SegmentResult(
            polygon=mask_to_polygon(mask),
            bbox=(float(x0), float(y0), float(x1 - x0), float(y1 - y0)),
            score=score,
            area=int(mask.sum()),
        )
