"""Deployment-artifact execution (E8-T7, E8-S6; engines since Serve-T1).

Runs an exported detection graph from what ships next to it — the model
card's input/output specification and class list — without the training
framework:

- ONNX through onnxruntime (CUDA or CPU provider, chosen explicitly)
- TensorRT engines through NVIDIA's tensorrt runtime (CUDA only, and only on
  the GPU architecture + TensorRT version that built them)
- TFLite through the LiteRT interpreter (CPU)

The decoding follows the card's documented output contract (boxes as
normalised cx, cy, w, h; class logits to sigmoid), which is the same for
every format because the TFLite and TensorRT graphs are built from the ONNX
export. One bundle description therefore serves all three, and a bundle
exported on one machine serves on another with nothing but `pip install
horos` plus that format's runtime (see `INSTALL_HINTS`).

R1: the runtimes are imported lazily, in `_graphs.py` only. R7: the device is
chosen explicitly and recorded; asking for what a runtime cannot do (CUDA on
a CPU-only onnxruntime, CPU for an engine, CUDA for TFLite) is an error,
never a silent fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from horos.backends.base import ImagePrediction, PredictedInstance, translate_backend_errors
from horos.backends.runtime._graphs import (
    FAMILY,
    INSTALL_HINTS,
    SUFFIX_KINDS,
    Graph,
    kind_for,
    make_graph,
    runtime_available,
)
from horos.errors import BackendError

__all__ = [
    "ArtifactModel",
    "INSTALL_HINTS",
    "SUFFIX_KINDS",
    "kind_for",
    "runtime_available",
]

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_DEFAULT_RESOLUTION = 640


class ArtifactModel:
    """An exported detector (ONNX / TensorRT / TFLite), callable like a
    backend's `infer_one`."""

    family = FAMILY

    def __init__(
        self,
        artifact: Path | str,
        *,
        card: dict[str, Any] | None = None,
        classes: list[str] | None = None,
        device: str | None = None,
        kind: str | None = None,
    ):
        self.artifact = Path(artifact)
        self.card = dict(card or {})
        self.classes = list(classes if classes is not None else self.card.get("classes", []))
        self.requested_device = device
        self.kind = kind_for(self.artifact, kind or self.card.get("format"))
        self.device: str | None = None   # resolved on first load
        self.runtime: str | None = None  # which library/provider actually runs it
        self._graph: Graph | None = None
        self._input_hw: tuple[int, int] | None = None

    # ------------------------------------------------------------ loading
    def load(self) -> Graph:
        if self._graph is not None:
            return self._graph
        if not self.artifact.is_file():
            raise BackendError(
                f"{self.kind} artifact not found: {self.artifact}", backend=self.family
            )
        hint_hw = self._resolve_input_hw(None)
        graph = make_graph(self.artifact, self.kind, self.requested_device, input_hw=hint_hw)
        graph.load()
        self._graph = graph
        self.device = graph.device
        self.runtime = graph.runtime
        self._input_hw = self._resolve_input_hw(graph.input_shape)
        logger.info(
            "loaded %s via %s, input %s", self.artifact.name, graph.runtime, self._input_hw
        )
        return graph

    def _resolve_input_hw(self, graph_shape) -> tuple[int, int]:
        """(height, width): the card's static shape, else the graph's, else the
        exported resolution hyperparameter, else 640."""
        shape = (self.card.get("input") or {}).get("shape")
        for candidate in (shape, graph_shape):
            if candidate and len(candidate) == 4:
                h, w = candidate[2], candidate[3]
                if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                    return h, w
        res = (self.card.get("hyperparameters") or {}).get("resolution")
        res = int(res) if isinstance(res, int | float) and res > 0 else _DEFAULT_RESOLUTION
        return res, res

    # ------------------------------------------------------------ inference
    def _preprocess(self, image: Path):
        import numpy as np
        from PIL import Image

        h, w = self._input_hw
        spec = self.card.get("input") or {}
        mean = np.asarray(spec.get("mean") or IMAGENET_MEAN, dtype=np.float32)
        std = np.asarray(spec.get("std") or IMAGENET_STD, dtype=np.float32)
        with Image.open(image) as im:
            width, height = im.size
            arr = np.asarray(im.convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.float32)
        arr = ((arr / 255.0 - mean) / std).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(arr), width, height

    def _outputs(self, graph: Graph, raw: dict[str, Any]) -> tuple[Any, Any]:
        """(boxes, logits) — by the card's / graph's output names, else by order."""
        boxes = raw.get("dets", raw.get("boxes"))
        logits = raw.get("labels", raw.get("logits", raw.get("scores")))
        if boxes is None or logits is None:
            values = [raw[name] for name in graph.output_names if name in raw]
            if len(values) < 2:
                raise BackendError(
                    f"expected two outputs (boxes, class logits), got {graph.output_names}",
                    backend=self.family,
                )
            # names did not survive conversion: boxes are the [.., 4] tensor when
            # that is unambiguous, otherwise trust the graph's output order
            four = [v for v in values[:2] if getattr(v, "shape", ())[-1:] == (4,)]
            if len(four) == 1:
                boxes = four[0]
                logits = values[1] if boxes is values[0] else values[0]
            else:
                boxes, logits = values[0], values[1]
        return boxes, logits

    def infer_one(self, image: Path | str, *, threshold: float = 0.5) -> ImagePrediction:
        import numpy as np

        graph = self.load()
        image = Path(image)
        with translate_backend_errors(self.family):
            tensor, width, height = self._preprocess(image)
            boxes, logits = self._outputs(graph, graph.run(tensor))
            boxes = np.asarray(boxes, dtype=np.float32)[0]
            logits = np.asarray(logits, dtype=np.float32)[0]
            scores = 1.0 / (1.0 + np.exp(-logits))
            classes = scores.argmax(axis=-1)
            best = scores.max(axis=-1)
            keep = np.nonzero(best >= threshold)[0]
            instances = []
            for i in keep[np.argsort(-best[keep])]:
                cx, cy, bw, bh = (float(v) for v in boxes[i][:4])
                x = max(0.0, (cx - bw / 2) * width)
                y = max(0.0, (cy - bh / 2) * height)
                w = min(float(width) - x, bw * width)
                h = min(float(height) - y, bh * height)
                cid = int(classes[i])
                instances.append(
                    PredictedInstance(
                        bbox=(x, y, max(0.0, w), max(0.0, h)),
                        score=float(best[i]),
                        category_id=cid,
                        category_name=self.classes[cid] if 0 <= cid < len(self.classes) else None,
                    )
                )
        return ImagePrediction(image=str(image), width=width, height=height, instances=instances)
