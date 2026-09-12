"""Deployment-artifact execution (E8-T7, E8-S6).

Runs an exported detection graph from what ships next to it — the model
card's input/output specification and class list — without the training
framework. Today that is ONNX through onnxruntime; the decoding follows the
card's documented output contract (boxes as normalised cx, cy, w, h; class
logits to sigmoid), so a bundle exported on one machine serves on another
with nothing but `pip install horos onnxruntime`.

R1: onnxruntime is imported lazily, here only. R7: the execution provider is
chosen explicitly and recorded; asking for CUDA on a build without it is an
error, never a silent CPU fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from horos.backends.base import ImagePrediction, PredictedInstance, translate_backend_errors
from horos.errors import BackendError

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_DEFAULT_RESOLUTION = 640


class ArtifactModel:
    """An exported ONNX detector, callable like a backend's `infer_one`."""

    family = "runtime"

    def __init__(
        self,
        artifact: Path | str,
        *,
        card: dict[str, Any] | None = None,
        classes: list[str] | None = None,
        device: str | None = None,
    ):
        self.artifact = Path(artifact)
        self.card = dict(card or {})
        self.classes = list(classes if classes is not None else self.card.get("classes", []))
        self.requested_device = device
        self.device: str | None = None  # resolved provider, set on first load
        self._session = None
        self._input_name: str | None = None
        self._input_hw: tuple[int, int] | None = None

    # ------------------------------------------------------------ loading
    def _providers(self, ort) -> list[str]:
        available = list(ort.get_available_providers())
        want = (self.requested_device or "auto").split(":")[0]
        if want == "cpu":
            return ["CPUExecutionProvider"]
        if want == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise BackendError(
                    "device 'cuda' requested but this onnxruntime build has no "
                    "CUDAExecutionProvider (install onnxruntime-gpu, or pass device='cpu')",
                    backend=self.family,
                )
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if want not in ("auto",):
            raise BackendError(
                f"device '{want}' is not supported by the ONNX runtime executor "
                f"(use 'cuda', 'cpu' or leave it unset)",
                backend=self.family,
            )
        return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]

    def load(self):
        if self._session is not None:
            return self._session
        if not self.artifact.is_file():
            raise BackendError(f"ONNX artifact not found: {self.artifact}", backend=self.family)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise BackendError(
                "Serving an ONNX bundle needs the 'onnxruntime' package "
                "(pip install onnxruntime, or onnxruntime-gpu for CUDA)",
                backend=self.family,
            ) from exc
        with translate_backend_errors(self.family):
            providers = self._providers(ort)
            options = ort.SessionOptions()
            options.log_severity_level = 3
            self._session = ort.InferenceSession(
                str(self.artifact), sess_options=options, providers=providers
            )
            used = self._session.get_providers()[0]
            self.device = "cuda" if used.startswith("CUDA") else "cpu"
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            self._input_hw = self._resolve_input_hw(inp.shape)
            logger.info(
                "loaded %s via %s, input %s", self.artifact.name, used, self._input_hw
            )
        return self._session

    def _resolve_input_hw(self, session_shape) -> tuple[int, int]:
        """(height, width): the card's static shape, else the graph's, else the
        exported resolution hyperparameter, else 640."""
        shape = (self.card.get("input") or {}).get("shape")
        for candidate in (shape, session_shape):
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

    def _outputs(self, session, raw: list) -> tuple[Any, Any]:
        """(boxes, logits) — by the card's / graph's output names, else by order."""
        names = [o.name for o in session.get_outputs()]
        by_name = dict(zip(names, raw, strict=False))
        boxes = by_name.get("dets", by_name.get("boxes"))
        logits = by_name.get("labels", by_name.get("logits", by_name.get("scores")))
        if boxes is None or logits is None:
            if len(raw) < 2:
                raise BackendError(
                    f"expected two outputs (boxes, class logits), got {names}",
                    backend=self.family,
                )
            boxes, logits = raw[0], raw[1]
        return boxes, logits

    def infer_one(self, image: Path | str, *, threshold: float = 0.5) -> ImagePrediction:
        import numpy as np

        session = self.load()
        image = Path(image)
        with translate_backend_errors(self.family):
            tensor, width, height = self._preprocess(image)
            raw = session.run(None, {self._input_name: tensor})
            boxes, logits = self._outputs(session, raw)
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
