"""ONNX → TFLite conversion and TFLite execution (E8-T3).

Design decision (confirmed 2026-09-12): the path is the existing ONNX export
→ onnx2tf (MIT) → TFLite. onnx2tf needs tensorflow (Apache 2.0, ~600 MB), so
the toolchain is an opt-in extra: `horos install --tflite`. Parity is checked
with the ai-edge-litert interpreter (Apache 2.0, the lightweight LiteRT
runtime), falling back to tensorflow's own interpreter.

The graph input is kept NCHW (onnx2tf would transpose it to NHWC by default)
so the TFLite bundle honours the same I/O contract the model card describes
for ONNX: one bundle description serves both formats, and the runtime
executor's pre/post-processing does not fork per format.
"""

from __future__ import annotations

import importlib.util
import logging
import shutil
from pathlib import Path
from typing import Any

from horos.errors import BackendError

logger = logging.getLogger(__name__)

TOOLCHAIN_MODULES = ("onnx2tf", "tensorflow")
INSTALL_HINT = (
    "TFLite export needs the conversion toolchain (onnx2tf + tensorflow, ~600 MB, "
    "Apache 2.0 / MIT). Run 'horos install --tflite' to add it, then retry."
)


def toolchain_available() -> bool:
    try:
        return all(importlib.util.find_spec(m) is not None for m in TOOLCHAIN_MODULES)
    except (ImportError, ValueError):
        return False


def interpreter_available() -> bool:
    try:
        return any(
            importlib.util.find_spec(m) is not None for m in ("ai_edge_litert", "tensorflow")
        )
    except (ImportError, ValueError):
        return False


def convert_onnx_to_tflite(
    onnx_path: Path,
    out_dir: Path,
    *,
    input_names: list[str] | None = None,
    precisions: tuple[str, ...] = ("float32", "float16"),
    stem: str | None = None,
) -> dict[str, Path]:
    """Convert `onnx_path` and place `<stem>_float32.tflite` (and float16) in
    `out_dir`. Returns {precision: path}. NCHW inputs listed in `input_names`
    are kept as-is instead of being transposed to NHWC."""
    if not toolchain_available():
        raise BackendError(INSTALL_HINT, backend="tflite")
    import onnx2tf

    onnx_path = Path(onnx_path)
    out_dir = Path(out_dir)
    work = out_dir / "_onnx2tf"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    stem = stem or onnx_path.stem
    try:
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(work),
            keep_ncw_or_nchw_or_ncdhw_input_names=list(input_names or []) or None,
            copy_onnx_input_output_names_to_tflite=True,
            output_signaturedefs=True,
            non_verbose=True,
        )
    except Exception as exc:  # noqa: BLE001 — translated into horos's error type
        raise BackendError(
            f"onnx2tf could not convert {onnx_path.name}: {type(exc).__name__}: {exc}",
            backend="tflite",
        ) from exc
    produced = {p.name: p for p in work.glob("*.tflite")}
    results: dict[str, Path] = {}
    for precision in precisions:
        match = next((p for name, p in produced.items() if name.endswith(f"_{precision}.tflite")),
                     None)
        if match is None:
            continue
        target = out_dir / f"{stem}_{precision}.tflite"
        shutil.move(str(match), target)
        results[precision] = target
    shutil.rmtree(work, ignore_errors=True)
    if "float32" not in results:
        raise BackendError(
            f"onnx2tf produced no float32 model for {onnx_path.name} "
            f"(got {sorted(produced) or 'nothing'})",
            backend="tflite",
        )
    logger.info("converted %s → %s", onnx_path.name, ", ".join(p.name for p in results.values()))
    return results


class TFLiteRunner:
    """Run a .tflite detector on NCHW float32 tensors, returning outputs in
    graph order (dets, labels) — the parity harness's second side."""

    def __init__(self, model_path: Path | str):
        self.model_path = Path(model_path)
        self._interp = None

    def _load(self):
        if self._interp is not None:
            return self._interp
        interpreter_cls = None
        try:
            from ai_edge_litert.interpreter import Interpreter as interpreter_cls
        except ImportError:
            try:
                import tensorflow as tf

                interpreter_cls = tf.lite.Interpreter
            except ImportError as exc:
                raise BackendError(
                    "Running a TFLite model needs 'ai-edge-litert' (or tensorflow) — "
                    "'horos install --tflite' adds both.",
                    backend="tflite",
                ) from exc
        interp = interpreter_cls(model_path=str(self.model_path))
        interp.allocate_tensors()
        self._interp = interp
        return interp

    def run(self, tensor) -> list[Any]:
        import numpy as np

        interp = self._load()
        inp = interp.get_input_details()[0]
        arr = np.asarray(tensor, dtype=np.float32)
        expected = list(inp["shape"])
        if list(arr.shape) != expected and len(expected) == 4 and expected[-1] == arr.shape[1]:
            arr = arr.transpose(0, 2, 3, 1)  # the graph kept NHWC after all
        interp.set_tensor(inp["index"], arr)
        interp.invoke()
        outputs = sorted(interp.get_output_details(), key=lambda d: d["index"])
        by_name = {d["name"]: interp.get_tensor(d["index"]) for d in outputs}
        # honour the exported contract's order when the names survived conversion
        ordered = []
        for key in ("dets", "labels"):
            hit = next((v for n, v in by_name.items() if n.split(":")[0].endswith(key)), None)
            if hit is not None:
                ordered.append(hit)
        if len(ordered) == 2:
            return ordered
        return [interp.get_tensor(d["index"]) for d in outputs]
