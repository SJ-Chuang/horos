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

Quantisation (E8-T3b, 2026-09-12). Two kinds are supported by the converter:

- "int8_dynamic": dynamic-range quantisation — int8 weights, float32
  activations and I/O, no calibration. What RF-DETR ships as its "int8"
  variant (~4x smaller file). Its graph goes through onnx2tf's legacy
  TensorFlow converter backend with Erf replaced by a tanh approximation:
  TFLite has no builtin Erf, and a Flex op would need the TF runtime; the
  approximation matches the exact float32 graph to ~4e-5 on RF-DETR.
  The converter path hands the graph an NHWC input; the executor handles
  either layout and the model card records it.
- "int8": static integer quantisation (int8 weights AND activations) through
  onnx2tf's `-oiqt` path, calibrated on preprocessed input tensors (onnx2tf
  feeds `(data - mean) / std`, so horos passes mean 0 / std 1). Verified on
  small graphs; RF-DETR cannot use it today — see docs/BACKLOG.md — so the
  RF-DETR backend does not offer it.

The float32 model always stays the bundle's primary artifact; quantised
files are recorded variants with their own parity check.
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


#: how onnx2tf names each precision; "int8" is the integer-quantised graph
#: that keeps float32 input/output (the full-integer variant is not shipped)
PRECISION_SUFFIXES: dict[str, str] = {
    "float32": "_float32.tflite",
    "float16": "_float16.tflite",
    "int8": "_integer_quant.tflite",
    "int8_dynamic": "_dynamic_range_quant.tflite",
}
#: onnx2tf's two TFLite builders: the fast FlatBuffer path (default; keeps
#: NCHW, exact ops) and TensorFlow's converter (slow, NHWC, but the only one
#: whose dynamic-range quantisation covers a transformer's weights)
BACKENDS = ("flatbuffer_direct", "tf_converter")


def convert_onnx_to_tflite(
    onnx_path: Path,
    out_dir: Path,
    *,
    input_names: list[str] | None = None,
    precisions: tuple[str, ...] = ("float32", "float16"),
    stem: str | None = None,
    calibration: Path | None = None,
    backend: str = "flatbuffer_direct",
    pseudo_operators: list[str] | None = None,
) -> dict[str, Path]:
    """Convert `onnx_path` and place `<stem>_<precision>.tflite` files in
    `out_dir` for each of `precisions`. Returns {precision: path}. NCHW
    inputs listed in `input_names` are kept as-is instead of being transposed
    to NHWC (the fast backend only). "int8" needs `calibration`: a .npy of
    already preprocessed input tensors, [N, ...] in the graph's input layout.
    `pseudo_operators` are ONNX ops onnx2tf replaces by builtin
    approximations (e.g. "Erf" → tanh form) instead of Flex ops."""
    if not toolchain_available():
        raise BackendError(INSTALL_HINT, backend="tflite")
    if backend not in BACKENDS:
        raise BackendError(f"unknown TFLite backend '{backend}' — one of {BACKENDS}",
                           backend="tflite")
    unknown = [p for p in precisions if p not in PRECISION_SUFFIXES]
    if unknown:
        raise BackendError(
            f"unknown TFLite precision(s) {unknown} — one of {sorted(PRECISION_SUFFIXES)}",
            backend="tflite",
        )
    if "int8" in precisions and (calibration is None or not Path(calibration).is_file()):
        raise BackendError(
            "int8 quantisation needs calibration data (a .npy of preprocessed inputs)",
            backend="tflite",
        )
    import onnx2tf

    onnx_path = Path(onnx_path)
    out_dir = Path(out_dir)
    work = out_dir / "_onnx2tf"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    stem = stem or onnx_path.stem
    kwargs: dict[str, Any] = {"tflite_backend": backend}
    if pseudo_operators:
        kwargs["replace_to_pseudo_operators"] = list(pseudo_operators)
    if "int8_dynamic" in precisions:
        kwargs["output_dynamic_range_quantized_tflite"] = True
    if "int8" in precisions:
        names = list(input_names or ["input"])
        kwargs.update(
            output_integer_quantized_tflite=True,
            quant_type="per-channel",
            # the data is preprocessed already: identity normalisation
            custom_input_op_name_np_data_path=[[n, str(calibration), 0.0, 1.0] for n in names],
        )
    try:
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(work),
            keep_ncw_or_nchw_or_ncdhw_input_names=list(input_names or []) or None,
            copy_onnx_input_output_names_to_tflite=True,
            output_signaturedefs=True,
            non_verbose=True,
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 — translated into horos's error type
        raise BackendError(
            f"onnx2tf could not convert {onnx_path.name}: {type(exc).__name__}: {exc}",
            backend="tflite",
        ) from exc
    produced = {p.name: p for p in work.glob("*.tflite")}
    results: dict[str, Path] = {}
    for precision in precisions:
        suffix = PRECISION_SUFFIXES[precision]
        match = next(
            (p for name, p in produced.items()
             if name.endswith(suffix) and not name.endswith("_full" + suffix)),
            None,
        )
        if match is None:
            continue
        target = out_dir / f"{stem}_{precision}.tflite"
        shutil.move(str(match), target)
        results[precision] = target
    shutil.rmtree(work, ignore_errors=True)
    missing = [p for p in precisions if p not in results]
    if missing:
        raise BackendError(
            f"onnx2tf produced no {', '.join(missing)} model for {onnx_path.name} "
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
