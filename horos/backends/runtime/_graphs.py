"""Graph runners: one per artifact format, all with the same tiny surface.

A runner owns the loaded graph and knows how to feed it one NCHW float32
tensor and hand back the raw outputs by name. Everything above it (image
preprocessing, decoding the model card's output contract) is shared in
`ArtifactModel`, so a bundle serves the same way whatever it was exported to.

R1: onnxruntime / tensorrt / LiteRT are imported lazily, in here only.
R7: the device is chosen explicitly and recorded; a request the runtime
cannot honour is an error, never a silent fallback.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

from horos.backends.base import translate_backend_errors
from horos.errors import BackendError

logger = logging.getLogger(__name__)

FAMILY = "runtime"

#: artifact suffix → format kind
SUFFIX_KINDS: dict[str, str] = {
    ".onnx": "onnx",
    ".trt": "tensorrt",
    ".engine": "tensorrt",
    ".plan": "tensorrt",
    ".tflite": "tflite",
}

INSTALL_HINTS: dict[str, str] = {
    "onnx": "Serving an ONNX bundle needs the 'onnxruntime' package "
            "(pip install onnxruntime, or onnxruntime-gpu for CUDA).",
    "tensorrt": "Serving a TensorRT engine needs NVIDIA's 'tensorrt' Python package on this "
                "machine — 'horos install --tensorrt' adds the wheels matching this GPU "
                "(NVIDIA TensorRT license); on Jetson use JetPack's tensorrt through a "
                "--system-site-packages venv.",
    "tflite": "Serving a TFLite model needs the 'ai-edge-litert' interpreter (or tensorflow) "
              "— 'horos install --tflite' adds it.",
}


def _find_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def runtime_available(kind: str) -> bool:
    """Can this interpreter execute `kind` artifacts? A cheap import-free
    probe so a service refuses at start-up, not after it was spawned."""
    if kind == "onnx":
        return _find_spec("onnxruntime")
    if kind == "tensorrt":
        return _find_spec("tensorrt")
    if kind == "tflite":
        return _find_spec("ai_edge_litert") or _find_spec("tensorflow")
    return False


def kind_for(artifact: Path, declared: str | None = None) -> str:
    kind = (declared or "").lower() or SUFFIX_KINDS.get(artifact.suffix.lower(), "")
    if kind not in INSTALL_HINTS:
        raise BackendError(
            f"Cannot execute {artifact.name}: expected an .onnx graph, a TensorRT engine "
            f"(.trt/.engine/.plan) or a .tflite model",
            backend=FAMILY,
        )
    return kind


def _want(requested_device: str | None) -> str:
    return (requested_device or "auto").split(":")[0].lower()


class Graph:
    """Loaded graph + the metadata `ArtifactModel` needs."""

    kind: str = ""
    device: str = ""        # resolved: "cuda" | "cpu"
    runtime: str = ""       # human-readable: which library/provider actually runs it
    input_name: str = ""
    input_shape: list[Any] | None = None   # as declared by the graph (may hold symbols)
    output_names: list[str] = []

    def __init__(self, artifact: Path, requested_device: str | None):
        self.artifact = Path(artifact)
        self.requested = requested_device

    def load(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def run(self, tensor) -> dict[str, Any]:  # pragma: no cover - abstract
        """{output name: numpy array} for one NCHW float32 batch."""
        raise NotImplementedError


# ------------------------------------------------------------------- ONNX


class OnnxGraph(Graph):
    kind = "onnx"

    def _providers(self, ort) -> list[str]:
        available = list(ort.get_available_providers())
        want = _want(self.requested)
        if want == "cpu":
            return ["CPUExecutionProvider"]
        if want == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise BackendError(
                    "device 'cuda' requested but this onnxruntime build has no "
                    "CUDAExecutionProvider (install onnxruntime-gpu, or pass device='cpu')",
                    backend=FAMILY,
                )
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if want != "auto":
            raise BackendError(
                f"device '{want}' is not supported by the ONNX runtime executor "
                f"(use 'cuda', 'cpu' or leave it unset)",
                backend=FAMILY,
            )
        return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise BackendError(INSTALL_HINTS["onnx"], backend=FAMILY) from exc
        with translate_backend_errors(FAMILY):
            options = ort.SessionOptions()
            options.log_severity_level = 3
            self._session = ort.InferenceSession(
                str(self.artifact), sess_options=options, providers=self._providers(ort)
            )
            used = self._session.get_providers()[0]
            self.device = "cuda" if used.startswith("CUDA") else "cpu"
            self.runtime = f"onnxruntime {ort.__version__} ({used})"
            inp = self._session.get_inputs()[0]
            self.input_name = inp.name
            self.input_shape = list(inp.shape)
            self.output_names = [o.name for o in self._session.get_outputs()]

    def run(self, tensor) -> dict[str, Any]:
        raw = self._session.run(None, {self.input_name: tensor})
        return dict(zip(self.output_names, raw, strict=False))


# --------------------------------------------------------------- TensorRT


class _CudaMemory:
    """Device buffers for the engine's I/O tensors. cuda-python's runtime
    bindings (Apache-2.0, ships with torch) when present, else torch's CUDA
    allocator — an engine needs one of them, and says so."""

    def __init__(self):
        self.backend = ""
        self._cudart = None
        self._torch = None
        self._buffers: dict[str, Any] = {}
        self.stream = 0
        try:
            from cuda.bindings import runtime as cudart  # cuda-python >= 12.6
        except ImportError:
            try:
                from cuda import cudart  # cuda-python < 12.6
            except ImportError:
                cudart = None
        if cudart is not None:
            self._cudart = cudart
            self.backend = "cuda-python"
            _, self.stream = self._check(cudart.cudaStreamCreate())
            return
        try:
            import torch
        except ImportError as exc:
            raise BackendError(
                "Executing a TensorRT engine needs CUDA device memory through 'cuda-python' "
                "(pip install cuda-python) or torch; neither is installed.",
                backend=FAMILY,
            ) from exc
        if not torch.cuda.is_available():
            raise BackendError(
                "A TensorRT engine needs a CUDA device, but torch reports none available.",
                backend=FAMILY,
            )
        self._torch = torch
        self.backend = "torch"
        self.stream = torch.cuda.current_stream().cuda_stream

    def _check(self, result):
        err, *rest = result if isinstance(result, tuple) else (result,)
        if int(err) != 0:
            raise BackendError(f"CUDA runtime error {err}", backend=FAMILY)
        return (err, *rest) if rest else (err, None)

    def allocate(self, name: str, nbytes: int) -> int:
        nbytes = max(int(nbytes), 1)
        if self._cudart is not None:
            _, ptr = self._check(self._cudart.cudaMalloc(nbytes))
            self._buffers[name] = (ptr, nbytes)
            return int(ptr)
        buf = self._torch.empty(nbytes, dtype=self._torch.uint8, device="cuda")
        self._buffers[name] = (buf, nbytes)
        return int(buf.data_ptr())

    def upload(self, name: str, array) -> None:
        import numpy as np

        array = np.ascontiguousarray(array)
        target, nbytes = self._buffers[name]
        if array.nbytes > nbytes:
            raise BackendError(
                f"input tensor {name} ({array.nbytes} bytes) exceeds the engine's "
                f"buffer ({nbytes} bytes)", backend=FAMILY,
            )
        if self._cudart is not None:
            kind = self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
            self._check(self._cudart.cudaMemcpyAsync(
                target, array.ctypes.data, array.nbytes, kind, self.stream))
            return
        flat = self._torch.from_numpy(array.reshape(-1).view(np.uint8))
        target[: array.nbytes].copy_(flat, non_blocking=False)

    def download(self, name: str, shape, dtype):
        import numpy as np

        target, _ = self._buffers[name]
        out = np.empty(shape, dtype=dtype)
        if self._cudart is not None:
            kind = self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
            self._check(self._cudart.cudaMemcpyAsync(
                out.ctypes.data, target, out.nbytes, kind, self.stream))
            self._check(self._cudart.cudaStreamSynchronize(self.stream))
            return out
        host = target[: out.nbytes].cpu().numpy()
        return host.view(dtype).reshape(shape).copy()

    def synchronize(self) -> None:
        if self._cudart is not None:
            self._check(self._cudart.cudaStreamSynchronize(self.stream))
        else:
            self._torch.cuda.synchronize()

    def release(self) -> None:
        if self._cudart is not None:
            for ptr, _ in self._buffers.values():
                self._cudart.cudaFree(ptr)
            if self.stream:
                self._cudart.cudaStreamDestroy(self.stream)
        self._buffers.clear()


class TensorRTGraph(Graph):
    """A serialized TensorRT engine (.trt/.engine/.plan) — CUDA only, and
    only on the GPU architecture + TensorRT version it was built for."""

    kind = "tensorrt"

    def __init__(self, artifact: Path, requested_device: str | None, *,
                 input_hw: tuple[int, int] | None = None):
        super().__init__(artifact, requested_device)
        self._hint_hw = input_hw
        self._memory: _CudaMemory | None = None
        self._outputs: dict[str, tuple[tuple[int, ...], Any]] = {}
        self._input_dtype = None
        self._input_static: tuple[int, ...] | None = None

    def load(self) -> None:
        want = _want(self.requested)
        if want not in ("auto", "cuda"):
            raise BackendError(
                f"a TensorRT engine executes on CUDA only; device '{want}' cannot run it "
                f"(serve the run's ONNX bundle for CPU)",
                backend=FAMILY,
            )
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise BackendError(INSTALL_HINTS["tensorrt"], backend=FAMILY) from exc
        import numpy as np

        with translate_backend_errors(FAMILY):
            trt_logger = trt.Logger(trt.Logger.ERROR)
            trt.init_libnvinfer_plugins(trt_logger, "")
            runtime = trt.Runtime(trt_logger)
            engine = runtime.deserialize_cuda_engine(self.artifact.read_bytes())
            if engine is None:
                raise BackendError(
                    f"TensorRT {trt.__version__} could not deserialize {self.artifact.name}: "
                    f"engines are bound to the GPU architecture and TensorRT version they "
                    f"were built with — rebuild it on this machine",
                    backend=FAMILY,
                )
            context = engine.create_execution_context()
            names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
            inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            outputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
            if len(inputs) != 1:
                raise BackendError(
                    f"expected one input tensor, the engine has {inputs}", backend=FAMILY
                )
            self.input_name = inputs[0]
            self.output_names = outputs
            declared = list(engine.get_tensor_shape(self.input_name))
            self.input_shape = declared
            static = self._static_input_shape(declared)
            if -1 in declared:
                context.set_input_shape(self.input_name, static)
            self._input_static = static
            self._input_dtype = trt.nptype(engine.get_tensor_dtype(self.input_name))
            memory = _CudaMemory()
            itemsize = np.dtype(self._input_dtype).itemsize
            context.set_tensor_address(
                self.input_name, memory.allocate(self.input_name, int(np.prod(static)) * itemsize)
            )
            for name in outputs:
                shape = tuple(int(v) for v in context.get_tensor_shape(name))
                dtype = trt.nptype(engine.get_tensor_dtype(name))
                nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
                context.set_tensor_address(name, memory.allocate(name, nbytes))
                self._outputs[name] = (shape, dtype)
            self._engine, self._context, self._memory = engine, context, memory
            self.device = "cuda"
            self.runtime = f"TensorRT {trt.__version__} ({memory.backend} memory)"

    def _static_input_shape(self, declared: list[int]) -> tuple[int, ...]:
        shape = list(declared)
        if len(shape) == 4:
            if shape[0] == -1:
                shape[0] = 1
            if self._hint_hw:
                for axis, value in ((2, self._hint_hw[0]), (3, self._hint_hw[1])):
                    if shape[axis] == -1:
                        shape[axis] = int(value)
        if -1 in shape:
            raise BackendError(
                f"the engine's input {declared} has dynamic axes the model card does not "
                f"pin; export with a static shape", backend=FAMILY,
            )
        return tuple(int(v) for v in shape)

    def run(self, tensor) -> dict[str, Any]:
        import numpy as np

        array = np.asarray(tensor).astype(self._input_dtype, copy=False)
        if tuple(array.shape) != self._input_static:
            raise BackendError(
                f"engine input is {self._input_static}, got {tuple(array.shape)}",
                backend=FAMILY,
            )
        memory = self._memory
        memory.upload(self.input_name, array)
        if not self._context.execute_async_v3(memory.stream):
            raise BackendError("TensorRT execution failed", backend=FAMILY)
        memory.synchronize()
        return {
            name: memory.download(name, shape, dtype)
            for name, (shape, dtype) in self._outputs.items()
        }


# ----------------------------------------------------------------- TFLite


class TFLiteGraph(Graph):
    """A .tflite model through the LiteRT interpreter (tensorflow's as a
    fallback). CPU only: that is what LiteRT executes here."""

    kind = "tflite"

    def load(self) -> None:
        want = _want(self.requested)
        if want not in ("auto", "cpu"):
            raise BackendError(
                f"the TFLite executor runs on CPU (LiteRT); device '{want}' cannot run it — "
                f"serve the run's ONNX bundle or TensorRT engine for GPU inference",
                backend=FAMILY,
            )
        interpreter_cls, label = None, ""
        try:
            import ai_edge_litert
            from ai_edge_litert.interpreter import Interpreter as interpreter_cls

            label = f"LiteRT {getattr(ai_edge_litert, '__version__', '')}".strip()
        except ImportError:
            try:
                import tensorflow as tf

                interpreter_cls = tf.lite.Interpreter
                label = f"tensorflow.lite {tf.__version__}"
            except ImportError as exc:
                raise BackendError(INSTALL_HINTS["tflite"], backend=FAMILY) from exc
        with translate_backend_errors(FAMILY):
            interp = interpreter_cls(model_path=str(self.artifact))
            interp.allocate_tensors()
            self._interp = interp
            self._runner = None
            signatures = {}
            try:
                signatures = dict(interp.get_signature_list() or {})
            except (AttributeError, ValueError):
                signatures = {}
            if len(signatures) == 1:
                # onnx2tf (and tensorflow's converter) write a SignatureDef: the
                # ONNX input/output names survive there even when the tensor
                # names became "PartitionedCall:N"
                key, spec = next(iter(signatures.items()))
                self._runner = interp.get_signature_runner(key)
                self.input_name = str(spec["inputs"][0])
                self.input_shape = [int(v) for v in
                                    self._runner.get_input_details()[self.input_name]["shape"]]
                self.output_names = [str(n) for n in spec["outputs"]]
            else:
                inp = interp.get_input_details()[0]
                self._input = inp
                self.input_name = _clean(inp["name"])
                self.input_shape = [int(v) for v in inp["shape"]]
                self._output_details = sorted(
                    interp.get_output_details(), key=lambda d: d["index"]
                )
                self.output_names = [_clean(d["name"]) for d in self._output_details]
            self.device = "cpu"
            self.runtime = label

    def run(self, tensor) -> dict[str, Any]:
        import numpy as np

        array = np.asarray(tensor, dtype=np.float32)
        expected = list(self.input_shape)
        if list(array.shape) != expected and len(expected) == 4 and expected[-1] == array.shape[1]:
            array = array.transpose(0, 2, 3, 1)  # the graph kept NHWC after all
        array = np.ascontiguousarray(array)
        if self._runner is not None:
            result = self._runner(**{self.input_name: array})
            return {str(name): np.asarray(value) for name, value in result.items()}
        interp = self._interp
        interp.set_tensor(self._input["index"], array)
        interp.invoke()
        return {
            _clean(d["name"]): interp.get_tensor(d["index"]) for d in self._output_details
        }


def _clean(name: str) -> str:
    """TFLite names survive conversion as 'dets', 'dets:0' or 'model/dets' —
    reduce them to the last path segment without the tensor index."""
    return str(name).split(":")[0].rsplit("/", 1)[-1]


def make_graph(artifact: Path, kind: str, requested_device: str | None, *,
               input_hw: tuple[int, int] | None = None) -> Graph:
    if kind == "onnx":
        return OnnxGraph(artifact, requested_device)
    if kind == "tensorrt":
        return TensorRTGraph(artifact, requested_device, input_hw=input_hw)
    if kind == "tflite":
        return TFLiteGraph(artifact, requested_device)
    raise BackendError(f"no executor for '{kind}' artifacts", backend=FAMILY)
