# horos

End-to-end tooling for perception tasks: annotate → train → evaluate → deploy.

- Models are adapters behind `horos/backends/`; nothing above that layer knows which
  architecture is underneath.
- License metadata travels with every model, run, and export artifact.
- Target deployment platform is NVIDIA Jetson.

## Install

```bash
pip install horos
```

### Jetson (read this — it matters)

On Jetson, torch **must** come from NVIDIA's JetPack-matched wheel. The PyPI torch has
no CUDA support on Jetson, and a plain `pip install horos` may silently replace your
CUDA-enabled torch with a CPU-only build — everything still runs, just an order of
magnitude slower.

Install without dependencies and provide the environment yourself:

```bash
pip install horos --no-deps
pip install pydantic flask pyyaml pillow "transformers>=5.1,<6" supervision
# torch/torchvision: use the NVIDIA wheel matching your JetPack version
```

`horos` warns at backend load time if it detects a Jetson platform where
`torch.cuda.is_available()` is False.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e . --no-deps
pip install pydantic flask pyyaml pillow pytest ruff
pytest tests/test_invariants.py && pytest
```
