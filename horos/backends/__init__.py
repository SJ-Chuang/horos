"""Backend loading (E4-T11) and the non-Apache license guard (E4-T9).

R1b: importing this module must not import torch/rfdetr/transformers. Backends
are resolved from the registry's entrypoint strings with importlib on first
`get_backend()` call — the single place lazy loading happens.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from horos.core import registry
from horos.errors import BackendError, LicenseError

if TYPE_CHECKING:
    from horos.backends.base import ModelBackend


def get_backend(
    key: str,
    *,
    acknowledge_non_apache: bool = False,
    device: str | None = None,
) -> ModelBackend:
    """Resolve a model key to a constructed backend instance.

    Non-Apache models (RF-DETR XL / 2XL, PML 1.0) are refused unless the caller
    passes ``acknowledge_non_apache=True`` — never silently allowed (§9).
    """
    info = registry.get_model_info(key)
    if info.requires_acknowledgement and not acknowledge_non_apache:
        raise LicenseError(
            f"Model '{key}' is licensed under {info.weights_license}, not Apache 2.0. "
            f"horos itself is Apache 2.0; using this model imposes additional terms "
            f"on you (see {info.license_url}). If you have reviewed the license and "
            f"accept it, pass acknowledge_non_apache=True."
        )

    module_name, _, class_name = info.entrypoint.partition(":")
    if not class_name:
        raise BackendError(
            f"Registry entrypoint for '{key}' is malformed: {info.entrypoint!r}"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise BackendError(
            f"Backend module '{module_name}' for model '{key}' could not be imported: "
            f"{exc}. Is the required dependency installed?",
            backend=info.family,
        ) from exc
    backend_cls = getattr(module, class_name, None)
    if backend_cls is None:
        raise BackendError(
            f"Backend module '{module_name}' has no class '{class_name}'",
            backend=info.family,
        )
    return backend_cls(info, device=device)
