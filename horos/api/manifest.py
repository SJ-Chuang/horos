"""Capability manifest (E9-T1).

Every public Python API function registers itself with @capability. The
manifest is the machine-readable answer to "what can horos do", and the
contract tests (E9-T2) compare it against the Web API's routes and the CLI.

Deliberate exceptions are first-class: a capability that is intentionally not
exposed over the Web must say why (`not_web_because=`) — parity gaps must be
declared, never implied by omission (§6 E9).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, Field

F = TypeVar("F", bound=Callable[..., Any])


class Capability(BaseModel):
    name: str  # dotted, e.g. "dataset.stats"
    summary: str
    func_ref: str  # "module:function"
    web_route: str | None = None  # Flask rule, e.g. "/api/v1/dataset/stats"
    web_methods: list[str] = Field(default_factory=list)
    not_web_because: str | None = None
    cli: str | None = None  # CLI subcommand name
    not_cli_because: str | None = None

    @property
    def web(self) -> bool:
        return self.web_route is not None


_REGISTRY: dict[str, Capability] = {}


def capability(
    name: str,
    *,
    summary: str,
    web_route: str | None = None,
    web_methods: tuple[str, ...] = ("POST",),
    not_web_because: str | None = None,
    cli: str | None = None,
    not_cli_because: str | None = None,
) -> Callable[[F], F]:
    if web_route is None and not_web_because is None:
        raise ValueError(
            f"capability '{name}': either expose a web_route or state "
            f"not_web_because — parity exceptions must be explicit (E9)"
        )
    if cli is None and not_cli_because is None:
        raise ValueError(
            f"capability '{name}': either name a cli subcommand or state "
            f"not_cli_because"
        )

    def decorator(func: F) -> F:
        if name in _REGISTRY:
            raise ValueError(f"capability '{name}' registered twice")
        _REGISTRY[name] = Capability(
            name=name,
            summary=summary,
            func_ref=f"{func.__module__}:{func.__qualname__}",
            web_route=web_route,
            web_methods=list(web_methods) if web_route else [],
            not_web_because=not_web_because,
            cli=cli,
            not_cli_because=not_cli_because,
        )
        func.__capability__ = name  # type: ignore[attr-defined]
        return func

    return decorator


def list_capabilities() -> list[Capability]:
    return sorted(_REGISTRY.values(), key=lambda c: c.name)


def get_capability(name: str) -> Capability | None:
    return _REGISTRY.get(name)
