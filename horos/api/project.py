"""Project lifecycle — the Python API layer over horos.core.project (R2)."""

from __future__ import annotations

from pathlib import Path

from horos.api.manifest import capability
from horos.core.project import Project

__all__ = ["create_project", "open_project"]


@capability(
    "project.create",
    summary="Create a new empty horos project directory",
    web_route="/api/v1/projects",
    web_methods=("POST",),
    cli="init",
)
def create_project(path: Path | str, name: str | None = None) -> Project:
    """Create a horos project at `path`. Fails if one already exists there."""
    return Project.create(Path(path), name=name)


@capability(
    "project.open",
    summary="Open an existing horos project",
    not_web_because="The Web API is started against one project; opening is implicit.",
    cli=None,
    not_cli_because="Every CLI command takes --project; opening is implicit.",
)
def open_project(path: Path | str) -> Project:
    """Open and structurally validate an existing project."""
    return Project.open(Path(path))
