"""Architecture invariants — run these first in CI.

E4-T7  (R1)  : horos/{core,api,web,ui} never import torch/rfdetr/transformers.
               Static AST scan — in a single environment the deps are always
               installed, so a runtime check could never catch a violation.
E4-T8  (R2)  : horos/ui never imports horos core modules; horos/web never
               imports horos.core or horos.backends (routes go through horos.api).
E4-T10 (R1b) : after `import horos` (and the whole non-backend surface),
               sys.modules contains no torch/rfdetr/transformers. Runtime check
               in a subprocess — only a real import reveals transitive pulls.
               The two checks are complementary, not interchangeable.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

HOROS_ROOT = Path(__file__).parent.parent / "horos"

ML_DEPS = {"torch", "torchvision", "rfdetr", "transformers"}
LAYERED_DIRS = ("core", "api", "web", "ui")


def _imported_roots(path: Path) -> set[str]:
    """Every module root imported anywhere in the file (incl. inside functions)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def _imported_modules(path: Path) -> set[str]:
    """Full dotted module names imported in the file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def _py_files(*dirs: str):
    for d in dirs:
        base = HOROS_ROOT / d
        if base.exists():
            yield from sorted(base.rglob("*.py"))
    # top-level modules belong to the layered world too
    yield HOROS_ROOT / "__init__.py"
    yield HOROS_ROOT / "errors.py"
    yield HOROS_ROOT / "cli.py"


def test_r1_no_ml_imports_outside_backends():
    """E4-T7: model dependencies may only appear under horos/backends/."""
    violations = []
    for path in _py_files(*LAYERED_DIRS):
        bad = _imported_roots(path) & ML_DEPS
        if bad:
            violations.append(f"{path.relative_to(HOROS_ROOT.parent)}: {sorted(bad)}")
    assert violations == [], "R1 violations:\n" + "\n".join(violations)


def test_r2_ui_imports_no_horos_internals():
    """E4-T8: the WebUI talks HTTP only — no horos.core/api/backends imports."""
    forbidden_prefixes = ("horos.core", "horos.api", "horos.backends")
    violations = []
    for path in sorted((HOROS_ROOT / "ui").rglob("*.py")):
        for module in _imported_modules(path):
            if module.startswith(forbidden_prefixes) or module in {"horos"}:
                violations.append(f"{path.relative_to(HOROS_ROOT.parent)}: {module}")
    assert violations == [], "R2 (ui) violations:\n" + "\n".join(violations)


def test_r2_web_routes_go_through_api_layer():
    """R2: Web API routes call horos.api, never horos.core or the backends."""
    forbidden_prefixes = ("horos.core", "horos.backends")
    violations = []
    for path in sorted((HOROS_ROOT / "web").rglob("*.py")):
        for module in _imported_modules(path):
            if module.startswith(forbidden_prefixes):
                violations.append(f"{path.relative_to(HOROS_ROOT.parent)}: {module}")
    assert violations == [], "R2 (web) violations:\n" + "\n".join(violations)


def test_r1b_import_horos_pulls_no_ml_dependency():
    """E4-T10: runtime lazy-loading invariant, checked in a clean subprocess."""
    script = """
import sys
import horos
import horos.errors
import horos.core.registry
import horos.core.project
import horos.api
import horos.backends
import horos.backends.base
from horos.web.app import create_app
app = create_app()
from horos.core.registry import list_models
assert len(list_models()) >= 4
leaked = sorted({'torch', 'torchvision', 'rfdetr', 'transformers'} & set(sys.modules))
assert not leaked, f"ML deps leaked into sys.modules: {leaked}"
print("clean")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_backend_dirs_are_the_only_ml_import_sites():
    """Positive control: the scanner does see the sanctioned import sites."""
    rfdetr_imports = _imported_roots(HOROS_ROOT / "backends" / "rfdetr" / "__init__.py")
    assert "rfdetr" in rfdetr_imports
