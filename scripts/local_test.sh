#!/usr/bin/env bash
set -Eeuo pipefail

# ----------------------------------------------------------------------
# local_test.sh
#
# Run local tests against the installed package inside .venv.
# Designed as a pre-commit / pre-push gate.
#
# horos specifics:
#   - tests/test_invariants.py (R1/R1b/R2 architecture checks) runs FIRST,
#     matching the CI ordering mandated by the project docs.
#   - the import check also asserts lazy backends (R1b): `import horos`
#     must not pull in torch/rfdetr/transformers.
#
# Usage:
#   bash scripts/local_test.sh                                  # Full test suite
#   bash scripts/local_test.sh --quick                          # Import + CLI only
#   bash scripts/local_test.sh --lint                           # Also run ruff check
#   bash scripts/local_test.sh --pytest-args "-k test_xxx -v"   # Extra pytest args
# ----------------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR=".venv"
QUICK="false"
LINT="false"
PACKAGE_NAME="horos"
CLI_NAME="horos"
PYTEST_EXTRA_ARGS=""

usage() {
  cat << 'EOF'
Usage:
  bash scripts/local_test.sh [options]

Options:
  --quick                   Only run import check and CLI --help (skip pytest).
  --lint                    Also run ruff check.
  --pytest-args "<args>"    Extra arguments to pass to pytest.
  -h, --help                Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      QUICK="true"
      shift
      ;;
    --lint)
      LINT="true"
      shift
      ;;
    --pytest-args)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --pytest-args requires a value."
        exit 1
      fi
      PYTEST_EXTRA_ARGS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

# ------------------------------------------------------------------
# Activate virtual environment
# ------------------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: Virtual environment not found at $VENV_DIR"
  echo "Run 'bash scripts/setup_local.sh' first."
  exit 1
fi

echo "==> Activating virtual environment: $VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "==> Python: $(which python) ($(python --version 2>&1))"
echo "==> Package: $PACKAGE_NAME"
echo "==> CLI:     $CLI_NAME"
echo "==> Mode:    $(if [[ "$QUICK" == "true" ]]; then echo "quick"; else echo "full"; fi)"
echo ""

FAILED=0

# ------------------------------------------------------------------
# Import check (includes the R1b lazy-backend assertion)
# ------------------------------------------------------------------
echo "==> Checking import: $PACKAGE_NAME (with R1b lazy-backend assertion)"
if python << 'PYCHECK'
import sys
import time

t0 = time.time()
import horos  # noqa: F401

dt = time.time() - t0
leaked = sorted({"torch", "rfdetr", "transformers"} & set(sys.modules))
if leaked:
    raise SystemExit(f"R1b violated: import horos pulled in {leaked}")
print(f"  horos {horos.__version__} imported OK ({dt:.2f}s, lazy backends intact)")
PYCHECK
then
  echo "    PASS"
else
  echo "    FAIL: import check failed"
  FAILED=1
fi
echo ""

# ------------------------------------------------------------------
# CLI --help check
# ------------------------------------------------------------------
echo "==> Checking CLI: $CLI_NAME --help"
if $CLI_NAME --help >/dev/null 2>&1; then
  echo "    PASS"
else
  echo "    FAIL: $CLI_NAME --help returned non-zero"
  FAILED=1
fi
echo ""

# ------------------------------------------------------------------
# Full pytest (unless --quick): invariants first, then everything else
# ------------------------------------------------------------------
if [[ "$QUICK" != "true" ]]; then
  echo "==> Running architecture invariants (tests/test_invariants.py)..."
  if python -m pytest tests/test_invariants.py -q; then
    echo "    PASS"
  else
    echo "    FAIL: architecture invariants violated (R1/R1b/R2)"
    FAILED=1
  fi
  echo ""

  echo "==> Running pytest..."
  if [[ -n "$PYTEST_EXTRA_ARGS" ]]; then
    echo "    Extra args: $PYTEST_EXTRA_ARGS"
    # Word-splitting is intentional here for pytest args
    # shellcheck disable=SC2086
    if python -m pytest $PYTEST_EXTRA_ARGS; then
      echo "    PASS"
    else
      echo "    FAIL: pytest exited with errors"
      FAILED=1
    fi
  else
    if python -m pytest; then
      echo "    PASS"
    else
      echo "    FAIL: pytest exited with errors"
      FAILED=1
    fi
  fi
  echo ""
fi

# ------------------------------------------------------------------
# Lint check (if --lint)
# ------------------------------------------------------------------
if [[ "$LINT" == "true" ]]; then
  echo "==> Running lint checks..."
  if command -v ruff >/dev/null 2>&1; then
    echo "    Running: ruff check ."
    if ruff check .; then
      echo "    ruff: PASS"
    else
      echo "    ruff: FAIL"
      FAILED=1
    fi
  else
    echo "    ruff not found, skipping. (bash scripts/setup_local.sh --dev)"
  fi
  echo ""
fi

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
if [[ "$FAILED" -ne 0 ]]; then
  echo "============================================"
  echo "  LOCAL TEST FAILED"
  echo "============================================"
  exit 1
fi

echo "============================================"
echo "  LOCAL TEST PASSED"
echo "============================================"
