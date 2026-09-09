#!/usr/bin/env bash
set -Eeuo pipefail

# ----------------------------------------------------------------------
# release.sh — version, build, publish.
#
#   bash scripts/release.sh                # show current version
#   bash scripts/release.sh 0.2.0          # set version (pyproject + __init__)
#   bash scripts/release.sh --build        # test + build + twine check
#   bash scripts/release.sh --test         # build + upload to TestPyPI
#   bash scripts/release.sh --pypi         # build + upload to PyPI (asks first)
#
# Extra flags: -y/--yes (skip the PyPI prompt), --skip-test (skip local_test.sh).
# Combine: bash scripts/release.sh 0.2.0 --pypi
#
# Credentials (one-time): pip install twine keyring, then
#   keyring set https://upload.pypi.org/legacy/ __token__
#   keyring set https://test.pypi.org/legacy/ __token__
# ----------------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PYTHON_BIN:-python3}"
VERSION=""
BUILD="false"
TESTPYPI="false"
PYPI="false"
SKIP_TEST="false"
YES="false"

for arg in "$@"; do
  case "$arg" in
    --build)     BUILD="true" ;;
    --test)      TESTPYPI="true"; BUILD="true" ;;
    --pypi)      PYPI="true"; BUILD="true" ;;
    -y|--yes)    YES="true" ;;
    --skip-test) SKIP_TEST="true" ;;
    -h|--help)   sed -n '4,19p' "$0"; exit 0 ;;
    -*)          echo "ERROR: unknown flag $arg (see --help)"; exit 1 ;;
    *)           VERSION="$arg" ;;
  esac
done

pyproject_version() { grep -m1 '^version' pyproject.toml | sed 's/.*"\(.*\)"/\1/'; }
init_version() { grep -m1 '^__version__' horos/__init__.py | sed 's/.*"\(.*\)"/\1/'; }

# ---------------------------------------------------------------- version
if [[ -n "$VERSION" ]]; then
  if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([a-zA-Z0-9.\-]+)?$ ]]; then
    echo "ERROR: invalid version '$VERSION' (expected e.g. 0.2.0, 0.2.0rc1)"
    exit 1
  fi
  sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
  sed -i.bak "s/^__version__ = \".*\"/__version__ = \"$VERSION\"/" horos/__init__.py
  rm -f pyproject.toml.bak horos/__init__.py.bak
fi

PV=$(pyproject_version)
IV=$(init_version)
echo "version: $PV (pyproject.toml) / $IV (horos/__init__.py)"
if [[ "$PV" != "$IV" ]]; then
  echo "ERROR: the two version files disagree — fix or rerun with a version."
  exit 1
fi

if [[ "$BUILD" == "false" ]]; then
  [[ -n "$VERSION" ]] && echo "Updated. Commit it with the project's [Chore] format."
  exit 0
fi

# ------------------------------------------------------------------ build
if [[ "$SKIP_TEST" != "true" ]]; then
  echo "==> scripts/local_test.sh --lint"
  bash "$PROJECT_ROOT/scripts/local_test.sh" --lint
fi

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "==> Building $PV ..."
rm -rf dist build
python -m pip install -q --upgrade build twine
python -m build
python -m twine check dist/*
ls -lh dist/

# ---------------------------------------------------------------- publish
if [[ "$TESTPYPI" == "true" ]]; then
  echo "==> Uploading to TestPyPI ..."
  python -m twine upload --repository testpypi --non-interactive dist/*
  echo "Done. Verify (the extra index is required — horos's dependencies live"
  echo "on real PyPI, not TestPyPI):"
  echo "  pip install --index-url https://test.pypi.org/simple/ \\"
  echo "    --extra-index-url https://pypi.org/simple/ horos==$PV"
fi

if [[ "$PYPI" == "true" ]]; then
  if [[ "$YES" != "true" ]]; then
    read -rp "Upload horos $PV to PyPI? [y/N] " CONFIRM
    [[ "$CONFIRM" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
  fi
  echo "==> Uploading to PyPI ..."
  python -m twine upload --repository pypi --non-interactive dist/*
  echo "Done. Verify: pip install horos==$PV"
fi
