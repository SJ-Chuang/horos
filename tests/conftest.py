import sys
from pathlib import Path

# Make tests/helpers importable (fake backend entrypoints resolve through it).
TESTS_ROOT = Path(__file__).parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))
