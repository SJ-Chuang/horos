"""horos — end-to-end tooling for perception tasks.

R1b: importing this package must never pull in torch / rfdetr / transformers.
Keep this module free of backend imports; `tests/test_invariants.py` enforces it
at runtime in a subprocess.
"""

__version__ = "0.1.1.dev0"

from horos import errors

__all__ = ["__version__", "errors"]
