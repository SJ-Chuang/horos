"""Cross-platform file primitives (R7).

Two facts of life that the first Windows CI run surfaced:

- Two processes (the training worker and the parent reconciling its state)
  may write the same JSON at the same time. An atomic write must use a tmp
  name unique to the writer, or one `os.replace` steals the other's tmp
  file and the second raises FileNotFoundError.
- On Windows a file another process has open cannot be replaced or deleted
  (sharing violation, `PermissionError`), and a reader that opens a file in
  the instant it is being replaced can hit the same error. These are
  transient: retrying for a moment is the standard answer. POSIX never
  raises them for these reasons, so the retries cost nothing there.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

#: how long a transient sharing violation is waited out before giving up
RETRY_ATTEMPTS = 40
RETRY_DELAY = 0.05  # seconds; ~2 s in total


def _retry(action, *, what: str, attempts: int = RETRY_ATTEMPTS, delay: float = RETRY_DELAY):
    for attempt in range(attempts):
        try:
            return action()
        except PermissionError:
            if attempt == attempts - 1:
                raise
            if attempt == 0:
                logger.debug("%s: sharing violation, retrying", what)
            gc.collect()  # an unreferenced but not yet closed handle is the usual culprit
            time.sleep(delay)
    return None  # pragma: no cover - the loop returns or raises


def replace_with_retry(src: Path, dst: Path) -> None:
    """`os.replace`, waiting out a Windows sharing violation on `dst`."""
    _retry(lambda: os.replace(src, dst), what=f"replace {dst.name}")


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write `text` so that a concurrent reader sees either the old or the
    new content, never a torn file, and concurrent writers never collide:
    a per-writer tmp file in the same directory, then an atomic replace."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(text, encoding=encoding)
        replace_with_retry(tmp, path)
    finally:
        if tmp.exists():  # the replace failed for good: do not leave litter
            try:
                tmp.unlink()
            except OSError:
                pass


def read_text_retry(path: Path, *, encoding: str = "utf-8") -> str:
    """`Path.read_text`, waiting out a replace that is in flight on Windows."""
    return _retry(lambda: Path(path).read_text(encoding), what=f"read {Path(path).name}")


def rmtree_retry(path: Path, *, ignore_errors: bool = False) -> None:
    """`shutil.rmtree`, waiting out handles another party still holds on
    Windows (a frame the browser is fetching while the user deletes)."""

    def _remove():
        shutil.rmtree(path)

    try:
        _retry(_remove, what=f"rmtree {Path(path).name}")
    except OSError:
        if not ignore_errors:
            raise
