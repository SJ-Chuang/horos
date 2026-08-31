"""Weights download & cache (E3-T7).

Runtime-downloaded weights live under ~/.horos/weights/ (never bundled, §1).
Downloads resume from a .part file via HTTP Range and report progress through
a callback so callers can forward R4 ProgressUpdated events.

transformers-hosted models (OWLv2) manage their own files; they are pointed at
`hf_cache_dir()` so everything horos fetches stays under one root the user can
inspect and delete.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from horos.errors import WeightsError

logger = logging.getLogger(__name__)

_CHUNK = 1 << 18  # 256 KiB

ProgressFn = Callable[[int, int | None], None]  # (bytes_done, total_or_None)


def weights_root() -> Path:
    root = Path(os.environ.get("HOROS_WEIGHTS_DIR", Path.home() / ".horos" / "weights"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def hf_cache_dir() -> Path:
    """Cache dir handed to transformers' from_pretrained(cache_dir=...)."""
    path = weights_root() / "hf"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cached_download(
    url: str,
    *,
    filename: str | None = None,
    dest_dir: Path | None = None,
    progress: ProgressFn | None = None,
) -> Path:
    """Download `url` into the weights cache, resuming a partial download.

    Returns the cached file path immediately when it already exists. The
    in-flight file is `<name>.part`; only a completed download is renamed to
    its final name, so a crash can never leave a truncated file that looks
    complete.
    """
    dest_dir = Path(dest_dir) if dest_dir else weights_root()
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = filename or url.rsplit("/", 1)[-1] or "weights.bin"
    final = dest_dir / name
    if final.exists():
        if progress:
            size = final.stat().st_size
            progress(size, size)
        return final

    part = dest_dir / f"{name}.part"
    offset = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
        logger.info("resuming download of %s from byte %d", name, offset)

    try:
        response = urllib.request.urlopen(request)  # noqa: S310 — model weight URLs
    except urllib.error.HTTPError as exc:
        if exc.code == 416:  # requested range beyond end: .part is already complete
            os.replace(part, final)
            return final
        raise WeightsError(f"Download failed for {url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise WeightsError(f"Download failed for {url}: {exc.reason}") from exc

    with response:
        resuming = response.status == 206
        if offset and not resuming:
            offset = 0  # server ignored Range — start over
        length = response.headers.get("Content-Length")
        total = (int(length) + offset) if length is not None else None
        mode = "ab" if offset else "wb"
        done = offset
        with part.open(mode) as fh:
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    if progress and total is None:
        progress(done, done)
    os.replace(part, final)
    return final
