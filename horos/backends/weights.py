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
import queue
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

from horos.backends.base import ProgressUpdated
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


def download_events(
    url: str,
    *,
    filename: str | None = None,
    dest_dir: Path | None = None,
    label: str = "downloading weights",
) -> Iterator[ProgressUpdated]:
    """`cached_download` as an R4 event stream; returns the downloaded path.

    Progress is throttled to whole-percent steps (~100 events per file) so a
    multi-hundred-MB download doesn't flood the event log. `current`/`total`
    are BYTES, not epochs — consumers that count epochs must filter on phase.
    Use `path = yield from download_events(...)` to get the file path back.
    """
    events: queue.Queue[ProgressUpdated] = queue.Queue()
    last = [-1]

    def on_progress(done: int, total: int | None) -> None:
        if total:
            tick = int(done * 100 / total)
            phase = f"{label}: {done >> 20} / {total >> 20} MB"
        else:  # no Content-Length: report every 32 MiB instead
            tick = done >> 25
            phase = f"{label}: {done >> 20} MB"
        if tick == last[0]:
            return
        last[0] = tick
        events.put(ProgressUpdated(current=done, total=total, phase=phase))

    result: list[Path] = []
    failure: list[BaseException] = []

    def run() -> None:
        try:
            result.append(
                cached_download(
                    url, filename=filename, dest_dir=dest_dir, progress=on_progress
                )
            )
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller's thread
            failure.append(exc)

    worker = threading.Thread(target=run, name="horos-weights-download", daemon=True)
    worker.start()
    while worker.is_alive() or not events.empty():
        try:
            yield events.get(timeout=0.5)
        except queue.Empty:
            continue
    worker.join()
    if failure:
        raise failure[0]
    return result[0]
