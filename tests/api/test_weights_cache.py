"""E3-T7: resumable weights download with progress reporting."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from horos.backends.weights import (
    cached_download,
    download_events,
    hf_cache_dir,
    weights_root,
)
from horos.errors import WeightsError

PAYLOAD = bytes(range(256)) * 512  # 128 KiB


class _RangeHandler(BaseHTTPRequestHandler):
    requests_seen: list[str] = []

    def do_GET(self):  # noqa: N802 — http.server API
        type(self).requests_seen.append(self.headers.get("Range") or "")
        if self.path == "/missing.bin":
            self.send_error(404)
            return
        header_range = self.headers.get("Range")
        if header_range:
            start = int(header_range.split("=")[1].rstrip("-").split("-")[0])
            if start >= len(PAYLOAD):
                self.send_error(416)
                return
            body = PAYLOAD[start:]
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}"
            )
        else:
            body = PAYLOAD
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep pytest output clean
        pass


@pytest.fixture
def server():
    _RangeHandler.requests_seen = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_full_download_with_progress(server, tmp_path):
    seen = []
    path = cached_download(
        f"{server}/weights.bin",
        dest_dir=tmp_path,
        progress=lambda done, total: seen.append((done, total)),
    )
    assert path.read_bytes() == PAYLOAD
    assert not path.with_name(path.name + ".part").exists()
    assert seen[-1] == (len(PAYLOAD), len(PAYLOAD))
    assert all(total == len(PAYLOAD) for _, total in seen)


def test_cached_file_short_circuits(server, tmp_path):
    cached_download(f"{server}/weights.bin", dest_dir=tmp_path)
    requests_before = len(_RangeHandler.requests_seen)
    again = cached_download(f"{server}/weights.bin", dest_dir=tmp_path)
    assert again.read_bytes() == PAYLOAD
    assert len(_RangeHandler.requests_seen) == requests_before  # no new request


def test_resume_from_partial(server, tmp_path):
    # simulate an interrupted download: half the payload sits in the .part file
    half = len(PAYLOAD) // 2
    (tmp_path / "weights.bin.part").write_bytes(PAYLOAD[:half])
    seen = []
    path = cached_download(
        f"{server}/weights.bin",
        dest_dir=tmp_path,
        progress=lambda done, total: seen.append((done, total)),
    )
    assert path.read_bytes() == PAYLOAD  # bytes are correct, not doubled
    assert _RangeHandler.requests_seen[-1] == f"bytes={half}-"
    assert seen[0][0] > half  # progress starts from the resumed offset


def test_complete_part_file_is_finalized(server, tmp_path):
    (tmp_path / "weights.bin.part").write_bytes(PAYLOAD)
    path = cached_download(f"{server}/weights.bin", dest_dir=tmp_path)
    assert path.read_bytes() == PAYLOAD


def test_http_error_is_explicit(server, tmp_path):
    with pytest.raises(WeightsError, match="404"):
        cached_download(f"{server}/missing.bin", dest_dir=tmp_path)


def test_cache_dirs_are_under_one_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOROS_WEIGHTS_DIR", str(tmp_path / "weights"))
    root = weights_root()
    assert root == tmp_path / "weights" and root.is_dir()
    assert hf_cache_dir() == root / "hf"


def _drain(gen):
    """Exhaust an event generator, capturing its StopIteration return value."""
    seen = []
    while True:
        try:
            seen.append(next(gen))
        except StopIteration as stop:
            return seen, stop.value


def test_download_events_stream_progress_and_return_path(server, tmp_path):
    events, path = _drain(
        download_events(
            f"{server}/weights.bin", dest_dir=tmp_path, label="downloading weights.bin"
        )
    )
    assert path.read_bytes() == PAYLOAD
    assert events, "a fresh download must emit progress"
    final = events[-1]
    assert final.type == "progress"
    assert (final.current, final.total) == (len(PAYLOAD), len(PAYLOAD))
    assert final.phase.startswith("downloading weights.bin:")
    # bytes must be strictly increasing — the throttle drops repeats, not order
    currents = [e.current for e in events]
    assert currents == sorted(set(currents))


def test_download_events_propagate_failure(server, tmp_path):
    with pytest.raises(WeightsError, match="404"):
        _drain(download_events(f"{server}/missing.bin", dest_dir=tmp_path))
