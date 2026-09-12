"""R7: the file primitives every state file goes through survive what the
Windows CI runner and concurrent worker processes do to them."""

from __future__ import annotations

import json
import shutil
import threading

import pytest

from horos.core import fsutil
from horos.core.fsutil import atomic_write_text, read_text_retry, rmtree_retry


def test_concurrent_writers_never_collide_and_never_tear(tmp_path):
    """The training worker and the parent both rewrite run.json; with a
    shared tmp name one os.replace stole the other's file (FileNotFoundError
    on Ubuntu CI). Per-writer tmp names make every write land whole."""
    target = tmp_path / "run.json"
    errors: list[BaseException] = []

    def writer(n: int):
        try:
            for i in range(200):
                atomic_write_text(target, json.dumps({"writer": n, "i": i}))
        except BaseException as exc:  # noqa: BLE001 — collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    final = json.loads(target.read_text("utf-8"))  # whole, one writer's last value
    assert final["i"] == 199
    assert list(tmp_path.iterdir()) == [target]  # no tmp litter left behind


def test_replace_and_read_wait_out_a_sharing_violation(tmp_path, monkeypatch):
    """Windows raises PermissionError while another process holds the file;
    it clears within milliseconds, so a short retry is the right answer."""
    monkeypatch.setattr(fsutil, "RETRY_DELAY", 0.001)
    target = tmp_path / "state.json"
    target.write_text("old", "utf-8")
    real_replace = fsutil.os.replace
    failures = {"n": 3}

    def flaky_replace(src, dst):
        if failures["n"]:
            failures["n"] -= 1
            raise PermissionError(13, "sharing violation")
        real_replace(src, dst)

    monkeypatch.setattr(fsutil.os, "replace", flaky_replace)
    atomic_write_text(target, "new")
    assert target.read_text("utf-8") == "new" and failures["n"] == 0

    calls = {"n": 0}
    real_read_text = type(target).read_text

    def flaky_read_text(self, encoding=None, errors=None, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(13, "sharing violation")
        return real_read_text(self, encoding, errors, *args, **kwargs)

    monkeypatch.setattr(type(target), "read_text", flaky_read_text)
    assert read_text_retry(target) == "new" and calls["n"] == 3


def test_a_permanent_permission_error_still_surfaces(tmp_path, monkeypatch):
    monkeypatch.setattr(fsutil, "RETRY_DELAY", 0.0)
    monkeypatch.setattr(fsutil, "RETRY_ATTEMPTS", 3)
    target = tmp_path / "state.json"

    def always(src, dst):
        raise PermissionError(13, "locked for good")

    monkeypatch.setattr(fsutil.os, "replace", always)
    with pytest.raises(PermissionError, match="locked for good"):
        atomic_write_text(target, "x")
    assert list(tmp_path.iterdir()) == []  # the tmp file was cleaned up


def test_rmtree_retries_while_a_handle_is_still_open(tmp_path, monkeypatch):
    """Deleting a media item while the browser still fetches a frame: on
    Windows the first unlink fails; the retry succeeds once the handle is
    released (or collected)."""
    monkeypatch.setattr(fsutil, "RETRY_DELAY", 0.001)
    tree = tmp_path / "media"
    (tree / "frames").mkdir(parents=True)
    (tree / "frames" / "00000.jpg").write_bytes(b"x")
    real_rmtree = shutil.rmtree
    failures = {"n": 2}

    def flaky_rmtree(path, *args, **kwargs):
        if failures["n"]:
            failures["n"] -= 1
            raise PermissionError(32, "being used by another process")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(fsutil.shutil, "rmtree", flaky_rmtree)
    rmtree_retry(tree)
    assert not tree.exists() and failures["n"] == 0

    monkeypatch.setattr(fsutil, "RETRY_ATTEMPTS", 2)
    failures["n"] = 99
    (tree / "frames").mkdir(parents=True)
    with pytest.raises(PermissionError):
        rmtree_retry(tree)
    rmtree_retry(tree, ignore_errors=True)  # swallowed on request
