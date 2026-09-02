"""E6-T2: media inference — photos, GIFs, and videos become per-frame
prediction galleries persisted under the run."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from helpers.data import write_sample_coco_dir
from PIL import Image

from horos.api.dataset import import_dataset
from horos.api.evaluate import _reset_backend_cache
from horos.api.media import (
    MAX_FRAMES,
    delete_media,
    get_media,
    list_media,
    media_inference_events,
)
from horos.api.project import create_project
from horos.api.train import TrainRunConfig, start_training, training_status
from horos.errors import ProjectError

TESTS_ROOT = Path(__file__).parent.parent
FAKE = "helpers.fake_backend:FakeBackend"

pytest.importorskip("imageio", reason="imageio not installed")


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", str(TESTS_ROOT) + (os.pathsep + existing if existing else "")
    )
    _reset_backend_cache()


@pytest.fixture
def trained(tmp_path):
    project = create_project(tmp_path / "proj")
    import_dataset(project, write_sample_coco_dir(tmp_path / "coco"))
    record = start_training(project, TrainRunConfig(entrypoint_override=FAKE, epochs=1))
    deadline = time.monotonic() + 30
    while training_status(project, record.run_id).run.state in ("pending", "running"):
        assert time.monotonic() < deadline
        time.sleep(0.2)
    return project, record.run_id


def _write_gif(path: Path, frames: int = 6, size=(64, 48)) -> Path:
    images = [
        Image.new("RGB", size, (30 * i % 255, 90, 160)) for i in range(frames)
    ]
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=100, loop=0
    )
    return path


def _run_stream(project, run_id, source, media_id="m1"):
    return list(
        media_inference_events(project, run_id, source, media_id=media_id)
    )


def test_gif_becomes_a_frame_gallery(trained, tmp_path):
    project, run_id = trained
    gif = _write_gif(tmp_path / "clip.gif", frames=6)
    events = _run_stream(project, run_id, gif)
    assert events[0].type == "started" and events[-1].type == "completed"
    assert events[-1].result["frames"] == 6

    item = get_media(project, run_id, "m1")
    assert item.kind == "video" and item.state == "completed"
    assert item.num_frames == 6 and len(item.frames) == 6
    media_dir = project.root / "runs" / run_id / "eval" / "media" / "m1"
    for frame in item.frames:
        assert (media_dir / frame.file_name).is_file()
        assert frame.width == 64 and frame.height == 48
        # the fake backend detects one instance per frame at score 0.9
        assert frame.instances and frame.instances[0]["score"] == 0.9


def test_single_image_is_a_one_frame_item(trained, tmp_path):
    project, run_id = trained
    photo = tmp_path / "photo.png"
    Image.new("RGB", (80, 60), (200, 40, 40)).save(photo)
    events = _run_stream(project, run_id, photo)
    assert events[-1].type == "completed" and events[-1].result["frames"] == 1
    item = get_media(project, run_id, "m1")
    assert item.kind == "image" and item.num_frames == 1


def test_long_media_is_stride_sampled_to_the_cap(trained, tmp_path, monkeypatch):
    import horos.api.media as media_mod

    monkeypatch.setattr(media_mod, "MAX_FRAMES", 4)
    project, run_id = trained
    gif = _write_gif(tmp_path / "long.gif", frames=10)
    events = _run_stream(project, run_id, gif)
    assert events[-1].type == "completed"
    item = get_media(project, run_id, "m1")
    assert item.num_frames <= 4
    # sampling keeps the ORIGINAL frame indices, spread across the clip
    indices = [f.index for f in item.frames]
    assert indices == sorted(indices) and indices[0] == 0 and indices[-1] >= 6
    assert MAX_FRAMES > 4  # the module constant itself is untouched


def test_listing_and_delete(trained, tmp_path):
    project, run_id = trained
    _run_stream(project, run_id, _write_gif(tmp_path / "a.gif", 3), media_id="a1")
    _run_stream(project, run_id, _write_gif(tmp_path / "b.gif", 2), media_id="b2")
    items = list_media(project, run_id)
    assert {m.media_id for m in items} == {"a1", "b2"}
    assert all(len(m.frames) == 1 for m in items)  # listings stay light

    assert delete_media(project, run_id, "a1") is True
    assert {m.media_id for m in list_media(project, run_id)} == {"b2"}
    with pytest.raises(ProjectError, match="no media item"):
        get_media(project, run_id, "a1")


def test_unsupported_suffix_is_refused(trained, tmp_path):
    project, run_id = trained
    weird = tmp_path / "notes.txt"
    weird.write_text("not media")
    with pytest.raises(ProjectError, match="Unsupported media type"):
        media_inference_events(project, run_id, weird, media_id="x")
