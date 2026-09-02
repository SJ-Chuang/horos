"""E6-T9 (first slice): inference and evaluation Web API endpoints."""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import pytest
from helpers.data import make_image, write_sample_coco_dir

from horos.web.app import create_app

TESTS_ROOT = Path(__file__).parent.parent
FAKE = "helpers.fake_backend:FakeBackend"

pytest.importorskip("pycocotools", reason="training stack not installed")


@pytest.fixture(autouse=True)
def worker_can_import_helpers(monkeypatch):
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", str(TESTS_ROOT) + (os.pathsep + existing if existing else "")
    )
    from horos.api.evaluate import _reset_backend_cache

    _reset_backend_cache()


@pytest.fixture
def trained_client(tmp_path):
    from horos.api import create_project, import_dataset

    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    app = create_app(proj.root)
    app.testing = True
    client = app.test_client()

    response = client.post(
        "/api/v1/train", json={"entrypoint_override": FAKE, "epochs": 1}
    )
    run_id = response.get_json()["run_id"]
    deadline = time.time() + 30
    while time.time() < deadline:
        state = client.get(f"/api/v1/train/runs/{run_id}").get_json()["run"]["state"]
        if state not in ("pending", "running"):
            break
        time.sleep(0.2)
    assert state == "completed"
    return client, run_id, tmp_path


def test_infer_upload_roundtrip(trained_client):
    client, run_id, tmp_path = trained_client
    image_path = make_image(tmp_path / "probe.png", 64, 48)
    response = client.post(
        f"/api/v1/train/runs/{run_id}/infer",
        data={"file": (io.BytesIO(image_path.read_bytes()), "probe.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["image"] == "probe.png"  # temp server path never leaks
    assert payload["instances"][0]["score"] == 0.9


def test_infer_without_file_is_a_client_error(trained_client):
    client, run_id, _ = trained_client
    response = client.post(f"/api/v1/train/runs/{run_id}/infer", data={})
    assert response.status_code == 400
    assert "multipart" in response.get_json()["error"]["message"]


def test_evaluate_job_and_persisted_report(trained_client):
    client, run_id, _ = trained_client
    response = client.post(
        f"/api/v1/train/runs/{run_id}/evaluate", json={"split": "valid"}
    )
    assert response.status_code == 202
    job_id = response.get_json()["job_id"]

    deadline = time.time() + 30
    while time.time() < deadline:
        job = client.get(f"/api/v1/jobs/{job_id}").get_json()
        if job["state"] != "running":
            break
        time.sleep(0.2)
    assert job["state"] == "completed"
    assert job["events"][-1]["result"]["split"] == "valid"

    report = client.get(f"/api/v1/train/runs/{run_id}/eval/valid")
    assert report.status_code == 200
    assert report.get_json()["num_images"] == 1


def test_evaluate_missing_split_fails_synchronously(trained_client):
    client, run_id, _ = trained_client
    response = client.post(
        f"/api/v1/train/runs/{run_id}/evaluate", json={"split": "nope"}
    )
    assert response.status_code == 400
    assert "no 'nope' split" in response.get_json()["error"]["message"]


def _gif_bytes(frames: int = 3) -> bytes:
    from PIL import Image

    images = [Image.new("RGB", (48, 32), (40 * i, 80, 120)) for i in range(frames)]
    buffer = io.BytesIO()
    images[0].save(
        buffer, format="GIF", save_all=True, append_images=images[1:],
        duration=80, loop=0,
    )
    return buffer.getvalue()


def _wait_job(client, job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/v1/jobs/{job_id}").get_json()
        if job["state"] != "running":
            return job
        time.sleep(0.1)
    raise AssertionError("media job never finished")


def test_media_gif_upload_gallery_and_frame_serving(trained_client):
    client, run_id, _ = trained_client
    response = client.post(
        f"/api/v1/train/runs/{run_id}/media",
        data={"file": (io.BytesIO(_gif_bytes(3)), "clip.gif")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 202
    body = response.get_json()
    assert _wait_job(client, body["job_id"])["state"] == "completed"

    listing = client.get(f"/api/v1/train/runs/{run_id}/media").get_json()
    assert len(listing) == 1 and listing[0]["media_id"] == body["media_id"]
    assert listing[0]["kind"] == "video" and listing[0]["num_frames"] == 3

    detail = client.get(
        f"/api/v1/train/runs/{run_id}/media/{body['media_id']}"
    ).get_json()
    assert len(detail["frames"]) == 3
    assert detail["frames"][0]["instances"][0]["score"] == 0.9

    frame = client.get(
        f"/api/v1/train/runs/{run_id}/media/{body['media_id']}/"
        f"{detail['frames'][0]['file_name']}"
    )
    assert frame.status_code == 200
    assert frame.content_type.startswith("image/jpeg")

    deleted = client.delete(
        f"/api/v1/train/runs/{run_id}/media/{body['media_id']}"
    )
    assert deleted.get_json()["deleted"] is True
    assert client.get(f"/api/v1/train/runs/{run_id}/media").get_json() == []


def test_media_upload_without_file_is_a_client_error(trained_client):
    client, run_id, _ = trained_client
    response = client.post(f"/api/v1/train/runs/{run_id}/media", data={})
    assert response.status_code == 400


def test_relative_project_root_still_serves_frame_files(tmp_path, monkeypatch):
    """`horos ui my-project` hands create_app a RELATIVE path; Flask's file
    helpers resolve relative directories against the package dir, not the cwd
    — every frame 404'd until create_app resolved the root absolute."""
    from horos.api import create_project, import_dataset

    monkeypatch.chdir(tmp_path)
    proj = create_project(Path("proj"))
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    app = create_app("proj")  # relative on purpose
    app.testing = True
    client = app.test_client()

    run_id = client.post(
        "/api/v1/train", json={"entrypoint_override": FAKE, "epochs": 1}
    ).get_json()["run_id"]
    deadline = time.time() + 30
    while time.time() < deadline:
        state = client.get(f"/api/v1/train/runs/{run_id}").get_json()["run"]["state"]
        if state not in ("pending", "running"):
            break
        time.sleep(0.2)

    body = client.post(
        f"/api/v1/train/runs/{run_id}/media",
        data={"file": (io.BytesIO(_gif_bytes(2)), "clip.gif")},
        content_type="multipart/form-data",
    ).get_json()
    assert _wait_job(client, body["job_id"])["state"] == "completed"
    frame = client.get(
        f"/api/v1/train/runs/{run_id}/media/{body['media_id']}/frames/00000.jpg"
    )
    assert frame.status_code == 200
