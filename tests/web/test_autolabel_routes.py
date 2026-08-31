"""E3-T9: autolabel Web API endpoints (thin routes over horos.api)."""

import time

import pytest
from helpers.data import make_image, write_sample_coco_dir
from helpers.fake_backend import FakeOpenVocabBackend

from horos.web.app import create_app

PROMPTS = {"prompts": {"forklift": ["forklift"], "person": ["person"]}}


@pytest.fixture
def project(tmp_path):
    from horos.api import create_project, import_dataset

    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    proj.add_image(make_image(tmp_path / "u1.png", 64, 48), width=64, height=48)
    return proj


@pytest.fixture
def client(project, monkeypatch):
    import horos.api.autolabel as autolabel_module
    import horos.backends

    monkeypatch.setattr(
        horos.backends, "get_backend", lambda key, **kw: FakeOpenVocabBackend()
    )
    monkeypatch.setattr(autolabel_module, "_ASSIST_BACKENDS", {})
    app = create_app(project.root)
    app.testing = True
    return app.test_client()


def _poll_done(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/v1/jobs/{job_id}").get_json()
        if body["state"] != "running":
            return body
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_autolabel_job_flow(client):
    response = client.post("/api/v1/autolabel", json=PROMPTS)
    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    body = _poll_done(client, job_id)
    assert body["state"] == "completed"
    assert body["events"][0]["type"] == "started"
    assert body["events"][-1]["type"] == "completed"
    # incremental polling
    tail = client.get(f"/api/v1/jobs/{job_id}?after={body['num_events'] - 1}").get_json()
    assert len(tail["events"]) == 1


def test_autolabel_requires_prompts(client):
    response = client.post("/api/v1/autolabel", json={})
    assert response.status_code == 400
    assert "prompts" in response.get_json()["error"]["message"]


def test_unknown_job_is_404_style_error(client):
    response = client.get("/api/v1/jobs/nope")
    assert response.status_code == 400
    assert "No such job" in response.get_json()["error"]["message"]


def test_pending_and_review_flow(client):
    job_id = client.post("/api/v1/autolabel", json=PROMPTS).get_json()["job_id"]
    _poll_done(client, job_id)
    pending = client.get("/api/v1/autolabel/pending").get_json()
    assert len(pending) == 1  # only the unannotated image got pre-labels
    image_id = pending[0]["image_id"]
    assert pending[0]["num_pending"] == 2
    reviewed = client.post(
        f"/api/v1/images/{image_id}/review",
        json={"action": "accept", "min_score": 0.85},
    ).get_json()
    assert reviewed == {"action": "accept", "count": 1}
    assert client.get("/api/v1/autolabel/pending").get_json() == []


def test_review_validates_action(client):
    response = client.post("/api/v1/images/1/review", json={"action": "merge"})
    assert response.status_code == 400


def test_assist_route(client, project):
    image_id = project.list_images()[0].id
    response = client.post(
        f"/api/v1/images/{image_id}/assist", json={**PROMPTS, "threshold": 0.5}
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["image_id"] == image_id
    assert sum(a["status"] == "pending" for a in body["annotations"]) == 2


def test_queue_reports_pending(client, project):
    job_id = client.post("/api/v1/autolabel", json=PROMPTS).get_json()["job_id"]
    _poll_done(client, job_id)
    queue = client.get("/api/v1/queue?mode=pending").get_json()
    assert len(queue) == 1
    assert queue[0]["num_pending"] == 2
    assert queue[0]["mean_pending_score"] == pytest.approx(0.85)
