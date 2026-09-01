"""E5-T9 (first slice): training Web API endpoints (thin routes over horos.api)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from helpers.data import write_sample_coco_dir

from horos.web.app import create_app

TESTS_ROOT = Path(__file__).parent.parent
FAKE = "helpers.fake_backend:FakeBackend"


@pytest.fixture(autouse=True)
def worker_can_import_helpers(monkeypatch):
    existing = os.environ.get("PYTHONPATH", "")
    joined = str(TESTS_ROOT) + (os.pathsep + existing if existing else "")
    monkeypatch.setenv("PYTHONPATH", joined)


@pytest.fixture
def client(tmp_path):
    from horos.api import create_project, import_dataset

    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    app = create_app(proj.root)
    app.testing = True
    return app.test_client()


def _wait_terminal(client, run_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/v1/train/runs/{run_id}").get_json()
        if payload["run"]["state"] not in ("pending", "running"):
            return payload
        time.sleep(0.2)
    pytest.fail(f"run {run_id} still active after {timeout}s")


def test_train_run_over_http(client):
    response = client.post(
        "/api/v1/train", json={"entrypoint_override": FAKE, "epochs": 2}
    )
    assert response.status_code == 202
    run_id = response.get_json()["run_id"]

    payload = _wait_terminal(client, run_id)
    assert payload["run"]["state"] == "completed"
    assert payload["events"][-1]["type"] == "completed"

    listing = client.get("/api/v1/train/runs").get_json()
    assert [r["run_id"] for r in listing] == [run_id]

    # stopping a finished run is a no-op, reported honestly
    stop = client.post(f"/api/v1/train/runs/{run_id}/stop")
    assert stop.get_json() == {"stopped": False}


def test_invalid_config_is_a_client_error(client):
    response = client.post("/api/v1/train", json={"epochs": "not-a-number"})
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "project_error"


def test_unknown_run_is_404_shaped(client):
    response = client.get("/api/v1/train/runs/nope")
    assert response.status_code == 400
    assert "No such training run" in response.get_json()["error"]["message"]


def test_derive_preview_over_http(client):
    response = client.post("/api/v1/train/derive", json={"batch_size": 8})
    assert response.status_code == 200
    plan = response.get_json()
    assert plan["values"]["batch_size"] == 8
    batch = next(d for d in plan["derivations"] if d["name"] == "batch_size")
    assert batch["overridden"] is True
    epochs = next(d for d in plan["derivations"] if d["name"] == "epochs")
    assert epochs["overridden"] is False and epochs["reason"]
