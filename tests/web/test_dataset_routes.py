"""E1-T9: dataset Web API endpoints (thin routes over horos.api)."""

import io
import zipfile

import pytest
from helpers.data import write_sample_coco_dir

from horos.web.app import create_app


@pytest.fixture
def project_root(tmp_path):
    from horos.api import create_project, import_dataset

    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    project = create_project(tmp_path / "proj")
    import_dataset(project, coco_dir)
    return project.root


@pytest.fixture
def client(project_root):
    app = create_app(project_root)
    app.testing = True
    return app.test_client()


def _zip_of(directory) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for f in directory.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(directory))
    buffer.seek(0)
    return buffer


def test_project_summary(client):
    body = client.get("/api/v1/project").get_json()
    assert body["name"] == "proj"
    assert body["num_images"] == 3
    assert {c["name"] for c in body["categories"]} == {"forklift", "pallet"}


def test_stats_route(client):
    body = client.get("/api/v1/dataset/stats").get_json()
    assert body["num_images"] == 3
    assert body["num_annotations"] == 4


def test_validation_route(client):
    body = client.get("/api/v1/dataset/validation").get_json()
    assert body["ok"] is True
    assert body["issues"] == []


def test_images_route(client):
    body = client.get("/api/v1/images").get_json()
    assert len(body) == 3
    assert {"id", "file_name", "width", "height", "split"} <= set(body[0])


def test_split_route(client):
    body = client.post(
        "/api/v1/dataset/split",
        json={"train": 1.0, "valid": 0.0, "test": 0.0, "seed": 5},
    ).get_json()
    assert body == {"train": 3, "valid": 0, "test": 0}


def test_upload_route(tmp_path):
    from horos.api import create_project

    project = create_project(tmp_path / "fresh")
    app = create_app(project.root)
    app.testing = True
    client = app.test_client()
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    response = client.post(
        "/api/v1/dataset/upload",
        data={"file": (_zip_of(coco_dir), "dataset.zip")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["num_images"] == 3
    assert body["instances_per_category"] == {"forklift": 2, "pallet": 2}


def test_import_by_path_route(tmp_path):
    from horos.api import create_project

    project = create_project(tmp_path / "fresh")
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    app = create_app(project.root)
    app.testing = True
    body = app.test_client().post(
        "/api/v1/dataset/import", json={"path": str(coco_dir)}
    ).get_json()
    assert body["format"] == "coco"
    assert body["num_images"] == 3


def test_export_route(client, tmp_path):
    body = client.post(
        "/api/v1/dataset/export",
        json={"out_dir": str(tmp_path / "out"), "format": "yolo"},
    ).get_json()
    assert body["path"].endswith("data.yaml")


def test_create_project_route(tmp_path):
    app = create_app()
    app.testing = True
    response = app.test_client().post(
        "/api/v1/projects", json={"path": str(tmp_path / "newproj"), "name": "n"}
    )
    assert response.status_code == 201
    assert response.get_json()["name"] == "n"


def test_models_route_carries_license(client):
    body = client.get("/api/v1/models").get_json()
    assert len(body) >= 4
    assert all(m["weights_license"] == "Apache-2.0" for m in body)


def test_capabilities_route(client):
    body = client.get("/api/v1/capabilities").get_json()
    features = {f["feature"] for f in body["features"]}
    assert "export_tensorrt" in features
