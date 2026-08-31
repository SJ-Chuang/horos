"""E9-T5: the CLI runs the whole E1 workflow without a browser (E9-S3)."""

import json

import pytest
from helpers.data import write_sample_coco_dir

from horos.cli import main


def _run(capsys, *argv) -> tuple[int, dict | list]:
    code = main(list(argv))
    out = capsys.readouterr().out
    return code, (json.loads(out) if out.strip() else None)


def test_full_workflow(tmp_path, capsys):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    proj = tmp_path / "proj"

    code, body = _run(capsys, "init", str(proj), "--name", "demo")
    assert code == 0 and body["name"] == "demo"

    code, body = _run(capsys, "import", str(coco_dir), "--project", str(proj))
    assert code == 0 and body["num_images"] == 3

    code, body = _run(capsys, "stats", "--project", str(proj))
    assert code == 0 and body["num_annotations"] == 4

    code, body = _run(capsys, "split", "--project", str(proj),
                      "--train", "1.0", "--valid", "0.0", "--test", "0.0")
    assert code == 0 and body["train"] == 3

    code, body = _run(capsys, "export", str(tmp_path / "out"),
                      "--project", str(proj), "--format", "yolo")
    assert code == 0 and body["path"].endswith("data.yaml")


def test_validate_exit_code_reflects_dataset_health(tmp_path, capsys):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    proj = tmp_path / "proj"
    _run(capsys, "init", str(proj))
    _run(capsys, "import", str(coco_dir), "--project", str(proj))

    code, body = _run(capsys, "validate", "--project", str(proj))
    assert code == 0 and body["ok"] is True

    # break it: delete an image file
    from horos.api import open_project

    project = open_project(proj)
    (project.images_dir / project.list_images()[0].file_name).unlink()
    code, body = _run(capsys, "validate", "--project", str(proj))
    assert code == 1 and body["ok"] is False


def test_convert(tmp_path, capsys):
    coco_dir = write_sample_coco_dir(tmp_path / "coco")
    code, body = _run(capsys, "convert", str(coco_dir), str(tmp_path / "yolo"),
                      "--to", "yolo")
    assert code == 0 and body["path"].endswith("data.yaml")


def test_models_lists_licenses(capsys):
    code, body = _run(capsys, "models")
    assert code == 0
    assert all(m["weights_license"] == "Apache-2.0" for m in body)


def test_capabilities(capsys):
    code, body = _run(capsys, "capabilities")
    assert code == 0
    assert {f["feature"] for f in body["features"]} >= {"training", "export_tensorrt"}


def test_horos_errors_exit_2_with_stderr(tmp_path, capsys):
    code = main(["stats", "--project", str(tmp_path / "nope")])
    captured = capsys.readouterr()
    assert code == 2
    assert "error [project_error]" in captured.err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0


def test_train_streams_events_and_exits_by_state(tmp_path, capsys, monkeypatch):
    """`horos train` runs in the foreground: it starts a run, prints the event
    stream as JSONL, and its exit code mirrors the terminal state (E5/E9-S3)."""
    import os

    from helpers.data import write_sample_coco_dir as _make

    import horos.api as api
    from horos.api.train import TrainRunConfig
    from horos.cli import main as cli_main

    tests_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", tests_root + (os.pathsep + existing if existing else "")
    )

    proj_dir = tmp_path / "proj"
    project = api.create_project(proj_dir)
    api.import_dataset(project, _make(tmp_path / "coco"))

    # the CLI builds the config itself; reroute it onto the fake backend
    original = api.start_training

    def with_fake(project, config):
        patched = TrainRunConfig(
            **config.model_dump(exclude={"entrypoint_override"}),
            entrypoint_override="helpers.fake_backend:FakeBackend",
        )
        return original(project, patched)

    monkeypatch.setattr(api, "start_training", with_fake)

    code = cli_main(["train", "--project", str(proj_dir), "--epochs", "2"])
    out = capsys.readouterr().out
    # events stream as one-line JSON; the final run record is pretty-printed —
    # decode the concatenated stream object by object
    decoder, pos, payloads = json.JSONDecoder(), 0, []
    while pos < len(out):
        remainder = out[pos:].lstrip()
        if not remainder:
            break
        obj, consumed = decoder.raw_decode(remainder)
        payloads.append(obj)
        pos += (len(out[pos:]) - len(remainder)) + consumed
    assert code == 0
    types = [p.get("type") for p in payloads]
    assert "started" in types and "completed" in types
    # the last JSON payload is the final run record
    assert payloads[-1]["state"] == "completed"
