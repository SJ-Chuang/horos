"""E7-T4: runs trained on different data are flagged as not comparable (E7-S4)."""

from __future__ import annotations

import json
import time

import pytest
from helpers.experiments import project_with_runs, train_fake

from horos.api.annotate import get_annotations, save_annotations
from horos.api.experiment import compare_runs, get_run_summary, query_runs
from horos.cli import main
from horos.errors import ProjectError


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """r1, r2 on the original data; then one train-split box moves and r3
    trains on the edited data."""
    project, (r1, r2) = project_with_runs(tmp_path_factory.mktemp("cmp"), epochs=(1, 2))
    view = get_annotations(project, 1)  # image 1 sits in the train split
    edited = [a.model_copy(update={"bbox": (a.bbox[0] + 1, *a.bbox[1:])}) for a in view.annotations]
    save_annotations(project, 1, edited, expected_version=view.version)
    time.sleep(1.05)
    r3 = train_fake(project, epochs=4)
    return project, r1, r2, r3


def test_runs_on_identical_data_are_comparable(world):
    project, r1, r2, _ = world
    judged = get_run_summary(project, r2.run_id, reference=r1.run_id).comparability
    assert judged is not None and judged.comparable
    assert judged.reference == r1.run_id and judged.changed_splits == []
    assert "same data" in judged.reason


def test_a_moved_train_box_breaks_comparability_by_split(world):
    project, r1, _, r3 = world
    judged = get_run_summary(project, r3.run_id, reference=r1.run_id).comparability
    assert judged is not None and not judged.comparable
    assert judged.changed_splits == ["train"] and not judged.classes_changed
    assert "train split(s) differ" in judged.reason and "different data" in judged.reason


def test_project_reference_flags_runs_older_than_the_last_edit(world):
    project, r1, r2, r3 = world
    result = query_runs(project)  # reference defaults to the project's data today
    assert result.reference == "project"
    by_id = {s.run.run_id: s.comparability for s in result.runs}
    assert by_id[r3.run_id].comparable
    assert not by_id[r1.run_id].comparable and not by_id[r2.run_id].comparable
    assert by_id[r1.run_id].changed_splits == ["train"]
    assert "project's current data" in by_id[r1.run_id].reason
    # reference=None skips the judgement entirely
    assert all(s.comparability is None for s in query_runs(project, reference=None).runs)


def test_class_scoped_run_is_judged_within_its_own_scope(world):
    project, *_ = world
    time.sleep(1.05)
    scoped = train_fake(project, epochs=1, categories=["pallet"], include_background=True)
    judged = get_run_summary(project, scoped.run_id).comparability
    # its fingerprint differs from the full-class runs, but the project's
    # pallet-only view has not changed since it trained: comparable
    assert judged is not None and judged.comparable
    assert scoped.dataset_fingerprint.classes == ["pallet"]


def test_compare_runs_highlights_what_changed(world):
    project, r1, r2, r3 = world
    comparison = compare_runs(project, [r1.run_id, r2.run_id, r3.run_id])
    assert [s.run.run_id for s in comparison.runs] == [r1.run_id, r2.run_id, r3.run_id]
    # comparability is judged against the first run
    flags = [s.comparability.comparable for s in comparison.runs]
    assert flags == [True, True, False]

    hparams = {row.name: row for row in comparison.hparams}
    assert hparams["epochs"].values == [1, 2, 4] and hparams["epochs"].differs
    assert all(hparams["epochs"].reasons)  # every value carries its reason
    assert not hparams["batch_size"].differs  # same derivation on the same data

    metrics = {row.name: row for row in comparison.metrics}
    assert metrics["loss"].values == pytest.approx([1.0, 0.5, 0.25]) and metrics["loss"].differs

    dataset = {row.name: row for row in comparison.dataset}
    assert dataset["fingerprint"].differs and not dataset["images"].differs


def test_compare_runs_input_validation(world):
    project, r1, *_ = world
    with pytest.raises(ProjectError, match="at least one"):
        compare_runs(project, [])
    with pytest.raises(ProjectError, match="twice"):
        compare_runs(project, [r1.run_id, r1.run_id])
    with pytest.raises(ProjectError, match="No such training run"):
        compare_runs(project, [r1.run_id, "ghost"])


def test_cli_compare_and_runs_report_comparability(world, capsys):
    project, r1, r2, r3 = world
    assert main(["compare", r1.run_id, r3.run_id, "--project", str(project.root)]) == 0
    body = json.loads(capsys.readouterr().out)
    assert [r["comparable"] for r in body["runs"]] == [True, False]
    assert {row["name"] for row in body["hparams"]} >= {"epochs"}
    assert all(row["differs"] for row in body["metrics"])  # only differing rows by default

    assert main(["runs", "--project", str(project.root), "--reference", r1.run_id]) == 0
    rows = {r["run_id"]: r for r in json.loads(capsys.readouterr().out)}
    assert rows[r2.run_id]["comparable"] is True and rows[r3.run_id]["comparable"] is False


def test_runs_whose_classes_were_deleted_are_flagged_not_crashed(tmp_path):
    """After 'delete all data' (or a class deletion) a scoped run's classes no
    longer exist: the Experiments listing must report it as not comparable,
    never raise 'Unknown category'."""
    from helpers.experiments import project_with_runs, train_fake

    from horos.api.labels import delete_category

    project, _ = project_with_runs(tmp_path, epochs=(1,))
    time.sleep(1.05)
    scoped = train_fake(project, epochs=1, categories=["pallet"], include_background=True)
    pallet = next(c for c in project.categories if c.name == "pallet")
    delete_category(project, pallet.id, force=True)

    result = query_runs(project)  # reference = the project's data today
    judged = {s.run.run_id: s.comparability for s in result.runs}[scoped.run_id]
    assert judged is not None and not judged.comparable and judged.classes_changed
    assert "no longer exist" in judged.reason and "pallet" in judged.reason
    # the single-run view and a run-to-run comparison keep working too
    assert get_run_summary(project, scoped.run_id).comparability.comparable is False
    assert compare_runs(project, [s.run.run_id for s in result.runs[:2]]).runs
