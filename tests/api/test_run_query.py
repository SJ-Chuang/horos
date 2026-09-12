"""E7-T3: query runs sorted by any metric, filtered by state or tag (E7-S3)."""

from __future__ import annotations

import json

import pytest
from helpers.experiments import project_with_runs, write_eval

from horos.api.experiment import query_runs, update_run_notes
from horos.cli import main
from horos.errors import ProjectError


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    project, records = project_with_runs(tmp_path_factory.mktemp("query"))
    r1, r2, r4 = records  # loss 1.0, 0.5, 0.25
    update_run_notes(project, r1.run_id, tags=["baseline"])
    update_run_notes(project, r4.run_id, tags=["baseline", "long"])
    write_eval(project, r2.run_id, map_50=0.9, map_5095=0.6)
    write_eval(project, r4.run_id, map_50=0.4, map_5095=0.3)
    return project, records


def test_default_order_is_newest_first(runs):
    project, records = runs
    result = query_runs(project)
    assert result.sort_by == "created_at" and result.descending
    assert [s.run.run_id for s in result.runs] == [r.run_id for r in reversed(records)]


def test_sort_by_a_score_key_both_directions(runs):
    project, (r1, r2, r4) = runs
    best_first = query_runs(project, sort_by="loss", descending=False)
    assert [s.run.run_id for s in best_first.runs] == [r4.run_id, r2.run_id, r1.run_id]
    assert [s.scores["loss"] for s in best_first.runs] == [0.25, 0.5, 1.0]
    worst_first = query_runs(project, sort_by="loss", descending=True)
    assert [s.run.run_id for s in worst_first.runs] == [r1.run_id, r2.run_id, r4.run_id]


def test_sort_by_evaluation_metric_puts_unevaluated_runs_last(runs):
    project, (r1, r2, r4) = runs
    result = query_runs(project, sort_by="eval.test.map_50")
    assert [s.run.run_id for s in result.runs] == [r2.run_id, r4.run_id, r1.run_id]
    ascending = query_runs(project, sort_by="eval.test.map_50", descending=False)
    # r1 has no evaluation: still last, not first
    assert [s.run.run_id for s in ascending.runs] == [r4.run_id, r2.run_id, r1.run_id]


def test_sort_keys_reflect_what_the_runs_actually_carry(runs):
    project, _ = runs
    keys = query_runs(project).sort_keys
    assert keys[:2] == ["created_at", "run_id"]
    assert "loss" in keys and "eval.test.map_50" in keys and "eval.test.map_5095" in keys
    assert "map50" not in keys  # the fake backend reports no mAP


def test_unknown_sort_key_lists_the_valid_ones(runs):
    project, _ = runs
    with pytest.raises(ProjectError, match="Unknown sort key 'mAP'.*created_at.*loss"):
        query_runs(project, sort_by="mAP")


def test_filter_by_tags_requires_all_tags(runs):
    project, (r1, r2, r4) = runs
    baseline = query_runs(project, tags=["Baseline"])
    assert {s.run.run_id for s in baseline.runs} == {r1.run_id, r4.run_id}
    both = query_runs(project, tags=["baseline", "long"])
    assert [s.run.run_id for s in both.runs] == [r4.run_id]
    assert query_runs(project, tags=["nope"]).runs == []


def test_filter_by_state(runs):
    project, records = runs
    assert len(query_runs(project, states=["completed"]).runs) == 3
    assert query_runs(project, states=["failed", "stopped"]).runs == []


def test_cli_runs_subcommand(runs, capsys):
    project, (r1, r2, r4) = runs
    code = main(["runs", "--project", str(project.root), "--sort", "loss", "--asc"])
    assert code == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["run_id"] for row in rows] == [r4.run_id, r2.run_id, r1.run_id]
    assert rows[0]["tags"] == ["baseline", "long"] and rows[0]["evals"]["test"]["map_50"] == 0.4
    assert rows[0]["fingerprint"].startswith("sha256:")

    code = main(["runs", "--project", str(project.root), "--tag", "long"])
    assert code == 0
    assert [row["run_id"] for row in json.loads(capsys.readouterr().out)] == [r4.run_id]

    code = main(["runs", "--project", str(project.root), "--sort", "bogus"])
    assert code != 0
