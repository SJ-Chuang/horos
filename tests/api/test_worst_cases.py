"""E6-T5: worst-case mining — the images a model gets most wrong, ranked by
error count with missed area as the tie-break (confirmed design)."""

from __future__ import annotations

import json

import pytest
from helpers.runs import completed_fake_run

from horos.api.error_analysis import analyze_detections, rank_worst, worst_cases
from horos.api.evaluate import _write_detections
from horos.errors import ProjectError

GT = {
    "images": [
        {"id": 1, "file_name": "clean.png", "width": 100, "height": 100},
        {"id": 2, "file_name": "one_small_miss.png", "width": 100, "height": 100},
        {"id": 3, "file_name": "one_big_miss.png", "width": 100, "height": 100},
        {"id": 4, "file_name": "three_errors.png", "width": 100, "height": 100},
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30]},
        {"id": 2, "image_id": 2, "category_id": 1, "bbox": [10, 10, 10, 10]},
        {"id": 3, "image_id": 3, "category_id": 1, "bbox": [10, 10, 60, 60]},
        {"id": 4, "image_id": 4, "category_id": 1, "bbox": [10, 10, 20, 20]},
        {"id": 5, "image_id": 4, "category_id": 2, "bbox": [50, 50, 20, 20]},
    ],
    "categories": [{"id": 1, "name": "block"}, {"id": 2, "name": "cone"}],
}


def _det(image_id, category_id, bbox, score=0.9):
    return {"image_id": image_id, "category_id": category_id, "bbox": bbox, "score": score}


DETECTIONS = [
    _det(1, 1, [10, 10, 30, 30]),  # image 1: perfect
    # images 2 and 3: nothing predicted -> one miss each, different sizes
    _det(4, 2, [10, 10, 20, 20]),  # image 4: block called cone (confusion)
    _det(4, 1, [80, 80, 10, 10]),  # image 4: false positive
    #                              image 4: the cone is missed
]


def _ranked(threshold=0.5):
    _, per_image = analyze_detections(GT, DETECTIONS, threshold=threshold, iou=0.5)
    return rank_worst(per_image)


def test_images_are_ranked_by_error_count_then_missed_area():
    ranked = _ranked()
    assert [img.file_name for img in ranked] == [
        "three_errors.png",
        "one_big_miss.png",
        "one_small_miss.png",
    ]
    worst = ranked[0]
    assert (worst.errors, worst.fn, worst.fp, worst.confused, worst.tp) == (3, 1, 1, 1, 0)
    assert ranked[1].missed_area == pytest.approx(0.36)
    assert ranked[2].missed_area == pytest.approx(0.01)


def test_clean_images_are_left_out():
    assert all(img.file_name != "clean.png" for img in _ranked())


def test_every_error_is_listed_with_its_kind_and_classes():
    worst = _ranked()[0]
    kinds = {item.kind: item for item in worst.items}
    assert set(kinds) == {"confused", "fp", "fn"}
    assert (kinds["confused"].gt_name, kinds["confused"].pred_name) == ("block", "cone")
    assert kinds["confused"].gt_bbox == (10, 10, 20, 20)
    assert kinds["fp"].pred_name == "block" and kinds["fp"].score == 0.9
    assert kinds["fn"].gt_name == "cone" and kinds["fn"].bbox == (50, 50, 20, 20)


def test_ranking_follows_the_threshold():
    # raising the threshold above every score turns image 1 into a miss too
    ranked = _ranked(threshold=0.95)
    assert len(ranked) == 4
    # two ground-truth boxes, both missed now that the confusion is gone
    assert ranked[0].file_name == "three_errors.png"
    assert ranked[0].errors == 2 and ranked[0].fn == 2 and ranked[0].confused == 0
    # the single-miss images order by missed area: 0.36, 0.09 (clean.png), 0.01
    assert [img.file_name for img in ranked[1:]] == [
        "one_big_miss.png",
        "clean.png",
        "one_small_miss.png",
    ]
    assert ranked[2].missed_area == pytest.approx(0.09)


# ------------------------------------------------------- project entry point


def _persist(project, run_id, split, detections):
    _write_detections(project, run_id, split, detections)


def _split_gt(project, run_id, split):
    path = project.root / "runs" / run_id / "dataset" / split / "_annotations.coco.json"
    return json.loads(path.read_text("utf-8"))


def test_worst_cases_reports_totals_and_truncates_to_top_k(tmp_path):
    project, run = completed_fake_run(tmp_path, epochs=1)
    gt = _split_gt(project, run.run_id, "train")
    assert len(gt["images"]) == 2
    _persist(project, run.run_id, "train", [])  # nothing predicted: every gt missed
    report = worst_cases(project, run.run_id, "train", top_k=1)
    assert report.total_images == 2 and report.images_with_errors == 2
    assert len(report.images) == 1 and report.top_k == 1
    assert report.images[0].errors >= 1 and report.images[0].fn == report.images[0].errors
    assert (report.threshold, report.iou) == (0.5, 0.5)


def test_worst_cases_of_a_perfect_run_is_empty(tmp_path):
    project, run = completed_fake_run(tmp_path, epochs=1)
    gt = _split_gt(project, run.run_id, "train")
    _persist(
        project,
        run.run_id,
        "train",
        [_det(a["image_id"], a["category_id"], list(a["bbox"])) for a in gt["annotations"]],
    )
    report = worst_cases(project, run.run_id, "train")
    assert report.images == [] and report.images_with_errors == 0
    assert report.total_images == 2


def test_worst_cases_validates_top_k_and_needs_detections(tmp_path):
    project, run = completed_fake_run(tmp_path, epochs=1)
    with pytest.raises(ProjectError, match="top_k"):
        worst_cases(project, run.run_id, "train", top_k=0)
    with pytest.raises(ProjectError, match="run an evaluation first"):
        worst_cases(project, run.run_id, "train")
