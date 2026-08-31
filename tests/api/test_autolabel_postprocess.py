"""E3-T4: confidence filtering and per-class NMS over raw predictions."""

from horos.api.autolabel import postprocess
from horos.backends.base import ImagePrediction, PredictedInstance

CLASSES = ["forklift", "forklift", "person"]  # prompts 0,1 -> forklift; 2 -> person


def _pred(instances):
    return ImagePrediction(image="x.png", instances=instances)


def _inst(bbox, score, idx):
    return PredictedInstance(bbox=bbox, score=score, category_id=idx)


def test_threshold_filters():
    pred = _pred([_inst((0, 0, 10, 10), 0.05, 0), _inst((0, 0, 10, 10), 0.6, 2)])
    kept = postprocess(pred, CLASSES, threshold=0.1)
    assert [(c, s) for c, _, s in kept] == [("person", 0.6)]


def test_nms_merges_same_class_overlaps():
    pred = _pred([
        _inst((0, 0, 100, 100), 0.9, 0),
        _inst((5, 5, 100, 100), 0.7, 0),  # heavy overlap, same class -> dropped
        _inst((300, 300, 50, 50), 0.6, 0),  # far away -> kept
    ])
    kept = postprocess(pred, CLASSES, nms_iou=0.5)
    assert len(kept) == 2
    assert kept[0][2] == 0.9 and kept[1][2] == 0.6


def test_nms_merges_across_prompts_of_same_class():
    # two different prompts firing on the same object keep only the better box
    pred = _pred([
        _inst((0, 0, 100, 100), 0.9, 0),
        _inst((2, 2, 100, 100), 0.8, 1),  # prompt 1 also maps to forklift
    ])
    kept = postprocess(pred, CLASSES, nms_iou=0.5)
    assert len(kept) == 1 and kept[0][0] == "forklift" and kept[0][2] == 0.9


def test_nms_keeps_overlapping_different_classes():
    pred = _pred([
        _inst((0, 0, 100, 100), 0.9, 0),
        _inst((0, 0, 100, 100), 0.8, 2),  # same box, different class -> kept
    ])
    kept = postprocess(pred, CLASSES, nms_iou=0.5)
    assert {c for c, _, _ in kept} == {"forklift", "person"}


def test_results_sorted_by_score():
    pred = _pred([
        _inst((0, 0, 10, 10), 0.3, 2),
        _inst((300, 0, 10, 10), 0.8, 0),
    ])
    kept = postprocess(pred, CLASSES)
    assert [s for _, _, s in kept] == [0.8, 0.3]
