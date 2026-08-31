"""E1-T7: dataset statistics — class distribution, relative area, image sizes."""

import pytest
from helpers.data import sample_dataset

from horos.core.dataset import Dataset
from horos.core.stats import compute_stats


@pytest.fixture
def stats():
    return compute_stats(sample_dataset())


def test_basic_counts(stats):
    assert stats.num_images == 3
    assert stats.num_annotations == 4
    assert stats.num_categories == 2


def test_per_class_instances_and_image_coverage(stats):
    by_name = {c.name: c for c in stats.per_class}
    assert by_name["forklift"].instances == 2
    assert by_name["pallet"].instances == 2
    assert by_name["forklift"].images == 2
    assert by_name["pallet"].images == 2


def test_split_counts(stats):
    assert stats.split_counts == {"train": 2, "valid": 1}


def test_image_size_distribution(stats):
    sizes = {(s.width, s.height): s.count for s in stats.image_sizes}
    assert sizes == {(64, 48): 2, (32, 32): 1}


def test_relative_area(stats):
    area = stats.relative_area
    assert area is not None
    # image 2: bbox 32x24 on 64x48 => exactly 0.25 relative area
    assert area.maximum == pytest.approx(0.25)
    assert 0 < area.minimum < area.maximum
    assert sum(area.histogram) == 4
    assert len(area.histogram) == 10


def test_imbalance_ratio(stats):
    assert stats.imbalance_ratio == pytest.approx(1.0)  # 2 vs 2


def test_annotated_vs_unannotated(stats):
    assert stats.annotated_images == 3
    assert stats.unannotated_images == 0


def test_empty_dataset_does_not_crash():
    stats = compute_stats(Dataset())
    assert stats.num_images == 0
    assert stats.relative_area is None
    assert stats.imbalance_ratio is None


def test_stats_serialize_for_web():
    payload = compute_stats(sample_dataset()).model_dump()
    assert payload["num_images"] == 3


def test_empty_categories_are_not_listed():
    from horos.core.dataset import Category, default_color

    dataset = sample_dataset()
    dataset.categories.append(
        Category(id=99, name="never_annotated", color=default_color(9))
    )
    stats = compute_stats(dataset)
    assert "never_annotated" not in [c.name for c in stats.per_class]
    assert stats.num_categories == len(stats.per_class) == 2
