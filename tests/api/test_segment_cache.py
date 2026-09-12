"""SAM-T2: the image encoder runs once per image; clicks reuse the embedding."""

from __future__ import annotations

import os
import time

import pytest
from helpers.data import make_image, write_sample_coco_dir
from helpers.fake_backend import FakePromptableSegmenter

from horos.api.dataset import import_dataset
from horos.api.project import create_project
from horos.api.segment import (
    EmbeddingCache,
    SegmentRequest,
    _reset_segmenters,
    prefetch_embedding,
    segment_image,
)


@pytest.fixture
def project(tmp_path):
    _reset_segmenters()
    proj = create_project(tmp_path / "proj")
    import_dataset(proj, write_sample_coco_dir(tmp_path / "coco"))
    yield proj
    _reset_segmenters()


def _click(x, y, **kw):
    return SegmentRequest(points=[(x, y)], labels=[1], **kw)


def test_clicks_on_one_image_embed_once(project):
    fake = FakePromptableSegmenter()
    first = segment_image(project, 1, _click(10, 10), backend=fake)
    second = segment_image(project, 1, _click(20, 20), backend=fake)
    third = segment_image(project, 1, SegmentRequest(box=(1, 1, 5, 5)), backend=fake)
    assert fake.embed_calls == [str(project.image_path(project.get_image(1)))]
    assert fake.segment_calls == 3
    assert (first.embedding_cached, second.embedding_cached, third.embedding_cached) == (
        False, True, True,
    )


def test_prefetch_warms_the_cache_for_the_first_click(project):
    fake = FakePromptableSegmenter()
    warm = prefetch_embedding(project, 2, backend=fake)
    assert warm.embedding_cached is False and warm.elapsed_ms >= 0
    assert prefetch_embedding(project, 2, backend=fake).embedding_cached is True
    assert segment_image(project, 2, _click(5, 5), backend=fake).embedding_cached is True
    assert len(fake.embed_calls) == 1


def test_a_changed_image_file_invalidates_its_embedding(project):
    fake = FakePromptableSegmenter()
    record = project.get_image(1)
    path = project.image_path(record)
    segment_image(project, 1, _click(10, 10), backend=fake)
    # touch with a different size and a later mtime: the encoder must rerun
    make_image(path, record.width, record.height, color=(1, 2, 3))
    future = time.time() + 5
    os.utime(path, (future, future))
    assert segment_image(project, 1, _click(10, 10), backend=fake).embedding_cached is False
    assert len(fake.embed_calls) == 2


def test_cache_is_per_model_and_bounded():
    cache = EmbeddingCache(capacity=2)
    calls = []

    def compute(tag):
        return lambda: calls.append(tag) or tag

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "a.png"
        path.write_bytes(b"x")
        assert cache.get_or_compute(("p", 1, "sam2.1-tiny", "auto"), path, compute("a"))[1] is False
        assert cache.get_or_compute(("p", 1, "sam-base", "auto"), path, compute("b"))[1] is False
        assert cache.get_or_compute(("p", 1, "sam2.1-tiny", "auto"), path, compute("a2"))[1] is True
        # a third key evicts the least recently used (sam-base)
        cache.get_or_compute(("p", 2, "sam2.1-tiny", "auto"), path, compute("c"))
        assert len(cache) == 2
        assert cache.peek(("p", 1, "sam-base", "auto"), path) is False
        assert cache.peek(("p", 1, "sam2.1-tiny", "auto"), path) is True
        assert calls == ["a", "b", "c"] and cache.hits == 1 and cache.misses == 3
