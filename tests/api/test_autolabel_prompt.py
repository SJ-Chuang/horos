"""E3-T2: text-prompt to class mapping — one class may carry several prompts."""

import pytest

from horos.api.autolabel import PromptSpec
from horos.errors import DatasetValidationError


def test_one_class_many_prompts():
    spec = PromptSpec(prompts={"forklift": ["forklift", "lift truck"], "person": ["person"]})
    texts, classes = spec.flat()
    assert texts == ["forklift", "lift truck", "person"]
    assert classes == ["forklift", "forklift", "person"]


def test_prompts_are_stripped():
    spec = PromptSpec(prompts={"vest": ["  safety vest ", ""]})
    texts, classes = spec.flat()
    assert texts == ["safety vest"] and classes == ["vest"]


def test_empty_spec_is_rejected():
    with pytest.raises(DatasetValidationError, match="empty"):
        PromptSpec(prompts={}).flat()


def test_class_without_prompts_is_rejected():
    with pytest.raises(DatasetValidationError, match="at least one"):
        PromptSpec(prompts={"forklift": ["", "  "]}).flat()


def test_blank_class_name_is_rejected():
    with pytest.raises(DatasetValidationError, match="class name"):
        PromptSpec(prompts={"  ": ["x"]}).flat()
