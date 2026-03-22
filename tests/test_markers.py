"""Tests for stage completion markers."""

import tempfile
from pathlib import Path

import pytest

from src.pipeline.utils.markers import (
    clear_stage_marker,
    is_stage_complete,
    mark_stage_complete,
)


@pytest.fixture
def config(tmp_path):
    return {"pipeline": {"markers_dir": str(tmp_path / "markers")}}


def test_mark_and_check(config):
    mark_stage_complete(config, "test_stage")
    assert is_stage_complete(config, "test_stage")


def test_not_complete(config):
    assert not is_stage_complete(config, "nonexistent_stage")


def test_clear_marker(config):
    mark_stage_complete(config, "test_stage")
    assert is_stage_complete(config, "test_stage")
    clear_stage_marker(config, "test_stage")
    assert not is_stage_complete(config, "test_stage")


def test_mark_with_metadata(config):
    mark_stage_complete(config, "test_stage", {"elapsed": 42.5})
    assert is_stage_complete(config, "test_stage")
