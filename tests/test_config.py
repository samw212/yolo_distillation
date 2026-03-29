"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.pipeline.utils.config import load_config, _set_nested, ensure_dirs


@pytest.fixture
def sample_config_file():
    """Create a temporary config file for testing."""
    config = {
        "classes": ["safety helmet", "fire", "smoke"],
        "paths": {
            "output_dir": "test_output",
            "data_dir": "test_output/data",
        },
        "pipeline": {
            "seed": 42,
            "log_level": "INFO",
            "markers_dir": "test_output/.markers",
        },
        "data_acquisition": {
            "workspace": "test-ws",
            "project": "test-proj",
            "version": 1,
            "format": "yolov8",
        },
        "auto_labeling": {
            "model_id": "IDEA-Research/grounding-dino-base",
            "box_threshold": 0.30,
            "text_threshold": 0.25,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        return f.name


def test_load_config(sample_config_file):
    """Test basic config loading."""
    config = load_config(sample_config_file)
    assert config["classes"] == ["safety helmet", "fire", "smoke"]
    assert config["auto_labeling"]["box_threshold"] == 0.30


def test_load_config_with_overrides(sample_config_file):
    """Test config loading with dot-notation overrides."""
    config = load_config(sample_config_file, {
        "auto_labeling.box_threshold": 0.5,
        "data_acquisition.workspace": "new-ws",
    })
    assert config["auto_labeling"]["box_threshold"] == 0.5
    assert config["data_acquisition"]["workspace"] == "new-ws"


def test_load_config_file_not_found():
    """Test error on missing config file."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")


def test_set_nested():
    """Test nested dictionary setting."""
    d = {"a": {"b": {"c": 1}}}
    _set_nested(d, "a.b.c", 2)
    assert d["a"]["b"]["c"] == 2

    _set_nested(d, "a.b.d", 3)
    assert d["a"]["b"]["d"] == 3


def test_ensure_dirs(sample_config_file, tmp_path):
    """Test directory creation."""
    config = load_config(sample_config_file, {
        "paths.output_dir": str(tmp_path / "output"),
        "paths.data_dir": str(tmp_path / "output" / "data"),
        "pipeline.markers_dir": str(tmp_path / "output" / ".markers"),
    })
    ensure_dirs(config)
    assert (tmp_path / "output").exists()
    assert (tmp_path / "output" / "data").exists()
    assert (tmp_path / "output" / ".markers").exists()


# Cleanup
@pytest.fixture(autouse=True)
def cleanup(sample_config_file):
    yield
    if os.path.exists(sample_config_file):
        os.unlink(sample_config_file)
