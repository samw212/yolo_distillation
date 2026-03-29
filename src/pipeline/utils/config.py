"""Configuration management for the distillation pipeline."""

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "configs/default.yaml", overrides: dict | None = None) -> dict:
    """Load pipeline configuration from YAML file with optional overrides.

    Args:
        config_path: Path to the YAML configuration file.
        overrides: Dictionary of override values (dot-notation keys supported).

    Returns:
        Merged configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if overrides:
        for key, value in overrides.items():
            _set_nested(config, key, value)

    # Resolve environment variables
    _resolve_env_vars(config)

    return config


def _set_nested(d: dict, key: str, value: Any) -> None:
    """Set a value in a nested dictionary using dot-notation key."""
    keys = key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _resolve_env_vars(config: dict) -> None:
    """Resolve ${ENV_VAR} patterns in string config values."""
    for key, value in config.items():
        if isinstance(value, dict):
            _resolve_env_vars(value)
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            config[key] = os.environ.get(env_var, value)


def ensure_dirs(config: dict) -> None:
    """Create all output directories specified in the config."""
    paths = config.get("paths", {})
    for name, path in paths.items():
        Path(path).mkdir(parents=True, exist_ok=True)

    # Also create markers directory
    markers_dir = config.get("pipeline", {}).get("markers_dir", "output/.markers")
    Path(markers_dir).mkdir(parents=True, exist_ok=True)
