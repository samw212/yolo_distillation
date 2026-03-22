"""Stage completion markers for pipeline resume support."""

import json
from datetime import datetime, timezone
from pathlib import Path


def mark_stage_complete(config: dict, stage_name: str, metadata: dict | None = None) -> None:
    """Mark a pipeline stage as completed.

    Args:
        config: Pipeline configuration dictionary.
        stage_name: Name of the completed stage.
        metadata: Optional metadata to store with the marker.
    """
    markers_dir = Path(config.get("pipeline", {}).get("markers_dir", "output/.markers"))
    markers_dir.mkdir(parents=True, exist_ok=True)

    marker_data = {
        "stage": stage_name,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }

    marker_file = markers_dir / f"{stage_name}.json"
    with open(marker_file, "w") as f:
        json.dump(marker_data, f, indent=2)


def is_stage_complete(config: dict, stage_name: str) -> bool:
    """Check if a pipeline stage has been completed.

    Args:
        config: Pipeline configuration dictionary.
        stage_name: Name of the stage to check.

    Returns:
        True if the stage marker exists.
    """
    markers_dir = Path(config.get("pipeline", {}).get("markers_dir", "output/.markers"))
    return (markers_dir / f"{stage_name}.json").exists()


def clear_stage_marker(config: dict, stage_name: str) -> None:
    """Remove a stage completion marker (to force re-run).

    Args:
        config: Pipeline configuration dictionary.
        stage_name: Name of the stage to clear.
    """
    markers_dir = Path(config.get("pipeline", {}).get("markers_dir", "output/.markers"))
    marker_file = markers_dir / f"{stage_name}.json"
    if marker_file.exists():
        marker_file.unlink()
