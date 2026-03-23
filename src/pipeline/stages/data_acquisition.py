"""Stage 1: Data Acquisition.

Supports two modes:
- "roboflow": Pull dataset from a Roboflow project via API
- "local": Use images from a local directory

For raw Roboflow projects (no versions), use the project's image export.
"""

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger("distill.data_acquisition")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(source_dir: Path) -> list[Path]:
    """Recursively collect all image files from a directory."""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(source_dir.rglob(f"*{ext}"))
        images.extend(source_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def _collect_labels(source_dir: Path) -> list[Path]:
    """Recursively collect all YOLO-format label files."""
    return sorted(source_dir.rglob("*.txt"))


def _run_roboflow(config: dict) -> dict:
    """Pull dataset from Roboflow API."""
    from roboflow import Roboflow

    acq_cfg = config["data_acquisition"]
    data_dir = Path(config["paths"]["data_dir"])
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ROBOFLOW_API_KEY environment variable is not set. "
            "Set it with: export ROBOFLOW_API_KEY=your_key"
        )

    logger.info("Connecting to Roboflow workspace: %s", acq_cfg["workspace"])
    rf = Roboflow(api_key=api_key)
    workspace = rf.workspace(acq_cfg["workspace"])
    project = workspace.project(acq_cfg["project"])

    download_dir = data_dir / "roboflow_download"

    # Handle raw projects (no version) vs versioned projects
    version_num = acq_cfg.get("version")
    if version_num:
        version = project.version(version_num)
        logger.info(
            "Downloading dataset: %s/%s v%d (format: %s)",
            acq_cfg["workspace"], acq_cfg["project"],
            version_num, acq_cfg.get("format", "yolov8"),
        )
        version.download(
            model_format=acq_cfg.get("format", "yolov8"),
            location=str(download_dir),
            overwrite=True,
        )
    else:
        # Raw project - download images directly
        logger.info(
            "Downloading raw project images: %s/%s",
            acq_cfg["workspace"], acq_cfg["project"],
        )
        project.download_raw(location=str(download_dir))

    return _organize_downloaded(download_dir, data_dir, acq_cfg)


def _run_local(config: dict) -> dict:
    """Use images from a local directory."""
    acq_cfg = config["data_acquisition"]
    data_dir = Path(config["paths"]["data_dir"])
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    local_dir = Path(acq_cfg.get("local_dir", str(data_dir)))
    logger.info("Using local data from: %s", local_dir)

    if not local_dir.exists():
        raise FileNotFoundError(f"Local data directory not found: {local_dir}")

    # Look for images in the local directory
    local_images_dir = local_dir / "images"
    if not local_images_dir.exists():
        local_images_dir = local_dir

    all_images = _collect_images(local_images_dir)

    max_images = acq_cfg.get("max_images")
    if max_images and max_images < len(all_images):
        logger.info("Limiting to %d images (out of %d)", max_images, len(all_images))
        all_images = all_images[:max_images]

    # Copy/symlink images to pipeline directory
    copied_images = []
    for img_path in all_images:
        dest = images_dir / img_path.name
        if not dest.exists():
            shutil.copy2(img_path, dest)
        copied_images.append(str(dest))

    # Collect ground truth labels if available
    gt_labels_dir = data_dir / "gt_labels"
    gt_count = 0
    for labels_dir_name in ["labels", "gt_labels"]:
        src_labels = local_dir / labels_dir_name
        if src_labels.exists():
            gt_labels_dir.mkdir(parents=True, exist_ok=True)
            for label_file in _collect_labels(src_labels):
                if label_file.name != "classes.txt":
                    shutil.copy2(label_file, gt_labels_dir / label_file.name)
                    gt_count += 1

    logger.info(
        "Local data acquisition complete: %d images, %d GT labels",
        len(copied_images), gt_count,
    )

    return {
        "image_paths": copied_images,
        "image_count": len(copied_images),
        "gt_label_count": gt_count,
        "gt_labels_dir": str(gt_labels_dir) if gt_count > 0 else None,
    }


def _organize_downloaded(download_dir: Path, data_dir: Path, acq_cfg: dict) -> dict:
    """Organize downloaded Roboflow data into pipeline structure."""
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    all_images = _collect_images(download_dir)

    max_images = acq_cfg.get("max_images")
    if max_images and max_images < len(all_images):
        logger.info("Limiting to %d images (out of %d)", max_images, len(all_images))
        all_images = all_images[:max_images]

    copied_images = []
    for img_path in all_images:
        dest = images_dir / img_path.name
        if not dest.exists():
            shutil.copy2(img_path, dest)
        copied_images.append(str(dest))

    # Collect existing ground truth labels
    gt_labels_dir = data_dir / "gt_labels"
    gt_count = 0
    for label_file in download_dir.rglob("*.txt"):
        if label_file.name == "classes.txt":
            continue
        # Check if it's a label file (in a labels/ directory)
        if "labels" in str(label_file.parent).lower():
            gt_labels_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_file, gt_labels_dir / label_file.name)
            gt_count += 1

    logger.info(
        "Data acquisition complete: %d images, %d GT labels",
        len(copied_images), gt_count,
    )

    return {
        "image_paths": copied_images,
        "image_count": len(copied_images),
        "gt_label_count": gt_count,
        "gt_labels_dir": str(gt_labels_dir) if gt_count > 0 else None,
        "download_dir": str(download_dir),
    }


def run(config: dict) -> dict:
    """Run data acquisition stage.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Dictionary with stage results.
    """
    acq_cfg = config["data_acquisition"]
    source = acq_cfg.get("source", "roboflow")

    if source == "local":
        return _run_local(config)
    else:
        return _run_roboflow(config)
