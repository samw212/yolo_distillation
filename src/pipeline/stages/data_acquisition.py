"""Stage 1: Data Acquisition from Roboflow.

Pulls the dataset from a Roboflow project and organizes images
for downstream auto-labeling with Grounding DINO.
"""

import logging
import os
import shutil
from pathlib import Path

from roboflow import Roboflow

logger = logging.getLogger("distill.data_acquisition")


def run(config: dict) -> dict:
    """Pull dataset from Roboflow and organize for the pipeline.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Dictionary with stage results (image paths, counts).
    """
    acq_cfg = config["data_acquisition"]
    data_dir = Path(config["paths"]["data_dir"])
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Roboflow client
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
    version = project.version(acq_cfg["version"])

    # Download dataset - use raw format to get unlabeled images
    # or yolov8 format if labels already exist (for evaluation comparison)
    download_dir = data_dir / "roboflow_download"
    logger.info(
        "Downloading dataset: %s/%s v%d (format: %s)",
        acq_cfg["workspace"],
        acq_cfg["project"],
        acq_cfg["version"],
        acq_cfg["format"],
    )

    dataset = version.download(
        model_format=acq_cfg["format"],
        location=str(download_dir),
        overwrite=True,
    )

    # Collect all images from the downloaded dataset
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = []

    for split_dir in ["train", "valid", "test"]:
        split_path = download_dir / split_dir / "images"
        if not split_path.exists():
            # Try alternate structure
            split_path = download_dir / split_dir
        if split_path.exists():
            for img_file in sorted(split_path.iterdir()):
                if img_file.suffix.lower() in image_extensions:
                    all_images.append(img_file)

    # Also check root-level images directory
    root_images = download_dir / "images"
    if root_images.exists():
        for img_file in sorted(root_images.iterdir()):
            if img_file.suffix.lower() in image_extensions:
                all_images.append(img_file)

    # Apply max_images limit if configured
    max_images = acq_cfg.get("max_images")
    if max_images and max_images < len(all_images):
        logger.info("Limiting to %d images (out of %d)", max_images, len(all_images))
        all_images = all_images[:max_images]

    # Copy images to pipeline data directory with consistent naming
    copied_images = []
    for img_path in all_images:
        dest = images_dir / img_path.name
        if not dest.exists():
            shutil.copy2(img_path, dest)
        copied_images.append(str(dest))

    # Collect existing ground truth labels if available (for evaluation)
    gt_labels_dir = data_dir / "gt_labels"
    gt_count = 0
    for split_dir in ["train", "valid", "test"]:
        labels_path = download_dir / split_dir / "labels"
        if labels_path.exists():
            gt_labels_dir.mkdir(parents=True, exist_ok=True)
            for label_file in labels_path.iterdir():
                if label_file.suffix == ".txt":
                    shutil.copy2(label_file, gt_labels_dir / label_file.name)
                    gt_count += 1

    logger.info(
        "Data acquisition complete: %d images collected, %d ground truth labels found",
        len(copied_images),
        gt_count,
    )

    return {
        "image_paths": copied_images,
        "image_count": len(copied_images),
        "gt_label_count": gt_count,
        "gt_labels_dir": str(gt_labels_dir) if gt_count > 0 else None,
        "download_dir": str(download_dir),
    }
