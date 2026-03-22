"""Stage 5: YOLOv11 Fine-tuning.

Trains a YOLOv11 model on the approved pseudo-labels (and any
human-corrected labels) using the Ultralytics framework.

This is the final knowledge transfer step: the detection capability
of Grounding DINO (teacher) is distilled into a lightweight
YOLOv11 model (student) through pseudo-label training.
"""

import logging
import random
import shutil
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

logger = logging.getLogger("distill.training")


def _prepare_dataset(config: dict) -> str:
    """Prepare the dataset in YOLO format with train/val/test splits.

    Creates a dataset.yaml file and organizes images and labels into
    the standard Ultralytics directory structure.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Path to the dataset.yaml file.
    """
    classes = config["classes"]
    train_cfg = config["training"]
    split_ratios = train_cfg.get("split", {"train": 0.8, "val": 0.15, "test": 0.05})

    images_dir = Path(config["paths"]["data_dir"]) / "images"
    approved_dir = Path(config["paths"]["labels_dir"]) / "approved"
    training_dir = Path(config["paths"]["training_dir"])

    # Create dataset directory structure
    dataset_dir = training_dir / "dataset"
    for split in ["train", "val", "test"]:
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Collect all images that have approved labels
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    labeled_images = []

    for label_file in sorted(approved_dir.glob("*.txt")):
        if label_file.stat().st_size == 0:
            continue  # Skip empty label files

        stem = label_file.stem
        # Find the corresponding image
        for ext in image_extensions:
            img_path = images_dir / f"{stem}{ext}"
            if img_path.exists():
                labeled_images.append((img_path, label_file))
                break

    if not labeled_images:
        raise FileNotFoundError(
            f"No labeled images found. Check {approved_dir} for label files "
            f"and {images_dir} for corresponding images."
        )

    logger.info("Found %d labeled images for training", len(labeled_images))

    # Shuffle and split
    seed = config.get("pipeline", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    random.shuffle(labeled_images)

    n = len(labeled_images)
    n_train = int(n * split_ratios["train"])
    n_val = int(n * split_ratios["val"])

    splits = {
        "train": labeled_images[:n_train],
        "val": labeled_images[n_train : n_train + n_val],
        "test": labeled_images[n_train + n_val :],
    }

    # Copy files to dataset structure
    for split_name, items in splits.items():
        for img_path, label_path in items:
            shutil.copy2(img_path, dataset_dir / split_name / "images" / img_path.name)
            shutil.copy2(label_path, dataset_dir / split_name / "labels" / label_path.name)
        logger.info("  %s split: %d images", split_name, len(items))

    # Create dataset.yaml
    dataset_yaml = {
        "path": str(dataset_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: cls for i, cls in enumerate(classes)},
        "nc": len(classes),
    }

    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    logger.info("Dataset YAML created: %s", yaml_path)

    return str(yaml_path)


def run(config: dict) -> dict:
    """Train YOLOv11 model on approved pseudo-labels.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Dictionary with training results (best model path, metrics).
    """
    train_cfg = config["training"]
    training_dir = Path(config["paths"]["training_dir"])

    # Prepare dataset
    dataset_yaml = _prepare_dataset(config)

    # Load pre-trained YOLOv11 model
    model_name = train_cfg.get("model", "yolo11s.pt")
    logger.info("Loading YOLOv11 model: %s", model_name)
    model = YOLO(model_name)

    # Resolve device
    device = train_cfg.get("device", "auto")
    if device == "auto":
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"

    # Build training arguments
    train_args = {
        "data": dataset_yaml,
        "epochs": train_cfg.get("epochs", 100),
        "batch": train_cfg.get("batch_size", 16),
        "imgsz": train_cfg.get("imgsz", 640),
        "patience": train_cfg.get("patience", 20),
        "lr0": train_cfg.get("lr0", 0.01),
        "lrf": train_cfg.get("lrf", 0.01),
        "momentum": train_cfg.get("momentum", 0.937),
        "weight_decay": train_cfg.get("weight_decay", 0.0005),
        "warmup_epochs": train_cfg.get("warmup_epochs", 3.0),
        "warmup_momentum": train_cfg.get("warmup_momentum", 0.8),
        "device": device,
        "workers": train_cfg.get("workers", 8),
        "project": str(training_dir / "runs"),
        "name": "distill_train",
        "exist_ok": True,
        "verbose": True,
        "seed": config.get("pipeline", {}).get("seed", 42),
    }

    # Add augmentation settings
    aug_cfg = train_cfg.get("augmentation", {})
    for key, value in aug_cfg.items():
        train_args[key] = value

    # Handle resume
    if train_cfg.get("resume", False):
        last_ckpt = training_dir / "runs" / "distill_train" / "weights" / "last.pt"
        if last_ckpt.exists():
            logger.info("Resuming training from %s", last_ckpt)
            model = YOLO(str(last_ckpt))
            train_args["resume"] = True

    # Train
    logger.info("Starting YOLOv11 training with %d epochs", train_args["epochs"])
    results = model.train(**train_args)

    # Collect training results
    best_model = training_dir / "runs" / "distill_train" / "weights" / "best.pt"
    last_model = training_dir / "runs" / "distill_train" / "weights" / "last.pt"

    training_results = {
        "best_model": str(best_model) if best_model.exists() else None,
        "last_model": str(last_model) if last_model.exists() else None,
        "dataset_yaml": dataset_yaml,
        "train_args": {k: str(v) if isinstance(v, Path) else v for k, v in train_args.items()},
    }

    # Extract key metrics from results
    if results and hasattr(results, "results_dict"):
        training_results["final_metrics"] = {
            k: float(v) if hasattr(v, "__float__") else str(v)
            for k, v in results.results_dict.items()
        }

    logger.info("Training complete. Best model: %s", best_model)

    return training_results
