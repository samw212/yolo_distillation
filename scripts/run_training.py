"""Run YOLOv11 training and evaluation directly on labeled data.

This script bypasses the auto-labeling and VLM review stages when
labels are already available (e.g., from ground truth or pre-labeled data).

Usage:
    python scripts/run_training.py [--epochs 100] [--batch-size 16] [--model yolo11s.pt]
"""

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("distill.run")


CLASS_NAMES = [
    "safety helmet", "fire", "smoke", "human", "ladder", "working platform"
]


def prepare_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    split_ratios: dict,
    seed: int = 42,
) -> str:
    """Prepare dataset in YOLO format with train/val/test splits.

    Returns:
        Path to dataset.yaml file.
    """
    random.seed(seed)
    np.random.seed(seed)

    dataset_dir = output_dir / "dataset"
    for split in ["train", "val", "test"]:
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Match images to labels
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    labeled_pairs = []

    for label_file in sorted(labels_dir.glob("*.txt")):
        if label_file.stat().st_size == 0:
            continue
        stem = label_file.stem
        for ext in image_extensions:
            img_path = images_dir / f"{stem}{ext}"
            if img_path.exists():
                labeled_pairs.append((img_path, label_file))
                break

    if not labeled_pairs:
        raise FileNotFoundError(
            f"No matched image-label pairs found in {images_dir} and {labels_dir}"
        )

    logger.info("Found %d labeled images", len(labeled_pairs))

    # Shuffle and split
    random.shuffle(labeled_pairs)
    n = len(labeled_pairs)
    n_train = int(n * split_ratios.get("train", 0.8))
    n_val = int(n * split_ratios.get("val", 0.15))

    splits = {
        "train": labeled_pairs[:n_train],
        "val": labeled_pairs[n_train:n_train + n_val],
        "test": labeled_pairs[n_train + n_val:],
    }

    for split_name, pairs in splits.items():
        for img_path, label_path in pairs:
            shutil.copy2(img_path, dataset_dir / split_name / "images" / img_path.name)
            shutil.copy2(label_path, dataset_dir / split_name / "labels" / label_path.name)
        logger.info("  %s: %d images", split_name, len(pairs))

    # Create dataset.yaml
    dataset_yaml = {
        "path": str(dataset_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
        "nc": len(CLASS_NAMES),
    }

    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    logger.info("Dataset YAML: %s", yaml_path)
    return str(yaml_path)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 on labeled data")
    parser.add_argument("--images-dir", default="output/data/images", help="Images directory")
    parser.add_argument("--labels-dir", default="output/data/gt_labels", help="Labels directory")
    parser.add_argument("--output-dir", default="output/training", help="Output directory")
    parser.add_argument("--model", default="yolo11s.pt", help="YOLO model")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="cpu", help="Device (cpu, 0, 0,1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("YOLOv11 Training Pipeline")
    logger.info("=" * 60)

    # Step 1: Prepare dataset
    logger.info("Step 1: Preparing dataset splits...")
    dataset_yaml = prepare_dataset(
        images_dir, labels_dir, output_dir,
        split_ratios={"train": 0.8, "val": 0.15, "test": 0.05},
        seed=args.seed,
    )

    # Step 2: Train model
    logger.info("Step 2: Training YOLOv11...")
    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=dataset_yaml,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        patience=15,
        seed=args.seed,
        project=str(output_dir / "runs"),
        name="distill_train",
        exist_ok=True,
        verbose=True,
        # Augmentation
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        scale=0.5,
    )

    # Step 3: Evaluate
    logger.info("Step 3: Evaluating on test set...")
    best_model_path = output_dir / "runs" / "distill_train" / "weights" / "best.pt"

    if best_model_path.exists():
        best_model = YOLO(str(best_model_path))
        val_results = best_model.val(
            data=dataset_yaml,
            split="test",
            verbose=True,
        )

        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        if val_results:
            box = val_results.box
            logger.info("  mAP@50:     %.4f", box.map50)
            logger.info("  mAP@50:95:  %.4f", box.map)
            logger.info("  Precision:  %.4f", box.mp)
            logger.info("  Recall:     %.4f", box.mr)
            f1 = 2 * box.mp * box.mr / (box.mp + box.mr) if (box.mp + box.mr) > 0 else 0
            logger.info("  F1:         %.4f", f1)
    else:
        logger.warning("Best model not found at %s", best_model_path)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("Best model: %s", best_model_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
