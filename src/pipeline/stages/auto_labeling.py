"""Stage 2: Auto-Labeling with Grounding DINO.

Uses Grounding DINO (open-vocabulary object detector) as a teacher model
to generate pseudo-labels for the dataset. This implements the Annotation
stage from the DART pipeline (arXiv:2407.09174).

The pseudo-labels are saved in YOLO format (class_id cx cy w h) normalized
to [0, 1] for direct use with Ultralytics training.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

logger = logging.getLogger("distill.auto_labeling")


def _get_device(device_cfg: str) -> str:
    """Resolve device configuration."""
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def _apply_nms(boxes: np.ndarray, scores: np.ndarray, labels: list, nms_threshold: float):
    """Apply class-aware Non-Maximum Suppression.

    Args:
        boxes: Array of shape (N, 4) in xyxy format.
        scores: Array of shape (N,).
        labels: List of N class labels.
        nms_threshold: IoU threshold for NMS.

    Returns:
        Filtered boxes, scores, and labels.
    """
    if len(boxes) == 0:
        return boxes, scores, labels

    # Class-aware NMS: apply NMS per class
    unique_labels = set(labels)
    keep_indices = []

    for cls_label in unique_labels:
        cls_mask = np.array([l == cls_label for l in labels])
        cls_indices = np.where(cls_mask)[0]

        if len(cls_indices) == 0:
            continue

        cls_boxes = boxes[cls_indices]
        cls_scores = scores[cls_indices]

        # Convert to torch for NMS
        boxes_t = torch.tensor(cls_boxes, dtype=torch.float32)
        scores_t = torch.tensor(cls_scores, dtype=torch.float32)

        from torchvision.ops import nms
        keep = nms(boxes_t, scores_t, nms_threshold)
        keep_indices.extend(cls_indices[keep.numpy()].tolist())

    keep_indices = sorted(keep_indices)
    return boxes[keep_indices], scores[keep_indices], [labels[i] for i in keep_indices]


def _filter_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list,
    img_w: int,
    img_h: int,
    min_area_ratio: float,
    max_area_ratio: float,
):
    """Filter boxes by size constraints.

    Args:
        boxes: Array of shape (N, 4) in xyxy format.
        scores: Array of shape (N,).
        labels: List of N class labels.
        img_w: Image width.
        img_h: Image height.
        min_area_ratio: Minimum box area as fraction of image area.
        max_area_ratio: Maximum box area as fraction of image area.

    Returns:
        Filtered boxes, scores, and labels.
    """
    if len(boxes) == 0:
        return boxes, scores, labels

    img_area = img_w * img_h
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_ratios = box_areas / img_area

    keep = (area_ratios >= min_area_ratio) & (area_ratios <= max_area_ratio)
    return boxes[keep], scores[keep], [l for l, k in zip(labels, keep) if k]


def _match_label_to_class(label: str, classes: list[str], class_prompts: dict[str, str]) -> int | None:
    """Match a Grounding DINO output label to a pipeline class index.

    Args:
        label: The label string from Grounding DINO.
        classes: List of target class names.
        class_prompts: Mapping of class names to their prompt strings.

    Returns:
        Class index or None if no match found.
    """
    label_lower = label.lower().strip()

    # Direct match
    for i, cls_name in enumerate(classes):
        if label_lower == cls_name.lower():
            return i

    # Check if label appears in any class prompt
    for i, cls_name in enumerate(classes):
        prompt = class_prompts.get(cls_name, cls_name)
        prompt_parts = [p.strip().lower() for p in prompt.split(".")]
        if label_lower in prompt_parts:
            return i

    # Substring match (label is part of class name or vice versa)
    for i, cls_name in enumerate(classes):
        if label_lower in cls_name.lower() or cls_name.lower() in label_lower:
            return i

    return None


def _xyxy_to_yolo(box: np.ndarray, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert xyxy box to YOLO format (cx, cy, w, h) normalized to [0, 1].

    Args:
        box: Array of shape (4,) in xyxy format.
        img_w: Image width.
        img_h: Image height.

    Returns:
        Tuple of (cx, cy, w, h) normalized.
    """
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return cx, cy, w, h


def run(config: dict) -> dict:
    """Run Grounding DINO auto-labeling on all images.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Dictionary with labeling results and statistics.
    """
    label_cfg = config["auto_labeling"]
    classes = config["classes"]
    class_prompts = label_cfg.get("class_prompts", {cls: cls for cls in classes})

    images_dir = Path(config["paths"]["data_dir"]) / "images"
    labels_dir = Path(config["paths"]["labels_dir"]) / "auto"
    labels_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device(label_cfg.get("device", "auto"))
    logger.info("Using device: %s", device)

    # Load Grounding DINO model
    model_id = label_cfg["model_id"]
    logger.info("Loading Grounding DINO model: %s", model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    # Build text prompt - concatenate all class prompts with " . " separator
    # Grounding DINO uses " . " to separate object categories
    text_prompt = " . ".join(class_prompts.values()) + " ."
    logger.info("Text prompt: %s", text_prompt)

    # Collect all images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted(
        [f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions]
    )

    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    logger.info("Processing %d images with Grounding DINO", len(image_files))

    # Statistics
    stats = {
        "total_images": len(image_files),
        "images_with_detections": 0,
        "total_detections": 0,
        "detections_per_class": {cls: 0 for cls in classes},
        "unmatched_labels": {},
        "confidence_scores": [],
    }

    # Detailed detection metadata (for VLM review stage)
    all_detections = {}

    box_threshold = label_cfg.get("box_threshold", 0.30)
    text_threshold = label_cfg.get("text_threshold", 0.25)
    nms_threshold = label_cfg.get("nms_threshold", 0.45)
    min_area = label_cfg.get("min_box_area_ratio", 0.0005)
    max_area = label_cfg.get("max_box_area_ratio", 0.95)

    for img_path in tqdm(image_files, desc="Auto-labeling"):
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        # Run Grounding DINO inference
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process results
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(img_h, img_w)],
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        raw_labels = results["labels"]

        # Apply NMS
        if len(boxes) > 0:
            boxes, scores, raw_labels = _apply_nms(boxes, scores, raw_labels, nms_threshold)

        # Filter by box size
        if len(boxes) > 0:
            boxes, scores, raw_labels = _filter_boxes(
                boxes, scores, raw_labels, img_w, img_h, min_area, max_area
            )

        # Match labels to pipeline classes and write YOLO format labels
        label_file = labels_dir / (img_path.stem + ".txt")
        image_detections = []
        yolo_lines = []

        for box, score, raw_label in zip(boxes, scores, raw_labels):
            class_idx = _match_label_to_class(raw_label, classes, class_prompts)

            if class_idx is None:
                stats["unmatched_labels"][raw_label] = (
                    stats["unmatched_labels"].get(raw_label, 0) + 1
                )
                continue

            cx, cy, w, h = _xyxy_to_yolo(box, img_w, img_h)
            yolo_lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            detection = {
                "class_idx": class_idx,
                "class_name": classes[class_idx],
                "confidence": float(score),
                "bbox_xyxy": box.tolist(),
                "bbox_yolo": [cx, cy, w, h],
                "raw_label": raw_label,
            }
            image_detections.append(detection)

            stats["detections_per_class"][classes[class_idx]] += 1
            stats["total_detections"] += 1
            stats["confidence_scores"].append(float(score))

        # Write label file
        with open(label_file, "w") as f:
            f.write("\n".join(yolo_lines))

        if yolo_lines:
            stats["images_with_detections"] += 1

        all_detections[img_path.name] = image_detections

    # Save detection metadata for downstream stages
    metadata_file = labels_dir / "detection_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(all_detections, f, indent=2)

    # Log summary statistics
    logger.info("Auto-labeling complete:")
    logger.info("  Total images: %d", stats["total_images"])
    logger.info("  Images with detections: %d", stats["images_with_detections"])
    logger.info("  Total detections: %d", stats["total_detections"])
    for cls, count in stats["detections_per_class"].items():
        logger.info("    %s: %d", cls, count)
    if stats["unmatched_labels"]:
        logger.warning("  Unmatched labels: %s", stats["unmatched_labels"])
    if stats["confidence_scores"]:
        scores_arr = np.array(stats["confidence_scores"])
        logger.info(
            "  Confidence: mean=%.3f, median=%.3f, min=%.3f, max=%.3f",
            scores_arr.mean(),
            np.median(scores_arr),
            scores_arr.min(),
            scores_arr.max(),
        )

    # Remove non-serializable numpy data from stats for return
    stats["confidence_scores"] = {
        "mean": float(scores_arr.mean()) if stats["confidence_scores"] else 0,
        "median": float(np.median(scores_arr)) if stats["confidence_scores"] else 0,
        "min": float(scores_arr.min()) if stats["confidence_scores"] else 0,
        "max": float(scores_arr.max()) if stats["confidence_scores"] else 0,
    }

    return {
        "labels_dir": str(labels_dir),
        "metadata_file": str(metadata_file),
        "stats": stats,
    }
