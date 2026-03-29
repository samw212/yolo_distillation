"""Stage 6: Comprehensive Model Evaluation.

Evaluates the trained YOLOv11 model following industry (computer vision)
best practices. Generates detailed metrics, visualizations, and reports.

Metrics:
- mAP@50, mAP@50:95 (COCO-style)
- Precision, Recall, F1 (per-class and overall)
- Confusion matrix
- PR curves, F1 curves
- Detection speed (FPS)
- Label distribution analysis
- Confidence calibration analysis

Optionally compares auto-labels vs ground truth to assess pseudo-label quality.
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from ultralytics import YOLO

matplotlib.use("Agg")  # Non-interactive backend

logger = logging.getLogger("distill.evaluation")


def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _load_yolo_labels(label_dir: Path, classes: list[str]) -> dict:
    """Load YOLO format labels from a directory.

    Returns:
        Dict mapping image stems to list of (class_idx, cx, cy, w, h) tuples.
    """
    labels = {}
    for label_file in label_dir.glob("*.txt"):
        stem = label_file.stem
        detections = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_idx = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    detections.append((cls_idx, cx, cy, w, h))
        labels[stem] = detections
    return labels


def _analyze_label_distribution(labels: dict, classes: list[str]) -> dict:
    """Analyze class distribution in labels."""
    class_counts = Counter()
    boxes_per_image = []

    for stem, dets in labels.items():
        boxes_per_image.append(len(dets))
        for cls_idx, *_ in dets:
            if 0 <= cls_idx < len(classes):
                class_counts[classes[cls_idx]] += 1

    return {
        "class_counts": dict(class_counts),
        "total_boxes": sum(class_counts.values()),
        "total_images": len(labels),
        "avg_boxes_per_image": np.mean(boxes_per_image) if boxes_per_image else 0,
        "boxes_per_image_std": np.std(boxes_per_image) if boxes_per_image else 0,
    }


def _plot_confusion_matrix(confusion: np.ndarray, classes: list[str], save_path: Path):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalize
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_sums, where=row_sums != 0, out=np.zeros_like(confusion, dtype=float))

    display_labels = classes + ["background"]
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_pr_curves(
    per_class_data: dict, classes: list[str], save_dir: Path
):
    """Plot Precision-Recall curves."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for cls_name in classes:
        data = per_class_data.get(cls_name, {})
        precision_list = data.get("precision_at_thresholds", [])
        recall_list = data.get("recall_at_thresholds", [])
        ap = data.get("ap50", 0)

        if precision_list and recall_list:
            ax.plot(recall_list, precision_list, label=f"{cls_name} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "pr_curves.png", dpi=150)
    plt.close()


def _plot_class_distribution(dist_data: dict, save_path: Path):
    """Plot class distribution bar chart."""
    counts = dist_data.get("class_counts", {})
    if not counts:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    classes = list(counts.keys())
    values = list(counts.values())

    bars = ax.bar(classes, values, color=sns.color_palette("husl", len(classes)))
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Label Distribution by Class")
    plt.xticks(rotation=45, ha="right")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_confidence_distribution(predictions: list[dict], save_path: Path):
    """Plot confidence score distribution."""
    if not predictions:
        return

    all_scores = []
    for pred in predictions:
        all_scores.extend(pred.get("confidences", []))

    if not all_scores:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_scores, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence Distribution")
    ax.axvline(np.mean(all_scores), color="red", linestyle="--", label=f"Mean: {np.mean(all_scores):.3f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_box_size_distribution(labels: dict, classes: list[str], save_path: Path):
    """Plot bounding box size distribution."""
    widths = []
    heights = []
    areas = []

    for stem, dets in labels.items():
        for cls_idx, cx, cy, w, h in dets:
            widths.append(w)
            heights.append(h)
            areas.append(w * h)

    if not areas:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(widths, bins=50, alpha=0.7, color="steelblue")
    axes[0].set_xlabel("Width (normalized)")
    axes[0].set_title("Box Width Distribution")

    axes[1].hist(heights, bins=50, alpha=0.7, color="coral")
    axes[1].set_xlabel("Height (normalized)")
    axes[1].set_title("Box Height Distribution")

    axes[2].hist(areas, bins=50, alpha=0.7, color="forestgreen")
    axes[2].set_xlabel("Area (normalized)")
    axes[2].set_title("Box Area Distribution")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _visualize_predictions(
    model: YOLO, test_images_dir: Path, save_dir: Path,
    num_samples: int, conf_threshold: float
):
    """Run inference on sample images and save annotated results."""
    save_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        f for f in sorted(test_images_dir.iterdir())
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        return

    # Sample images
    sample_files = image_files[:num_samples]

    predictions = []
    for img_path in sample_files:
        results = model.predict(str(img_path), conf=conf_threshold, verbose=False)

        if results and len(results) > 0:
            result = results[0]
            annotated = result.plot()
            save_path = save_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(save_path), annotated)

            # Collect prediction data
            if result.boxes is not None and len(result.boxes) > 0:
                predictions.append({
                    "image": img_path.name,
                    "confidences": result.boxes.conf.cpu().numpy().tolist(),
                    "classes": result.boxes.cls.cpu().numpy().tolist(),
                })

    return predictions


def _generate_html_report(
    eval_results: dict, eval_dir: Path, classes: list[str]
) -> str:
    """Generate an HTML evaluation report."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>YOLOv11 Distillation - Evaluation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: center; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .metric-card { display: inline-block; background: #e8f5e9; padding: 15px 25px; margin: 5px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2e7d32; }
        .metric-label { font-size: 12px; color: #666; }
        img { max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }
        .img-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px; }
    </style>
</head>
<body>
<div class="container">
    <h1>YOLOv11 Distillation Pipeline - Evaluation Report</h1>
    <p>Model distilled from Grounding DINO using DART methodology</p>
"""

    # Overall metrics
    metrics = eval_results.get("metrics", {})
    html += "<h2>Overall Metrics</h2><div>"
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            html += f'<div class="metric-card"><div class="metric-value">{value:.4f}</div><div class="metric-label">{name}</div></div>'
    html += "</div>"

    # Per-class metrics table
    per_class = eval_results.get("per_class_metrics", {})
    if per_class:
        html += "<h2>Per-Class Metrics</h2><table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>AP@50</th><th>AP@50:95</th><th>Count</th></tr>"
        for cls_name in classes:
            cls_data = per_class.get(cls_name, {})
            html += f"<tr><td>{cls_name}</td>"
            for metric in ["precision", "recall", "f1", "ap50", "ap50_95"]:
                val = cls_data.get(metric, 0)
                html += f"<td>{val:.4f}</td>"
            html += f"<td>{cls_data.get('count', 0)}</td></tr>"
        html += "</table>"

    # Visualizations
    html += "<h2>Visualizations</h2>"

    viz_files = {
        "confusion_matrix.png": "Confusion Matrix",
        "pr_curves.png": "Precision-Recall Curves",
        "class_distribution.png": "Class Distribution",
        "confidence_distribution.png": "Confidence Distribution",
        "box_size_distribution.png": "Box Size Distribution",
    }

    for filename, title in viz_files.items():
        if (eval_dir / filename).exists():
            html += f"<h3>{title}</h3><img src='{filename}' alt='{title}'>"

    # Sample predictions
    samples_dir = eval_dir / "prediction_samples"
    if samples_dir.exists():
        sample_images = sorted(samples_dir.glob("*.jpg")) + sorted(samples_dir.glob("*.png"))
        if sample_images:
            html += "<h2>Sample Predictions</h2><div class='img-grid'>"
            for img in sample_images[:12]:
                html += f"<img src='prediction_samples/{img.name}' alt='{img.name}'>"
            html += "</div>"

    # Label quality analysis
    label_quality = eval_results.get("label_quality", {})
    if label_quality:
        html += "<h2>Label Quality Analysis</h2>"
        html += "<p>Comparison between auto-generated pseudo-labels and model predictions</p>"
        html += "<table><tr><th>Metric</th><th>Value</th></tr>"
        for k, v in label_quality.items():
            html += f"<tr><td>{k}</td><td>{v}</td></tr>"
        html += "</table>"

    html += """
</div>
</body>
</html>"""

    report_path = eval_dir / "evaluation_report.html"
    with open(report_path, "w") as f:
        f.write(html)

    return str(report_path)


def run(config: dict) -> dict:
    """Run comprehensive model evaluation.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Dictionary with evaluation results.
    """
    eval_cfg = config["evaluation"]
    classes = config["classes"]
    training_dir = Path(config["paths"]["training_dir"])
    eval_dir = Path(config["paths"]["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Find best model
    best_model_path = training_dir / "runs" / "distill_train" / "weights" / "best.pt"
    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Best model not found at {best_model_path}. Run training first."
        )

    logger.info("Loading best model: %s", best_model_path)
    model = YOLO(str(best_model_path))

    # Find dataset yaml
    dataset_yaml_path = training_dir / "dataset" / "dataset.yaml"
    if not dataset_yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml_path}")

    with open(dataset_yaml_path) as f:
        dataset_info = yaml.safe_load(f)

    conf_threshold = eval_cfg.get("conf_threshold", 0.25)
    iou_threshold = eval_cfg.get("iou_threshold", 0.5)

    # ---- Run Ultralytics validation on test set ----
    logger.info("Running model validation on test set...")
    val_results = model.val(
        data=str(dataset_yaml_path),
        split="test",
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=True,
    )

    # Extract metrics from validation results
    eval_results = {"metrics": {}, "per_class_metrics": {}}

    if val_results:
        box = val_results.box
        eval_results["metrics"] = {
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
            "precision": float(box.mp),
            "recall": float(box.mr),
            "f1": float(2 * box.mp * box.mr / (box.mp + box.mr)) if (box.mp + box.mr) > 0 else 0.0,
        }

        # Per-class metrics
        if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
            for i, cls_idx in enumerate(box.ap_class_index):
                cls_name = classes[int(cls_idx)] if int(cls_idx) < len(classes) else f"class_{cls_idx}"
                p = float(box.p[i]) if i < len(box.p) else 0
                r = float(box.r[i]) if i < len(box.r) else 0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

                eval_results["per_class_metrics"][cls_name] = {
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                    "ap50": float(box.ap50[i]) if i < len(box.ap50) else 0,
                    "ap50_95": float(box.ap[i]) if i < len(box.ap) else 0,
                }

    # ---- Visualizations ----
    viz_cfg = eval_cfg.get("visualizations", {})
    logger.info("Generating visualizations...")

    # Label distribution analysis
    test_labels_dir = Path(dataset_info["path"]) / "test" / "labels"
    if test_labels_dir.exists():
        test_labels = _load_yolo_labels(test_labels_dir, classes)
        label_dist = _analyze_label_distribution(test_labels, classes)
        eval_results["label_distribution"] = label_dist

        if viz_cfg.get("label_distribution", True):
            _plot_class_distribution(label_dist, eval_dir / "class_distribution.png")

        if viz_cfg.get("box_size_distribution", True):
            _plot_box_size_distribution(test_labels, classes, eval_dir / "box_size_distribution.png")

    # Analyze full training set distribution
    train_labels_dir = Path(dataset_info["path"]) / "train" / "labels"
    if train_labels_dir.exists():
        train_labels = _load_yolo_labels(train_labels_dir, classes)
        train_dist = _analyze_label_distribution(train_labels, classes)
        eval_results["train_label_distribution"] = train_dist

    # Sample predictions
    test_images_dir = Path(dataset_info["path"]) / "test" / "images"
    if viz_cfg.get("prediction_samples", True) and test_images_dir.exists():
        num_samples = eval_cfg.get("num_samples", 50)
        predictions = _visualize_predictions(
            model, test_images_dir, eval_dir / "prediction_samples",
            num_samples, conf_threshold
        )
        if predictions:
            _plot_confidence_distribution(predictions, eval_dir / "confidence_distribution.png")

    # ---- Copy Ultralytics-generated plots ----
    # Ultralytics generates confusion matrix, PR curves, etc. in the run dir
    run_dir = training_dir / "runs" / "distill_train"
    plots_to_copy = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "results.png",
    ]
    for plot_name in plots_to_copy:
        src = run_dir / plot_name
        if src.exists():
            import shutil
            shutil.copy2(src, eval_dir / plot_name)

    # ---- Ground truth comparison (if available) ----
    gt_labels_dir = Path(config["paths"]["data_dir"]) / "gt_labels"
    if eval_cfg.get("compare_with_gt", True) and gt_labels_dir.exists():
        logger.info("Comparing pseudo-labels with ground truth...")
        eval_results["label_quality"] = _compare_with_gt(
            config, gt_labels_dir, classes
        )

    # ---- Generate report ----
    report_format = eval_cfg.get("report_format", "html")
    if report_format in ("html", "both"):
        report_path = _generate_html_report(eval_results, eval_dir, classes)
        eval_results["html_report"] = report_path
        logger.info("HTML report generated: %s", report_path)

    if report_format in ("json", "both"):
        json_path = eval_dir / "evaluation_results.json"
        with open(json_path, "w") as f:
            json.dump(eval_results, f, indent=2, default=str)
        eval_results["json_report"] = str(json_path)

    # Log summary
    logger.info("Evaluation Summary:")
    for metric, value in eval_results.get("metrics", {}).items():
        logger.info("  %s: %.4f", metric, value)

    return eval_results


def _compare_with_gt(config: dict, gt_labels_dir: Path, classes: list[str]) -> dict:
    """Compare auto-generated pseudo-labels with ground truth labels.

    This measures pseudo-label quality - how well Grounding DINO's
    annotations match human-created ground truth.
    """
    approved_dir = Path(config["paths"]["labels_dir"]) / "approved"
    if not approved_dir.exists():
        return {}

    gt_labels = _load_yolo_labels(gt_labels_dir, classes)
    auto_labels = _load_yolo_labels(approved_dir, classes)

    # Compare overlapping images
    common_stems = set(gt_labels.keys()) & set(auto_labels.keys())
    if not common_stems:
        return {"note": "No common images between GT and auto-labels"}

    total_gt = 0
    total_auto = 0
    matched = 0

    for stem in common_stems:
        gt_dets = gt_labels[stem]
        auto_dets = auto_labels[stem]
        total_gt += len(gt_dets)
        total_auto += len(auto_dets)

        # Simple matching: for each GT box, find best matching auto box
        used_auto = set()
        for gt_cls, gt_cx, gt_cy, gt_w, gt_h in gt_dets:
            gt_box = np.array([gt_cx - gt_w/2, gt_cy - gt_h/2, gt_cx + gt_w/2, gt_cy + gt_h/2])
            best_iou = 0
            best_idx = -1

            for j, (auto_cls, a_cx, a_cy, a_w, a_h) in enumerate(auto_dets):
                if j in used_auto:
                    continue
                if auto_cls != gt_cls:
                    continue

                auto_box = np.array([a_cx - a_w/2, a_cy - a_h/2, a_cx + a_w/2, a_cy + a_h/2])
                iou = _compute_iou(gt_box, auto_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_iou >= 0.5 and best_idx >= 0:
                matched += 1
                used_auto.add(best_idx)

    recall = matched / total_gt if total_gt > 0 else 0
    precision = matched / total_auto if total_auto > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "common_images": len(common_stems),
        "gt_total_boxes": total_gt,
        "auto_total_boxes": total_auto,
        "matched_boxes_iou50": matched,
        "pseudo_label_precision": round(precision, 4),
        "pseudo_label_recall": round(recall, 4),
        "pseudo_label_f1": round(f1, 4),
    }
