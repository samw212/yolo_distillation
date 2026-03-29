"""Stage 4: Human Labeling Export and Import.

Exports images that failed VLM review to CVAT XML format for human
annotation. After manual correction, imports the corrected labels
back into the pipeline.

The CVAT XML format is widely supported by annotation tools and
provides a standard interchange format.
"""

import json
import logging
import shutil
from pathlib import Path

from lxml import etree

logger = logging.getLogger("distill.human_labeling")


def _create_cvat_xml(
    images: list[dict],
    classes: list[str],
    task_name: str,
    include_pre_annotations: bool,
    include_rejection_reasons: bool,
) -> etree.Element:
    """Create CVAT XML annotation file.

    Args:
        images: List of image dicts with detection info.
        classes: List of class names.
        task_name: Name of the CVAT task.
        include_pre_annotations: Whether to include auto-generated labels.
        include_rejection_reasons: Whether to include VLM rejection reasons.

    Returns:
        XML Element tree root.
    """
    root = etree.Element("annotations")

    # Version info
    version = etree.SubElement(root, "version")
    version.text = "1.1"

    # Meta
    meta = etree.SubElement(root, "meta")
    task = etree.SubElement(meta, "task")
    name_elem = etree.SubElement(task, "name")
    name_elem.text = task_name
    size_elem = etree.SubElement(task, "size")
    size_elem.text = str(len(images))

    # Labels definition
    labels_elem = etree.SubElement(task, "labels")
    for cls_name in classes:
        label = etree.SubElement(labels_elem, "label")
        label_name = etree.SubElement(label, "name")
        label_name.text = cls_name
        label_type = etree.SubElement(label, "type")
        label_type.text = "rectangle"

    # Image annotations
    for idx, img_data in enumerate(images):
        image_elem = etree.SubElement(root, "image")
        image_elem.set("id", str(idx))
        image_elem.set("name", img_data["image"])
        image_elem.set("width", str(img_data.get("width", 0)))
        image_elem.set("height", str(img_data.get("height", 0)))

        if include_pre_annotations:
            # Include all detections (both approved and flagged) as pre-annotations
            all_dets = img_data.get("approved", []) + img_data.get("flagged", [])
            for det in all_dets:
                box = etree.SubElement(image_elem, "box")
                box.set("label", det["class_name"])

                x1, y1, x2, y2 = det["bbox_xyxy"]
                box.set("xtl", f"{x1:.2f}")
                box.set("ytl", f"{y1:.2f}")
                box.set("xbr", f"{x2:.2f}")
                box.set("ybr", f"{y2:.2f}")
                box.set("occluded", "0")

                # Add review status as attribute
                status = det.get("review_status", "unknown")
                attr = etree.SubElement(box, "attribute")
                attr.set("name", "review_status")
                attr.text = status

                if include_rejection_reasons and status == "vlm_rejected":
                    reason_attr = etree.SubElement(box, "attribute")
                    reason_attr.set("name", "rejection_reason")
                    reason_attr.text = det.get("review_reason", "")

                # Add confidence
                conf_attr = etree.SubElement(box, "attribute")
                conf_attr.set("name", "confidence")
                conf_attr.text = f"{det.get('confidence', 0):.3f}"

    return root


def export_for_review(config: dict) -> dict:
    """Export flagged images to CVAT format for human annotation.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Dictionary with export results.
    """
    human_cfg = config["human_labeling"]
    classes = config["classes"]
    cvat_cfg = human_cfg.get("cvat", {})

    flagged_dir = Path(config["paths"]["labels_dir"]) / "flagged"
    images_dir = Path(config["paths"]["data_dir"]) / "images"
    export_dir = Path(config["paths"]["export_dir"])
    export_images_dir = export_dir / "images"

    export_dir.mkdir(parents=True, exist_ok=True)
    export_images_dir.mkdir(parents=True, exist_ok=True)

    # Collect all flagged image data
    flagged_files = sorted(flagged_dir.glob("*.json"))

    if not flagged_files:
        logger.info("No flagged images to export for human review")
        return {"exported_count": 0, "export_dir": str(export_dir)}

    images_data = []
    for flagged_file in flagged_files:
        with open(flagged_file) as f:
            data = json.load(f)

        img_name = data["image"]
        img_path = images_dir / img_name

        if img_path.exists():
            # Get image dimensions
            import cv2
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                data["width"] = w
                data["height"] = h

                # Copy image to export directory
                shutil.copy2(img_path, export_images_dir / img_name)
                images_data.append(data)

    # Create CVAT XML
    xml_root = _create_cvat_xml(
        images=images_data,
        classes=classes,
        task_name=cvat_cfg.get("task_name", "yolo_distillation_review"),
        include_pre_annotations=cvat_cfg.get("include_pre_annotations", True),
        include_rejection_reasons=cvat_cfg.get("include_rejection_reasons", True),
    )

    xml_file = export_dir / "annotations.xml"
    tree = etree.ElementTree(xml_root)
    tree.write(str(xml_file), pretty_print=True, xml_declaration=True, encoding="utf-8")

    logger.info(
        "Exported %d flagged images to CVAT format: %s",
        len(images_data),
        xml_file,
    )

    return {
        "exported_count": len(images_data),
        "export_dir": str(export_dir),
        "cvat_xml": str(xml_file),
    }


def import_corrected_labels(config: dict) -> dict:
    """Import human-corrected labels back into the pipeline.

    Supports CVAT XML, YOLO TXT, and COCO JSON import formats.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Dictionary with import results.
    """
    human_cfg = config["human_labeling"]
    classes = config["classes"]

    import_dir = Path(human_cfg.get("import_dir", "output/human_corrected"))
    import_format = human_cfg.get("import_format", "cvat_xml")
    approved_dir = Path(config["paths"]["labels_dir"]) / "approved"
    approved_dir.mkdir(parents=True, exist_ok=True)

    if not import_dir.exists():
        logger.warning(
            "Human-corrected labels directory not found: %s. "
            "Skipping import. Place corrected labels here and re-run.",
            import_dir,
        )
        return {"imported_count": 0}

    imported_count = 0

    if import_format == "cvat_xml":
        imported_count = _import_cvat_xml(import_dir, approved_dir, classes)
    elif import_format == "yolo_txt":
        imported_count = _import_yolo_txt(import_dir, approved_dir)
    elif import_format == "coco_json":
        imported_count = _import_coco_json(import_dir, approved_dir, classes)
    else:
        raise ValueError(f"Unsupported import format: {import_format}")

    logger.info("Imported %d corrected label files", imported_count)

    return {"imported_count": imported_count, "import_dir": str(import_dir)}


def _import_cvat_xml(import_dir: Path, approved_dir: Path, classes: list[str]) -> int:
    """Import labels from CVAT XML format."""
    xml_files = list(import_dir.glob("*.xml"))
    if not xml_files:
        logger.warning("No CVAT XML files found in %s", import_dir)
        return 0

    class_to_idx = {cls.lower(): i for i, cls in enumerate(classes)}
    imported = 0

    for xml_file in xml_files:
        tree = etree.parse(str(xml_file))
        root = tree.getroot()

        for image_elem in root.findall(".//image"):
            img_name = image_elem.get("name", "")
            img_w = int(image_elem.get("width", 0))
            img_h = int(image_elem.get("height", 0))

            if img_w == 0 or img_h == 0:
                continue

            lines = []
            for box in image_elem.findall("box"):
                label = box.get("label", "").lower()
                class_idx = class_to_idx.get(label)
                if class_idx is None:
                    continue

                x1 = float(box.get("xtl", 0))
                y1 = float(box.get("ytl", 0))
                x2 = float(box.get("xbr", 0))
                y2 = float(box.get("ybr", 0))

                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if lines:
                stem = Path(img_name).stem
                label_file = approved_dir / f"{stem}.txt"
                with open(label_file, "w") as f:
                    f.write("\n".join(lines))
                imported += 1

    return imported


def _import_yolo_txt(import_dir: Path, approved_dir: Path) -> int:
    """Import labels from YOLO TXT format (direct copy)."""
    imported = 0
    for txt_file in import_dir.glob("*.txt"):
        if txt_file.name == "classes.txt":
            continue
        shutil.copy2(txt_file, approved_dir / txt_file.name)
        imported += 1
    return imported


def _import_coco_json(import_dir: Path, approved_dir: Path, classes: list[str]) -> int:
    """Import labels from COCO JSON format."""
    json_files = list(import_dir.glob("*.json"))
    if not json_files:
        return 0

    imported = 0
    for json_file in json_files:
        with open(json_file) as f:
            coco_data = json.load(f)

        # Build category mapping
        coco_cat_to_idx = {}
        for cat in coco_data.get("categories", []):
            cat_name = cat["name"].lower()
            for i, cls in enumerate(classes):
                if cat_name == cls.lower():
                    coco_cat_to_idx[cat["id"]] = i
                    break

        # Build image lookup
        image_info = {img["id"]: img for img in coco_data.get("images", [])}

        # Group annotations by image
        from collections import defaultdict
        img_annotations = defaultdict(list)
        for ann in coco_data.get("annotations", []):
            img_annotations[ann["image_id"]].append(ann)

        for img_id, anns in img_annotations.items():
            img = image_info.get(img_id)
            if not img:
                continue

            img_w = img["width"]
            img_h = img["height"]

            lines = []
            for ann in anns:
                class_idx = coco_cat_to_idx.get(ann["category_id"])
                if class_idx is None:
                    continue

                x, y, w, h = ann["bbox"]  # COCO format: x, y, w, h (absolute)
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if lines:
                stem = Path(img.get("file_name", "")).stem
                if stem:
                    label_file = approved_dir / f"{stem}.txt"
                    with open(label_file, "w") as f:
                        f.write("\n".join(lines))
                    imported += 1

    return imported


def run(config: dict) -> dict:
    """Run the human labeling stage: export flagged images and import corrections.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Combined export and import results.
    """
    # Export flagged images for review
    export_result = export_for_review(config)

    # Try to import any existing corrections
    import_result = import_corrected_labels(config)

    return {
        "export": export_result,
        "import": import_result,
    }
