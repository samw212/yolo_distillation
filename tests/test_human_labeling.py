"""Tests for human labeling export/import utilities."""

import json
import tempfile
from pathlib import Path

import pytest
from lxml import etree

from src.pipeline.stages.human_labeling import _create_cvat_xml


def test_create_cvat_xml():
    """Test CVAT XML creation."""
    images = [
        {
            "image": "img001.jpg",
            "width": 640,
            "height": 480,
            "approved": [
                {
                    "class_name": "fire",
                    "bbox_xyxy": [100, 100, 200, 200],
                    "review_status": "auto_approved",
                    "confidence": 0.85,
                }
            ],
            "flagged": [
                {
                    "class_name": "smoke",
                    "bbox_xyxy": [300, 300, 400, 400],
                    "review_status": "vlm_rejected",
                    "review_reason": "Looks like fog, not smoke",
                    "confidence": 0.35,
                }
            ],
        }
    ]

    classes = ["safety helmet", "fire", "smoke", "human", "ladder", "working platform"]

    root = _create_cvat_xml(
        images=images,
        classes=classes,
        task_name="test_task",
        include_pre_annotations=True,
        include_rejection_reasons=True,
    )

    # Verify XML structure
    assert root.tag == "annotations"
    assert root.find(".//version").text == "1.1"
    assert root.find(".//task/name").text == "test_task"

    # Check labels
    label_names = [l.find("name").text for l in root.findall(".//labels/label")]
    assert "fire" in label_names
    assert "smoke" in label_names

    # Check image annotations
    image_elems = root.findall(".//image")
    assert len(image_elems) == 1
    assert image_elems[0].get("name") == "img001.jpg"

    # Check boxes
    boxes = image_elems[0].findall("box")
    assert len(boxes) == 2

    # Check rejection reason attribute
    rejected_box = [b for b in boxes if b.get("label") == "smoke"][0]
    reason_attr = [a for a in rejected_box.findall("attribute") if a.get("name") == "rejection_reason"]
    assert len(reason_attr) == 1
    assert "fog" in reason_attr[0].text.lower()
