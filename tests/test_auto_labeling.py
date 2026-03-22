"""Tests for auto-labeling utilities (unit tests that don't require GPU)."""

import numpy as np
import pytest

from src.pipeline.stages.auto_labeling import (
    _filter_boxes,
    _match_label_to_class,
    _xyxy_to_yolo,
)


class TestXYXYToYOLO:
    """Test bounding box format conversion."""

    def test_basic_conversion(self):
        box = np.array([100, 100, 200, 200])
        cx, cy, w, h = _xyxy_to_yolo(box, 640, 480)
        assert abs(cx - 150 / 640) < 1e-6
        assert abs(cy - 150 / 480) < 1e-6
        assert abs(w - 100 / 640) < 1e-6
        assert abs(h - 100 / 480) < 1e-6

    def test_full_image_box(self):
        box = np.array([0, 0, 640, 480])
        cx, cy, w, h = _xyxy_to_yolo(box, 640, 480)
        assert abs(cx - 0.5) < 1e-6
        assert abs(cy - 0.5) < 1e-6
        assert abs(w - 1.0) < 1e-6
        assert abs(h - 1.0) < 1e-6

    def test_small_box(self):
        box = np.array([10, 10, 20, 20])
        cx, cy, w, h = _xyxy_to_yolo(box, 640, 480)
        assert 0 <= cx <= 1
        assert 0 <= cy <= 1
        assert 0 <= w <= 1
        assert 0 <= h <= 1


class TestMatchLabelToClass:
    """Test label matching logic."""

    CLASSES = ["safety helmet", "fire", "smoke", "human", "ladder", "working platform"]
    PROMPTS = {
        "safety helmet": "safety helmet . hard hat . construction helmet",
        "fire": "fire . flames . burning",
        "smoke": "smoke . fumes . haze",
        "human": "person . human . worker . man . woman",
        "ladder": "ladder . step ladder . extension ladder",
        "working platform": "working platform . scaffolding . elevated platform",
    }

    def test_exact_match(self):
        assert _match_label_to_class("fire", self.CLASSES, self.PROMPTS) == 1
        assert _match_label_to_class("smoke", self.CLASSES, self.PROMPTS) == 2

    def test_prompt_match(self):
        assert _match_label_to_class("hard hat", self.CLASSES, self.PROMPTS) == 0
        assert _match_label_to_class("person", self.CLASSES, self.PROMPTS) == 3
        assert _match_label_to_class("worker", self.CLASSES, self.PROMPTS) == 3

    def test_case_insensitive(self):
        assert _match_label_to_class("Fire", self.CLASSES, self.PROMPTS) == 1
        assert _match_label_to_class("SMOKE", self.CLASSES, self.PROMPTS) == 2

    def test_no_match(self):
        assert _match_label_to_class("car", self.CLASSES, self.PROMPTS) is None
        assert _match_label_to_class("tree", self.CLASSES, self.PROMPTS) is None

    def test_substring_match(self):
        assert _match_label_to_class("safety", self.CLASSES, self.PROMPTS) == 0


class TestFilterBoxes:
    """Test box size filtering."""

    def test_filter_tiny_boxes(self):
        boxes = np.array([[0, 0, 1, 1], [100, 100, 300, 300]])
        scores = np.array([0.9, 0.8])
        labels = ["a", "b"]

        filtered_boxes, filtered_scores, filtered_labels = _filter_boxes(
            boxes, scores, labels, 640, 480, min_area_ratio=0.001, max_area_ratio=0.95
        )

        # First box is 1x1 = 1 pixel, area_ratio = 1/(640*480) < 0.001
        assert len(filtered_boxes) == 1
        assert filtered_labels == ["b"]

    def test_filter_huge_boxes(self):
        boxes = np.array([[0, 0, 630, 470], [100, 100, 200, 200]])
        scores = np.array([0.9, 0.8])
        labels = ["a", "b"]

        filtered_boxes, filtered_scores, filtered_labels = _filter_boxes(
            boxes, scores, labels, 640, 480, min_area_ratio=0.001, max_area_ratio=0.90
        )

        # First box covers ~96% of image > 0.90
        assert len(filtered_boxes) == 1
        assert filtered_labels == ["b"]

    def test_empty_boxes(self):
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        labels = []

        result = _filter_boxes(boxes, scores, labels, 640, 480, 0.001, 0.95)
        assert len(result[0]) == 0
