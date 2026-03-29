"""Generate synthetic dataset for pipeline testing.

Creates simple synthetic images with colored shapes representing each
target class, along with YOLO-format ground truth labels. This allows
the full pipeline to be tested end-to-end in environments without
external API access.

Usage:
    python scripts/generate_synthetic_data.py --output output/data --num-images 500
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


# Class definitions with visual representations
CLASSES = {
    0: {"name": "safety_helmet", "color": (0, 255, 255), "shape": "semicircle"},
    1: {"name": "fire", "color": (0, 80, 255), "shape": "triangle"},
    2: {"name": "smoke", "color": (180, 180, 180), "shape": "cloud"},
    3: {"name": "human", "color": (0, 200, 0), "shape": "rectangle_tall"},
    4: {"name": "ladder", "color": (139, 90, 43), "shape": "ladder"},
    5: {"name": "working_platform", "color": (100, 100, 200), "shape": "rectangle_wide"},
}

CLASS_NAMES = [
    "safety helmet", "fire", "smoke", "human", "ladder", "working platform"
]


def draw_semicircle(img, x, y, w, h, color):
    """Draw a helmet-like semicircle."""
    cx, cy = x + w // 2, y + h // 2
    axes = (w // 2, h // 2)
    cv2.ellipse(img, (cx, cy + h // 4), axes, 0, 180, 360, color, -1)
    cv2.rectangle(img, (x + w // 6, cy), (x + w - w // 6, y + h), color, -1)


def draw_triangle(img, x, y, w, h, color):
    """Draw fire-like triangle."""
    pts = np.array([
        [x + w // 2, y],
        [x + w, y + h],
        [x, y + h],
    ], np.int32)
    cv2.fillPoly(img, [pts], color)
    # Add inner yellow
    inner_pts = np.array([
        [x + w // 2, y + h // 3],
        [x + 2 * w // 3, y + h],
        [x + w // 3, y + h],
    ], np.int32)
    cv2.fillPoly(img, [inner_pts], (0, 200, 255))


def draw_cloud(img, x, y, w, h, color):
    """Draw smoke-like cloud shape."""
    for _ in range(5):
        cx = x + random.randint(w // 4, 3 * w // 4)
        cy = y + random.randint(h // 4, 3 * h // 4)
        rx = random.randint(w // 4, w // 2)
        ry = random.randint(h // 4, h // 2)
        alpha = random.randint(100, 200)
        overlay = img.copy()
        cv2.ellipse(overlay, (cx, cy), (rx, ry), 0, 0, 360, color, -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)


def draw_rectangle_tall(img, x, y, w, h, color):
    """Draw human-like tall rectangle."""
    # Body
    cv2.rectangle(img, (x + w // 4, y + h // 6), (x + 3 * w // 4, y + h), color, -1)
    # Head
    cv2.circle(img, (x + w // 2, y + h // 8), h // 8, color, -1)


def draw_ladder(img, x, y, w, h, color):
    """Draw ladder shape."""
    # Side rails
    cv2.rectangle(img, (x, y), (x + w // 6, y + h), color, -1)
    cv2.rectangle(img, (x + 5 * w // 6, y), (x + w, y + h), color, -1)
    # Rungs
    num_rungs = max(3, h // 30)
    for i in range(num_rungs):
        ry = y + (i + 1) * h // (num_rungs + 1)
        cv2.rectangle(img, (x, ry - 2), (x + w, ry + 2), color, -1)


def draw_rectangle_wide(img, x, y, w, h, color):
    """Draw platform-like wide rectangle."""
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    # Add railing lines
    cv2.line(img, (x, y), (x + w, y), (80, 80, 80), 2)
    # Support legs
    cv2.rectangle(img, (x + w // 8, y + h), (x + w // 8 + 5, y + h + h // 2), (80, 80, 80), -1)
    cv2.rectangle(img, (x + 7 * w // 8, y + h), (x + 7 * w // 8 + 5, y + h + h // 2), (80, 80, 80), -1)


DRAW_FUNCTIONS = {
    "semicircle": draw_semicircle,
    "triangle": draw_triangle,
    "cloud": draw_cloud,
    "rectangle_tall": draw_rectangle_tall,
    "ladder": draw_ladder,
    "rectangle_wide": draw_rectangle_wide,
}


def generate_background(h, w):
    """Generate a random background."""
    bg_type = random.choice(["solid", "gradient", "noise"])

    if bg_type == "solid":
        color = [random.randint(100, 240) for _ in range(3)]
        img = np.full((h, w, 3), color, dtype=np.uint8)
    elif bg_type == "gradient":
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            start = random.randint(50, 200)
            end = random.randint(50, 200)
            img[:, :, c] = np.linspace(start, end, h).reshape(-1, 1).astype(np.uint8)
    else:
        img = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (15, 15), 0)

    return img


def generate_image(img_size=(640, 640), min_objects=1, max_objects=6):
    """Generate one synthetic image with random objects.

    Returns:
        Tuple of (image, labels) where labels is list of (class_id, cx, cy, w, h).
    """
    h, w = img_size
    img = generate_background(h, w)
    labels = []

    num_objects = random.randint(min_objects, max_objects)
    placed_boxes = []

    for _ in range(num_objects):
        cls_id = random.choice(list(CLASSES.keys()))
        cls_info = CLASSES[cls_id]

        # Random size based on class
        if cls_info["shape"] == "rectangle_tall":
            obj_w = random.randint(40, 120)
            obj_h = random.randint(80, 200)
        elif cls_info["shape"] == "rectangle_wide":
            obj_w = random.randint(100, 250)
            obj_h = random.randint(30, 80)
        elif cls_info["shape"] == "ladder":
            obj_w = random.randint(30, 80)
            obj_h = random.randint(100, 250)
        elif cls_info["shape"] == "cloud":
            obj_w = random.randint(80, 200)
            obj_h = random.randint(60, 150)
        else:
            obj_w = random.randint(30, 100)
            obj_h = random.randint(30, 100)

        # Random position
        max_x = w - obj_w - 5
        max_y = h - obj_h - 5
        if max_x <= 5 or max_y <= 5:
            continue

        x = random.randint(5, max_x)
        y = random.randint(5, max_y)

        # Check overlap with existing boxes
        new_box = (x, y, x + obj_w, y + obj_h)
        overlap = False
        for box in placed_boxes:
            ix1 = max(new_box[0], box[0])
            iy1 = max(new_box[1], box[1])
            ix2 = min(new_box[2], box[2])
            iy2 = min(new_box[3], box[3])
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                box_area = obj_w * obj_h
                if inter / box_area > 0.3:
                    overlap = True
                    break
        if overlap:
            continue

        # Add noise to color
        color = list(cls_info["color"])
        color = tuple(max(0, min(255, c + random.randint(-30, 30))) for c in color)

        # Draw the object
        draw_fn = DRAW_FUNCTIONS[cls_info["shape"]]
        draw_fn(img, x, y, obj_w, obj_h, color)

        placed_boxes.append(new_box)

        # YOLO format: class_id cx cy w h (normalized)
        cx = (x + obj_w / 2) / w
        cy = (y + obj_h / 2) / h
        nw = obj_w / w
        nh = obj_h / h
        labels.append((cls_id, cx, cy, nw, nh))

    # Add noise and blur for realism
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img, labels


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset")
    parser.add_argument("--output", default="output/data", help="Output directory")
    parser.add_argument("--num-images", type=int, default=500, help="Number of images")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "gt_labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_images} synthetic images...")

    total_objects = 0
    class_counts = {cls_id: 0 for cls_id in CLASSES}

    for i in range(args.num_images):
        img, labels = generate_image(img_size=(args.img_size, args.img_size))

        # Save image
        img_name = f"synth_{i:05d}.jpg"
        cv2.imwrite(str(images_dir / img_name), img)

        # Save label
        label_name = f"synth_{i:05d}.txt"
        with open(labels_dir / label_name, "w") as f:
            for cls_id, cx, cy, w, h in labels:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                class_counts[cls_id] += 1
                total_objects += 1

    # Save classes.txt
    with open(output_dir / "classes.txt", "w") as f:
        for name in CLASS_NAMES:
            f.write(f"{name}\n")

    print(f"\nGenerated {args.num_images} images with {total_objects} objects")
    print("Class distribution:")
    for cls_id, count in class_counts.items():
        print(f"  {CLASS_NAMES[cls_id]}: {count}")
    print(f"\nImages: {images_dir}")
    print(f"Labels: {labels_dir}")


if __name__ == "__main__":
    main()
