# YOLOv11 Distillation Pipeline

Knowledge distillation from [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) to [YOLOv11](https://docs.ultralytics.com/models/yolo11/), based on the [DART](https://arxiv.org/abs/2407.09174) and [Auto-Labeling](https://arxiv.org/abs/2506.02359) methodologies.

## Overview

This pipeline transfers the open-vocabulary detection capability of Grounding DINO (a large vision-language model) into a lightweight YOLOv11 model through pseudo-label training. The result is a fast, deployable detector trained without manual annotation.

**Target Classes:** Safety Helmet, Fire, Smoke, Human, Ladder, Working Platform

## Pipeline Stages

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  1. Data Pull    │───▶│  2. Auto-Label   │───▶│  3. VLM Review   │
│  (Roboflow)      │    │  (Grounding DINO)│    │  (Qwen2.5-VL)    │
└──────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                         │
                                              ┌──────────┴─────────┐
                                              ▼                    ▼
                                     ┌─────────────┐    ┌──────────────────┐
                                     │  Approved    │    │  4. Human Label  │
                                     │  Labels      │    │  (CVAT Export)   │
                                     └──────┬──────┘    └────────┬─────────┘
                                            │                    │
                                            ▼                    ▼
                                     ┌──────────────────────────────────┐
                                     │  5. Train YOLOv11               │
                                     │  (Fine-tune on approved labels) │
                                     └──────────────┬───────────────────┘
                                                    │
                                                    ▼
                                     ┌──────────────────────────────────┐
                                     │  6. Evaluate                    │
                                     │  (mAP, PR curves, report)       │
                                     └──────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export ROBOFLOW_API_KEY=your_api_key_here
```

### 3. Configure

Edit `configs/default.yaml` with your Roboflow workspace/project details and any parameter adjustments.

### 4. Run the Full Pipeline

```bash
python -m src run \
    --workspace your-workspace \
    --project your-project \
    --version 1
```

### 5. Run Individual Stages

```bash
# Pull data from Roboflow
python -m src pull-data --workspace my-ws --project my-proj --version 1

# Auto-label with Grounding DINO
python -m src auto-label

# VLM review (optional)
python -m src vlm-review

# Export flagged images for human annotation
python -m src export-labels

# Import corrected labels
python -m src import-labels --import-dir output/human_corrected

# Train YOLOv11
python -m src train --epochs 100 --batch-size 16

# Evaluate
python -m src evaluate

# Check pipeline status
python -m src status
```

## Project Structure

```
yolo_distillation/
├── configs/
│   └── default.yaml          # Pipeline configuration
├── src/
│   ├── __init__.py
│   ├── __main__.py            # Entry point
│   ├── cli.py                 # CLI interface
│   └── pipeline/
│       ├── __init__.py
│       ├── orchestrator.py    # Stage coordination
│       ├── stages/
│       │   ├── data_acquisition.py   # Stage 1: Roboflow data pull
│       │   ├── auto_labeling.py      # Stage 2: Grounding DINO
│       │   ├── vlm_review.py         # Stage 3: Qwen2.5-VL review
│       │   ├── human_labeling.py     # Stage 4: CVAT export/import
│       │   ├── training.py           # Stage 5: YOLOv11 fine-tuning
│       │   └── evaluation.py         # Stage 6: Metrics & reports
│       └── utils/
│           ├── config.py             # Configuration management
│           ├── logging.py            # Logging setup
│           └── markers.py            # Stage completion tracking
├── tests/
│   ├── test_config.py
│   ├── test_auto_labeling.py
│   ├── test_markers.py
│   └── test_human_labeling.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Configuration

All parameters are controlled via `configs/default.yaml`. Key settings:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `auto_labeling` | `box_threshold` | 0.30 | Grounding DINO box confidence |
| `auto_labeling` | `text_threshold` | 0.25 | Grounding DINO text confidence |
| `vlm_review` | `enabled` | true | Enable/disable VLM review |
| `vlm_review` | `auto_approve_threshold` | 0.65 | Auto-approve above this conf. |
| `vlm_review` | `strategy` | "patch" | "patch" or "full" review mode |
| `training` | `model` | yolo11s.pt | YOLOv11 model size |
| `training` | `epochs` | 100 | Training epochs |
| `training` | `imgsz` | 640 | Training image size |

## Evaluation Metrics

The evaluation stage generates:
- **mAP@50** and **mAP@50:95** (COCO-standard)
- **Precision, Recall, F1** (per-class and overall)
- **Confusion matrix** (normalized)
- **PR curves** and **F1 curves**
- **Confidence distribution** analysis
- **Box size distribution** analysis
- **Sample prediction visualizations**
- **HTML report** with all results
- **Pseudo-label quality analysis** (if ground truth available)

## References

- [DART: An Automated End-to-End Object Detection Pipeline](https://arxiv.org/abs/2407.09174) - Data diversification, annotation, review, and training
- [Auto-Labeling Data for Object Detection](https://arxiv.org/abs/2506.02359) - Vision-language model auto-labeling methodology
- [Grounding DINO](https://arxiv.org/abs/2303.05499) - Open-set object detection with grounded pre-training
- [Ultralytics YOLOv11](https://docs.ultralytics.com/models/yolo11/) - Real-time object detection
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) - Vision-language model for label verification
