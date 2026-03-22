"""Pipeline Orchestrator.

Coordinates the execution of all pipeline stages in sequence,
handling stage dependencies, completion markers, and error recovery.

Pipeline stages (following DART methodology):
1. Data Acquisition  - Pull images from Roboflow
2. Auto-Labeling     - Generate pseudo-labels with Grounding DINO
3. VLM Review        - Verify labels with Qwen2.5-VL (optional)
4. Human Labeling    - Export flagged images, import corrections
5. Training          - Fine-tune YOLOv11 on approved labels
6. Evaluation        - Comprehensive model evaluation
"""

import logging
import time
from pathlib import Path

from src.pipeline.stages import (
    auto_labeling,
    data_acquisition,
    evaluation,
    human_labeling,
    training,
    vlm_review,
)
from src.pipeline.utils.config import ensure_dirs, load_config
from src.pipeline.utils.logging import setup_logging
from src.pipeline.utils.markers import is_stage_complete, mark_stage_complete

logger = logging.getLogger("distill.orchestrator")

STAGES = [
    ("data_acquisition", data_acquisition),
    ("auto_labeling", auto_labeling),
    ("vlm_review", vlm_review),
    ("human_labeling", human_labeling),
    ("training", training),
    ("evaluation", evaluation),
]


def run_pipeline(
    config_path: str = "configs/default.yaml",
    overrides: dict | None = None,
    stages: list[str] | None = None,
    skip_stages: list[str] | None = None,
    force: bool = False,
) -> dict:
    """Run the full distillation pipeline.

    Args:
        config_path: Path to YAML configuration file.
        overrides: Dictionary of config overrides.
        stages: If specified, only run these stages.
        skip_stages: Stages to skip.
        force: If True, ignore completion markers and re-run all stages.

    Returns:
        Dictionary with results from each stage.
    """
    config = load_config(config_path, overrides)
    setup_logging(config)
    ensure_dirs(config)

    skip_completed = config.get("pipeline", {}).get("skip_completed", True) and not force
    skip_stages = set(skip_stages or [])

    logger.info("=" * 60)
    logger.info("YOLOv11 Distillation Pipeline")
    logger.info("Based on DART (arXiv:2407.09174) & Auto-Labeling (arXiv:2506.02359)")
    logger.info("=" * 60)
    logger.info("Classes: %s", ", ".join(config["classes"]))

    results = {}

    for stage_name, stage_module in STAGES:
        # Filter stages if specified
        if stages and stage_name not in stages:
            logger.info("[SKIP] %s (not in requested stages)", stage_name)
            continue

        if stage_name in skip_stages:
            logger.info("[SKIP] %s (explicitly skipped)", stage_name)
            continue

        # Check completion marker
        if skip_completed and is_stage_complete(config, stage_name):
            logger.info("[DONE] %s (already completed, use --force to re-run)", stage_name)
            continue

        # Run stage
        logger.info("-" * 60)
        logger.info("[START] Stage: %s", stage_name)
        start_time = time.time()

        try:
            result = stage_module.run(config)
            elapsed = time.time() - start_time

            results[stage_name] = result
            mark_stage_complete(config, stage_name, {"elapsed_seconds": elapsed})

            logger.info("[DONE] %s (%.1f seconds)", stage_name, elapsed)

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("[FAIL] %s after %.1f seconds: %s", stage_name, elapsed, e)
            results[stage_name] = {"error": str(e)}
            raise

    logger.info("=" * 60)
    logger.info("Pipeline complete")
    logger.info("=" * 60)

    return results
