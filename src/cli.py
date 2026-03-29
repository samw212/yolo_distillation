"""Command-line interface for the YOLOv11 Distillation Pipeline.

Usage:
    # Run full pipeline
    python -m src.cli run --config configs/default.yaml

    # Run specific stages
    python -m src.cli run --stages data_acquisition auto_labeling

    # Skip VLM review
    python -m src.cli run --skip vlm_review

    # Force re-run all stages
    python -m src.cli run --force

    # Run individual stages
    python -m src.cli pull-data --workspace my-ws --project my-proj --version 1
    python -m src.cli auto-label
    python -m src.cli vlm-review
    python -m src.cli export-labels
    python -m src.cli import-labels --import-dir output/human_corrected
    python -m src.cli train
    python -m src.cli evaluate
"""

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.option(
    "--config", "-c",
    default="configs/default.yaml",
    help="Path to configuration YAML file.",
)
@click.pass_context
def cli(ctx, config):
    """YOLOv11 Distillation Pipeline - Knowledge transfer from Grounding DINO."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@cli.command()
@click.option("--stages", "-s", multiple=True, help="Only run specific stages.")
@click.option("--skip", "-k", multiple=True, help="Skip specific stages.")
@click.option("--force", "-f", is_flag=True, help="Force re-run completed stages.")
@click.option("--workspace", help="Roboflow workspace (overrides config).")
@click.option("--project", help="Roboflow project (overrides config).")
@click.option("--version", type=int, help="Roboflow dataset version (overrides config).")
@click.pass_context
def run(ctx, stages, skip, force, workspace, project, version):
    """Run the full distillation pipeline (or selected stages)."""
    from src.pipeline.orchestrator import run_pipeline

    overrides = {}
    if workspace:
        overrides["data_acquisition.workspace"] = workspace
    if project:
        overrides["data_acquisition.project"] = project
    if version:
        overrides["data_acquisition.version"] = version

    results = run_pipeline(
        config_path=ctx.obj["config_path"],
        overrides=overrides or None,
        stages=list(stages) if stages else None,
        skip_stages=list(skip) if skip else None,
        force=force,
    )

    _display_results(results)


@cli.command("pull-data")
@click.option("--workspace", required=True, help="Roboflow workspace name.")
@click.option("--project", required=True, help="Roboflow project name.")
@click.option("--version", required=True, type=int, help="Dataset version.")
@click.option("--max-images", type=int, help="Limit number of images.")
@click.pass_context
def pull_data(ctx, workspace, project, version, max_images):
    """Pull dataset from Roboflow."""
    from src.pipeline.orchestrator import run_pipeline

    overrides = {
        "data_acquisition.workspace": workspace,
        "data_acquisition.project": project,
        "data_acquisition.version": version,
    }
    if max_images:
        overrides["data_acquisition.max_images"] = max_images

    results = run_pipeline(
        config_path=ctx.obj["config_path"],
        overrides=overrides,
        stages=["data_acquisition"],
        force=True,
    )
    _display_results(results)


@cli.command("auto-label")
@click.option("--box-threshold", type=float, help="Override box confidence threshold.")
@click.option("--text-threshold", type=float, help="Override text confidence threshold.")
@click.pass_context
def auto_label(ctx, box_threshold, text_threshold):
    """Run Grounding DINO auto-labeling on dataset images."""
    from src.pipeline.orchestrator import run_pipeline

    overrides = {}
    if box_threshold is not None:
        overrides["auto_labeling.box_threshold"] = box_threshold
    if text_threshold is not None:
        overrides["auto_labeling.text_threshold"] = text_threshold

    results = run_pipeline(
        config_path=ctx.obj["config_path"],
        overrides=overrides or None,
        stages=["auto_labeling"],
        force=True,
    )
    _display_results(results)


@cli.command("vlm-review")
@click.option("--disable", is_flag=True, help="Disable VLM review (auto-approve all).")
@click.option("--auto-approve-threshold", type=float, help="Confidence threshold for auto-approval.")
@click.pass_context
def vlm_review_cmd(ctx, disable, auto_approve_threshold):
    """Run Qwen VLM review of pseudo-labels."""
    from src.pipeline.orchestrator import run_pipeline

    overrides = {}
    if disable:
        overrides["vlm_review.enabled"] = False
    if auto_approve_threshold is not None:
        overrides["vlm_review.auto_approve_threshold"] = auto_approve_threshold

    results = run_pipeline(
        config_path=ctx.obj["config_path"],
        overrides=overrides or None,
        stages=["vlm_review"],
        force=True,
    )
    _display_results(results)


@cli.command("export-labels")
@click.pass_context
def export_labels(ctx):
    """Export flagged images to CVAT format for human annotation."""
    from src.pipeline.orchestrator import run_pipeline

    results = run_pipeline(
        config_path=ctx.obj["config_path"],
        stages=["human_labeling"],
        force=True,
    )
    _display_results(results)


@cli.command("import-labels")
@click.option(
    "--import-dir",
    required=True,
    help="Directory containing human-corrected labels.",
)
@click.option(
    "--format",
    "import_format",
    type=click.Choice(["cvat_xml", "yolo_txt", "coco_json"]),
    default="cvat_xml",
    help="Import format.",
)
@click.pass_context
def import_labels(ctx, import_dir, import_format):
    """Import human-corrected labels back into the pipeline."""
    from src.pipeline.stages.human_labeling import import_corrected_labels
    from src.pipeline.utils.config import load_config

    config = load_config(ctx.obj["config_path"], {
        "human_labeling.import_dir": import_dir,
        "human_labeling.import_format": import_format,
    })
    result = import_corrected_labels(config)
    console.print(f"[green]Imported {result['imported_count']} label files[/green]")


@cli.command()
@click.option("--epochs", type=int, help="Override number of training epochs.")
@click.option("--batch-size", type=int, help="Override batch size.")
@click.option("--model", help="Override YOLO model (e.g., yolo11n.pt, yolo11m.pt).")
@click.option("--resume", is_flag=True, help="Resume from last checkpoint.")
@click.pass_context
def train(ctx, epochs, batch_size, model, resume):
    """Train YOLOv11 on approved pseudo-labels."""
    from src.pipeline.orchestrator import run_pipeline

    overrides = {}
    if epochs is not None:
        overrides["training.epochs"] = epochs
    if batch_size is not None:
        overrides["training.batch_size"] = batch_size
    if model:
        overrides["training.model"] = model
    if resume:
        overrides["training.resume"] = True

    results = run_pipeline(
        config_path=ctx.obj["config_path"],
        overrides=overrides or None,
        stages=["training"],
        force=True,
    )
    _display_results(results)


@cli.command()
@click.option("--conf", type=float, help="Override confidence threshold.")
@click.pass_context
def evaluate(ctx, conf):
    """Run comprehensive model evaluation."""
    from src.pipeline.orchestrator import run_pipeline

    overrides = {}
    if conf is not None:
        overrides["evaluation.conf_threshold"] = conf

    results = run_pipeline(
        config_path=ctx.obj["config_path"],
        overrides=overrides or None,
        stages=["evaluation"],
        force=True,
    )
    _display_results(results)


@cli.command()
@click.pass_context
def status(ctx):
    """Show pipeline status (completed/pending stages)."""
    from src.pipeline.utils.config import load_config
    from src.pipeline.utils.markers import is_stage_complete

    config = load_config(ctx.obj["config_path"])

    table = Table(title="Pipeline Status")
    table.add_column("Stage", style="cyan")
    table.add_column("Status", style="bold")

    stage_names = [
        "data_acquisition",
        "auto_labeling",
        "vlm_review",
        "human_labeling",
        "training",
        "evaluation",
    ]

    for stage in stage_names:
        if is_stage_complete(config, stage):
            table.add_row(stage, "[green]Completed[/green]")
        else:
            table.add_row(stage, "[yellow]Pending[/yellow]")

    console.print(table)


def _display_results(results: dict):
    """Display pipeline results in a formatted table."""
    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return

    for stage_name, result in results.items():
        if isinstance(result, dict) and "error" in result:
            console.print(f"[red]{stage_name}: FAILED - {result['error']}[/red]")
        else:
            console.print(f"[green]{stage_name}: Completed[/green]")

            # Show key metrics if available
            if isinstance(result, dict):
                metrics = result.get("metrics") or result.get("stats")
                if metrics and isinstance(metrics, dict):
                    table = Table(title=f"{stage_name} Results")
                    table.add_column("Metric")
                    table.add_column("Value")
                    for k, v in metrics.items():
                        if isinstance(v, float):
                            table.add_row(str(k), f"{v:.4f}")
                        elif isinstance(v, dict):
                            for sub_k, sub_v in v.items():
                                table.add_row(f"  {sub_k}", str(sub_v))
                        else:
                            table.add_row(str(k), str(v))
                    console.print(table)


if __name__ == "__main__":
    cli()
