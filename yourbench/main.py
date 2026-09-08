#!/usr/bin/env python3
"""YourBench CLI - Dynamic Evaluation Set Generation with Large Language Models.

A modern CLI using Rich for beautiful output, progress tracking, and useful commands.
"""

import os
import sys
import json
import atexit
from pathlib import Path
from datetime import datetime

import typer
from dotenv import load_dotenv
from loguru import logger
from rich.panel import Panel
from rich.table import Table
from rich.console import Console


load_dotenv()
console = Console()


def configure_logging(debug: bool = False, log_dir: Path = None, quiet: bool = False):
    """Configure structured logging with file output."""
    logger.remove()  # Remove default handler

    log_level = "DEBUG" if debug else os.getenv("YOURBENCH_LOG_LEVEL", "INFO")

    if not quiet:
        # Console handler with structured format for logs with stage
        console_format = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[stage]: <16}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stderr,
            format=console_format,
            level=log_level,
            filter=lambda record: "stage" in record["extra"],
            enqueue=True,
        )

        # Fallback console handler for logs without stage
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level=log_level,
            filter=lambda record: "stage" not in record["extra"],
            enqueue=True,
        )

    # File handler - JSON structured logs
    if log_dir is None:
        log_dir = Path(os.getenv("YOURBENCH_LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"yourbench_{timestamp}.jsonl"

    logger.add(
        str(log_file),
        format="{message}",
        level="DEBUG",
        serialize=True,
        enqueue=True,
        rotation="100 MB",
    )

    # Summary log file (INFO and above only)
    summary_file = log_dir / f"yourbench_{timestamp}_summary.log"
    logger.add(
        str(summary_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra} | {message}",
        level="INFO",
        filter=lambda record: record["level"].no >= 20,
        enqueue=True,
    )

    if not quiet:
        logger.info(f"Logging configured. JSON logs: {log_file}, Summary: {summary_file}")
    return log_file, summary_file


def cleanup_logging():
    """Ensure all logs are flushed and closed."""
    logger.complete()


atexit.register(cleanup_logging)


app = typer.Typer(
    name="yourbench",
    help="YourBench - Dynamic Evaluation Set Generation with Large Language Models.",
    pretty_exceptions_show_locals=False,
    add_completion=False,
)


def _print_banner():
    """Print YourBench banner."""
    banner = """
[bold blue]\u256d\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u256e[/bold blue]
[bold blue]\u2502[/bold blue]  [bold cyan]YourBench[/bold cyan] - Dynamic Evaluation Set Generation           [bold blue]\u2502[/bold blue]
[bold blue]\u2502[/bold blue]  [dim]Build domain-specific benchmarks from your documents[/dim]     [bold blue]\u2502[/bold blue]
[bold blue]\u2570\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u256f[/bold blue]
"""
    console.print(banner)


def _print_config_summary(config) -> None:
    """Print a summary of the configuration."""
    from yourbench.conf.loader import get_enabled_stages

    table = Table(title="Configuration Summary", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # Dataset info
    hf = config.hf_configuration
    if hf.hf_dataset_name:
        org_prefix = f"{hf.hf_organization}/" if hf.hf_organization else ""
        table.add_row("Dataset", f"{org_prefix}{hf.hf_dataset_name}")
    table.add_row("Push to Hub", "\u2713" if hf.push_to_hub else "\u2717")
    table.add_row("Private", "\u2713" if hf.private else "\u2717")

    # Models
    if config.model_list:
        models = ", ".join(m.model_name for m in config.model_list)
        table.add_row("Models", models)

    # Enabled stages
    stages = get_enabled_stages(config)
    if stages:
        table.add_row("Stages", ", ".join(stages))

    console.print(table)
    console.print()


@app.command()
def create(
    brief: str = typer.Argument(..., help="Describe the benchmark you want to build"),
    source: Path = typer.Option(..., "--source", help="Directory of source documents"),
    output: Path = typer.Option(..., "--output", help="New or empty output directory"),
    model: str = typer.Option("", "--model", envvar="YOURBENCH_MODEL", help="Model identifier"),
    provider: str | None = typer.Option(None, "--provider", help="Inference provider"),
    base_url: str | None = typer.Option(None, "--base-url", help="Compatible inference endpoint"),
    api_key_env: str | None = typer.Option(None, "--api-key-env", help="Name of the API key environment variable"),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", min=1, help="Provider output-token limit per response, including planning"
    ),
    concurrency: int = typer.Option(8, "--concurrency", min=1, help="Maximum simultaneous requests per model"),
    plan_only: bool = typer.Option(
        False, "--plan-only", help="Make one planning call and save a recipe without generation"
    ),
) -> None:
    """Build a benchmark from a natural-language brief; save a reusable local recipe."""
    from yourbench.planning import create_recipe
    from yourbench.conf.loader import load_config
    from yourbench.pipeline.handler import run_pipeline_with_progress

    try:
        intent, config_path = create_recipe(
            brief,
            source,
            output,
            model,
            provider,
            base_url,
            api_key_env,
            max_tokens=max_tokens,
            concurrency=concurrency,
        )
        console.print(f"Saved recipe: {config_path}", markup=False)
        for assumption in intent.assumptions:
            console.print(f"Assumption: {assumption}", markup=False)
        if not plan_only:
            config = load_config(config_path)
            run_pipeline_with_progress(config, console=console)
            console.print(f"Benchmark saved to {config_path.parent}", markup=False)
    except Exception as exc:
        from pydantic import ValidationError

        message = "Planner returned an invalid benchmark intent" if isinstance(exc, ValidationError) else str(exc)
        console.print(f"Creation failed: {message}", markup=False)
        raise typer.Exit(1) from None


@app.command()
def run(
    config_path: str = typer.Argument(..., help="YAML recipe or generated output directory"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output (only errors)"),
    no_banner: bool = typer.Option(False, "--no-banner", help="Hide the banner"),
) -> None:
    """Run YourBench pipeline with a config file."""
    configure_logging(debug=debug, quiet=quiet)

    if not quiet and not no_banner:
        _print_banner()

    from yourbench.api import _recipe_path

    config_file = _recipe_path(config_path)
    if not config_file.exists():
        console.print(f"[bold red]\u2717[/bold red] Config file not found: {config_path}")
        raise typer.Exit(1)

    if config_file.suffix not in {".yaml", ".yml"}:
        console.print(f"[bold red]\u2717[/bold red] Config must be a YAML file (.yaml or .yml): {config_path}")
        raise typer.Exit(1)

    from yourbench.conf.loader import load_config
    from yourbench.pipeline.handler import run_pipeline_with_progress

    try:
        with console.status("[bold cyan]Loading configuration..."):
            config = load_config(config_file)
            if debug:
                config.debug = True

        if not quiet:
            _print_config_summary(config)

        run_pipeline_with_progress(config, debug=debug, quiet=quiet, console=console)

        if not quiet:
            console.print()
            console.print(
                Panel.fit("[bold green]\u2713 Pipeline completed successfully![/bold green]", border_style="green")
            )

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        console.print(f"[bold red]\u2717[/bold red] Pipeline failed: {e}")
        raise typer.Exit(1)


@app.command("inspect")
def inspect_result(
    path: Path = typer.Argument(..., help="Output directory or YAML recipe"),
    as_json: bool = typer.Option(False, "--json", help="Emit a machine-readable summary"),
) -> None:
    """Read local run status and dataset sizes without model calls or credentials."""
    from yourbench import load_result

    try:
        summary = load_result(path).summary()
    except Exception as error:
        console.print(f"Cannot inspect local result ({type(error).__name__})", markup=False)
        raise typer.Exit(1) from None
    if as_json:
        # Avoid Rich wrapping JSON strings or applying terminal markup.
        typer.echo(json.dumps(summary, indent=2))
        return
    console.print(f"Status: {summary['status']}", markup=False)
    console.print(f"Recipe: {summary['config_path']}", markup=False)
    table = Table("Subset", "Rows")
    for name, details in summary["subsets"].items():
        table.add_row(name, str(details["rows"]))
    console.print(table)
    if not summary["subsets"]:
        console.print("No local datasets found.")
    if summary["status"] != "completed":
        console.print("Stored artifacts may be partial or left over from an earlier run.")


@app.command()
def validate(
    config_path: str = typer.Argument(..., help="Path to YAML config file to validate"),
) -> None:
    """Validate a configuration file without running the pipeline."""
    _print_banner()

    config_file = Path(config_path)
    if not config_file.exists():
        console.print(f"[bold red]\u2717[/bold red] Config file not found: {config_path}")
        raise typer.Exit(1)

    from yourbench.conf.loader import load_config, get_enabled_stages
    from yourbench.conf.schema import ConfigValidationError

    try:
        with console.status("[bold cyan]Validating configuration..."):
            config = load_config(config_file)

        console.print("[bold green]\u2713[/bold green] Configuration is valid!")
        console.print()
        _print_config_summary(config)

        # Show detailed stage information
        stages = get_enabled_stages(config)
        if stages:
            console.print(f"[cyan]Enabled stages ({len(stages)}):[/cyan]")
            for i, stage in enumerate(stages, 1):
                console.print(f"  {i}. {stage}")
        else:
            console.print("[yellow]\u26a0[/yellow] No pipeline stages enabled")

    except ConfigValidationError as e:
        console.print(f"[bold red]\u2717[/bold red] Validation failed: {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]\u2717[/bold red] Error loading config: {e}")
        raise typer.Exit(1)


@app.command()
def init(
    output: Path = typer.Option(Path("config.yaml"), "--output", "-o", help="Output file path"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing file"),
    model: str = typer.Option("YOUR_MODEL_ID", "--model", envvar="YOURBENCH_MODEL", help="Model identifier"),
    source: Path = typer.Option(
        Path("data/raw"), "--source", help="Source directory, resolved from current directory"
    ),
) -> None:
    """Write a local starter recipe with all required stages; edit credentials before running."""
    import yaml

    if output.exists() and not force:
        console.print(f"File already exists: {output}. Use --force to overwrite.", markup=False)
        raise typer.Exit(1)
    config = {
        "hf_configuration": {
            "hf_dataset_name": "benchmark",
            "push_to_hub": False,
            "upload_card": False,
            "private": True,
            "local_dataset_dir": "datasets",
            "export_jsonl": True,
            "jsonl_export_dir": "jsonl",
        },
        "model_list": [{"model_name": model, "max_concurrent_requests": 8}],
        "pipeline": {
            "ingestion": {
                "source_documents_dir": str(source.expanduser().resolve()),
                "output_dir": "processed",
            },
            "summarization": {},
            "chunking": {},
            "single_hop_question_generation": {"question_mode": "open-ended"},
            "prepare_lighteval": {},
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    console.print(f"Saved {output}. Configure the model endpoint/key, then run: yourbench run {output}", markup=False)


@app.command()
def stages() -> None:
    """Show available pipeline stages and their descriptions."""
    _print_banner()

    table = Table(title="Pipeline Stages", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Stage", style="cyan")
    table.add_column("Description", style="white")

    from yourbench.pipeline.registry import STAGES

    for i, stage in enumerate(STAGES.values(), 1):
        table.add_row(str(i), stage.name, stage.title)

    console.print(table)
    console.print()
    console.print("[dim]Enable stages by adding them to your config's pipeline section.[/dim]")


@app.command()
def estimate(
    config_path: str = typer.Argument(..., help="Path to YAML config file"),
) -> None:
    """Estimate token usage for a pipeline run."""
    _print_banner()

    config_file = Path(config_path)
    if not config_file.exists():
        console.print(f"[bold red]✗[/bold red] Config file not found: {config_path}")
        raise typer.Exit(1)

    from yourbench.conf.loader import load_config
    from yourbench.utils.token_estimation import (
        format_token_count,
        format_token_range,
        estimate_pipeline_tokens,
    )

    try:
        with console.status("[bold cyan]Analyzing configuration..."):
            config = load_config(config_file)
            estimates = estimate_pipeline_tokens(config)

        # Source info
        console.print("[bold]Source Documents:[/bold]")
        console.print(f"  Files: {estimates.get('source_file_count', 0)}")
        console.print(f"  Estimated tokens: {format_token_count(estimates['source_tokens'])}")
        console.print()

        # Stage breakdown
        table = Table(title="Token Estimation by Stage", show_header=True, header_style="bold magenta")
        table.add_column("Stage", style="cyan")
        table.add_column("Input Tokens", style="green", justify="right")
        table.add_column("Output Tokens (25%-75%)", style="yellow", justify="right")
        table.add_column("API Calls", style="blue", justify="right")
        table.add_column("Notes", style="dim")

        for stage, info in estimates["stages"].items():
            input_tok = format_token_count(info.get("input_tokens", 0)) if info.get("input_tokens") else "-"
            out_low = info.get("output_tokens_low", 0)
            out_high = info.get("output_tokens_high", 0)
            output_tok = format_token_range(out_low, out_high) if out_low or out_high else "-"
            calls = str(info.get("calls", "-")) if info.get("calls") else "-"
            note = info.get("note", "")
            table.add_row(stage.replace("_", " ").title(), input_tok, output_tok, calls, note)

        console.print(table)
        console.print()

        # Totals
        console.print(
            Panel.fit(
                f"[bold]Total Estimated Usage:[/bold]\n"
                f"  Input tokens:  [green]{format_token_count(estimates['total_input_tokens'])}[/green]\n"
                f"  Output tokens: [yellow]{format_token_range(estimates['total_output_tokens_low'], estimates['total_output_tokens_high'])}[/yellow]\n"
                f"  Total:         [bold cyan]{format_token_range(estimates['total_tokens_low'], estimates['total_tokens_high'])}[/bold cyan]",
                title="Summary",
                border_style="blue",
            )
        )

        console.print()
        console.print(
            "[dim]Note: These are rough estimates. Actual usage may vary based on document content and model responses.[/dim]"
        )

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Error: {e}")
        raise typer.Exit(1)


@app.command("version")
def version_command() -> None:
    """Show YourBench version."""
    from importlib.metadata import version as get_version

    try:
        v = get_version("yourbench")
    except Exception:
        v = "development"

    console.print(
        Panel.fit(f"[bold cyan]YourBench[/bold cyan] version [bold green]{v}[/bold green]", border_style="blue")
    )


def main() -> None:
    """Entry point for the CLI."""
    # Handle version flag
    if len(sys.argv) == 2 and sys.argv[1] in {"--version", "-v"}:
        version_command()
        return

    # If no arguments, show help
    if len(sys.argv) == 1:
        _print_banner()
        app()
        return

    # If first arg looks like a path (not a command), assume it's 'run'
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        if not first_arg.startswith("-") and Path(first_arg).suffix in {".yaml", ".yml"}:
            sys.argv = [sys.argv[0], "run"] + sys.argv[1:]

    app()


if __name__ == "__main__":
    main()
