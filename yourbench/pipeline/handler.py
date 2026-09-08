"""Validated stage execution shared by the CLI and Python API."""

import json
import time
import uuid
import importlib
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.progress import Progress, TextColumn, SpinnerColumn, TimeElapsedColumn

from yourbench.conf.loader import get_enabled_stages
from yourbench.pipeline.registry import STAGES, artifacts_for_stage
from yourbench.utils.dataset_engine import custom_load_dataset


class PipelineError(RuntimeError):
    """A required stage input or output is missing."""


def _get_stage_function(stage: str):
    spec = STAGES[stage]
    return importlib.import_module(f"yourbench.pipeline.{spec.module or spec.name}").run


def validate_pipeline(config) -> None:
    """Check model assignments and artifact dependencies before spending tokens."""
    available = set()
    for stage in get_enabled_stages(config):
        spec = STAGES[stage]
        if spec.requires_model or (stage == "ingestion" and config.pipeline.ingestion.llm_ingestion):
            names = config.model_roles.get(stage) or [m.model_name for m in config.model_list[:1]]
            if not names:
                raise PipelineError(f"Stage '{stage}' requires a model; configure model_list or OPENAI_MODEL")
        if stage == "ingestion":
            source = Path(config.pipeline.ingestion.source_documents_dir)
            extensions = config.pipeline.ingestion.supported_file_extensions
            output = Path(config.pipeline.ingestion.output_dir).resolve()
            if not source.is_dir() or not any(
                p.is_file() and p.suffix.lower() in extensions and not p.resolve().is_relative_to(output)
                for p in source.rglob("*")
            ):
                raise PipelineError(f"No supported source documents found in {source}")
        inputs, outputs = artifacts_for_stage(stage, config)
        if stage in {"question_rewriting", "prepare_lighteval"}:
            from yourbench.pipeline.prepare_lighteval import QUESTION_INPUTS

            selected = []
            for _, field, generation, default in QUESTION_INPUTS:
                subset = (
                    default if stage == "question_rewriting" else getattr(config.pipeline.prepare_lighteval, field)
                )
                required = getattr(config.pipeline, generation).run or subset != default
                if required or subset in available:
                    selected.append(subset)
                    continue
                try:
                    existing = custom_load_dataset(config, subset)
                except FileNotFoundError:
                    continue
                if len(existing):
                    selected.append(subset)
                    available.add(subset)
            if not selected:
                raise PipelineError(f"Stage '{stage}' requires at least one question subset")
            inputs = (*inputs, *selected)
            if stage == "question_rewriting":
                outputs = tuple(f"{subset}_rewritten" for subset in selected)
                export = config.pipeline.prepare_lighteval
                for _, field, _, default in QUESTION_INPUTS:
                    if default in selected and field not in export.model_fields_set:
                        setattr(export, field, f"{default}_rewritten")
        for subset in inputs:
            if subset not in available:
                try:
                    dataset = custom_load_dataset(config, subset)
                except FileNotFoundError as exc:
                    raise PipelineError(
                        f"Stage '{stage}' requires '{subset}'; enable its producer or supply saved data"
                    ) from exc
                if not len(dataset):
                    raise PipelineError(f"Stage '{stage}' requires nonempty '{subset}'")
                available.add(subset)
        available.update(outputs)


def run_stage(stage: str, config) -> float:
    start = time.perf_counter()
    _get_stage_function(stage)(config)
    return time.perf_counter() - start


def run_pipeline(config_path: str, debug: bool = False) -> None:
    from yourbench.conf.loader import load_config

    run_pipeline_with_config(load_config(config_path), debug=debug)


def run_pipeline_with_config(config, debug: bool = False) -> None:
    run_pipeline_with_progress(config, debug=debug, quiet=True)


def run_pipeline_with_progress(
    config, debug: bool = False, quiet: bool = False, console: Console | None = None
) -> None:
    config.debug = config.debug or debug
    enabled = get_enabled_stages(config)
    if not enabled:
        raise PipelineError("No pipeline stages enabled")
    state = {"run_id": str(uuid.uuid4()), "status": "running", "stages": enabled, "completed": []}
    hf = config.hf_configuration
    status_path = Path(hf.local_dataset_dir).parent / "run.json" if hf.local_saving and hf.local_dataset_dir else None

    def save_status():
        if status_path:
            status_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            temporary.replace(status_path)

    save_status()
    try:
        validate_pipeline(config)
        console = console or Console()
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console, disable=quiet
        ) as progress:
            task = progress.add_task("Pipeline", total=len(enabled))
            for stage in enabled:
                state["current_stage"] = stage
                save_status()
                progress.update(task, description=STAGES[stage].title)
                elapsed = run_stage(stage, config)
                state["completed"].append({"stage": stage, "seconds": elapsed})
                progress.advance(task)
                logger.success(f"Completed {stage} in {elapsed:.2f}s")
        from yourbench.utils.dataset_card import upload_dataset_card

        upload_dataset_card(config)
        state["status"] = "completed"
    except BaseException as error:
        state["status"] = "failed"
        state["error_type"] = type(error).__name__
        raise
    finally:
        save_status()
