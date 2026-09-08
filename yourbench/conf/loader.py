"""
Simplified configuration loader using Pydantic for validation.

Loads YAML configs and validates against Pydantic schemas.
"""

import os
from typing import Any
from pathlib import Path

import yaml
from loguru import logger

from yourbench.utils.env import expand_env_recursive
from yourbench.conf.schema import (
    ModelConfig,
    YourbenchConfig,
    ConfigValidationError,
)
from yourbench.conf.prompts import DEFAULT_PROMPTS, load_prompt
from yourbench.pipeline.registry import STAGES


STAGE_ORDER = list(STAGES)

# Prompt field paths: (config path tuple, default prompt key)
PROMPT_FIELDS = [
    (("pipeline", "ingestion", "pdf_llm_prompt"), "pdf_llm_prompt"),
    (("pipeline", "summarization", "summarization_user_prompt"), "summarization_user_prompt"),
    (("pipeline", "summarization", "combine_summaries_user_prompt"), "combine_summaries_user_prompt"),
    (("pipeline", "single_hop_question_generation", "single_hop_system_prompt"), "single_hop_system_prompt"),
    (
        ("pipeline", "single_hop_question_generation", "single_hop_system_prompt_multi"),
        "single_hop_system_prompt_multi",
    ),
    (("pipeline", "single_hop_question_generation", "single_hop_user_prompt"), "single_hop_user_prompt"),
    (("pipeline", "multi_hop_question_generation", "multi_hop_system_prompt"), "multi_hop_system_prompt"),
    (("pipeline", "multi_hop_question_generation", "multi_hop_system_prompt_multi"), "multi_hop_system_prompt_multi"),
    (("pipeline", "multi_hop_question_generation", "multi_hop_user_prompt"), "multi_hop_user_prompt"),
    (("pipeline", "cross_document_question_generation", "multi_hop_system_prompt"), "multi_hop_system_prompt"),
    (
        ("pipeline", "cross_document_question_generation", "multi_hop_system_prompt_multi"),
        "multi_hop_system_prompt_multi",
    ),
    (("pipeline", "cross_document_question_generation", "multi_hop_user_prompt"), "multi_hop_user_prompt"),
    (("pipeline", "question_rewriting", "question_rewriting_system_prompt"), "question_rewriting_system_prompt"),
    (("pipeline", "question_rewriting", "question_rewriting_user_prompt"), "question_rewriting_user_prompt"),
]


def load_config(yaml_path: str | Path) -> YourbenchConfig:
    """Load a yourbench config from YAML file.

    1. Parse YAML
    2. Expand $VAR environment variables
    3. Handle legacy field names
    4. Mark enabled stages (presence = run)
    5. Auto-load OpenAI from env if no models
    6. Validate with Pydantic
    7. Load prompts
    8. Assign model roles
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    # Load raw YAML
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    return resolve_config(data, base_dir=yaml_path.resolve().parent)


def _expand_execution_values(data, field=""):
    """Natural-language content is literal, never a credential interpolation surface."""
    if field == "additional_instructions" or "prompt" in field:
        if isinstance(data, str) and data.startswith("file:"):
            return expand_env_recursive(data)
        return data
    if isinstance(data, dict):
        return {key: _expand_execution_values(value, key) for key, value in data.items()}
    if isinstance(data, list):
        return [_expand_execution_values(value, field) for value in data]
    return expand_env_recursive(data)


def resolve_config(data: dict[str, Any], base_dir: str | Path | None = None) -> YourbenchConfig:
    """Resolve YAML or generated configuration through the same validation path.

    Relative file paths are relative to the configuration file, or base_dir.
    """
    from copy import deepcopy

    if not isinstance(data, dict):
        raise ConfigValidationError("Configuration must be a mapping")
    data = _handle_legacy_fields(deepcopy(data))
    data = _auto_load_openai_from_env(data)
    data = _expand_execution_values(data)
    data = _mark_enabled_stages(data)
    try:
        config = YourbenchConfig.model_validate(data)
    except Exception as exc:
        # Pydantic input values can contain credentials. Exclude them from errors.
        from pydantic import ValidationError

        if isinstance(exc, ValidationError):
            details = "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
                for e in exc.errors(include_input=False, include_url=False)
            )
            raise ConfigValidationError(details) from None
        raise ConfigValidationError("Invalid configuration") from None

    from yourbench.utils.env import validate_env_expanded

    for model in config.model_list:
        for field in ("api_key", "base_url", "model_name"):
            value = getattr(model, field)
            if value:
                validate_env_expanded(value, f"model_list.{field}")

    base = Path(base_dir or Path.cwd()).resolve()
    for obj, fields in (
        (config.hf_configuration, ("local_dataset_dir", "jsonl_export_dir")),
        (config.pipeline.ingestion, ("source_documents_dir", "output_dir")),
    ):
        for field in fields:
            value = getattr(obj, field)
            if value:
                setattr(obj, field, str((base / Path(value).expanduser()).resolve()))
    for stage in STAGE_ORDER:
        cfg = getattr(config.pipeline, stage)
        if getattr(cfg, "question_schema", None):
            cfg.question_schema = str((base / Path(cfg.question_schema).expanduser()).resolve())
    _load_prompts(config, base)
    _assign_model_roles(config)
    if config.pipeline.question_rewriting.run:
        export = config.pipeline.prepare_lighteval
        for field, subset in (
            ("single_hop_subset", "single_hop_questions"),
            ("multi_hop_subset", "multi_hop_questions"),
            ("cross_doc_subset", "cross_document_questions"),
        ):
            stage_name = subset.replace("_questions", "_question_generation")
            if getattr(config.pipeline, stage_name).run and field not in export.model_fields_set:
                setattr(export, field, subset + "_rewritten")
    return config


def _handle_legacy_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Handle legacy field renames."""
    if "models" in data and "model_list" not in data:
        data["model_list"] = data.pop("models")
        logger.debug("Renamed 'models' -> 'model_list'")

    if "pipeline_config" in data and "pipeline" not in data:
        data["pipeline"] = data.pop("pipeline_config")

    return data


def _mark_enabled_stages(data: dict[str, Any]) -> dict[str, Any]:
    """Mark stages as run=True if present in config (presence = enabled)."""
    pipeline = data.get("pipeline", {})
    if not pipeline:
        return data

    for stage in STAGE_ORDER:
        if stage in pipeline:
            stage_cfg = pipeline[stage]
            if stage_cfg is None:
                # Empty stage (e.g., "summarization:") means run=True
                pipeline[stage] = {"run": True}
            elif isinstance(stage_cfg, dict) and "run" not in stage_cfg:
                stage_cfg["run"] = True

    data["pipeline"] = pipeline
    return data


def _auto_load_openai_from_env(data: dict[str, Any]) -> dict[str, Any]:
    """Create OpenAI model from env vars if no models configured."""
    if data.get("model_list") or data.get("models"):
        return data

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not os.getenv("OPENAI_MODEL"):
        return data

    openai_model = {
        "model_name": os.environ["OPENAI_MODEL"],
        "api_key": "$OPENAI_API_KEY",
        "max_concurrent_requests": 128,
    }

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if base_url:
        openai_model["base_url"] = base_url

    data["model_list"] = [openai_model]
    logger.info(f"Auto-loaded OpenAI model from environment: {openai_model['model_name']}")
    return data


def _load_prompts(config: YourbenchConfig, base_dir: Path | None = None) -> None:
    """Load prompts once, failing explicitly for missing user files."""
    for path_tuple, default_key in PROMPT_FIELDS:
        obj = config
        for key in path_tuple[:-1]:
            obj = getattr(obj, key)
        field = path_tuple[-1]
        value = getattr(obj, field, "")
        setattr(obj, field, load_prompt(value, DEFAULT_PROMPTS.get(default_key, ""), base_dir=base_dir))


def _assign_model_roles(config: YourbenchConfig) -> None:
    """Assign default model to stages without explicit model_roles."""
    if not config.model_list:
        return

    default_model = config.model_list[0].model_name
    if not default_model:
        return

    for stage in STAGE_ORDER:
        if stage not in config.model_roles:
            config.model_roles[stage] = [default_model]


# Helper functions for config access
def get_enabled_stages(config: YourbenchConfig) -> list[str]:
    """Return list of enabled pipeline stages in execution order."""
    return [
        s
        for s in STAGE_ORDER
        if getattr(config.pipeline, s, None) and getattr(getattr(config.pipeline, s), "run", False)
    ]


def is_stage_enabled(config: YourbenchConfig, stage: str) -> bool:
    """Check if a pipeline stage is enabled."""
    stage_cfg = getattr(config.pipeline, stage, None)
    return stage_cfg and getattr(stage_cfg, "run", False)


def get_model_for_stage(config: YourbenchConfig, stage: str) -> str | None:
    """Get the primary model name for a stage."""
    models = config.model_roles.get(stage, [])
    if models:
        return models[0]
    if config.model_list:
        return config.model_list[0].model_name
    return None


def get_model_config(config: YourbenchConfig, model_name: str) -> ModelConfig | None:
    """Get model config by name."""
    for model in config.model_list:
        if model.model_name == model_name:
            return model
    return None
