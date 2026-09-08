from __future__ import annotations
from typing import Any

from loguru import logger

from datasets import Dataset
from yourbench.utils.schema_loader import load_schema_from_spec
from yourbench.utils.chunking_utils import get_sampling_cfg
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
from yourbench.utils.parsing_engine import (
    parse_multi_hop_responses,
    parse_single_hop_responses,
    _remove_duplicate_questions,
)
from yourbench.utils.prompt_builder import build_system_prompt
from yourbench.utils.logging_context import log_step, log_stage
from yourbench.utils.question_models import question_dataset
from yourbench.utils.cross_document_utils import create_cross_document_dataset
from yourbench.utils.inference.inference_core import run_inference
from yourbench.utils.inference.inference_builders import (
    build_multi_hop_inference_calls,
    build_single_hop_inference_calls,
)


def _get_system_prompt(stage_cfg: Any, mode: str, is_multi: bool = False) -> str:
    """Get system prompt, rendering the selected default or custom schema."""
    prefix = "multi_hop_" if is_multi else "single_hop_"
    suffix = "_multi" if mode == "multi-choice" else ""
    template = getattr(stage_cfg, f"{prefix}system_prompt{suffix}")

    schema_spec = getattr(stage_cfg, "question_schema", None)
    schema_class = load_schema_from_spec(schema_spec, mode)
    return build_system_prompt(template, schema_class)


def _validate_mode(mode: str) -> str:
    """Ensure question mode is valid."""
    mode = (mode or "open-ended").strip().lower()
    if mode not in {"open-ended", "multi-choice"}:
        logger.warning(f"Invalid question_mode '{mode}', defaulting to 'open-ended'")
        return "open-ended"
    return mode


def _get_mode_from_config(stage_cfg: Any) -> str:
    """Extract and validate question_mode from stage config."""
    raw_mode = getattr(stage_cfg, "question_mode", None) or ""
    return _validate_mode(raw_mode)


def _build_and_run_inference(
    dataset: Dataset, system_msg: dict, stage_cfg: Any, builder_func: callable, step_name: str, config
) -> tuple[dict, list]:
    """Common pattern: build calls, run inference, return responses + index map."""
    if builder_func is build_single_hop_inference_calls:
        calls, index_map = builder_func(dataset, system_msg, stage_cfg, get_sampling_cfg(stage_cfg))
    else:
        calls, index_map = builder_func(dataset, system_msg, stage_cfg)
    if not calls:
        raise ValueError(f"No valid inference calls for {step_name}")

    responses = run_inference(config=config, step_name=step_name, inference_calls=calls)
    return responses, index_map


def _save_questions(rows: list[dict], config, subset: str) -> None:
    """Save question rows after deduplication."""
    if not (clean_rows := _remove_duplicate_questions(rows)):
        raise ValueError(f"No valid questions generated for {subset}")

    logger.info(f"Saving {len(clean_rows)} {subset}")
    custom_save_dataset(
        question_dataset(clean_rows),
        config=config,
        subset=subset,
        push_to_hub=config.hf_configuration.push_to_hub,
    )


def run_single_hop(config) -> None:
    """Generate single-hop questions from individual chunks."""
    with log_stage("single_hop_generation"):
        if not (stage_cfg := config.pipeline.single_hop_question_generation).run:
            logger.info("single_hop_question_generation disabled")
            return

        mode = _get_mode_from_config(stage_cfg)
        logger.info(f"Single-shot mode: {mode}")

        system_msg = {"role": "system", "content": _get_system_prompt(stage_cfg, mode)}

        with log_step("loading_dataset"):
            dataset = custom_load_dataset(config=config, subset="chunked")
            logger.debug(f"Loaded {len(dataset) if dataset else 0} documents")

        with log_step("generating_questions"):
            responses, index_map = _build_and_run_inference(
                dataset,
                system_msg,
                stage_cfg,
                build_single_hop_inference_calls,
                "single_hop_question_generation",
                config,
            )

        with log_step("saving_questions"):
            rows = parse_single_hop_responses(responses, index_map, stage_cfg)
            _save_questions(rows, config, "single_hop_questions")


def run_multi_hop(config) -> None:
    """Generate multi-hop questions."""
    stage_cfg = config.pipeline.multi_hop_question_generation
    if not stage_cfg.run:
        logger.info("Multi-hop question generation disabled")
        return

    mode = _get_mode_from_config(stage_cfg)
    system_msg = {"role": "system", "content": _get_system_prompt(stage_cfg, mode, is_multi=True)}

    chunked_ds = custom_load_dataset(config=config, subset="chunked")
    logger.info(f"Loaded {len(chunked_ds)} documents for multi-hop")

    _process_questions(
        chunked_ds, "multi_hop_questions", system_msg, stage_cfg, config, "multi_hop_question_generation"
    )


def run_cross_document(config) -> None:
    """Generate cross-document questions."""
    stage_cfg = config.pipeline.cross_document_question_generation
    if not stage_cfg.run:
        logger.info("Cross-document question generation disabled")
        return

    mode = _get_mode_from_config(stage_cfg)
    system_msg = {"role": "system", "content": _get_system_prompt(stage_cfg, mode, is_multi=True)}

    chunked_ds = custom_load_dataset(config=config, subset="chunked")
    logger.info(f"Loaded {len(chunked_ds)} documents for cross-document")

    cross_cfg = {
        "enable": True,
        "max_combinations": stage_cfg.max_combinations,
        "chunks_per_document": stage_cfg.chunks_per_document,
        "num_docs_per_combination": stage_cfg.num_docs_per_combination,
        "random_seed": stage_cfg.random_seed,
    }

    logger.info("Starting cross-document generation")
    cross_ds = create_cross_document_dataset(chunked_ds, cross_cfg)
    _process_questions(
        cross_ds, "cross_document_questions", system_msg, stage_cfg, config, "cross_document_question_generation"
    )


def _process_questions(dataset: Dataset, label: str, system_msg: dict, stage_cfg: Any, config, step_name: str) -> None:
    """Process and save a set of questions."""
    if not dataset or len(dataset) == 0:
        raise ValueError(f"No valid input documents for {label}")

    responses, index_map = _build_and_run_inference(
        dataset, system_msg, stage_cfg, build_multi_hop_inference_calls, step_name, config
    )

    rows = parse_multi_hop_responses(responses, index_map, stage_cfg)
    source_map = {row["document_id"]: row.get("sources") for row in dataset if row.get("sources")}
    for row in rows:
        if row["document_id"] in source_map:
            row["sources"] = source_map[row["document_id"]]
    _save_questions(rows, config, label)
