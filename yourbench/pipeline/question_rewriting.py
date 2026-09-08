"""
Question question_rewriting Pipeline Stage

This module implements a stage that takes generated questions (both single-hop and multi-hop)
and rewrites them using an LLM while preserving their meaning and answerability.

Features:
- Preserves question meaning and answerability
- Maintains all metadata from original questions
- Works with both single-hop and multi-hop questions
- Configurable question_rewriting instructions
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from loguru import logger

from datasets import Dataset
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
from yourbench.utils.parsing_engine import extract_content_from_xml_tags
from yourbench.utils.logging_context import log_stage
from yourbench.utils.inference.inference_core import InferenceCall, run_inference


STAGE_TAG = ["question_rewriting"]


@dataclass
class RewrittenQuestion:
    """Container for a rewritten question with metadata."""

    original_question: str
    rewritten_question: str
    question_rewriting_model: str
    question_rewriting_rationale: str


def _parse_question_rewriting_response(response: str) -> Optional[RewrittenQuestion]:
    """
    Parse the model's question_rewriting response to extract the rewritten question and rationale.

    Args:
        response: Raw model response

    Returns:
        RewrittenQuestion object or None if parsing fails
    """
    try:
        rewritten_q = extract_content_from_xml_tags(response, "rewritten_question")
        rationale = extract_content_from_xml_tags(response, "question_rewriting_rationale")

        if not rewritten_q:
            logger.warning("No rewritten question found in response")
            return None

        return RewrittenQuestion(
            original_question="",  # Will be filled by caller
            rewritten_question=rewritten_q.strip(),
            question_rewriting_model="",  # Will be filled by caller
            question_rewriting_rationale=rationale.strip() if rationale else "",
        )
    except Exception as e:
        logger.error(f"Error parsing question_rewriting response: {e}")
        return None


def _build_question_rewriting_calls(
    dataset: Dataset, system_prompt: str, user_prompt_template: str, additional_instructions: str
) -> tuple[List[InferenceCall], List[int]]:
    """
    Build inference calls for question_rewriting questions.

    Returns:
        Tuple of (inference_calls, row_indices)
    """
    calls = []
    indices = []

    for idx, row in enumerate(dataset):
        # Extract relevant fields
        question = row.get("question", "")
        if not question:
            logger.warning(f"Skipping row {idx} - no question found")
            continue

        # Get chunks based on question type
        chunks_data = row.get("chunks", "")
        if isinstance(chunks_data, list):
            # For both multihop and single-hop, if chunks are a list, join them.
            # This correctly handles empty, single-item, and multi-item lists.
            # We use map(str, ...) to safely handle any non-string elements.
            chunk_text = "\n\n".join(map(str, chunks_data))
        else:
            # For single-hop, chunks might be a single item (e.g., a string).
            # We convert it to a string. Falsy values (like None or empty string) will result in an empty string.
            chunk_text = str(chunks_data) if chunks_data else ""

        summary = row.get("document_summary", "")
        answer = row.get("self_answer", "")

        # Build user prompt
        user_prompt = user_prompt_template.format(
            original_question=question,
            answer=answer,
            chunk_text=chunk_text,
            document_summary=summary,
            additional_instructions=additional_instructions,
        )

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        calls.append(InferenceCall(messages=messages, tags=STAGE_TAG))
        indices.append(idx)

    return calls, indices


def _process_question_rewriting_responses(
    responses: Dict[str, List[str]], indices: List[int], original_dataset: Dataset
) -> List[Dict[str, Any]]:
    """
    Process model responses and create rewritten dataset rows.
    """
    rewritten_rows = []

    for model_name, model_responses in responses.items():
        if len(model_responses) != len(indices):
            logger.warning(
                f"Response count mismatch for model {model_name}. "
                f"Expected {len(indices)} but got {len(model_responses)}. "
                "This can happen if some inference calls failed. "
                "Processing the responses that were returned."
            )

        for response, dataset_idx in zip(model_responses, indices):
            if not response:
                logger.warning(f"Skipping failed or empty response for original dataset row {dataset_idx}")
                continue

            original_row = original_dataset[dataset_idx]

            # Parse the question_rewriting response
            rewritten = _parse_question_rewriting_response(response)
            if not rewritten:
                logger.warning(f"Failed to parse response for row {dataset_idx} - skipping this row")
                continue

            # Create new row with all original data plus question_rewriting info
            new_row_dict = dict(original_row)
            new_row_dict.update({
                "original_question": original_row["question"],
                "question": rewritten.rewritten_question,
                "question_rewriting_model": model_name,
                "question_rewriting_rationale": rewritten.question_rewriting_rationale,
                "raw_question_rewriting_response": response,
            })

            # Ensure question_mode is present (required by QuestionRow but may be missing from older datasets)
            if "question_mode" not in new_row_dict:
                new_row_dict["question_mode"] = "open-ended"  # Default for older datasets

            rewritten_rows.append(new_row_dict)

    return rewritten_rows


def _process_question_type(
    config,
    question_type: str,
    load_subset: str,
    save_subset: str,
    system_prompt: str,
    user_prompt_template: str,
    additional_instructions: str,
) -> bool:
    """Rewrite an available question subset; preserve all payload and provenance fields."""
    try:
        dataset = custom_load_dataset(config=config, subset=load_subset)
    except FileNotFoundError:
        return False
    if not len(dataset):
        raise ValueError(f"Cannot rewrite empty subset '{load_subset}'")
    calls, indices = _build_question_rewriting_calls(
        dataset, system_prompt, user_prompt_template, additional_instructions
    )
    if len(calls) != len(dataset):
        raise ValueError(f"Subset '{load_subset}' contains rows without questions")
    responses = run_inference(config, "question_rewriting", calls)
    rewritten_rows = _process_question_rewriting_responses(responses, indices, dataset)
    if len(rewritten_rows) != len(indices) * len(responses) or not rewritten_rows:
        raise ValueError(f"Incomplete or invalid rewriting responses for '{load_subset}'")
    keys = set().union(*(row.keys() for row in rewritten_rows))
    rewritten_ds = Dataset.from_list([{key: row.get(key) for key in keys} for row in rewritten_rows])
    custom_save_dataset(rewritten_ds, config, subset=save_subset, push_to_hub=config.hf_configuration.push_to_hub)
    return True


def run(config) -> None:
    from yourbench.pipeline.registry import QUESTION_SUBSETS

    stage_cfg = config.pipeline.question_rewriting
    if not stage_cfg.run:
        return
    completed = 0
    with log_stage("question_rewriting"):
        for stage, subset in QUESTION_SUBSETS.items():
            rewritten = _process_question_type(
                config,
                stage,
                subset,
                f"{subset}_rewritten",
                stage_cfg.question_rewriting_system_prompt,
                stage_cfg.question_rewriting_user_prompt,
                stage_cfg.additional_instructions,
            )
            if not rewritten and getattr(config.pipeline, stage).run:
                raise FileNotFoundError(f"Expected generated subset '{subset}' for rewriting")
            completed += rewritten
    if not completed:
        raise ValueError("No question subsets found for rewriting")
