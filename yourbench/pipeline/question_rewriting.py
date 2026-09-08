"""Rewrite questions against their actual source passages, preserving original records."""

import json
from typing import Annotated

from pydantic import BaseModel, ValidationError, StringConstraints

from yourbench.pipeline.registry import QUESTION_SUBSETS
from yourbench.utils.dataset_engine import MissingSubsetError, custom_load_dataset, custom_save_dataset
from yourbench.utils.parsing_engine import decode_response_json
from yourbench.utils.logging_context import log_stage
from yourbench.utils.question_models import question_dataset
from yourbench.pipeline.prepare_lighteval import make_record, build_document_lookup
from yourbench.utils.inference.inference_core import InferenceCall, run_inference


class RewriteResponse(BaseModel):
    model_config = {"extra": "forbid"}
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    rationale: str


def _build_question_rewriting_calls(dataset, stage, documents):
    calls = []
    for row in dataset:
        if not row["question"].strip():
            raise ValueError("Cannot rewrite an empty question")
        context = make_record(row, "rewriting", documents)
        content = stage.question_rewriting_user_prompt.format(
            original_question=row["question"],
            answer=context["ground_truth_answer"],
            choices=json.dumps(row.get("choices") or [], ensure_ascii=False),
            chunk_text=json.dumps(
                [{**source, "text": text} for source, text in zip(context["sources"], context["chunks"], strict=True)],
                ensure_ascii=False,
            ),
            document_summary=context["document_summary"],
            additional_instructions=stage.additional_instructions,
        )
        calls.append(
            InferenceCall(
                messages=[
                    {"role": "system", "content": stage.question_rewriting_system_prompt},
                    {"role": "user", "content": content},
                ],
                tags=["question_rewriting"],
            )
        )
    return calls


def _process_question_rewriting_responses(responses, original_dataset):
    if not responses:
        raise ValueError("No rewriting responses")
    rows = []
    for model, replies in responses.items():
        if len(replies) != len(original_dataset):
            raise ValueError(f"Incomplete rewriting batch: expected {len(original_dataset)}, received {len(replies)}")
        for row, raw in zip(original_dataset, replies, strict=True):
            try:
                rewritten = RewriteResponse.model_validate(decode_response_json(raw), strict=True)
            except (ValidationError, ValueError):
                raise ValueError("Invalid rewriting response: expected JSON question and rationale") from None
            rows.append({
                **row,
                "original_question": row["question"],
                "question": rewritten.question,
                "question_rewriting_model": model,
                "question_rewriting_rationale": rewritten.rationale,
                "raw_question_rewriting_response": raw,
            })
    return rows


def run(config) -> None:
    stage = config.pipeline.question_rewriting
    with log_stage("question_rewriting"):
        documents = build_document_lookup(custom_load_dataset(config=config, subset="chunked"))
        completed = 0
        for generation, subset in QUESTION_SUBSETS.items():
            try:
                dataset = custom_load_dataset(config=config, subset=subset)
            except MissingSubsetError:
                if getattr(config.pipeline, generation).run:
                    raise
                continue
            if not len(dataset):
                raise ValueError(f"Cannot rewrite empty subset '{subset}'")
            calls = _build_question_rewriting_calls(dataset, stage, documents)
            responses = run_inference(config, "question_rewriting", calls)
            rows = _process_question_rewriting_responses(responses, dataset)
            rewritten = question_dataset(rows)
            custom_save_dataset(
                rewritten, config, subset=f"{subset}_rewritten", push_to_hub=config.hf_configuration.push_to_hub
            )
            completed += 1
        if not completed:
            raise ValueError("No question subsets found for rewriting")
