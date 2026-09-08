"""Hierarchical summaries with one validated JSON response per input chunk."""

from typing import Annotated

from pydantic import BaseModel, ValidationError, StringConstraints

from yourbench.utils.chunking_utils import split_into_token_chunks
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
from yourbench.utils.parsing_engine import decode_response_json
from yourbench.utils.logging_context import log_stage
from yourbench.utils.inference.inference_core import InferenceCall, run_inference


class SummaryResponse(BaseModel):
    model_config = {"extra": "forbid"}
    summary: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _read_summaries(responses: dict[str, list[str]], expected: int) -> tuple[str, list[str]]:
    if len(responses) != 1:
        raise ValueError("Summarization requires exactly one model")
    model, raw = next(iter(responses.items()))
    if len(raw) != expected:
        raise ValueError(f"Incomplete summarization batch: expected {expected}, received {len(raw)}")
    try:
        return model, [SummaryResponse.model_validate(decode_response_json(text), strict=True).summary for text in raw]
    except (ValidationError, ValueError):
        raise ValueError("Invalid summarization response: expected JSON with a nonempty summary") from None


def run(config) -> None:
    with log_stage("summarization"):
        cfg = config.pipeline.summarization
        assigned = config.model_roles.get("summarization", [])
        if len(assigned) > 1:
            raise ValueError("Assign exactly one model to summarization")
        dataset = custom_load_dataset(config=config, subset="ingested")
        if not len(dataset):
            raise ValueError("No documents to summarize")
        calls, document_indices = [], []
        for index, text in enumerate(dataset["document_text"]):
            if not text.strip():
                raise ValueError(f"Cannot summarize empty document at row {index}")
            for chunk in split_into_token_chunks(text, cfg.max_tokens, cfg.token_overlap, cfg.encoding_name):
                calls.append(
                    InferenceCall(
                        messages=[{"role": "user", "content": cfg.summarization_user_prompt.format(document=chunk)}],
                        tags=["chunk_summary"],
                    )
                )
                document_indices.append(index)
        responses = run_inference(config=config, step_name="summarization", inference_calls=calls)
        model, summaries = _read_summaries(responses, len(calls))
        grouped = [[] for _ in dataset]
        for index, summary in zip(document_indices, summaries, strict=True):
            grouped[index].append(summary)
        combine_calls, combine_indices = [], []
        for index, parts in enumerate(grouped):
            if len(parts) > 1:
                combine_calls.append(
                    InferenceCall(
                        messages=[
                            {
                                "role": "user",
                                "content": cfg.combine_summaries_user_prompt.format(
                                    chunk_summaries="\n\n".join(parts)
                                ),
                            }
                        ],
                        tags=["merge_summary"],
                    )
                )
                combine_indices.append(index)
        final = [parts[0] for parts in grouped]
        if combine_calls:
            combined = run_inference(config=config, step_name="summarization", inference_calls=combine_calls)
            combine_model, merged = _read_summaries(combined, len(combine_calls))
            if combine_model != model:
                raise ValueError("Summarization model changed between batches")
            for index, summary in zip(combine_indices, merged, strict=True):
                final[index] = summary
        dataset = dataset.add_column("document_summary", final)
        dataset = dataset.add_column("summarization_model", [model] * len(dataset))
        custom_save_dataset(dataset, config, subset="summarized", push_to_hub=config.hf_configuration.push_to_hub)
