import tiktoken
from loguru import logger

from datasets import Dataset
from yourbench.utils.chunking_utils import split_into_token_chunks
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
from yourbench.utils.parsing_engine import extract_content_from_xml_tags
from yourbench.utils.logging_context import log_step, log_stage
from yourbench.utils.inference.inference_core import InferenceCall, run_inference


def run(config) -> None:
    """Execute hierarchical document summarization."""
    with log_stage("summarization"):
        cfg = config.pipeline.summarization
        dataset = custom_load_dataset(config=config, subset="ingested")
        if not dataset:
            logger.warning("No documents to summarize")
            return

        logger.info(f"Summarizing {len(dataset)} documents")

        # Stage 1: Chunk summaries
        with log_step("chunk_summaries", num_docs=len(dataset)):
            calls, mapping = _build_calls(
                dataset, cfg.max_tokens, cfg.token_overlap, cfg.encoding_name, cfg.summarization_user_prompt
            )
            logger.debug(f"Created {len(calls)} summarization calls")
            responses = run_inference(config=config, step_name="summarization", inference_calls=calls)
            model_name, chunks_by_doc = _parse_chunk_responses(responses, mapping, len(dataset))

        # Stage 2: Combine summaries for multi-chunk docs
        with log_step("combine_summaries"):
            combine_calls, combine_indices = _build_combine_calls(chunks_by_doc, cfg.combine_summaries_user_prompt)
            if combine_calls:
                logger.debug(f"Combining summaries for {len(combine_calls)} multi-chunk documents")
                combine_responses = run_inference(
                    config=config, step_name="summarization", inference_calls=combine_calls
                )
                combined = list(combine_responses.values())[0] if combine_responses else []
                final_summaries = _merge_summaries(chunks_by_doc, combined, combine_indices)
            else:
                final_summaries = [chunks[0] if chunks else "" for chunks in chunks_by_doc]

        if any(not summary.strip() for summary in final_summaries):
            raise ValueError("Summarization returned empty or unparseable summaries")
        # Save results
        with log_step("saving_results"):
            dataset = dataset.add_column("document_summary", final_summaries)
            dataset = dataset.add_column("summarization_model", [model_name] * len(dataset))
            custom_save_dataset(
                dataset=dataset, config=config, subset="summarized", push_to_hub=config.hf_configuration.push_to_hub
            )
            logger.success(f"Summarization complete for {len(dataset)} documents")


def _build_calls(
    dataset: Dataset, max_tokens: int, overlap: int, encoding: str, prompt: str
) -> tuple[list[InferenceCall], list[tuple[int, int]]]:
    """Build inference calls for chunked summaries."""
    enc = _get_encoder(encoding)
    calls, mapping = [], []

    for i, text in enumerate(dataset["document_text"]):
        if len(enc.encode(text)) <= max_tokens:
            calls.append(_make_call(text, prompt))
            mapping.append((i, -1))
        else:
            chunks = split_into_token_chunks(text, max_tokens, overlap, encoding)
            for j, chunk in enumerate(chunks):
                calls.append(_make_call(chunk, prompt))
                mapping.append((i, j))

    return calls, mapping


def _make_call(text: str, prompt: str) -> InferenceCall:
    """Create a summarization inference call."""
    return InferenceCall(messages=[{"role": "user", "content": prompt.format(document=text)}], tags=["chunk_summary"])


def _get_encoder(encoding_name: str) -> tiktoken.Encoding:
    """Get tiktoken encoder with fallback."""
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(f"Unknown encoding '{encoding_name}': {str(e)[:60]}...")
        return tiktoken.get_encoding("cl100k_base")


def _parse_chunk_responses(responses: dict, mapping: list, num_docs: int) -> tuple[str, list[list[str]]]:
    """Parse chunk summaries back to per-document lists."""
    model_name = list(responses.keys())[0] if responses else "unknown"
    raw_responses = responses.get(model_name, [])

    # Ensure response count matches
    if len(raw_responses) < len(mapping):
        raw_responses.extend([""] * (len(mapping) - len(raw_responses)))

    # Group by document
    summaries_by_doc = [[] for _ in range(num_docs)]
    for resp, (doc_idx, _) in zip(raw_responses, mapping):
        summary = (
            extract_content_from_xml_tags(resp, "chunk_summary")
            or extract_content_from_xml_tags(resp, "final_summary")
            or ""
        )
        summaries_by_doc[doc_idx].append(summary.strip())

    return model_name, summaries_by_doc


def _build_combine_calls(summaries_by_doc: list[list[str]], prompt: str) -> tuple[list[InferenceCall], list[int]]:
    """Build calls to combine multi-chunk summaries."""
    calls, indices = [], []

    for i, summaries in enumerate(summaries_by_doc):
        valid = [s for s in summaries if s]
        if len(valid) > 1:
            bullet_list = "\n".join(f"- {s}" for s in valid)
            calls.append(
                InferenceCall(
                    messages=[{"role": "user", "content": prompt.format(chunk_summaries=bullet_list)}],
                    tags=["merge_summary"],
                )
            )
            indices.append(i)

    return calls, indices


def _merge_summaries(chunks_by_doc: list[list[str]], combined: list[str], indices: list[int]) -> list[str]:
    """Merge combined summaries into final list."""
    final = [chunks[0] if chunks else "" for chunks in chunks_by_doc]

    for resp, idx in zip(combined, indices):
        parsed = extract_content_from_xml_tags(resp, "final_summary")
        final[idx] = parsed.strip() if parsed else "No summary available."

    return final
