"""Lossless evaluation export with explicit document/chunk provenance."""

from typing import Any

from loguru import logger

from datasets import Dataset
from yourbench.utils.dataset_engine import MissingSubsetError, custom_load_dataset, custom_save_dataset
from yourbench.utils.logging_context import log_stage
from yourbench.utils.question_models import question_dataset


QUESTION_INPUTS = (
    ("single_hop", "single_hop_subset", "single_hop_question_generation", "single_hop_questions"),
    ("multi_hop", "multi_hop_subset", "multi_hop_question_generation", "multi_hop_questions"),
    ("cross_document", "cross_doc_subset", "cross_document_question_generation", "cross_document_questions"),
)


def build_document_lookup(chunked, summarized=()) -> dict:
    """Index chunks by document, so identical chunk IDs in different documents work."""
    lookup = {}
    for row in chunked:
        chunks = {chunk["chunk_id"]: chunk["chunk_text"] for chunk in row.get("chunks", [])}
        for group in row.get("multihop_chunks", []):
            chunks.update(zip(group.get("chunk_ids", []), group.get("chunks_text", []), strict=True))
        lookup[row["document_id"]] = {
            "text": row.get("document_text", ""),
            "summary": row.get("document_summary", "") or "",
            "chunks": chunks,
        }
    for row in summarized:
        if row["document_id"] in lookup:
            lookup[row["document_id"]]["summary"] = row.get("document_summary", "") or ""
    return lookup


def make_record(row: dict[str, Any], kind: str, documents: dict, fallback_mode="open-ended") -> dict:
    """Keep the generated payload and add evaluation fields at this boundary."""
    sources = row.get("sources") or [
        {"document_id": row["document_id"], "chunk_id": cid}
        for cid in (row.get("source_chunk_ids") or ([row["chunk_id"]] if row.get("chunk_id") else []))
    ]
    if not sources:
        raise ValueError(f"Question has no source references: {row.get('question', '')!r}")
    chunks = []
    document_ids = []
    for source in sources:
        doc_id, chunk_id = source["document_id"], source["chunk_id"]
        if doc_id not in documents or chunk_id not in documents[doc_id]["chunks"]:
            raise ValueError(f"Cannot resolve source document {doc_id!r}, chunk {chunk_id!r}")
        chunks.append(documents[doc_id]["chunks"][chunk_id])
        if doc_id not in document_ids:
            document_ids.append(doc_id)
    answer = row.get("self_answer", row.get("answer", ""))
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Evaluation question has an empty answer")
    mode = row.get("question_mode") or fallback_mode
    choices = row.get("choices") or []
    if mode == "multi-choice":
        if len(answer) != 1 or not "A" <= answer <= "Z" or not 0 <= ord(answer) - 65 < len(choices):
            raise ValueError("Multiple-choice answer does not reference an existing choice")
        gold = [ord(answer) - 65]
    else:
        # LightEval gold values are indices into choices, also for free text.
        # Keeping the same type permits mixed question modes in one Arrow table.
        choices = [answer]
        gold = [0]
    return {
        **row,
        "question_mode": mode,
        "ground_truth_answer": answer,
        "gold": gold,
        "choices": choices,
        "question_category": row.get("self_assessed_question_type", row.get("question_type", "unknown")),
        "kind": kind,
        "sources": sources,
        "document_ids": document_ids,
        "chunk_ids": [source["chunk_id"] for source in sources],
        "question_generating_model": row.get("generating_model", ""),
        "chunks": chunks,
        "documents": [documents[doc_id]["text"] for doc_id in document_ids],
        "document": "\n\n".join(documents[doc_id]["text"] for doc_id in document_ids),
        "document_summary": "\n\n".join(documents[doc_id]["summary"] for doc_id in document_ids),
    }


def run(config) -> None:
    with log_stage("prepare_lighteval"):
        _run_impl(config)


def _run_impl(config) -> None:
    stage = config.pipeline.prepare_lighteval
    inputs = []
    for kind, subset_field, generation_field, default_subset in QUESTION_INPUTS:
        subset = getattr(stage, subset_field)
        generation = getattr(config.pipeline, generation_field)
        required = generation.run or subset != default_subset
        try:
            dataset = custom_load_dataset(config=config, subset=subset)
        except MissingSubsetError:
            if required:
                raise
            continue
        inputs.append((kind, getattr(generation, "question_mode", "open-ended"), dataset))
    records = []
    if any(len(dataset) for _, _, dataset in inputs):
        chunked = custom_load_dataset(config=config, subset=stage.chunked_subset)
        try:
            summarized = custom_load_dataset(config=config, subset=stage.summarized_subset)
        except MissingSubsetError:
            summarized = []
        documents = build_document_lookup(chunked, summarized)
        records = [make_record(row, kind, documents, mode) for kind, mode, dataset in inputs for row in dataset]
    # Arrow infers columns from the first record; explicitly retain the union.
    dataset = (
        question_dataset(records)
        if records
        else Dataset.from_dict({"question": [], "ground_truth_answer": [], "gold": [], "sources": []})
    )
    custom_save_dataset(
        dataset=dataset,
        config=config,
        subset=stage.output_subset,
        push_to_hub=config.hf_configuration.push_to_hub,
    )
    logger.success(f"Prepared {len(dataset)} evaluation records in {stage.output_subset}")
