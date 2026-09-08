"""Document chunking pipeline stage."""

import hashlib

import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from yourbench.utils.chunking_utils import split_into_token_chunks
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
from yourbench.utils.logging_context import log_step, log_stage


def _get_rng(seed: str) -> np.random.Generator:
    """Get deterministic RNG from string seed."""
    seed_int = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed_int)


def _chunk_text(
    text: str, doc_id: str, max_tokens: int, overlap: int = 0, encoding_name: str = "cl100k_base"
) -> list[dict]:
    """Split text into token-based chunks."""
    if not text.strip():
        return []

    chunks = split_into_token_chunks(text, max_tokens, overlap=overlap, encoding_name=encoding_name)
    return [{"chunk_id": f"{doc_id}_{i}", "chunk_text": chunk} for i, chunk in enumerate(chunks)]


def _sample_multihop_combinations(n_chunks: int, h_min: int, h_max: int, factor: int, doc_id: str) -> list[list[int]]:
    """Generate random multi-hop chunk combinations."""

    # If we have only 1 chunk, create a single-chunk combination for cross-document use
    if n_chunks == 1:
        return [[0]]

    # Original logic for multiple chunks per document
    if n_chunks < h_min or h_min > h_max or h_min <= 0:
        return []

    h_max = min(h_max, n_chunks)
    target_count = max(1, n_chunks // max(1, factor))

    # Generate combinations of different sizes
    rng = _get_rng(doc_id)
    combinations_list = []

    for size in range(h_min, h_max + 1):
        n_combos = max(1, target_count // (h_max - h_min + 1))
        if n_combos >= n_chunks:
            # If requesting more combos than possible, just take a few
            n_combos = min(5, n_chunks // size)

        # Generate unique combinations for this size
        all_indices = list(range(n_chunks))
        for _ in range(n_combos):
            if len(all_indices) >= size:
                combo = sorted(rng.choice(all_indices, size=size, replace=False))
                # Convert numpy int64 to regular int to avoid serialization issues
                combo = [int(x) for x in combo]
                combinations_list.append(combo)

    # Deduplicate
    seen = set()
    unique_combos = []
    for combo in combinations_list:
        key = tuple(combo)
        if key not in seen:
            seen.add(key)
            unique_combos.append(combo)

    result = unique_combos[:target_count]
    return result


def _process_document(row: dict, cfg) -> tuple[list[dict], list[dict]]:
    """Process a single document into chunks and multihop combinations."""
    doc_text = row.get("document_text", "")
    doc_id = row.get("document_id", f"doc_{hashlib.sha256(doc_text.encode()).hexdigest()[:16]}")

    # Create single-hop chunks
    chunks = _chunk_text(doc_text, doc_id, cfg.l_max_tokens, cfg.token_overlap, cfg.encoding_name)
    if not chunks:
        return [], []

    # Create multi-hop combinations
    combos = _sample_multihop_combinations(len(chunks), cfg.h_min, cfg.h_max, cfg.num_multihops_factor, doc_id)

    multihop_chunks = [
        {"chunk_ids": [chunks[i]["chunk_id"] for i in combo], "chunks_text": [chunks[i]["chunk_text"] for i in combo]}
        for combo in combos
    ]

    return chunks, multihop_chunks


def run(config) -> None:
    """Execute chunking pipeline stage."""

    with log_stage("chunking"):
        logger.info("Starting chunking stage...")
        cfg = config.pipeline.chunking

        # Load dataset
        with log_step("loading_dataset"):
            dataset = custom_load_dataset(config=config, subset=getattr(cfg, "input_subset", "summarized"))
            logger.info(f"Processing {len(dataset)} documents")

        # Process all documents
        all_chunks = []
        all_multihops = []

        with log_step("chunking_documents", num_docs=len(dataset)):
            for row in tqdm(dataset, desc="Chunking"):
                chunks, multihops = _process_document(row, cfg)
                all_chunks.append(chunks)
                all_multihops.append(multihops)

        # Add to dataset and save
        with log_step("saving_chunked_dataset"):
            dataset = dataset.add_column("chunks", all_chunks)
            dataset = dataset.add_column("multihop_chunks", all_multihops)
            custom_save_dataset(
                dataset=dataset, config=config, subset="chunked", push_to_hub=config.hf_configuration.push_to_hub
            )

            # Log statistics
            total_chunks = sum(len(c) for c in all_chunks)
            total_multihop = sum(len(m) for m in all_multihops)
            logger.success(f"Chunking complete: {total_chunks} chunks, {total_multihop} multihop combinations")
