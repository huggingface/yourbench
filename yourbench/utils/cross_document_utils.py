"""Cross-document dataset utilities for multi-document question generation."""

import json
import math
import random
import hashlib
from typing import Any, Set, List, TypeVar, Sequence

from loguru import logger

from datasets import Dataset


T = TypeVar("T")


def _unrank_comb(n: int, k: int, rank: int) -> List[int]:
    """
    Return the k-combination of [0, n) corresponding to the given rank
    in colexicographic (colex) order.

    Colexicographic order sorts combinations by increasing values of the
    largest element, then second largest, and so on (i.e., right-to-left
    significance).

    Parameters
    ----------
    n : int
        Size of the universe (exclusive upper bound of elements).
    k : int
        Size of each combination.
    rank : int
        Integer in the range [0, C(n, k)) specifying the position of the combination
        in colexicographic order.

    Returns
    -------
    List[int]
        A strictly increasing list of k integers in the range [0, n),
        representing the rank-th combination in colex order.

    Raises
    ------
    ValueError
        If k is not in [0, n] or rank is not in [0, C(n, k)).
    """
    if not 0 <= k <= n:
        raise ValueError(f"require 0 ≤ k ≤ n, got k={k}, n={n}")
    max_rank = math.comb(n, k)
    if not 0 <= rank < max_rank:
        raise ValueError(f"rank must be in [0,{max_rank - 1}], got {rank}")

    combo: List[int] = []
    for i in range(k, 0, -1):
        # largest c such that C(c, i) ≤ rank (binary search)
        lo, hi = i - 1, n - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if math.comb(mid, i) <= rank:
                lo = mid
            else:
                hi = mid - 1
        combo.append(lo)
        rank -= math.comb(lo, i)
        n = lo  # next digit must be < current one
    combo.reverse()
    return combo


def _floyd_sample_indices(total: int, sample_size: int, *, rng: random.Random | None = None) -> Set[int]:
    """Select sample_size unique integers ∈ [0, total) uniformly at random"""
    if sample_size > total:
        raise ValueError("sample_size cannot exceed total")
    if rng is None:
        rng = random

    chosen: Set[int] = set()
    for j in range(total - sample_size, total):
        t = rng.randrange(0, j + 1)
        chosen.add(t if t not in chosen else j)
    return chosen


def _sample_exact_combinations(
    objects: Sequence[T], k: int, N: int, *, rng: random.Random | None = None
) -> List[List[T]]:
    """Draw N distinct k-combinations from objects exactly uniformly.

    The function first uses Bob Floyd to pick N distinct ranks in
    `[0, C(n,k))` (where `n = len(objects)`), then converts each rank to its
    combination via `_unrank_comb`, and finally maps the integer indices back
    to the actual objects.
    """
    n = len(objects)
    if not 0 <= k <= n:
        raise ValueError("require 0 ≤ k ≤ n")
    total = math.comb(n, k)
    if N > total:
        raise ValueError("cannot request more combinations than exist")
    if rng is None:
        rng = random

    ranks = _floyd_sample_indices(total, N, rng=rng)
    combos: List[List[T]] = []
    for r in ranks:
        idxs = _unrank_comb(n, k, r)
        combos.append([objects[i] for i in idxs])
    return combos


def create_cross_document_dataset(dataset: Dataset, stage_cfg: dict[str, Any]) -> Dataset:
    """Creates a cross-document Dataset by combining multi-hop chunks from different documents.

    Args:
        dataset: A HuggingFace Dataset where each row may contain a 'multihop_chunks' list and 'document_summary'.
        stage_cfg: Stage-specific config containing:
            - 'max_combinations' (int): The maximum number of cross-document combinations to generate.
            - 'chunks_per_document' (int): The number of chunks to sample from each document.
            - 'num_docs_per_combination' (List[int]): A list [min, max] specifying the range of documents to combine.
            - 'random_seed' (int): Seed for the random number generator.

    Returns:
        A new Dataset with cross-document combinations, preserving a similar schema but with an aggregated summary.
    """
    # Extract and validate configuration
    max_combinations = (
        stage_cfg.get("max_combinations", 100) if isinstance(stage_cfg, dict) else stage_cfg.max_combinations
    )
    chunks_per_document = (
        stage_cfg.get("chunks_per_document", 1) if isinstance(stage_cfg, dict) else stage_cfg.chunks_per_document
    )
    num_docs_range = (
        stage_cfg.get("num_docs_per_combination", [2, 5])
        if isinstance(stage_cfg, dict)
        else stage_cfg.num_docs_per_combination
    )
    random_seed = stage_cfg.get("random_seed", 42) if isinstance(stage_cfg, dict) else stage_cfg.random_seed

    min_docs, max_docs = num_docs_range

    # Cross-document reasoning only needs evidence from each document, not
    # an existing within-document multi-hop group.
    docs = []
    for idx, row in enumerate(dataset):
        valid_chunks = [
            chunk
            for chunk in (row.get("multihop_chunks") or [])
            if isinstance(chunk, dict) and chunk.get("chunk_ids") and chunk.get("chunks_text")
        ]
        if not valid_chunks:
            valid_chunks = [
                {"chunk_ids": [chunk["chunk_id"]], "chunks_text": [chunk["chunk_text"]]}
                for chunk in (row.get("chunks") or [])
                if chunk.get("chunk_id") and chunk.get("chunk_text")
            ]
        if valid_chunks:
            docs.append({
                "document_id": row.get("document_id", f"doc_{idx}"),
                "original_index": idx,
                "document_summary": row.get("document_summary", ""),
                "multihop_chunks": valid_chunks,
            })

    if len(docs) < min_docs:
        logger.warning(f"Found only {len(docs)} document(s) with valid 'multihop_chunks'. Need at least {min_docs}.")
        return Dataset.from_list([])

    logger.info(f"Found {len(docs)} documents with valid multihop_chunks")

    # Initialize random number generator
    rng = random.Random(random_seed)

    # Generate combinations efficiently using exact uniform sampling
    cross_rows = []

    # Strategy: distribute combinations across different group sizes
    # Calculate total possible combinations across all group sizes
    total_possible_combinations = sum(
        math.comb(len(docs), k) for k in range(min_docs, min(max_docs + 1, len(docs) + 1))
    )

    logger.info(f"Total possible combinations: {total_possible_combinations}")

    # Cap max_combinations to what's actually possible
    actual_max_combinations = min(max_combinations, total_possible_combinations)

    # For each possible number of documents to combine
    for num_docs_to_combine in range(min_docs, min(max_docs + 1, len(docs) + 1)):
        # Calculate how many combinations we can make with this number of docs
        combinations_for_this_size = math.comb(len(docs), num_docs_to_combine)

        if combinations_for_this_size == 0:
            continue

        # Determine how many combinations to generate for this group size
        remaining_combinations = actual_max_combinations - len(cross_rows)
        if remaining_combinations <= 0:
            break

        # Simple proportional allocation
        proportion = combinations_for_this_size / total_possible_combinations
        target_for_this_size = max(1, int(proportion * actual_max_combinations))
        actual_for_this_size = min(target_for_this_size, combinations_for_this_size, remaining_combinations)

        if actual_for_this_size <= 0:
            continue

        logger.info(f"Generating {actual_for_this_size} combinations with {num_docs_to_combine} documents")

        # Use exact uniform sampling to get distinct combinations
        try:
            doc_combinations = _sample_exact_combinations(docs, num_docs_to_combine, actual_for_this_size, rng=rng)
        except ValueError as e:
            logger.warning(f"Could not generate combinations for {num_docs_to_combine} docs: {e}")
            continue

        # Process each combination
        for doc_group in doc_combinations:
            sampled_chunks_from_group = []
            sources = []
            doc_ids_for_tracing = []

            # Sample chunks from each document in the group
            for doc in doc_group:
                doc_ids_for_tracing.append(doc["document_id"])

                if not doc["multihop_chunks"]:
                    continue

                # Sample the specified number of chunks from this document
                num_chunks_to_sample = min(chunks_per_document, len(doc["multihop_chunks"]))
                if num_chunks_to_sample == 1:
                    sampled_chunks = [rng.choice(doc["multihop_chunks"])]
                else:
                    sampled_chunks = rng.sample(doc["multihop_chunks"], num_chunks_to_sample)

                sampled_chunks_from_group.extend(sampled_chunks)
                sources.extend(
                    {"document_id": doc["document_id"], "chunk_id": cid}
                    for chunk in sampled_chunks
                    for cid in chunk["chunk_ids"]
                )

            # Validation: ensure we have chunks from the expected number of documents
            # (This addresses the original validation mismatch issue)
            expected_total_chunks = len(doc_group) * chunks_per_document
            if len(sampled_chunks_from_group) < len(doc_group):
                logger.warning(f"Insufficient chunks sampled from document group {doc_ids_for_tracing}")
                continue

            # Combine chunks from all documents in the group
            combined_ids = []
            combined_texts = []

            for chunk in sampled_chunks_from_group:
                chunk_ids = chunk.get("chunk_ids", [])
                chunk_texts = chunk.get("chunks_text", [])

                if isinstance(chunk_ids, list):
                    combined_ids.extend(chunk_ids)
                else:
                    combined_ids.append(chunk_ids)

                if isinstance(chunk_texts, list):
                    combined_texts.extend(chunk_texts)
                else:
                    combined_texts.append(chunk_texts)

            # Create combined multihop chunk
            combined_multihop_chunk = {
                "chunk_ids": combined_ids,
                "chunks_text": combined_texts,
            }

            # Combine document summaries
            doc_summaries = [
                doc["document_summary"]
                for doc in doc_group
                if doc.get("document_summary") and doc["document_summary"].strip()
            ]

            combined_summary = ""
            if doc_summaries:
                header = "Here are the summaries from the various documents involved in the chunking:"
                summary_bullets = "\n".join(f"- {s}" for s in doc_summaries)
                combined_summary = f"{header}\n\n{summary_bullets}"

            # Create readable and collision-resistant ID
            doc_ids_sorted = sorted(doc_ids_for_tracing)
            # Hash structured identity: delimiter-joined IDs can collide, and
            # different evidence selections must not share a provenance key.
            identity = json.dumps(
                {"document_ids": doc_ids_sorted, "sources": sources},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            cross_doc_id = f"cross_{hashlib.sha256(identity.encode()).hexdigest()}"

            # Add comprehensive metadata for traceability
            metadata = {
                "source_documents": doc_ids_sorted,
                "num_source_docs": len(doc_group),
                "chunks_per_doc": chunks_per_document,
                "total_chunks_sampled": len(sampled_chunks_from_group),
                "source_indices": sorted([doc["original_index"] for doc in doc_group]),
                "generation_method": "exact_uniform_sampling",
            }

            cross_rows.append({
                "document_id": cross_doc_id,
                "document_summary": combined_summary,
                "chunks": [],  # keep consistent with original schema
                "multihop_chunks": [combined_multihop_chunk],
                "sources": sources,
                "cross_document_metadata": metadata,  # add traceability
            })

    if not cross_rows:
        logger.warning("No cross-document combinations were generated.")
        return Dataset.from_list([])

    logger.info(f"Generated {len(cross_rows)} cross-document combinations successfully")
    return Dataset.from_list(cross_rows)
