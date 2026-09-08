import time
from typing import Any, Dict, List
from contextlib import contextmanager
from dataclasses import field, dataclass

from loguru import logger

from yourbench.utils.chunking_utils import sample_multihop_groups, sample_single_hop_chunks
from yourbench.utils.inference.inference_core import InferenceCall


@dataclass
class InferenceJob:
    """Enhanced inference job with metadata tracking."""

    inference_calls: List[InferenceCall]
    job_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class BuilderMetrics:
    """Metrics for tracking inference call generation."""

    total_documents: int = 0
    total_chunks_processed: int = 0
    total_calls_generated: int = 0
    skipped_chunks: int = 0
    avg_chunk_length: float = 0.0
    processing_time: float = 0.0
    error_count: int = 0
    warnings: List[str] = field(default_factory=list)


@contextmanager
def _builder_context(builder_type: str, dataset_size: int):
    """Context manager for builder metrics and timing."""
    start_time = time.time()
    metrics = BuilderMetrics()
    logger.info(f"Building {builder_type} inference calls for {dataset_size} documents")
    try:
        yield metrics
    finally:
        metrics.processing_time = time.time() - start_time


def _log_builder_completion(metrics: BuilderMetrics, builder_type: str, extra_info: str = ""):
    """Log completion message and warnings."""
    logger.info(
        f"{builder_type} builder completed: {metrics.total_calls_generated} calls from "
        f"{metrics.total_documents} documents, {metrics.total_chunks_processed} chunks processed "
        f"{extra_info}(errors: {metrics.error_count}) in {metrics.processing_time:.2f}s"
    )
    if metrics.warnings:
        logger.warning(f"Builder warnings: {len(metrics.warnings)} total")
        for warning in metrics.warnings[:5]:
            logger.warning(f"  - {warning}")
        if len(metrics.warnings) > 5:
            logger.warning(f"  ... and {len(metrics.warnings) - 5} more warnings")


def _build_tags(base_tags: List[str], row: Dict, extra_tags: List[str] = None) -> List[str]:
    """Build tags list with common patterns."""
    tags = list(base_tags)
    if "document_type" in row:
        tags.append(f"type_{row['document_type']}")
    if extra_tags:
        tags.extend(extra_tags)
    return tags


def _create_call(messages: List[Dict], tags: List[str], stage_cfg) -> InferenceCall:
    """Create an InferenceCall with common config extraction."""
    return InferenceCall(
        messages=messages,
        tags=tags,
        temperature=getattr(stage_cfg, "temperature", None),
        max_retries=getattr(stage_cfg, "max_retries", 3),
    )


def build_single_hop_inference_calls(dataset, system_msg, stage_cfg, sampling_cfg):
    """Build single-shot inference calls with enhanced tracking."""
    calls = []
    index_map = []

    with _builder_context("single-shot", len(dataset)) as metrics:
        for idx, row in enumerate(dataset):
            try:
                metrics.total_documents += 1
                document_chunks = row.get("chunks") or []

                if not document_chunks:
                    metrics.warnings.append(f"Document {idx} has no chunks")
                    continue

                selected_chunks = sample_single_hop_chunks(document_chunks, sampling_cfg)

                for ch_idx, chunk in enumerate(selected_chunks):
                    try:
                        metrics.total_chunks_processed += 1
                        chunk_id = chunk.get("chunk_id", f"{idx}_{ch_idx}")
                        chunk_text = chunk.get("chunk_text", "")

                        if not chunk_text.strip():
                            metrics.skipped_chunks += 1
                            metrics.warnings.append(f"Empty chunk {chunk_id}")
                            continue

                        user_msg = {
                            "role": "user",
                            "content": stage_cfg.single_hop_user_prompt.format(
                                title=row.get("document_filename", f"doc_{idx}"),
                                document_summary=row.get("document_summary", ""),
                                text_chunk=chunk_text,
                                additional_instructions=stage_cfg.additional_instructions,
                            ),
                        }

                        tags = _build_tags(
                            ["single_hop_qa", f"doc_{idx}", f"chunk_{ch_idx}", f"chunk_len_{len(chunk_text)}"],
                            row,
                        )
                        call = _create_call([system_msg, user_msg], tags, stage_cfg)

                        calls.append(call)
                        index_map.append((idx, row.get("document_id", f"doc_{idx}"), chunk_id))
                        metrics.total_calls_generated += 1

                    except Exception as e:
                        metrics.error_count += 1
                        metrics.warnings.append(f"Error processing chunk {ch_idx} in document {idx}: {e}")
                        logger.warning(f"Error processing chunk {ch_idx} in document {idx}: {e}")

            except Exception as e:
                metrics.error_count += 1
                metrics.warnings.append(f"Error processing document {idx}: {e}")
                logger.error(f"Error processing document {idx}: {e}")

        _log_builder_completion(metrics, "Single-shot", f"(skipped: {metrics.skipped_chunks}) ")

    return calls, index_map


def build_multi_hop_inference_calls(dataset, system_msg, stage_cfg):
    """Build multi-hop inference calls with enhanced tracking."""
    calls = []
    index_map = []

    with _builder_context("multi-hop", len(dataset)) as metrics:
        for idx, row in enumerate(dataset):
            try:
                metrics.total_documents += 1
                multihop_chunks = row.get("multihop_chunks") or []

                if not multihop_chunks:
                    metrics.warnings.append(f"Document {idx} has no multihop chunks")
                    continue

                chunk_sampling = getattr(stage_cfg, "chunk_sampling", {})
                groups = sample_multihop_groups(multihop_chunks, chunk_sampling)

                for group_idx, group in enumerate(groups):
                    try:
                        if not isinstance(group, dict):
                            metrics.warnings.append(f"Multihop group {group_idx} in document {idx} is not a dict")
                            continue

                        chunk_ids = group.get("chunk_ids", [])
                        texts = group.get("chunks_text", [])

                        if not texts:
                            metrics.warnings.append(f"Group {group_idx} in document {idx} has empty chunks_text")
                            continue

                        metrics.total_chunks_processed += len(texts)
                        full_text = "".join([f"<text_chunk_{i}>{t}</text_chunk_{i}>\n" for i, t in enumerate(texts)])

                        user_msg = {
                            "role": "user",
                            "content": stage_cfg.multi_hop_user_prompt.format(
                                title=row.get("document_filename", f"doc_{idx}"),
                                document_summary=row.get("document_summary", ""),
                                chunks=full_text,
                                additional_instructions=stage_cfg.additional_instructions,
                            ),
                        }

                        chunk_category = (
                            "few_chunks" if len(texts) <= 2 else "medium_chunks" if len(texts) <= 5 else "many_chunks"
                        )
                        tags = _build_tags(
                            [
                                "multi_hop_qa",
                                f"doc_{idx}",
                                f"group_{group_idx}",
                                f"chunks_{len(texts)}",
                                f"total_len_{len(full_text)}",
                            ],
                            row,
                            [chunk_category],
                        )
                        call = _create_call([system_msg, user_msg], tags, stage_cfg)

                        calls.append(call)
                        index_map.append((idx, row.get("document_id", f"doc_{idx}"), chunk_ids))
                        metrics.total_calls_generated += 1

                    except Exception as e:
                        metrics.error_count += 1
                        metrics.warnings.append(f"Error processing group {group_idx} in document {idx}: {e}")
                        logger.warning(f"Error processing group {group_idx} in document {idx}: {e}")

            except Exception as e:
                metrics.error_count += 1
                metrics.warnings.append(f"Error processing document {idx}: {e}")
                logger.error(f"Error processing document {idx}: {e}")

        _log_builder_completion(metrics, "Multi-hop")

    return calls, index_map


def get_builder_performance_summary(calls: List[InferenceCall], processing_time: float) -> Dict[str, Any]:
    """Generate performance summary for builder operations."""
    if not calls:
        return {"total_calls": 0, "processing_time": processing_time}

    tag_counts = {}
    message_lengths = []

    for call in calls:
        for tag in call.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        total_length = sum(len(str(msg.get("content", ""))) for msg in call.messages)
        message_lengths.append(total_length)

    avg_message_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0

    return {
        "total_calls": len(calls),
        "processing_time": processing_time,
        "avg_message_length": avg_message_length,
        "min_message_length": min(message_lengths) if message_lengths else 0,
        "max_message_length": max(message_lengths) if message_lengths else 0,
        "tag_distribution": tag_counts,
        "calls_per_second": len(calls) / processing_time if processing_time > 0 else 0,
    }
