"""Build ordered requests and source mappings; malformed inputs fail at construction."""

import json
from typing import NamedTuple

from yourbench.utils.chunking_utils import sample_single_hop_chunks
from yourbench.utils.inference.inference_core import InferenceCall


class SourceIndex(NamedTuple):
    row_index: int
    document_id: str
    chunk_ids: str | list[str]


def build_single_hop_inference_calls(dataset, system_msg, stage_cfg, sampling_cfg):
    calls, indices = [], []
    for index, row in enumerate(dataset):
        for chunk in sample_single_hop_chunks(row["chunks"], sampling_cfg):
            if not chunk["chunk_text"].strip():
                raise ValueError(f"Empty source chunk in document {row['document_id']!r}")
            content = stage_cfg.single_hop_user_prompt.format(
                title=row.get("document_filename", row["document_id"]),
                document_summary=row.get("document_summary", ""),
                text_chunk=chunk["chunk_text"],
                additional_instructions=stage_cfg.additional_instructions,
            )
            calls.append(
                InferenceCall(messages=[system_msg, {"role": "user", "content": content}], tags=["single_hop_qa"])
            )
            indices.append(SourceIndex(index, row["document_id"], chunk["chunk_id"]))
    return calls, indices


def build_multi_hop_inference_calls(dataset, system_msg, stage_cfg):
    calls, indices = [], []
    for index, row in enumerate(dataset):
        for group in row["multihop_chunks"]:
            chunk_ids, texts = group["chunk_ids"], group["chunks_text"]
            if not chunk_ids or len(chunk_ids) != len(texts) or any(not text.strip() for text in texts):
                raise ValueError(f"Invalid source group in document {row['document_id']!r}")
            chunks = json.dumps(
                [{"chunk_id": cid, "text": text} for cid, text in zip(chunk_ids, texts, strict=True)],
                ensure_ascii=False,
            )
            content = stage_cfg.multi_hop_user_prompt.format(
                title=row.get("document_filename", row["document_id"]),
                document_summary=row.get("document_summary", ""),
                chunks=chunks,
                additional_instructions=stage_cfg.additional_instructions,
            )
            calls.append(
                InferenceCall(messages=[system_msg, {"role": "user", "content": content}], tags=["multi_hop_qa"])
            )
            indices.append(SourceIndex(index, row["document_id"], chunk_ids))
    return calls, indices
