"""Request-level regression for a global brief overwhelming per-source evidence."""

import json
from types import SimpleNamespace

import pytest

from datasets import Dataset
from yourbench.conf.prompts import load_prompt_from_package
from yourbench.utils.chunking_utils import ChunkSamplingConfig
from yourbench.utils.cross_document_utils import create_cross_document_dataset
from yourbench.utils.inference.inference_builders import (
    build_multi_hop_inference_calls,
    build_single_hop_inference_calls,
)


BRIEF = "Cover returns and premium shipping; compare both policies across documents. Preserve {literal} phrasing."
RETURNS = "Unopened items may be returned within 30 days."
SHIPPING = "Premium members receive next-day delivery at no charge."


@pytest.fixture
def documents():
    return Dataset.from_list([
        {
            "document_id": doc_id,
            "document_filename": f"{doc_id}.md",
            "document_summary": summary,
            "chunks": [{"chunk_id": "chunk-0", "chunk_text": text}],
        }
        for doc_id, summary, text in [
            ("returns", "Return eligibility", RETURNS),
            ("shipping", "Premium delivery benefits", SHIPPING),
        ]
    ])


@pytest.mark.parametrize("custom_template", [False, True])
def test_global_brief_preserved_but_single_hop_evidence_stays_in_its_document(documents, custom_template):
    template = (
        "{additional_instructions}\n{text_chunk}"
        if custom_template
        else load_prompt_from_package("question_generation/single_hop_user_prompt.md")
    )
    stage = SimpleNamespace(single_hop_user_prompt=template, additional_instructions=BRIEF)
    original_system = {"role": "system", "content": "Generate a JSON array using the configured schema."}
    calls, mapping = build_single_hop_inference_calls(documents, original_system, stage, ChunkSamplingConfig())

    assert mapping == [(0, "returns", "chunk-0"), (1, "shipping", "chunk-0")]
    for call, included, excluded in [(calls[0], RETURNS, SHIPPING), (calls[1], SHIPPING, RETURNS)]:
        system, user = call.messages
        assert BRIEF in user["content"]
        assert included in user["content"]
        assert excluded not in "\n".join(message["content"] for message in call.messages)
        assert "supported entirely by the supplied text chunk" in system["content"]
        assert "not answer or citation evidence" in system["content"]
        assert (
            "Omit unsupported topics instead of creating questions answered with 'not specified' or abstention"
            in system["content"]
        )
        assert "return the empty JSON array []" in system["content"]
        assert system["content"].startswith(original_system["content"])
    assert original_system == {"role": "system", "content": "Generate a JSON array using the configured schema."}
    assert stage.additional_instructions == BRIEF


def test_cross_document_requests_combine_evidence_with_unambiguous_source_mapping(documents):
    grouped = create_cross_document_dataset(
        documents,
        {
            "max_combinations": 1,
            "chunks_per_document": 1,
            "num_docs_per_combination": [2, 2],
            "random_seed": 42,
        },
    )
    stage = SimpleNamespace(multi_hop_user_prompt="{chunks}\n{additional_instructions}", additional_instructions=BRIEF)
    system = {"role": "system", "content": "Generate JSON."}
    calls, mapping = build_multi_hop_inference_calls(grouped, system, stage)

    assert len(calls) == 1
    evidence_json, brief = calls[0].messages[1]["content"].split("\n", 1)
    evidence = json.loads(evidence_json)
    assert [chunk["text"] for chunk in evidence] == [RETURNS, SHIPPING]
    assert brief == BRIEF
    assert grouped[0]["sources"] == [
        {"document_id": "returns", "chunk_id": "chunk-0"},
        {"document_id": "shipping", "chunk_id": "chunk-0"},
    ]
    assert mapping == [(0, grouped[0]["document_id"], [chunk["chunk_id"] for chunk in evidence])]
    assert "Combine evidence from at least two supplied chunks" in calls[0].messages[0]["content"]
    assert "supported entirely by the supplied source chunks" in calls[0].messages[0]["content"]
    assert system["content"] == "Generate JSON."
