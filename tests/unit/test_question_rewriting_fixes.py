"""Rewriting contract exercised through real local datasets and rendered prompts."""

import json

import pytest

from datasets import Dataset
from yourbench.pipeline import question_rewriting
from yourbench.conf.loader import resolve_config
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset


@pytest.fixture
def rewrite_case(tmp_path):
    config = resolve_config({
        "hf_configuration": {
            "hf_dataset_name": "rewrite",
            "local_dataset_dir": str(tmp_path / "data"),
            "push_to_hub": False,
        },
        "model_list": [{"model_name": "editor"}],
        "pipeline": {"question_rewriting": {}},
    })
    docs = Dataset.from_list([
        {
            "document_id": "policy-a",
            "document_text": "Refunds within 30 days",
            "document_summary": "Refund rule",
            "chunks": [{"chunk_id": "same", "chunk_text": "Refunds within 30 days"}],
        },
        {
            "document_id": "policy-b",
            "document_text": "No refunds on sale items",
            "document_summary": "Sale exception",
            "chunks": [{"chunk_id": "same", "chunk_text": "No refunds on sale items"}],
        },
    ])
    rows = [
        {
            "question": "How policies differ?",
            "self_answer": "Sale items are excluded",
            "document_id": "cross",
            "sources": [
                {"document_id": "policy-a", "chunk_id": "same"},
                {"document_id": "policy-b", "chunk_id": "same"},
            ],
            "question_mode": "open-ended",
            "custom_rubric": ["Compare exception"],
            "question_data_json": '{"question":"How policies differ?"}',
        },
        {
            "question": "Sale refundable?",
            "self_answer": "No",
            "document_id": "policy-b",
            "sources": [{"document_id": "policy-b", "chunk_id": "same"}],
            "question_mode": "open-ended",
            "custom_rubric": ["State exclusion"],
            "question_data_json": '{"question":"Sale refundable?"}',
        },
    ]
    custom_save_dataset(docs, config, subset="chunked", push_to_hub=False)
    custom_save_dataset(Dataset.from_list(rows), config, subset="cross_document_questions", push_to_hub=False)
    return config, rows


def test_rewriting_uses_owned_sources_and_preserves_payload_for_every_model(rewrite_case, monkeypatch):
    config, original = rewrite_case

    def infer(config, step, calls):
        assert step == "question_rewriting"
        assert len(calls) == 2
        first, second = [call.messages[1]["content"] for call in calls]
        assert "Refunds within 30 days" in first and "No refunds on sale items" in first
        assert '"document_id": "policy-a"' in first and '"document_id": "policy-b"' in first
        assert "No refunds on sale items" in second and "Refunds within 30 days" not in second
        assert "Sale items are excluded" in first
        assert '"question"' in calls[0].messages[0]["content"]
        return {
            model: [
                json.dumps({"question": f"{model} revised {i}?", "rationale": "Clarified wording"}) for i in range(2)
            ]
            for model in ["editor", "reviewer"]
        }

    monkeypatch.setattr(question_rewriting, "run_inference", infer)
    question_rewriting.run(config)
    result = custom_load_dataset(config, subset="cross_document_questions_rewritten")
    assert len(result) == 4
    for index, row in enumerate(result):
        source = original[index % 2]
        assert row["question"] == f"{['editor', 'reviewer'][index // 2]} revised {index % 2}?"
        assert row["original_question"] == source["question"]
        for field in ["self_answer", "sources", "custom_rubric", "question_data_json"]:
            assert row[field] == source[field]


@pytest.mark.parametrize(
    "replies",
    [
        ['{"question":"First?","question":"Second?","rationale":"Duplicate"}'] * 2,
        ['{"question":"Question?","rationale":NaN}'] * 2,
        [],
        ['{"question":"Only one?","rationale":"Edit"}'],
        ['{"question":"Good?","rationale":"Edit"}', '{"question":"","rationale":"Bad"}'],
        ['{"question":"Good?","rationale":"Edit"}', "<rewritten_question>Legacy</rewritten_question>"],
    ],
)
def test_incomplete_or_invalid_rewrite_never_saves_partial_subset(rewrite_case, monkeypatch, replies):
    config, _ = rewrite_case
    monkeypatch.setattr(question_rewriting, "run_inference", lambda *args: {"editor": replies})
    with pytest.raises(ValueError, match="rewrit"):
        question_rewriting.run(config)
    with pytest.raises(FileNotFoundError):
        custom_load_dataset(config, subset="cross_document_questions_rewritten")


def test_unresolved_sources_fail_before_request(rewrite_case, monkeypatch):
    config, rows = rewrite_case
    rows[0]["sources"][0]["chunk_id"] = "missing"
    custom_save_dataset(Dataset.from_list(rows), config, subset="cross_document_questions", push_to_hub=False)

    def unexpected(*args):
        pytest.fail("Inference must not run with missing evidence")

    monkeypatch.setattr(question_rewriting, "run_inference", unexpected)
    with pytest.raises(ValueError, match="Cannot resolve"):
        question_rewriting.run(config)
