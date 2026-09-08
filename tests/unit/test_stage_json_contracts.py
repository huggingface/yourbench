"""Actual dataset persistence and complete ordered batches for summary and generation."""

import json
from types import SimpleNamespace

import pytest

from datasets import Dataset
from yourbench.pipeline import summarization
from yourbench.conf.loader import resolve_config
from yourbench.utils.chunking_utils import ChunkSamplingConfig
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
from yourbench.utils.inference.inference_builders import (
    build_multi_hop_inference_calls,
    build_single_hop_inference_calls,
)


@pytest.fixture
def summary_case(tmp_path):
    config = resolve_config({
        "hf_configuration": {
            "hf_dataset_name": "summaries",
            "local_dataset_dir": str(tmp_path / "data"),
            "push_to_hub": False,
        },
        "model_list": [{"model_name": "summarizer"}],
        "pipeline": {"summarization": {"max_tokens": 3, "token_overlap": 0}},
    })
    dataset = Dataset.from_dict({
        "document_id": ["a", "b", "c"],
        "document_text": ["one two three four five six", "red blue", "north south east west"],
    })
    custom_save_dataset(dataset, config, subset="ingested", push_to_hub=False)
    return config


def test_hierarchical_summary_maps_interleaved_document_sizes(summary_case, monkeypatch):
    batches = []

    def infer(config, step_name, inference_calls):
        batches.append(inference_calls)
        if len(batches) == 1:
            assert len(inference_calls) == 5
            expected = ["one two three", " four five six", "red blue", "north south east", " west"]
            for call, fragment in zip(inference_calls, expected, strict=True):
                assert call.messages[0]["content"].endswith(fragment)
                assert '"summary"' in call.messages[0]["content"]
            return {"summarizer": [json.dumps({"summary": f"part {i}"}) for i in range(5)]}
        assert len(inference_calls) == 2
        assert "part 0\n\npart 1" in inference_calls[0].messages[0]["content"]
        assert "part 3\n\npart 4" in inference_calls[1].messages[0]["content"]
        return {"summarizer": ['{"summary":"Combined A"}', '{"summary":"Combined C"}']}

    monkeypatch.setattr(summarization, "run_inference", infer)
    summarization.run(summary_case)
    result = custom_load_dataset(summary_case, subset="summarized")
    assert result["document_id"] == ["a", "b", "c"]
    assert result["document_summary"] == ["Combined A", "part 2", "Combined C"]
    assert result["summarization_model"] == ["summarizer"] * 3


@pytest.mark.parametrize(
    "bad",
    [
        '{"summary":""}',
        "not json",
        "<final_summary>Old format</final_summary>",
        '{"summary":"first","summary":"second"}',
        '{"summary":NaN}',
    ],
)
def test_invalid_summary_does_not_create_output(summary_case, monkeypatch, bad):
    monkeypatch.setattr(summarization, "run_inference", lambda **kwargs: {"summarizer": [bad] * 5})
    with pytest.raises(ValueError, match="Invalid summarization"):
        summarization.run(summary_case)
    with pytest.raises(FileNotFoundError):
        custom_load_dataset(summary_case, subset="summarized")


def test_missing_merge_response_cannot_fallback_to_first_chunk(summary_case, monkeypatch):
    def infer(**kwargs):
        if kwargs["inference_calls"][0].tags == ["chunk_summary"]:
            return {"summarizer": ['{"summary":"part"}'] * 5}
        return {"summarizer": ['{"summary":"only first document"}']}

    monkeypatch.setattr(summarization, "run_inference", infer)
    with pytest.raises(ValueError, match="expected 2, received 1"):
        summarization.run(summary_case)
    with pytest.raises(FileNotFoundError):
        custom_load_dataset(summary_case, subset="summarized")


def test_multiple_summary_models_rejected_before_requests(summary_case, monkeypatch):
    summary_case.model_roles["summarization"] = ["first", "second"]

    def unexpected(**kwargs):
        pytest.fail("Model ambiguity should fail before a request")

    monkeypatch.setattr(summarization, "run_inference", unexpected)
    with pytest.raises(ValueError, match="exactly one"):
        summarization.run(summary_case)


def test_builders_keep_exact_source_mapping_and_literal_document_content():
    docs = Dataset.from_list([
        {
            "document_id": "a",
            "document_summary": "summary a",
            "chunks": [{"chunk_id": "c1", "chunk_text": "{literal} α"}, {"chunk_id": "c2", "chunk_text": "second"}],
            "multihop_chunks": [{"chunk_ids": ["c1", "c2"], "chunks_text": ["{literal} α", "second"]}],
        },
        {
            "document_id": "b",
            "document_summary": "summary b",
            "chunks": [{"chunk_id": "c1", "chunk_text": "other doc"}],
            "multihop_chunks": [{"chunk_ids": ["c1"], "chunks_text": ["other doc"]}],
        },
    ])
    stage = SimpleNamespace(
        single_hop_user_prompt="{title}\n{document_summary}\n{text_chunk}\n{additional_instructions}",
        multi_hop_user_prompt="{chunks}",
        additional_instructions="Evaluate exceptions",
    )
    system = {"role": "system", "content": "Generate JSON"}
    calls, mapping = build_single_hop_inference_calls(docs, system, stage, ChunkSamplingConfig())
    assert mapping == [(0, "a", "c1"), (0, "a", "c2"), (1, "b", "c1")]
    assert "{literal} α" in calls[0].messages[1]["content"]
    assert "other doc" in calls[2].messages[1]["content"]
    multi_calls, mapping = build_multi_hop_inference_calls(docs, system, stage)
    assert mapping == [(0, "a", ["c1", "c2"]), (1, "b", ["c1"])]
    assert json.loads(multi_calls[0].messages[1]["content"]) == [
        {"chunk_id": "c1", "text": "{literal} α"},
        {"chunk_id": "c2", "text": "second"},
    ]


def test_builder_rejects_bad_later_group_instead_of_saving_partial_work():
    docs = Dataset.from_list([
        {
            "document_id": "a",
            "multihop_chunks": [
                {"chunk_ids": ["c1"], "chunks_text": ["valid"]},
                {"chunk_ids": ["c2", "c3"], "chunks_text": ["missing partner"]},
            ],
        }
    ])
    with pytest.raises(ValueError, match="Invalid source group"):
        build_multi_hop_inference_calls(
            docs,
            {"role": "system", "content": "JSON"},
            SimpleNamespace(multi_hop_user_prompt="{chunks}", additional_instructions=""),
        )
