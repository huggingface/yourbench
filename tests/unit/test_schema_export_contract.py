import json
from types import SimpleNamespace

import pytest

from datasets import Dataset
from yourbench.utils.dataset_engine import MissingSubsetError
from yourbench.utils.parsing_engine import shuffle_mcq, parse_single_hop_responses, _remove_duplicate_questions
from yourbench.pipeline.prepare_lighteval import _run_impl, make_record, build_document_lookup
from yourbench.utils.cross_document_utils import create_cross_document_dataset
from yourbench.pipeline.question_generation._core import _get_system_prompt


@pytest.mark.parametrize("mode", ["open-ended", "multi-choice"])
@pytest.mark.parametrize("multi", [False, True])
def test_default_schema_is_rendered(mode, multi):
    name = ("multi_hop_" if multi else "single_hop_") + "system_prompt" + ("_multi" if mode == "multi-choice" else "")
    stage = SimpleNamespace(**{name: "{schema_definition}"})
    prompt = _get_system_prompt(stage, mode, multi)
    assert "{schema_definition}" not in prompt
    assert "question" in prompt


def test_custom_schema_validates_and_preserves_aliases(tmp_path):
    schema = tmp_path / "schema.py"
    schema.write_text(
        "from pydantic import BaseModel\nclass DataFormat(BaseModel):\n    question: str\n    answer: str\n    difficulty: str\n    rubric: list[str]\n"
    )
    stage = SimpleNamespace(question_schema=str(schema), question_mode="open-ended", additional_instructions="")
    good = {"question": "Why?", "answer": "Because.", "difficulty": "hard", "rubric": ["Evidence"]}
    rows = parse_single_hop_responses(
        {"model": [json.dumps([good, {"question": "Bad", "answer": "x"}])]}, [(0, "doc", "chunk")], stage
    )
    assert len(rows) == 1
    assert rows[0]["difficulty"] == "hard"
    assert "estimated_difficulty" not in rows[0]
    assert rows[0]["question_data"] == good
    assert rows[0]["sources"] == [{"document_id": "doc", "chunk_id": "chunk"}]
    exported = make_record(
        rows[0], "single_hop", {"doc": {"text": "Source", "summary": "", "chunks": {"chunk": "Source"}}}
    )
    assert exported["rubric"] == ["Evidence"]


def test_cross_document_preserves_real_ids_and_chunk_ownership():
    docs = Dataset.from_list([
        {
            "document_id": doc_id,
            "document_text": text,
            "chunks": [{"chunk_id": "1", "chunk_text": text}],
            "multihop_chunks": [{"chunk_ids": ["1"], "chunks_text": [text]}],
        }
        for doc_id, text in [("a/b", "First"), ("a.b", "Second")]
    ])
    cross = create_cross_document_dataset(
        docs, {"max_combinations": 1, "chunks_per_document": 1, "num_docs_per_combination": [2, 2], "random_seed": 42}
    )[0]
    row = {
        "question": "Compare",
        "self_answer": "Different",
        "sources": cross["sources"],
        "document_id": cross["document_id"],
    }
    exported = make_record(row, "cross_document", build_document_lookup(docs))
    assert exported["document_ids"] == ["a/b", "a.b"]
    assert exported["chunks"] == ["First", "Second"]


def test_custom_mcq_count_and_invalid_answer(tmp_path):
    schema = tmp_path / "mcq.py"
    schema.write_text(
        "from pydantic import BaseModel\nclass DataFormat(BaseModel):\n    question: str\n    answer: str\n    choices: list[str]\n"
    )
    stage = SimpleNamespace(question_schema=str(schema), question_mode="multi-choice")
    good = {"question": "Which?", "answer": "C", "choices": ["One", "Two", "Three"]}
    rows = parse_single_hop_responses({"m": [json.dumps([good, {**good, "answer": "D"}])]}, [(0, "d", "c")], stage)
    assert len(rows) == 1
    assert len(rows[0]["choices"]) == 3
    assert rows[0]["choices"][ord(rows[0]["answer"]) - 65].endswith("Three")
    with pytest.raises(ValueError, match="existing choice"):
        shuffle_mcq({**good, "answer": "Z"})


def test_dedup_preserves_numbers_and_operators():
    rows = [{"question": text} for text in ["Year 2025?", "Year 2026?", "a + b?", "a - b?", " YEAR 2025? "]]
    assert len(_remove_duplicate_questions(rows)) == 4


def test_export_union_and_empty_subset(monkeypatch):
    import yourbench.pipeline.prepare_lighteval as export

    pipeline = SimpleNamespace(
        prepare_lighteval=SimpleNamespace(
            single_hop_subset="single_hop_questions",
            multi_hop_subset="multi_hop_questions",
            cross_doc_subset="cross_document_questions",
            chunked_subset="chunked",
            summarized_subset="summarized",
            output_subset="custom_output",
        ),
        **{
            name: SimpleNamespace(run=False, question_mode="open-ended")
            for name in [
                "single_hop_question_generation",
                "multi_hop_question_generation",
                "cross_document_question_generation",
            ]
        },
    )
    config = SimpleNamespace(pipeline=pipeline, hf_configuration=SimpleNamespace(push_to_hub=False))
    subsets = {}
    saved = []

    def load(**kwargs):
        if kwargs["subset"] not in subsets:
            raise MissingSubsetError(kwargs["subset"])
        return subsets[kwargs["subset"]]

    monkeypatch.setattr(export, "custom_load_dataset", load)
    monkeypatch.setattr(export, "custom_save_dataset", lambda **kwargs: saved.append(kwargs))
    _run_impl(config)
    assert saved[-1]["subset"] == "custom_output"
    first = {"question": "Q", "self_answer": "A", "document_id": "d", "chunk_id": "c"}
    subsets.update(
        single_hop_questions=[first, {**first, "question": "Other", "custom_rubric": ["Yes"]}],
        chunked=[{"document_id": "d", "chunks": [{"chunk_id": "c", "chunk_text": "Source"}]}],
    )
    _run_impl(config)
    assert saved[-1]["dataset"][1]["custom_rubric"] == ["Yes"]
    pipeline.single_hop_question_generation.run = True
    del subsets["single_hop_questions"]
    with pytest.raises(FileNotFoundError):
        _run_impl(config)


def test_mixed_modes_have_compatible_arrow_gold_and_unresolved_sources_fail():
    documents = {"d": {"text": "Source", "summary": "", "chunks": {"c": "Source"}}}
    base = {"question": "Why?", "self_answer": "Explanation", "document_id": "d", "chunk_id": "c"}
    free = make_record(base, "single_hop", documents)
    mcq = make_record(
        {**base, "self_answer": "B", "choices": ["One", "Two"], "question_mode": "multi-choice"},
        "single_hop",
        documents,
    )
    dataset = Dataset.from_list([free, mcq])
    assert dataset[0]["gold"] == [0]
    assert dataset[0]["choices"] == ["Explanation"]
    assert dataset[1]["gold"] == [1]
    with pytest.raises(ValueError, match="Cannot resolve source"):
        make_record({**base, "chunk_id": "missing"}, "single_hop", documents)
    with pytest.raises(ValueError, match="existing choice"):
        make_record(
            {**base, "self_answer": "Z", "question_mode": "multi-choice", "choices": ["One", "Two"]},
            "single_hop",
            documents,
        )


def test_response_alignment_and_default_invalid_payloads():
    stage = SimpleNamespace(question_mode="open-ended")
    with pytest.raises(ValueError, match="Response count"):
        parse_single_hop_responses({"m": []}, [(0, "d", "c")], stage)
    rows = parse_single_hop_responses(
        {"m": [json.dumps([{"question": "Q", "answer": ""}, {"question": "Q"}])]}, [(0, "d", "c")], stage
    )
    assert rows == []


def test_cross_document_accepts_one_chunk_per_document():
    docs = Dataset.from_list([
        {"document_id": doc_id, "chunks": [{"chunk_id": "1", "chunk_text": doc_id}]} for doc_id in ["one", "two"]
    ])
    cross = create_cross_document_dataset(
        docs, {"max_combinations": 1, "chunks_per_document": 1, "num_docs_per_combination": [2, 2]}
    )
    assert len(cross) == 1
    assert cross[0]["sources"] == [{"document_id": "one", "chunk_id": "1"}, {"document_id": "two", "chunk_id": "1"}]


def test_cross_generation_to_export_with_two_short_documents(monkeypatch):
    from yourbench.conf.loader import resolve_config
    from yourbench.pipeline.question_generation import _core

    config = resolve_config({
        "model_list": [{"model_name": "local-test"}],
        "pipeline": {"cross_document_question_generation": {"run": True, "max_combinations": 1}},
    })
    docs = Dataset.from_list([
        {
            "document_id": doc_id,
            "document_text": text,
            "chunks": [{"chunk_id": "1", "chunk_text": text}],
            "multihop_chunks": [],
        }
        for doc_id, text in [("policy/old", "Refunds take five days."), ("policy.new", "Refunds take two days.")]
    ])
    monkeypatch.setattr(_core, "custom_load_dataset", lambda **kwargs: docs)
    saved = []
    monkeypatch.setattr(_core, "custom_save_dataset", lambda dataset, **kwargs: saved.append(dataset))

    def inference(**kwargs):
        assert len(kwargs["inference_calls"]) == 1
        call = kwargs["inference_calls"][0]
        assert "five days" in call.messages[1]["content"]
        assert "two days" in call.messages[1]["content"]
        return {
            "local-test": [
                json.dumps([
                    {"question": "How did refund time change?", "answer": "It decreased from five days to two."}
                ])
            ]
        }

    monkeypatch.setattr(_core, "run_inference", inference)
    _core.run_cross_document(config)
    record = make_record(saved[0][0], "cross_document", build_document_lookup(docs))
    assert record["document_ids"] == ["policy/old", "policy.new"]
    assert record["chunks"] == ["Refunds take five days.", "Refunds take two days."]


@pytest.mark.parametrize("mode", ["", "   ", None])
def test_legacy_blank_mode_defaults_to_open_ended(mode):
    rows = parse_single_hop_responses(
        {"m": ['[{"question":"Q?","answer":"A"}]']}, [(0, "d", "c")], {"question_mode": mode}
    )
    assert rows[0]["question_mode"] == "open-ended"


def test_cross_identity_distinguishes_delimiters_and_evidence():
    docs = Dataset.from_list([
        {
            "document_id": doc_id,
            "chunks": [{"chunk_id": "1", "chunk_text": "first"}, {"chunk_id": "2", "chunk_text": "second"}],
        }
        for doc_id in ["a", "b_c", "a_b", "c"]
    ])
    config = {"max_combinations": 6, "chunks_per_document": 1, "num_docs_per_combination": [2, 2]}
    combinations = create_cross_document_dataset(docs, config)
    assert len(set(combinations["document_id"])) == 6
    by_documents = {
        tuple(sorted(row["cross_document_metadata"]["source_documents"])): row["document_id"] for row in combinations
    }
    assert by_documents[("a", "b_c")] != by_documents[("a_b", "c")]
    selections = {}
    for seed in range(4):
        row = create_cross_document_dataset(docs.select([0, 1]), {**config, "random_seed": seed})[0]
        selections[tuple(source["chunk_id"] for source in row["sources"])] = row["document_id"]
    assert len(selections) > 1
    assert len(set(selections.values())) == len(selections)


def test_schema_conflict_is_clear_and_does_not_expose_values():
    from yourbench.utils.question_models import question_dataset

    with pytest.raises(ValueError, match="Question schema conflict.*rubric") as caught:
        question_dataset([{"rubric": "private-payload"}, {"rubric": ["private-other"]}])
    assert "private" not in str(caught.value)
    assert caught.value.__suppress_context__
    compatible = question_dataset([{"rubric": {"score": 1}}, {"rubric": {"score": 2}, "extra": ["ok"]}])
    assert compatible[1]["extra"] == ["ok"]


def test_custom_payload_cannot_override_execution_metadata(tmp_path):
    schema = tmp_path / "metadata.py"
    schema.write_text(
        "from pydantic import BaseModel\nclass DataFormat(BaseModel):\n    question: str\n    answer: str\n    document_id: str\n    source_chunk_ids: list[str]\n    additional_instructions: str\n    generating_model: str\n    raw_response: str\n"
    )
    candidate = {
        "question": "Q",
        "answer": "A",
        "document_id": "fake",
        "source_chunk_ids": ["fake"],
        "additional_instructions": "fake",
        "generating_model": "fake",
        "raw_response": "fake",
    }
    rows = parse_single_hop_responses(
        {"real": [json.dumps([candidate])]},
        [(0, "real-doc", "real-chunk")],
        {"question_schema": str(schema), "additional_instructions": ""},
    )
    row = rows[0]
    assert row["document_id"] == "real-doc"
    assert row["generating_model"] == "real"
    assert "source_chunk_ids" not in row
    assert row.get("additional_instructions", "") == ""
    assert row["question_data"] == candidate
    assert row["raw_response"] != "fake"


def test_custom_field_meanings_survive_generation_and_export(tmp_path):
    schema = tmp_path / "semantics.py"
    schema.write_text(
        "from pydantic import BaseModel\nclass DataFormat(BaseModel):\n"
        "    question: str\n    answer: str\n    reasoning: list[str]\n"
        "    difficulty: str\n    complexity: dict[str, str]\n"
    )
    payload = {
        "question": "What is the runtime?",
        "answer": "Linear.",
        "reasoning": ["Inspect the loop", "Count iterations"],
        "difficulty": "requires proof, not a numeric rating",
        "complexity": {"runtime": "O(n)", "space": "O(1)"},
    }
    row = parse_single_hop_responses(
        {"m": [json.dumps([payload])]}, [(0, "d", "c")], {"question_schema": str(schema)}
    )[0]
    exported = make_record(
        row, "single_hop", {"d": {"text": "A single loop.", "summary": "", "chunks": {"c": "A single loop."}}}
    )
    for field, expected in payload.items():
        assert exported[field] == expected
    assert "estimated_difficulty" not in exported
    assert "thought_process" not in exported
    assert exported["question_data"] == payload
