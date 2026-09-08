"""Adversarial response tests exercise JSON boundaries, not extraction heuristics."""

import json

import pytest

from yourbench.utils.parsing_engine import (
    shuffle_mcq,
    decode_response_json,
    parse_single_hop_responses,
    parse_qa_pairs_from_response,
)


@pytest.mark.parametrize(
    "wrapper", [lambda x: x, lambda x: f"```json\n{x}\n```", lambda x: f"<output_json>{x}</output_json>"]
)
@pytest.mark.parametrize("ascii_only", [True, False])
def test_escaping_unicode_and_nested_brackets_round_trip(wrapper, ascii_only):
    payload = [
        {
            "question": 'What does "[\\]" mean? 日本語 🤖',
            "answer": "literal </output_json> and ``` delimiters",
            "evidence": [{"quotes": ['a]b{c"d', "line\nbreak"]}],
        }
    ]
    encoded = json.dumps(payload, ensure_ascii=ascii_only)
    assert parse_qa_pairs_from_response("\n " + wrapper(encoded) + " \n") == payload


@pytest.mark.parametrize(
    "response",
    [
        'preface [{"question":"Q","answer":"A"}]',
        '[{"question":"Q","answer":"A"}] trailing',
        '[{"question":"Q"}][{"question":"Other"}]',
        "```json\n[]\n```\n```json\n[]\n```",
        "<output_json>[]</output_json><output_json>[]</output_json>",
        '{"questions":[{"question":"Q","answer":"A"}]}',
        '[[{"question":"Q","answer":"A"}]]',
        '["bad", {"question":"Q","answer":"A"}]',
        '[{"question":"Q","question":"Other"}]',
        '[{"question":"Q","score":NaN}]',
        '[{"question":"Q","score":Infinity}]',
        '<output_json>[{"question":"Q"}]',
        '[{"question":"Q"}',
    ],
)
def test_invalid_or_ambiguous_response_is_never_salvaged(response):
    with pytest.raises(ValueError):
        parse_qa_pairs_from_response(response)


def test_decode_error_does_not_echo_payload():
    with pytest.raises(ValueError) as caught:
        decode_response_json('private-source-text {"secret":"value"}')
    assert "private" not in str(caught.value)
    assert "secret" not in str(caught.value)


def test_duplicate_choices_keep_original_correct_option_identity(monkeypatch):
    # Reverse a known permutation: the second identical option stays the second
    # correct identity, rather than accidentally selecting the first text match.
    monkeypatch.setattr("random.Random.shuffle", lambda self, order: order.reverse())
    original = {"question": "Q", "choices": ["(A) same", "(B) same", "(C) other"], "answer": "B"}
    result = shuffle_mcq(original)
    assert result["choices"] == ["(A) other", "(B) same", "(C) same"]
    assert result["answer"] == "B"
    assert original["choices"][0] == "(A) same"


def test_custom_nested_constraints_are_enforced_without_partial_salvage(tmp_path):
    schema = tmp_path / "schema.py"
    schema.write_text(
        'from pydantic import BaseModel, Field\nclass Evidence(BaseModel):\n    quote: str = Field(pattern="^source:")\nclass DataFormat(BaseModel):\n    question: str\n    answer: str\n    evidence: list[Evidence] = Field(min_length=2)\n'
    )
    good = {"question": "Q", "answer": "A", "evidence": [{"quote": "source: one"}, {"quote": "source: two"}]}
    bad = {**good, "evidence": [{"quote": "unsupported"}, {"quote": "source: two"}]}
    rows = parse_single_hop_responses(
        {"m": [json.dumps([bad, good])]}, [(0, "d", "c")], {"question_schema": str(schema)}
    )
    assert len(rows) == 1
    assert rows[0]["question_data"] == good


@pytest.mark.parametrize("response", ['{"score":1e400}', '{"nested":[{"score":-1e400}]}', "[1e400]", "[[-1e400]]"])
def test_overflowing_json_numbers_are_rejected_at_every_depth(response):
    with pytest.raises(ValueError, match="invalid or ambiguous"):
        decode_response_json(response)
