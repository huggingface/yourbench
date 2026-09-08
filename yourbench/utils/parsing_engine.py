import re
import json
import math
import random
import hashlib
from typing import Any

from loguru import logger
from pydantic import TypeAdapter, ValidationError

from yourbench.utils.schema_loader import load_schema_from_spec


def _unique_object(pairs):
    """Duplicate keys are ambiguous even though Python's JSON decoder accepts them."""
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON keys")
    return result


def _reject_constant(value):
    raise ValueError("Non-finite JSON numbers are unsupported")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite JSON numbers are unsupported")
    return number


def decode_response_json(raw_response: str) -> Any:
    """Decode one JSON value, optionally in a whole fence or legacy output_json envelope.

    Prose, duplicate keys, non-finite numbers and concatenated payloads are rejected.
    JSON handles escaping and nesting; we never search inside a malformed response.
    """
    if not isinstance(raw_response, str):
        raise ValueError("Expected one JSON response")
    text = raw_response.strip()
    envelope = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```|<output_json>(.*?)</output_json>", text, re.DOTALL)
    if envelope:
        text = next(group for group in envelope.groups() if group is not None)
    try:
        return json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_constant, parse_float=_finite_float
        )
    except ValueError:
        raise ValueError("Expected one JSON response; output was invalid or ambiguous") from None


def parse_qa_pairs_from_response(raw_response: str) -> list[dict[str, Any]]:
    """Require an array of objects; never salvage objects from an invalid container."""
    try:
        return TypeAdapter(list[dict[str, Any]]).validate_python(decode_response_json(raw_response), strict=True)
    except ValidationError:
        raise ValueError("Expected a JSON array of question objects") from None


def _config_value(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


EXECUTION_FIELDS = {
    "document_id",
    "chunk_id",
    "source_chunk_ids",
    "sources",
    "question_data",
    "generating_model",
    "raw_response",
    "additional_instructions",
    "original_question",
    "question_rewriting_model",
    "question_rewriting_rationale",
    "raw_question_rewriting_response",
}


def _question_record(candidate, schema, mode, source, model, raw, instructions, multi_hop):
    """Validate the declared schema and attach execution facts without guessing aliases."""
    validated = schema.model_validate(candidate).model_dump(mode="json")
    for field in ("question", "answer"):
        if not isinstance(validated.get(field), str) or not validated[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    payload = shuffle_mcq(validated) if mode == "multi-choice" else validated
    _, document_id, chunk_ids = source[:3]
    ids = chunk_ids if multi_hop else [chunk_ids]
    return {
        **{key: value for key, value in payload.items() if key not in EXECUTION_FIELDS},
        "question": payload["question"].strip(),
        "answer": payload["answer"].strip(),
        "self_answer": payload["answer"].strip(),
        "self_assessed_question_type": payload.get("question_type", ""),
        "question_mode": mode,
        "document_id": document_id,
        "source_chunk_ids" if multi_hop else "chunk_id": chunk_ids,
        "additional_instructions": instructions,
        "generating_model": model,
        "raw_response": raw,
        "question_data": validated,
        "sources": [{"document_id": document_id, "chunk_id": cid} for cid in ids],
    }


def _parse_responses(responses, index_map, stage_cfg, *, multi_hop=False):
    """Decode each response and validate candidates against the selected schema."""
    mode = (_config_value(stage_cfg, "question_mode", "open-ended") or "open-ended").strip().lower() or "open-ended"
    schema = load_schema_from_spec(_config_value(stage_cfg, "question_schema"), mode)
    instructions = _config_value(stage_cfg, "additional_instructions", "")
    rows = []
    for model, replies in responses.items():
        if len(replies) != len(index_map):
            raise ValueError(f"Response count for {model}: {len(replies)}; expected {len(index_map)}")
        for index, (raw, source) in enumerate(zip(replies, index_map, strict=True)):
            for candidate in parse_qa_pairs_from_response(raw):
                try:
                    rows.append(_question_record(candidate, schema, mode, source, model, raw, instructions, multi_hop))
                except (ValidationError, ValueError) as error:
                    logger.warning(f"Rejected question from {model} at response {index}: {type(error).__name__}")
    return rows


def parse_single_hop_responses(responses, index_map, stage_cfg):
    return _parse_responses(responses, index_map, stage_cfg)


def parse_multi_hop_responses(responses, index_map, stage_cfg):
    return _parse_responses(responses, index_map, stage_cfg, multi_hop=True)


def shuffle_mcq(question_dict: dict) -> dict:
    """Shuffle option indices deterministically, preserving duplicate option identity."""
    result = dict(question_dict)
    choices = result.get("choices", [])
    answer = str(result.get("answer", "")).strip().upper()
    if not isinstance(choices, list) or not 2 <= len(choices) <= 26:
        raise ValueError("Multiple-choice questions require 2 to 26 choices")
    if len(answer) != 1 or not "A" <= answer <= "Z" or ord(answer) - ord("A") >= len(choices):
        raise ValueError("Multiple-choice answer must reference an existing choice")
    if not all(isinstance(choice, str) for choice in choices):
        raise ValueError("Choices must be strings")
    raw_choices = [re.sub(r"^\s*(?:\([A-Z]\)|[A-Z][.)])\s*", "", choice) for choice in choices]
    order = list(range(len(choices)))
    seed = int(hashlib.sha256(repr((raw_choices, answer)).encode()).hexdigest(), 16)
    random.Random(seed).shuffle(order)
    result["choices"] = [f"({chr(65 + i)}) {raw_choices[source]}" for i, source in enumerate(order)]
    result["answer"] = chr(65 + order.index(ord(answer) - 65))
    return result


def _remove_duplicate_questions(rows: list[dict]) -> list[dict]:
    """
    Removes duplicate question entries based on case-folded question text.
    Whitespace is collapsed; meaningful numbers and punctuation are preserved
    The original question format is preserved in the output.
    """
    seen_questions = set()
    deduped_rows = []

    for row in rows:
        question = row.get("question")
        if question is None:
            deduped_rows.append(row)
            continue

        # Normalize for deduplication
        norm_question = " ".join(question.casefold().split())

        if norm_question not in seen_questions:
            seen_questions.add(norm_question)
            deduped_rows.append(row)

    removed = len(rows) - len(deduped_rows)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate questions. Final count: {len(deduped_rows)}")
    else:
        logger.info(f"No duplicate questions detected. Final count: {len(deduped_rows)}")

    return deduped_rows
