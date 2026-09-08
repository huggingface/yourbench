import re
import json
import random
import hashlib
from typing import Any
from dataclasses import fields

from loguru import logger
from pydantic import ValidationError

from yourbench.utils.schema_loader import load_schema_from_spec
from yourbench.utils.question_models import QuestionRow


# Field alias mapping for custom schemas
# Maps custom field names to standard QuestionRow field names
FIELD_ALIASES: dict[str, str] = {
    "reasoning": "thought_process",
    "explanation": "thought_process",
    "rationale": "thought_process",
    "thinking": "thought_process",
    "difficulty": "estimated_difficulty",
    "complexity": "estimated_difficulty",
}

# Maps string difficulty values to numeric values
DIFFICULTY_MAPPINGS: dict[str, int] = {
    "beginner": 2,
    "easy": 2,
    "intermediate": 5,
    "medium": 5,
    "advanced": 7,
    "hard": 7,
    "expert": 9,
    "very hard": 9,
}


def _normalize_pair_fields(pair: dict) -> dict:
    """Map custom schema fields to standard QuestionRow fields."""
    normalized = dict(pair)
    for old, new in FIELD_ALIASES.items():
        if old in normalized and new not in normalized:
            normalized[new] = normalized.pop(old)
    # Convert string difficulty to numeric
    if "estimated_difficulty" in normalized and isinstance(normalized["estimated_difficulty"], str):
        normalized["estimated_difficulty"] = DIFFICULTY_MAPPINGS.get(normalized["estimated_difficulty"].lower(), 5)
    return normalized


def _is_valid_question_list(items: list) -> bool:
    """Check if a list appears to contain question dicts.

    Returns True if at least one item is a dict with a 'question' key.
    This helps filter out arrays of strings or other non-question data.
    """
    if not items:
        return False
    for item in items:
        if isinstance(item, dict) and ("question" in item or "answer" in item):
            return True
    return False


# JSON parsing functions


def _attempt_json_parse(json_str: str) -> Any:
    """
    Attempt to parse a JSON string. Return parsed object if success,
    or None if parsing fails.
    """
    try:
        return json.loads(json_str)
    except Exception:
        return None


def _maybe_strip_triple_backticks(text_in: str) -> str:
    """
    Removes triple backticks (``` or ```json) from the beginning
    and end of a string, if present.
    """
    if not text_in or not isinstance(text_in, str):
        return ""
    try:
        pattern = r"^\s*```(?:json)?\s*([\s\S]*?)\s*```$"
        match = re.match(pattern, text_in)
        if match:
            return match.group(1)
    except Exception as e:
        logger.debug(f"Error stripping backticks: {e}")
    return text_in


def _best_effort_json_extract(full_text: str) -> list[str]:
    """
    Collect bracket-delimited substrings that might be valid JSON.
    Uses balanced bracket parsing to correctly extract nested JSON structures.
    Returns a list of candidates (which may be empty).
    """
    if not full_text or not isinstance(full_text, str):
        return []
    candidates = []
    try:
        i = 0
        while i < len(full_text):
            if full_text[i] in "[{":
                start = i
                open_char = full_text[i]
                close_char = "]" if open_char == "[" else "}"
                depth = 1
                in_string = False
                escape_next = False
                j = i + 1

                while j < len(full_text):
                    ch = full_text[j]

                    if escape_next:
                        escape_next = False
                        j += 1
                        continue

                    if ch == "\\" and in_string:
                        escape_next = True
                    elif ch == '"':
                        in_string = not in_string
                    elif not in_string:
                        if ch in "[{":
                            depth += 1
                        elif ch in "]}":
                            depth -= 1
                            if depth == 0:
                                candidate = full_text[start : j + 1].strip()
                                if candidate:
                                    candidates.append(candidate)
                                i = j
                                break
                    j += 1
            i += 1
    except Exception as e:
        logger.debug(f"Error in best-effort JSON extraction: {e}")
    return candidates


def _extract_tag_content(text: str, tag: str) -> str:
    """
    Extract text enclosed in <tag>...</tag> from the given string.
    Returns an empty string if the tag is not found.
    """
    try:
        pattern = rf"<{tag}\s*>([\s\S]*?)</{tag}>"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    except Exception as e:
        logger.debug(f"Error extracting tag content for '{tag}': {e}")
    return ""


def extract_content_from_xml_tags(full_content, xml_tag):
    # This function extracts the content between the XML tags
    # It uses regex to find the content and includes error handling

    # Define the regex patterns to match the content
    pattern_with_closing_tag = f"<{xml_tag}>(.*?)</{xml_tag}>"
    pattern_without_closing_tag = f"<{xml_tag}>(.*)"

    try:
        # First, try to find matches with both opening and closing tags
        matches_with_closing = re.findall(pattern_with_closing_tag, full_content, re.DOTALL)
        if matches_with_closing:
            return matches_with_closing[0].strip()

        # If no matches found, try to find content with only opening tag
        matches_without_closing = re.findall(pattern_without_closing_tag, full_content, re.DOTALL)
        if matches_without_closing:
            return matches_without_closing[0].strip()

        # If still no matches found, return an empty string
        return ""

    except Exception as extraction_error:
        logger.error(f"Error extracting content from XML tags: {extraction_error}")
        return ""


def parse_qa_pairs_from_response(raw_response: str) -> list[dict[str, Any]]:
    """
    Attempt to parse question-answer pairs from a raw LLM response.

    The function searches in this priority order:
        1. <output_json>...</output_json> tags.
        2. ```json fenced code blocks.
        3. Best-effort bracket-based extraction.

    If any candidate JSON is found, it attempts to parse it. If parsing
    succeeds and yields a list, it returns that list. Otherwise, it
    returns an empty list.

    Even if this returns an empty list, callers are expected to store
    the raw response (e.g., so the pipeline does not lose data).

    Args:
        raw_response (str): The complete raw response string from the model.

    Returns:
        A list of dict objects, each presumably containing
        question-answer information. If no valid parse is found,
        an empty list is returned.
    """
    if not raw_response or not isinstance(raw_response, str):
        return []

    # 1) Check for <output_json>...</output_json>
    extracted_json_str = _extract_tag_content(raw_response, "output_json")
    if extracted_json_str.strip():
        possible_parsed = _attempt_json_parse(_maybe_strip_triple_backticks(extracted_json_str))
        if isinstance(possible_parsed, list) and _is_valid_question_list(possible_parsed):
            return possible_parsed

    # 2) Check for ```json fenced code block
    fence_pattern = r"```json\s*([\s\S]*?)\s*```"
    fence_match = re.search(fence_pattern, raw_response)
    if fence_match:
        possible_parsed = _attempt_json_parse(fence_match.group(1).strip())
        if isinstance(possible_parsed, list) and _is_valid_question_list(possible_parsed):
            return possible_parsed

    # 3) Best-effort bracket-based extraction
    bracket_candidates = _best_effort_json_extract(raw_response)
    logger.debug(f"Found {len(bracket_candidates)} bracket candidates")
    for candidate in bracket_candidates:
        possible_parsed = _attempt_json_parse(candidate)
        if possible_parsed is not None:
            logger.debug(f"Parsed JSON type: {type(possible_parsed).__name__}")
            # Handle {"questions": [...]} wrapper format from some models
            if isinstance(possible_parsed, dict):
                logger.debug(f"Dict keys: {list(possible_parsed.keys())[:5]}")
                for key in ["questions", "question_list", "qa_pairs", "pairs", "data"]:
                    if key in possible_parsed and isinstance(possible_parsed[key], list):
                        if _is_valid_question_list(possible_parsed[key]):
                            logger.debug(f"Extracted questions from '{key}' key")
                            return possible_parsed[key]
        if isinstance(possible_parsed, list) and _is_valid_question_list(possible_parsed):
            return possible_parsed

    # If no valid parse was found, return empty.
    logger.debug("No valid question list found in response")
    return []


# QA response parsing utils


def _config_value(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def _parse_responses(responses, index_map, stage_cfg, *, multi_hop=False):
    """Validate model payloads once and attach authoritative execution metadata."""
    mode = (_config_value(stage_cfg, "question_mode", "open-ended") or "open-ended").strip().lower() or "open-ended"
    schema_spec = _config_value(stage_cfg, "question_schema")
    schema = load_schema_from_spec(schema_spec, mode)
    rows = []
    for model, replies in responses.items():
        if len(replies) != len(index_map):
            raise ValueError(f"Response count for {model}: {len(replies)}; expected {len(index_map)}")
        for index, raw in enumerate(replies):
            for candidate in parse_qa_pairs_from_response(raw):
                if not isinstance(candidate, dict):
                    continue
                try:
                    # Default legacy responses may omit descriptive metadata; the
                    # actual question, answer and MCQ constraints remain required.
                    payload = dict(candidate)
                    if not schema_spec:
                        payload = _normalize_pair_fields(payload)
                        for key, default in {
                            "thought_process": "",
                            "question_type": "factual",
                            "estimated_difficulty": 5,
                            "citations": [],
                        }.items():
                            payload.setdefault(key, default)
                    validated = schema.model_validate(payload).model_dump(mode="json")
                    if not str(validated.get("question", "")).strip():
                        raise ValueError("Question must be nonempty")
                    if not str(validated.get("answer", "")).strip():
                        raise ValueError("Answer must be nonempty")
                    pair = _normalize_pair_fields(validated)
                    if mode == "multi-choice":
                        pair = shuffle_mcq(pair)
                    pair["question_mode"] = mode
                    _, document_id, chunk_ids = index_map[index][:3]
                    factory = QuestionRow.from_multi_hop if multi_hop else QuestionRow.from_single_hop
                    record = factory(
                        pair,
                        chunk_ids,
                        document_id,
                        model,
                        raw,
                        _config_value(stage_cfg, "additional_instructions", ""),
                    ).to_dict(format="multi-hop" if multi_hop else "single-hop")
                    # Preserve schema fields verbatim, including aliases and nested
                    # structures, while execution metadata cannot be overwritten.
                    reserved = {field.name for field in fields(QuestionRow)}
                    record = {**{key: value for key, value in validated.items() if key not in reserved}, **record}
                    record["answer"] = pair["answer"]
                    if mode == "multi-choice":
                        record["choices"] = pair["choices"]
                    record["question_data"] = validated
                    record["sources"] = [
                        {"document_id": document_id, "chunk_id": cid}
                        for cid in (chunk_ids if multi_hop else [chunk_ids])
                    ]
                    rows.append(record)
                except (ValidationError, ValueError, TypeError) as error:
                    logger.warning(f"Rejected question from {model} at response {index}: {error}")
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
