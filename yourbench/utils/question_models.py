from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import field, dataclass


def force_int_in_range(value: Any, min_val: int, max_val: int) -> int:
    try:
        ivalue = int(value)
    except (ValueError, TypeError):
        ivalue = (min_val + max_val) // 2
    return max(min_val, min(ivalue, max_val))


def validate_list(some_list: list[str]) -> list[str]:
    if not isinstance(some_list, list):
        return []
    try:
        return [str(value) for value in some_list]
    except Exception:
        return []


@dataclass
class QuestionRow:
    document_id: str
    additional_instructions: str
    question: str
    self_answer: str
    estimated_difficulty: int
    self_assessed_question_type: str
    question_mode: str
    generating_model: str
    thought_process: str
    raw_response: str

    citations: List[str] = field(default_factory=list)
    choices: Optional[List[str]] = field(default_factory=list)

    chunk_id: Optional[str] = None
    source_chunk_ids: Optional[List[str]] = None

    # Fields for question rewriting
    original_question: Optional[str] = None
    question_rewriting_model: Optional[str] = None
    question_rewriting_rationale: Optional[str] = None
    raw_question_rewriting_response: Optional[str] = None

    def __post_init__(self) -> None:
        self.question = str(self.question).strip()
        self.self_answer = str(self.self_answer).strip()
        self.estimated_difficulty = force_int_in_range(self.estimated_difficulty, 1, 10)
        self.self_assessed_question_type = str(self.self_assessed_question_type).strip()
        self.thought_process = str(self.thought_process)
        self.citations = validate_list(self.citations)
        self.question_mode = str(self.question_mode).strip().lower()

        if self.question_mode == "multi-choice":
            self.choices = validate_list(self.choices)
            if not 2 <= len(self.choices) <= 26:
                raise ValueError("Multi-choice questions require 2 to 26 choices.")
        else:
            self.choices = []

        if self.chunk_id and self.source_chunk_ids:
            raise ValueError("Cannot have both chunk_id and source_chunk_ids.")
        if not self.chunk_id and not self.source_chunk_ids:
            raise ValueError("Must have either chunk_id or source_chunk_ids.")

    @property
    def answer(self) -> str:
        return self.self_answer

    @property
    def question_type(self) -> str:
        return self.self_assessed_question_type

    def is_multi_hop(self) -> bool:
        return self.source_chunk_ids is not None

    def is_single_hop(self) -> bool:
        return self.chunk_id is not None

    @classmethod
    def from_single_hop(
        cls,
        pair: Dict[str, Any],
        chunk_id: str,
        document_id: str,
        model: str,
        raw_response: str,
        additional_instructions: str = "",
    ) -> QuestionRow:
        return cls(
            chunk_id=chunk_id,
            source_chunk_ids=None,
            document_id=document_id,
            additional_instructions=additional_instructions,
            question=str(pair.get("question", "")).strip(),
            self_answer=str(pair.get("answer", "")).strip(),
            choices=pair.get("choices"),
            estimated_difficulty=force_int_in_range(pair.get("estimated_difficulty", 5), 1, 10),
            self_assessed_question_type=str(pair.get("question_type", "")).strip(),
            question_mode=str(pair.get("question_mode", "")).strip().lower(),
            generating_model=model,
            thought_process=str(pair.get("thought_process", "")),
            raw_response=raw_response,
            citations=validate_list(pair.get("citations", [])),
        )

    @classmethod
    def from_multi_hop(
        cls,
        pair: Dict[str, Any],
        source_chunk_ids: List[str],
        document_id: str,
        model: str,
        raw_response: str,
        additional_instructions: str = "",
    ) -> QuestionRow:
        return cls(
            chunk_id=None,
            source_chunk_ids=source_chunk_ids,
            document_id=document_id,
            additional_instructions=additional_instructions,
            question=str(pair.get("question", "")).strip(),
            self_answer=str(pair.get("answer", "")).strip(),
            choices=pair.get("choices"),
            estimated_difficulty=force_int_in_range(pair.get("estimated_difficulty", 5), 1, 10),
            self_assessed_question_type=str(pair.get("question_type", "")).strip(),
            question_mode=str(pair.get("question_mode", "")).strip().lower(),
            generating_model=model,
            thought_process=str(pair.get("thought_process", "")),
            raw_response=raw_response,
            citations=validate_list(pair.get("citations", [])),
        )

    def to_dict(self, format: str = "unified", exclude_empty: bool = True) -> Dict[str, Any]:
        base = {
            "document_id": self.document_id,
            "additional_instructions": self.additional_instructions,
            "question": self.question,
            "question_mode": self.question_mode,
            "self_answer": self.self_answer,
            "estimated_difficulty": self.estimated_difficulty,
            "self_assessed_question_type": self.self_assessed_question_type,
            "generating_model": self.generating_model,
            "thought_process": self.thought_process,
            "raw_response": self.raw_response,
            "citations": self.citations,
            "original_question": self.original_question,
            "question_rewriting_model": self.question_rewriting_model,
            "question_rewriting_rationale": self.question_rewriting_rationale,
            "raw_question_rewriting_response": self.raw_question_rewriting_response,
        }

        if self.question_mode == "multi-choice":
            base["choices"] = self.choices

        if format == "multi-hop":
            result = {**base, "source_chunk_ids": self.source_chunk_ids}
        elif format == "single-hop":
            result = {**base, "chunk_id": self.chunk_id}
        else:
            result = {**base, "chunk_id": self.chunk_id, "source_chunk_ids": self.source_chunk_ids}

        if exclude_empty:
            result = self._filter_empty(result)
        return result

    @staticmethod
    def _filter_empty(d: Dict[str, Any]) -> Dict[str, Any]:
        """Remove keys with empty/None values, preserving required fields."""
        # Fields that should always be included even if empty
        required_fields = {"document_id", "question", "generating_model"}
        return {k: v for k, v in d.items() if k in required_fields or (v is not None and v != "" and v != [])}


def question_dataset(rows: list[dict]):
    """Build a lossless Arrow table or report incompatible schema columns safely."""
    import pyarrow as pa

    from datasets import Dataset

    columns = dict.fromkeys(key for row in rows for key in row)
    aligned = [{key: row.get(key) for key in columns} for row in rows]
    try:
        return Dataset.from_list(aligned)
    except (pa.ArrowException, TypeError):
        conflicts = []
        for key in columns:
            try:
                pa.array([row.get(key) for row in rows])
            except (pa.ArrowException, TypeError):
                conflicts.append(key)
        names = ", ".join(conflicts) or "unknown column"
        # Arrow's exception includes example values. Do not expose raw payloads.
        raise ValueError(
            f"Question schema conflict in columns: {names}. "
            "Use compatible field types across question schemas or export them separately."
        ) from None
