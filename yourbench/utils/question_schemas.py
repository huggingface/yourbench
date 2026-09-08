"""Default Pydantic question schemas for YourBench.

These schemas define the expected output format from LLMs during question
generation. Users can provide custom schemas via config to modify the output.
"""

from typing import Literal

from pydantic import Field, BaseModel


# Valid question types for open-ended and multi-choice modes
OPEN_ENDED_QUESTION_TYPES = Literal[
    "analytical",
    "application-based",
    "clarification",
    "counterfactual",
    "conceptual",
    "true-false",
    "factual",
    "open-ended",
    "false-premise",
    "edge-case",
]

MULTI_CHOICE_QUESTION_TYPES = Literal[
    "analytical",
    "application-based",
    "clarification",
    "counterfactual",
    "conceptual",
    "true-false",
    "factual",
    "false-premise",
    "edge-case",
]


class OpenEndedQuestion(BaseModel):
    """Default schema for open-ended questions generated from documents."""

    thought_process: str = Field(
        default="", description="Explain why this question effectively tests understanding of the document content."
    )
    question_type: OPEN_ENDED_QUESTION_TYPES = Field(
        default="factual", description="The type of question that best categorizes this entry."
    )
    question: str = Field(
        description="The question text. Do not include meta-references like 'according to the text'."
    )
    answer: str = Field(description="Complete, accurate answer to the question.")
    estimated_difficulty: int = Field(
        default=5, ge=1, le=10, description="Difficulty rating from 1 (easiest) to 10 (hardest)."
    )
    citations: list[str] = Field(
        default_factory=list, description="Exact quotes from the source text that support the answer."
    )


class MultiChoiceQuestion(BaseModel):
    """Default schema for multiple-choice questions generated from documents."""

    thought_process: str = Field(
        default="", description="Explain why this question effectively tests understanding of the document content."
    )
    question_type: MULTI_CHOICE_QUESTION_TYPES = Field(
        default="factual", description="The type of question that best categorizes this entry."
    )
    question: str = Field(
        description="The question text. Do not include meta-references like 'according to the text'."
    )
    choices: list[str] = Field(
        min_length=4,
        max_length=4,
        description="Exactly 4 answer choices formatted as '(A) text', '(B) text', '(C) text', '(D) text'.",
    )
    answer: str = Field(
        pattern=r"^[A-D]$",
        description="The correct answer letter (A, B, C, or D).",
    )
    estimated_difficulty: int = Field(
        default=5, ge=1, le=10, description="Difficulty rating from 1 (easiest) to 10 (hardest)."
    )
    citations: list[str] = Field(
        default_factory=list, description="Exact quotes from the source text that support the correct answer."
    )


# Mapping from mode to default schema
DEFAULT_SCHEMAS: dict[str, type[BaseModel]] = {
    "open-ended": OpenEndedQuestion,
    "multi-choice": MultiChoiceQuestion,
}


def get_default_schema(question_mode: str) -> type[BaseModel]:
    """Get the default question schema for a given mode."""
    mode = question_mode.strip().lower()
    if mode not in DEFAULT_SCHEMAS:
        raise ValueError(f"Unknown question mode: {mode}. Must be 'open-ended' or 'multi-choice'.")
    return DEFAULT_SCHEMAS[mode]
