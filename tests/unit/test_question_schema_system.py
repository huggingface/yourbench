"""Consolidated tests for the custom question schema system.

Covers: schema loading, prompt generation, field normalization, and default schemas.
"""

import json
import tempfile
from typing import Literal

import pytest
from pydantic import Field, BaseModel, ValidationError

from yourbench.utils.schema_loader import SCHEMA_CLASS_NAME, SchemaLoadError, load_schema_from_spec
from yourbench.utils.question_schemas import (
    OpenEndedQuestion,
    MultiChoiceQuestion,
    get_default_schema,
)
from yourbench.utils.schema_prompt_generator import generate_schema_instructions


class TestDefaultSchemas:
    """Tests for OpenEndedQuestion and MultiChoiceQuestion defaults."""

    def test_open_ended_valid(self):
        q = OpenEndedQuestion(
            thought_process="Tests core concept",
            question_type="analytical",
            question="What is X?",
            answer="X is...",
            estimated_difficulty=5,
            citations=["source"],
        )
        assert q.question_type == "analytical"

    def test_open_ended_difficulty_bounds(self):
        with pytest.raises(ValidationError):
            OpenEndedQuestion(
                thought_process="T",
                question_type="factual",
                question="Q?",
                answer="A",
                estimated_difficulty=11,
                citations=[],
            )

    def test_open_ended_invalid_type(self):
        with pytest.raises(ValidationError):
            OpenEndedQuestion(
                thought_process="T",
                question_type="invalid",
                question="Q?",
                answer="A",
                estimated_difficulty=5,
                citations=[],
            )

    def test_multi_choice_valid(self):
        q = MultiChoiceQuestion(
            thought_process="T",
            question_type="factual",
            question="Which?",
            choices=["(A) 1", "(B) 2", "(C) 3", "(D) 4"],
            answer="A",
            estimated_difficulty=3,
            citations=["s"],
        )
        assert q.answer == "A"

    def test_multi_choice_requires_four_choices(self):
        with pytest.raises(ValidationError):
            MultiChoiceQuestion(
                thought_process="T",
                question_type="factual",
                question="Q?",
                choices=["(A) 1", "(B) 2", "(C) 3"],
                answer="A",
                estimated_difficulty=5,
                citations=[],
            )

    def test_multi_choice_answer_letter_validation(self):
        with pytest.raises(ValidationError):
            MultiChoiceQuestion(
                thought_process="T",
                question_type="factual",
                question="Q?",
                choices=["(A) 1", "(B) 2", "(C) 3", "(D) 4"],
                answer="E",
                estimated_difficulty=5,
                citations=[],
            )

    def test_get_default_schema_modes(self):
        assert get_default_schema("open-ended") == OpenEndedQuestion
        assert get_default_schema("multi-choice") == MultiChoiceQuestion
        assert get_default_schema("OPEN-ENDED") == OpenEndedQuestion

    def test_get_default_schema_invalid(self):
        with pytest.raises(ValueError, match="Unknown question mode"):
            get_default_schema("invalid")


class TestSchemaLoader:
    """Tests for load_schema_from_spec."""

    def test_none_returns_default(self):
        assert load_schema_from_spec(None, "open-ended") == OpenEndedQuestion
        assert load_schema_from_spec(None, "multi-choice") == MultiChoiceQuestion

    def test_missing_file_raises(self):
        with pytest.raises(SchemaLoadError, match="not found"):
            load_schema_from_spec("/nonexistent/path.py", "open-ended")

    def test_non_python_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("not python\n")
            f.flush()
            with pytest.raises(SchemaLoadError, match="must be a Python file"):
                load_schema_from_spec(f.name, "open-ended")

    def test_missing_class_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("from pydantic import BaseModel\nclass Other(BaseModel): pass\n")
            f.flush()
            with pytest.raises(SchemaLoadError, match=f"'{SCHEMA_CLASS_NAME}' not found in"):
                load_schema_from_spec(f.name, "open-ended")

    def test_non_basemodel_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(f"class {SCHEMA_CLASS_NAME}: pass\n")
            f.flush()
            with pytest.raises(SchemaLoadError, match="must be a Pydantic"):
                load_schema_from_spec(f.name, "open-ended")

    def test_valid_custom_schema(self):
        code = f"from pydantic import BaseModel\nclass {SCHEMA_CLASS_NAME}(BaseModel):\n    question: str\n"
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(code)
            f.flush()
            schema = load_schema_from_spec(f.name, "open-ended")
            assert schema.__name__ == SCHEMA_CLASS_NAME

    def test_load_complex_schema(self):
        """Test loading a schema with multiple fields and Literal types."""
        code = f"""from typing import Literal
from pydantic import BaseModel, Field
class {SCHEMA_CLASS_NAME}(BaseModel):
    reasoning: str = Field(description="Why")
    question: str
    difficulty: Literal["easy", "hard"] = Field(description="Level")
    citations: list[str]
"""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(code)
            f.flush()
            schema = load_schema_from_spec(f.name, "open-ended")
            assert schema.__name__ == SCHEMA_CLASS_NAME
            assert "reasoning" in schema.model_fields


class TestPromptGeneration:
    def test_nested_schema_constraints_survive_prompt_rendering(self):
        class Evidence(BaseModel):
            quote: str = Field(pattern=r"^source:")
            score: float = Field(gt=0, le=1)

        class CustomQuestion(BaseModel):
            evidence: list[Evidence] = Field(min_length=2)
            category: Literal["a", "b", "c", "d", "e", "f", "g"]

        specification = json.loads(generate_schema_instructions(CustomQuestion).split("\n", 1)[1])
        assert specification["type"] == "array"
        assert specification["$defs"]["Evidence"]["properties"]["quote"]["pattern"] == "^source:"
        assert specification["$defs"]["Evidence"]["properties"]["score"]["exclusiveMinimum"] == 0
        assert specification["$defs"]["CustomQuestion"]["properties"]["evidence"]["minItems"] == 2
        assert specification["$defs"]["CustomQuestion"]["properties"]["category"]["enum"] == list("abcdefg")


class TestPromptBuilder:
    """Tests for the prompt_builder module."""

    def test_build_substitutes_schema_without_invented_examples(self):
        from yourbench.utils.prompt_builder import build_system_prompt

        result = build_system_prompt("Header\n{schema_definition}", MultiChoiceQuestion)
        assert "{schema_definition}" not in result
        assert '"minItems": 4' in result
        assert '"pattern": "^[A-D]$"' in result
        assert "output_json" not in result

    def test_build_preserves_template_without_placeholders(self):
        """build_system_prompt leaves templates without placeholders unchanged."""
        from yourbench.utils.prompt_builder import build_system_prompt

        template = "This is a static prompt with no placeholders."
        result = build_system_prompt(template, OpenEndedQuestion)

        assert result == template

    def test_build_includes_schema_fields(self):
        """build_system_prompt includes schema field names in output."""
        from yourbench.utils.prompt_builder import build_system_prompt

        template = "{schema_definition}"
        result = build_system_prompt(template, OpenEndedQuestion)

        assert "question" in result
        assert "answer" in result
        assert "citations" in result
