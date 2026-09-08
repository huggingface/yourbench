"""Render question generation prompts from their actual Pydantic schema."""

from pydantic import BaseModel

from yourbench.utils.schema_prompt_generator import generate_schema_instructions


def build_system_prompt(template: str, schema: type[BaseModel]) -> str:
    return template.replace("{schema_definition}", generate_schema_instructions(schema))
