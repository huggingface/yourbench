"""Render the actual validation schema rather than maintaining a parallel type system."""

import json

from pydantic import BaseModel, TypeAdapter


def generate_schema_instructions(schema: type[BaseModel]) -> str:
    """Describe the complete output array, including nested schemas and constraints."""
    specification = TypeAdapter(list[schema]).json_schema()
    return "Return only a JSON array conforming to this JSON Schema:\n" + json.dumps(specification, indent=2)
