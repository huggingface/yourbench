# Custom Question Schemas

Define custom output formats for generated questions using Pydantic models.

## Quick Start

**1. Create a schema file:**

```python
# schemas/my_schema.py
from pydantic import BaseModel, Field

class DataFormat(BaseModel):
    question: str = Field(description="The question text")
    answer: str = Field(description="Complete answer")
    citations: list[str] = Field(description="Source quotes from the document")
```

**2. Reference it in your config:**

```yaml
pipeline:
  single_hop_question_generation:
    question_schema: ./schemas/my_schema.py
```

**3. Run the pipeline:**

```bash
yourbench run config.yaml
```

Generated questions will use your schema format.

## Schema Requirements

| Requirement | Details |
|-------------|--------|
| Class name | Must be `DataFormat` |
| Base class | Must inherit from `pydantic.BaseModel` |
| Field descriptions | Use `Field(description=...)` to guide the LLM |
| File extension | Must be `.py` |

## Field Types

Supported Pydantic field types:

```python
from typing import Literal
from pydantic import BaseModel, Field

class DataFormat(BaseModel):
    # Basic types
    question: str = Field(description="Question text")
    answer: str = Field(description="Answer text")
    
    # Lists
    citations: list[str] = Field(description="Source quotes")
    keywords: list[str] = Field(description="Key terms")
    
    # Enums / Literals
    difficulty: Literal["easy", "medium", "hard"] = Field(description="Difficulty")
    category: Literal["factual", "analytical", "conceptual"] = Field(description="Type")
    
    # Integers with constraints
    score: int = Field(ge=1, le=10, description="Quality score 1-10")
    
    # Optional fields with defaults
    notes: str = Field(default="", description="Additional notes")
```

## Field semantics

YourBench preserves custom field names and values literally. A `difficulty` string stays a string; a `reasoning` list stays a list. There is no inferred mapping to `estimated_difficulty` or `thought_process`. Use explicit Pydantic validators or aliases when your schema needs conversions. Default question schemas define their own metadata defaults.

## Example Schemas

### Technical Documentation

```python
from pydantic import BaseModel, Field
from typing import Literal

class DataFormat(BaseModel):
    question: str = Field(description="Technical question about the API/code")
    answer: str = Field(description="Detailed technical answer")
    prerequisites: list[str] = Field(description="Required knowledge")
    code_snippet: str = Field(default="", description="Code example if relevant")
    difficulty: Literal["beginner", "intermediate", "advanced"] = Field(
        description="Technical difficulty level"
    )
    citations: list[str] = Field(description="Source quotes")
```

### Educational Assessment

```python
from pydantic import BaseModel, Field
from typing import Literal

class DataFormat(BaseModel):
    question: str = Field(description="Question testing comprehension")
    answer: str = Field(description="Expected correct answer")
    bloom_level: Literal[
        "remember", "understand", "apply", "analyze", "evaluate", "create"
    ] = Field(description="Bloom's taxonomy level")
    marking_scheme: str = Field(description="How to evaluate responses")
    common_mistakes: list[str] = Field(description="Typical student errors")
    citations: list[str] = Field(description="Source material")
```

### Socratic Dialogue

```python
from pydantic import BaseModel, Field
from typing import Literal

class DataFormat(BaseModel):
    question: str = Field(description="Probing question to stimulate thinking")
    answer: str = Field(description="Ideal response demonstrating understanding")
    dialectic_goal: str = Field(description="What insight should emerge")
    follow_up_questions: list[str] = Field(description="Questions to dig deeper")
    expected_reasoning: str = Field(description="Thought process to demonstrate")
    depth_level: Literal["surface", "analytical", "philosophical"] = Field(
        description="Conceptual depth required"
    )
    citations: list[str] = Field(description="Source material")
```

### Multiple Choice

```python
from pydantic import BaseModel, Field

class DataFormat(BaseModel):
    question: str = Field(description="Question stem")
    choices: list[str] = Field(
        min_length=4, max_length=4,
        description="Four choices: (A) text, (B) text, (C) text, (D) text"
    )
    answer: str = Field(pattern=r"^[A-D]$", description="Correct letter A-D")
    distractor_explanations: list[str] = Field(
        description="Why each wrong answer is plausible but incorrect"
    )
    citations: list[str] = Field(description="Source material")
```

## Default Schema

If no `question_schema` is specified, YourBench uses:

**Open-ended mode:**
```python
class OpenEndedQuestion(BaseModel):
    thought_process: str
    question_type: Literal["analytical", "application-based", "clarification", ...]
    question: str
    answer: str
    estimated_difficulty: int  # 1-10
    citations: list[str]
```

**Multi-choice mode:**
```python
class MultiChoiceQuestion(BaseModel):
    thought_process: str
    question_type: Literal["analytical", "application-based", "clarification", ...]
    question: str
    choices: list[str]  # Exactly 4
    answer: str  # A, B, C, or D
    estimated_difficulty: int  # 1-10
    citations: list[str]
```

## Troubleshooting

**"Class 'DataFormat' not found"**
- Your schema file must export a class named exactly `DataFormat`
- Check for typos in the class name

**"Must be a Pydantic BaseModel subclass"**
- Ensure your class inherits from `pydantic.BaseModel`
- Check your Pydantic import: `from pydantic import BaseModel`

**"Schema file not found"**
- Use a relative path from the config file location
- Or use an absolute path

**LLM produces wrong format**
- Add more detailed `description` to each field
- Use `Literal` types to constrain values
- Add `additional_instructions` in the pipeline config

## Validation and export contracts

Custom `DataFormat` models are validated against each generated candidate before attaching runtime metadata. Required fields, constraints, and nested structures must validate; invalid candidates are rejected and a run with no valid questions fails. Custom schemas must expose nonempty `question` and `answer` fields. Default question schemas are also rendered into prompts automatically.

`question_data` preserves the validated original payload, including pre-shuffle multiple-choice data. Runtime fields such as source references and model identity cannot be overridden by generated fields. Additional fields survive evaluation export. Compatible schemas can be combined; conflicting types for a shared field produce an explicit schema-conflict error. Export such schemas separately or make the shared types consistent.

Evaluation `gold` is always a list of choice indices. Open-ended rows use `choices: [answer]`, `gold: [0]`, with the text also in `ground_truth_answer`. Custom multiple-choice schemas can use 2–26 choices; the default still specifies four. Canonical `sources` contains document/chunk pairs and resolves cross-document evidence without assuming globally unique chunk IDs.
