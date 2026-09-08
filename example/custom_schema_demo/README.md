# Custom Pydantic schema

This recipe loads `schemas/technical_qa.py`, whose `DataFormat` adds constrained difficulty labels, prerequisites and key concepts to question, answer and citations. It uses the included policy corpus for a small reproducible input; replace that corpus with technical documents for domain-specific use.

Set endpoint variables from [the examples guide](../README.md), then run:

```bash
yourbench run example/custom_schema_demo/config.yaml
```

The schema file path is relative to this recipe:

```yaml
pipeline:
  single_hop_question_generation:
    question_schema: ./schemas/technical_qa.py
```

Switch to `./schemas/educational_assessment.py` for learning objectives and Bloom-level metadata. Custom fields keep their exact names and values through export when their types are compatible. For example, `difficulty: intermediate` remains a string and is not converted to a numeric `estimated_difficulty`.

Schema files are executable Python: use code you trust. Each must define a Pydantic `DataFormat` class with nonempty string `question` and `answer` fields. MCQ schemas also require `choices` and a valid answer letter. See [custom schemas](../../docs/CUSTOM_SCHEMAS.md) for validation and export behavior.
