# Shared prompt templates

| File | Purpose |
| --- | --- |
| [single_shot_technical.md](single_shot_technical.md) | Technical documentation questions |
| [single_shot_business.md](single_shot_business.md) | Business decision questions |
| [multi_hop_business.md](multi_hop_business.md) | Questions joining several business passages |

Paths are relative to the YAML file. For a recipe inside `example/default_example/`:

```yaml
pipeline:
  single_hop_question_generation:
    single_hop_system_prompt: ../prompts/single_shot_technical.md
  multi_hop_question_generation:
    multi_hop_system_prompt: ../prompts/multi_hop_business.md
```

These are system prompts for question generation. Keep the output to a single JSON array matching the appended schema and quote only supplied sources. Use `additional_instructions` when a short instruction is enough.
