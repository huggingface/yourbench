# Custom generation prompts

The recipe loads system prompts from `custom_prompts/` and uses the included policy documents. Set endpoint variables from [the examples guide](../README.md), then run:

```bash
yourbench run example/custom_prompts_demo/config.yaml
```

Prompt paths resolve relative to the YAML file. For example:

```yaml
pipeline:
  single_hop_question_generation:
    single_hop_system_prompt: ./custom_prompts/single_shot_system_prompt.md
```

Keep the output contract when customizing prompts: return one JSON array matching the supplied schema, with verbatim source citations. The framework appends the selected schema's JSON Schema to guide the response. Prefer `additional_instructions` for small tone or topic changes; replacing a system prompt gives you responsibility for grounding and quality instructions.

See [shared templates](../prompts/) and [custom schemas](../../docs/CUSTOM_SCHEMAS.md).
