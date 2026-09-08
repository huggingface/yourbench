# Runnable examples

These recipes use the small fictional policy corpus in [sample_documents](sample_documents/) and save datasets and JSONL under each example's `output/` directory. They do not publish to the Hub. Source and output paths resolve relative to each configuration file.

Install the checkout with `pip install -e .`. Set `YOURBENCH_MODEL`, `YOURBENCH_BASE_URL` and `YOURBENCH_API_KEY` for an endpoint you can access, then run:

```bash
yourbench validate example/default_example/config.yaml
yourbench run example/default_example/config.yaml
```

`validate` does not generate questions. Running calls your model and may incur charges. Model availability and output counts depend on your endpoint and responses.

| Example | Purpose |
| --- | --- |
| [default_example](default_example/) | Minimal document-to-question pipeline |
| [harry_potter_quizz](harry_potter_quizz/) | Multiple-choice format; uses the policy corpus unless replaced |
| [custom_prompts_demo](custom_prompts_demo/) | Custom generation instructions |
| [local_vllm_private_data](local_vllm_private_data/) | Local compatible endpoint without a required API key |
| [rich_pdf_extraction_with_gemini](rich_pdf_extraction_with_gemini/) | PDF page-image ingestion with a vision-capable model |
| [custom_schema_demo](custom_schema_demo/) | Custom Pydantic fields and constraints |

Directory names from older tutorials are retained, but no example requires a particular commercial model. For your own data, replace `source_documents_dir`; for publication, explicitly configure Hub saving. The PDF demo uses [sample_pdf](sample_pdf/) instead of the Markdown corpus.

See [CLI usage](../docs/CLI.md), [model endpoints](../docs/USING_OPENAI_COMPATIBLE_MODELS.md), and [shared prompts](prompts/).
