# OpenAI-compatible model endpoints

YourBench sends chat-completion requests through the Hugging Face inference client. To use an OpenAI-compatible server, configure its chat API base URL and the model identifier it serves. A native API with a different request protocol needs a compatible gateway or supported inference provider.

## Natural-language CLI

```bash
yourbench create "Generate difficult policy questions with supporting quotations" \
  --source ./documents --output ./benchmark --model MODEL_ID \
  --base-url http://localhost:8000/v1
```

For an authenticated server, set your key in an environment variable and add `--api-key-env YOURBENCH_API_KEY`. This flag takes the variable's name, not its value. Saved recipes keep the environment reference. An unauthenticated local server can omit it; if `HF_TOKEN` is set, the inference client may use that as its default key.

## Reusable YAML

```yaml
hf_configuration:
  push_to_hub: false
  local_saving: true
  local_dataset_dir: ./datasets
  export_jsonl: true
  jsonl_export_dir: ./jsonl

model_list:
  - model_name: $YOURBENCH_MODEL
    base_url: $YOURBENCH_BASE_URL
    api_key: $YOURBENCH_API_KEY
    max_concurrent_requests: 4

pipeline:
  ingestion:
    source_documents_dir: ./documents
    output_dir: ./processed
  summarization: {}
  chunking: {}
  single_hop_question_generation: {}
  prepare_lighteval: {}
```

Set all three referenced environment variables before loading this recipe. Use the model name and base URL supplied by your service. For a server with no authentication, omit `api_key`. File paths resolve relative to the recipe, so it can be run from another directory.

```bash
yourbench validate config.yaml
yourbench run config.yaml
```

`run` reads model settings from YAML; it does not accept `--model`, `--base-url`, or `--model-extra-parameters` overrides. Those first two flags belong to `create`.

## Request options

Put endpoint-specific request options in `model_list[].extra_parameters`:

```yaml
model_list:
  - model_name: MODEL_ID
    base_url: http://localhost:8000/v1
    max_concurrent_requests: 4
    extra_parameters:
      temperature: 0.2
      max_tokens: 2048
```

Only use parameters supported by your endpoint and model. YourBench does not translate arbitrary provider protocols. A provider rejection stops the run; it is not treated as an empty successful response.

When using Hugging Face routing instead of a direct compatible URL, set `provider` and the appropriate key. Hub dataset publication is a separate setting and is not required for inference.

## Diagnose integration issues

- Authentication errors: verify the referenced variable is set and the key belongs to that endpoint.
- Missing model errors: use the served model identifier, including any gateway alias.
- Unsupported request options: remove or correct `extra_parameters` according to the endpoint contract.
- Context-limit errors: reduce summary or chunk sizes, keeping room for prompt and output tokens.
- Invalid JSON: check the model response contract and custom prompts. YourBench does not search malformed prose for salvageable JSON.

Retries cover transient failures such as timeouts, rate limits and server errors. Permanent request errors fail immediately. A failed call cancels sibling work in its batch and closes shared clients. See [CLI usage](CLI.md) for run artifacts and [configuration](CONFIGURATION.md) for model roles.
