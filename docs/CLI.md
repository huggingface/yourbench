# YourBench CLI

## Create from a brief

```bash
yourbench create \
  "Build difficult support-policy questions about exceptions and conflicting rules" \
  --source ./docs --model YOUR_MODEL --output ./benchmark
```

YourBench interprets the brief, saves an inspectable recipe, and runs it. All dataset
artifacts stay local by default. The model can select question formats and generation
strategies; source paths, credentials, model selection and publication settings come
from your explicit options.

For an OpenAI-compatible endpoint:

```bash
yourbench create "Evaluate knowledge of our refund policy" \
  --source ./docs --model YOUR_MODEL --output ./benchmark \
  --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY
```

Set the named environment variable before running. `--api-key-env` takes its **name**,
not its value. Saved recipes contain an environment reference, never the resolved key.
Use `--provider` to select a Hugging Face inference provider. If `HF_TOKEN` is present,
it is used by default. A local compatible endpoint can omit the key flag.

| Option | Meaning |
| --- | --- |
| `--source DIR` | Existing directory of Markdown, text or PDF documents |
| `--model ID` | Required model, or set `YOURBENCH_MODEL` |
| `--output DIR` | New or empty directory outside the source tree |
| `--provider NAME` | Inference provider |
| `--base-url URL` | Compatible model endpoint |
| `--api-key-env NAME` | Environment variable containing the API key |
| `--plan-only` | Interpret and save, without running generation |

`--plan-only` makes a planning model request, so it can incur API usage. Inspect the
result, edit `config.yaml` if needed, and run it later:

```bash
yourbench create "Test difficult refund decisions" --source ./docs \
  --model YOUR_MODEL --output ./benchmark --plan-only
yourbench run ./benchmark/config.yaml
```

The output directory contains:

- `plan.json`: original brief, interpreted instructions and assumptions.
- `config.yaml`: reusable execution recipe with absolute data paths.
- `processed/`, `datasets/`, `jsonl/`: artifacts produced when generation runs.

The current pipeline supports grounded open-ended and multiple-choice questions from
single passages, multiple passages and multiple documents. It does not enforce exact
example counts or dollar budgets. The planner is instructed to report unsupported
requirements; reported unsupported requirements stop creation before generation.
Planning receives the supported document count, without document contents. Cross-document
generation requires at least two supported documents. It does not verify document content during planning. Review the saved assumptions and
recipe when these decisions matter. Model-generated questions still need quality review.

## Execute or inspect a recipe

```bash
yourbench run config.yaml             # Execute enabled stages
yourbench run config.yaml --quiet     # Minimal console output
yourbench run config.yaml --debug     # Detailed logging
yourbench validate config.yaml        # Validate configuration without generation
yourbench estimate config.yaml        # Approximate token usage; not a spending cap
yourbench stages                     # Show the registered stages
yourbench version
```

A YAML filename can also be passed directly: `yourbench config.yaml`. `run` accepts
`--no-banner`. Paths in YAML resolve relative to the configuration file's directory.
Model calls require credentials appropriate to the configured endpoint/provider.

## Start with YAML

```bash
yourbench init --output config.yaml --source ./docs --model YOUR_MODEL
```

`init` writes a local starter recipe with ingestion, summarization, chunking,
single-hop generation and evaluation preparation enabled. Set model credentials and
an endpoint/provider in the generated YAML before execution. It never calls a model.
Use `--force` to replace an existing configuration. This command is now noninteractive;
use `create` to describe the benchmark in natural language.
