<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/yourbench_banner_dark_mode.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/yourbench_banner_light_mode.svg">
  <img alt="YourBench Logo" src="docs/assets/yourbench_banner_light_mode.svg" width="50%">
</picture>

<h2>YourBench: A Dynamic Benchmark Generation Framework</h2>

<a href="https://github.com/huggingface/yourbench/stargazers">
  <img src="https://img.shields.io/github/stars/huggingface/yourbench?style=social" alt="GitHub Repo stars">
</a>

<p>
  <strong>
    [<a href="https://github.com/huggingface/yourbench">GitHub</a>] · 
    [<a href="https://huggingface.co/datasets/sumuks/tempora">Dataset</a>] · 
    [<a href="https://github.com/huggingface/yourbench/tree/main/docs">Documentation</a>] · 
    [<a href="https://arxiv.org/abs/2504.01833">Paper</a>]
  </strong>
</p>

</div>

---

Generate QA pairs and evaluation datasets from source documents. YourBench transforms PDFs, Word documents, and text into structured benchmark datasets with configurable output formats. The library is open source; hosted model calls may incur provider charges.

## Features

- **Document Ingestion** – Parse PDFs, Word docs, HTML, and text files into standardized Markdown
- **Question Generation** – Create single-hop and multi-hop questions with customizable schemas
- **Custom Output Schemas** – Define your own Pydantic models for question/answer format
- **Multi-Model Support** – Use different LLMs for different pipeline stages
- **HuggingFace Integration** – Push datasets directly to the Hub or save locally
- **Reviewable Outputs** – Source references, citation scores, and exact normalized-question deduplication

## Quick Start

Describe the evaluation you want and point YourBench at your documents:

```bash
pip install -e .
yourbench create "Test understanding of policy exceptions and difficult customer questions" \
  --source ./documents --model YOUR_MODEL_ID --output ./benchmark
```

For an OpenAI-compatible endpoint:

```bash
yourbench create "Build questions about policy exceptions" \
  --source ./documents --model YOUR_MODEL_ID --output ./benchmark \
  --base-url http://localhost:8000/v1 --api-key-env MODEL_API_KEY
```

Set `MODEL_API_KEY` in your environment, or omit `--api-key-env` for an unauthenticated local endpoint. Hugging Face providers use `HF_TOKEN` when available.

YourBench interprets the brief, saves `plan.json` and `config.yaml`, then generates local datasets and JSONL under the output directory. Add `--plan-only` to inspect the interpretation first (this still makes a model call). Rerun a saved recipe with:

```bash
yourbench run ./benchmark
yourbench inspect ./benchmark
```

The brief can specify domain, audience, language, difficulty, and question style. Exact counts, dollar budgets, conversational tasks, and executable evaluators are currently unsupported and should be reported by the planner. Generated answers still require evaluation of their quality; schema validation checks structure, not factual correctness.

Use `--max-tokens 4000 --concurrency 2` to bound each response and simultaneous requests, including planning. These are not total cost or question-count limits.

The natural-language frontend defaults to local output. YAML configurations remain supported for explicit stage/model settings. See [CLI reference](docs/CLI.md), [configuration changes](docs/CONFIGURATION.md#configuration-changes-in-the-natural-language-redesign), and [schema/export contracts](docs/CUSTOM_SCHEMAS.md#validation-and-export-contracts).

## Use from Python

```python
from yourbench import create, load_result

result = create(
    "Test understanding of policy exceptions",
    source="./documents", output="./benchmark", model="MODEL_ID",
    base_url="http://localhost:8000/v1", max_tokens=4000, concurrency=2,
)
questions = result.load_dataset()

# Later, without model credentials or another inference call:
print(load_result("./benchmark").summary())
```

For an authenticated endpoint, set the key in the environment and pass `api_key_env="MODEL_API_KEY"`. See the [Python API guide](docs/PYTHON_API.md) for planning, rerunning, reading subsets, and notebook usage.

## Installation

Requires **Python 3.12**.

```bash
# With uv (recommended)
uv pip install yourbench

# With pip
pip install yourbench
```

**From source:**

```bash
git clone https://github.com/huggingface/yourbench.git
cd yourbench
pip install -e .
```

## Usage

**Minimal config:**

```yaml
hf_configuration:
  hf_dataset_name: my-benchmark
  push_to_hub: false
  upload_card: false
  export_jsonl: true

model_list:
  - model_name: MODEL_ID
    base_url: https://api.openai.com/v1
    api_key: $OPENAI_API_KEY

pipeline:
  ingestion:
    source_documents_dir: ./my-documents
  summarization:
  chunking:
  single_hop_question_generation:
  prepare_lighteval:
```

```bash
yourbench run config.yaml
```

**With custom output schema:**

```yaml
pipeline:
  single_hop_question_generation:
    question_schema: ./my_schema.py  # Must export DataFormat class
```

```python
# my_schema.py
from pydantic import BaseModel, Field

class DataFormat(BaseModel):
    question: str = Field(description="The question")
    answer: str = Field(description="The answer")
    difficulty: str = Field(description="easy, medium, or hard")
```

## CLI Commands

YourBench provides several CLI commands:

| Command | Description |
|---------|-------------|
| `yourbench create "brief" --source DIR --model MODEL --output DIR` | Interpret an objective and generate a local benchmark |
| `yourbench run <config-or-output>` | Run enabled stages from a saved recipe |
| `yourbench inspect <config-or-output> [--json]` | Read local status and subset sizes without inference |
| `yourbench validate <config>` | Check config without running |
| `yourbench estimate <config>` | Estimate token usage |
| `yourbench init` | Generate a local starter config |
| `yourbench stages` | List available pipeline stages |
| `yourbench version` | Show version |

See [CLI Reference](./docs/CLI.md) for full documentation.

## Documentation

| Guide | Description |
|-------|-------------|
| [Python API](./docs/PYTHON_API.md) | Create, run, and read local results from Python |
| [Configuration](./docs/CONFIGURATION.md) | Full config reference with all options |
| [Custom Schemas](./docs/CUSTOM_SCHEMAS.md) | Define your own output formats |
| [How It Works](./docs/PRINCIPLES.md) | Pipeline architecture and stages |
| [CLI Reference](./docs/CLI.md) | All CLI commands and options |
| [FAQ](./docs/FAQ.md) | Common questions and troubleshooting |
| [OpenAI-Compatible Models](./docs/USING_OPENAI_COMPATIBLE_MODELS.md) | Use vLLM, Ollama, etc. |
| [Dataset Columns](./docs/DATASET_COLUMNS_DESCRIPTION.md) | Output field descriptions |
| [Academic Paper](./docs/academic/paper.pdf) | COLM 2025 submission |

## Try Online

No installation needed:

- **[Demo Space](https://huggingface.co/spaces/yourbench/demo)** – Upload a document, get a benchmark
- **[Advanced Space](https://huggingface.co/spaces/yourbench/advanced)** – Full config control in browser

## Example Configs

The `example/` folder contains ready-to-use configurations:

- `default_example/` – Basic setup with sample documents
- `harry_potter_quizz/` – Multiple-choice quiz with a replaceable sample corpus
- `custom_prompts_demo/` – Custom prompts for domain-specific questions
- `local_vllm_private_data/` – Use local models for private data
- `rich_pdf_extraction_with_gemini/` – PDF extraction using a compatible vision model

Set the endpoint variables used by the examples (the sample documents are included):

```bash
export YOURBENCH_MODEL=MODEL_ID
export YOURBENCH_BASE_URL=http://localhost:8000/v1
# For an authenticated endpoint, set YOURBENCH_API_KEY in your environment.
# For an unauthenticated local endpoint, use a nonempty placeholder:
export YOURBENCH_API_KEY=not-needed
yourbench run example/default_example/config.yaml
```

See the [examples guide](example/README.md) for the six recipes and their required model capabilities.

## API Keys

Set in environment or `.env` file:

```bash
HF_TOKEN=hf_xxx              # For Hub upload
OPENAI_API_KEY=sk-xxx        # For OpenAI models
```

Use `$VAR_NAME` in config to reference environment variables.

## Contributing

PRs welcome! Open an issue first for major changes.

## 📈 Progress

<div align="center">
  <a href="https://star-history.com/#huggingface/yourbench&Date">
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=huggingface/yourbench&type=Date">
  </a>
</div>

## 📜 License

Apache 2.0 – see [LICENSE](LICENSE).

## 📚 Citation

```bibtex
@misc{shashidhar2025yourbencheasycustomevaluation,
      title={YourBench: Easy Custom Evaluation Sets for Everyone},
      author={Sumuk Shashidhar and Clémentine Fourrier and Alina Lozovskia and Thomas Wolf and Gokhan Tur and Dilek Hakkani-Tür},
      year={2025},
      eprint={2504.01833},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.01833}
}
```
