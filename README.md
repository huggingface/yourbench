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

Generate high-quality QA pairs and evaluation datasets from any source documents. YourBench transforms your PDFs, Word docs, and text files into structured benchmark datasets with configurable output formats. Appearing at COLM 2025. **100% free and open source.**

## Features

- **Document Ingestion** – Parse PDFs, Word docs, HTML, and text files into standardized Markdown
- **Question Generation** – Create single-hop and multi-hop questions with customizable schemas
- **Custom Output Schemas** – Define your own Pydantic models for question/answer format
- **Multi-Model Support** – Use different LLMs for different pipeline stages
- **HuggingFace Integration** – Push datasets directly to the Hub or save locally
- **Quality Filtering** – Citation scoring and deduplication built-in

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
yourbench run ./benchmark/config.yaml
```

The brief can specify domain, audience, language, difficulty, and question style. Exact counts, dollar budgets, conversational tasks, and executable evaluators are currently unsupported and should be reported by the planner. Generated answers still require evaluation of their quality; schema validation checks structure, not factual correctness.

The new frontend defaults to local output. YAML configurations remain supported for explicit stage/model settings. See [CLI reference](docs/CLI.md), [configuration changes](docs/CONFIGURATION.md#configuration-changes-in-the-natural-language-redesign), and [schema/export contracts](docs/CUSTOM_SCHEMAS.md#validation-and-export-contracts).

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

model_list:
  - model_name: openai/gpt-4o-mini
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
| `yourbench run <config>` | Run the full pipeline |
| `yourbench validate <config>` | Check config without running |
| `yourbench estimate <config>` | Estimate token usage |
| `yourbench init` | Generate a local starter config |
| `yourbench stages` | List available pipeline stages |
| `yourbench version` | Show version |

See [CLI Reference](./docs/CLI.md) for full documentation.

## Documentation

| Guide | Description |
|-------|-------------|
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
- `harry_potter_quizz/` – Generate quiz questions from books
- `custom_prompts_demo/` – Custom prompts for domain-specific questions
- `local_vllm_private_data/` – Use local models for private data
- `rich_pdf_extraction_with_gemini/` – LLM-based PDF extraction for charts/figures

Run any example:

```bash
yourbench run example/default_example/config.yaml
```

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
