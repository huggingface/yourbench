# Python API

Use the same planner and runner as the CLI from a script. Python 3.12 is required. Install the checkout with `pip install -e .`.

## Generate from a brief

```python
from yourbench import create

result = create(
    "Test customer-support understanding of refund exceptions",
    source="./documents",
    output="./benchmark",
    model="MODEL_ID",
    base_url="http://localhost:8000/v1",
    api_key_env="MODEL_API_KEY",
    max_tokens=4000,
    concurrency=2,
)
print(result.status)
questions = result.load_dataset()  # Hugging Face Dataset
print(questions[0]["question"])
print(questions[0]["ground_truth_answer"])
```

Set `MODEL_API_KEY` in the process environment before calling `create`. Pass the variable's **name**, never its value. For an unauthenticated local endpoint, omit `api_key_env`; for a Hugging Face provider, use `provider` instead of `base_url`. Unlike the CLI, the Python API does not load `.env` files automatically. If needed, load your chosen file explicitly with `dotenv.load_dotenv(path)`.

The output must be new or empty and outside the source tree. `create` makes a planning call, saves `plan.json` and `config.yaml`, runs the selected stages, and returns a `BenchmarkResult`. Exceptions propagate to the caller. The generated recipe disables Hub publication and saves datasets and JSONL locally.

`max_tokens` is an optional provider output-token limit **per response**, including planning. It is not a total run budget or question count. Too small a value can truncate JSON and fail a run. `concurrency` is the maximum simultaneous requests per model (default 8). Both settings are saved in the recipe.

## Inspect the plan before generating

```python
from yourbench import create, run

planned = create(
    "Build multiple-choice questions about policy exceptions",
    source="./documents",
    output="./planned-benchmark",
    model="MODEL_ID",
    base_url="http://localhost:8000/v1",
    plan_only=True,
)
print(planned.config_path)
# Inspect plan.json and edit config.yaml if needed.
result = run(planned.config_path)
```

`plan_only=True` still calls the model. It leaves question generation for `run`. `run` accepts a YAML path or a generated output directory. It reruns enabled stages, rather than resuming only unfinished stages. Generated recipes replace named subsets by default; `concat_if_exist: true` explicitly changes this to append.

For programmatically assembled configurations, the lower-level `yourbench.conf.loader.resolve_config(mapping, base_dir=...)` and `yourbench.pipeline.handler.run_pipeline_with_config(config)` remain available. The small public API uses saved recipes so runs can be inspected and repeated.

## Read existing results without credentials

```python
from yourbench import load_result

result = load_result("./benchmark")
print(result.summary())
questions = result.load_dataset()
source_documents = result.load_dataset("ingested")
```

Remote-only recipes expose no local datasets or local run status. Reading results never calls a model, downloads a dataset, loads a custom Python question schema, or resolves model credentials. Storage paths may still use environment variables, which must be set. Paths are interpreted relative to the recipe, and absolute paths are used as saved. After moving a benchmark directory, update its recipe's storage paths before loading it; the reader does not guess a new location.

`BenchmarkResult` provides:

| Member | Meaning |
| --- | --- |
| `config_path` | Absolute recipe path |
| `dataset_dir` | Configured local Arrow dataset store |
| `jsonl_dir` | Configured JSONL directory when local export is enabled, otherwise `None` |
| `output_subset` | Evaluation subset, normally `prepared_lighteval` |
| `status` | Status from `run.json`, `planned` when only a plan exists, otherwise `unknown` |
| `summary()` | Status, run ID, paths, and each local subset's row count and columns |
| `load_dataset(subset=None)` | Local Hugging Face `Dataset`; defaults to the evaluation subset |

A missing subset or corrupt dataset raises an error. If a run failed, prior datasets can still be present and readable. Check `status` before treating the artifacts as the result of a completed run. Completion means the selected stages executed; it does not certify factual correctness or benchmark quality.

The CLI exposes the same local inspection through `yourbench inspect ./benchmark`, or `yourbench inspect ./benchmark --json` for scripts.

## Async applications and notebooks

The execution API is synchronous. From a notebook or another application with an active event loop, run it in a worker thread:

```python
import asyncio
from yourbench import run

result = await asyncio.to_thread(run, "./benchmark/config.yaml")
```

Do not run concurrent writers against the same dataset directory. Cancelling an await on `asyncio.to_thread` does not stop the worker's pipeline; allow it to finish before touching its output.

## What to check before using generated questions

Inspect the question, answer, citations, and `sources` together. Questions should be answerable from their attached passages. Generation prompts reserve combined reasoning for the supplied multi-hop evidence and omit topics absent from a single-hop chunk, but this is model guidance, not semantic enforcement.

A real two-document trial produced 18 questions and a rerun produced 21. Both passed structural/source checks; two first-run single-hop questions unnecessarily abstained about rules available in the other document. This prompted stricter evidence-scoping instructions. It demonstrates why a completed run and valid schema do not replace a quality review. See [dataset columns](DATASET_COLUMNS_DESCRIPTION.md) and [custom schemas](CUSTOM_SCHEMAS.md).

`run.json` describes the last recorded pipeline execution. Errors while loading a recipe happen before a new execution is recorded and may leave the previous status unchanged; always check the exception or CLI exit code for the current attempt.
