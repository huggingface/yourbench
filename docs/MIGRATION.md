# Natural-language redesign: changes and migration

This guide covers the CLI, library API, runtime, storage, and output changes in the natural-language redesign. It describes the source checkout; it is not a claim that a new package release has been published.

## New entry points

Start from an objective instead of assembling a pipeline by hand:

```bash
yourbench create "Test customer-support understanding of policy exceptions" \
  --source ./documents --output ./benchmark --model MODEL_ID \
  --base-url http://localhost:8000/v1 --api-key-env MODEL_API_KEY \
  --max-tokens 4000 --concurrency 2

yourbench inspect ./benchmark --json
yourbench run ./benchmark
```

Set the named key variable before running, or omit `--api-key-env` for an unauthenticated endpoint. `--max-tokens` limits each response, including planning; it is not a total spending limit. `--concurrency` sets the per-model request limit. `--plan-only` saves the interpretation and recipe after a model call, without generating questions.

`create` saves `plan.json`, `config.yaml`, local datasets, JSONL, and a run manifest. The planner selects supported question formats and generation strategies; it cannot choose filesystem paths, credentials, executable schema files, or publication destinations. It reports unsupported requests such as exact question counts, spending caps, conversational evaluations, and executable evaluators.

The public Python API provides `create`, `run`, `load_result`, and `BenchmarkResult`:

```python
from yourbench import load_result

result = load_result("./benchmark")
print(result.status)
print(result.summary())
questions = result.load_dataset()
```

Reading results needs no model credentials or inference calls. `run` and `load_result` accept a YAML recipe or its output directory. See [Python API](PYTHON_API.md) and [CLI reference](CLI.md).

## Configuration changes

| Previous behavior or configuration | New behavior / migration |
| --- | --- |
| Working-directory-dependent paths | Relative paths resolve beside the YAML recipe. Update paths when moving a recipe; absolute saved paths remain absolute. |
| Missing prompt path could fall back silently | Missing files are errors. Use `file:path.md` for files and `inline:instructions` for literal text. |
| Environment expansion in arbitrary prompt content | Prompts and `additional_instructions` remain literal; execution settings and explicit prompt file paths may expand environment references. |
| Implicit named OpenAI model | Automatic OpenAI configuration requires both `OPENAI_MODEL` and `OPENAI_API_KEY`. Select the model explicitly for `create`. |
| `pipeline.ingestion.upload_to_hub` | Removed. Use `hf_configuration.push_to_hub` for every stage. |
| Multiple summary/rewrite models | Summarization and rewriting each require exactly one model; generation supports multiple assigned models. |
| Implicit summary dependency | To skip summaries, explicitly set `pipeline.chunking.input_subset: ingested`. |
| Inconsistent sampling and overlap | Chunking honors `token_overlap` and `encoding_name`. Single-hop sampling uses `random` or `first`, with `num_samples` per document. |
| XML summary/rewrite responses | Summary prompts must produce `{"summary": "..."}`; rewrite prompts must produce `{"question": "...", "rationale": "..."}`. |

Generated recipes are local-only, private, and replace existing named subsets on rerun. Existing YAML defaults still permit Hub operations unless disabled; set `push_to_hub: false` for local-only runs. This also disables remote dataset reads and dataset-card publication. At least one persistence destination must be enabled. JSONL export requires local saving and a configured export directory.

The supported legacy top-level names `models` and `pipeline_config` still map to `model_list` and `pipeline`. Unknown fields otherwise fail validation. See [configuration reference](CONFIGURATION.md) for all settings.

## Response and schema contracts

Generation prompts request a JSON array of question objects. One decoder accepts bare JSON, a complete Markdown JSON fence, or a complete legacy `<output_json>` envelope. It rejects surrounding prose, concatenated payloads, duplicate keys, and nonfinite numbers, including overflowing exponents. It no longer searches malformed text for fragments to salvage.

Candidates validate against the actual selected Pydantic schema; prompts include that schema's JSON Schema. Invalid candidates are rejected, and generation with no valid questions fails. The default schemas carry their own metadata defaults. There is no inferred `reasoning`/`explanation` alias or string-to-number difficulty conversion. Express desired conversions in the custom schema; custom fields keep their validated names and values.

`question_data` preserves the validated original payload, including choices before shuffling. Generated fields cannot replace authoritative runtime metadata. Custom fields survive generation, rewriting, and export when their Arrow types are compatible; incompatible shared types raise an explicit conflict instead of dropping fields.

Multiple-choice shuffling preserves the identity of the correct option, even when option text repeats. Default multiple-choice questions have four options; custom schemas can use 2–26. `gold` is always a list of choice indices. Open-ended export uses `choices: [answer]`, `gold: [0]`, and `ground_truth_answer: answer`.

Canonical `sources` stores document/chunk pairs; cross-document evidence does not assume globally unique chunk IDs. Rewriting receives resolved source passages and preserves answers, custom data, and provenance. Default export follows rewritten subsets when rewriting is enabled. See [custom schemas](CUSTOM_SCHEMAS.md) and [dataset columns](DATASET_COLUMNS_DESCRIPTION.md).

## Runtime and storage changes

- One stage catalogue defines ordering and artifact dependencies. Preflight checks required inputs and persistence settings before generation.
- Inference preserves request order, bounds concurrency per model, retries transient failures, cancels sibling work on failure, and closes clients. Failed requests no longer become fabricated empty answers.
- Inference metrics use `logs/inference.jsonl`; redundant CSV/atexit reporting was removed. Token counts are local estimates, not provider billing totals.
- Ingestion converts each file once and publishes the ingested dataset only after all documents convert. Conversion failures are errors. `llm_ingestion` applies to PDF pages. Converted filenames retain their original extension (`policy.txt.md`), and document IDs depend on relative source path plus text, making corpus moves stable.
- Dataset writes serialize into a sibling staging directory before replacement and restore the old store if promotion fails. JSONL files are replaced after successful serialization. Corrupt datasets and missing physical shards are errors, not missing subsets eligible for remote fallback.
- `MissingSubsetError` distinguishes absent named artifacts from damaged storage. Remote append requires reading the existing target; use `concat_if_exist: false` for first publication.
- `run.json` records stage completion and failure. A failed run can leave earlier artifacts. Configuration-loading errors happen before a new manifest is recorded, so check the exception or CLI exit code for the current attempt.

These guarantees cover individual writes, not a whole-run transaction, crash recovery, or concurrent writers to the same directory. The execution API is synchronous; use `asyncio.to_thread` from an active event loop.

## Documentation and examples

The README, CLI, configuration, provider, FAQ, schema, and output-column guides now describe the implemented behavior. Six example recipes include small fictional Markdown/PDF inputs, explicit local output settings, and configurable endpoints. Their preflight and chunk viability are exercised without model calls. See [examples](../example/README.md).

Citation scoring adds overlap scores; it does not filter rows or establish factual correctness. Deduplication normalizes question casing and whitespace within a generated subset; it is not semantic deduplication.

## Verification and limits

At the pre-merge checkpoint, 245 tests pass, including real CLI subprocesses against a local HTTP server, Python API execution, credential-free inspection, corrupt-storage and interrupted-write tests, concurrency/cancellation tests, and offline example checks. Five deliberately injected runtime defects were caught in isolated copies; the [mutation report](superpowers/plans/2026-09-07-flat-core-verification.md) includes reproduction steps. Ruff, formatting, wheel/source builds, and imports from the built wheel were checked.

A separate real-model trial used two fictional policy documents: the first run produced 18 questions and a rerun produced 21. All 11 logical calls succeeded without retries; dataset/JSONL parity, source references, answer indices, and all 98 citations passed verification. Two first-run single-hop questions asked about information present only in the other document and unnecessarily abstained. Subsequent prompts explicitly scope single-hop questions to their supplied chunk and combined questions to their supplied passages. Request tests cover that guidance; a new live semantic evaluation has not established that the issue is eliminated.

The live trial covered open-ended single-hop and cross-document generation, local ingestion/export, and reruns. It did not verify every model/provider, PDF extraction, multiple-choice generation, or rewriting against a real model. A successful run is not a guarantee of question quality, exact count, or cost.
