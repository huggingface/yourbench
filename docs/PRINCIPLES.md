# How YourBench works

YourBench turns a benchmark objective and a source directory into a repeatable document-to-question workflow.

## Describe the objective

`yourbench create "your brief" --source DIR --model MODEL --output DIR` sends the brief and supported document count to a planning model. The model returns an intent: question mode, generation strategies, instructions, assumptions, and unsupported requirements. A deterministic compiler produces the execution configuration. Model output cannot select paths, credentials, executable schemas, or publication destinations.

The saved `plan.json` makes the interpretation inspectable. The saved `config.yaml` can be edited and rerun. `--plan-only` stops after planning; it still uses the selected model.

## Execute the recipe

Stages communicate through named dataset subsets. The runner checks required inputs before execution and records status in `run.json` beside the local dataset directory.

1. **Ingestion** converts supported files into text/Markdown with stable document IDs. Distinct source formats retain separate output files.
2. **Summarization** generates document summaries, combining chunk summaries for long inputs.
3. **Chunking** divides documents by token count, respecting configured overlap and encoding. Explicit `input_subset: ingested` permits operation without summaries.
4. **Generation** selects individual passages, combinations within a document, or passages across documents. Shared prompt rendering and parsing validate generated questions against their selected schema.
5. **Deduplication**, within each generated subset, removes repeated case-folded question text with collapsed whitespace. Numbers and punctuation remain significant. This is not semantic deduplication.
6. **Optional rewriting** rewrites question wording while preserving payload fields and source references. Active or resumed inputs are routed to rewritten export subsets unless explicitly overridden.
7. **Evaluation preparation** joins questions with their source documents/chunks and produces consistent choice-index `gold` values.
8. **Optional citation scoring** adds fuzzy text-overlap scores. It does not remove rows or prove factual correctness.

A central stage catalogue supplies execution order, module routing, display labels, and artifact contracts. Model calls share a bounded asynchronous runtime with per-model concurrency, transient retries, and explicit errors. Individual invalid question candidates can be rejected; generation that produces no valid questions fails.

## Data and limitations

`question_data` retains the validated original model payload. Canonical `sources` identifies each document/chunk pair, including evidence spanning documents. Additional custom fields survive export when their Arrow types are compatible; conflicting shared types require separate subsets or a consistent schema.

The natural-language frontend defaults to local datasets and JSONL. Existing YAML can explicitly enable Hugging Face Hub publication. In local-only mode, missing data never silently triggers a remote read.

This is a benchmark generation tool, not a guarantee that every generated question is useful or correct. Exact question counts, dollar spending caps, interactive conversations, and executable evaluators are not implemented. The planner is instructed to surface these unsupported requests. Failed reruns may leave previous artifacts on disk; use `run.json` to identify the most recent run's outcome.

## Library and inspection

The public Python functions `create`, `run`, and `load_result` return a `BenchmarkResult` that exposes local datasets and run metadata. `yourbench inspect OUTPUT --json` reads the same metadata without credentials or inference. The status is the last recorded pipeline execution; configuration-loading errors can leave an older status unchanged. See the [Python API](PYTHON_API.md).

Per-response token limits and per-model concurrency can be specified at creation and are carried into the saved recipe. Generation requests include stage-specific evidence instructions: single-hop questions must be answerable from their one chunk, while combined reasoning uses only the supplied multi-hop passages. These instructions guide the model; they are not semantic validation.
