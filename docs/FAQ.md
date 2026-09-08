# YourBench FAQ

## What does it do?

YourBench turns local documents into question-and-answer datasets. Start with a natural-language brief using `yourbench create`, or edit a YAML recipe. It generates benchmarks; it does not run candidate models against them or establish that the questions are correct.

## How do I start?

From a checkout, install with `pip install -e .`, then follow [the CLI guide](CLI.md). `yourbench init` creates a starter recipe without calling a model. `yourbench create --plan-only` calls the planner once and saves its interpretation for review.

The [examples](../example/README.md) include a small source corpus and save locally. Select a model you can access and configure its credentials before running them.

## What does “local” mean?

`hf_configuration.push_to_hub: false` keeps dataset publication disabled. A remote inference endpoint still receives the document passages used in its requests. To keep inference on your machine too, configure a local OpenAI-compatible server. `HF_HUB_OFFLINE=1` disables Hub access; it does not disable model requests to a configured endpoint.

## Which stages should I enable?

The usual order is ingestion, summarization, chunking, question generation and `prepare_lighteval`. Generation supports single passages, multiple passages from one document, and cross-document combinations. Rewriting and citation scoring are optional. `yourbench stages` lists registered stages.

Including a stage in YAML enables it, even with `{}` or an empty value. Omit it or set `run: false` to disable it. Stages execute in dependency order, not YAML order. Saved subsets can supply inputs when earlier stages are disabled.

## Can I skip summaries?

Yes. Enable ingestion and set `pipeline.chunking.input_subset: ingested`. Summaries provide document context but can contain errors. Generation should still be answerable from the actual supplied passages.

## Can I use multiple models?

Yes. Declare each model once in `model_list`, then assign model names in `model_roles`. Unassigned roles use the first model. Question generation can use multiple models. Summarization and PDF LLM ingestion each require exactly one model. Concurrency is bounded separately per model.

## What documents and chunking methods are supported?

The default extension list is `.md`, `.txt`, `.pdf`. Add extensions such as `.html` explicitly in `supported_file_extensions` when needed. Ingestion converts files to text; `llm_ingestion: true` uses page images for PDFs and requires a vision-capable endpoint.

Chunking uses token windows with configurable overlap. Set `l_max_tokens` and `token_overlap` directly under `pipeline.chunking`; there is no semantic chunker or nested `chunking_configuration`. For large documents, summarization splits and combines summaries. Configure sizes for your model's context window, allowing space for prompts and responses.

## Where are the results?

A recipe created by `create` stores `config.yaml`, `plan.json`, `run.json`, `datasets/`, `jsonl/` and `processed/` beneath its output directory. The final default subset is `prepared_lighteval`. See [dataset columns](DATASET_COLUMNS_DESCRIPTION.md) for loading and answer-index conventions.

In hand-written YAML, paths resolve relative to the YAML file. Local and JSONL destinations are configured separately. Set `push_to_hub: true` explicitly only when you want publication, along with the destination and Hub credentials.

## What happens on reruns or failure?

With `concat_if_exist: false`, saving replaces the named subset while retaining others. A stage failure stops the run; `run.json` records completed stages and failure status. Earlier completed stages remain saved. Dataset writes stage new content and restore the old directory if promotion fails; this is not a whole-run transaction or concurrent-writer guarantee.

Corrupt existing datasets raise errors instead of silently being replaced. Remote append requires an existing readable Hub subset. Use separate output directories for independent runs.

## Does citation scoring prove correctness?

No. `citation_score_filtering` adds overlap scores; it does not remove rows or verify semantic correctness. Apply your own threshold or review after scoring. Inspect source scope, answer correctness, distractors and unsupported abstentions as well as citations. A successful pipeline run can still produce weak questions.

## How do I control generation volume?

Use `single_hop_question_generation.chunk_sampling`, reduce cross-document `max_combinations`, and tune `chunking.num_multihops_factor`. `yourbench estimate` estimates tokens without generation; it is not a spending cap. Neither exact question counts nor dollar budgets are enforced.

## Why did a response fail parsing?

Question generation expects one JSON array matching the selected Pydantic schema. Whole JSON fences and the legacy whole `<output_json>` envelope are accepted, but surrounding prose, duplicate keys and nonfinite numbers are rejected. Fields are not guessed from synonyms. Malformed envelopes fail the stage; individual schema-invalid questions are rejected, and generation with no valid questions fails.

Custom summaries return `{"summary": "..."}`; custom rewrites return `{"question": "...", "rationale": "..."}`. See [custom schemas](CUSTOM_SCHEMAS.md) and [configuration](CONFIGURATION.md).
