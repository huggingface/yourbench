# Natural-language redesign implementation plan

Goal: Natural-language creation backed by a smaller reliable benchmark core.
Spec: ../specs/2026-09-07-natural-language-redesign.md
Architecture: model interprets intent; deterministic compiler chooses execution configuration; existing stages execute with explicit failures and validated data.
Global constraints: Python 3.12, no new agent framework, no paid calls/publication in tests, environment references only in saved config.

## Parallel work
- [x] Frontend: own yourbench/main.py, new yourbench/planning.py, frontend tests and docs/CLI.md. `create` plans, persists and optionally runs; preserve old commands. Call load_config on compiled YAML and handler.run_pipeline_with_progress. Save raw config before environment resolution; reject model-generated executable paths/settings. Tests inject planner responses and cover invalid plans, no model, secrets, missing sources, plan-only, execution.
- [x] Inference: own utils/inference/inference_core.py and inference_tracking.py and tests/unit/test_inference_core.py. Preserve successful run_inference(config, step_name, calls)->dict[str,list[str]]. Raise typed errors for missing models, permanent/exhausted requests. Reuse client per model, bounded transient retries, seed and merged extras, one logical request metric, closure on failure. Test concurrency ordering and cancellation cleanup.
- [x] Schema/data: own generation/_core.py, parsing_engine.py, question_models.py, schema_loader.py, prepare_lighteval.py, cross_document_utils.py, related tests. Validate custom schema and always render defaults; consolidate parsing and export; preserve fields and canonical sources; support variable MCQ choice counts when schema allows. Tests prove bad custom output rejection and cross-doc context preservation.
- [x] Coordinator: own conf/*, pipeline/handler.py, ingestion.py, chunking.py, question_rewriting.py, utils/chunking_utils.py, dataset_engine.py, dataset_card.py, README and integration tests. Fix configuration expansion/paths, sampling, ingestion validation, stage registry/dependencies, rewriting routing, local IO and card serialization. Avoid overlapping agent files; coordinate interfaces by message.

## Integration and verification
- [x] Establish baseline tests in worktree venv.
- [x] Run targeted behavior tests per task.
- [x] Review each worker diff and resolve findings.
- [x] Full test suite and Ruff for touched code.
- [x] CLI end-to-end with temporary source docs and local model stub; plan, create, rerun saved recipe, inspect output and missing/invalid failure paths.
- [x] Whole-change independent review, fix significant findings, repeat relevant checks.
- [x] Commit reviewed implementation and report branch/worktree and measured results.

## Verification and decisions

- Baseline: 91 tests passed. Final implementation: 161 tests passed; Ruff check and formatting passed for every changed/new Python file; git diff --check passed.
- Actual CLI integration uses a local HTTP model endpoint: planning-only, generation, saved-recipe rerun, stable source identity, JSONL output, failed-generation exit/status and credential-reference persistence. No paid model or Hub calls used.
- Three agents implemented frontend, inference and schema/export independently and reviewed one another's changes. Coordinator resolved integration findings and added configuration, storage, ingestion, rewriting and CLI regression tests. Final scoped reviewer reported no remaining blocker; separately identified remote append suppression was fixed and tested.
- Source shrank from 6,605 to 5,794 Python lines, including the new frontend and registry (811 fewer, 12.3%).
- Interface decisions: YAML paths are config-relative; init is noninteractive; no silent default model name; export gold is always choice indices; natural-language text never expands environment credentials; unsupported hard counts/budgets are surfaced by planner; incompatible custom Arrow field types require separate exports or consistent types.
- Work retained on feat/natural-language-redesign in ../yourbench-redesign. No push/publication or main-branch merge.
