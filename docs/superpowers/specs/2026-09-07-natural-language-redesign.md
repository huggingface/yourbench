# Natural-language benchmark creation and reliable execution

User authorization: implement collaboratively, radically simplify brittle code, natural-language frontend, interface changes allowed. Preserve useful YAML execution for repeatability.

## Experience
`yourbench create BRIEF --source DIR --model ID --output DIR [--plan-only]` interprets a user's evaluation objective into a small validated intent, deterministically compiles a runnable config, saves brief/intent/config artifacts, and executes unless plan-only. Paths, credentials, provider selection and publication are controlled by explicit inputs, never by model output. Default local saving and JSONL export, no Hub publication. Models are selected explicitly or via environment; secrets are represented only by environment references in saved config. Surface assumptions and reject unsupported requirements instead of claiming exact counts or dollar budgets without enforcement. Existing YAML run remains supported.

## Reliable core
Always render question prompts against a schema. Validate custom-schema responses against that schema before normalizing; preserve custom fields through export. Share generation parsing and export transformations; carry canonical document/chunk source references for all question strategies. Inference reuses and closes clients, retries transient failures with bounded backoff, passes seed, and raises typed errors rather than returning empty success. Configuration rejects missing prompt files and unknown model assignments; stage orchestration validates required inputs and uses one runner. Chunking honors configured overlap/encoding; sampling honors configured settings. Generated recipes are inspectable and rerunnable.

## Scope and constraints
Python 3.12. No new agent framework or general DAG engine. Use installed Hugging Face client. No real paid model calls or Hub writes during tests; verify the full user path against a local mock compatible endpoint. Existing code can change interfaces where needed. Add behavior tests for new contracts and preserve valid existing tests. No secrets in code, logs, test output or artifacts. Changes live in ../yourbench-redesign on feat/natural-language-redesign.
