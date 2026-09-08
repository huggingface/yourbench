# Flat-core runtime verification

The inference changes were checked with real coroutine scheduling and five deliberate mutations in independent temporary source copies. The shared checkout was never mutated. Every selected mutation produced pytest exit code 1, rather than a subprocess timeout or import failure.

| Deliberate mutation | Test selector | Observed result |
| --- | --- | --- |
| Replace per-model semaphore capacity with 1000 | `test_per_model_concurrency_is_saturated_and_bounded` | 3 failed |
| Reverse each model's returned response slice | `test_batch_preserves_order_despite_completion_order` | 1 failed |
| Remove shared-client close registration | `test_failure_cancels_siblings_before_client_closes` | 1 failed |
| Retry permanent errors until exhaustion | `test_http_failures_apply_policy_through_execution` | 2 failed, 2 passed; permanent cases failed and transient cases passed |
| Increment aggregate request count by two | `test_metrics_log_and_totals_agree_under_event_permutations` | 1 failed |

The scheduling test holds requests behind an async event until two independent model pools saturate. It checks their actual in-flight counts and peak concurrency across three different capacity assignments, then releases them and verifies ordered outputs and closed clients. The metrics test permutes events and checks totals against the persisted event records. The HTTP policy test executes the retry loop using actual `httpx.HTTPStatusError` instances, including 400, 401, 429, and 503 responses.

Before and after the mutation experiment, the unmodified inference test module passed all 25 cases, and Ruff passed for the owned runtime files. This report makes no full-suite claim. The original machine-readable observations were also saved to `/tmp/yourbench-inference-mutation-report.json`; the procedure below is the durable reproduction.

## Reproduce

Run from the repository root with its existing virtual environment. Each experiment receives a fresh copy of `yourbench/` and the inference test module; no network or model calls are involved. The script asserts each replacement still exists so implementation drift cannot silently turn a mutation into a no-op.

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import json
import shutil
import subprocess
import tempfile

root = Path.cwd()
python = str(root / '.venv/bin/python')
cases = [
    ('ignore_concurrency_limit', 'inference_core.py',
     'asyncio.Semaphore(model.max_concurrent_requests)', 'asyncio.Semaphore(1000)',
     'test_per_model_concurrency_is_saturated_and_bounded'),
    ('reverse_outputs', 'inference_core.py',
     'results[i * count : (i + 1) * count]',
     'list(reversed(results[i * count : (i + 1) * count]))',
     'test_batch_preserves_order_despite_completion_order'),
    ('leak_shared_clients', 'inference_core.py',
     'stack.push_async_callback(client.close)', 'pass',
     'test_failure_cancels_siblings_before_client_closes'),
    ('retry_permanent_errors', 'inference_core.py',
     'if not _is_transient(error) or attempt + 1 == inference_call.max_retries:',
     'if attempt + 1 == inference_call.max_retries:',
     'test_http_failures_apply_policy_through_execution'),
    ('double_count_calls', 'inference_tracking.py',
     '"calls": 1,', '"calls": 2,',
     'test_metrics_log_and_totals_agree_under_event_permutations'),
]
results = []
with tempfile.TemporaryDirectory(prefix='yourbench-inference-mutations-') as folder:
    for name, module, old, new, selector in cases:
        case = Path(folder) / name
        shutil.copytree(root / 'yourbench', case / 'yourbench',
                        ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(root / 'tests/unit/test_inference_core.py',
                     case / 'test_inference_core.py')
        target = case / 'yourbench/utils/inference' / module
        source = target.read_text()
        assert old in source, f'Mutation no longer matches: {name}'
        target.write_text(source.replace(old, new, 1))
        run = subprocess.run(
            [python, '-m', 'pytest', 'test_inference_core.py', '-q', '-k', selector],
            cwd=case, capture_output=True, text=True, timeout=15,
        )
        assert run.returncode == 1, (name, run.returncode, run.stdout, run.stderr)
        results.append({'mutation': name, 'result': run.stdout.strip().splitlines()[-1]})
print(json.dumps(results, indent=2))
PY
```

A detected mutation proves the corresponding test distinguishes the healthy implementation from that specific defect. It does not establish correctness for every possible scheduling or provider failure.
