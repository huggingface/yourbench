"""Exercise the actual CLI, HTTP client, pipeline, and persisted artifacts offline."""

import os
import sys
import json
import threading
import subprocess
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import yaml
import pytest


@pytest.fixture
def model_server():
    calls = []
    state = {"fail_generation": False, "custom_fields": {}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(request)
            messages = request["messages"]
            system = messages[0]["content"] if messages[0]["role"] == "system" else ""
            if system.startswith("Translate the user's benchmark brief"):
                content = json.dumps({
                    "question_mode": "open-ended",
                    "strategies": ["single-hop"],
                    "additional_instructions": "Focus on returns exceptions.",
                    "assumptions": [],
                    "unsupported_requests": [],
                })
            elif not system:
                content = json.dumps({"summary": "Returns are allowed within thirty days."})
            elif system.startswith("Rewrite the question"):
                content = json.dumps({
                    "question": "Within how many days can a customer return an item?",
                    "rationale": "Clarified the customer action without changing the time limit.",
                })
            elif state["fail_generation"]:
                content = "Invalid question output"
            else:
                assert "{schema_definition}" not in system
                content = json.dumps([
                    {
                        "question": "How long is the returns window?",
                        "answer": "Thirty days.",
                        "question_type": "factual",
                        "thought_process": "Tests the policy time limit.",
                        "estimated_difficulty": 3,
                        "citations": ["Returns are allowed within thirty days."],
                        **state["custom_fields"],
                    }
                ])
            payload = json.dumps({
                "id": "mock",
                "object": "chat.completion",
                "created": 1,
                "model": "local-test",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", calls, state
    server.shutdown()
    server.server_close()
    thread.join()


def invoke(tmp_path, args):
    env = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "LANG", "TIKTOKEN_CACHE_DIR", "SSL_CERT_FILE"}
    }
    env.update({"HF_HUB_DISABLE_TELEMETRY": "1", "YOURBENCH_TEST_KEY": "local-test-secret"})
    return subprocess.run(
        [sys.executable, "-m", "yourbench", *args], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=45
    )


def test_create_plan_execute_rerun_and_failure(tmp_path, model_server):
    endpoint, calls, state = model_server
    source = tmp_path / "source"
    source.mkdir()
    (source / "policy.txt").write_text("Returns are allowed within thirty days.")
    output = tmp_path / "benchmark"
    args = [
        "create",
        "Test policy exceptions",
        "--source",
        str(source),
        "--output",
        str(output),
        "--model",
        "local-test",
        "--base-url",
        endpoint,
        "--api-key-env",
        "YOURBENCH_TEST_KEY",
    ]
    args += ["--max-tokens", "2000", "--concurrency", "2"]
    planned = invoke(tmp_path, [*args, "--plan-only"])
    assert planned.returncode == 0, planned.stdout + planned.stderr
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 2000
    config = output / "config.yaml"
    assert "local-test-secret" not in config.read_text()
    assert "YOURBENCH_TEST_KEY" in config.read_text()
    assert not (output / "datasets").exists()
    executed = invoke(tmp_path, ["run", str(config), "--quiet"])
    assert executed.returncode == 0, executed.stdout + executed.stderr
    records = [json.loads(line) for line in (output / "jsonl" / "prepared_lighteval.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["ground_truth_answer"] == "Thirty days."
    assert records[0]["chunks"] == ["Returns are allowed within thirty days."]
    assert records[0]["sources"][0]["document_id"]
    assert records[0]["gold"] == [0]
    assert not (tmp_path / "questions_and_answers.jsonl").exists()
    assert json.loads((output / "run.json").read_text())["status"] == "completed"
    # Regenerate a saved recipe, preserving source identity and replacing old subsets.
    rerun = invoke(tmp_path, ["run", str(output), "--quiet"])
    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    records2 = [json.loads(line) for line in (output / "jsonl" / "prepared_lighteval.jsonl").read_text().splitlines()]
    assert records2[0]["sources"] == records[0]["sources"]
    state["fail_generation"] = True
    failed = invoke(tmp_path, ["run", str(config), "--quiet"])
    assert failed.returncode != 0
    assert json.loads((output / "run.json").read_text())["status"] == "failed"
    assert "local-test-secret" not in failed.stdout + failed.stderr


def test_create_executes_without_plan_only(tmp_path, model_server):
    endpoint, calls, _ = model_server
    source = tmp_path / "source"
    source.mkdir()
    (source / "policy.txt").write_text("Returns are allowed within thirty days.")
    output = tmp_path / "benchmark"
    result = invoke(
        tmp_path,
        [
            "create",
            "Test policies",
            "--source",
            str(source),
            "--output",
            str(output),
            "--model",
            "local-test",
            "--base-url",
            endpoint,
            "--api-key-env",
            "YOURBENCH_TEST_KEY",
        ],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(calls) == 3
    assert (output / "jsonl" / "prepared_lighteval.jsonl").is_file()


def test_saved_recipe_rewrites_and_exports_custom_payload_with_provenance(tmp_path, model_server):
    endpoint, calls, state = model_server
    source = tmp_path / "source"
    source.mkdir()
    source_text = "Returns are allowed within thirty days."
    (source / "policy.txt").write_text(source_text)
    output = tmp_path / "benchmark"
    planned = invoke(
        tmp_path,
        [
            "create",
            "Test returns policy comprehension",
            "--source",
            str(source),
            "--output",
            str(output),
            "--model",
            "local-test",
            "--base-url",
            endpoint,
            "--api-key-env",
            "YOURBENCH_TEST_KEY",
            "--plan-only",
        ],
    )
    assert planned.returncode == 0, planned.stdout + planned.stderr
    schema = output / "question_schema.py"
    schema.write_text(
        "from pydantic import BaseModel, Field\n"
        "class RubricItem(BaseModel):\n    criterion: str\n    weight: int = Field(ge=1)\n"
        "class DataFormat(BaseModel):\n"
        "    question: str\n    answer: str\n    citations: list[str]\n"
        "    rubric: list[RubricItem]\n    difficulty: str\n"
    )
    custom_fields = {
        "rubric": [{"criterion": "States the thirty-day limit", "weight": 2}],
        "difficulty": "Requires reading the policy, not a numeric difficulty score",
    }
    state["custom_fields"] = custom_fields
    config_path = output / "config.yaml"
    recipe = yaml.safe_load(config_path.read_text())
    recipe["pipeline"]["single_hop_question_generation"]["question_schema"] = "question_schema.py"
    recipe["pipeline"]["question_rewriting"] = {"run": True}
    config_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    result = invoke(tmp_path, ["run", str(config_path), "--quiet"])
    assert result.returncode == 0, result.stdout + result.stderr

    def saved_rows(subset):
        return [json.loads(line) for line in (output / "jsonl" / f"{subset}.jsonl").read_text().splitlines()]

    generated = saved_rows("single_hop_questions")
    rewritten = saved_rows("single_hop_questions_rewritten")
    exported = saved_rows("prepared_lighteval")
    assert len(generated) == len(rewritten) == len(exported) == 1
    before, after, final = generated[0], rewritten[0], exported[0]
    assert before["question"] == "How long is the returns window?"
    assert after["question"] == final["question"] == "Within how many days can a customer return an item?"
    assert final["original_question"] == before["question"]
    for field in [
        "answer",
        "self_answer",
        "sources",
        "document_id",
        "chunk_id",
        "question_data",
        "citations",
        *custom_fields,
    ]:
        assert after[field] == final[field] == before[field]
    assert final["ground_truth_answer"] == "Thirty days."
    assert final["choices"] == ["Thirty days."]
    assert final["gold"] == [0]
    assert final["chunks"] == [source_text]
    assert final["question_rewriting_model"] == "local-test"
    assert final["question_rewriting_rationale"]
    assert final["question_data"]["question"] == before["question"]
    assert "estimated_difficulty" not in final
    assert {field: final[field] for field in custom_fields} == custom_fields

    rewriting_calls = [call for call in calls if call["messages"][0]["content"].startswith("Rewrite the question")]
    assert len(rewriting_calls) == 1
    prompt = rewriting_calls[0]["messages"][1]["content"]
    assert before["question"] in prompt and before["answer"] in prompt
    assert source_text in prompt
    assert before["sources"][0]["document_id"] in prompt
    assert before["sources"][0]["chunk_id"] in prompt
    assert json.loads((output / "run.json").read_text())["status"] == "completed"
    assert "local-test-secret" not in config_path.read_text() + result.stdout + result.stderr


def test_public_python_api_executes_bounded_recipe_and_reads_without_credentials(tmp_path, model_server, monkeypatch):
    from yourbench import run, create, load_result

    endpoint, calls, state = model_server
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHON_API_TEST_KEY", "local-test-secret")
    source = tmp_path / "source"
    source.mkdir()
    (source / "policy.md").write_text("Returns are allowed within thirty days.")
    result = create(
        "Test returns policy comprehension",
        source=source,
        output=tmp_path / "benchmark",
        model="local-test",
        base_url=endpoint,
        api_key_env="PYTHON_API_TEST_KEY",
        max_tokens=2000,
        concurrency=2,
    )
    assert result.status == "completed"
    assert len(calls) == 3
    assert all(call["max_tokens"] == 2000 for call in calls)
    first = result.load_dataset().to_list()
    assert first[0]["ground_truth_answer"] == "Thirty days."
    monkeypatch.delenv("PYTHON_API_TEST_KEY")
    offline = load_result(result.config_path.parent)
    assert offline.summary()["subsets"]["prepared_lighteval"]["rows"] == 1
    assert offline.load_dataset().to_list() == first
    assert len(calls) == 3
    monkeypatch.setenv("PYTHON_API_TEST_KEY", "rotated-local-secret")
    rerun = run(result.config_path.parent)
    assert rerun.load_dataset().to_list()[0]["sources"] == first[0]["sources"]
    assert len(calls) == 5
    # Failure propagates through the public API and inspection reports failure,
    # even though the previous evaluation artifact remains readable.
    before_failure = rerun.load_dataset().to_list()
    state["fail_generation"] = True
    with pytest.raises(ValueError):
        run(result.config_path)
    assert load_result(result.config_path).status == "failed"
    assert load_result(result.config_path).load_dataset().to_list() == before_failure
