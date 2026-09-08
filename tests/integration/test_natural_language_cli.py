"""Exercise the actual CLI, HTTP client, pipeline, and persisted artifacts offline."""

import os
import sys
import json
import threading
import subprocess
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import pytest


@pytest.fixture
def model_server():
    calls = []
    state = {"fail_generation": False}

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
                content = "<chunk_summary>Returns are allowed within thirty days.</chunk_summary>"
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
    planned = invoke(tmp_path, [*args, "--plan-only"])
    assert planned.returncode == 0, planned.stdout + planned.stderr
    assert len(calls) == 1
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
    rerun = invoke(tmp_path, ["run", str(config), "--quiet"])
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
