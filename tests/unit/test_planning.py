"""Behavior checks for the natural-language creation path without paid requests."""

import json
from unittest.mock import Mock

import yaml
import pytest
from typer.testing import CliRunner

from yourbench import planning
from yourbench.main import app
from yourbench.conf.loader import load_config


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "sources"
    path.mkdir()
    (path / "policy.md").write_text("Refunds are available within 30 days.")
    return path


@pytest.fixture
def planner(monkeypatch):
    response = {
        "question_mode": "open-ended",
        "strategies": ["single-hop"],
        "additional_instructions": "Focus on policy exceptions.",
        "assumptions": ["Questions are for support agents."],
        "unsupported_requests": [],
    }
    mock = Mock(return_value={"test-model": [json.dumps(response)]})
    monkeypatch.setattr(planning, "run_inference", mock)
    return mock


def arguments(source, output, *extra):
    return [
        "create",
        "Evaluate difficult refund cases",
        "--source",
        str(source),
        "--output",
        str(output),
        "--model",
        "test-model",
        *extra,
    ]


def test_plan_only_persists_reusable_local_recipe(source, tmp_path, planner, monkeypatch):
    pipeline = Mock()
    monkeypatch.setattr("yourbench.pipeline.handler.run_pipeline_with_progress", pipeline)
    output = tmp_path / "benchmark"
    result = CliRunner().invoke(app, arguments(source, output, "--plan-only"))
    assert result.exit_code == 0, result.output
    pipeline.assert_not_called()
    config = load_config(output / "config.yaml")
    assert not config.hf_configuration.push_to_hub
    assert config.hf_configuration.export_jsonl
    assert config.pipeline.single_hop_question_generation.run
    assert not config.pipeline.multi_hop_question_generation.run
    instructions = config.pipeline.single_hop_question_generation.additional_instructions
    assert "Evaluate difficult refund cases" in instructions
    assert "Focus on policy exceptions" in instructions
    assert "schema_definition" in config.pipeline.single_hop_question_generation.single_hop_system_prompt
    assert json.loads((output / "plan.json").read_text())["brief"] == "Evaluate difficult refund cases"
    assert "Assumption:" in result.output


def test_create_runs_compiled_config(source, tmp_path, planner, monkeypatch):
    pipeline = Mock()
    monkeypatch.setattr("yourbench.pipeline.handler.run_pipeline_with_progress", pipeline)
    result = CliRunner().invoke(app, arguments(source, tmp_path / "benchmark"))
    assert result.exit_code == 0, result.output
    pipeline.assert_called_once()
    assert pipeline.call_args.args[0].pipeline.ingestion.source_documents_dir == str(source)


def test_key_is_resolved_for_request_but_saved_as_reference(source, tmp_path, planner, monkeypatch):
    monkeypatch.setenv("TEST_PLANNER_KEY", "private-test-value")
    output = tmp_path / "benchmark"
    result = CliRunner().invoke(
        app,
        arguments(
            source,
            output,
            "--plan-only",
            "--api-key-env",
            "TEST_PLANNER_KEY",
            "--base-url",
            "http://localhost:8000/v1",
        ),
    )
    assert result.exit_code == 0, result.output
    assert planner.call_args.args[0].model_list[0].api_key == "private-test-value"
    raw = (output / "config.yaml").read_text()
    assert "${TEST_PLANNER_KEY}" in raw
    assert "private-test-value" not in raw + (output / "plan.json").read_text() + result.output
    assert load_config(output / "config.yaml").model_list[0].api_key == "private-test-value"


@pytest.mark.parametrize("response", ["", "not-json", "{}", '{"question_mode":"open-ended","push_to_hub":true}'])
def test_invalid_plan_fails_without_running_or_saving(source, tmp_path, planner, monkeypatch, response):
    planner.return_value = {"test-model": [response]}
    pipeline = Mock()
    monkeypatch.setattr("yourbench.pipeline.handler.run_pipeline_with_progress", pipeline)
    output = tmp_path / "benchmark"
    result = CliRunner().invoke(app, arguments(source, output))
    assert result.exit_code == 1
    assert not output.exists()
    pipeline.assert_not_called()


def test_unsupported_requirements_are_not_silently_ignored(source, tmp_path, planner):
    response = json.loads(planner.return_value["test-model"][0])
    response["unsupported_requests"] = ["A hard $20 budget"]
    planner.return_value = {"test-model": [json.dumps(response)]}
    result = CliRunner().invoke(app, arguments(source, tmp_path / "benchmark"))
    assert result.exit_code == 1
    assert "hard $20 budget" in result.output


def test_invalid_local_inputs_fail_before_planner(source, tmp_path, planner):
    for input_source, output in [(tmp_path / "missing", tmp_path / "out"), (source, source / "out")]:
        result = CliRunner().invoke(app, arguments(input_source, output))
        assert result.exit_code == 1
    planner.assert_not_called()


def test_no_model_is_not_silently_defaulted(source, tmp_path, planner, monkeypatch):
    monkeypatch.delenv("YOURBENCH_MODEL", raising=False)
    args = arguments(source, tmp_path / "benchmark")[:-2]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert "Select a model" in result.output
    planner.assert_not_called()


def test_existing_artifacts_are_not_overwritten(source, tmp_path, planner):
    output = tmp_path / "benchmark"
    output.mkdir()
    config = output / "config.yaml"
    config.write_text("original")
    result = CliRunner().invoke(app, arguments(source, output))
    assert result.exit_code == 1
    assert config.read_text() == "original"
    planner.assert_not_called()


def test_model_cannot_choose_executable_settings(source, tmp_path, planner):
    response = json.loads(planner.return_value["test-model"][0])
    response["question_schema"] = "/tmp/untrusted.py:Schema"
    planner.return_value = {"test-model": [json.dumps(response)]}
    result = CliRunner().invoke(app, arguments(source, tmp_path / "benchmark"))
    assert result.exit_code == 1


def test_all_strategies_compile(source, tmp_path, planner):
    (source / "second.txt").write_text("Another policy.")
    response = json.loads(planner.return_value["test-model"][0])
    response["strategies"] = ["single-hop", "multi-hop", "cross-document", "single-hop"]
    planner.return_value = {"test-model": [json.dumps(response)]}
    intent, config_path = planning.create_recipe("Difficult questions", source, tmp_path / "out", "test-model")
    assert len(intent.strategies) == 3
    raw = yaml.safe_load(config_path.read_text())
    for stage in [
        "single_hop_question_generation",
        "multi_hop_question_generation",
        "cross_document_question_generation",
    ]:
        assert raw["pipeline"][stage]["run"]


def test_entrypoint_recognizes_create(monkeypatch):
    import yourbench.main as cli

    dispatch = Mock()
    monkeypatch.setattr(cli, "app", dispatch)
    monkeypatch.setattr(cli.sys, "argv", ["yourbench", "create", "My benchmark"])
    cli.main()
    assert cli.sys.argv[1] == "create"
    dispatch.assert_called_once()


def test_init_writes_complete_local_recipe(source, tmp_path):
    config_path = tmp_path / "config.yaml"
    result = CliRunner().invoke(
        app, ["init", "--output", str(config_path), "--source", str(source), "--model", "test-model"]
    )
    assert result.exit_code == 0, result.output
    config = load_config(config_path)
    assert config.pipeline.summarization.run and config.pipeline.chunking.run
    assert not config.hf_configuration.push_to_hub
    result = CliRunner().invoke(app, ["init", "--output", str(config_path)])
    assert result.exit_code == 1


def test_credentials_cannot_be_embedded_in_endpoint(source, tmp_path, planner):
    from urllib.parse import urlunsplit

    # A deliberately invalid endpoint, assembled without a credential-shaped literal.
    endpoint = urlunsplit(("https", "test-user:test-password@example.invalid", "/v1", "", ""))
    result = CliRunner().invoke(app, arguments(source, tmp_path / "out", "--base-url", endpoint))
    assert result.exit_code == 1
    assert "test-password" not in result.output
    planner.assert_not_called()


def test_planner_gets_only_corpus_count_not_document_contents(source, tmp_path, planner):
    (source / "ignored.bin").write_text("Not a supported document")
    planning.create_recipe("Evaluate policies", source, tmp_path / "out", "test-model")
    messages = planner.call_args.args[2][0].messages
    assert "1 supported documents" in messages[1]["content"]
    assert "Refunds are available" not in str(messages)


def test_single_document_cross_doc_plan_is_rejected(source, tmp_path, planner, monkeypatch):
    response = json.loads(planner.return_value["test-model"][0])
    response["strategies"] = ["cross-document"]
    planner.return_value = {"test-model": [json.dumps(response)]}
    pipeline = Mock()
    monkeypatch.setattr("yourbench.pipeline.handler.run_pipeline_with_progress", pipeline)
    result = CliRunner().invoke(app, arguments(source, tmp_path / "out"))
    assert result.exit_code == 1
    assert "at least 2" in result.output
    pipeline.assert_not_called()
    assert not (tmp_path / "out").exists()


def test_invalid_planner_data_is_not_echoed(source, tmp_path, planner):
    planner.return_value = {"test-model": ['{"unexpected": "sensitive-provider-response"}']}
    result = CliRunner().invoke(app, arguments(source, tmp_path / "out"))
    assert result.exit_code == 1
    assert "invalid benchmark intent" in result.output
    assert "sensitive-provider-response" not in result.output


@pytest.mark.parametrize("option", ["max_tokens", "concurrency"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8"])
def test_invalid_model_limits_fail_before_planning_or_writing(source, tmp_path, planner, option, value):
    output = tmp_path / "benchmark"
    with pytest.raises(ValueError, match=f"{option} must be a positive integer"):
        planning.create_recipe("Evaluate policies", source, output, "test-model", **{option: value})
    planner.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("limits", [{}, {"max_tokens": 1, "concurrency": 1}, {"max_tokens": 2048, "concurrency": 3}])
def test_model_limits_apply_to_planner_and_reusable_recipe(source, tmp_path, planner, monkeypatch, limits):
    monkeypatch.setenv("TEST_PLANNER_KEY", "first-private-value")
    output = tmp_path / "benchmark"
    _, config_path = planning.create_recipe(
        "Evaluate policies",
        source,
        output,
        "test-model",
        None,
        "http://localhost:8000/v1",
        "TEST_PLANNER_KEY",
        **limits,
    )
    runtime_model = planner.call_args.args[0].model_list[0]
    expected_parameters = {"max_tokens": limits["max_tokens"]} if "max_tokens" in limits else {}
    assert runtime_model.max_concurrent_requests == limits.get("concurrency", 8)
    assert runtime_model.extra_parameters == expected_parameters
    assert runtime_model.api_key == "first-private-value"
    saved_model = yaml.safe_load(config_path.read_text())["model_list"][0]
    assert saved_model["max_concurrent_requests"] == runtime_model.max_concurrent_requests
    assert saved_model.get("extra_parameters", {}) == expected_parameters
    assert saved_model["api_key"] == "${TEST_PLANNER_KEY}"
    assert "first-private-value" not in "".join(path.read_text() for path in output.iterdir())
    monkeypatch.setenv("TEST_PLANNER_KEY", "rotated-private-value")
    reloaded_model = load_config(config_path).model_list[0]
    assert reloaded_model.api_key == "rotated-private-value"
    assert reloaded_model.max_concurrent_requests == runtime_model.max_concurrent_requests
    assert reloaded_model.extra_parameters == expected_parameters
