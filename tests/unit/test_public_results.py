"""Saved results remain usable without credentials, providers, or custom schema code."""

import json
from unittest.mock import patch

import yaml
import pytest
from typer.testing import CliRunner

from datasets import Dataset, DatasetDict
from yourbench import load_result
from yourbench.main import app


def saved_result(tmp_path):
    root = tmp_path / "benchmark"
    root.mkdir()
    schema = root / "schema.py"
    schema.write_text("raise AssertionError('Reading results must not execute schemas')")
    recipe = {
        "hf_configuration": {
            "local_dataset_dir": "store",
            "jsonl_export_dir": "exports",
            "push_to_hub": True,
            "hf_token": "${UNAVAILABLE_HUB_TOKEN}",
        },
        "model_list": [{"model_name": "remote", "api_key": "${UNAVAILABLE_MODEL_KEY}"}],
        "pipeline": {
            "single_hop_question_generation": {"question_schema": "schema.py"},
            "prepare_lighteval": {"output_subset": "evaluation"},
        },
    }
    (root / "config.yaml").write_text(yaml.safe_dump(recipe))
    rows = [{"question": "Refund?", "answer": "Yes", "nested": {"rubric": ["timing", "eligibility"]}}]
    DatasetDict({
        "evaluation": Dataset.from_list(rows),
        "ingested": Dataset.from_dict({"text": ["policy"]}),
    }).save_to_disk(str(root / "store"))
    (root / "run.json").write_text(json.dumps({"status": "completed", "run_id": "test-run"}))
    return root, rows


def test_result_reads_relative_paths_without_credentials_or_schema_execution(tmp_path, monkeypatch):
    root, rows = saved_result(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    with patch("huggingface_hub.HfApi.repo_info", side_effect=AssertionError("Network forbidden")):
        result = load_result(root)
        assert result.status == "completed"
        assert result.load_dataset().to_list() == rows
        assert result.load_dataset("ingested")["text"] == ["policy"]
        summary = result.summary()
    assert summary["run_id"] == "test-run"
    assert summary["subsets"]["evaluation"]["rows"] == 1
    assert summary["dataset_dir"] == str(root / "store")
    assert "UNAVAILABLE" not in json.dumps(summary)
    assert load_result(root / "config.yaml") == result


def test_inspection_reports_failed_run_even_when_old_rows_remain(tmp_path):
    root, rows = saved_result(tmp_path)
    (root / "run.json").write_text(json.dumps({"status": "failed", "run_id": "failed-new-run"}))
    command = CliRunner().invoke(app, ["inspect", str(root), "--json"])
    assert command.exit_code == 0, command.output
    summary = json.loads(command.output)
    assert summary["status"] == "failed"
    assert summary["subsets"]["evaluation"]["rows"] == 1
    human = CliRunner().invoke(app, ["inspect", str(root)])
    assert "left over" in human.output
    assert load_result(root).load_dataset().to_list() == rows


def test_inspection_preserves_corruption_error_instead_of_reporting_no_rows(tmp_path):
    root, _ = saved_result(tmp_path)
    next((root / "store" / "evaluation").glob("*.arrow")).unlink()
    with pytest.raises(FileNotFoundError):
        load_result(root).summary()
    command = CliRunner().invoke(app, ["inspect", str(root), "--json"])
    assert command.exit_code == 1
    assert "FileNotFoundError" in command.output
    assert '"subsets"' not in command.output


def test_planned_output_and_default_storage_paths(tmp_path):
    (tmp_path / "config.yaml").write_text("pipeline: {}")
    (tmp_path / "plan.json").write_text("{}")
    result = load_result(tmp_path)
    assert result.status == "planned"
    assert result.dataset_dir == tmp_path / "data" / "saved_dataset"
    assert result.summary()["subsets"] == {}
    with pytest.raises(FileNotFoundError):
        result.load_dataset()


def test_remote_only_recipe_does_not_claim_stale_local_artifacts(tmp_path):
    root, _ = saved_result(tmp_path)
    path = root / "config.yaml"
    data = yaml.safe_load(path.read_text())
    data["hf_configuration"]["local_saving"] = False
    path.write_text(yaml.safe_dump(data))
    result = load_result(root)
    assert result.dataset_dir is None
    assert result.status == "unknown"
    assert result.summary()["subsets"] == {}
    with pytest.raises(ValueError, match="no local"):
        result.load_dataset()


def test_reader_honors_legacy_pipeline_alias_without_execution(tmp_path):
    root, rows = saved_result(tmp_path)
    path = root / "config.yaml"
    data = yaml.safe_load(path.read_text())
    data["pipeline_config"] = data.pop("pipeline")
    path.write_text(yaml.safe_dump(data))
    assert load_result(path).load_dataset().to_list() == rows


def test_cli_run_accepts_directory_and_reports_empty_recipe(tmp_path):
    (tmp_path / "config.yaml").write_text("pipeline: {}")
    result = CliRunner().invoke(app, ["run", str(tmp_path), "--quiet"])
    assert result.exit_code == 1
    assert "No pipeline stages enabled" in result.output
    assert "must be a YAML" not in result.output
