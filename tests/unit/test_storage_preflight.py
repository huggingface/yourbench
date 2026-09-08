"""Invalid persistence settings and damaged inputs must fail before work is lost."""

from pathlib import Path

import pytest

from datasets import Dataset
from yourbench.conf.loader import resolve_config
from yourbench.pipeline.handler import PipelineError, validate_pipeline
from yourbench.utils.dataset_engine import ConfigurationError, custom_load_dataset, custom_save_dataset
from yourbench.pipeline.prepare_lighteval import run as export


def config_for(tmp_path, **stages):
    return resolve_config({
        "hf_configuration": {
            "hf_dataset_name": "test",
            "local_dataset_dir": str(tmp_path / "store"),
            "push_to_hub": False,
        },
        "model_list": [{"model_name": "model"}],
        "pipeline": stages,
    })


def snapshot(path):
    return {str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()}


@pytest.mark.parametrize("setting", ["no_destination", "missing_export", "export_is_file", "export_parent_is_file"])
def test_invalid_settings_do_not_change_existing_store(tmp_path, monkeypatch, setting):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    config = config_for(tmp_path)
    custom_save_dataset(Dataset.from_dict({"value": ["original"]}), config, subset="existing")
    before = snapshot(tmp_path / "store")
    if setting == "no_destination":
        config.hf_configuration.local_saving = False
    else:
        config.hf_configuration.export_jsonl = True
        config.hf_configuration.jsonl_export_dir = ""
        if setting != "missing_export":
            blocker = tmp_path / "blocked"
            blocker.write_text("Do not change this file")
            config.hf_configuration.jsonl_export_dir = str(
                blocker if setting == "export_is_file" else blocker / "child"
            )
    with pytest.raises(ConfigurationError):
        custom_save_dataset(Dataset.from_dict({"value": ["replacement"]}), config, subset="existing")
    assert snapshot(tmp_path / "store") == before
    assert custom_load_dataset(config, "existing")["value"] == ["original"]
    with pytest.raises(ConfigurationError):
        validate_pipeline(config)


def test_rewriting_preflight_requires_source_dataset(tmp_path):
    config = config_for(tmp_path, question_rewriting={})
    custom_save_dataset(
        Dataset.from_dict({"question": ["Q?"], "self_answer": ["A"]}), config, subset="single_hop_questions"
    )
    with pytest.raises(PipelineError, match="chunked"):
        validate_pipeline(config)


@pytest.mark.parametrize("operation", [validate_pipeline, export])
def test_damaged_optional_question_store_is_not_treated_as_absent(tmp_path, operation):
    config = config_for(tmp_path, prepare_lighteval={})
    docs = Dataset.from_list([{"document_id": "d", "chunks": [{"chunk_id": "c", "chunk_text": "source"}]}])
    custom_save_dataset(docs, config, subset="chunked")
    custom_save_dataset(
        Dataset.from_dict({"question": ["Q?"], "self_answer": ["A"], "document_id": ["d"], "chunk_id": ["c"]}),
        config,
        subset="single_hop_questions",
    )
    shard = next((Path(config.hf_configuration.local_dataset_dir) / "single_hop_questions").glob("*.arrow"))
    shard.unlink()
    with pytest.raises(FileNotFoundError) as failure:
        operation(config)
    assert not isinstance(failure.value, PipelineError)
    assert "arrow" in str(failure.value)


def test_explicit_no_destination_override_also_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    config = config_for(tmp_path)
    with pytest.raises(ConfigurationError, match="no dataset destination"):
        custom_save_dataset(Dataset.from_dict({"value": [1]}), config, save_local=False, push_to_hub=False)
    assert not (tmp_path / "store").exists()
