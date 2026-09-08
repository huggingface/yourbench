"""Fresh Hub publication and resumed-input preflight have different dependencies."""

import json
from unittest.mock import patch

import pytest

from datasets import Dataset
from datasets.exceptions import DatasetNotFoundError
from yourbench.conf.loader import resolve_config
from yourbench.pipeline.handler import PipelineError, validate_pipeline, run_pipeline_with_config
from yourbench.utils.dataset_engine import MissingSubsetError, custom_load_dataset


def recipe(tmp_path, *, rewriting=False):
    source = tmp_path / "source"
    source.mkdir()
    (source / "policy.txt").write_text("Returns are accepted within 30 days.")
    pipeline = {
        "ingestion": {"source_documents_dir": "source"},
        "chunking": {"input_subset": "ingested"},
        "single_hop_question_generation": {},
        **({"question_rewriting": {}} if rewriting else {}),
        "prepare_lighteval": {},
    }
    return resolve_config(
        {
            "hf_configuration": {
                "hf_dataset_name": "test/new-repository",
                "local_dataset_dir": "datasets",
                "push_to_hub": True,
                "upload_card": False,
            },
            "model_list": [{"model_name": "test"}],
            "pipeline": pipeline,
        },
        base_dir=tmp_path,
    )


@pytest.mark.parametrize("rewriting", [False, True])
def test_fresh_hub_preflight_uses_scheduled_outputs_without_remote_reads(tmp_path, rewriting):
    config = recipe(tmp_path, rewriting=rewriting)
    with patch(
        "yourbench.utils.dataset_engine._load_hub",
        side_effect=DatasetNotFoundError("Repository will be created by ingestion"),
    ) as read:
        validate_pipeline(config)
    read.assert_not_called()


def test_fresh_pipeline_publishes_before_optional_remote_reads(tmp_path):
    config = recipe(tmp_path)
    published = {}

    def publish(dataset, *, config_name, **kwargs):
        assert kwargs["repo_id"] == "test/new-repository"
        published[config_name] = dataset.to_list()

    def read_remote(repo_id, subset, token):
        if not published:
            raise DatasetNotFoundError("Repository does not exist before first publication")
        if subset not in published:
            raise MissingSubsetError(subset)
        return Dataset.from_list(published[subset])

    def infer(*, config, step_name, inference_calls):
        assert step_name == "single_hop_question_generation"
        assert len(inference_calls) == 1
        assert "Returns are accepted within 30 days." in inference_calls[0].messages[1]["content"]
        return {"test": [json.dumps([{"question": "How long is the return window?", "answer": "30 days"}])]}

    with (
        patch.object(Dataset, "push_to_hub", autospec=True, side_effect=publish),
        patch("yourbench.utils.dataset_engine._load_hub", side_effect=read_remote),
        patch("yourbench.pipeline.question_generation._core.run_inference", side_effect=infer),
    ):
        run_pipeline_with_config(config)
    assert list(published) == ["ingested", "chunked", "single_hop_questions", "prepared_lighteval"]
    assert published["prepared_lighteval"][0]["ground_truth_answer"] == "30 days"
    assert published["prepared_lighteval"] == custom_load_dataset(config, "prepared_lighteval").to_list()
    assert json.loads((tmp_path / "run.json").read_text())["status"] == "completed"


def test_explicit_required_custom_subset_is_still_checked(tmp_path):
    config = recipe(tmp_path)
    config.pipeline.prepare_lighteval.multi_hop_subset = "required-custom"
    with patch("yourbench.utils.dataset_engine._load_hub", side_effect=MissingSubsetError("absent")) as read:
        with pytest.raises(PipelineError, match="required-custom"):
            validate_pipeline(config)
    assert read.call_args.args[1] == "required-custom"


def test_resumed_export_does_not_hide_remote_access_failure(tmp_path):
    config = recipe(tmp_path)
    for stage in ("ingestion", "chunking", "single_hop_question_generation"):
        getattr(config.pipeline, stage).run = False
    with patch("yourbench.utils.dataset_engine._load_hub", side_effect=PermissionError("denied")):
        with pytest.raises(PermissionError, match="denied"):
            validate_pipeline(config)
