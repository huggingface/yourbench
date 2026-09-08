from unittest.mock import patch

import pytest

from datasets import Dataset
from yourbench.conf.loader import load_config
from yourbench.conf.schema import ChunkingConfig, YourbenchConfig
from yourbench.pipeline.chunking import _process_document, _sample_multihop_combinations
from yourbench.utils.chunking_utils import get_sampling_cfg, split_into_token_chunks, sample_single_hop_chunks
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset


def test_auto_model_resolves_credentials_without_printing(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-private-value")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text("pipeline: {}")
    config = load_config(path)
    assert config.model_list[0].api_key == "test-private-value"
    assert config.model_list[0].base_url == "https://api.openai.com/v1"
    assert "test-private-value" not in repr(config)


def test_prompt_paths_resolve_beside_config(tmp_path):
    (tmp_path / "prompt.md").write_text("A custom prompt")
    path = tmp_path / "config.yaml"
    path.write_text("pipeline:\n  summarization:\n    summarization_user_prompt: prompt.md\n")
    assert load_config(path).pipeline.summarization.summarization_user_prompt == "A custom prompt"
    (tmp_path / "prompt.md").unlink()
    with pytest.raises((ValueError, FileNotFoundError), match="prompt"):
        load_config(path)


def test_invalid_roles_and_duplicate_models_rejected():
    with pytest.raises(ValueError, match="duplicate|unique"):
        YourbenchConfig(model_list=[{"model_name": "a"}, {"model_name": "a"}])
    with pytest.raises(ValueError, match="unknown|Unknown"):
        YourbenchConfig(model_list=[{"model_name": "a"}], model_roles={"summarization": ["b"]})


def test_chunking_honors_parameters_and_repeats():
    cfg = ChunkingConfig(l_max_tokens=20, token_overlap=4, encoding_name="p50k_base")
    with patch("yourbench.pipeline.chunking.split_into_token_chunks", return_value=["hello"]) as split:
        _process_document({"document_id": "doc", "document_text": "hello world"}, cfg)
    assert split.call_args.kwargs == {"overlap": 4, "encoding_name": "p50k_base"}
    assert _sample_multihop_combinations(20, 2, 5, 1, "doc") == _sample_multihop_combinations(20, 2, 5, 1, "doc")
    with pytest.raises(ValueError):
        split_into_token_chunks("abc", 10, 10)


def test_sampling_honors_count_without_changing_global_random():
    import random

    config = YourbenchConfig(
        pipeline={
            "single_hop_question_generation": {"chunk_sampling": {"enable": True, "num_samples": 2, "random_seed": 1}}
        }
    )
    sampling = get_sampling_cfg(config.pipeline.single_hop_question_generation)
    chunks = [{"chunk_id": str(i)} for i in range(10)]
    state = random.getstate()
    assert len(sample_single_hop_chunks(chunks, sampling)) == 2
    assert sample_single_hop_chunks(chunks, sampling) == sample_single_hop_chunks(chunks, sampling)
    assert random.getstate() == state


def test_local_storage_never_calls_hub_and_missing_subset_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    config = YourbenchConfig(
        hf_configuration={
            "hf_dataset_name": "local",
            "push_to_hub": False,
            "local_dataset_dir": str(tmp_path / "saved"),
        }
    )
    with (
        patch("yourbench.utils.dataset_engine.whoami", side_effect=AssertionError("network")),
        patch("yourbench.utils.dataset_engine._load_hub", side_effect=AssertionError("network")),
    ):
        custom_save_dataset(Dataset.from_list([{"text": "hello"}]), config, "ingested", push_to_hub=False)
        assert len(custom_load_dataset(config, "ingested")) == 1
        with pytest.raises(FileNotFoundError):
            custom_load_dataset(config, "chunked")


def test_card_upload_disabled_when_local_only():
    from yourbench.utils.dataset_card import upload_dataset_card

    config = YourbenchConfig(hf_configuration={"push_to_hub": False})
    with patch("yourbench.utils.dataset_card._generate_and_upload_dataset_card") as upload:
        upload_dataset_card(config)
    upload.assert_not_called()


def test_instructions_never_expand_credentials(monkeypatch):
    from yourbench.conf.loader import resolve_config

    monkeypatch.setenv("HF_TOKEN", "test-private-value")
    config = resolve_config({
        "pipeline": {
            "single_hop_question_generation": {
                "additional_instructions": "Explain $HF_TOKEN and ${HF_TOKEN} literally",
                "single_hop_user_prompt": "inline:Discuss $HF_TOKEN",
            }
        }
    })
    stage = config.pipeline.single_hop_question_generation
    assert stage.additional_instructions == "Explain $HF_TOKEN and ${HF_TOKEN} literally"
    assert stage.single_hop_user_prompt == "Discuss $HF_TOKEN"


def test_missing_credential_reference_fails(monkeypatch):
    from yourbench.conf.loader import resolve_config

    monkeypatch.delenv("YOURBENCH_TEST_MISSING", raising=False)
    with pytest.raises(ValueError, match="YOURBENCH_TEST_MISSING"):
        resolve_config({"model_list": [{"model_name": "test", "api_key": "$YOURBENCH_TEST_MISSING"}]})


def test_rewriting_routes_only_active_generation():
    from yourbench.conf.loader import resolve_config

    cfg = resolve_config({
        "pipeline": {"single_hop_question_generation": {}, "question_rewriting": {}, "prepare_lighteval": {}}
    })
    assert cfg.pipeline.prepare_lighteval.single_hop_subset == "single_hop_questions_rewritten"
    assert cfg.pipeline.prepare_lighteval.multi_hop_subset == "multi_hop_questions"
    assert cfg.pipeline.prepare_lighteval.cross_doc_subset == "cross_document_questions"


def test_preflight_missing_questions_before_any_model_call(tmp_path):
    from yourbench.conf.loader import resolve_config
    from yourbench.pipeline.handler import PipelineError, validate_pipeline

    source = tmp_path / "source"
    source.mkdir()
    (source / "text.txt").write_text("Document")
    cfg = resolve_config(
        {
            "hf_configuration": {"hf_dataset_name": "test", "push_to_hub": False},
            "model_list": [{"model_name": "test"}],
            "pipeline": {
                "ingestion": {"source_documents_dir": str(source)},
                "summarization": {},
                "chunking": {},
                "prepare_lighteval": {},
            },
        },
        base_dir=tmp_path,
    )
    with pytest.raises(PipelineError, match="question subset"):
        validate_pipeline(cfg)


def test_preflight_allows_explicit_unsummarized_input(tmp_path):
    from yourbench.conf.loader import resolve_config
    from yourbench.pipeline.handler import validate_pipeline

    source = tmp_path / "source"
    source.mkdir()
    (source / "text.txt").write_text("Document")
    cfg = resolve_config(
        {
            "hf_configuration": {"hf_dataset_name": "test", "push_to_hub": False},
            "pipeline": {"ingestion": {"source_documents_dir": str(source)}, "chunking": {"input_subset": "ingested"}},
        },
        base_dir=tmp_path,
    )
    validate_pipeline(cfg)


def test_resumed_rewriting_selects_existing_outputs(tmp_path):
    from yourbench.conf.loader import resolve_config
    from yourbench.pipeline.handler import validate_pipeline

    cfg = resolve_config(
        {
            "hf_configuration": {"hf_dataset_name": "test", "push_to_hub": False},
            "model_list": [{"model_name": "test"}],
            "pipeline": {"question_rewriting": {}, "prepare_lighteval": {}},
        },
        base_dir=tmp_path,
    )
    custom_save_dataset(Dataset.from_list([{"question": "Original?"}]), cfg, "single_hop_questions")
    custom_save_dataset(Dataset.from_list([{"document_id": "a"}]), cfg, "chunked")
    validate_pipeline(cfg)
    assert cfg.pipeline.prepare_lighteval.single_hop_subset == "single_hop_questions_rewritten"
    assert cfg.pipeline.prepare_lighteval.multi_hop_subset == "multi_hop_questions"


def test_ingestion_preserves_same_stem_formats(tmp_path):
    from yourbench.conf.loader import resolve_config
    from yourbench.pipeline.ingestion import run

    source = tmp_path / "source"
    source.mkdir()
    (source / "report.txt").write_text("Text report")
    (source / "report.md").write_text("Markdown report")
    cfg = resolve_config(
        {
            "hf_configuration": {"hf_dataset_name": "test", "push_to_hub": False},
            "pipeline": {"ingestion": {"source_documents_dir": str(source)}},
        },
        base_dir=tmp_path,
    )
    run(cfg)
    ingested = custom_load_dataset(cfg, "ingested")
    assert set(ingested["document_text"]) == {"Text report", "Markdown report"}
    assert len(set(ingested["document_id"])) == 2


def test_failed_remote_append_does_not_publish_replacement():
    cfg = YourbenchConfig(
        hf_configuration={
            "hf_dataset_name": "org/test",
            "hf_organization": "org",
            "push_to_hub": True,
            "local_saving": False,
            "concat_if_exist": True,
        }
    )
    dataset = Dataset.from_list([{"text": "new"}])
    with (
        patch("yourbench.utils.dataset_engine._load_hub", side_effect=ValueError("incompatible schema")),
        patch.object(Dataset, "push_to_hub") as publish,
    ):
        with pytest.raises(ValueError, match="incompatible"):
            custom_save_dataset(dataset, cfg, "questions")
    publish.assert_not_called()
