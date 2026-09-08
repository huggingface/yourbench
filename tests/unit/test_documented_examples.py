"""Example recipes must find real sources and pass preflight without remote access."""

from pathlib import Path
from unittest.mock import patch

import pytest

from yourbench.conf.loader import load_config
from yourbench.pipeline.handler import validate_pipeline


EXAMPLES = sorted((Path(__file__).resolve().parents[2] / "example").glob("*/config.yaml"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.parent.name)
def test_shipped_example_has_executable_local_configuration(path, monkeypatch):
    monkeypatch.setenv("YOURBENCH_MODEL", "test-model")
    monkeypatch.setenv("YOURBENCH_API_KEY", "dummy-key")
    monkeypatch.setenv("YOURBENCH_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with patch("huggingface_hub.HfApi.repo_info", side_effect=AssertionError("Example attempted network")):
        config = load_config(path)
        assert config.hf_configuration.push_to_hub is False
        assert config.hf_configuration.local_saving is True
        validate_pipeline(config)

    from yourbench.pipeline.chunking import _process_document
    from yourbench.pipeline.ingestion import source_files

    sources = source_files(config.pipeline.ingestion)
    for source in sources:
        if source.suffix == ".pdf":
            import fitz

            with fitz.open(source) as pdf:
                text = "\n".join(page.get_text() for page in pdf)
        else:
            text = source.read_text(encoding="utf-8")
        chunks, groups = _process_document(
            {"document_id": source.stem, "document_text": text}, config.pipeline.chunking
        )
        assert chunks and all(chunk["chunk_text"].strip() for chunk in chunks)
        if config.pipeline.multi_hop_question_generation.run:
            assert groups and any(len(group["chunk_ids"]) >= 2 for group in groups)
    if config.pipeline.cross_document_question_generation.run:
        assert len(sources) >= 2
