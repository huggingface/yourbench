"""Document discovery and ingestion use real files and exact content assertions."""

import pytest

from yourbench.conf.loader import resolve_config
from yourbench.pipeline.ingestion import run, source_files
from yourbench.utils.dataset_engine import custom_load_dataset


def recipe(tmp_path, source):
    return resolve_config(
        {
            "hf_configuration": {"hf_dataset_name": "test", "push_to_hub": False},
            "pipeline": {"ingestion": {"source_documents_dir": str(source)}},
        },
        base_dir=tmp_path,
    )


def test_move_corpus_preserves_identity_but_content_changes_do_not(tmp_path):
    import shutil

    source = tmp_path / "first"
    source.mkdir()
    (source / "a.txt").write_text("Policy A: “thirty days” 日本語")
    (source / "nested").mkdir()
    (source / "nested" / "a.txt").write_text("Policy B: “fourteen days”")
    cfg = recipe(tmp_path / "run1", source)
    run(cfg)
    original = custom_load_dataset(cfg, "ingested").to_list()
    moved = tmp_path / "moved"
    shutil.copytree(source, moved)
    cfg2 = recipe(tmp_path / "run2", moved)
    run(cfg2)
    assert custom_load_dataset(cfg2, "ingested").to_list() == original
    (moved / "a.txt").write_text("Policy A: changed")
    run(cfg2)
    changed = custom_load_dataset(cfg2, "ingested").to_list()
    assert changed[0]["document_id"] != original[0]["document_id"]
    assert changed[1] == original[1]


def test_output_directory_is_excluded_without_excluding_similarly_named_sources(tmp_path):
    source = tmp_path / "input-output-policies"
    source.mkdir()
    (source / "policy.txt").write_text("Actual source")
    output = source / "generated"
    output.mkdir()
    (output / "prior.md").write_text("Old output must not become source")
    cfg = recipe(tmp_path, source)
    cfg.pipeline.ingestion.output_dir = str(output)
    assert source_files(cfg.pipeline.ingestion) == [source / "policy.txt"]
    run(cfg)
    assert custom_load_dataset(cfg, "ingested")["document_text"] == ["Actual source"]


def test_failed_source_does_not_publish_partial_new_dataset(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("Original")
    cfg = recipe(tmp_path, source)
    run(cfg)
    before = custom_load_dataset(cfg, "ingested").to_list()
    (source / "a.txt").write_text("Changed first document")
    (source / "z.txt").write_bytes(b"invalid utf8 \xff")
    with pytest.raises(UnicodeDecodeError):
        run(cfg)
    assert custom_load_dataset(cfg, "ingested").to_list() == before
