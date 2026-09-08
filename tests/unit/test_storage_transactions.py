"""Storage behavior under real serialization, corruption and interrupted publication."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from datasets import Dataset, DatasetDict, load_from_disk
from yourbench.conf.schema import YourbenchConfig
from yourbench.utils.dataset_engine import _safe_save, custom_load_dataset, custom_save_dataset


def config_for(path, **options):
    return YourbenchConfig(
        hf_configuration={"hf_dataset_name": "local", "push_to_hub": False, "local_dataset_dir": str(path), **options}
    )


def snapshot(path):
    return {str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()}


def test_serialization_failure_preserves_existing_dataset_byte_for_byte(tmp_path):
    target = tmp_path / "data"
    old = DatasetDict({"questions": Dataset.from_list([{"question": "Original?", "answer": "原本"}])})
    _safe_save(old, target)
    before = snapshot(target)

    def failing_write(self, location, **kwargs):
        Path(location).mkdir(parents=True, exist_ok=True)
        (Path(location) / "partial.arrow").write_bytes(b"incomplete new data")
        raise OSError("injected serialization failure")

    with patch.object(DatasetDict, "save_to_disk", failing_write), pytest.raises(OSError):
        _safe_save(DatasetDict({"questions": Dataset.from_list([{"question": "Replacement?"}])}), target)
    assert snapshot(target) == before
    assert load_from_disk(str(target))["questions"][0]["answer"] == "原本"
    assert list(tmp_path.iterdir()) == [target]


def test_promotion_failure_restores_readable_old_dataset(tmp_path):
    target = tmp_path / "data"
    _safe_save(Dataset.from_list([{"id": "old"}]), target)
    original_replace = Path.replace

    def interrupted_replace(source, destination):
        if source.name == "new":
            raise OSError("injected promotion failure")
        return original_replace(source, destination)

    with patch.object(Path, "replace", interrupted_replace), pytest.raises(OSError, match="promotion"):
        _safe_save(Dataset.from_list([{"id": "new"}]), target)
    assert load_from_disk(str(target)).to_list() == [{"id": "old"}]
    assert list(tmp_path.iterdir()) == [target]


def test_corrupt_existing_store_is_never_treated_as_empty(tmp_path):
    target = tmp_path / "data"
    cfg = config_for(target)
    custom_save_dataset(Dataset.from_list([{"id": "original"}]), cfg, "questions")
    (target / "questions" / "state.json").write_text("{bad JSON")
    before = snapshot(target)
    with pytest.raises(json.JSONDecodeError):
        custom_save_dataset(Dataset.from_list([{"id": "replacement"}]), cfg, "questions")
    assert snapshot(target) == before


def test_missing_shard_does_not_fall_back_to_hub(tmp_path):
    target = tmp_path / "data"
    cfg = config_for(target)
    custom_save_dataset(Dataset.from_list([{"id": "original"}]), cfg, "questions")
    next(target.rglob("*.arrow")).unlink()
    cfg.hf_configuration.push_to_hub = True
    cfg.hf_configuration.hf_organization = "test"
    with (
        patch("huggingface_hub.HfApi.repo_info", side_effect=AssertionError("metadata network")),
        patch("yourbench.utils.dataset_engine._load_hub", side_effect=AssertionError("remote fallback")),
    ):
        with pytest.raises(FileNotFoundError):
            custom_load_dataset(cfg, "questions")


def test_append_partition_invariance_preserves_other_subsets(tmp_path):
    rows = [{"id": i, "payload": {"text": f"Item {i} é", "values": [i, i + 1]}} for i in range(12)]
    target = tmp_path / "data"
    cfg = config_for(target, concat_if_exist=True)
    custom_save_dataset(Dataset.from_list([{"sentinel": "untouched"}]), cfg, "other")
    for start, end in [(0, 1), (1, 7), (7, 12)]:
        custom_save_dataset(Dataset.from_list(rows[start:end]), cfg, "questions")
    assert custom_load_dataset(cfg, "questions").to_list() == rows
    assert custom_load_dataset(cfg, "other").to_list() == [{"sentinel": "untouched"}]


def test_failed_append_leaves_all_previous_subsets_unchanged(tmp_path):
    target = tmp_path / "data"
    cfg = config_for(target, concat_if_exist=True)
    custom_save_dataset(Dataset.from_list([{"id": 1}]), cfg, "questions")
    before = snapshot(target)
    with pytest.raises(ValueError):
        custom_save_dataset(Dataset.from_list([{"id": {"nested": "incompatible"}}]), cfg, "questions")
    assert snapshot(target) == before


def test_jsonl_failure_preserves_old_file_without_partial_rows(tmp_path):
    from yourbench.utils.dataset_engine import _export_to_jsonl

    valid = Dataset.from_list([{"value": 1.5}])
    _export_to_jsonl(valid, tmp_path, "questions")
    before = (tmp_path / "questions.jsonl").read_bytes()
    # First new row serializes; second is not valid JSON and must not truncate the old file.
    invalid = Dataset.from_list([{"value": 2.5}, {"value": float("inf")}])
    with pytest.raises(ValueError):
        _export_to_jsonl(invalid, tmp_path, "questions")
    assert (tmp_path / "questions.jsonl").read_bytes() == before
    assert sorted(file.name for file in tmp_path.iterdir()) == ["questions.jsonl"]


def test_storage_read_error_prevents_replacement(tmp_path):
    cfg = config_for(tmp_path / "data")
    custom_save_dataset(Dataset.from_list([{"id": "old"}]), cfg, "questions")
    before = snapshot(tmp_path / "data")
    with patch("yourbench.utils.dataset_engine.load_from_disk", side_effect=PermissionError("denied")):
        with pytest.raises(PermissionError):
            custom_save_dataset(Dataset.from_list([{"id": "new"}]), cfg, "questions")
    assert snapshot(tmp_path / "data") == before
