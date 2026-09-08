"""Named dataset storage with explicit missing-artifact and commit boundaries."""

import os
import json
import shutil
import tempfile
from typing import Any
from pathlib import Path
from dataclasses import field, dataclass

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets, get_dataset_config_names
from huggingface_hub import whoami
from huggingface_hub.utils import validate_repo_id
from yourbench.utils.env import validate_env_expanded


class ConfigurationError(ValueError):
    """Storage configuration is incomplete."""


class MissingSubsetError(FileNotFoundError):
    """A store exists, but does not contain the requested named artifact."""


@dataclass(frozen=True)
class HFSettings:
    dataset_name: str
    organization: str | None
    token: str | None = field(repr=False)
    local_dir: Path | None
    concat_if_exist: bool = False
    private: bool = True
    export_jsonl: bool = False
    jsonl_export_dir: Path | None = None

    @property
    def repo_id(self):
        if self.organization and "/" not in self.dataset_name:
            return f"{self.organization}/{self.dataset_name}"
        return self.dataset_name


def _is_offline() -> bool:
    return os.getenv("HF_HUB_OFFLINE", "0").lower() in {"1", "true", "yes"}


def _extract_settings(config) -> HFSettings:
    """Read configuration without network calls or side effects."""
    hf = config.hf_configuration
    name = getattr(hf, "hf_dataset_name", "")
    if not name:
        raise ConfigurationError("hf_dataset_name is required")
    local_dir = getattr(hf, "local_dataset_dir", None)
    export_dir = getattr(hf, "jsonl_export_dir", None)
    return HFSettings(
        dataset_name=validate_env_expanded(name, "hf_dataset_name"),
        organization=getattr(hf, "hf_organization", "") or None,
        token=getattr(hf, "hf_token", "") or os.getenv("HF_TOKEN"),
        local_dir=Path(local_dir).expanduser().resolve() if local_dir else None,
        concat_if_exist=getattr(hf, "concat_if_exist", False),
        private=getattr(hf, "private", True),
        export_jsonl=getattr(hf, "export_jsonl", False),
        jsonl_export_dir=Path(export_dir).expanduser().resolve() if export_dir else None,
    )


def validate_storage(config, *, save_local: bool | None = None, push_to_hub: bool | None = None):
    """Resolve write destinations and reject invalid settings before any side effects."""
    settings = _extract_settings(config)
    hf = config.hf_configuration
    local = getattr(hf, "local_saving", True) if save_local is None else save_local
    remote = getattr(hf, "push_to_hub", True) if push_to_hub is None else push_to_hub
    local, remote = (True, False) if _is_offline() else (local, remote)
    if not local and not remote:
        raise ConfigurationError("Enable local_saving or push_to_hub; no dataset destination is configured")
    if local and settings.local_dir is None:
        raise ConfigurationError("Local saving requires local_dataset_dir")
    if settings.export_jsonl and (not local or settings.jsonl_export_dir is None):
        raise ConfigurationError("JSONL export requires local saving and jsonl_export_dir")
    directories = [settings.local_dir] if local else []
    if settings.export_jsonl:
        directories.append(settings.jsonl_export_dir)
    for directory in directories:
        for path in (directory, *directory.parents):
            if path.exists() and not path.is_dir():
                raise ConfigurationError(f"Storage directory is blocked by a file: {path}")
    return settings, local, remote


def _remote_repo(settings: HFSettings) -> str:
    """Resolve account identity only for an actual remote operation."""
    repo = settings.repo_id
    if "/" not in repo and settings.token:
        repo = f"{whoami(token=settings.token)['name']}/{repo}"
    validate_repo_id(repo)
    return repo


def _load_local(path: Path, subset: str | None) -> Dataset | DatasetDict:
    dataset = load_from_disk(str(path))
    if subset is None:
        return dataset
    if not isinstance(dataset, DatasetDict) or subset not in dataset:
        raise MissingSubsetError(f"Subset '{subset}' is not present in {path}")
    return dataset[subset]


def _load_hub(repo_id: str, subset: str | None, token: str | None) -> Dataset:
    # Check documented metadata rather than parsing exception messages.
    if subset is not None and subset not in get_dataset_config_names(repo_id, token=token):
        raise MissingSubsetError(f"Subset '{subset}' is not present in {repo_id}")
    return load_dataset(repo_id, name=subset, split="train", token=token)


def _merge_datasets(existing, new: Dataset, subset: str | None, concat_if_exist=False):
    if subset is None:
        if isinstance(existing, DatasetDict):
            raise ConfigurationError("Specify a subset when writing to a named dataset store")
        return concatenate_datasets([existing, new]) if existing is not None and concat_if_exist else new
    subsets = (
        dict(existing)
        if isinstance(existing, DatasetDict)
        else ({"default": existing} if existing is not None else {})
    )
    previous = subsets.get(subset)
    subsets[subset] = concatenate_datasets([previous, new]) if previous is not None and concat_if_exist else new
    return DatasetDict(subsets)


def _safe_save(dataset: Dataset | DatasetDict, path: Path) -> None:
    """Serialize first, then replace. Restore the previous store on promotion failure.

    If rollback itself fails, retain the backup directory for recovery.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    new, previous = staging / "new", staging / "previous"
    try:
        dataset.save_to_disk(str(new))
        if path.exists():
            path.replace(previous)
        try:
            new.replace(path)
        except BaseException:
            if previous.exists():
                previous.replace(path)
            raise
        if previous.exists():
            shutil.rmtree(previous)
    finally:
        if not previous.exists():
            shutil.rmtree(staging)


def _write_jsonl(dataset: Dataset, destination: Path) -> None:
    """Do not truncate a previous export if serialization fails."""
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        temporary = Path(staging) / "rows.jsonl"
        with temporary.open("w", encoding="utf-8") as stream:
            for row in dataset:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(destination)


def _export_to_jsonl(dataset: Dataset | DatasetDict, export_dir: Path, subset: str | None = None) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    subsets = dict(dataset) if isinstance(dataset, DatasetDict) else {subset or "dataset": dataset}
    for name, rows in subsets.items():
        if Path(name).name != name or name in {".", ".."}:
            raise ConfigurationError("Subset names must be single filenames for JSONL export")
        _write_jsonl(rows, export_dir / f"{name}.jsonl")
    if isinstance(dataset, DatasetDict):
        index = {"subsets": list(subsets), "total_rows": sum(len(rows) for rows in subsets.values())}
        temporary = export_dir / ".index.json.tmp"
        temporary.write_text(json.dumps(index, indent=2), encoding="utf-8")
        temporary.replace(export_dir / "index.json")


def custom_load_dataset(config: Any, subset: str | None = None) -> Dataset | DatasetDict:
    settings = _extract_settings(config)
    local = settings.local_dir
    if local and local.exists() and any(local.iterdir()):
        try:
            return _load_local(local, subset)
        except MissingSubsetError:
            pass  # Only absent named artifacts permit fallback, never damaged data.
    if _is_offline() or not getattr(config.hf_configuration, "push_to_hub", True):
        raise MissingSubsetError(f"Dataset subset '{subset}' not found locally in {local}")
    return _load_hub(_remote_repo(settings), subset, settings.token)


def custom_save_dataset(
    dataset: Dataset,
    config: Any,
    subset: str | None = None,
    *,
    save_local: bool | None = None,
    push_to_hub: bool | None = None,
) -> None:
    settings, local_enabled, remote_enabled = validate_storage(config, save_local=save_local, push_to_hub=push_to_hub)
    if subset is not None and (not subset or Path(subset).name != subset or subset in {".", ".."}):
        raise ConfigurationError("Subset names must be single filenames")
    if local_enabled:
        path = settings.local_dir
        existing = load_from_disk(str(path)) if path.exists() and any(path.iterdir()) else None
        merged = _merge_datasets(existing, dataset, subset, settings.concat_if_exist)
        _safe_save(merged, path)
        if settings.export_jsonl:
            _export_to_jsonl(merged, settings.jsonl_export_dir, subset)
    if remote_enabled:
        repo = _remote_repo(settings)
        if settings.concat_if_exist:
            # Append must read the old rows successfully; never replace an unreadable target.
            dataset = concatenate_datasets([_load_hub(repo, subset, settings.token), dataset])
        dataset.push_to_hub(
            repo_id=repo, private=settings.private, config_name=subset or "default", token=settings.token
        )


def replace_dataset_columns(dataset: Dataset, columns_data: dict[str, list]) -> Dataset:
    dataset = dataset.remove_columns([name for name in columns_data if name in dataset.column_names])
    for name, values in columns_data.items():
        dataset = dataset.add_column(name, values)
    return dataset
