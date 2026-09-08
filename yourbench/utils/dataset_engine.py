import os
import json
import shutil
import tempfile
from typing import Any, TypeVar
from pathlib import Path
from dataclasses import dataclass

from loguru import logger

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets
from huggingface_hub import HfApi, whoami
from huggingface_hub.utils import HFValidationError
from yourbench.utils.env import validate_env_expanded
from yourbench.utils.dataset_card import upload_dataset_card


__all__ = ["custom_load_dataset", "custom_save_dataset", "upload_dataset_card"]

T = TypeVar("T")


class ConfigurationError(Exception):
    """Configuration error."""


@dataclass(slots=True, frozen=True)
class HFSettings:
    """Normalized HuggingFace configuration."""

    dataset_name: str
    organization: str | None
    token: str | None
    local_dir: Path | None
    concat_if_exist: bool = False
    private: bool = True
    export_jsonl: bool = False
    jsonl_export_dir: Path | None = None

    @property
    def repo_id(self) -> str:
        """Full repository identifier."""
        if "/" in self.dataset_name:
            return self.dataset_name
        return f"{self.organization}/{self.dataset_name}" if self.organization else self.dataset_name


def _is_offline() -> bool:
    """Check if offline mode enabled."""
    return os.environ.get("HF_HUB_OFFLINE", "0").lower() in ("1", "true", "yes")


def _expand_var(value: str, field: str) -> str:
    """Ensure value is not unexpanded $VAR placeholder."""
    try:
        return validate_env_expanded(value, field)
    except ValueError as e:
        raise ConfigurationError(str(e)) from e


def _extract_settings(config) -> HFSettings:
    """Parse and validate configuration (OmegaConf DictConfig or dict)."""
    # Get hf_configuration - works with both attribute and dict access
    hf = config.hf_configuration
    if not hf:
        raise ConfigurationError("'hf_configuration' section missing")

    # Helper to get value from hf config (supports both dict and DictConfig)
    def get_val(key, default=None):
        return getattr(hf, key, default)

    dataset_name = get_val("hf_dataset_name", "")
    if not dataset_name:
        raise ConfigurationError("'hf_dataset_name' required")
    dataset_name = _expand_var(dataset_name, "hf_dataset_name")

    org_raw = get_val("hf_organization", "")
    token = get_val("hf_token", "") or os.getenv("HF_TOKEN")
    organization = _resolve_organization(org_raw, token) if get_val("push_to_hub", True) else (org_raw or None)

    local_raw = get_val("local_dataset_dir")
    local_dir = Path(local_raw).expanduser().resolve() if local_raw else None

    jsonl_raw = get_val("jsonl_export_dir")
    jsonl_dir = Path(jsonl_raw).expanduser().resolve() if jsonl_raw else None

    return HFSettings(
        dataset_name=dataset_name,
        organization=organization,
        token=token,
        local_dir=local_dir,
        concat_if_exist=get_val("concat_if_exist", False),
        private=get_val("private", True),
        export_jsonl=get_val("export_jsonl", False),
        jsonl_export_dir=jsonl_dir,
    )


def _resolve_organization(org: str | None, token: str | None) -> str | None:
    """Resolve organization, fetching from HF if needed."""
    if _is_offline() or (org and not org.startswith("$")):
        return org

    if org and org.startswith("$"):
        var_name = org[1:].split("/")[0]
        logger.warning(f"Environment variable '{var_name}' in 'hf_organization' not set")

    if not token:
        return None

    try:
        if username := whoami(token=token).get("name"):
            logger.info(f"Using '{username}' as organization")
            return username
    except HFValidationError:
        logger.warning("Invalid HF token")
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Network error fetching organization: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching organization: {e}")

    return None


def _validate_repo(settings: HFSettings) -> None:
    """Validate repository ID format."""
    if _is_offline():
        return

    try:
        HfApi().repo_info(repo_id=settings.repo_id, repo_type="dataset", token=settings.token)
    except HFValidationError as e:
        raise ConfigurationError(f"Invalid repo ID '{settings.repo_id}': {e}") from e
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Network error validating repo: {e}")
    except Exception as e:
        if "404" not in str(e):
            logger.error(f"Unexpected error validating repo: {e}")
            raise


def _load_local(path: Path, subset: str | None) -> Dataset:
    """Load dataset from local path."""
    logger.info(f"Loading '{subset or 'default'}' from {path}")
    dataset = load_from_disk(str(path))

    if subset is None:
        return dataset

    if not isinstance(dataset, DatasetDict):
        raise FileNotFoundError(f"Named subset '{subset}' requires a DatasetDict at {path}")

    if subset in dataset:
        return dataset[subset]

    # Provide a helpful error message showing available subsets
    available_subsets = list(dataset.keys())
    raise FileNotFoundError(f"Subset '{subset}' not found in local dataset. Available subsets: {available_subsets}")


def _load_hub(repo_id: str, subset: str | None, token: str | None) -> Dataset:
    """Load dataset from HuggingFace Hub."""
    logger.info(f"Loading '{subset or 'default'}' from Hub: {repo_id}")

    try:
        dataset = load_dataset(repo_id, name=subset, split="train", token=token)
        if len(dataset) == 0:
            raise ValueError(f"Dataset from Hub is empty (repo: {repo_id}, subset: {subset})")
        return dataset
    except ValueError as e:
        if "BuilderConfig" in str(e) and "not found" in str(e):
            raise FileNotFoundError(f"Subset '{subset}' not found on Hub") from e
        if "split" in str(e):
            raise ConfigurationError("Split 'train' not found in dataset") from e
        raise


def _merge_datasets(
    existing: Dataset | DatasetDict, new: Dataset, subset: str | None, concat_if_exist: bool = False
) -> Dataset | DatasetDict:
    """Merge new dataset with existing. If subset exists and concat_if_exist is True, new data is concatenated."""
    if subset is None:
        if isinstance(existing, Dataset):
            if concat_if_exist:
                return concatenate_datasets([existing, new])
            else:
                return new
        return new

    if not isinstance(existing, DatasetDict):
        existing = DatasetDict({"default": existing})

    if subset in existing and concat_if_exist:
        new = concatenate_datasets([existing[subset], new])

    existing[subset] = new
    return existing


def _safe_save(dataset: Dataset | DatasetDict, path: Path) -> None:
    """Save dataset, handling overwrite issues."""
    try:
        dataset.save_to_disk(str(path))
        logger.success(f"Saved to {path}")
    except PermissionError as e:
        if "can't overwrite itself" not in str(e):
            raise

        with tempfile.TemporaryDirectory() as tmp:
            dataset.save_to_disk(tmp)
            shutil.rmtree(path, ignore_errors=True)
            shutil.copytree(tmp, path)
        logger.success(f"Saved to {path} (via temp)")


def _export_to_jsonl(dataset: Dataset | DatasetDict, export_dir: Path, subset: str | None = None) -> None:
    """Export dataset to JSONL format."""
    export_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(dataset, Dataset):
        # Single dataset - export to single file
        file_name = f"{subset}.jsonl" if subset else "dataset.jsonl"
        file_path = export_dir / file_name

        logger.info(f"Exporting dataset to JSONL: {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            for row in dataset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.success(f"Exported {len(dataset)} rows to {file_path}")

    elif isinstance(dataset, DatasetDict):
        # Multiple subsets - export each to separate file
        logger.info(f"Exporting DatasetDict with {len(dataset)} subsets to JSONL")
        for subset_name, subset_data in dataset.items():
            # Use recursive call for all subsets to handle prepared_lighteval specially
            _export_to_jsonl(subset_data, export_dir, subset_name)

        # Create an index file listing all subsets
        index_path = export_dir / "index.json"
        index_data = {
            "subsets": list(dataset.keys()),
            "total_rows": sum(len(subset_data) for subset_data in dataset.values()),
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)
        logger.info(f"Created index file: {index_path}")


def custom_load_dataset(config: Any, subset: str | None = None) -> Dataset:
    """Load a named local artifact, using Hub only when remote access is enabled.

    Missing artifacts raise FileNotFoundError; corruption and network failures
    propagate unchanged instead of being disguised as absent data.
    """
    settings = _extract_settings(config)
    if settings.local_dir and settings.local_dir.exists() and any(settings.local_dir.iterdir()):
        try:
            return _load_local(settings.local_dir, subset)
        except FileNotFoundError:
            if _is_offline() or not getattr(config.hf_configuration, "push_to_hub", True):
                raise
    if _is_offline() or not getattr(config.hf_configuration, "push_to_hub", True):
        raise FileNotFoundError(f"Dataset subset '{subset}' not found locally in {settings.local_dir}")
    _validate_repo(settings)
    return _load_hub(settings.repo_id, subset, settings.token)


def custom_save_dataset(
    dataset: Dataset,
    config: Any,
    subset: str | None = None,
    *,
    save_local: bool | None = None,
    push_to_hub: bool | None = None,
) -> None:
    """Save dataset locally and/or push to Hub."""
    settings = _extract_settings(config)
    hf = config.hf_configuration
    save_local = getattr(hf, "local_saving", True) if save_local is None else save_local
    push_to_hub = getattr(hf, "push_to_hub", True) if push_to_hub is None else push_to_hub

    if _is_offline():
        save_local = True
        push_to_hub = False
        logger.info("Offline mode - only saving locally")

    if save_local and settings.local_dir:
        logger.info(f"Saving to {settings.local_dir}")

        existing = None
        if settings.local_dir.exists():
            try:
                existing = load_from_disk(str(settings.local_dir))
            except (FileNotFoundError, PermissionError, OSError) as e:
                logger.warning(f"Error loading existing dataset from disk: {e}")
            except Exception as e:
                logger.error(f"Unexpected error loading existing dataset: {e}")
                raise

        merged = (
            _merge_datasets(existing, dataset, subset, settings.concat_if_exist)
            if existing
            else (DatasetDict({subset: dataset}) if subset else dataset)
        )

        settings.local_dir.parent.mkdir(parents=True, exist_ok=True)
        _safe_save(merged, settings.local_dir)

        # Export to JSONL if enabled
        if settings.export_jsonl and settings.jsonl_export_dir:
            logger.info("JSONL export is enabled")
            _export_to_jsonl(merged, settings.jsonl_export_dir, subset)

    if push_to_hub and not _is_offline():
        if settings.concat_if_exist:
            try:
                existing = _load_hub(settings.repo_id, subset, settings.token)
            except FileNotFoundError:
                pass
            else:
                dataset = concatenate_datasets([existing, dataset])
                logger.info("Concatenated with existing remote")

        _validate_repo(settings)
        logger.info(f"Pushing to Hub: {settings.repo_id}")
        dataset.push_to_hub(
            repo_id=settings.repo_id,
            private=settings.private,
            config_name=subset or "default",
            token=settings.token,
        )
        logger.success(f"Pushed to Hub: {settings.repo_id}")


def replace_dataset_columns(
    dataset: Dataset, columns_data: dict[str, list], preserve_metadata: bool = False
) -> Dataset:
    """Replace columns by removing existing and adding new ones."""
    to_remove = [col for col in columns_data if col in dataset.column_names]

    if to_remove:
        logger.info(f"Removing columns: {to_remove}")
        dataset = dataset.remove_columns(to_remove)

    for name, data in columns_data.items():
        dataset = dataset.add_column(name, data)

    return dataset
