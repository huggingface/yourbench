"""Dataset card generation and upload functionality for HuggingFace Hub."""

import os
from typing import Any

from loguru import logger

from huggingface_hub import DatasetCard, DatasetCardData
from yourbench.conf.loader import get_enabled_stages
from yourbench.conf.prompts import load_prompt_from_package as _load_prompt_from_package


def _is_offline() -> bool:
    """Check if offline mode enabled."""
    return os.environ.get("HF_HUB_OFFLINE", "0").lower() in ("1", "true", "yes")


# Dataset card generation functions


def extract_readme_metadata(repo_id: str, token: str | None = None) -> str:
    """Extracts the metadata from the README.md file of the dataset repository.
    We have to download the previous README.md file in the repo, extract the metadata from it.
    Args:
        repo_id: The ID of the repository to push to, from the `push_to_hub` method.
        token: The token to authenticate with the Hugging Face Hub, from the `push_to_hub` method.
    Returns:
        The metadata extracted from the README.md file of the dataset repository as a str.
    """
    try:
        import re
        from pathlib import Path

        from huggingface_hub.file_download import hf_hub_download

        readme_path = Path(hf_hub_download(repo_id, "README.md", repo_type="dataset", token=token))
        # Extract the content between the '---' markers
        metadata_match = re.findall(r"---\n(.*?)\n---", readme_path.read_text(), re.DOTALL)

        if not metadata_match:
            logger.debug("No YAML metadata found in the README.md")
            return ""

        return metadata_match[0]

    except Exception as e:
        logger.debug(f"Failed to extract metadata from README.md: {e}")
        return ""


def extract_dataset_info(repo_id: str, token: str | None = None) -> str:
    """
    Extract dataset_info section from README metadata.

    Args:
        repo_id: The dataset repository ID
        token: Optional HuggingFace token for authentication

    Returns:
        The dataset_info section as a string, or empty string if not found
    """
    readme_metadata = extract_readme_metadata(repo_id=repo_id, token=token)
    if not readme_metadata:
        return ""

    section_prefix = "dataset_info:"
    if section_prefix not in readme_metadata:
        return ""

    try:
        # Extract the part after `dataset_info:` prefix
        config_data = section_prefix + readme_metadata.split(section_prefix)[1]
        return config_data
    except IndexError:
        logger.debug("Failed to extract dataset_info section from metadata")
        return ""


def _serialize_config_for_card(config: Any) -> str:
    """Publish a compact config without credentials or provider-specific extras."""
    import yaml

    from yourbench.conf.schema import YourbenchConfig
    from yourbench.conf.prompts import DEFAULT_PROMPTS

    if not isinstance(config, YourbenchConfig):
        # Compatibility for direct API callers; the CLI resolves Pydantic once.
        from omegaconf import OmegaConf

        config = YourbenchConfig.model_validate(OmegaConf.to_container(config, resolve=True))
    data = config.model_dump(exclude_defaults=True)
    hf = data.get("hf_configuration", {})
    hf.pop("hf_token", None)
    for model in data.get("model_list", []):
        if model.get("api_key"):
            model["api_key"] = "$API_KEY"
        model.pop("extra_parameters", None)
    data["pipeline"] = {
        name: getattr(config.pipeline, name).model_dump(exclude_defaults=True) for name in get_enabled_stages(config)
    }
    for stage in data["pipeline"].values():
        for field, value in list(stage.items()):
            if "prompt" in field:
                default = _load_prompt_from_package(DEFAULT_PROMPTS.get(field, ""))
                if value == default:
                    del stage[field]
                else:
                    stage[field] = f"custom_{field}.md"
    # Public cards must never publish an embedded URL credential/query.
    from urllib.parse import urlsplit, urlunsplit

    for model in data.get("model_list", []):
        if model.get("base_url"):
            url = urlsplit(model["base_url"])
            host = url.hostname or ""
            if url.port:
                host += f":{url.port}"
            model["base_url"] = urlunsplit((url.scheme, host, url.path, "", ""))
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _get_pipeline_subset_info(config: Any) -> str:
    """Describe enabled stages from the execution catalogue."""
    from yourbench.pipeline.registry import STAGES

    return "\n".join(f"- **{name}**: {STAGES[name].title}" for name in get_enabled_stages(config))


# Helper function to extract settings without circular import
def _extract_settings_impl(config: Any):
    """Extract HF settings from config. Import here to avoid circular dependency."""
    from yourbench.utils.dataset_engine import _extract_settings

    return _extract_settings(config)


def _generate_and_upload_dataset_card(config: Any, template_path: str | None = None) -> None:
    """Generate and upload a dataset card to Hugging Face Hub.

    Raises exceptions on failure - caller should handle.
    """
    settings = _extract_settings_impl(config)
    from yourbench.utils.dataset_engine import _remote_repo

    dataset_repo_name = _remote_repo(settings)
    token = settings.token

    logger.info(f"Uploading card for dataset: {dataset_repo_name}")

    # Load template
    if not template_path:
        current_dir = os.path.dirname(__file__)
        template_path = os.path.join(current_dir, "yourbench_card_template.md")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        template_str = f.read()

    # Extract dataset_info from existing README if available
    config_data = extract_dataset_info(repo_id=dataset_repo_name, token=token)

    # Get pretty_name
    hf_config = config.hf_configuration
    pretty_name = getattr(hf_config, "pretty_name", None)
    if not pretty_name:
        dataset_name = dataset_repo_name.split("/")[-1]
        pretty_name = dataset_name.replace("-", " ").replace("_", " ").title()

    card_data = DatasetCardData(pretty_name=pretty_name)

    # Get YourBench version
    from importlib.metadata import PackageNotFoundError, version

    try:
        version_str = version("yourbench")
    except PackageNotFoundError:
        version_str = "dev"

    footer = getattr(hf_config, "footer", None) or "*(This dataset card was automatically generated by YourBench)*"

    template_vars = {
        "pretty_name": card_data.pretty_name,
        "yourbench_version": version_str,
        "config_yaml": _serialize_config_for_card(config),
        "pipeline_subsets": _get_pipeline_subset_info(config),
        "config_data": config_data,
        "footer": footer,
    }

    card = DatasetCard.from_template(card_data=card_data, template_str=template_str, **template_vars)
    card.push_to_hub(dataset_repo_name, token=token)

    logger.success(f"Dataset card uploaded to: https://huggingface.co/datasets/{dataset_repo_name}")


def upload_dataset_card(config: Any) -> None:
    """Upload a dataset card to Hugging Face Hub.

    Checks config settings before uploading. Raises on failure.
    """
    hf_config = config.hf_configuration
    upload_card = getattr(hf_config, "upload_card", True)
    if upload_card is None:
        upload_card = True

    if not upload_card or not getattr(hf_config, "push_to_hub", True):
        logger.info("Dataset card upload disabled in configuration")
        return

    if _is_offline():
        logger.info("Offline mode enabled, skipping dataset card upload")
        return

    _generate_and_upload_dataset_card(config)
