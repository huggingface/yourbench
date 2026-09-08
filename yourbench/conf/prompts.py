"""Prompt loading utilities for Hydra configs."""

from pathlib import Path
from importlib.resources import files

from loguru import logger


DEFAULT_PROMPTS = {
    "pdf_llm_prompt": "ingestion/pdf_llm_prompt.md",
    "summarization_user_prompt": "summarization/summarization_user_prompt.md",
    "combine_summaries_user_prompt": "summarization/combine_summaries_user_prompt.md",
    "single_hop_system_prompt": "question_generation/single_hop_system_prompt.md",
    "single_hop_system_prompt_multi": "question_generation/single_hop_system_prompt_multi.md",
    "single_hop_user_prompt": "question_generation/single_hop_user_prompt.md",
    "multi_hop_system_prompt": "question_generation/multi_hop_system_prompt.md",
    "multi_hop_system_prompt_multi": "question_generation/multi_hop_system_prompt_multi.md",
    "multi_hop_user_prompt": "question_generation/multi_hop_user_prompt.md",
    "question_rewriting_system_prompt": "question_rewriting/question_rewriting_system_prompt.md",
    "question_rewriting_user_prompt": "question_rewriting/question_rewriting_user_prompt.md",
}


def load_prompt_from_package(package_path: str) -> str | None:
    """Load prompt content from package resources."""
    try:
        prompts_files = files("yourbench.prompts")
        parts = package_path.split("/")
        current = prompts_files

        for part in parts[:-1]:
            current = current / part

        file_resource = current / parts[-1]
        if file_resource.is_file():
            return file_resource.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.debug(f"Failed to load prompt from package {package_path}: {e}")
    return None


def load_prompt(value: str, default_package_path: str = "", *, base_dir: Path | None = None) -> str:
    """Load inline text or a prompt file. Missing explicit files are errors."""
    if not value:
        if not default_package_path:
            return ""
        content = load_prompt_from_package(default_package_path)
        if content is None:
            raise FileNotFoundError(f"Default prompt not found: {default_package_path}")
        return content
    if value.startswith("inline:"):
        return value[len("inline:") :]
    explicit_file = value.startswith("file:")
    candidate = value[len("file:") :] if explicit_file else value
    if not explicit_file and ("\n" in candidate or len(candidate) > 300):
        return candidate
    path = Path(candidate).expanduser()
    if explicit_file or path.suffix.lower() in {".md", ".txt", ".prompt"}:
        resolved = (base_dir or Path.cwd()) / path
        if resolved.is_file():
            return resolved.read_text(encoding="utf-8").strip()
        if not explicit_file:
            content = load_prompt_from_package(candidate)
            if content is not None:
                return content
        raise FileNotFoundError(f"Prompt file not found: {resolved}")
    return value
