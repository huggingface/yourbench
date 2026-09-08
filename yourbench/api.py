"""Small public entry points for creating, running, and reading local benchmarks."""

import json
from pathlib import Path
from dataclasses import dataclass


def _recipe_path(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    return path / "config.yaml" if path.is_dir() else path


@dataclass(frozen=True)
class BenchmarkResult:
    """References to local artifacts, not a guarantee of question quality.

    Reading artifacts never runs a model or fetches data from the Hub. After a failed
    run, stored datasets may belong to earlier completed stages or a previous run.
    """

    config_path: Path
    dataset_dir: Path | None
    jsonl_dir: Path | None
    output_subset: str

    @property
    def manifest_path(self) -> Path | None:
        return self.dataset_dir.parent / "run.json" if self.dataset_dir else None

    @property
    def status(self) -> str:
        return self._manifest().get("status", "unknown")

    def _manifest(self) -> dict:
        if self.manifest_path is None or not self.manifest_path.exists():
            return {"status": "planned" if (self.config_path.parent / "plan.json").exists() else "unknown"}
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Run manifest must be a JSON object")
        return manifest

    def load_dataset(self, subset: str | None = None):
        """Load a local Hugging Face Dataset; default to the configured evaluation subset."""
        from yourbench.utils.dataset_engine import _load_local

        if self.dataset_dir is None:
            raise ValueError("This recipe has no local dataset directory")
        return _load_local(self.dataset_dir, self.output_subset if subset is None else subset)

    def summary(self) -> dict:
        """Read status and local subset sizes without resolving model credentials."""
        from datasets import DatasetDict, load_from_disk

        subsets = {}
        if self.dataset_dir and self.dataset_dir.exists() and any(self.dataset_dir.iterdir()):
            dataset = load_from_disk(str(self.dataset_dir))
            named = dataset if isinstance(dataset, DatasetDict) else {"default": dataset}
            subsets = {name: {"rows": len(rows), "columns": rows.column_names} for name, rows in named.items()}
        manifest = self._manifest()
        return {
            "status": manifest.get("status", "unknown"),
            "run_id": manifest.get("run_id"),
            "config_path": str(self.config_path),
            "dataset_dir": str(self.dataset_dir) if self.dataset_dir else None,
            "jsonl_dir": str(self.jsonl_dir) if self.jsonl_dir else None,
            "output_subset": self.output_subset,
            "subsets": subsets,
        }


def load_result(path: str | Path) -> BenchmarkResult:
    """Open an output directory or YAML recipe without model credentials or execution.

    Storage paths follow the recipe, including absolute paths. After moving an
    output directory, update its recipe's paths before reading or rerunning it.
    """
    import yaml

    from yourbench.utils.env import expand_env_value, validate_env_expanded
    from yourbench.conf.schema import HFConfig, LightevalConfig

    recipe = _recipe_path(path)
    data = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Recipe must be a YAML mapping")
    hf = HFConfig.model_validate(data.get("hf_configuration") or {})

    def directory(field):
        value = getattr(hf, field)
        if not value:
            return None
        value = validate_env_expanded(expand_env_value(value), field)
        return (recipe.parent / Path(value).expanduser()).resolve()

    pipeline = data.get("pipeline", data.get("pipeline_config")) or {}
    export = LightevalConfig.model_validate(pipeline.get("prepare_lighteval") or {})
    return BenchmarkResult(
        recipe,
        directory("local_dataset_dir") if hf.local_saving else None,
        directory("jsonl_export_dir") if hf.local_saving and hf.export_jsonl else None,
        export.output_subset,
    )


def run(path: str | Path) -> BenchmarkResult:
    """Run a saved recipe, then return handles to its local artifacts.

    This synchronous function propagates configuration, model, and storage errors.
    In an async application, use ``await asyncio.to_thread(run, path)``.
    """
    from yourbench.conf.loader import load_config
    from yourbench.pipeline.handler import run_pipeline_with_config

    recipe = _recipe_path(path)
    run_pipeline_with_config(load_config(recipe))
    return load_result(recipe)


def create(
    brief: str,
    *,
    source: str | Path,
    output: str | Path,
    model: str,
    base_url: str | None = None,
    api_key_env: str | None = None,
    provider: str | None = None,
    plan_only: bool = False,
    max_tokens: int | None = None,
    concurrency: int = 8,
) -> BenchmarkResult:
    """Interpret a brief, save a recipe, and generate a local benchmark.

    ``plan_only=True`` still makes a model call. ``max_tokens`` is a per-response
    provider limit, not a total token or dollar budget. Pass a key's environment
    variable name rather than its value; recipes preserve only that reference.
    """
    from yourbench.planning import create_recipe

    _, recipe = create_recipe(
        brief,
        Path(source),
        Path(output),
        model,
        provider,
        base_url,
        api_key_env,
        max_tokens=max_tokens,
        concurrency=concurrency,
    )
    return load_result(recipe) if plan_only else run(recipe)
