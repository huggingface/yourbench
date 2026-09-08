"""YourBench package version from installed distribution metadata."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("yourbench")
except PackageNotFoundError:
    __version__ = "development"


from yourbench.api import BenchmarkResult, run, create, load_result


__all__ = ["BenchmarkResult", "create", "load_result", "run", "__version__"]
