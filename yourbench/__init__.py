"""YourBench package version from installed distribution metadata."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("yourbench")
except PackageNotFoundError:
    __version__ = "development"
