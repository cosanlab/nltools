"""Expose the installed nltools version. Single source of truth: pyproject.toml.

`__version__` is read from the installed package metadata (populated by hatchling
from the `version` field in pyproject.toml), so the release bump in pyproject.toml
flows through automatically — there is no version string to keep in sync here.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nltools")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"
