"""Ridge regression algorithms and utilities.

This package contains ridge regression implementations and backend abstractions.
"""

from . import backends
from ._core import ridge_svd, ridge_cv

__all__ = ["backends", "ridge_svd", "ridge_cv"]
