"""Backend abstraction for ridge regression.

This module provides a module-based backend system following himalaya's approach.
Supports NumPy (CPU), PyTorch CPU, and PyTorch CUDA backends.

The backend can be switched at runtime to leverage different computational backends
(CPU or GPU) without changing code.

Examples
--------
>>> from nltools.algorithms.ridge.backends import set_backend, get_backend
>>> backend = set_backend("numpy")  # Use NumPy (default)
>>> backend.name
'numpy'

>>> # Switch to GPU (if available)
>>> backend = set_backend("torch_cuda")
>>> X_gpu = backend.asarray(X, device="cuda")
>>> U, s, Vt = backend.svd(X_gpu)  # GPU-accelerated
"""

from .utils import (
    ALL_BACKENDS,
    set_backend,
    get_backend,
    _dtype_to_str,
    warn_if_not_float32,
)

__all__ = [
    "ALL_BACKENDS",
    "set_backend",
    "get_backend",
    "_dtype_to_str",
    "warn_if_not_float32",
]
