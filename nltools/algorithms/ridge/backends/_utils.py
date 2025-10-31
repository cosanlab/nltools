"""Backend utilities for ridge regression.

Provides backend switching and utility functions following himalaya's approach.
"""

import importlib
import types
import warnings

# Available backends (torch, torch_cuda require optional dependencies)
ALL_BACKENDS = [
    "numpy",
    "torch",
    "torch_cuda",
]

# Current backend (default: numpy)
CURRENT_BACKEND = "numpy"


def set_backend(backend, on_error="raise"):
    """Set the backend using a global variable, and return the backend module.

    Parameters
    ----------
    backend : str or module
        Name or module of the backend.
    on_error : str in {"raise", "warn"}
        Define what is done if the backend fails to be loaded.
        If "warn", this function only warns, and keeps the previous backend.
        If "raise", this function raises on errors.

    Returns
    -------
    module : python module
        Module of the backend.

    Raises
    ------
    ValueError
        If backend is not in ALL_BACKENDS.
    ImportError
        If backend dependencies are not installed (when on_error="raise").

    Examples
    --------
    >>> from nltools.algorithms.ridge.backends import set_backend
    >>> backend = set_backend("numpy")
    >>> backend.name
    'numpy'
    """
    global CURRENT_BACKEND

    try:
        if isinstance(backend, types.ModuleType):  # get name from module
            backend = backend.name

        if backend not in ALL_BACKENDS:
            raise ValueError(f"Unknown backend={backend!r}")

        module = importlib.import_module(__package__ + "." + backend)
        CURRENT_BACKEND = backend
    except Exception as error:
        if on_error == "raise":
            raise error
        elif on_error == "warn":
            warnings.warn(
                f"Setting backend to {backend} failed: {str(error)}. "
                f"Falling back to {CURRENT_BACKEND} backend."
            )
            module = get_backend()
        else:
            raise ValueError(f"Unknown value on_error={on_error!r}")

    return module


def get_backend():
    """Get the current backend module.

    Returns
    -------
    module : python module
        Module of the backend.

    Examples
    --------
    >>> from nltools.algorithms.ridge.backends import get_backend
    >>> backend = get_backend()
    >>> backend.name
    'numpy'
    """
    module = importlib.import_module(__package__ + "." + CURRENT_BACKEND)
    return module


def _dtype_to_str(dtype):
    """Cast dtype to string, such as "float32", or "float64".

    Parameters
    ----------
    dtype : dtype or str or None
        Data type to convert.

    Returns
    -------
    str or None
        String representation of dtype.
    """
    if isinstance(dtype, str):
        return dtype
    elif dtype is None:
        return None
    elif hasattr(dtype, "name"):  # works for numpy and cupy dtype instances
        return dtype.name
    elif "torch." in str(dtype):  # works for torch
        return str(dtype)[6:]
    else:
        # Try to convert to numpy dtype (handles np.float32, np.float64 types)
        try:
            import numpy as np

            dt = np.dtype(dtype)
            return dt.name
        except (TypeError, ValueError):
            pass
        raise NotImplementedError(f"Cannot convert dtype {dtype} to string")


_already_warned = [False]


def warn_if_not_float32(dtype):
    """Warn if X is not float32.

    GPU backends are much faster with single precision.

    Parameters
    ----------
    dtype : dtype
        Data type to check.
    """
    if _already_warned[0]:  # avoid warning multiple times
        return None

    if _dtype_to_str(dtype) != "float32":
        backend = get_backend()
        warnings.warn(
            f"GPU backend {backend.name} is much faster with single "
            f"precision floats (float32), got input in {dtype}. "
            "Consider casting your data to float32.",
            UserWarning,
        )
        _already_warned[0] = True
