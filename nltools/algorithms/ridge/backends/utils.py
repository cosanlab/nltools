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

    Args:
        backend: Name or module of the backend. Must be one of: "numpy", "torch",
            "torch_cuda".
        on_error: What to do if backend fails to load. Options:
            - "raise": Raise an exception (default).
            - "warn": Warn and keep previous backend.

    Returns:
        module: Module of the backend (e.g., numpy, torch backend module).

    Raises:
        ValueError: If backend is not in ALL_BACKENDS.
        ImportError: If backend dependencies are not installed (when on_error="raise").

    Examples:
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

    Returns:
        module: Module of the backend (e.g., numpy, torch backend module).

    Examples:
        >>> from nltools.algorithms.ridge.backends import get_backend
        >>> backend = get_backend()
        >>> backend.name
        'numpy'
    """
    module = importlib.import_module(__package__ + "." + CURRENT_BACKEND)
    return module


def _dtype_to_str(dtype):
    """Cast dtype to string, such as "float32", or "float64".

    Converts numpy, torch, cupy, and other dtype objects to their string
    representation. Handles None values and string inputs.

    Args:
        dtype: Data type to convert. Can be:
            - str: Returned as-is
            - None: Returned as None
            - numpy/cupy dtype: Uses dtype.name
            - torch dtype: Converts from "torch.float32" format
            - Other: Attempts numpy.dtype conversion

    Returns:
        str or None: String representation of dtype (e.g., "float32", "float64"),
            or None if input was None.

    Raises:
        NotImplementedError: If dtype cannot be converted to string.

    Examples:
        >>> import numpy as np
        >>> _dtype_to_str(np.float32)
        'float32'
        >>> _dtype_to_str("float64")
        'float64'
        >>> _dtype_to_str(None)
        None
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
    """Warn if dtype is not float32.

    GPU backends are much faster with single precision (float32). This function
    warns the user once if they are using a different dtype, encouraging them
    to cast to float32 for better performance.

    Args:
        dtype: Data type to check. Can be any dtype object or string.

    Notes:
        - Warning is only shown once per session (uses internal flag).
        - Only warns for GPU backends (checked via get_backend()).
        - Warning is suppressed if dtype is already float32.
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
