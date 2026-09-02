"""Shared validation utilities for algorithms module.

This module provides common validation functions to reduce code duplication
and ensure consistent error handling across the algorithms module.

These functions are used throughout the algorithms module to validate input
parameters. They provide consistent error messages and behavior.

Examples:
    ```python
    from nltools.algorithms.inference.validation import validate_device_parameter

    validate_device_parameter("cpu")  # OK
    validate_device_parameter("invalid")  # raises ValueError
    ```
"""

import numpy as np

from .utils import _normalize_tail_internal


def validate_device_parameter(device: str | None, *, allow_auto: bool = False) -> None:
    """Validate device parameter.

    Args:
        device: Device parameter value (None, 'cpu', or 'gpu')
        allow_auto: Also accept 'auto' (entry points that resolve the device
            themselves, e.g. `phase_randomize`)

    Raises:
        ValueError: If device is not one of the accepted values
    """
    allowed = [None, "cpu", "gpu"] + (["auto"] if allow_auto else [])
    if device not in allowed:
        options = (
            "None, 'cpu', 'gpu', or 'auto'" if allow_auto else "None, 'cpu', or 'gpu'"
        )
        raise ValueError(f"device must be {options}, got {device!r}")


def validate_device_parameter_matrix(device: str | None) -> None:
    """Validate device parameter for matrix operations.

    Args:
        device: Parallel parameter value

    Raises:
        ValueError: If device is not None or 'cpu' (GPU not yet supported)
    """
    if device not in [None, "cpu"]:
        raise ValueError(
            f"device must be None or 'cpu', got {device!r}. "
            "GPU support not yet implemented for matrix permutation tests."
        )


def validate_tail_parameter(tail: int | str) -> str:
    """Validate the public tail vocabulary and normalize to the internal form.

    The public vocabulary (v0.6.0) is deliberately two-valued — the *direction*
    of a one-tailed test is fixed by the test's convention, never chosen from
    the data (a data-driven direction would silently halve every p-value):

    Args:
        tail: Tail parameter value. Can be:
            - 2 or 'two' (default everywhere): two-tailed test (|obs| vs |null|)
            - 1 or 'one': one-tailed test in the test's canonical positive
              direction (correlation/ISC/similarity > 0, mean > popmean,
              group1 > group2). To test the negative direction, negate your
              data, swap the groups, or flip the contrast.

    Returns:
        Normalized internal tail string: 'two' or 'upper'

    Raises:
        ValueError: If tail is not a valid option (including the removed
            v0.5 forms 'upper'/'lower'/-1)

    Notes:
        For multiple comparisons correction (FDR, Bonferroni) a fixed direction
        across all tests is essential — which is exactly why the direction is
        part of the vocabulary, not the data. See GH #315.
    """
    # One mapping table lives in `_normalize_tail_internal`; the public layer
    # only rejects the internal-only directional forms it must not accept.
    if tail not in (-1, "upper", "lower"):
        try:
            return _normalize_tail_internal(tail)
        except ValueError:
            pass
    raise ValueError(
        f"tail must be 2|'two' (two-tailed) or 1|'one' (one-tailed, the test's "
        f"positive direction), got {tail!r}. The 'upper'/'lower'/-1 forms were "
        "removed in v0.6.0: to test the negative direction, negate your data, "
        "swap the groups, or flip the contrast."
    )


def validate_array_shape(
    array: np.ndarray,
    expected_ndim: int,
    name: str = "array",
) -> None:
    """Validate array dimensionality.

    Args:
        array: Array to validate
        expected_ndim: Expected number of dimensions
        name: Name of array for error message

    Raises:
        ValueError: If array has wrong number of dimensions
    """
    if array.ndim != expected_ndim:
        raise ValueError(
            f"{name} must be {expected_ndim}D, got shape {array.shape} ({array.ndim}D)"
        )


def validate_array_shape_range(
    array: np.ndarray,
    min_ndim: int,
    max_ndim: int,
    name: str = "array",
) -> None:
    """Validate array dimensionality is within a range.

    Args:
        array: Array to validate
        min_ndim: Minimum number of dimensions (inclusive)
        max_ndim: Maximum number of dimensions (inclusive)
        name: Name of array for error message

    Raises:
        ValueError: If array has wrong number of dimensions
    """
    if not (min_ndim <= array.ndim <= max_ndim):
        raise ValueError(
            f"{name} must be {min_ndim}D to {max_ndim}D, got shape {array.shape} "
            f"({array.ndim}D)"
        )


def validate_same_shape(
    array1: np.ndarray,
    array2: np.ndarray,
    name1: str = "array1",
    name2: str = "array2",
) -> None:
    """Validate two arrays have same shape.

    Args:
        array1: First array
        array2: Second array
        name1: Name of first array for error message
        name2: Name of second array for error message

    Raises:
        ValueError: If arrays have different shapes
    """
    if array1.shape != array2.shape:
        raise ValueError(
            f"{name1} and {name2} must have same shape, "
            f"got {array1.shape} and {array2.shape}"
        )


def validate_metric_parameter(
    metric: str,
    allowed: list[str],
    name: str = "metric",
) -> None:
    """Validate metric parameter.

    Args:
        metric: Metric parameter value
        allowed: List of allowed metric values
        name: Name of parameter for error message

    Raises:
        ValueError: If metric is not in allowed list
    """
    if metric not in allowed:
        allowed_str = ", ".join(f"'{m}'" for m in allowed)
        raise ValueError(f"{name} must be one of [{allowed_str}], got {metric!r}")


def validate_how_parameter(how: str) -> None:
    """Validate 'how' parameter for matrix operations.

    Args:
        how: How parameter value

    Raises:
        ValueError: If how is not 'upper', 'lower', or 'full'
    """
    if how not in ["upper", "lower", "full"]:
        raise ValueError(f"how must be 'upper', 'lower', or 'full', got {how!r}")


def validate_square_matrix(matrix: np.ndarray, name: str = "matrix") -> None:
    """Validate matrix is square.

    Args:
        matrix: Matrix to validate
        name: Name of matrix for error message

    Raises:
        ValueError: If matrix is not square
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square, got shape {matrix.shape}")


def validate_percentiles(percentiles: tuple[float, float]) -> None:
    """Validate percentile values for confidence intervals.

    Args:
        percentiles: Percentile values (lower, upper)

    Raises:
        ValueError: If percentiles are invalid
    """
    if not isinstance(percentiles, (tuple, list)) or len(percentiles) != 2:
        raise ValueError(f"percentiles must be a tuple of 2 values, got {percentiles}")

    lower, upper = percentiles

    if not (0 < lower < 50):
        raise ValueError(f"Lower percentile must be between 0 and 50, got {lower}")

    if not (50 < upper < 100):
        raise ValueError(f"Upper percentile must be between 50 and 100, got {upper}")

    if lower >= upper:
        raise ValueError(
            f"Lower percentile ({lower}) must be less than upper ({upper})"
        )


def validate_shape_compatibility(
    X: np.ndarray,
    y: np.ndarray,
    X_name: str = "X",
    y_name: str = "y",
) -> None:
    """Validate that X and y have compatible shapes for regression.

    Args:
        X: Feature matrix
        y: Target vector or matrix
        X_name: Name of X for error message
        y_name: Name of y for error message

    Raises:
        ValueError: If shapes are incompatible
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{X_name} and {y_name} must have same first dimension (n_samples), "
            f"got {X.shape[0]} and {y.shape[0]}"
        )


def validate_bootstrap_method(
    method: str, simple_methods: list[str], fitted_methods: list[str]
) -> None:
    """Validate bootstrap method name.

    Args:
        method: Method name to validate
        simple_methods: List of simple method names
        fitted_methods: List of fitted method names

    Raises:
        ValueError: If method is not supported
    """
    supported = simple_methods + fitted_methods
    if method not in supported:
        raise ValueError(
            f"Unsupported method '{method}'. "
            f"Supported methods: {simple_methods} (simple methods), "
            f"{fitted_methods} (fitted model methods). "
            f"For fitted methods, you must call .fit() first."
        )


def validate_bootstrap_data(data: np.ndarray, method: str) -> None:
    """Validate input data for bootstrapping.

    Args:
        data: Data to validate
        method: Bootstrap method

    Raises:
        ValueError: If data is invalid (wrong shape, too few samples, etc.)
    """
    # Check dimensionality
    if data.ndim not in [1, 2]:
        raise ValueError(
            f"Data must be 1D or 2D, got shape {data.shape}. "
            f"For 3D+ data, you may need to reshape or select specific dimensions."
        )

    # Check number of samples
    n_samples = data.shape[0] if data.ndim == 2 else len(data)
    if n_samples < 2:
        raise ValueError(
            f"Need at least 2 samples for bootstrap, got {n_samples}. "
            f"Bootstrap requires resampling, which needs multiple samples."
        )
