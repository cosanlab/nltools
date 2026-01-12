"""Shared validation utilities for algorithms module.

This module provides common validation functions to reduce code duplication
and ensure consistent error handling across the algorithms module.

Usage:
    These functions are used throughout the algorithms module to validate
    input parameters. They provide consistent error messages and behavior.

    Example:
        >>> from nltools.algorithms._validation import validate_parallel_parameter
        >>> validate_parallel_parameter("cpu")  # OK
        >>> validate_parallel_parameter("invalid")  # Raises ValueError
"""

from typing import Optional, Tuple
import numpy as np


def validate_parallel_parameter(parallel: Optional[str]) -> None:
    """Validate parallel parameter.

    Args:
        parallel: Parallel parameter value

    Raises:
        ValueError: If parallel is not None, 'cpu', or 'gpu'
    """
    if parallel is not None and parallel not in ["cpu", "gpu"]:
        raise ValueError(f"parallel must be None, 'cpu', or 'gpu', got: {parallel!r}")


def validate_parallel_parameter_matrix(parallel: Optional[str]) -> None:
    """Validate parallel parameter for matrix operations.

    Args:
        parallel: Parallel parameter value

    Raises:
        ValueError: If parallel is not None or 'cpu' (GPU not yet supported)
    """
    if parallel not in [None, "cpu"]:
        raise ValueError(
            f"parallel must be None or 'cpu', got {parallel!r}. "
            "GPU support not yet implemented for matrix permutation tests."
        )


def validate_tail_parameter(tail: int | str) -> str:
    """Validate and normalize tail parameter.

    Args:
        tail: Tail parameter value. Can be:
            - 'two' or 2: Two-tailed test (|obs| > |null|)
            - 'upper' or 1: One-tailed upper (obs > null, for testing positive effects)
            - 'lower' or -1: One-tailed lower (obs < null, for testing negative effects)

    Returns:
        Normalized tail string: 'two', 'upper', or 'lower'

    Raises:
        ValueError: If tail is not a valid option

    Notes:
        For multiple comparisons correction (FDR, Bonferroni), use 'upper' or 'lower'
        to ensure consistent direction across all tests. The old tail=1 behavior
        (auto-detecting direction per test based on sign) can lead to incorrect
        MCP-adjusted p-values. See GH #315.
    """
    # Normalize to string
    if tail == 2 or tail == "two":
        return "two"
    elif tail == 1 or tail == "upper":
        return "upper"
    elif tail == -1 or tail == "lower":
        return "lower"
    else:
        raise ValueError(
            f"tail must be 'two', 'upper', 'lower' (or 2, 1, -1), got {tail}"
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


def validate_same_first_dimension(
    array1: np.ndarray,
    array2: np.ndarray,
    name1: str = "array1",
    name2: str = "array2",
) -> None:
    """Validate two arrays have same first dimension.

    Args:
        array1: First array
        array2: Second array
        name1: Name of first array for error message
        name2: Name of second array for error message

    Raises:
        ValueError: If arrays have different first dimensions
    """
    if array1.shape[0] != array2.shape[0]:
        raise ValueError(
            f"{name1} and {name2} must have same first dimension, "
            f"got {array1.shape[0]} and {array2.shape[0]}"
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


def validate_n_samples(
    n_samples: int, min_samples: int = 2, name: str = "n_samples"
) -> None:
    """Validate number of samples.

    Args:
        n_samples: Number of samples
        min_samples: Minimum required samples
        name: Name of parameter for error message

    Raises:
        ValueError: If n_samples is less than min_samples
    """
    if n_samples < min_samples:
        raise ValueError(f"{name} must be at least {min_samples}, got {n_samples}")


def validate_percentiles(percentiles: Tuple[float, float]) -> None:
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


def validate_alpha(alpha: float, name: str = "alpha") -> None:
    """Validate regularization parameter alpha.

    Args:
        alpha: Regularization parameter
        name: Name of parameter for error message

    Raises:
        ValueError: If alpha is negative
    """
    if alpha < 0:
        raise ValueError(f"{name} must be >= 0, got {alpha}")


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


def validate_isc_parameters(
    metric: str,
    summary_statistic: str,
    method: Optional[str] = None,
) -> None:
    """Validate ISC parameter values.

    Args:
        metric: Summary statistic metric
        summary_statistic: ISC computation method
        method: Resampling method (optional)

    Raises:
        ValueError: If any parameter is invalid
    """
    if metric not in ["median", "mean"]:
        raise ValueError(f"metric must be 'median' or 'mean', got {metric}")

    if summary_statistic not in ["leave-one-out", "pairwise"]:
        raise ValueError(
            f"summary_statistic must be 'leave-one-out' or 'pairwise', "
            f"got {summary_statistic}"
        )

    if method is not None:
        if method not in ["permute", "bootstrap"]:
            raise ValueError(f"method must be 'permute' or 'bootstrap', got {method}")
