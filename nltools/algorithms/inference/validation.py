"""Shared input validation for the algorithms module.

One home for the argument checks the permutation, bootstrap, and matrix tests
share, so every entry point raises the same `ValueError` for the same mistake.

Examples:
    ```python
    from nltools.algorithms.inference.validation import validate_tail_parameter

    validate_tail_parameter(2)  # → 'two'
    validate_tail_parameter("invalid")  # raises ValueError
    ```
"""

import numpy as np

from .utils import _normalize_tail_internal


def validate_tail_parameter(tail: int | str) -> str:
    """Validate the public tail vocabulary and normalize to the internal form.

    The public vocabulary is deliberately two-valued: the *direction* of a
    one-tailed test is fixed by the test's convention, never chosen from the
    data (a data-driven direction would silently halve every p-value). A fixed
    direction across all tests is what keeps multiple-comparison correction
    (FDR, Bonferroni) valid (GH #315).

    Args:
        tail (int | str): `2` or `'two'` (the default everywhere) for a
            two-tailed test (`|obs|` vs `|null|`); `1` or `'one'` for a
            one-tailed test in the test's canonical positive direction
            (correlation/ISC/similarity > 0, mean > popmean, group1 > group2).
            To test the negative direction, negate your data, swap the groups,
            or flip the contrast.

    Returns:
        str: Normalized internal tail, `'two'` or `'upper'`.

    Raises:
        ValueError: If `tail` is not a valid option (including the removed
            `'upper'`/`'lower'`/`-1` forms).
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
        array (np.ndarray): Array to validate.
        expected_ndim (int): Expected number of dimensions.
        name (str): Name of the array for the error message.

    Raises:
        ValueError: If the array has the wrong number of dimensions.
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
    """Validate that array dimensionality falls within a range.

    Args:
        array (np.ndarray): Array to validate.
        min_ndim (int): Minimum number of dimensions (inclusive).
        max_ndim (int): Maximum number of dimensions (inclusive).
        name (str): Name of the array for the error message.

    Raises:
        ValueError: If the array has the wrong number of dimensions.
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
    """Validate that two arrays have the same shape.

    Args:
        array1 (np.ndarray): First array.
        array2 (np.ndarray): Second array.
        name1 (str): Name of the first array for the error message.
        name2 (str): Name of the second array for the error message.

    Raises:
        ValueError: If the arrays have different shapes.
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
    """Validate a metric name against an allowed list.

    Args:
        metric (str): Metric name to validate.
        allowed (list[str]): Allowed metric names.
        name (str): Name of the parameter for the error message.

    Raises:
        ValueError: If `metric` is not in `allowed`.
    """
    if metric not in allowed:
        allowed_str = ", ".join(f"'{m}'" for m in allowed)
        raise ValueError(f"{name} must be one of [{allowed_str}], got {metric!r}")


def validate_how_parameter(how: str) -> None:
    """Validate the `how` parameter for matrix operations.

    Args:
        how (str): `'upper'`, `'lower'`, or `'full'`.

    Raises:
        ValueError: If `how` is not one of those values.
    """
    if how not in ["upper", "lower", "full"]:
        raise ValueError(f"how must be 'upper', 'lower', or 'full', got {how!r}")


def validate_square_matrix(matrix: np.ndarray, name: str = "matrix") -> None:
    """Validate that a matrix is square.

    Args:
        matrix (np.ndarray): Matrix to validate.
        name (str): Name of the matrix for the error message.

    Raises:
        ValueError: If the matrix is not square.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square, got shape {matrix.shape}")


def validate_shape_compatibility(
    X: np.ndarray,
    y: np.ndarray,
    X_name: str = "X",
    y_name: str = "y",
) -> None:
    """Validate that X and y have the same number of samples.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector or matrix.
        X_name (str): Name of X for the error message.
        y_name (str): Name of y for the error message.

    Raises:
        ValueError: If the first dimensions differ.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{X_name} and {y_name} must have same first dimension (n_samples), "
            f"got {X.shape[0]} and {y.shape[0]}"
        )


def validate_bootstrap_method(
    method: str, simple_methods: list[str], fitted_methods: list[str]
) -> None:
    """Validate a bootstrap method name.

    Args:
        method (str): Method name to validate.
        simple_methods (list[str]): Methods that need no fitted model.
        fitted_methods (list[str]): Methods that require a prior `.fit()`.

    Raises:
        ValueError: If `method` is in neither list.
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
        data (np.ndarray): 1D or 2D data with at least 2 samples along axis 0.
        method (str): Bootstrap method name (reserved for method-specific checks).

    Raises:
        ValueError: If the data is not 1D/2D or has fewer than 2 samples.
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


def validate_n_samples(n_samples: int) -> None:
    """Reject a replicate count a bootstrap cannot be computed from.

    Two replicates are the fewest a `ddof=1` standard error can be computed
    from, so that is the hard floor. The separate quality advisory lives in
    `_advise_on_n_samples`, in `nltools/algorithms/inference/bootstrap.py`.

    Args:
        n_samples (int): Number of bootstrap replicates.

    Raises:
        TypeError: If `n_samples` is not an integer.
        ValueError: If `n_samples` is below 2.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise TypeError(f"n_samples must be an integer, got {type(n_samples).__name__}")

    if n_samples < 2:
        raise ValueError(
            f"n_samples must be at least 2, got {n_samples}. "
            f"A bootstrap standard error needs at least two replicates. "
            f"Recommended: n_samples >= 1000 for confidence intervals."
        )


def validate_confidence_level(confidence_level: float) -> None:
    """Validate the interval confidence level.

    Args:
        confidence_level (float): Requested level.

    Raises:
        TypeError: If `confidence_level` is not a real number.
        ValueError: If it is not finite and strictly between zero and one.
    """
    if isinstance(confidence_level, bool) or not isinstance(
        confidence_level, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"confidence_level must be a number, got {type(confidence_level).__name__}"
        )
    value = float(confidence_level)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(
            f"confidence_level must be finite and strictly between 0 and 1, got "
            f"{confidence_level!r}. Use 0.95 for a 95% interval."
        )


def validate_memory_budget(memory_budget_gb: float | None) -> None:
    """Validate an explicit working-memory budget.

    Args:
        memory_budget_gb (float | None): Budget in GB, or None to measure the
            device.

    Raises:
        TypeError: If a supplied budget is not a real number.
        ValueError: If a supplied budget is not finite and positive.
    """
    if memory_budget_gb is None:
        return
    if isinstance(memory_budget_gb, bool) or not isinstance(
        memory_budget_gb, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"memory_budget_gb must be a number or None, got "
            f"{type(memory_budget_gb).__name__}"
        )
    value = float(memory_budget_gb)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"memory_budget_gb must be finite and positive, got {memory_budget_gb!r}."
        )
