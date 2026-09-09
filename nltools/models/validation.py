"""Input validation helpers shared by the nltools estimators.

`Glm` and `Ridge` are independent estimators with different input contracts, so
they share these small private functions instead of a common base class. Each
takes the estimator as its first argument and reads only `is_fitted_` and
`n_features_in_` from it.
"""

from __future__ import annotations

import numpy as np


def _check_is_fitted(model: object) -> None:
    """Raise if `model` has not been fitted yet.

    Args:
        model (object): Estimator carrying an `is_fitted_` flag.

    Raises:
        ValueError: If the model has not been fitted yet.
    """
    if not model.is_fitted_:
        raise ValueError(
            f"{model.__class__.__name__} instance is not fitted yet. "
            "Call 'fit' with appropriate arguments before using this model."
        )


def _validate_X(model: object, X: np.ndarray, reset: bool = True) -> np.ndarray:
    """Validate a feature matrix and return it as an ndarray.

    Args:
        model (object): Estimator the features belong to; read for
            `n_features_in_` when `reset` is False.
        X (array-like): Input data to validate.
        reset (bool): If True, any feature count is allowed because this is a
            new fit. If False, the feature count must match training.

    Returns:
        np.ndarray: Validated and converted input array.

    Raises:
        ValueError: If `X` is not 2-D, or its features do not match training.
    """
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got {X.ndim}D array instead. "
            f"Reshape your data using array.reshape(-1, 1) for single feature."
        )

    if not reset and hasattr(model, "n_features_in_"):
        if X.shape[1] != model.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {model.__class__.__name__} "
                f"was fitted with {model.n_features_in_} features."
            )

    return X


def _validate_X_y(
    model: object, X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a feature matrix and its targets and return them as ndarrays.

    Args:
        model (object): Estimator the data belongs to.
        X (array-like): Input data.
        y (array-like): Target values.

    Returns:
        tuple[np.ndarray, np.ndarray]: The validated `X` and `y`.

    Raises:
        ValueError: If `X` is not 2-D, `y` has more than two dimensions, or
            their sample counts differ.
    """
    X = _validate_X(model, X)
    y = np.asarray(y)

    if y.ndim > 2:
        raise ValueError(f"y should be 1D or 2D array, got {y.ndim}D array instead.")

    if y.shape[0] != X.shape[0]:
        raise ValueError(
            f"X and y have inconsistent number of samples: {X.shape[0]} vs {y.shape[0]}"
        )

    return X, y
