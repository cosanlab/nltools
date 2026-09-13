"""The one validation check the nltools estimators share.

`_Glm` and `_Ridge` are independent estimators with different input contracts, so
they share this small private function instead of a common base class.
"""

from __future__ import annotations

from sklearn.exceptions import NotFittedError


def _check_is_fitted(model: object) -> None:
    """Raise if `model` has not been fitted yet.

    Args:
        model (object): Estimator carrying an `is_fitted_` flag.

    Raises:
        NotFittedError: If the model has not been fitted yet. `NotFittedError`
            subclasses both `ValueError` and `AttributeError`.
    """
    if not model.is_fitted_:
        raise NotFittedError(
            f"This {type(model).__name__} instance is not fitted yet. "
            "Call 'fit' with appropriate arguments before using this estimator."
        )
