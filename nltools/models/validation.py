"""The one validation check the nltools estimators share.

`Glm` and `Ridge` are independent estimators with different input contracts, so
they share this small private function instead of a common base class. It takes
the estimator as its first argument and reads only `is_fitted_` from it.

Feature and target validation is deliberately not shared: `Ridge` accepts named
feature-space mappings that no other estimator models, so it validates its own
inputs.
"""

from __future__ import annotations


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
