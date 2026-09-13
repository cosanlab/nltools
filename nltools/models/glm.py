"""General linear model estimator built on Nilearn's `run_glm`.

`_Glm` fits one run represented by a `(DesignMatrix, y)` pair. It knows nothing
about `BrainData`, masks, NIfTI images, events, or multi-run orchestration: the
caller supplies a precomputed design (including any intercept column) and a
preprocessed response. Fitting delegates to `nilearn.glm.first_level.run_glm`
and contrast inference to Nilearn's `Contrast`, so every number this module
returns is Nilearn's.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from nilearn.glm import Contrast, expression_to_contrast_vector
from nilearn.glm.first_level import run_glm

from .results import ContrastResult
from .validation import _check_is_fitted

if TYPE_CHECKING:
    from nltools.data import DesignMatrix

# `arN` with N a positive integer, matching Nilearn's autoregressive orders.
AR_NOISE_MODEL = re.compile(r"^ar[1-9][0-9]*$")


@dataclass(frozen=True)
class _GlmFitState:
    """Immutable record of one fitted Nilearn GLM, sufficient to compute contrasts.

    Nilearn's `RegressionResults` also hold the response, the whitened
    response, the regression model, and the residuals. `_Glm` extracts this
    much and discards those objects, so the retained state stays proportional
    to the design rather than to the data. Its arrays keep the dtypes Nilearn
    produced.

    Attributes:
        feature_names (tuple[str, ...]): Fitted design column names in
            coefficient order.
        labels (np.ndarray): Nilearn's per-target model label, shape
            `(n_targets,)`. Every target sharing a label was fitted by the same
            regression model.
        coefficients (np.ndarray): Fitted coefficients, shape
            `(n_features, n_targets)`.
        covariances (dict[Any, np.ndarray]): Normalized parameter covariance of
            shape `(n_features, n_features)` for each fitted label.
        dispersion (np.ndarray): Per-target dispersion (the whitened residual
            mean square), shape `(n_targets,)`.
        residual_degrees_of_freedom (float): Residual degrees of freedom of the
            fitted design.
    """

    feature_names: tuple[str, ...]
    labels: np.ndarray
    coefficients: np.ndarray
    covariances: dict[Any, np.ndarray]
    dispersion: np.ndarray
    residual_degrees_of_freedom: float


def _check_noise_model(noise_model: object) -> None:
    """Raise unless `noise_model` is `'ols'` or `'arN'` with N a positive integer."""
    if not isinstance(noise_model, str):
        raise TypeError(
            f"noise_model must be a string, got {type(noise_model).__name__}."
        )
    if noise_model != "ols" and AR_NOISE_MODEL.match(noise_model) is None:
        raise ValueError(
            "noise_model must be 'ols' or 'arN' with N a positive integer "
            f"(e.g. 'ar1', 'ar2'); got {noise_model!r}."
        )


def _check_bins(bins: object) -> None:
    """Raise unless `bins` is a positive integer."""
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)):
        raise TypeError(f"bins must be an integer, got {type(bins).__name__}.")
    if bins < 1:
        raise ValueError(f"bins must be a positive integer, got {bins}.")


def _check_design_matrix(X: object, method: str) -> None:
    """Raise unless `X` is a `DesignMatrix`.

    Args:
        X (object): The candidate design.
        method (str): Name of the calling method, used in the message.

    Raises:
        TypeError: If `X` is not a `DesignMatrix`.
    """
    from nltools.data import DesignMatrix

    if not isinstance(X, DesignMatrix):
        raise TypeError(
            f"_Glm.{method} requires a DesignMatrix, got {type(X).__name__}. Raw "
            "arrays and other DataFrame types cannot preserve the fitted "
            "coefficient-to-regressor relationship."
        )


def _extract_fit_state(
    labels: np.ndarray,
    results: dict,
    feature_names: tuple[str, ...],
    n_targets: int,
) -> tuple[_GlmFitState, np.ndarray]:
    """Copy the compact fitted state and R-squared out of Nilearn's results.

    Args:
        labels (np.ndarray): Per-target model labels from `run_glm`.
        results (dict): Label to `RegressionResults` mapping from `run_glm`.
        feature_names (tuple[str, ...]): Fitted design column names.
        n_targets (int): Number of fitted targets.

    Returns:
        tuple[_GlmFitState, np.ndarray]: The retained state and the per-target
            `r_square` values copied from Nilearn.
    """
    first = next(iter(results.values()))
    coefficients = np.zeros((len(feature_names), n_targets), dtype=first.theta.dtype)
    dispersion = np.zeros(n_targets, dtype=np.asarray(first.dispersion).dtype)
    covariances = {}
    # Nilearn computes `r_square` as a bare division by the target's own
    # variance, so a constant target — an empty voxel inside a mask, which any
    # real brain mask contains — makes it 0/0 or x/0. The ratio is genuinely
    # undefined there and comes back non-finite; reading it must not raise
    # numpy's RuntimeWarning on the caller's behalf. The values are copied, not
    # recomputed, so nothing else about them changes.
    with np.errstate(divide="ignore", invalid="ignore"):
        r_square = np.zeros(n_targets, dtype=np.asarray(first.r_square).dtype)
        for label, result in results.items():
            target_mask = labels == label
            coefficients[:, target_mask] = result.theta
            dispersion[target_mask] = result.dispersion
            r_square[target_mask] = result.r_square
            covariances[label] = np.array(result.cov, copy=True)

    state = _GlmFitState(
        feature_names=feature_names,
        labels=np.array(labels, copy=True),
        coefficients=coefficients,
        covariances=covariances,
        dispersion=dispersion,
        residual_degrees_of_freedom=float(first.df_residuals),
    )
    return state, r_square


def _resolve_contrast(contrast: object, feature_names: tuple[str, ...]) -> np.ndarray:
    """Resolve one contrast definition to a validated float64 weight vector.

    Args:
        contrast (object): A string expression over the fitted column names or
            a real-valued flat vector with one weight per fitted column.
        feature_names (tuple[str, ...]): Fitted design column names.

    Returns:
        np.ndarray: The resolved weights, shape `(n_features,)`, dtype float64.

    Raises:
        TypeError: If `contrast` is boolean, complex, or otherwise nonnumeric.
        ValueError: If the expression is invalid or names an unknown column, or
            if the resolved vector is empty, not one-dimensional, the wrong
            length, non-finite, or all zero.
    """
    if isinstance(contrast, str):
        weights = expression_to_contrast_vector(contrast, list(feature_names))
    elif isinstance(contrast, (bool, np.bool_)):
        raise TypeError(
            "A contrast must be a string expression or a numeric vector; got a boolean."
        )
    else:
        weights = np.asarray(contrast)
        if weights.dtype == np.bool_:
            raise TypeError("A contrast must be real-valued; got boolean weights.")
        if not np.issubdtype(weights.dtype, np.number):
            raise TypeError(
                "A contrast must be a string expression or a numeric vector; got "
                f"an array of dtype {weights.dtype}."
            )
        if np.issubdtype(weights.dtype, np.complexfloating):
            raise TypeError("A contrast must be real-valued; got complex weights.")

    weights = np.asarray(weights, dtype=np.float64)
    if weights.size == 0:
        raise ValueError("A contrast must have at least one weight; got an empty one.")
    if weights.ndim != 1:
        raise ValueError(
            f"A contrast must be one-dimensional; got {weights.ndim} dimensions. "
            "_Glm does not compute F-contrasts."
        )
    if weights.shape[0] != len(feature_names):
        raise ValueError(
            "A contrast must have exactly one weight per fitted design column "
            f"({len(feature_names)}); got {weights.shape[0]}. Available columns "
            f"are: {list(feature_names)}."
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("A contrast must contain only finite weights.")
    if not np.any(weights):
        raise ValueError("A contrast must have at least one nonzero weight.")
    return weights


def _contrast_effect(state: _GlmFitState, weights: np.ndarray) -> np.ndarray:
    """Return the contrast effect `weights @ coefficients` as a new array."""
    return weights @ state.coefficients


def _contrast_statistics(state: _GlmFitState, weights: np.ndarray) -> dict[str, Any]:
    """Compute one t-contrast and its inferential statistics from `state`.

    Effect and variance use the same matrix operations as Nilearn's functional
    `compute_contrast`; the statistic, p-value, and z-score come from Nilearn's
    public `Contrast`. Every returned array owns its data.

    Args:
        state (_GlmFitState): The retained fitted state.
        weights (np.ndarray): A resolved float64 contrast vector.

    Returns:
        dict[str, Any]: `effect`, `variance`, `standard_error`, `statistic`,
            `z_score`, `p_value`, and `degrees_of_freedom`.
    """
    effect = _contrast_effect(state, weights)
    variance = np.zeros(state.labels.shape[0], dtype=np.float64)
    for label, covariance in state.covariances.items():
        target_mask = state.labels == label
        variance[target_mask] = (weights @ covariance @ weights) * state.dispersion[
            target_mask
        ]

    contrast = Contrast(
        effect=effect,
        variance=variance,
        dim=1,
        dof=state.residual_degrees_of_freedom,
        stat_type="t",
    )
    return {
        "effect": effect,
        "variance": variance,
        "standard_error": np.sqrt(variance),
        "statistic": np.array(contrast.stat(), copy=True),
        "z_score": np.array(contrast.z_score(), copy=True),
        "p_value": np.array(contrast.p_value(), copy=True),
        "degrees_of_freedom": float(contrast.dof),
    }


class _Glm:
    """General linear model over a precomputed design matrix and a response.

    Fits ordinary least squares or an autoregressive noise model with Nilearn's
    `run_glm`, then exposes coefficients, predictions, residuals, R-squared, and
    contrasts. The model is `y = X @ beta + error`: `_Glm` never adds or
    estimates an intercept, so include an intercept column in `X` when the model
    needs one. Predictions and residuals are always in observation space, for
    autoregressive fits as well as OLS.

    Args:
        noise_model (str): `'ols'` for ordinary least squares, or `'arN'` with
            `N` a positive integer for Nilearn's autoregressive model of that
            order (`'ar1'`, `'ar2'`, ...). Default `'ols'`.
        bins (int): Nilearn's discretization of the estimated AR coefficients —
            the maximum number of histogram bins for AR(1), and the maximum
            number of K-means clusters for higher orders. Must be positive.
            Default 100.
        n_jobs (int): Number of CPUs Nilearn uses to fit autoregressive groups
            in parallel, following joblib's convention (`-1` is all cores). The
            default OLS fit does not use this path. Default 1.
        random_state (int | None): Seeds the K-means step for autoregressive
            models of order two or greater. It does not affect OLS or AR(1)
            results. Default None.

    Attributes:
        coef_ (np.ndarray): Fitted coefficients, shape `(n_features,)` for a
            one-dimensional `y` and `(n_features, n_targets)` otherwise.
        predicted_ (np.ndarray): Training predictions `X @ coef_`, same shape as
            the fitted `y`.
        residuals_ (np.ndarray): Training residuals `y - predicted_`, same shape
            as the fitted `y`.
        r2_ (float | np.ndarray): Nilearn's `RegressionResults.r_square`, copied
            rather than recomputed. It is the variance ratio
            `variance(whitened_design @ coef_) / variance(whitened_y)`. For OLS
            the whitening is the identity, so with an intercept in the design
            this equals conventional R-squared; for autoregressive noise models
            it is a pseudo-R-squared in the whitened space. A constant target
            has zero variance, so its ratio is undefined and comes back
            non-finite rather than raising a warning. A float for a
            one-dimensional `y`, otherwise shape `(n_targets,)`.
        n_samples_ (int): Fitted sample count.
        n_features_in_ (int): Fitted feature count.
        feature_names_in_ (tuple[str, ...]): Fitted design column names in
            coefficient order.
        n_targets_ (int): Fitted target count; one for a one-dimensional `y`.
        is_fitted_ (bool): True after a successful fit.

    Examples:
        ```python
        import numpy as np
        from nltools.data import DesignMatrix
        from nltools.models import _Glm

        n_samples = 100
        rng = np.random.default_rng(0)
        design = DesignMatrix(
            {
                "condition_a": rng.normal(size=n_samples),
                "condition_b": rng.normal(size=n_samples),
                "intercept": np.ones(n_samples),
            },
            sampling_freq=0.5,
        )
        y = rng.normal(size=(n_samples, 50))

        model = _Glm(noise_model="ar1").fit(design, y)
        effects = model.compute_contrasts("condition_a - condition_b")
        result = model.compute_contrasts("condition_a - condition_b", inference=True)
        result.statistic  # → t-statistic per target
        ```
    """

    def __init__(
        self,
        *,
        noise_model: str = "ols",
        bins: int = 100,
        n_jobs: int = 1,
        random_state: int | None = None,
    ) -> None:
        _check_noise_model(noise_model)
        _check_bins(bins)
        self.noise_model = noise_model
        self.bins = bins
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.is_fitted_ = False

    def fit(self, X: DesignMatrix, y) -> _Glm:
        """Fit the model to one design matrix and response.

        A one-dimensional `y` is expanded to a single column for Nilearn and the
        target axis is squeezed back out of every fitted attribute and contrast
        result.

        Args:
            X (DesignMatrix): Design of shape `(n_samples, n_features)`,
                including any intercept column the model needs.
            y (array-like): Response of shape `(n_samples,)` or
                `(n_samples, n_targets)`.

        Returns:
            _Glm: The fitted model, for method chaining.

        Raises:
            TypeError: If `X` is not a `DesignMatrix`.
            ValueError: If `y` is not one- or two-dimensional, or its sample
                count does not match `X`.
        """
        _check_design_matrix(X, "fit")
        response = np.asarray(y)
        if response.ndim not in (1, 2):
            raise ValueError(f"y must be 1-D or 2-D, got {response.ndim} dimensions.")
        if response.shape[0] != X.shape[0]:
            raise ValueError(
                "X and y must have the same number of samples: X has "
                f"{X.shape[0]} and y has {response.shape[0]}."
            )

        squeeze_targets = response.ndim == 1
        if squeeze_targets:
            response = response[:, None]
        design = X.to_numpy()

        labels, results = run_glm(
            response,
            design,
            noise_model=self.noise_model,
            bins=self.bins,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        state, r_square = _extract_fit_state(
            labels, results, tuple(X.columns), response.shape[1]
        )
        del labels, results

        predicted = design @ state.coefficients
        residuals = response - predicted

        self._fit_state = state
        self._squeeze_targets = squeeze_targets
        # Copy so a caller mutating the public attribute cannot reach the
        # retained state that later contrasts are computed from.
        self.coef_ = (
            state.coefficients[:, 0] if squeeze_targets else state.coefficients
        ).copy()
        self.predicted_ = predicted[:, 0] if squeeze_targets else predicted
        self.residuals_ = residuals[:, 0] if squeeze_targets else residuals
        self.r2_ = float(r_square[0]) if squeeze_targets else r_square
        self.n_samples_ = response.shape[0]
        self.n_features_in_ = design.shape[1]
        self.feature_names_in_ = state.feature_names
        self.n_targets_ = response.shape[1]
        self.is_fitted_ = True
        return self

    def predict(self, X: DesignMatrix) -> np.ndarray:
        """Apply the fitted coefficients to a design matrix.

        `X` must carry exactly the fitted column names. They may appear in any
        order; the columns are reordered to `feature_names_in_` before
        multiplying, so the coefficient-to-regressor relationship survives.

        Args:
            X (DesignMatrix): Design with the fitted column names, in any order.

        Returns:
            np.ndarray: `X @ coef_`, shape `(n_samples,)` for a model fitted on
                a one-dimensional `y` and `(n_samples, n_targets)` otherwise.

        Raises:
            TypeError: If `X` is not a `DesignMatrix`.
            ValueError: If the model is not fitted, or `X` has missing or
                additional columns.
        """
        _check_is_fitted(self)
        _check_design_matrix(X, "predict")

        columns = list(X.columns)
        missing = sorted(set(self.feature_names_in_) - set(columns))
        additional = sorted(set(columns) - set(self.feature_names_in_))
        if missing or additional:
            raise ValueError(
                "X must have exactly the fitted design columns "
                f"{list(self.feature_names_in_)}. Missing: {missing}. "
                f"Additional: {additional}."
            )

        return X[list(self.feature_names_in_)].to_numpy() @ self.coef_

    def compute_contrasts(
        self,
        contrasts,
        *,
        inference: bool = False,
    ) -> float | np.ndarray | ContrastResult | dict:
        """Compute one contrast or a named mapping of contrasts on the fitted model.

        A contrast is a string expression over the fitted design column names —
        `"condition_a - condition_b"`, `"2 * condition_a - condition_b"` — or a
        real-valued vector with one weight per fitted column. Several contrasts
        must be supplied as a mapping of names to those definitions, which
        leaves every flat numeric sequence unambiguously available as one
        contrast vector.

        The default returns the effect `contrast @ coef_` only, the appropriate
        input to a second-level model. With `inference=True`, the contrast is
        tested against zero and every inferential output is returned together;
        the p-value is Nilearn's one-sided upper-tail value, so negating the
        contrast tests the opposite direction.

        Args:
            contrasts (str | array-like | Mapping): One contrast definition, or
                a mapping of string names to contrast definitions.
            inference (bool): If True, return `ContrastResult` records instead of
                bare effects. Default False.

        Returns:
            float | np.ndarray | ContrastResult | dict: One effect, or one
                `ContrastResult` when `inference=True`; a dictionary with the
                same keys for a mapping. Effects and inferential fields are
                floats for a model fitted on a one-dimensional `y`, shape `(1,)`
                for a `(n_samples, 1)` response, and shape `(n_targets,)`
                otherwise.

        Raises:
            RuntimeError: If the model has not been fitted.
            TypeError: If `inference` is not a bool, a mapping key is not a
                string, or a contrast is boolean, complex, or nonnumeric.
            ValueError: If the mapping is empty, an expression is invalid or
                names an unknown column, or a resolved contrast is empty,
                non-finite, all zero, wrongly sized, or not one-dimensional.

        Examples:
            ```python
            model.compute_contrasts("condition_a - condition_b")
            model.compute_contrasts([1, -1, 0])
            model.compute_contrasts({"a_vs_b": "condition_a - condition_b"})
            model.compute_contrasts("condition_a", inference=True).p_value
            ```
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "_Glm instance is not fitted yet. Call 'fit' with a DesignMatrix "
                "and a response before computing contrasts."
            )
        if not isinstance(inference, (bool, np.bool_)):
            raise TypeError(f"inference must be a bool, got {type(inference)}.")

        if isinstance(contrasts, Mapping):
            if not contrasts:
                raise ValueError(
                    "contrasts is an empty mapping; supply at least one named contrast."
                )
            for name in contrasts:
                if not isinstance(name, str):
                    raise TypeError(
                        f"Contrast names must be strings, got {type(name).__name__}."
                    )
            return {
                name: self._one_contrast(definition, inference)
                for name, definition in contrasts.items()
            }
        return self._one_contrast(contrasts, inference)

    def _one_contrast(self, contrast, inference: bool):
        """Resolve and compute one contrast definition."""
        weights = _resolve_contrast(contrast, self.feature_names_in_)
        if not inference:
            return self._payload(_contrast_effect(self._fit_state, weights))

        values = _contrast_statistics(self._fit_state, weights)
        return ContrastResult(
            effect=self._payload(values["effect"]),
            variance=self._payload(values["variance"]),
            standard_error=self._payload(values["standard_error"]),
            statistic=self._payload(values["statistic"]),
            z_score=self._payload(values["z_score"]),
            p_value=self._payload(values["p_value"]),
            degrees_of_freedom=values["degrees_of_freedom"],
        )

    def _payload(self, values: np.ndarray) -> float | np.ndarray:
        """Apply the fitted squeeze rule to one per-target array."""
        return float(values[0]) if self._squeeze_targets else values
