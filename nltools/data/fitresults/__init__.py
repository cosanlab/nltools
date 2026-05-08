"""Immutable container for model fitting results.

This module provides the Fit dataclass, which stores results from model fitting
operations in nltools. It uses pure numpy arrays and has no dependencies on
BrainData or other nltools data structures, making it suitable for standalone
use with inference algorithms.

Examples
--------
**Using with BrainData workflow:**

>>> from nltools.data import BrainData
>>> brain = BrainData(data="brain_data.nii.gz")
>>> fit = brain.fit(X=design_matrix, mode="ridge", cv_dict={"type": "kfold", "n_splits": 5})
>>> print(fit.available())
['fitted_values', 'weights', 'scores', 'cv_scores', 'cv_mean_score', 'cv_predictions', 'cv_folds']

**Using with inference algorithms directly:**

>>> from nltools.algorithms import ridge_cv
>>> import numpy as np
>>> X = np.random.randn(100, 5)
>>> y = np.random.randn(100, 1000)
>>> fit = ridge_cv(X, y, alpha=1.0, cv_dict={"type": "kfold", "n_splits": 5})
>>> fit.cv_mean_score.shape
(1000,)

**Serialization/deserialization:**

>>> # Save all non-None results
>>> np.savez("fit_results.npz", **fit.asdict())
>>>
>>> # Load and reconstruct
>>> loaded = np.load("fit_results.npz")
>>> fit_reconstructed = Fit(**{k: loaded[k] for k in loaded.files})

**Export to .npz:**

>>> # Export only specific fields
>>> import numpy as np
>>> np.savez("weights_and_scores.npz",
...          weights=fit.weights,
...          scores=fit.scores)

**Introspection:**

>>> # Check what's available
>>> if 'cv_scores' in fit.available():
...     print(f"CV R² range: [{fit.cv_mean_score.min():.3f}, {fit.cv_mean_score.max():.3f}]")
>>>
>>> # Get as dict and convert to a polars DataFrame (for scalar and 1D arrays)
>>> import polars as pl
>>> results_dict = fit.asdict()
>>> df = pl.DataFrame({k: v for k, v in results_dict.items() if v.ndim <= 1})
"""

from dataclasses import asdict as dataclass_asdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Fit:
    """Immutable container for model fitting results.

    Pure numpy arrays with minimal introspection methods. This allows
    users to work directly with nltools inference algorithms without
    requiring BrainData objects.

    Attributes depend on model type and CV usage:

    **Ridge (no CV):**
        weights (ndarray): Coefficients, shape (n_features, n_voxels)
        scores (ndarray): R² scores, shape (n_voxels,)
        fitted_values (ndarray): Training predictions, shape (n_samples, n_voxels)

    **Ridge (with CV):**
        All above plus:
        cv_scores (ndarray): Per-fold R², shape (n_folds, n_voxels)
        cv_mean_score (ndarray): Mean R² across folds, shape (n_voxels,)
        cv_predictions (ndarray): Out-of-fold predictions, shape (n_samples, n_voxels)
        cv_folds (ndarray): Fold indices, shape (n_samples,)
        cv_best_alpha (float): Selected alpha (if alpha='auto')
        cv_alpha_scores (ndarray): Alpha selection scores (if alpha='auto')

    **GLM:**
        betas (ndarray): Beta coefficients, shape (n_regressors, n_voxels)
        t_stats (ndarray): T-statistics, shape (n_regressors, n_voxels)
        p_values (ndarray): P-values, shape (n_regressors, n_voxels)
        se (ndarray): Standard errors, shape (n_regressors, n_voxels)
        residuals (ndarray): Residuals, shape (n_samples, n_voxels)
        fitted_values (ndarray): Fitted values, shape (n_samples, n_voxels)
        r2 (ndarray): R² values, shape (n_voxels,)

    Attributes
    ----------
    fitted_values : ndarray
        Fitted values or predictions, always present
    weights : ndarray, optional
        Model coefficients (Ridge)
    scores : ndarray, optional
        R² scores (Ridge)
    betas : ndarray, optional
        Beta coefficients (GLM)
    t_stats : ndarray, optional
        T-statistics (GLM)
    p_values : ndarray, optional
        P-values (GLM)
    se : ndarray, optional
        Standard errors (GLM)
    residuals : ndarray, optional
        Residuals (GLM)
    r2 : ndarray, optional
        R² values (GLM)
    cv_scores : ndarray, optional
        Per-fold cross-validation scores
    cv_mean_score : ndarray, optional
        Mean cross-validation score across folds
    cv_predictions : ndarray, optional
        Out-of-fold predictions
    cv_folds : ndarray, optional
        Fold indices for each sample
    cv_best_alpha : float, optional
        Best alpha selected via cross-validation
    cv_alpha_scores : ndarray, optional
        Cross-validation scores for each alpha tested

    Methods
    -------
    available() : list
        Returns list of non-None attribute names (excludes private fields)
    asdict(include_none=False) : dict
        Converts to dictionary, optionally excluding None values

    Examples
    --------
    **Creating a Fit object (Ridge without CV):**

    >>> import numpy as np
    >>> from nltools.data.fitresults import Fit
    >>> fit = Fit(
    ...     fitted_values=np.random.randn(100, 1000),
    ...     weights=np.random.randn(5, 1000),
    ...     scores=np.random.randn(1000)
    ... )
    >>> fit.available()
    ['fitted_values', 'weights', 'scores']

    **Creating a Fit object (Ridge with CV):**

    >>> fit_cv = Fit(
    ...     fitted_values=np.random.randn(100, 1000),
    ...     weights=np.random.randn(5, 1000),
    ...     scores=np.random.randn(1000),
    ...     cv_scores=np.random.randn(5, 1000),
    ...     cv_mean_score=np.random.randn(1000),
    ...     cv_predictions=np.random.randn(100, 1000),
    ...     cv_folds=np.arange(100) % 5
    ... )
    >>> 'cv_scores' in fit_cv.available()
    True

    **Immutability:**

    >>> try:
    ...     fit.scores = np.zeros(1000)  # Will raise FrozenInstanceError
    ... except AttributeError:
    ...     print("Cannot modify frozen dataclass")
    Cannot modify frozen dataclass

    **Export/serialization:**

    >>> # Save to .npz
    >>> np.savez("results.npz", **fit.asdict())
    >>>
    >>> # Load and reconstruct
    >>> loaded = np.load("results.npz")
    >>> fit_reloaded = Fit(**{k: loaded[k] for k in loaded.files})

    Notes
    -----
    - Frozen dataclass ensures results cannot be accidentally modified
    - All attributes are numpy arrays (except cv_best_alpha which is float)
    - None values indicate that field was not computed for this model/method
    - Private fields (starting with _) are excluded from available() and asdict()
    """

    # Always available
    fitted_values: np.ndarray

    # Ridge-specific
    weights: np.ndarray | None = None
    scores: np.ndarray | None = None

    # GLM-specific
    betas: np.ndarray | None = None
    t_stats: np.ndarray | None = None
    p_values: np.ndarray | None = None
    se: np.ndarray | None = None
    residuals: np.ndarray | None = None
    r2: np.ndarray | None = None

    # CV-specific
    cv_scores: np.ndarray | None = None
    cv_mean_score: np.ndarray | None = None
    cv_predictions: np.ndarray | None = None
    cv_folds: np.ndarray | None = None
    cv_best_alpha: float | None = None
    cv_alpha_scores: np.ndarray | None = None

    def available(self) -> list:
        """Return list of non-None attribute names.

        Excludes private fields (starting with _).

        Returns:
            Names of attributes that are not None.

        Examples
        --------
        >>> import numpy as np
        >>> from nltools.data.fitresults import Fit
        >>> fit = Fit(
        ...     fitted_values=np.random.randn(100, 1000),
        ...     weights=np.random.randn(5, 1000)
        ... )
        >>> fit.available()
        ['fitted_values', 'weights']
        >>> 'scores' in fit.available()
        False
        """
        return [
            field_name
            for field_name in self.__dataclass_fields__
            if not field_name.startswith("_") and getattr(self, field_name) is not None
        ]

    def asdict(self, include_none: bool = False) -> dict:
        """Convert to dictionary.

        Args:
            include_none: If True, include attributes with None values.
                Private fields (starting with _) are always excluded.

        Returns:
            Dictionary of attribute names to values.

        Examples
        --------
        >>> import numpy as np
        >>> from nltools.data.fitresults import Fit
        >>> fit = Fit(
        ...     fitted_values=np.random.randn(100, 1000),
        ...     weights=np.random.randn(5, 1000),
        ...     scores=None
        ... )
        >>> d = fit.asdict(include_none=False)
        >>> 'scores' in d
        False
        >>> d = fit.asdict(include_none=True)
        >>> 'scores' in d
        True
        >>> d['scores'] is None
        True
        """
        # Get full dict from dataclass
        full_dict = dataclass_asdict(self)

        # Filter out private fields (always)
        filtered = {k: v for k, v in full_dict.items() if not k.startswith("_")}

        # Optionally filter None values
        if not include_none:
            filtered = {k: v for k, v in filtered.items() if v is not None}

        return filtered


@dataclass(frozen=True)
class Predict:
    """Immutable container for prediction / MVPA decoding results.

    Mirrors :class:`Fit`: frozen, all fields default to ``None``, populated
    based on the dispatch path (``method``, ``y`` vs ``X``, ``refit``) used
    by :meth:`BrainData.predict`. Fields not applicable to the call remain
    ``None`` and are filtered from :meth:`available` and :meth:`asdict`.

    Field groups by call shape:

    **MVPA cross-validation** (``y`` provided):
        predictions, scores, mean_score, std_score, cv_folds.

    **Whole-brain decoding**:
        weight_map, fold_weight_maps (also populated for ROI when each ROI
        emits a coefficient vector).

    **Searchlight / ROI**:
        accuracy_map (voxel-shaped accuracy values).

    **Refit on full data** (``refit=True``):
        final_estimator, final_weight_map.

    Note: encoding-model timeseries prediction (``bd.predict(X=...)``) returns
    a ``BrainData`` directly, not a ``Predict`` — the natural container for a
    voxel timeseries.

    Attributes
    ----------
    predictions : ndarray, optional
        Out-of-fold predictions in original sample order, shape (n_samples,).
    scores : ndarray, optional
        Per-fold score, shape (n_folds,).
    mean_score : float, optional
        Mean score across folds.
    std_score : float, optional
        Standard deviation of scores across folds.
    cv_folds : ndarray, optional
        Fold index per sample, shape (n_samples,).
    accuracy_map : ndarray, optional
        Voxel-shaped accuracy map for searchlight / ROI methods.
    weight_map : ndarray, optional
        Mean linear-classifier coefficients across folds, projected to voxel
        space, shape (n_voxels,).
    fold_weight_maps : ndarray, optional
        Per-fold linear-classifier coefficients, shape (n_folds, n_voxels).
    final_estimator : sklearn estimator, optional
        Estimator refit on full data when ``refit=True``.
    final_weight_map : ndarray, optional
        Coefficients from the full-data refit, shape (n_voxels,).

    Methods
    -------
    available() : list
        Names of non-None fields (excludes private).
    asdict(include_none=False) : dict
        Convert to dict for serialization (private fields always excluded).
    """

    # MVPA / classification — populated when y given
    predictions: np.ndarray | None = None
    scores: np.ndarray | None = None
    mean_score: float | None = None
    std_score: float | None = None
    cv_folds: np.ndarray | None = None

    # Searchlight / ROI
    accuracy_map: np.ndarray | None = None

    # Linear-model weight maps
    weight_map: np.ndarray | None = None
    fold_weight_maps: np.ndarray | None = None

    # refit=True
    final_estimator: Any | None = None
    final_weight_map: np.ndarray | None = None

    def available(self) -> list:
        """Return names of non-None fields (excludes private)."""
        return [
            field_name
            for field_name in self.__dataclass_fields__
            if not field_name.startswith("_") and getattr(self, field_name) is not None
        ]

    def asdict(self, include_none: bool = False) -> dict:
        """Convert to dictionary.

        Args:
            include_none: If True, include fields with None values.
                Private fields (starting with _) are always excluded.

        Returns:
            Dictionary of field names to values.
        """
        full_dict = dataclass_asdict(self)
        filtered = {k: v for k, v in full_dict.items() if not k.startswith("_")}
        if not include_none:
            filtered = {k: v for k, v in filtered.items() if v is not None}
        return filtered
