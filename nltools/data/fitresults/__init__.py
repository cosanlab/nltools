"""Immutable container for model fitting results.

This module provides the Fit dataclass, which stores results from model fitting
operations in nltools. It uses pure numpy arrays and has no dependencies on
BrainData or other nltools data structures, making it suitable for standalone
use with inference algorithms.

Examples:
    Using with BrainData workflow:

    >>> from nltools.data import BrainData
    >>> brain = BrainData(data="brain_data.nii.gz")
    >>> fit = brain.fit(model="ridge", X=design_matrix, cv=5)
    >>> print(fit.available())
    ['fitted_values', 'weights', 'scores', 'cv_scores', 'cv_mean_score', 'cv_predictions', 'cv_folds']

    Using with inference algorithms directly:

    >>> from nltools.algorithms import ridge_cv
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randn(100, 1000)
    >>> result = ridge_cv(X, y, cv=5)
    >>> result["cv_scores"].shape
    (5, 20, 1000)

    Serialization/deserialization:

    >>> # Save all non-None results
    >>> np.savez("fit_results.npz", **fit.asdict())
    >>>
    >>> # Load and reconstruct
    >>> loaded = np.load("fit_results.npz")
    >>> fit_reconstructed = Fit(**{k: loaded[k] for k in loaded.files})

    Export to .npz:

    >>> # Export only specific fields
    >>> import numpy as np
    >>> np.savez("weights_and_scores.npz",
    ...          weights=fit.weights,
    ...          scores=fit.scores)

    Introspection:

    >>> # Check what's available
    >>> if 'cv_scores' in fit.available():
    ...     print(f"CV R² range: [{fit.cv_mean_score.min():.3f}, {fit.cv_mean_score.max():.3f}]")
    >>>
    >>> # Get as dict and convert to a polars DataFrame (for scalar and 1D arrays)
    >>> import polars as pl
    >>> results_dict = fit.asdict()
    >>> df = pl.DataFrame({k: v for k, v in results_dict.items() if v.ndim <= 1})
"""

from collections.abc import Iterator
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

    Attributes:
        fitted_values (ndarray): Fitted values or predictions, always present.
        weights (ndarray | None): Model coefficients (Ridge).
        scores (ndarray | None): R² scores (Ridge).
        betas (ndarray | None): Beta coefficients (GLM).
        t_stats (ndarray | None): T-statistics (GLM).
        p_values (ndarray | None): P-values (GLM).
        se (ndarray | None): Standard errors (GLM).
        residuals (ndarray | None): Residuals (GLM).
        r2 (ndarray | None): R² values (GLM).
        cv_scores (ndarray | None): Per-fold cross-validation scores.
        cv_mean_score (ndarray | None): Mean cross-validation score across folds.
        cv_predictions (ndarray | None): Out-of-fold predictions.
        cv_folds (ndarray | None): Fold indices for each sample.
        cv_best_alpha (float | None): Best alpha selected via cross-validation.
        cv_alpha_scores (ndarray | None): Cross-validation scores for each alpha tested.

    Note:
        Methods: `available` returns the list of non-None attribute names
        (excludes private fields); `asdict` converts to a dictionary,
        optionally excluding None values.

    Examples:
        Creating a Fit object (Ridge without CV):

        >>> import numpy as np
        >>> from nltools.data.fitresults import Fit
        >>> fit = Fit(
        ...     fitted_values=np.random.randn(100, 1000),
        ...     weights=np.random.randn(5, 1000),
        ...     scores=np.random.randn(1000)
        ... )
        >>> fit.available()
        ['fitted_values', 'weights', 'scores']

        Creating a Fit object (Ridge with CV):

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

        Immutability:

        >>> try:
        ...     fit.scores = np.zeros(1000)  # Will raise FrozenInstanceError
        ... except AttributeError:
        ...     print("Cannot modify frozen dataclass")
        Cannot modify frozen dataclass

        Export/serialization:

        >>> # Save to .npz
        >>> np.savez("results.npz", **fit.asdict())
        >>>
        >>> # Load and reconstruct
        >>> loaded = np.load("results.npz")
        >>> fit_reloaded = Fit(**{k: loaded[k] for k in loaded.files})

    Note:
        - Frozen dataclass ensures results cannot be accidentally modified.
        - All attributes are numpy arrays (except cv_best_alpha which is float).
        - None values indicate that field was not computed for this model/method.
        - Private fields (starting with _) are excluded from available() and asdict().
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

        Examples:
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

        Examples:
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

    Mirrors `Fit`: frozen, all fields default to `None`, populated
    based on the dispatch path (`spatial_scale`, `y` vs `X`, `refit`) used
    by `BrainData.predict`. Fields not applicable to the call remain
    `None` and are filtered from `available` and `asdict`.

    **Brain-space outputs are `BrainData` objects**, not raw arrays —
    so `result.weight_map.plot()` works directly. Drop down to numpy via
    `result.weight_map.data` if needed. Non-spatial fields (`predictions`,
    `cv_folds`, scalar scores) stay as numpy.

    Field shapes by dispatch:

    **spatial_scale='whole_brain'** (with `y`):
        - `predictions`: `(n_samples,)` ndarray, OOF predictions from CV
        - `scores`: `(n_folds,)` ndarray, per-fold score
        - `mean_score`: float, mean across folds
        - `std_score`: float, std across folds
        - `cv_folds`: `(n_samples,)` ndarray, fold index per sample
        - `weight_map`: BrainData `(1, n_voxels)`, `coef_` from one
          model fit on the **full** `(X, y)`. The publishable map.
        - `fold_weight_maps`: BrainData `(n_folds, n_voxels)`, per-fold
          `coef_` stack — for stability analysis (e.g., across-fold std).
        - `estimator`: the fitted all-data sklearn estimator (use for
          `.predict()` on new data).

    **spatial_scale='roi'** (with `y`):
        - `scores`: `(n_folds, n_rois)` ndarray
        - `mean_score`: `(n_rois,)` ndarray, mean across folds per parcel
        - `std_score`: `(n_rois,)` ndarray
        - `roi_labels`: `(n_rois,)` ndarray of atlas integer IDs in the
          same order as `mean_score` / `std_score` / `scores` axis 1
        - `accuracy_map`: BrainData `(1, n_voxels)`, every voxel inside
          parcel *i* set to that parcel's mean accuracy (others NaN)
        - `weight_map`: BrainData `(1, n_voxels)`, per-parcel `coef_`
          from each parcel's all-data fit, written back into voxel space
          (atlas is a label image so reassembly is disjoint). Voxels outside
          any parcel are NaN. Magnitudes across parcels are not directly
          comparable — different parcels live on different X distributions.
        - `fold_weight_maps`: BrainData `(n_folds, n_voxels)`
        - `estimator`: `dict[int, sklearn]` keyed by atlas label

        If any parcel can't expose `.coef_` (non-linear model, `SelectKBest`
        in pipeline), `weight_map` / `fold_weight_maps` / `estimator`
        all collapse to `None` for the whole call.

    **spatial_scale='searchlight'** (with `y`):
        - `accuracy_map`: BrainData `(1, n_voxels)`, sphere-centered
          accuracy at each voxel

    Note:
        Encoding-model timeseries prediction (`bd.predict(X=...)`) returns
        a `BrainData` directly, not a `Predict` — the natural container for a
        voxel timeseries.

        Why the all-data fit is canonical: the CV mean of per-fold `coef_`
        vectors doesn't correspond to any actual fitted estimator (each fold
        saw a different subset). The all-data refit is a single, real model
        with all the information used. CV gives the honest *score*; the refit
        gives the publishable *map*. `fold_weight_maps` is still exposed for
        stability analysis, and the CV-mean is one line away if you want it
        (`fold_weight_maps.data.mean(axis=0)`).

        Methods: `available` returns the names of non-None fields (excludes
        private); `asdict` converts to a dict for serialization (private fields
        always excluded).
    """

    # MVPA / classification — populated when y given
    predictions: np.ndarray | None = None
    scores: np.ndarray | None = None
    mean_score: Any = None  # float (whole_brain) or ndarray (n_rois,) (roi)
    std_score: Any = None  # float (whole_brain) or ndarray (n_rois,) (roi)
    cv_folds: np.ndarray | None = None

    # ROI — atlas labels in the order of mean_score / std_score / scores axis 1
    roi_labels: np.ndarray | None = None

    # Brain-space maps — BrainData when populated by the runners
    accuracy_map: Any = None  # BrainData
    weight_map: Any = None  # BrainData (all-data fit, the publishable map)
    fold_weight_maps: Any = None  # BrainData (per-fold stack, stability analysis)

    # All-data fitted estimator (whole_brain only)
    estimator: Any = None

    # Label-permutation null (BrainCollection.predict_group(n_permute=))
    permutation_scores: np.ndarray | None = None
    permutation_pvalue: float | None = None

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


@dataclass(frozen=True)
class PredictCollection:
    """Immutable container for per-subject decoding results.

    Returned by ``BrainCollection.predict(y=...)``: one `Predict` per subject
    (each an independent within-subject model), plus the collection's
    per-subject metadata. Sequence-like — ``len``, iteration, and integer
    indexing all address the underlying `Predict` objects.

    The stacking properties are the bridge to second-level inference: the
    per-subject maps become one ``BrainData (n_subjects, n_voxels)``, ready
    for a group test.

    Attributes:
        results: One `Predict` per subject, in collection order.
        metadata: Optional per-subject metadata (polars DataFrame, one row
            per subject), carried over from the source collection.
        paths: Optional on-disk predict-bundle paths, populated when the
            producing call cached its results (``cache=True``/``'auto'``).

    Note:
        Properties: ``mean_scores`` / ``std_scores`` stack each subject's
        score summary — ``(n_subjects,)`` for whole-brain decoding,
        ``(n_subjects, n_rois)`` for ROI. ``scores`` renders the whole-brain
        case as a polars DataFrame alongside the metadata. ``weight_maps`` /
        ``accuracy_maps`` stack the per-subject brain maps into one
        ``BrainData``. ``available`` returns the field names populated on
        *every* subject's result.

    Examples:
        ```python
        pc = collection.predict(y="condition", cv=5)
        pc.scores                    # per-subject accuracy table
        pc[0].weight_map.plot()      # one subject's decoder map

        # Second-level inference on the decoder maps:
        from nltools.algorithms import one_sample_permutation_test
        group = one_sample_permutation_test(pc.weight_maps.data)
        ```
    """

    results: tuple
    metadata: Any = None  # pl.DataFrame | None
    paths: tuple | None = None

    def __post_init__(self):
        results = tuple(self.results)
        if not results:
            raise ValueError("PredictCollection cannot be empty.")
        for i, r in enumerate(results):
            if not isinstance(r, Predict):
                raise TypeError(
                    f"results[{i}] is {type(r).__name__}, expected Predict."
                )
        object.__setattr__(self, "results", results)
        if self.metadata is not None and self.metadata.shape[0] != len(results):
            raise ValueError(
                f"metadata has {self.metadata.shape[0]} rows for "
                f"{len(results)} results."
            )
        if self.paths is not None:
            paths = tuple(self.paths)
            if len(paths) != len(results):
                raise ValueError(
                    f"paths has {len(paths)} entries for {len(results)} results."
                )
            object.__setattr__(self, "paths", paths)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[Predict]:
        return iter(self.results)

    def __getitem__(self, idx: int) -> Predict:
        return self.results[idx]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_subjects={len(self.results)}, "
            f"available={self.available()})"
        )

    def _stack_field(self, field: str) -> np.ndarray:
        values = [getattr(r, field) for r in self.results]
        missing = [i for i, v in enumerate(values) if v is None]
        if missing:
            raise ValueError(
                f"{field} is not populated for subjects {missing} — it is "
                f"unavailable for their dispatch mode (see `Predict`)."
            )
        return np.stack([np.asarray(v) for v in values], axis=0)

    @property
    def mean_scores(self) -> np.ndarray:
        """Per-subject mean CV score — ``(n_subjects,)`` or ``(n_subjects, n_rois)``."""
        return self._stack_field("mean_score")

    @property
    def std_scores(self) -> np.ndarray:
        """Per-subject score std across folds, matching ``mean_scores``' shape."""
        return self._stack_field("std_score")

    @property
    def scores(self):
        """Per-subject score table (polars): metadata + mean/std score columns.

        Only defined when each subject's score summary is a scalar
        (whole-brain decoding); for ROI decoding use ``mean_scores`` /
        ``std_scores`` — the array forms.
        """
        import polars as pl

        mean = self.mean_scores
        if mean.ndim != 1:
            raise ValueError(
                "scores is a per-subject scalar table; these results carry "
                "array-valued score summaries (ROI decoding) — use "
                "mean_scores / std_scores instead."
            )
        base = (
            self.metadata
            if self.metadata is not None
            else pl.DataFrame({"subject": np.arange(len(self))})
        )
        return base.with_columns(
            pl.Series("mean_score", mean),
            pl.Series("std_score", self._stack_field("std_score")),
        )

    def _stack_maps(self, field: str):
        maps = [getattr(r, field) for r in self.results]
        missing = [i for i, m in enumerate(maps) if m is None]
        if missing:
            raise ValueError(
                f"{field} is not populated for subjects {missing} — it is "
                f"unavailable for their dispatch mode (see `Predict`)."
            )
        from nltools.data import BrainData

        stacked = np.vstack([np.asarray(m.data).reshape(1, -1) for m in maps])
        return BrainData(stacked, mask=maps[0].mask)

    @property
    def weight_maps(self):
        """Per-subject decoder maps as one ``BrainData (n_subjects, n_voxels)``."""
        return self._stack_maps("weight_map")

    @property
    def accuracy_maps(self):
        """Per-subject accuracy maps as one ``BrainData (n_subjects, n_voxels)``."""
        return self._stack_maps("accuracy_map")

    def available(self) -> list:
        """Return field names populated on every subject's `Predict`."""
        common = set(self.results[0].available())
        for r in self.results[1:]:
            common &= set(r.available())
        return sorted(common)
