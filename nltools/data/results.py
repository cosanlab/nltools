"""Structural result records returned by fitting, decoding and resampling."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict as dataclass_asdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic

import numpy as np

from nltools.models.results import Payload

if TYPE_CHECKING:
    from .braindata import BrainData


#: Fields that may be populated for each spatial scale, and the subset that
#: every result of that scale must carry. Anything outside `permitted` must be
#: `None`, so an invalid field combination cannot be constructed.
_MODE_FIELDS = {
    "whole_brain": {
        "required": ("predictions", "cv_folds", "scores", "estimator", "weight_map"),
        "optional": ("classes",),
    },
    "roi": {
        "required": ("scores", "roi_labels", "score_map", "weight_map"),
        "optional": ("classes",),
    },
    "searchlight": {
        "required": ("score_map",),
        "optional": ("classes",),
    },
}

#: Fields that describe the call itself rather than one spatial scale's output.
#: They always carry a value, including a `scoring` of `None`, so they are exempt
#: from both the per-mode "None means not applicable" rule and the None filter in
#: `available` and `asdict`.
_MODE_INDEPENDENT_FIELDS = ("spatial_scale", "scoring")


def _fold_mean(scores, axis=None):
    """Reduce fold scores to their cross-fold mean, ignoring failed folds.

    `PredictResult.mean_score` and the ROI runner's painted `score_map` must report the
    same number for the same parcel, so both call this one reduction.
    """
    return np.nanmean(scores, axis=axis)


@dataclass(frozen=True)
class PredictResult:
    """Frozen structural record for `BrainData.predict` decoding results.

    ``spatial_scale`` is the discriminator: it decides which fields carry a
    value and which stay ``None``. Construction validates that combination, so
    an empty or mixed-mode record cannot exist; the shapes within it are the
    producer's responsibility. Field bindings cannot be rebound, but the
    payloads they hold remain usable, and the record takes independent
    ownership of every array, brain map, and estimator it stores.

    **Brain-space outputs are `BrainData` objects**, not raw arrays, so
    ``result.weight_map.plot()`` works directly (``.data`` gives the array).
    Non-spatial fields are numpy.

    **Populated by `spatial_scale`.** ``'whole_brain'``: ``predictions``,
    ``cv_folds``, ``scores``, ``estimator``, ``weight_map``. ``'roi'``:
    ``scores``, ``roi_labels``, ``score_map``, ``weight_map``.
    ``'searchlight'``: ``score_map``. ``classes`` accompanies any classifier;
    ``scoring`` records the caller's scoring specification in every mode.

    **Why the all-data fit is the canonical map.** The mean of per-fold
    ``coef_`` vectors corresponds to no actual fitted estimator (each fold saw
    a different subset), and fits on overlapping training folds are not
    independent uncertainty samples. The record therefore exposes one
    coefficient map, from the estimator refitted on all observations after
    cross-validation: cross-validation gives the honest *score*, the refit
    gives the publishable *map*.

    Attributes:
        spatial_scale (str): ``'whole_brain'``, ``'roi'``, or
            ``'searchlight'``.
        scoring (str | callable | None): The scoring specification the caller
            passed. ``None`` records that the estimator's own ``score`` method
            was used; it does not by itself name that method's metric.
        classes (ndarray | None): Classifier class labels, ``(n_classes,)``.
            ``None`` for regression.
        predictions (ndarray | None): Out-of-fold predictions, one per row,
            ``(n_samples,)`` (whole-brain only).
        cv_folds (ndarray | None): Fold index per row, ``(n_samples,)``
            (whole-brain only).
        scores (ndarray | None): Per-fold score — ``(n_folds,)`` for
            whole-brain, ``(n_folds, n_rois)`` for ROI.
        estimator (Any): The all-data fitted sklearn estimator (whole-brain
            only); use it to ``.predict()`` on new data.
        weight_map (BrainData | None): Coefficients of the estimator refit on
            all data, ``(n_voxels,)`` or ``(n_classes, n_voxels)`` for
            multiclass — one map for regression and binary classification, one
            map per class in ``classes`` order for multiclass. For ROI, each
            parcel's coefficients are written into its voxels (NaN outside
            parcels); magnitudes are not comparable across parcels.
        roi_labels (ndarray | None): Atlas integer ids, ``(n_rois,)``, in the
            order of the ``scores`` parcel axis (ROI only).
        score_map (BrainData | None): ``(n_voxels,)`` map of cross-validated
            scores — for ROI, every voxel of parcel *i* holds that parcel's
            mean fold score (NaN outside parcels); for searchlight, the
            sphere-centered mean fold score at each voxel.
        mean_score (float | ndarray): Mean of ``scores`` across folds, computed
            on demand — a float for whole-brain, ``(n_rois,)`` for ROI.
            Accessing it on a searchlight result raises `AttributeError`.
        std_score (float | ndarray): Standard deviation of ``scores`` across
            folds, in ``mean_score``'s form and with the same searchlight rule.

    Note:
        Encoding-model timeseries prediction (``bd.predict(X=...)``) returns a
        `BrainData` directly rather than a `PredictResult` — the natural container
        for a voxel timeseries.
    """

    spatial_scale: str
    scoring: Any = None
    classes: np.ndarray | None = None
    predictions: np.ndarray | None = None
    cv_folds: np.ndarray | None = None
    scores: np.ndarray | None = None
    estimator: Any = None
    weight_map: BrainData | None = None
    roi_labels: np.ndarray | None = None
    score_map: BrainData | None = None

    def __post_init__(self):
        """Validate the field combination, then take ownership."""
        self._validate_mode()
        for name in self.__dataclass_fields__:
            if name in _MODE_INDEPENDENT_FIELDS:
                # A scoring name or callable is the caller's specification, not
                # a payload the record owns; copying a callable scorer would
                # change what `scoring` reports.
                continue
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, deepcopy(value))

    def _validate_mode(self) -> None:
        """Require exactly the fields the spec's shape table lists for this scale."""
        if self.spatial_scale not in _MODE_FIELDS:
            raise ValueError(
                f"spatial_scale must be one of {sorted(_MODE_FIELDS)}; "
                f"got {self.spatial_scale!r}."
            )
        mode = _MODE_FIELDS[self.spatial_scale]
        permitted = set(mode["required"]) | set(mode["optional"])
        for name in mode["required"]:
            if getattr(self, name) is None:
                raise ValueError(
                    f"{name} is required for spatial_scale="
                    f"{self.spatial_scale!r} and cannot be None."
                )
        for name in self.__dataclass_fields__:
            if name in _MODE_INDEPENDENT_FIELDS or name in permitted:
                continue
            if getattr(self, name) is not None:
                raise ValueError(
                    f"{name} does not apply to spatial_scale="
                    f"{self.spatial_scale!r} and must be None. That scale "
                    f"populates {sorted(permitted)}."
                )

    @property
    def mean_score(self):
        """Mean score across folds — a float for whole-brain, per parcel for ROI."""
        return self._summarize(_fold_mean, "mean_score")

    @property
    def std_score(self):
        """Score standard deviation across folds, in `mean_score`'s form."""
        return self._summarize(np.nanstd, "std_score")

    def _summarize(self, reduction, name: str):
        """Derive one cross-fold summary from `scores`."""
        if self.spatial_scale == "searchlight":
            raise AttributeError(
                f"{name} does not exist for a searchlight result: searchlight "
                f"stores its cross-fold mean directly in score_map, with one "
                f"value per sphere center."
            )
        if self.spatial_scale == "roi":
            return reduction(self.scores, axis=0)
        return float(reduction(self.scores))

    def __setstate__(self, state):
        """Restore a pickled record, rejecting one written by an older nltools.

        Unpickling is the only path that bypasses `__post_init__`, and joblib
        caches (the tutorials memoize `predict` results) are full of pickles. An
        old record's fields are disjoint enough to detect, and without this check
        its `score_map` would silently read as the class default `None`.
        """
        if "spatial_scale" not in state or set(state) - set(self.__dataclass_fields__):
            raise ValueError(
                "This PredictResult was pickled by an older nltools and cannot be "
                "restored: its field set predates the spatial_scale "
                "discriminator. Clear the cache (for the tutorials, "
                "`uv run poe tutorials-clean-cache`) and rerun."
            )
        self.__dict__.update(state)

    def available(self) -> list:
        """Return names of the fields this result carries (excludes private).

        `spatial_scale` and `scoring` always count: a `scoring` of `None` records
        that the estimator's own `score` method was used, which is a value, not an
        absent field.
        """
        return [
            field_name
            for field_name in self.__dataclass_fields__
            if not field_name.startswith("_") and self._is_reported(field_name)
        ]

    def _is_reported(self, field_name: str) -> bool:
        """Whether a field appears in `available` and the default `asdict`."""
        return (
            field_name in _MODE_INDEPENDENT_FIELDS
            or getattr(self, field_name) is not None
        )

    def asdict(self, include_none: bool = False) -> dict:
        """Convert to dictionary.

        Args:
            include_none: If True, include every field that does not apply to
                this spatial scale, whose value is None. `spatial_scale` and
                `scoring` are always included. Private fields (starting with _)
                are always excluded.

        Returns:
            Dictionary of field names to values.
        """
        full_dict = dataclass_asdict(self)
        filtered = {k: v for k, v in full_dict.items() if not k.startswith("_")}
        if not include_none:
            filtered = {k: v for k, v in filtered.items() if self._is_reported(k)}
        return filtered


@dataclass(frozen=True)
class BootstrapResult(Generic[Payload]):
    """Frozen record of one bootstrap statistic's estimate and uncertainty.

    The single result structure every supported `bootstrap` statistic returns.
    Its payload is whatever the producer works in: `BrainData` for the
    `BrainData` facade, `Adjacency` for the `Adjacency` facade. The four
    summary payloads share one data shape.

    Field bindings cannot be rebound. The payloads stay usable, but the record
    takes independent ownership of each one, so mutating a returned payload
    never reaches the source object or a sibling payload.

    The record deliberately exposes no replicate mean and no `z`, `p`, or
    `tail` output: those need a separately defined bootstrap hypothesis test.
    For a normal-approximation stand-in, users compute it themselves from
    `estimate` and `standard_error`.

    Attributes:
        estimate (Payload): The statistic evaluated once on the original full
            sample — not the mean of the replicates.
        standard_error (Payload): Elementwise standard deviation of the
            bootstrap replicates, with `ddof=1`.
        ci_lower (Payload): Lower bound of the central percentile interval at
            the requested `confidence_level`.
        ci_upper (Payload): Upper bound of that interval. The bounds are
            elementwise marginal: the nominal level applies separately to each
            voxel, feature, or test row, with no simultaneous-coverage claim.
        samples (np.ndarray | None): Every replicate, bootstrap axis first,
            when `return_samples=True`; `None` otherwise.
    """

    estimate: Payload
    standard_error: Payload
    ci_lower: Payload
    ci_upper: Payload
    samples: np.ndarray | None = None

    def __post_init__(self):
        """Take independent ownership of every payload the record stores."""
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, deepcopy(value))


@dataclass(frozen=True)
class FitResult:
    """Frozen record of one `BrainData.fit`, held on `BrainData.model`.

    `fit` still returns the `BrainData` it fitted, so chaining survives; this
    record is the one declared attribute it sets. Every map is a `BrainData` on
    the fitted object's mask, so `result.betas.plot()` works directly and
    `.data` gives the array.

    Field bindings cannot be rebound and the record takes independent ownership
    of every map and design it stores. The fitted estimator is the exception:
    it is internal state the facade methods drive, not a payload to read, so it
    is neither copied nor shown in the repr, and it is `None` on a record
    restored from HDF5.

    Attributes:
        kind (str): Which estimator produced the fit, `'glm'` or `'ridge'`.
        betas (BrainData): One coefficient map per column of `design`, in
            column order. Ridge users know these as the model's weights.
        predicted (BrainData): The fitted response, one row per training
            observation, with the training row metadata retained.
        residual (BrainData): Response minus `predicted`, in the same shape.
        r2 (BrainData): One in-sample fit-quality map. A GLM's carries
            Nilearn's whitened variance-ratio semantics: conventional
            R-squared for an OLS fit with an intercept, a pseudo-R-squared in
            the whitened space for an autoregressive one.
        design (DesignMatrix | np.ndarray | dict[str, np.ndarray]): What the
            model was fit on — the `DesignMatrix` for a GLM, the feature matrix
            for ridge, or the named feature spaces for banded ridge.
        alpha (BrainData | None): Ridge only: the penalty the fit used, one
            value per voxel, broadcast when a single alpha was shared. `None`
            for a GLM.
        cv (BaseCrossValidator | None): Ridge only: the cross-validator alpha
            selection resolved to. `None` for a GLM and for a fixed-alpha fit.
    """

    kind: str
    betas: BrainData
    predicted: BrainData
    residual: BrainData
    r2: BrainData
    design: Any
    alpha: BrainData | None = None
    cv: Any = None
    #: The fitted `_Glm` or `_Ridge` that `compute_contrasts`, `predict` and
    #: `bootstrap` drive. Excluded from the ownership copy because it is not a
    #: payload the user reads, and copying a device-backed ridge fit would
    #: duplicate its solver state for nothing.
    _estimator: Any = field(default=None, repr=False)

    def __post_init__(self):
        """Take independent ownership of every payload the record stores."""
        for name in self.__dataclass_fields__:
            if name == "_estimator":
                continue
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, deepcopy(value))

    def write(self, directory, prefix=None) -> list:
        """Write the fit to `directory` as NIfTI maps, a design CSV and a sidecar.

        The whole "fit, then save" workflow in one call. Each map becomes
        `<prefix>_betas.nii.gz`, `_predicted`, `_residual`, `_r2` and — for
        ridge — `_alpha`; the design becomes `<prefix>_design.csv` (one
        `_design-<space>.csv` per feature space for a banded ridge); and
        `<prefix>_fit.json` records the kind of fit and the design's column
        names. Nothing here is BIDS.

        Args:
            directory (str | Path): Where to write. Created if it does not exist.
            prefix (str | None): Prepended to every filename as `<prefix>_`.
                Default None writes the bare names.

        Returns:
            list[Path]: Every file written.

        Examples:
            ```python
            data.fit(model="glm", X=design)
            data.model.write("derivatives/sub-01", prefix="sub-01_task-rest")
            ```
        """
        from .results_io import _write_fit

        return _write_fit(self, directory, prefix)
