---
title: data.results
label: page-data-results
---

Structural result records returned by decoding and resampling operations.

**Classes:**

Name | Description
---- | -----------
[`BootstrapResult`](#data-results-bootstrapresult) | Frozen record of one bootstrap statistic's estimate and uncertainty.
[`Predict`](#data-results-predict) | Frozen structural record for `BrainData.predict` decoding results.



## Classes

(data-results-bootstrapresult)=
### `BootstrapResult`

```python
BootstrapResult(estimate: Payload, standard_error: Payload, ci_lower: Payload, ci_upper: Payload, samples: np.ndarray | None = None)
```

Frozen record of one bootstrap statistic's estimate and uncertainty.

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

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`estimate` | <code>Payload</code> | The statistic evaluated once on the original full sample — not the mean of the replicates.
`standard_error` | <code>Payload</code> | Elementwise standard deviation of the bootstrap replicates, with `ddof=1`.
`ci_lower` | <code>Payload</code> | Lower bound of the central percentile interval at the requested `confidence_level`.
`ci_upper` | <code>Payload</code> | Upper bound of that interval. The bounds are elementwise marginal: the nominal level applies separately to each voxel, feature, or test row, with no simultaneous-coverage claim.
`samples` | <code>ndarray \| None</code> | Every replicate, bootstrap axis first, when `return_samples=True`; `None` otherwise.

(data-results-predict)=
### `Predict`

```python
Predict(spatial_scale: str, scoring: Any = None, classes: np.ndarray | None = None, predictions: np.ndarray | None = None, cv_folds: np.ndarray | None = None, scores: np.ndarray | None = None, estimator: Any = None, weight_map: BrainData | None = None, roi_labels: np.ndarray | None = None, score_map: BrainData | None = None)
```

Frozen structural record for `BrainData.predict` decoding results.

``spatial_scale`` is the discriminator: it decides which fields carry a
value and which stay ``None``. Construction validates that combination and
the shapes it implies, so an empty or mixed-mode record cannot exist. Field
bindings cannot be rebound, but the payloads they hold remain usable, and
the record takes independent ownership of every array, brain map, and
estimator it stores.

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

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`spatial_scale` | <code>str</code> | ``'whole_brain'``, ``'roi'``, or ``'searchlight'``.
`scoring` | <code>str \| callable \| None</code> | The scoring specification the caller passed. ``None`` records that the estimator's own ``score`` method was used; it does not by itself name that method's metric.
[`classes`](#classes) | <code>ndarray \| None</code> | Classifier class labels, ``(n_classes,)``. ``None`` for regression.
`predictions` | <code>ndarray \| None</code> | Out-of-fold predictions, one per row, ``(n_samples,)`` (whole-brain only).
`cv_folds` | <code>ndarray \| None</code> | Fold index per row, ``(n_samples,)`` (whole-brain only).
`scores` | <code>ndarray \| None</code> | Per-fold score — ``(n_folds,)`` for whole-brain, ``(n_folds, n_rois)`` for ROI.
`estimator` | <code>Any</code> | The all-data fitted sklearn estimator (whole-brain only); use it to ``.predict()`` on new data.
`weight_map` | <code>[BrainData](#page-data-brain-data) \| None</code> | Coefficients of the estimator refit on all data, ``(n_voxels,)`` or ``(n_classes, n_voxels)`` for multiclass — one map for regression and binary classification, one map per class in ``classes`` order for multiclass. For ROI, each parcel's coefficients are written into its voxels (NaN outside parcels); magnitudes are not comparable across parcels.
`roi_labels` | <code>ndarray \| None</code> | Atlas integer ids, ``(n_rois,)``, in the order of the ``scores`` parcel axis (ROI only).
`score_map` | <code>[BrainData](#page-data-brain-data) \| None</code> | ``(n_voxels,)`` map of cross-validated scores — for ROI, every voxel of parcel *i* holds that parcel's mean fold score (NaN outside parcels); for searchlight, the sphere-centered mean fold score at each voxel.
`mean_score` | <code>float \| ndarray</code> | Mean of ``scores`` across folds, computed on demand — a float for whole-brain, ``(n_rois,)`` for ROI. Accessing it on a searchlight result raises `AttributeError`.
`std_score` | <code>float \| ndarray</code> | Standard deviation of ``scores`` across folds, in ``mean_score``'s form and with the same searchlight rule.

<details class="note" open markdown="1">
<summary>Note</summary>

Encoding-model timeseries prediction (``bd.predict(X=...)``) returns a
`BrainData` directly rather than a `Predict` — the natural container
for a voxel timeseries.

</details>

**Methods:**

Name | Description
---- | -----------
[`asdict`](#data-results-asdict) | Convert to dictionary.
[`available`](#data-results-available) | Return names of the fields this result carries (excludes private).

#### Methods

(data-results-asdict)=
##### `asdict`

```python
asdict(include_none: bool = False) -> dict
```

Convert to dictionary.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`include_none` | <code>bool</code> | If True, include every field that does not apply to this spatial scale, whose value is None. `spatial_scale` and `scoring` are always included. Private fields (starting with _) are always excluded. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Dictionary of field names to values.

(data-results-available)=
##### `available`

```python
available() -> list
```

Return names of the fields this result carries (excludes private).

`spatial_scale` and `scoring` always count: a `scoring` of `None` records
that the estimator's own `score` method was used, which is a value, not an
absent field.
