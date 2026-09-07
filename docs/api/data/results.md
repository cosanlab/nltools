---
title: data.results
label: page-data-results
---

Structural result records returned by decoding operations.

**Classes:**

Name | Description
---- | -----------
[`Predict`](#data-results-predict) | Frozen structural record for `BrainData.predict` decoding results.
[`PredictCollection`](#data-results-predictcollection) | Frozen structural record for per-subject decoding results.



## Classes

(data-results-predict)=
### `Predict`

```python
Predict(predictions: np.ndarray | None = None, scores: np.ndarray | None = None, mean_score: float | np.ndarray | None = None, std_score: float | np.ndarray | None = None, cv_folds: np.ndarray | None = None, roi_labels: np.ndarray | None = None, accuracy_map: BrainData | None = None, weight_map: BrainData | None = None, fold_weight_maps: BrainData | None = None, estimator: Any = None, permutation_scores: np.ndarray | None = None, permutation_pvalue: float | np.ndarray | BrainData | None = None)
```

Frozen structural record for `BrainData.predict` decoding results.

Fields cannot be rebound, but their mutable payloads remain usable. The
record takes independent ownership of arrays, brain maps, and estimators
when constructed. Which fields are populated depends on ``spatial_scale``;
fields not applicable to the call stay ``None`` and are dropped by
`available` and `asdict`.

**Brain-space outputs are `BrainData` objects**, not raw arrays, so
``result.weight_map.plot()`` works directly (``.data`` gives the array).
Non-spatial fields (``predictions``, ``cv_folds``, scalar scores) are
numpy.

**Populated by `spatial_scale`.** ``'whole_brain'``: ``predictions``,
``scores``, ``mean_score``, ``std_score``, ``cv_folds``, ``weight_map``,
``fold_weight_maps``, ``estimator``. ``'roi'``: ``scores``,
``mean_score``, ``std_score``, ``roi_labels``, ``accuracy_map``,
``weight_map``, ``fold_weight_maps``, ``estimator`` — and if any parcel's
model cannot expose ``coef_`` (a non-linear model, or feature selection in
the pipeline), ``weight_map`` / ``fold_weight_maps`` / ``estimator`` are
all ``None`` for the whole call. ``'searchlight'``: ``accuracy_map`` only.

**Why the all-data fit is the canonical map.** The mean of per-fold
``coef_`` vectors corresponds to no actual fitted estimator (each fold saw
a different subset). The all-data refit is one real model using all the
information; CV gives the honest *score*, the refit gives the publishable
*map*. ``fold_weight_maps`` is still exposed for stability analysis, and
the CV mean is ``fold_weight_maps.data.mean(axis=0)``.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`predictions` | <code>ndarray \| None</code> | Out-of-fold CV predictions, ``(n_samples,)`` (whole-brain only).
`scores` | <code>ndarray \| None</code> | Per-fold score — ``(n_folds,)`` for whole-brain, ``(n_folds, n_rois)`` for ROI.
`mean_score` | <code>float \| ndarray \| None</code> | Mean score across folds — a float for whole-brain, ``(n_rois,)`` for ROI.
`std_score` | <code>float \| ndarray \| None</code> | Score standard deviation across folds, same form as ``mean_score``.
`cv_folds` | <code>ndarray \| None</code> | Fold index per sample, ``(n_samples,)`` (whole-brain only).
`roi_labels` | <code>ndarray \| None</code> | Atlas integer ids, ``(n_rois,)``, in the order of ``mean_score`` / ``std_score`` / ``scores`` axis 1 (ROI only).
`accuracy_map` | <code>[BrainData](#page-data-brain-data) \| None</code> | ``(1, n_voxels)`` map — for ROI, every voxel in parcel *i* holds that parcel's mean score (NaN outside parcels); for searchlight, the sphere-centered score at each voxel.
`weight_map` | <code>[BrainData](#page-data-brain-data) \| None</code> | ``(1, n_voxels)`` ``coef_`` of the model refit on all data — the publishable map. For ROI, each parcel's coefficients are written back into voxel space (NaN outside parcels); magnitudes are not comparable across parcels.
`fold_weight_maps` | <code>[BrainData](#page-data-brain-data) \| None</code> | ``(n_folds, n_voxels)`` stack of per-fold ``coef_`` for stability analysis.
`estimator` | <code>Any</code> | The fitted all-data sklearn estimator (whole-brain; use it to ``.predict()`` on new data), or a ``dict[int, estimator]`` keyed by atlas label (ROI). ``None`` when read back from a cached bundle.
`permutation_scores` | <code>ndarray \| None</code> | Label-permutation null from `BrainCollection.predict_group` — ``(n_permute,)`` for whole-brain, ``(n_permute, n_rois)`` for ROI, ``(n_permute, n_voxels)`` for searchlight.
`permutation_pvalue` | <code>float \| ndarray \| [BrainData](#page-data-brain-data) \| None</code> | Upper-tail permutation p-value — a float, ``(n_rois,)``, or a ``(1, n_voxels)`` `BrainData` map, matching ``permutation_scores``.

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
[`available`](#data-results-available) | Return names of non-None fields (excludes private).



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
`include_none` | <code>bool</code> | If True, include fields with None values. Private fields (starting with _) are always excluded. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Dictionary of field names to values.

(data-results-available)=
##### `available`

```python
available() -> list
```

Return names of non-None fields (excludes private).

(data-results-predictcollection)=
### `PredictCollection`

```python
PredictCollection(results: tuple[Predict, ...], metadata: pl.DataFrame | None = None, paths: tuple[Path | None, ...] | None = None)
```

Frozen structural record for per-subject decoding results.

Returned by ``BrainCollection.predict(y=...)``: one `Predict` per subject
(each an independent within-subject model), plus the collection's
per-subject metadata. Sequence-like — ``len``, iteration, and integer
indexing all address the underlying `Predict` objects.

The stacking properties are the bridge to second-level inference: the
per-subject maps become one ``BrainData (n_subjects, n_voxels)``, ready
for a group test.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`results` | <code>tuple[[Predict](#data-results-predict), ...]</code> | One `Predict` per subject, in collection order.
`metadata` | <code>DataFrame \| None</code> | Per-subject metadata (one row per subject), carried over from the source collection.
`paths` | <code>tuple[Path \| None, ...] \| None</code> | On-disk predict-bundle paths, populated when the producing call cached its results (``cache=True`` / ``'auto'``).
`mean_scores` | <code>ndarray</code> | Stacked per-subject mean CV score — ``(n_subjects,)`` for whole-brain decoding, ``(n_subjects, n_rois)`` for ROI.
`std_scores` | <code>ndarray</code> | Stacked per-subject score standard deviation across folds, same shape as ``mean_scores``.
`scores` | <code>DataFrame</code> | Per-subject score table — the metadata plus ``mean_score`` / ``std_score`` columns. Whole-brain decoding only; ROI results raise (use ``mean_scores`` / ``std_scores``).
`weight_maps` | <code>[BrainData](#page-data-brain-data)</code> | Per-subject decoder maps stacked into one ``(n_subjects, n_voxels)`` `BrainData`.
`accuracy_maps` | <code>[BrainData](#page-data-brain-data)</code> | Per-subject accuracy maps stacked into one ``(n_subjects, n_voxels)`` `BrainData`.

**Methods:**

Name | Description
---- | -----------
[`available`](#data-results-available) | Return field names populated on every subject's `Predict`.



**Examples:**

```python
pc = collection.predict(y="condition", cv=5)
pc.scores                    # per-subject accuracy table
pc[0].weight_map.plot()      # one subject's decoder map

# Second-level inference on the decoder maps:
from nltools.algorithms import one_sample_permutation_test
group = one_sample_permutation_test(pc.weight_maps.data)
```

#### Methods

##### `available`

```python
available() -> list
```

Return field names populated on every subject's `Predict`.
