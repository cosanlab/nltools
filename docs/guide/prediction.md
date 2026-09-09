---
title: "Prediction: encoding & decoding"
---

[`BrainData.predict`](../api/data/brain_data.md#data-brain-data-predict) is decoding: predict a
per-image label or value `y` from voxel patterns, cross-validated. One call returns a frozen
[`Predict`](../api/data/fitresults.md#data-fitresults-predict) result holding `predictions`,
`scores`, `mean_score`, `cv_folds`, the `weight_map` (the model refit on all the data, the map you
publish), and `fold_weight_maps`. Encoding runs the other way, predicting voxel timeseries from
stimulus features, and is a ridge problem. Use
[`BrainData.fit`](../api/data/brain_data.md#data-brain-data-fit)`(model='ridge')` or the
[`Ridge`](../api/models.md#models-ridge) estimator directly.

`spatial_scale=` sets what a "pattern" means. `'whole_brain'` fits one model on every in-mask
voxel. `'roi'` needs `roi_mask=` (a labeled parcellation) and fits one model per parcel, returning
per-parcel scores plus an `accuracy_map` with every voxel filled by its parcel's score.
`'searchlight'` fits one model per sphere of radius `radius_mm` and returns a per-voxel map. It is
the slow one, so cache the result.

Goal | Use | Notes
--- | --- | ---
Decode a label or value | `predict(y=, model=, cv=)` | `y` is an array, or a string naming a column of `.Y`
Pick an estimator | `model='svm'`, `'logistic'`, `'lda'`, `'ridge_classifier'`, `'ridge'`, `'lasso'`, `'svr'`, or any sklearn estimator | Only linear models expose a `weight_map`
Cross-validation | `cv=5`, `cv='loo'`, `cv='logo'` + `groups=`, or an sklearn splitter | [`resolve_cv`](../api/tasks/prediction.md#tasks-prediction-resolve-cv) turns any of these into a splitter
Stratify a continuous target | [`KFoldStratified`](../api/tasks/prediction.md#tasks-prediction-kfoldstratified) | Deals `y`-ordered samples round-robin into folds
Region-by-region | `spatial_scale='roi', roi_mask=atlas` | Answers "is this region informative on its own?"
Voxel-by-voxel | `spatial_scale='searchlight', radius_mm=8.0` | Thousands of models; `n_jobs` defaults to `1` here on purpose
Classifier performance | [`Roc`](../api/tasks/prediction.md#tasks-prediction-roc) | `calculate()` then `summary()` or `plot()`
Encoding (features → voxels) | [`Ridge`](../api/models.md#models-ridge) | `per_target_alpha=True` (the default) picks a per-voxel alpha; a named mapping of feature spaces makes it banded

## Decoding

```python
high_pain = (pain.X["PainLevel"].to_numpy() > 2).astype(int)

result = pain.predict(y=high_pain, model="svm", cv=5, random_state=0)
result.mean_score      # cross-validated accuracy
result.weight_map      # BrainData: the classifier refit on all the data
```

Leave-one-subject-out is `cv='logo'` plus a `groups=` column. Naming a string for `y` or `groups`
looks it up in `.Y`, so attach that table first (`pain.Y = pain.X`).

```python
grouped = pain.predict(
    y=high_pain, model="svm", cv="logo", groups="SubjectID", random_state=0
)
```

For an ROI or searchlight map, change `spatial_scale` and nothing else:

```python
roi_result = pain.predict(
    y=high_pain, model="svm", spatial_scale="roi", roi_mask=atlas, random_state=0
)
roi_result.mean_score      # one score per parcel
roi_result.accuracy_map    # those scores painted back into voxel space
```

## Encoding

```python
from nltools.models import Ridge

model = Ridge(alpha=np.logspace(0, 6, 20), cv=5).fit(X, brain.data)
model.coef_, model.alpha_, model.cv_scores_
```

A scalar `alpha` with `cv=None` fits it as given; a sequence of alphas with a `cv` selects one,
per voxel by default (`per_target_alpha=False` shares a single alpha). Pass `X` as a *mapping* of
named feature spaces and `Ridge` becomes banded ridge, sampling each space's weight from a
Dirichlet controlled by `dirichlet_concentration=` and exposing the result as
`feature_space_weights_`. Both forms accept `device='gpu'`; see
[Performance & GPU](../performance.md).

## Gotchas

- `predict` never mutates the object; `inplace=False` is the default. `fit` does mutate by default.
- `standardize=True` (the default) z-scores voxels *inside* each training fold, not before the
  split, so there is no leakage.
- `n_jobs` defaults to `1` on `BrainData.predict` because searchlight copies the brain into every
  worker.
- A ROC on a regression model's `predictions` needs a binary `binary_outcome`. Pass the labels,
  not the continuous target.

Next: [Similarity & RSA](similarity-and-rsa.md), or the
[MVPA tutorial](../tutorials/workflows/03_mvpa.md) and
[encoding tutorial](../tutorials/workflows/02_encoding.md).
