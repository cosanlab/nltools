---
title: "Prediction: encoding & decoding"
---

[`BrainData.predict`](../api/data/brain_data.md#data-brain-data-predict) is decoding: predict a
per-image label or value `y` from voxel patterns, cross-validated. One call returns a frozen
[`Predict`](../api/data/results.md#data-results-predict) result whose `spatial_scale` field
says which of its fields carry values. Encoding runs the other way, predicting voxel timeseries
from stimulus features, and is a ridge problem. Use
[`BrainData.fit`](../api/data/brain_data.md#data-brain-data-fit)`(model='ridge')` or the
[`Ridge`](../api/models.md#models-ridge) estimator directly.

`spatial_scale=` sets what a "pattern" means. `'whole_brain'` fits one model on every in-mask
voxel. `'roi'` needs `roi_mask=` (a labeled parcellation) and fits one model per parcel, returning
per-parcel scores plus a `score_map` with every voxel filled by its parcel's score.
`'searchlight'` fits one model per sphere of radius `radius` (in millimeters) and returns a
per-voxel map. It is the slow one, so cache the result.

## What comes back

Every field exists on every `Predict`; `None` means the field does not apply to the scale you
asked for. Constructing a mixed combination is impossible — the record validates itself.

Field | `'whole_brain'` | `'roi'` | `'searchlight'`
--- | --- | --- | ---
`spatial_scale` | `'whole_brain'` | `'roi'` | `'searchlight'`
`scoring` | what you passed | what you passed | what you passed
`classes` | class labels, or `None` for regression | same | same
`predictions` | `(n_samples,)` out-of-fold | `None` | `None`
`cv_folds` | `(n_samples,)` fold index | `None` | `None`
`scores` | `(n_folds,)` | `(n_folds, n_rois)` | `None`
`estimator` | the all-data fit | `None` | `None`
`weight_map` | `BrainData` (see below) | `BrainData` (see below) | `None`
`roi_labels` | `None` | `(n_rois,)` | `None`
`score_map` | `None` | `BrainData` | `BrainData`

`mean_score` and `std_score` are computed from `scores` on demand — a float for whole-brain, one
value per parcel for ROI. A searchlight result has no cross-fold summary to compute: its
`score_map` already holds the mean score at every sphere center, so asking for either raises
`AttributeError`.

`scoring=None` (the default) records that the estimator's own `score` method was used; it does not
name that method's metric. `weight_map` is the estimator refit on all observations after
cross-validation — the map you publish. It is one signed map for regression and binary
classification. It is `None` today when the estimator exposes no `coef_` (a non-linear model, or a
pipeline whose preprocessing cannot be reversed) and for a multiclass classifier, whose one map per
class is not built yet; averaging coefficients across classes describes no fitted decision
boundary, so nothing is returned in its place. Fold-specific coefficient maps are deliberately
absent: fits on overlapping training folds are not independent uncertainty samples.

Goal | Use | Notes
--- | --- | ---
Decode a label or value | `predict(y=, estimator=, cv=)` | `y` is an array, or a string naming a column of `.Y`
Pick an estimator | `estimator='linear_svc'`, `'logistic_regression'`, `'linear_discriminant_analysis'`, `'ridge_classifier'`, `'ridge'`, `'lasso'`, `'linear_svr'`, or any sklearn estimator | Only linear models expose a `weight_map`
Cross-validation | `cv=None` (a deterministic five folds), `cv=5`, or an sklearn splitter such as `LeaveOneGroupOut()` + `groups=` | Test folds must partition the rows, so shuffle-split and repeated splitters raise
Stratify a continuous target | [`KFoldStratified`](../api/tasks/prediction.md#tasks-prediction-kfoldstratified) | Deals `y`-ordered samples round-robin into folds
Region-by-region | `spatial_scale='roi', roi_mask=atlas` | Answers "is this region informative on its own?"
Voxel-by-voxel | `spatial_scale='searchlight', radius=8.0` | Thousands of models; `n_jobs` defaults to `1` here on purpose
Classifier performance | [`Roc`](../api/tasks/prediction.md#tasks-prediction-roc) | `calculate()` then `summary()` or `plot()`
Encoding (features → voxels) | [`Ridge`](../api/models.md#models-ridge), or `fit(model='ridge', ridge_*=...)` | `per_target_alpha=True` (the default) picks a per-voxel alpha; a named mapping of feature spaces makes it banded

## Decoding

```python
high_pain = (pain.X["PainLevel"].to_numpy() > 2).astype(int)

result = pain.predict(y=high_pain, estimator="linear_svc", cv=5)
result.mean_score      # cross-validated accuracy
result.weight_map      # BrainData: the classifier refit on all the data
```

Leave-one-subject-out is `cv=LeaveOneGroupOut()` plus a `groups=` column. Naming a string for `y`
or `groups` looks it up in `.Y`, so attach that table first (`pain.Y = pain.X`).

```python
from sklearn.model_selection import LeaveOneGroupOut

grouped = pain.predict(
    y=high_pain, estimator="linear_svc", cv=LeaveOneGroupOut(), groups="SubjectID"
)
```

For an ROI or searchlight map, change `spatial_scale` and nothing else:

```python
roi_result = pain.predict(
    y=high_pain, estimator="linear_svc", spatial_scale="roi", roi_mask=atlas
)
roi_result.mean_score      # one score per parcel
roi_result.score_map       # those scores painted back into voxel space
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

The same fit through the facade carries a `ridge_` prefix on every estimator option, and attaches
`ridge_weights`, `ridge_fitted_values`, and `ridge_r2`:

```python
brain.fit(model="ridge", X=X, ridge_alpha=np.logspace(0, 6, 20), ridge_cv=5)
brain.ridge_r2                     # full-data R² per voxel
brain.predict()                    # an owned copy of ridge_fitted_values
brain.model_.alpha_                # the selection lives on the estimator
```

Fitting keeps no copy of `X`, so a coefficient or prediction bootstrap takes the training features
explicitly and holds the selected hyperparameters fixed across replicates:

```python
boot = brain.bootstrap(stat="weights", X=X, n_samples=1000)
boot["mean"], boot["ci_lower"], boot["ci_upper"]
```

## Gotchas

- `predict` never mutates the object and attaches nothing to it. `fit` does mutate by default.
- The built-in shortcuts z-score voxels *inside* each training fold, not before the split, so
  there is no leakage. A caller-supplied estimator or `Pipeline` is used exactly as given.
- `cv=None` and an integer `cv` do not shuffle, so the folds are reproducible across calls. Rows
  ordered by condition make contiguous folds degenerate — pass
  `cv=KFold(n_splits=5, shuffle=True, random_state=0)` when that is a risk.
- `n_jobs` defaults to `1` on `BrainData.predict` because searchlight copies the brain into every
  worker.
- A ROC on a regression model's `predictions` needs a binary `binary_outcome`. Pass the labels,
  not the continuous target.

Next: [Similarity & RSA](similarity-and-rsa.md), or the
[MVPA tutorial](../tutorials/workflows/03_mvpa.md) and
[encoding tutorial](../tutorials/workflows/02_encoding.md).
