# BrainData specification

This working specification defines the approved `BrainData` contracts and
records unresolved decisions explicitly. Code, tests, and docstrings must
implement the settled sections. Compatibility notes and migration
history belong elsewhere.

The estimator-specific details referenced here are authoritative in
[`glm.md`](glm.md) and [`ridge.md`](ridge.md).

## Purpose and ownership

`BrainData` is the stateful facade for masked brain arrays, their row-aligned
metadata, and fitted analysis state. Its methods coordinate validation, copying,
metadata, and result wrapping. Numerical and domain logic belongs in internal
functions or the `Glm` and `Ridge` estimators.

`BrainData` remains the return type of `fit`. There is no separate fitted-data
wrapper. Model prediction and MVPA decoding continue to share the existing
`predict` method; there is no separate public `decode` method.

`copy()` returns a complete, independently owned snapshot. The copy includes
data, mask and masker state, row metadata, a fitted estimator, and attached fit
results when present. Mutating the copy must not affect the source.

## Row metadata

`.X` and `.Y` describe rows of `data`. They are valid only while their rows
refer to the same observations in the same order.

- An operation that preserves observations and their order retains `.X` and
  `.Y`. This includes spatial smoothing and masking, value transformations,
  arithmetic, stored training predictions, and training residuals.
- An operation that subsets or reorders observations applies the identical
  selection or ordering to `.X` and `.Y`.
- An output whose leading axis no longer represents the source observations
  clears `.X` and `.Y`. This includes coefficients, contrasts, inferential and
  fit-quality maps, reductions, group statistics, and bootstrap results.
- No-argument fitted-model prediction returns stored training predictions and
  retains their aligned row metadata.
- Prediction for an explicit new design clears the source row metadata. The
  prediction design is not installed as the result's `.X`.

One shared result-construction policy must enforce these rules. Individual
methods must not mechanically copy row metadata and then repair mismatches.

## Model fitting

The public signature is:

```python
BrainData.fit(
    model="glm",
    *,
    X,
    ridge_alpha=1.0,
    ridge_cv=None,
    ridge_search_iterations=100,
    ridge_dirichlet_concentration=(0.1, 1.0),
    ridge_device="cpu",
    ridge_memory_budget_gb=None,
    ridge_per_target_alpha=True,
    ridge_prefer_conservative_alpha=False,
    ridge_progress_bar=False,
    glm_noise_model="ols",
    glm_bins=100,
    glm_n_jobs=1,
    inplace=True,
    random_state=None,
) -> BrainData
```

`X` is required and keyword-only. A GLM requires a precomputed `DesignMatrix`.
Ordinary Ridge requires one numerical matrix. Banded Ridge requires a non-empty
mapping from feature-space names to matrices.

Every model-specific argument uses a `glm_` or `ridge_` prefix. `random_state`
retains its unprefixed name because both estimators use it.
`ridge_progress_bar` maps to `Ridge.progress_bar`; there is no GLM progress
argument. Supplying a non-default option for the unselected estimator raises
`ValueError`; an irrelevant option must never be silently accepted.

`fit` does not accept preprocessing, intercept, HRF, drift, filtering,
smoothing, event, report, caching, or raw third-party arguments. It has no
`**kwargs`. Users prepare the response and design explicitly with the existing
`BrainData` and `DesignMatrix` operations.

With `inplace=True`, fitting replaces all previous fitted state on `self` and
returns `self`. With `inplace=False`, fitting begins from an independent,
fit-state-free copy, leaves every part of the source untouched, and returns the
fitted copy.

## Fitted state

A successful GLM fit attaches only:

- `model_`: the fitted `Glm`;
- `glm_betas`: one map per design column;
- `glm_residual`: one row per training observation;
- `glm_predicted`: one row per training observation; and
- `glm_r2`: one fit-quality map.

A successful Ridge fit attaches only:

- `model_`: the fitted `Ridge`;
- `ridge_weights`: one map per feature;
- `ridge_fitted_values`: one row per training observation; and
- `ridge_r2`: one R-squared map.

Estimator selection state remains on `model_`. `BrainData` does not duplicate
selected alphas, cross-validation scores, banded feature weights, or feature
metadata.

The fitted object does not retain `X_`, `design_matrix`, `cv_results_`,
`ridge_scores`, eager `glm_t`, `glm_p`, or `glm_se`, or a Ridge intercept.

`_FIT_STATE_ATTRIBUTES` is an exhaustive enumeration of every fitted estimator
and attached model result. Clearing fitted state uses only that enumeration; it
does not combine a partial list with predicates or special-case assignments.

Any in-place operation that changes data values, the row axis, or the voxel axis
clears all fitted state, including stored predictions. Derived analytical
results start without fitted state. `copy()` preserves fitted state
independently. A refit clears all old state before attaching new state, so GLM
and Ridge result families cannot coexist accidentally.

## Prediction and decoding

The public signature is:

```python
BrainData.predict(
    *,
    X: DesignMatrix | ArrayLike | Mapping[str, ArrayLike] | None = None,
    y: ArrayLike | str | None = None,
    estimator: str | BaseEstimator = "linear_svc",
    cv: int | BaseCrossValidator | None = None,
    groups: ArrayLike | str | None = None,
    scoring: str | Callable | None = None,
    spatial_scale: Literal["whole_brain", "roi", "searchlight"] = "whole_brain",
    roi_mask: NiimgLike | None = None,
    radius_mm: float = 10.0,
    n_jobs: int = 1,
    progress_bar: bool = False,
) -> BrainData | Predict
```

Static overloads expose the mode-specific return types: fitted-model prediction
returns `BrainData`, while MVPA returns `Predict`. They do not change the single
runtime signature.

`predict` resolves exactly one mode before doing any work:

- An explicit `y=` requests MVPA decoding.
- An explicit `X=` requests prediction from a fitted `Glm` or `Ridge`.
- With neither argument and a fitted model, it returns an independent copy of
  the stored training predictions.
- With neither argument, no fitted model, and exactly one `.Y` column, it runs
  MVPA using that column.
- Supplying both `X` and `y`, or any argument and fitted-state combination not
  listed above, raises before prediction begins.

For MVPA, `y` must be one-dimensional with one value per `BrainData` row. A
string `y` names one column of `.Y`. Multioutput and multilabel targets are not
accepted. An array-like `groups` must also contain one value per row; a string
`groups` names one `.Y` column. Missing columns, a multi-column `.Y` without an
explicit `y`, or a row-count mismatch raises before model fitting.

Fitted-model prediction wins over attached `.Y` on a no-argument call.

`predict` never mutates `self` and has no `inplace` argument. MVPA returns a
`Predict` result. Fitted-model prediction returns a new, independently owned
`BrainData`. `predict` does not attach dynamic `predict_*` attributes to the
source `BrainData`.

MVPA has no separate `standardize`, `reduce`, or `n_components` arguments.
The public argument is named `estimator`, not `model` or `decoder`. The built-in
estimator shortcuts are `"linear_svc"`, `"logistic_regression"`,
`"linear_discriminant_analysis"`, `"ridge_classifier"`, `"ridge"`, `"lasso"`,
and `"linear_svr"`. All built-in pipelines use linear estimators. The default
is `"linear_svc"`; ambiguous abbreviations such as `"svm"`, `"logistic"`,
`"lda"`, and `"svr"` are not accepted.

Built-in classification shortcuts use one-vs-rest for multiclass targets.
MVPA does not wrap a caller-supplied classifier or override its multiclass
strategy. Callers who want one-vs-rest behavior must supply a
`OneVsRestClassifier`.

Each shortcut selects a predefined pipeline that performs preprocessing and
estimation within each cross-validation fold. For a caller-supplied
scikit-learn estimator or `Pipeline`, MVPA does not add, remove, or reconfigure
preprocessing steps. Callers include any custom scaling, dimensionality
reduction, or feature selection in that estimator or pipeline.

Every MVPA pipeline must end in an estimator that exposes `coef_`. For
whole-brain and ROI decoding, every preprocessing step must also allow those
coefficients to be projected back to the original whole-brain or parcel voxel
axis. An incompatible estimator or pipeline raises `ValueError`. Every
successful whole-brain or ROI result includes `Predict.weight_map`. Searchlight
uses the same transformer whitelist but does not combine coefficients from
overlapping local models into one map.

Supported preprocessing steps are `StandardScaler`, `PCA`, `VarianceThreshold`,
`GenericUnivariateSelect`, `SelectPercentile`, `SelectKBest`, `SelectFpr`,
`SelectFdr`, `SelectFwe`, `SelectFromModel`, `RFE`, `RFECV`,
`SequentialFeatureSelector`, and `None` or `"passthrough"`. Pipelines may
compose these steps in any order whose fitted feature widths align. Other
transformers are invalid, even if they implement `inverse_transform`, because
an inverse data transformation is not generally a coefficient back-projection.

Back-projection starts with coefficients shaped `(n_maps, n_final_features)`
and walks fitted preprocessing steps in reverse order:

- A feature selector expands the feature axis to its fitted input width and
  inserts exact zeros at unselected positions.
- Unwhitened `PCA` applies `weights @ components_`.
- Whitened `PCA` computes `scale = sqrt(explained_variance_)`, replaces values
  of `scale` below `finfo(scale.dtype).eps` with that epsilon, divides each
  component weight by `scale`, and then applies `weights @ components_`.
- `StandardScaler(with_std=True)` divides weights by `scale_`.
  `StandardScaler(with_std=False)` leaves them unchanged.

Each step validates its fitted input and output widths, and the final projected
width must match the original whole-brain or parcel voxel axis. Centering adds
an offset to the raw-space decision function but does not change its slope map.
Therefore, `raw_data @ weight_map` need not reproduce the full decision
function. Callers use the fitted whole-brain `Predict.estimator` for prediction.

`OneVsRestClassifier` is handled explicitly because it does not expose one
combined `coef_`. For binary classification, its sole fitted child must expose
one coefficient row, representing `classes_[1]` versus `classes_[0]`. For
multiclass classification, child `i` must expose one coefficient row for
`classes_[i]`; those rows are stacked in class order. Supported shared
preprocessing must appear before `OneVsRestClassifier`, which must be the final
pipeline step.

For regression, `Predict.weight_map` contains one coefficient map. For binary
classification, it contains one signed map for `classes_[1]` versus
`classes_[0]`. For multiclass classification, it contains one map per class in
`classes_` order. `Predict.classes` stores the classifier's class labels.
It is `None` for regression. Coefficient maps are never averaged across classes.

`Predict` does not expose fold-specific coefficient maps. Fits on overlapping
training folds are neither independent uncertainty samples nor a substitute
for a defined inferential procedure. The canonical coefficient map comes from
the estimator refitted on all observations after cross-validation.

`scoring` defaults to `None` and follows scikit-learn's single-metric scoring
contract. `None` uses the estimator's `score` method; a scoring name or callable
overrides it. The nltools-specific `"auto"` value is removed. Multimetric
mappings are not accepted because `Predict.scores` contains one value per
cross-validation fold.

`cv` and `groups` apply only to MVPA. Prediction from a fitted `Glm` or `Ridge`
never constructs or evaluates cross-validation folds; it delegates directly to
the fitted estimator. In MVPA, `groups` is optional and is passed to the
selected scikit-learn cross-validation splitter, including splitters such as
`LeaveOneGroupOut`.

MVPA follows scikit-learn's cross-validation grammar. `cv=None` selects its
deterministic five-fold `KFold` or `StratifiedKFold`; an integer selects that
many folds; and a scikit-learn cross-validation splitter is used as supplied.
The nltools-specific `"loo"` and `"logo"` aliases are removed. `predict` has no
`random_state` argument: callers configure randomness on the estimator or
splitter that owns it.

`n_jobs` controls the outer independent work for each MVPA spatial mode:
cross-validation folds for whole-brain decoding, parcels for ROI decoding, and
spheres for searchlight decoding. `BrainCollection` manages member-level
parallelism and runs each member's inner prediction with `n_jobs=1`.

MVPA requires the cross-validation test folds to partition the observations:
each observation appears in exactly one test fold. This keeps
`Predict.predictions` aligned one-to-one with the original rows and gives every
observation one `Predict.cv_folds` value. Repeated, overlapping, or incomplete
test folds raise before model fitting.

MVPA uses one frozen `Predict` result class. Its required `spatial_scale`
discriminator is `"whole_brain"`, `"roi"`, or `"searchlight"`. Every field
exists on every instance; `None` marks fields that do not apply to the selected
mode. Construction validates the permitted non-`None` fields and their shapes,
so an empty result or invalid field combination cannot be constructed. Separate
public result classes are not introduced for the three spatial modes.

The exact stored fields are:

| Field | Whole brain | ROI | Searchlight |
| --- | --- | --- | --- |
| `spatial_scale` | `"whole_brain"` | `"roi"` | `"searchlight"` |
| `scoring` | caller's specification | caller's specification | caller's specification |
| `classes` | `(n_classes,)` or `None` | `(n_classes,)` or `None` | `(n_classes,)` or `None` |
| `predictions` | `(n_samples,)` | `None` | `None` |
| `cv_folds` | `(n_samples,)` | `None` | `None` |
| `scores` | `(n_folds,)` | `(n_folds, n_rois)` | `None` |
| `estimator` | fitted all-data estimator | `None` | `None` |
| `weight_map` | one map or `(n_classes,)` maps | one map or `(n_classes,)` maps | `None` |
| `roi_labels` | `None` | `(n_rois,)` | `None` |
| `score_map` | `None` | one map | one map |

`scoring` stores the scoring specification supplied by the caller. For
`scoring=None`, it records that the fitted estimator's `score` method was used;
it does not by itself identify that method's metric. Internal caching preserves
a callable scorer losslessly or raises before the prediction begins.
Construction rejects every field combination not listed in the table.

Whole-brain results contain row-aligned out-of-fold `predictions`, `cv_folds`,
one `scores` value per fold, the all-data fitted estimator, and its canonical
`weight_map`. Retaining the estimator lets callers apply the complete decoding
pipeline to new brain data. Internal collection caching must preserve it.

ROI results contain `scores` with shape `(n_folds, n_rois)`, ordered
`roi_labels`, a `score_map`, and a `weight_map` assembled from the full-data
estimator fitted within each parcel. `score_map` paints each parcel's mean fold
score into its voxels. Because parcels do not overlap, each parcel's
coefficients have one unambiguous destination on the voxel axis. Binary
classification and regression produce one assembled coefficient map;
multiclass classification produces one assembled map per class. ROI does not
expose its internal estimator mapping or parcel-wise out-of-fold prediction
matrix.

Searchlight results contain one `score_map`. They do not expose predictions,
fold assignments, estimators, or coefficient maps for overlapping local
neighborhoods.

`Predict.mean_score` and `Predict.std_score` derive from `scores` for
whole-brain and ROI results; they are not stored independently. Accessing either
property on a searchlight result raises `AttributeError`. Searchlight stores its
cross-fold mean directly in `score_map`. The misleading `accuracy_map` name is
removed because the selected scorer may not measure accuracy.

For a fitted GLM, an explicit `X` must be a `DesignMatrix` with exactly the
fitted column-name set. Reordered columns are aligned to fitted order; missing,
additional, or duplicate columns raise. For Ridge, `X` is one ordinary matrix
or a named mapping matching the fitted banded feature spaces and widths.

## GLM contrasts

The public method is:

```python
compute_contrasts(
    contrasts: str | NumericVector | Mapping[str, str | NumericVector],
    *,
    inference: bool = False,
)
```

A string expression or flat numeric vector describes one contrast and produces
one result. A mapping describes multiple named contrasts and produces a
dictionary with the same keys. A mapping is the only batch form; unnamed
sequences of contrast definitions are invalid because a flat sequence already
represents one numeric contrast.

`BrainData` forwards contrast definitions to its fitted `Glm`. It does not parse
expressions or implement contrast arithmetic or inference independently.

With `inference=False`, one contrast returns an effect `BrainData`; a mapping
returns keyed effect objects. Numeric effect calculation equals
`contrast @ glm_betas.data`. String expressions provide the named syntax that
plain coefficient arithmetic cannot.

With `inference=True`, one contrast returns
`ContrastResult[BrainData]`; a mapping returns keyed results. The frozen generic
`ContrastResult` is exported from `nltools.models` and contains `effect`,
`variance`, `standard_error`, `statistic`, `z_score`, `p_value`, and
`degrees_of_freedom`. All inferential semantics and validation are defined by
the GLM specification.

## Group inference

`ttest()` remains the concise intercept-only group test. Its input is a stack of
subject-level effect maps, not first-level statistic maps, and its default
p-value is two-sided. Its existing signature and result structure remain open
to a separate audit; they are not implicitly replaced by `ContrastResult`.

A multi-regressor second-level analysis uses `fit(model="glm", ...)` with an
OLS `Glm` and a second-level `DesignMatrix` containing one row per effect map.
`compute_contrasts()` then tests the fitted second-level coefficients. This
estimates variance across effect maps and does not propagate first-level effect
variance.

## Bootstrap results

Every supported bootstrap statistic returns one stable result structure:

```python
@dataclass(frozen=True)
class BootstrapResult(Generic[Payload]):
    estimate: Payload
    standard_error: Payload
    ci_lower: Payload
    ci_upper: Payload
    samples: Samples | None = None
```

`Samples` is a placeholder for the unresolved retained-sample payload types
listed under Open design questions; it is not yet an approved public alias.

`estimate` is the statistic evaluated once on the original full sample.
`standard_error` is the sample standard deviation of the bootstrap replicates.
The result does not expose the replicate mean as the estimate and does not
provide generic `z`, `p`, or `tail` outputs. Those quantities would require a
separately designed bootstrap hypothesis test with null resampling and a
pivotal or studentized statistic.

`bootstrap` accepts one `confidence_level`, which defaults to `0.95`; it does
not accept separate percentile bounds. It returns the central percentile
interval using NumPy's linear interpolation. For `B` replicates and confidence
level `c`, the streaming accumulator retains the running variance and this many
of the smallest and largest values per output element:

```text
k = ceil((B - 1) * (1 - c) / 2) + 1
```

This reproduces the interval from the complete distribution without retaining
every replicate. At 95% confidence it stores approximately 5% of the replicate
values. Aggregation runs on the CPU and is mergeable across bounded batches or
workers.

`return_samples` controls only whether the complete distribution is retained
and returned. It never changes interval semantics. Returned samples place the
bootstrap axis first. A different confidence level requires a new bootstrap run
unless the complete distribution was retained.

Tail storage is memory-efficient, not constant-memory. It scales with
`(1 - confidence_level) * B * output_size`, where `B` is the number of
bootstrap replicates. The centralized backend planner uses `memory_budget_gb`
to preflight the output and choose output blocks, worker concurrency, and
device batches. GPU estimators may calculate replicates on the GPU, but
accumulation remains on the CPU.

Non-model bootstrap statistics are the explicit strings `"mean"`, `"median"`,
`"std"`, `"sum"`, `"min"`, and `"max"`. `bootstrap` does not accept arbitrary
callables or dynamically dispatch to other `BrainData` methods. Each statistic
reduces the resampled observation axis and returns one spatial estimate.

## Ridge bootstrap behavior

Ridge coefficient and prediction bootstraps require the original training `X`
explicitly because fitted objects do not retain it. Prediction bootstraps also
require `X_test`. The fitted `Ridge` supplies `alpha_` and, for a banded model,
`feature_space_weights_`; it does not supply the observations being resampled.
Calling either model-bootstrap mode without `X` raises even when the same
features were passed to `fit`.

Ordinary Ridge uses matrices. Banded Ridge uses named mappings aligned to the
fitted feature-space names and widths. Every training feature space and
`BrainData.data` is resampled with the same row indices. `X_test` may contain a
different number of rows but must preserve the fitted feature structure.

Bootstrap refits hold selected hyperparameters fixed. Ordinary Ridge retains
`alpha_`; banded Ridge retains both `alpha_` and
`feature_space_weights_`. Bootstrap computation uses the shared
fixed-hyperparameter refit path and never reruns cross-validation or random
search.

`"weights"` summarizes coefficients from the fitted encoding Ridge model.
`"predict"` summarizes brain-response predictions obtained by applying each
refitted coefficient array to `X_test`. Neither mode bootstraps MVPA decoding or
silently converts decoding weight maps into encoding-model results.

## Open design questions

The following contracts are intentionally unresolved and must be settled before
this specification is complete:

- the complete `bootstrap()` signature around the settled non-model statistics
  and Ridge `"weights"` and `"predict"` modes;
- the concrete payload types for `BootstrapResult.samples`, especially for
  coefficient distributions with both bootstrap and feature axes;
- the exact output-shape convention for scalar, single-map, and stacked results;
- standalone serialization of fitted estimators, compact GLM contrast state,
  row metadata, and attached results; and
- the exact internal result-construction representation that enforces the row
  metadata policy without duplicating logic.

## Required tests

Tests must establish:

- exact public signatures, defaults, keyword-only boundaries, and absence of
  removed aliases;
- complete source immutability and ownership under `fit(inplace=False)`;
- replacement rather than coexistence of GLM and Ridge state;
- exhaustive fit-state clearing after every in-place data or axis mutation;
- row-metadata preservation, aligned selection, and clearing for every output
  axis category;
- exact estimator delegation and attached result shapes for GLM, ordinary
  Ridge, and banded Ridge;
- deterministic `predict` dispatch and all invalid argument/state combinations;
- exact `predict` signature and rejection of every removed argument and alias;
- one-dimensional MVPA targets and partition-only cross-validation;
- exact built-in estimator pipelines and caller-supplied pipeline preservation;
- coefficient back-projection through every supported preprocessing step;
- binary, multiclass one-vs-rest, native multiclass, and regression weight-map
  shapes without class averaging;
- strict whole-brain, ROI, and searchlight `Predict` field combinations;
- derived score summaries and the absence of duplicated or fold-specific
  coefficient state;
- GLM named-column and Ridge named-feature-space prediction alignment;
- effect-only and inferential contrasts for single and mapping inputs;
- exact streaming percentile intervals versus a retained full distribution;
- deterministic and mergeable bootstrap aggregation;
- bootstrap output-memory preflight and CPU/GPU batching; and
- Ridge bootstrap refitting without hyperparameter reselection.
