# BrainData specification

> Release scope: BrainCollection and collection-only requirements in this specification
> are deferred to 0.6.1. Retained BrainData and estimator contracts remain targets for 0.6.0.

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
functions or the `_Glm` and `_Ridge` estimators.

`BrainData` remains the return type of `fit`. There is no separate fitted-data
wrapper. Model prediction and MVPA decoding continue to share the existing
`predict` method; there is no separate public `decode` method.

`copy()` returns a complete, independently owned snapshot. The copy includes
data, mask and masker state, row metadata, a fitted estimator, and attached fit
results when present. Mutating the copy must not affect the source. Python's
`copy.copy()` and `copy.deepcopy()` produce the same complete snapshot; there
is no public shallow-copy operation.

One internal graph-copy engine owns object allocation, `deepcopy` memo handling,
attribute traversal, and alias preservation. It is shared with `Adjacency`, so it
lives in `nltools/data/ownership.py`; the result constructors built on it stay in
`nltools/data/braindata/utils.py`. Callers use narrow semantic entry points rather
than selecting independent copy flags:

```python
_copy_complete(source, memo=None)
_copy_for_fit(source)
_result_from_array(source, data, *, rows: Literal["preserve", "clear"])
_result_from_selection(source, index)
_result_from_rows(source, data, *, X, Y)
_result_with_mask(
    source,
    data,
    mask,
    *,
    rows: Literal["preserve", "clear"],
)
```

`copy()`, `__copy__`, and `__deepcopy__` delegate to `_copy_complete`
(`nltools/data/ownership.py`).
`_copy_for_fit` excludes every attribute in `_FIT_STATE_ATTRIBUTES` before
copying retained state, so `fit(inplace=False)` does not copy an old estimator
merely to delete it. Both copy operations independently own every retained
mutable value.

The array, replacement-row, and replacement-mask constructors require final
data owned independently from the source. The selection constructor derives
its data and row metadata by applying `index` once to the source data, `.X`,
and `.Y`. Every result constructor excludes fitted state and applies the
row-metadata policy below atomically. `_result_from_rows` validates replacement
data, `.X`, and `.Y` row counts before returning.

All results independently own valid mask state. A result that preserves the
voxel axis independently copies compatible masker state. `_result_with_mask`
owns the replacement mask, recomputes all mask-derived spatial state, and
resets or rebuilds the masker rather than retaining one fitted for another
voxel axis.

All semantic entry points use the same graph-copy engine and expose no
ownership or data-copy controls. Every mutable value retained by the result is
independently owned.

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
- An operation that creates a new row axis supplies its complete replacement
  `.X` and `.Y` to `_result_from_rows` rather than attaching metadata after
  construction. Pairwise transformation clears `.X` and installs only its
  generated `.Y`.

With `ignore_attrs=False`, `append()` requires each metadata family to be empty
on both operands or populated with compatible schemas on both operands. A
one-sided or schema-incompatible `.X` or `.Y` raises. With `ignore_attrs=True`,
the result clears both metadata frames.

One shared result-construction policy must enforce these rules. Individual
methods must not mechanically copy row metadata and then repair mismatches.

## Spatial transformations

Spatial resampling and masking are separate operations:

```python
BrainData.resample(
    *,
    img=None,
    resolution=None,
    interpolation=None,
) -> BrainData

BrainData.apply_mask(mask) -> BrainData
```

`resample()` accepts exactly one of `img` or `resolution`. An `img` supplies only
the target grid; its intensity values do not define the output mask. A positive
`resolution` supplies an isotropic voxel size in millimeters. `interpolation`
is `"nearest"`, `"linear"`, `"continuous"`, or `None` for the data-aware
default. The method uses nearest-neighbor interpolation to resample the source
mask onto the target grid, then installs an independent copy on the result.

`apply_mask()` changes mask support without changing the grid. The supplied mask
must be a single three-dimensional image on the same grid and with the same
affine as the source. A mismatch raises rather than resampling either operand.
To apply a mask from another grid, callers first use `resample()` explicitly.

Both operations return independently owned results. They preserve row-aligned
`.X` and `.Y`. They always clear every fitted-state attribute because either
operation can change the voxel axis. There is no `resample_to()` alias or
`resample_mask_to_brain` flag.

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
retains its unprefixed name because both estimators accept it.
`ridge_progress_bar` maps to `_Ridge.progress_bar`; there is no GLM progress
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

- `model_`: the fitted `_Glm`;
- `glm_betas`: one map per design column;
- `glm_residual`: one row per training observation;
- `glm_predicted`: one row per training observation; and
- `glm_r2`: one fit-quality map.

A successful Ridge fit attaches only:

- `model_`: the fitted `_Ridge`;
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
    estimator_kwargs: dict | None = None,
    cv: int | BaseCrossValidator | None = None,
    groups: ArrayLike | str | None = None,
    scoring: str | Callable | None = None,
    spatial_scale: Literal["whole_brain", "roi", "searchlight"] = "whole_brain",
    roi_mask: NiimgLike | None = None,
    radius: float = 10.0,
    plot: bool = False,
    n_jobs: int = 1,
    progress_bar: bool = False,
) -> BrainData | Predict
```

Static overloads expose the mode-specific return types: fitted-model prediction
returns `BrainData`, while MVPA returns `Predict`. They do not change the single
runtime signature.

`predict` resolves exactly one mode before doing any work:

- An explicit `y=` requests MVPA decoding.
- An explicit `X=` requests prediction from a fitted `_Glm` or `_Ridge`.
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

The two ridge shortcuts select their penalty inside each training fold, by an
inner cross-validation over `RIDGE_ALPHA_GRID` — ten log-spaced values from
`1e-3` to `1e6` — rather than fitting at scikit-learn's default `alpha=1`, which
is effectively no penalty at whole-brain scale. `estimator_kwargs` is merged
over a shortcut's own constructor options, so a caller's key wins; passing it
alongside a caller-supplied estimator raises `ValueError`, because that
estimator is used exactly as given.

`plot=False` by default. With `plot=True`, whole-brain decoding draws its
cross-validated figures as a side effect and returns the same `Predict`:
the predicted-versus-actual scatter for a regression, the ROC of the out-of-fold
decision values plus the margin or probability figure for a binary
classification, and the weight map in both cases. A multiclass target and any
spatial scale other than `"whole_brain"` raise before fitting, because neither
produces the per-observation values those figures are drawn from.

Built-in classification shortcuts use one-vs-rest for multiclass targets.
MVPA does not wrap a caller-supplied classifier or override its multiclass
strategy. Callers who want one-vs-rest behavior must supply a
`OneVsRestClassifier`.

Each shortcut selects a predefined pipeline that performs preprocessing and
estimation within each cross-validation fold. For a caller-supplied
scikit-learn estimator or `Pipeline`, MVPA does not add, remove, or reconfigure
preprocessing steps. Callers include any custom scaling, dimensionality
reduction, or feature selection in that estimator or pipeline.

Every MVPA pipeline's preprocessing steps must come from the supported
transformer whitelist below, in every spatial scale. Whole-brain and ROI
pipelines must additionally end in an estimator that exposes `coef_`, because
those two scales extract a weight map: their coefficients must project back to
the original whole-brain or parcel voxel axis. An incompatible estimator or
pipeline raises `ValueError`. Every successful whole-brain or ROI result
includes `Predict.weight_map`. Searchlight builds no coefficient map — it would
have to combine coefficients from overlapping local models — so it requires the
whitelist but not `coef_`.

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

`cv` and `groups` apply only to MVPA. Prediction from a fitted `_Glm` or `_Ridge`
never constructs or evaluates cross-validation folds; it delegates directly to
the fitted estimator. In MVPA, `groups` is optional. With a caller-supplied
splitter it is passed to that splitter's `split()`, including splitters such as
`LeaveOneGroupOut`; with `cv=None` or an integer it selects the group-aware
splitter.

MVPA follows scikit-learn's cross-validation grammar. `cv=None` selects five
folds and an integer selects that many, and both mean a deterministic,
unshuffled stratified K-fold: `StratifiedKFold` on the class labels for a
classifier, and `StratifiedKFold` on quantile bins of `y` for a regressor so
the outcome distribution matches across folds. When `groups` is supplied both
become `StratifiedGroupKFold` on the same strata, so no group straddles the
train/test boundary. Quantile stratification needs at least two rows per fold;
a shorter continuous target raises before splitting rather than surfacing
scikit-learn's message about a class the caller never had. A scikit-learn cross-validation splitter is used as
supplied. The nltools-specific `"loo"` and `"logo"` aliases are removed.
`predict` has no `random_state` argument: callers configure randomness on the
estimator or splitter that owns it.

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

`BrainData` forwards contrast definitions to its fitted `_Glm`. It does not parse
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

`ttest()` remains the concise intercept-only group test on a stack of
subject-level effect maps, with two-sided p-values by default. The shared
[one-sample t-test contract](ttest.md) defines its dictionary results,
permutation nulls, shapes and ownership. It does not use `ContrastResult`.

A multi-regressor second-level analysis uses `fit(model="glm", ...)` with an
OLS `_Glm` and a second-level `DesignMatrix` containing one row per effect map.
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
    samples: np.ndarray | None = None
```

For `BrainData.bootstrap`, `Payload` is always `BrainData`. The four summary
payloads have identical `.data.shape`, own their data independently, preserve
the source mask, and clear row metadata and fitted state. When present,
`samples` is an independently owned NumPy array. The frozen result prevents
field rebinding; it does not make the contained objects immutable.

The record is defined in `nltools.data.results` for internal result
construction but is not re-exported from `nltools.data` or another public
namespace. Users receive it from `bootstrap` and interact with its named
fields; they do not need to import or construct it.

`BrainData.bootstrap` has this public signature:

```python
BrainData.bootstrap(
    statistic,
    *,
    X=None,
    X_test=None,
    n_samples=5000,
    confidence_level=0.95,
    device="cpu",
    memory_budget_gb=None,
    return_samples=False,
    n_jobs=-1,
    random_state=None,
    progress_bar=False,
) -> BootstrapResult[BrainData]
```

`statistic` is required and accepts only `"mean"`, `"median"`, `"std"`,
`"sum"`, `"min"`, `"max"`, `"weights"`, or `"predict"`. An unknown value
raises `ValueError`. The removed `stat` keyword raises `TypeError`. `n_samples`
is the number of bootstrap replicates.

Arguments are validated by mode before resampling:

- A basic statistic does not require a fitted model and rejects `X`, `X_test`,
  and `device="gpu"`.
- `"weights"` requires a fitted `_Ridge` and explicit training features `X`. It
  rejects `X_test`.
- `"predict"` requires a fitted `_Ridge`, explicit training features `X`, and
  evaluation features `X_test`.
- Both Ridge modes reject any fitted estimator other than `_Ridge`.

For Ridge, `X` must contain the training feature values in their original row
order; it need not be the same Python object passed to `fit`. `device` accepts
only `"cpu"` or `"gpu"`. An explicit GPU request runs Ridge refits on an
available CUDA or MPS backend or raises. `memory_budget_gb` governs retained
output and CPU-worker planning for every mode, and GPU-batch planning for the
two Ridge modes. `n_jobs` is the CPU-worker ceiling, including aggregation
around GPU refits; the planner may use fewer workers.

`n_samples` must be an integer of at least two. `confidence_level` must be
finite and strictly between zero and one. `memory_budget_gb`, when supplied,
must be finite and positive.

This method implements an IID row bootstrap. `BrainData.data` must be a
two-dimensional stack containing at least two observations. Each replicate
draws exactly `n_obs` row indices with replacement. For Ridge, the same indices
resample the response data and every training feature space together.

Rows must be exchangeable for the resulting uncertainty estimates to be
meaningful. The method does not implement grouped, clustered, stratified, or
block resampling. In particular, users must not treat an autocorrelated fMRI
time series as IID rows; dependent observations require a separately designed
resampling procedure.

### Bootstrap output shapes

A basic statistic returns one spatial map, so every summary has
`data.shape == (n_voxels,)`, matching the existing single-map convention for
`BrainData` reductions. Its retained samples have shape
`(n_samples, n_voxels)`.

Ridge weights retain their feature axis. Every summary has shape
`(n_features, n_voxels)`, and retained samples have shape
`(n_samples, n_features, n_voxels)`. Banded features use the concatenated
fitted feature order defined by `_Ridge`; they do not introduce a different
return type.

Ridge predictions retain the `X_test` row axis. Every summary has shape
`(n_test, n_voxels)`, and retained samples have shape
`(n_samples, n_test, n_voxels)`. A singleton feature or test-row axis is not
squeezed.

`estimate` is the statistic evaluated once on the original full sample. For
`"weights"`, it is an independent copy of the fitted full-data coefficients.
For `"predict"`, it is the fitted full-data model evaluated at `X_test`.
`standard_error` is the elementwise sample standard deviation of the bootstrap
replicates with `ddof=1`. The result does not expose the replicate mean or
generic `z`, `p`, or `tail` outputs. Those quantities require a separately
defined bootstrap hypothesis test.

Basic statistics use the corresponding NumPy reduction over the observation
axis:

```text
"mean"   -> np.mean(data, axis=0)
"median" -> np.median(data, axis=0)
"std"    -> np.std(data, axis=0, ddof=0)
"sum"    -> np.sum(data, axis=0)
"min"    -> np.min(data, axis=0)
"max"    -> np.max(data, axis=0)
```

Every replicate applies the same operation after resampling rows. In
particular, `"std"` matches `BrainData.std()` and NumPy's population standard
deviation. This is distinct from `BootstrapResult.standard_error`, which uses
`ddof=1` across bootstrap replicates. The operations propagate non-finite
values; `bootstrap` does not substitute their `nan*` variants.

`bootstrap` accepts one `confidence_level`, which defaults to `0.95`; it does
not accept separate percentile bounds. It returns the central percentile
interval using NumPy's linear interpolation. These are elementwise marginal
intervals: the nominal confidence level applies separately to each voxel,
feature, or test-row element. The result does not imply simultaneous coverage
or multiple-comparison control across an output map or stack.

For `B` replicates and confidence level `c`, the streaming accumulator retains
the running variance and this many of the smallest and largest values per
output element:

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

The full-sample estimate and every completed replicate are converted to CPU
NumPy `float64` arrays. Aggregation and optional sample retention use those
converted values, so all five result fields are `float64`. GPU Ridge refits
still use Himalaya's required `float32` computation; conversion occurs only
after each fit. Memory preflight budgets eight bytes per retained output value.

When `return_samples=True`, recomputing the sample standard deviation and
linear percentiles from `samples` must match `standard_error`, `ci_lower`, and
`ci_upper` within float64 numerical tolerance.

`random_state` deterministically derives one independent seed per replicate.
Those seeds define the same resampled row indices regardless of CPU worker
count, memory-driven batching, or CPU/GPU execution. Retained samples remain in
replicate-index order rather than worker-completion order. Scheduling changes
and GPU out-of-memory batch splitting must not change which resamples are
evaluated.

An integer `random_state` reproduces that seed sequence. `None` draws fresh
entropy and is intentionally nondeterministic.

A successful call contains exactly the `n_samples` preassigned replicates. A
recoverable out-of-memory condition may retry the same replicate and row
indices with a smaller batch. A terminal fitting or numerical failure raises
the entire call and identifies the replicate index. Failed replicates are never
dropped, and replacement row samples are never drawn.

The same `random_state` always produces the same resample indices and replicate
order. Numerical reproducibility of the fitted values follows the selected
backend and its libraries. CPU and GPU results are compared within explicit
tolerances rather than byte-for-byte because GPU Ridge arithmetic remains
`float32`; converting completed outputs to `float64` does not recreate CPU
arithmetic.

Tail storage is memory-efficient, not constant-memory. It scales with
`(1 - confidence_level) * B * output_size`, where `B` is the number of
bootstrap replicates. The centralized backend planner uses `memory_budget_gb`
to preflight output storage and choose output blocks, worker concurrency, and
device batches. GPU Ridge refits run on the GPU; accumulation remains on the
CPU.

If mandatory tail storage, retained samples, or final summary payloads cannot
fit the memory budget, `bootstrap` raises before resampling. It does not weaken
the confidence interval, reduce `n_samples`, or disable `return_samples`.

`bootstrap` does not accept callable statistics or dispatch dynamically to
other `BrainData` methods.

## Ridge bootstrap behavior

Ridge bootstrap resamples the explicitly supplied training `X`; fitted objects
do not retain a hidden copy. The fitted `_Ridge` supplies `alpha_` and, for a
banded model, `feature_space_weights_`, but not the observations or features
being resampled.

Ordinary Ridge uses matrices. Banded Ridge uses named mappings aligned to the
fitted feature-space names and widths. Every training feature space and
`BrainData.data` is resampled with the same row indices. `X_test` may contain a
different number of rows but must preserve the fitted feature structure.

Bootstrap refits hold selected hyperparameters fixed. Ordinary Ridge retains
`alpha_`; banded Ridge retains both `alpha_` and
`feature_space_weights_`. Bootstrap computation uses the shared
fixed-hyperparameter refit path and never reruns cross-validation or random
search.

A `"weights"` replicate is the coefficient array from one
fixed-hyperparameter refit. A `"predict"` replicate applies that refitted
coefficient array to the unchanged `X_test`. Neither mode runs MVPA or returns
decoding weight maps.

## Persistence

Public `BrainData.write` persists the data container, not a fitted-object
snapshot. Writing a fitted object is allowed, but loading the result always
produces an unfitted `BrainData`.

NIfTI output is an image export containing the data and spatial geometry. It
does not retain row metadata. HDF5 output is portable `BrainData` persistence
containing the data, mask by value, and row-aligned `.X` and `.Y`. If a mask is
file-backed, HDF5 retains only the basename of its filename, never the parent
path. The reconstructed in-memory mask reports that basename from
`get_filename()`; a mask created in memory continues to report `None`. Embedded
mask data and geometry are authoritative, and no operation reopens the retained
basename. Neither format stores `model_`, attached fit maps, result records,
masker caches, or execution settings.

The HDF5 input boundary reads only the current layout; it recognizes a file
written by nltools 0.5.1 or earlier and raises, directing the user to export
that file to NIfTI or CSV under 0.5.1 before upgrading.

Internal `BrainCollection` caches are deferred to 0.6.1 along with the rest of
`BrainCollection`; nothing in 0.6.0 implements them. When they arrive they use a
separate, explicitly versioned format, preserve the complete fitted-member state
specified in `braincollection.md`, and are not accepted as public `BrainData`
input files.

`Predict`, contrast, and bootstrap records are separate returned values rather
than attached `BrainData` state. Writing a `BrainData` payload extracted from
one of those records writes only that map or stack.

## Required tests

Tests must establish:

- exact public signatures, defaults, keyword-only boundaries, and absence of
  removed aliases;
- complete source immutability and ownership under `fit(inplace=False)`;
- equivalent independent snapshots from `BrainData.copy()`, `copy.copy()`, and
  `copy.deepcopy()`;
- replacement rather than coexistence of GLM and Ridge state;
- exhaustive fit-state clearing after every in-place data or axis mutation;
- row-metadata preservation, aligned selection, and clearing for every output
  axis category;
- exact `resample()` and `apply_mask()` signatures; exactly one of `img` or
  positive `resolution`; target images used only as grids; nearest-neighbor
  source-mask resampling; same-grid mask application without implicit
  resampling; preservation of `.X` and `.Y`; fitted-state clearing; independent
  output ownership; and absence of `resample_to()` and
  `resample_mask_to_brain`;
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
