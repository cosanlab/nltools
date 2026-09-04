# BrainCollection specification

This working specification defines the approved `BrainCollection` contracts and
records unresolved decisions explicitly. Code, tests, and docstrings must
implement the settled sections. Compatibility notes and migration
history belong elsewhere.

The member-level contracts referenced here are authoritative in
[`braindata.md`](braindata.md), [`glm.md`](glm.md), and [`ridge.md`](ridge.md).

## Purpose and foundational invariant

`BrainCollection` makes operations over multiple `BrainData` objects concise
without changing the meaning of the corresponding member operation. It is a
generic collection: members may represent subjects, runs, sessions, or other
analysis units.

Every member has the complete state and behavior of a `BrainData`, whether it
resides in memory or in the cache. A storage choice must not create a second
kind of collection member. Indexing, iteration, transformation, fitting,
prediction, contrasts, and reductions behave identically for cached and
in-memory members.

`fit()` returns a new collection of fitted members independently owned from the
source and from one another. Each member and every attached `BrainData` result
follows the ownership contract in `braindata.md`. Fitting does not mutate the
source collection or any source member. Caching changes storage location and
resource use only.

API names, errors, and docstrings use “member,” not “subject,” except when a
specific metadata field or statistical operation genuinely refers to subjects.

## Construction and indexing

`len(collection)` is the member count. There is no `n_subjects` property and no
redundant `n_items` property.

Indexing supports integers, slices, integer lists, boolean masks, and Polars
expressions. Subject-name string indexing is not supported because one subject
may have multiple runs or sessions. Users select semantic groups through
metadata and receive every matching member:

```python
runs = collection.filter(pl.col("subject") == "01")
```

## BIDS discovery

`from_bids` is a thin discovery factory:

```python
BrainCollection.from_bids(
    root,
    *,
    mask,
    task=None,
    space=None,
    sub_labels=None,
    img_filters=None,
    derivatives_folder="derivatives",
    cache_dir="./.nltools_cache",
)
```

It uses Nilearn's public `get_bids_files()` and `parse_bids_filename()`
functions, populates BIDS entity metadata, and delegates to `from_paths()`.

It does not load or pair events, confounds, sample masks, or repetition times.
It does not construct a `DesignMatrix`, infer a model, or create a Nilearn
`FirstLevelModel`. Designs and preprocessing remain explicit inputs to their
own APIs.

## Model fitting

The exact `BrainCollection.fit()` signature remains open. Its settled boundary
is:

- `model`, estimator-specific options, collection execution controls, and
  caching controls are explicit keyword parameters;
- GLM and Ridge options use the same names and defaults as `BrainData.fit`;
- `n_jobs` controls outer member-level CPU concurrency;
- `progress_bar` reports outer member-level progress, while
  `ridge_progress_bar` controls each fitted Ridge estimator;
- `glm_n_jobs=1` controls Nilearn's inner AR-group concurrency by default and
  avoids accidental nested parallelism;
- preprocessing, intercept, HRF, drift, filtering, smoothing, events, reports,
  and raw third-party arguments are absent; and
- there is no `**kwargs` or compatibility alias handling.

Collection GLM fitting supports OLS and autoregressive models. Each fitted
member contains the complete `BrainData` GLM state specified in
`braindata.md`. Cached GLM members preserve the compact contrast state
losslessly and without downcasting it to float32. Cached and in-memory members
use the same GLM contrast implementation.

Collection Ridge fitting supports ordinary and banded models. Each fitted
member contains the complete `BrainData` Ridge state. Its fitted `Ridge`
preserves coefficients, selected alpha, selection scores, named feature-space
structure, and selected feature-space weights. Cached members do not retain
training matrices or copy estimator state onto the `BrainData` facade.

## Ridge input grammar

`X` has one unambiguous structural grammar:

- one matrix supplies an ordinary feature matrix shared by every member;
- a list of matrices supplies one ordinary matrix per member;
- one named mapping supplies shared banded feature spaces;
- a list of named mappings supplies banded feature spaces per member;
- a callable returns one matrix or one named mapping for the current member;
  and
- `None` uses the collection's paired ordinary designs.

An outer list must have the same length as the collection. Each member validates
its own sample count and feature widths. Banded mappings for every member must
contain the same feature-space name set; order may differ and is aligned by
name.

Prediction accepts the corresponding shared or per-member matrix or mapping
forms. Each fitted estimator validates its own feature structure.

## Fitted member storage

Cached fitted members use the same complete internal cache representation as
other cached `BrainData` members. Model-only bundles are not collection
members. This requirement does not settle the public serialization API.

Hydration restores each member as an independently owned object graph. Its
mutable state does not alias the collection or another member, and valid aliases
within the member are preserved. Serialized mask bytes may be deduplicated, but
hydration creates separate mutable mask and masker state for each member. The
loader does not infer a bundle type from whichever fields happen to be present
or recognize obsolete development bundle layouts.

Cached execution must not change return types or supported methods. A fitted
collection remains valid for indexing, iteration, member transformations,
prediction, contrasts, and reductions.

## Prediction and decoding

The public signature is:

```python
BrainCollection.predict(
    *,
    X: PredictionDesign | Sequence[PredictionDesign] | None = None,
    y: ArrayLike | Sequence[ArrayLike] | str | None = None,
    estimator: str | BaseEstimator = "linear_svc",
    cv: int | BaseCrossValidator | None = None,
    groups: ArrayLike | Sequence[ArrayLike] | str | None = None,
    scoring: str | Callable | None = None,
    spatial_scale: Literal["whole_brain", "roi", "searchlight"] = "whole_brain",
    roi_mask: NiimgLike | None = None,
    radius_mm: float = 10.0,
    cache: Literal["auto", True, False] = "auto",
    n_jobs: int = -1,
    progress_bar: bool = False,
) -> BrainCollection | PredictCollection
```

`PredictionDesign` means a `DesignMatrix` for GLM, one matrix for ordinary
Ridge, or a named feature-space mapping for banded Ridge. A single design is
shared by every member; a sequence supplies one design per member. Prediction
does not accept a callable `X`.

For MVPA, a string `y` or `groups` selects that `.Y` column from every member. A
flat array is shared by every member; a sequence of one-dimensional arrays
supplies one value per member. The estimator, cross-validation, scoring, and
spatial contracts are otherwise identical to `BrainData.predict`.

`cv`, `groups`, `estimator`, `scoring`, and the spatial controls apply only to
MVPA. `cache`, `n_jobs`, and `progress_bar` apply to both modes. `n_jobs`
controls member-level work; each member runs with `n_jobs=1`. The collection
validates the mode, collection state, and every shared or per-member input
before scheduling workers.

The old `X_new`, `model`, `standardize`, and `random_state` arguments are not
accepted.

The collection resolves one mode before scheduling any workers:

- An explicit `y=` requests MVPA decoding.
- An explicit `X=` requests prediction from fitted member models.
- With neither argument and fitted members, it returns independently owned
  stored training predictions.
- With neither argument, no fitted models, and exactly one `.Y` column per
  member, it runs MVPA with those columns.
- Supplying both `X` and `y`, mixed fitted/unfitted member state, or another
  unresolved combination raises before partial work begins.

For stored training predictions, every member retains aligned row metadata.
Predictions for explicit new designs clear source row metadata. In-memory and
cached prediction delegate to the fitted `Glm` or `Ridge`; collection code must
not reimplement matrix multiplication, add an intercept, or special-case one
storage mode.

MVPA returns a `PredictCollection` whose members are the validated `Predict`
results defined by the BrainData specification. Every member uses the
collection call's `spatial_scale` and `scoring` arguments. Classifier results
must have identical class labels in the same order. ROI results must also have
identical `roi_labels` in the same order. The collection raises on a mismatch;
it does not reorder labels after fitting.

`PredictCollection` is a frozen result containing only:

```python
PredictCollection(
    results: tuple[Predict, ...],
    metadata: pl.DataFrame | None = None,
)
```

`results` contains one independently owned result per collection member, in
collection order. When present, `metadata` has one row per result in the same
order. Construction validates result modes, scoring specifications, class
labels, ROI labels, masks, and metadata row count before creating the object.

`PredictCollection` supports `len`, iteration, and integer indexing over
`results`. Its other attributes are derived properties:

| Property | Whole brain | ROI | Searchlight |
| --- | --- | --- | --- |
| `mean_scores` | `(n_members,)` | `(n_members, n_rois)` | `None` |
| `std_scores` | `(n_members,)` | `(n_members, n_rois)` | `None` |
| `score_table` | one row per member | one row per member and ROI | `None` |
| `weight_maps` | member stacks | member stacks | `None` |
| `score_maps` | `None` | one map per member | one map per member |

For regression and binary classification, `weight_maps` is a `BrainData` with
one row per collection member. For multiclass classification, it is a
dictionary keyed by class label. Each value is a `BrainData` with one row per
member for that class. Dictionary iteration follows the shared class-label
order. This preserves separate member and class axes; it never averages maps
across classes.

For whole-brain results, `score_table` contains one row per member with
`mean_score` and `std_score` columns plus aligned member metadata. For ROI
results, it is long-form with one row per member and ROI, repeated member
metadata, `roi_label`, `mean_score`, and `std_score` columns.

`score_maps` stacks member maps into a `BrainData` with one row per member.
Every derived `BrainData` preserves collection order and carries aligned member
metadata.

The public name is `score_table`, not `scores`, because `Predict.scores`
already names the raw fold-level arrays. Cache paths are internal execution
metadata and are not stored on `PredictCollection`. Cached and in-memory
execution return the same result type and derived values.

### Group MVPA

`predict_group` runs one group-level MVPA analysis in which collection members
are observations and returns one `Predict`:

```python
BrainCollection.predict_group(
    *,
    y: ArrayLike | str,
    estimator: str | BaseEstimator = "linear_svc",
    cv: int | BaseCrossValidator | None = None,
    groups: ArrayLike | str | None = None,
    scoring: str | Callable | None = None,
    spatial_scale: Literal["whole_brain", "roi", "searchlight"] = "whole_brain",
    roi_mask: NiimgLike | None = None,
    radius_mm: float = 10.0,
    n_jobs: int = -1,
    progress_bar: bool = False,
) -> Predict
```

The method requires exactly one map per collection member. Each member's
`BrainData.data` must have shape `(n_voxels,)` or `(1, n_voxels)`. All members
must have spatially equivalent mask values, affine, voxel order, and feature
count; equivalence never requires shared Python object identity. The method
stacks maps in collection order using NumPy's dtype-promotion rules; it never
unconditionally downcasts them. It then delegates once to `BrainData.predict`.
All estimator, cross-validation, scoring, spatial, result, and error semantics
come from that implementation.

An array-like `y` or `groups` must be one-dimensional with one value per
collection member. A string names a collection-metadata column, which is read
in collection order. Missing metadata, an unknown column, or a length mismatch
raises before stacking or fitting.

For whole-brain group MVPA, `Predict.predictions` and `Predict.cv_folds` have
one value per collection member in collection order. `predict_group` does not
mutate the collection or its members, and the returned `Predict` owns its arrays
and brain maps.

`BrainCollection.predict(y=...)` runs a separate MVPA analysis within each
member and returns a `PredictCollection`. `BrainCollection.predict_group(y=...)`
instead treats members as observations in one group-level analysis and returns
one `Predict`.

`predict_group` does not create group labels, choose a group-aware splitter, or
run permutation tests. Callers pass `groups` and a compatible cross-validation
splitter explicitly. The removed `model`, `standardize`, `n_permute`, and
`random_state` arguments raise `TypeError`. `Predict` has no permutation-only
fields.

## GLM contrasts

The public method accepts one contrast or a named mapping:

```python
compute_contrasts(
    contrasts: str | NumericVector | Mapping[str, str | NumericVector],
    *,
    inference: bool = False,
)
```

A string or flat numeric vector returns one collection result. A mapping returns
a dictionary with the same keys. With `inference=False`, payloads are effect
`BrainCollection` objects. With `inference=True`, payloads are
`ContrastResult[BrainCollection]`. Degrees of freedom are ordered by collection
member.

Each string expression is resolved against that member's fitted feature names,
so unrelated nuisance columns and column order may differ. Every referenced
name must exist in every member.

For a numeric vector, the first member's fitted feature order is canonical.
Every other member must have exactly the same feature-name set and is reordered
to canonical order. Missing or additional features raise rather than allowing
one vector to describe different scientific contrasts.

All parsing and computation delegate to fitted `Glm` objects and the shared GLM
contrast core. `BrainCollection` does not implement a second expression parser,
OLS covariance formula, or cached-only inference path.

## Execution and ownership

Execution creates one immutable task description per member. Workers write only
to distinct atomic cache targets. The coordinating process relays warnings with
member context. Exceptions identify the failing member. A failed operation
never returns a partial collection.

`n_jobs` is a concurrency ceiling, not a promise to start that many workers.
The resource planner may use fewer workers when the operation's memory budget
requires it.

No operation may mutate a materialized source member. Worker inputs are
independently owned snapshots or read-only representations, and worker outputs
become independently owned result members.

Integer indexing and iteration return complete, independently owned
`BrainData` snapshots. They never expose an internal in-memory or hydrated
member object directly.

## Open design questions

The following contracts are intentionally unresolved and must be settled before
this specification is complete:

- the exact `fit()` signature, including the placement and defaults of caching
  and execution controls;
- whether the Ridge memory budget governs each member or the entire collection,
  how it constrains outer workers, and whether GPU member fits are serialized;
- whether public string-based `apply()` survives, whether `map()` remains the
  single generic callable operation, and how callable results are validated;
- whether synchronous `transform_designs()` retains any execution controls;
- whether constructors manufacture a `subject` metadata column for generic
  members;
- callable context after removal of BIDS-specific events, confounds, sample
  masks, and repetition times;
- empty-collection behavior and whether supplied in-memory members must already
  use the collection mask or are remasked on construction; and
- the boundary between complete collection serialization and intentional image
  export.

## Required tests

Tests must establish:

- complete semantic parity between cached and in-memory members;
- fit source immutability and independent ownership of every returned member;
- generic member terminology, indexing forms, and metadata-based filtering;
- thin BIDS discovery and the absence of event, design, confound, censoring,
  repetition-time, and model side effects;
- ordinary and banded Ridge fitting with shared, per-member, reordered,
  missing, and additional feature spaces;
- OLS and AR GLM fitting and lossless cached contrast state;
- global prediction-mode validation before worker scheduling;
- stored versus new-design prediction metadata behavior;
- exact collection prediction signatures and shared versus per-member inputs;
- strict `PredictCollection` construction, metadata alignment, and derived
  whole-brain, ROI, and searchlight properties;
- binary/regression and class-keyed multiclass weight-map stacking;
- thin `predict_group` validation and delegation without dtype downcasting,
  synthetic groups, permutation state, or duplicated MVPA computation;
- single and mapped GLM contrasts, including canonical numeric alignment;
- complete `ContrastResult[BrainCollection]` assembly in member order;
- clear failures for irrelevant estimator options and all removed aliases;
- budget-based worker and batch sizing, plus explicit-GPU behavior;
- validation of generic callable results; and
- round-trip cache hydration of the complete fitted-member state.
