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

Explicit construction uses one constructor:

```python
BrainCollection(
    brains: Sequence[BrainData | PathLike],
    *,
    mask,
    designs: Sequence[DesignMatrix | PathLike | None] | None = None,
    metadata: pl.DataFrame | None = None,
    lazy=True,
    cache_dir=None,
)
```

`brains` may mix in-memory members and paths. `lazy` applies to every member or
design supplied as a path. With `lazy=True`, those paths load on demand; with
`lazy=False`, they all load during construction. In-memory objects are
unaffected. This constructor is the only public ingress for caller-supplied
member and design paths. There are no `from_paths()`, `from_bids()`, or
`from_glob()` factories or compatibility aliases. `read()` is the separate,
strict archive-loading API; it assembles manifest-declared inputs and delegates
to the constructor.

When supplied, `designs` has exactly one entry for each member in the same
order. Each entry is an in-memory `DesignMatrix`, a path, or `None`; its storage
position associates it with the corresponding member. The constructor copies
in-memory designs and metadata rather than retaining caller-owned mutable
objects.

A stored design path must be an `.h5` or `.hdf5` file written by
`DesignMatrix.write()`. Text files are data sources rather than complete
precomputed designs and are not accepted here. Construction validates each
stored design's row count against its member, using in-memory shapes or file
metadata without loading voxel arrays.

Paths remain an internal storage detail and never cross the public boundary.
`designs` returns independently owned `DesignMatrix` values or `None`.
`iter_pairs()`, `fit(X=None)`, and `transform_designs()` likewise receive values
rather than paths. Accessing a lazy design materializes and copies it rather
than exposing the collection's stored object.

Public accessors for mutable state return independent snapshots. `mask` returns
an independently owned image; `metadata` returns an independent Polars clone or
`None`; and `designs` materializes independently owned designs. For each
position, `iter_pairs()` returns an independently owned member and design, or
`None` in place of a missing design. Mutating any snapshot cannot change the
collection. `shape` returns a freshly constructed value.

A stored member design is the optional `DesignMatrix` at the same position as a
collection member. “Stored” describes that positional relationship, not a
different design type. A supplied design is an `X` used only for the current
operation. Every stored member design must have the same row count as its
member.

Stored member designs are preserved only when an operation guarantees the same
member row identity and order:

- row-preserving transformations, `fit()`, and prediction from stored training
  inputs preserve independent copies;
- `transform_designs()` replaces them;
- prediction from a newly supplied `X`, GLM contrasts, and every other
  row-replacing operation clear them; and
- `map()` always clears them because an arbitrary callback can reorder rows
  without changing the row count.

A design supplied directly to `fit()` or `predict()` is an operation input and
is never installed as a stored member design.

`len(collection)` is the member count. There is no `n_subjects` property and no
redundant `n_items` property.

Generic construction with `metadata=None` leaves metadata unset; it does not
invent a `subject` column or another public member identifier. Collection order
provides positional identity. Callers perform path discovery and metadata
extraction, then pass the resulting paths and explicit metadata to the
constructor. Metadata-dependent filtering raises when the collection has no
metadata.

`metadata` accepts only a Polars `DataFrame` or `None`. The constructor stores
an independent clone of a supplied frame, preserving its row order, column
order, names, values, and dtypes. It requires exactly one row per member. An
empty collection accepts `None` or a zero-row frame; any other row count raises.
Pandas objects, mappings, and all other input types raise without coercion, as
does a frame with duplicate columns.

An in-memory member must already match the collection mask in spatial shape,
affine, boolean support, voxel order, and feature width. Matching only the voxel
count is insufficient. If a member differs, construction identifies its
position and raises. It does not remask or resample the member because either
operation could change the meaning of its data and invalidate fitted state.

The collection stores a complete independent copy of each valid in-memory
member, including its fitted state. Each stored member receives its own copy of
the collection's canonical mask and a compatible masker. Members never share
mutable mask or masker state with the caller, the collection, or one another.

Explicit construction, slicing, and filtering may produce a valid empty
collection. It has length zero, an empty iterator, and shape
`(0, None, n_voxels)`. On an empty collection, slicing, filtering, `map()`,
row-preserving transformations, and `transform_designs()` return an empty
in-memory collection without scheduling work or creating cache artifacts.

`fit()`, `predict()`, `compute_contrasts()`, reductions, inference, group
operations, alignment, `write()`, and `export_images()` require at least one
member. They raise `ValueError("BrainCollection is empty")` before scheduling
work. `read()` raises it when the manifest declares no members.

Indexing accepts exactly:

- a Python or NumPy integer, including a valid negative index, returns an
  independently owned `BrainData`;
- a slice returns a `BrainCollection`;
- a one-dimensional sequence of integers returns a `BrainCollection` and
  preserves its order and duplicate indices;
- a one-dimensional sequence of non-null booleans with exactly one value per
  member returns a `BrainCollection`; and
- a boolean Polars expression over metadata returns a `BrainCollection`.

Every returned `BrainData` is independently owned from the source. Members of a
returned `BrainCollection` are independently owned from the source and one
another, including repeated indices. Empty selections return a valid empty
collection. A Polars expression requires metadata.

Every other selector raises rather than being coerced, including strings,
floats, scalar booleans, multidimensional arrays, nullable boolean masks, and
non-boolean Polars expressions. Subject-name string indexing is not supported
because one subject may have multiple runs or sessions. Users select semantic
groups through metadata and receive every matching member:

```python
runs = collection.filter(pl.col("subject") == "01")
```

The public filtering signature is:

```python
collection.filter(expr: pl.Expr) -> BrainCollection
```

`filter()` is a metadata query only. It requires collection metadata and
evaluates `expr` to exactly one non-null Boolean value per member. It raises if
metadata is absent or the result is non-Boolean, contains a null, or has the
wrong length. The method does not accept callables or array-like masks. Pass a
Boolean mask to the indexing API instead. For data-dependent selection, callers
explicitly compute such a mask and index the collection with it.

## Spatial transformations

The collection mirrors the member-level spatial operations:

```python
BrainCollection.resample(
    *,
    img=None,
    resolution=None,
    interpolation=None,
    cache="auto",
    n_jobs=-1,
    progress_bar=False,
) -> BrainCollection

BrainCollection.apply_mask(
    mask,
    *,
    cache="auto",
    n_jobs=-1,
    progress_bar=False,
) -> BrainCollection
```

`resample()` uses the exact `img`, `resolution`, and `interpolation` grammar of
`BrainData.resample()`. Before scheduling work, it resolves one target grid and
uses nearest-neighbor interpolation to resample the collection's canonical mask
onto it. Every member uses that grid and mask. Target-image intensities never
define mask support.

Before scheduling, `apply_mask()` validates and independently copies one
canonical output mask. The supplied mask must be a single three-dimensional
image on the collection's current grid and with its current affine. Every member
receives that mask. The method never resamples the mask or member data.

Both methods validate every completed member against the new canonical mask
before publishing the result. They preserve row-aligned member `.X` and `.Y`
and independently copy stored member designs because spatial transformations
do not change row identity. They clear every member's fitted state, leave the
source unchanged, and follow the standard collection cache policy. There is no
`resample_to()` or `resample_mask_to_brain` alias and no generic mask-changing
mode on `map()`.

## Thresholding

```python
BrainCollection.threshold(
    *,
    upper=None,
    lower=None,
    binarize=False,
    coerce_nan=True,
    cluster_threshold=0,
    cache="auto",
    n_jobs=-1,
    progress_bar=False,
) -> BrainCollection
```

`threshold()` mirrors the parameter grammar and numerical behavior of
`BrainData.threshold()`, including percentile strings, NaN handling, and
cluster filtering. Each member is thresholded independently.

The method preserves the canonical mask, row-aligned member `.X` and `.Y`,
and stored member designs. It returns independently owned results, clears every
member's fitted state, leaves the source unchanged, and follows the standard
collection cache policy. A dedicated method preserves stored designs because
thresholding preserves row identity; generic `map()` clears them.

## Model fitting

The public signature is:

```python
BrainCollection.fit(
    model="glm",
    *,
    X=None,
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
    cache="auto",
    n_jobs=-1,
    random_state=None,
    progress_bar=False,
) -> BrainCollection
```

`X=None` uses the collection's stored member designs. An explicit `X` follows
the shared or per-member grammar below for ordinary or banded designs. Unlike
`BrainData.fit`, collection fitting has no `inplace` mode: it always returns a
new independently owned collection.

The following constraints also apply:

- `model`, estimator-specific options, collection execution controls, and
  caching controls are explicit parameters;
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

### Ridge memory and concurrency

`ridge_memory_budget_gb` limits aggregate incremental Ridge working memory
across the entire `fit()` call. It is not a separate allowance for each member.
As with the member-level Ridge budget, source inputs, retained outputs, allocator
overhead, and third-party library memory are excluded. An explicit value must be
positive and finite.

With `ridge_memory_budget_gb=None`, the coordinator resolves the budget once
before creating workers or cache steps. CPU uses available host memory with
conservative headroom, CUDA uses free accelerator memory with conservative
headroom, and MPS uses available unified memory with conservative headroom.
Automatic CUDA planning must use the resolved CUDA backend rather than host
memory. The private per-batch ceiling described in `ridge.md` may cap an
individual batch, but it does not reduce the aggregate budget used to plan the
call.

For CPU Ridge, `n_jobs` is a ceiling. The resource planner reduces the outer
worker count until each concurrent member can receive an equal share of the
aggregate budget large enough for the largest member's minimum working set.
Each worker receives that share as its private Ridge budget, and inner
third-party computation cannot create a nested member-level worker pool.

For GPU Ridge, member fits are serialized regardless of `n_jobs`, and each fit
receives the full device budget. CUDA targets may remain in host memory as an
internal optimization; their storage is excluded from the accelerator budget.
MPS planning counts simultaneously live device allocations and CPU-fallback
allocations against the same unified-memory budget.

The centralized backend planner derives private target, alpha, and refit batch
sizes from the assigned budget. On a device OOM, it clears the device cache and
retries the same member with a smaller batch plan. Cross-validation splits and
banded Dirichlet candidates are reused exactly. Recovery never changes results
by dropping work, substituting candidates, or falling back from an explicitly
requested GPU. An OOM at the smallest legal batch raises `MemoryError` with
member context.

Caching is resolved before resource planning. Cached execution writes each
complete member atomically and releases its temporary state before continuing.
Uncached execution may retain final fitted members whose combined size exceeds
the working-memory budget. A preflight capacity failure raises before workers
or cache steps are created, and no failure exposes a partial fitted collection.

`fit()` forwards `random_state` unchanged to every member, so its results match
an explicit serial loop that fits each member with the same argument. Banded
Ridge therefore uses the same Dirichlet candidate set for every member. Worker
count, scheduling, and cache policy cannot change a member's candidates or
result.

Each member materializes its cross-validation splits and random-search
candidates once and reuses them for every OOM retry.

## Fit input grammar

Ridge resolves `X` in this order:

1. A mapping supplies shared named banded feature spaces.
2. Any value that can be coerced without error to a finite, numeric,
   two-dimensional array supplies one shared ordinary matrix. A nested numeric
   list therefore denotes one shared matrix.
3. Otherwise, a sequence with exactly one entry per member supplies per-member
   inputs. All entries must be two-dimensional ordinary matrices, or all must be
   named banded mappings. A call cannot mix ordinary and banded members.
4. `None` uses each member's stored ordinary design.

Per-member matrices with one feature or one row require an explicit
two-dimensional container; the parser does not infer them from a list of
one-dimensional rows. Banded mappings for every member must contain the same
feature-space name set. Mapping order may differ and is aligned by name.

For GLM, `X` is exactly one shared `DesignMatrix`, a sequence containing one
`DesignMatrix` per member, or `None` to use every stored member design. Ridge
matrices and mappings are not accepted as GLM designs. Before creating workers
or cache artifacts, the coordinator resolves every fit input and validates each
member's sample count and feature widths. An error identifies the member.

Prediction accepts the corresponding shared or per-member matrix or mapping
forms. Each fitted estimator validates its own feature structure.

## Cache policy

Every collection-returning operation with a `cache` parameter resolves it once
before scheduling work:

- `cache="auto"` resolves to `True` when at least one source member is
  path-backed and to `False` when every source member is in memory;
- `cache=True` stores every result in the complete internal member format and
  returns a path-backed collection; and
- `cache=False` returns an independently owned in-memory collection.

Stored design paths do not affect automatic resolution. Materializing inputs
during the operation cannot change the resolved policy. A valid empty result is
returned in memory without creating a cache step. Cache policy affects only
storage and resource use; it does not change result values, fitted state, return
types, or supported operations.

## Fitted member storage

Cached fitted members use the same complete internal cache representation as
other cached `BrainData` members. Model-only bundles are not collection
members. Public persistence is specified separately below.

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
MVPA. `n_jobs` and `progress_bar` apply to both fitted-model prediction and
MVPA. `n_jobs` controls member-level work; each member runs with `n_jobs=1`.
The collection validates the mode, collection state, and every shared or
per-member input before scheduling workers.

`cache` applies only to fitted-model prediction, which returns a
`BrainCollection` and follows the standard collection cache policy. MVPA always
returns a materialized `PredictCollection`: `cache="auto"` means no caching,
and an explicit `cache=True` or `cache=False` raises instead of being ignored.

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

Every MVPA result remains fully in memory and retains the complete `Predict`
state, including the all-data fitted estimator for each member. MVPA has no
cache bundles, estimator-stripping step, callable-scorer serialization, or
public cache read path.

`PredictCollection` is a frozen result containing only:

```python
PredictCollection(
    results: tuple[Predict, ...],
    metadata: pl.DataFrame | None = None,
)
```

`PredictCollection` is defined in the internal results module and is not
exported from the package namespace. Users receive it from collection
prediction. It must contain at least one result: empty collection prediction
raises before fitting, and its derived properties require a common result mode.

The constructor independently copies each input `Predict` into `results` in
collection order. When present, `metadata` is an independent Polars clone with
one row per result in the same order. Construction validates result modes,
scoring specifications, class labels, ROI labels, masks, and metadata row count
before creating the object.

`PredictCollection` supports `len`, iteration, and integer indexing over
`results`. Integer indexing and iteration return independently owned `Predict`
snapshots rather than references to stored results. Freezing the outer result
prevents attribute reassignment but does not make nested arrays and brain maps
immutable. Its other attributes are derived properties:

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

These result columns are not generated inside a `DesignMatrix`, so the `.nl_`
namespace does not apply. Before constructing `score_table`, whole-brain
results reject metadata containing `mean_score` or `std_score`; ROI results also
reject `roi_label`. A collision never overwrites or suffixes a metadata column.

`weight_maps` and `score_maps` stack member maps in collection order. Each
resulting `BrainData` stores an independent copy of collection metadata in `.X`
in the same order; when metadata is absent, `.X` is empty. Its `.Y` is always
empty. For multiclass `weight_maps`, every class-specific `BrainData` is
independently owned and follows the same `.X` and `.Y` rules. No ad hoc
member-metadata attribute is added to `BrainData`.

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
BrainCollection.compute_contrasts(
    contrasts: str | NumericVector | Mapping[str, str | NumericVector],
    *,
    inference: bool = False,
    cache: Literal["auto", True, False] = "auto",
    n_jobs: int = -1,
    progress_bar: bool = False,
)
```

A string or flat numeric vector returns one collection result. A mapping returns
a dictionary with the same keys. With `inference=False`, payloads are effect
`BrainCollection` objects. With `inference=True`, payloads are
`ContrastResult[BrainCollection]`. Degrees of freedom are ordered by collection
member.

The collection validates every requested contrast against every fitted member
before scheduling. Each member is hydrated once, and one worker computes all
requested contrasts for it. A mapping therefore does not create a separate pass
over members for each contrast. The standard cache policy applies to every
payload collection. The call is atomic across members, contrasts, and
inferential fields; a failure returns and exposes none of them.

Every effect collection and every `BrainCollection` field of an inferential
`ContrastResult` preserves collection metadata in member order, with its own
independent copy when metadata is present, and clears stored member designs.
Each member contains one map with empty `.X` and `.Y`.

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

## Generic member mapping

The sole generic member operation is:

```python
BrainCollection.map(
    fn: Callable[[BrainData], BrainData],
    *,
    cache="auto",
    n_jobs=-1,
    progress_bar=False,
) -> BrainCollection
```

Each callback receives an independently owned copy of the current member. It
must return a `BrainData` with the collection's exact spatial shape, affine,
boolean mask support, voxel order, and feature width. A wrong return type or
changed mask raises with member context. The stored result is an independent
copy of the callback's return value and shares no mutable state with any other
result.

`map()` cannot change the collection mask. Mask-changing operations such as
resampling use dedicated methods that establish one new canonical mask and
validate every returned member against it. These methods call the internal
execution primitive directly; they do not dispatch through `map()` or a
method-name string.

There is no public `apply()` method or compatibility alias. A string-based
dispatcher would bypass explicit public signatures and duplicate the dedicated
methods.

## Design transformation

```python
BrainCollection.transform_designs(
    fn: Callable[[DesignMatrix], DesignMatrix],
) -> BrainCollection
```

`transform_designs()` runs synchronously in the coordinating process. It does
not schedule workers or write to the cache. It calls `fn` only for non-`None`
entries and preserves every `None`. Each call receives an independent copy of
the stored member design and must return a `DesignMatrix` whose row count
matches that member. A type or row-count error identifies the member.

The returned collection owns an independent copy of every transformed design.
It neither mutates the source collection nor retains aliases to source designs
or callback return values. The method has no `n_jobs`, `progress_bar`, or
`cache` parameters.

## Persistence and image export

Collection persistence and image export use separate public methods:

```python
collection.write(directory: PathLike) -> None

BrainCollection.read(
    directory: PathLike,
    *,
    lazy=True,
    cache_dir=None,
) -> BrainCollection

collection.export_images(
    directory: PathLike,
    *,
    include_metadata=True,
) -> tuple[Path, ...]
```

`write()` creates a versioned directory archive that round-trips the portable
collection data: member order, the canonical mask, each member's data and
row-aligned `.X` and `.Y`, every stored member design, and metadata with its
schema and values intact. Metadata remains absent when it is `None`. Loading an
archive always produces unfitted members, matching public `BrainData`
persistence. Fitted models, attached fit maps, masker caches, original source
paths, loaded state, cache lineage, and execution settings are not stored.

The required manifest declares the exact artifact kind and schema version and
records member order and design presence; readers do not infer either from the
directory contents. The archive stores one mask by value, lossless metadata
when present, ordinal member files, and ordinal design files for members that
have designs. A file-backed mask retains only its original basename, while a
programmatically created mask retains `None`, following `braindata.md`. Each
hydrated member receives independently owned mask and masker state.

`read()` accepts only this public collection archive. It requires no separate
mask argument and never accepts an internal cache directory or falls back to
globbing images. With `lazy=True`, member and design files load on demand;
`lazy=False` loads them during construction. A missing, malformed, wrong-kind,
or unsupported-version archive raises without a compatibility fallback or
field-based format detection.

`export_images()` creates a deliberately lossy image-exchange directory. It
writes fixed ordinal NIfTI filenames in member order, one copy of the canonical
mask, and `metadata.csv` only when metadata exists and
`include_metadata=True`. It exports only each member's current data and spatial
geometry. It omits `.X`, `.Y`, stored member designs, fitted state, attached fit
maps, maskers, source paths, and cache lineage. There is no image-directory mode
in `read()`; users may explicitly construct a new unfitted collection from the
exported files.

Both `write()` and `export_images()` reject an existing destination instead of
merging or overwriting it. They build a uniquely named sibling staging
directory and publish the complete directory atomically; a failure removes only
that staging directory and leaves no partial destination. Both reject an empty
collection with `ValueError("BrainCollection is empty")`.

Internal caches remain separate, disposable, explicitly versioned execution
artifacts. They preserve complete fitted-member state for continued collection
operations but are not public archives. Public persistence and internal caching
may share typed member and design serialization helpers, but not top-level
layouts, lifecycle rules, or format detection.

## Cache lifecycle

Caching is an internal execution detail. `BrainCollection` has no public cache
lifecycle API and does not expose `load()`, `unload()`, `is_loaded`,
`memory_estimate`, `cache_root`, `steps`, `cleanup()`, or `cleanup_all()`.

`cache_dir` selects the parent for caches created by a collection and its
operations. With `cache_dir=None`, nltools uses the system temporary directory;
an explicit path is used only as the parent. When a cache is needed, nltools
creates a uniquely named child and owns only that child. It never treats the
parent as disposable or deletes unrelated files within it.

Each cache has one internal owner shared by every collection or result that
depends on its artifacts. The owner keeps the cache available until the last
dependent releases it, then removes the owned child through internal
finalization. Cleanup uses neither a public protocol nor a `__del__` method.
Source image and design paths supplied by the user are never owned or removed.

`lazy=False` eagerly materializes path-backed inputs. `cache=False` returns
in-memory results. Neither behavior adds a public loaded-state concept. Public
`write()` is the sole durable collection persistence API. Internal caches
cannot be reopened through a public method.

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

## Required tests

Tests must establish:

- complete semantic parity between cached and in-memory members;
- fit source immutability and independent ownership of every returned member;
- generic member terminology, exact selector return types, independent copies,
  strict selector validation, preserved selection order and duplicates, empty
  selections, and metadata-based filtering;
- the exact `filter(expr: pl.Expr)` signature; metadata as a requirement;
  exactly one non-null Boolean result per member; rejection of results that are
  non-Boolean, contain nulls, or have the wrong length; rejection of callables
  and array-like masks; Boolean-mask indexing; and explicit mask computation for
  data-dependent selection;
- Polars-only metadata ingress; an independent clone preserving row and column
  order, names, values, and dtypes; exactly one row per member; `None` or a
  zero-row frame for an empty collection; rejection without coercion of pandas,
  mappings, duplicate columns, and all other inputs; and no synthetic metadata;
- empty construction, slicing, filtering, `map()`, row-preserving operations,
  and `transform_designs()` returning in-memory without scheduling or cache
  work; the exact pre-scheduling `ValueError` from `fit()`, `predict()`,
  `compute_contrasts()`, reductions, inference, group operations, alignment,
  `write()`, and `export_images()`; and rejection of zero-member archive
  manifests;
- lazy loading of every brain path and every HDF5 design written by
  `DesignMatrix.write()`, with in-memory inputs unaffected; eager loading of all
  paths with `lazy=False`; rejection of other design paths; row-count validation
  from shapes or metadata without loading voxel arrays; and path-free,
  independently owned design access;
- independent snapshot access for the canonical mask image, a Polars metadata
  clone or `None`, materialized designs, and each member and design from
  `iter_pairs()`, with no mutation of collection state through returned objects;
  and a fresh `shape` value;
- no public cache-lifecycle API; the system temporary directory as parent when
  `cache_dir=None`; an explicit `cache_dir` used only as a parent; one uniquely
  named owned child per cache; preservation of the parent and unrelated files;
  one internal owner shared across dependent collections and results; removal
  through internal finalization after the last dependent releases the cache; no
  `__del__` cleanup; no deletion of user-supplied member or design paths; and
  `write()` as the sole durable persistence API;
- one-time pre-scheduling cache resolution, including the exact `cache="auto"`
  rule from source member paths; explicit path-backed and in-memory result modes;
  independence from stored design paths and later materialization; and in-memory
  empty results without cache artifacts;
- the constructor as the only ingress for caller-supplied member and design
  paths; strict archive `read()` as the sole alternative path-loading API; no
  `from_paths()`, `from_bids()`, or `from_glob()` factories or compatibility
  aliases; and caller-owned path discovery and metadata extraction;
- exact collection `resample()` and `apply_mask()` signatures; exactly one of
  `img` or `resolution` for `resample()`; one canonical output grid and mask
  resolved before scheduling; cached and in-memory mask agreement; target
  images used only as grids; nearest-neighbor source-mask resampling; same-grid
  mask application without implicit resampling; preservation of member `.X`
  and `.Y` and stored member designs; fitted-state clearing; source
  immutability; and absence of `resample_to()`, `resample_mask_to_brain`, and a
  generic mask-changing `map()` mode;
- the exact collection `threshold()` signature and numerical parity with
  member-level thresholding, including percentile strings, `binarize`,
  `coerce_nan`, and `cluster_threshold`; preservation of the canonical mask,
  member `.X` and `.Y`, and stored designs; independent result ownership,
  unconditional fitted-state clearing, source immutability, and parity across
  cache modes and worker counts;
- ordinary and banded Ridge fitting with shared, per-member, reordered,
  missing, and additional feature spaces; parsing nested numeric lists as shared
  matrices; rejection of mixed per-member modes; and GLM input restricted to
  shared, per-member, or stored `DesignMatrix` values;
- one-time aggregate Ridge budget resolution on CPU, CUDA, and MPS; memory-capped
  CPU worker counts; centralized target, alpha, and refit batch sizing;
  serialized GPU fits; deterministic same-member OOM retries without fallback;
  and preflight failure without partial outputs or cache artifacts;
- unchanged `random_state` forwarding to every member and equivalence to an
  explicit serial loop; identical banded candidate sets across members,
  scheduling, worker counts, and cache modes; and one-time per-member
  materialization and retry reuse of cross-validation splits and random-search
  candidates;
- OLS and AR GLM fitting and lossless cached contrast state;
- global prediction-mode validation before worker scheduling;
- stored versus new-design prediction metadata behavior;
- exact collection prediction signatures and shared versus per-member inputs;
  `n_jobs` and `progress_bar` behavior in both modes;
- standard collection caching only for fitted-model prediction; uncached MVPA
  under `cache="auto"`; rejection of explicit boolean cache values in MVPA
  mode; and complete in-memory `Predict` state, including all-data fitted
  estimators, without cache bundles, estimator stripping, callable-scorer
  serialization, or a public cache read path;
- strict nonempty `PredictCollection` construction in the internal results
  module, with no package-namespace export; independently copied stored results;
  aligned, independently cloned Polars metadata; independently owned `Predict`
  snapshots from integer indexing and iteration; and derived whole-brain, ROI,
  and searchlight properties;
- binary/regression and class-keyed multiclass weight-map stacking;
- collection order for every derived weight-map or score-map stack;
  independently owned metadata in `.X`, or empty `.X` when metadata is absent;
  empty `.Y`; independent class-specific multiclass maps; and no ad hoc
  `BrainData` metadata attribute;
- `mean_score` and `std_score` columns for whole-brain `score_table`, plus
  `roi_label` for ROI; pre-construction rejection of metadata collisions with
  those mode-specific names; and no overwriting or suffixing;
- thin `predict_group` validation and delegation without dtype downcasting,
  synthetic groups, permutation state, or duplicated MVPA computation;
- single and mapped GLM contrasts, including canonical numeric alignment;
- exact contrast signature and control order; global validation before
  scheduling; one hydration and one all-contrast worker task per member;
  standard cache policy for every payload; atomic completion and publication
  across all members, contrasts, and inferential fields; complete
  `ContrastResult[BrainCollection]` fields and degrees of freedom in member
  order; independently copied metadata; cleared stored designs; and empty member
  `.X` and `.Y`;
- clear failures for irrelevant estimator options and all removed aliases;
- exact `map()` signature; independent callback inputs and stored outputs;
  `BrainData` return-type and exact-mask validation with member-context errors;
  dedicated mask-changing execution; and no public `apply()` or string alias;
- exact design-transformation signature with no execution or cache controls;
  synchronous coordinator execution without cache work; callbacks only for
  non-`None` entries; preserved `None` entries; independent inputs and outputs;
  source immutability; and type and row-count errors with member context;
- exact persistence and export signatures; lossless unfitted collection
  round-trips; lazy and eager reads; independent hydration; mask-basename
  handling; strict manifest and schema rejection; lossy image export with fixed
  ordinal filenames; atomic staging and publication; and refusal to overwrite,
  read caches, infer archives from image directories, or write or export an
  empty collection;
- round-trip cache hydration of the complete fitted-member state.
