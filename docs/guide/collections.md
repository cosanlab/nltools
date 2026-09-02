---
title: Working with many subjects
---

A [`BrainCollection`](../api/data/brain_collection.md) is one `BrainData` per subject, sharing one
mask, optionally paired with a per-subject `DesignMatrix` and one row of `metadata`. Its API
mirrors `BrainData`: call `smooth`, `fit`, `compute_contrasts`, `predict`, `align` on the
collection and each runs per subject, in parallel, returning a new collection.

Two facts explain most of its behavior. Items are lazy: each is a file path until something needs
it, so building a collection over 200 subjects costs nothing. Results are path-backed: after a
parallel step, workers write each subject's output to disk and the parent holds paths, so peak
memory is roughly `n_workers × 1 subject` rather than the whole study. Group reductions
(`mean`, `ttest`, `isc`) stream from those paths and return small in-memory `BrainData`.

Goal | Use | Notes
--- | --- | ---
Build from disk | [`from_bids`](../api/data/brain_collection.md#data-brain-collection-from-bids), [`from_glob`](../api/data/brain_collection.md#data-brain-collection-from-glob), [`from_paths`](../api/data/brain_collection.md#data-brain-collection-from-paths) | `from_bids` pairs events and confounds for you; all take `mask=`
Per-subject GLM | [`fit`](../api/data/brain_collection.md#data-brain-collection-fit)`(model='glm', X=design)` → [`compute_contrasts`](../api/data/brain_collection.md#data-brain-collection-compute-contrasts) | `X=` takes one design, a list, or a callable of the item
Group statistics | [`ttest`](../api/data/brain_collection.md#data-brain-collection-ttest), [`ttest2`](../api/data/brain_collection.md#data-brain-collection-ttest2), [`anova`](../api/data/brain_collection.md#data-brain-collection-anova), [`permutation_test`](../api/data/brain_collection.md#data-brain-collection-permutation-test) | Return dicts of `BrainData`; never cached
Anything else per subject | [`map`](../api/data/brain_collection.md#data-brain-collection-map)`(fn)`, [`apply`](../api/data/brain_collection.md#data-brain-collection-apply)`('method_name', ...)` | `apply` calls a `BrainData` method by name; `map` takes a function
Decode within each subject | [`predict`](../api/data/brain_collection.md#data-brain-collection-predict) | Returns a `PredictCollection`, one result per subject
Decode across subjects | [`predict_group`](../api/data/brain_collection.md#data-brain-collection-predict-group) | Pools every image; `cv='logo'` by default so folds respect subjects
Whole-brain ISC | [`isc`](../api/data/brain_collection.md#data-brain-collection-isc), [`isc_test`](../api/data/brain_collection.md#data-brain-collection-isc-test) | See [Intersubject correlation](intersubject.md)
Functional alignment | [`align`](../api/data/brain_collection.md#data-brain-collection-align) | `return_model=True` keeps the transforms
Subset | `bc[i]`, `bc[i:j]`, `bc[bool_mask]`, `bc['sub-01']`, [`filter`](../api/data/brain_collection.md#data-brain-collection-filter) | A single index gives `BrainData`; anything plural gives a collection
Persist | [`write`](../api/data/brain_collection.md#data-brain-collection-write) / [`read`](../api/data/brain_collection.md#data-brain-collection-read) | `write` also saves `metadata.csv`; `read` needs the same `mask=`

## The chain

```python
from nltools.data import BrainCollection

bc = BrainCollection.from_glob("sub-*_bold.nii.gz", mask=mask)
bc.n_subjects, bc.memory_estimate()

fitted = bc.fit(model="glm", X=design, n_jobs=2)
contrasts = fitted.compute_contrasts("face - house", statistic="beta", n_jobs=2)
group = contrasts.ttest()          # {'mean', 't', 'z', 'p'}
contrasts.steps()                  # the cache directories this chain wrote
```

`steps()` is the audit trail: one directory per parallel step, named with a timestamp and the
operation, with one file per subject. `write(directory)` exports a collection to a normal folder of
NIfTIs plus `metadata.csv`; `read` brings it back.

## Caching

Every collection-returning method takes `cache=`:

Value | Behavior
--- | ---
`'auto'` (default) | Follow the source. All items loaded → keep the result in memory; any item path-backed → write through to disk.
`True` | Always write to disk, even from a loaded source.
`False` | Always keep in memory; loads any path-backed items first.

```python
warm = bc.load().smooth(fwhm=6, n_jobs=2)      # 'auto' + loaded source → in memory, no step dir
cached = bc.smooth(fwhm=6, cache=True, n_jobs=2)
```

The cache root is `./.nltools_cache/{run_id}/`, resolved once at construction (`cache_dir=` or the
`NLTOOLS_CACHE_DIR` env var override it; `cache_dir=None` uses a temp dir removed at exit).
`bc.cleanup()` deletes the whole root, including for every collection derived from `bc`, since
clones share it. Methods that return `BrainData`, a dict, or a scalar take no `cache=`: their output
is small by construction.

## Arbitrary work

```python
zscored = bc.apply("standardize", axis=0, method="zscore", n_jobs=2)
halved = bc.map(lambda brain: brain[:50], n_jobs=2)
```

`apply` is for `BrainData` methods you would otherwise call in a loop; `map` is for anything else.
Both pickle the function to workers, so a `map` function must be importable or a simple lambda over
picklable state.

## Gotchas

- `n_jobs` is subject-level parallelism; `device` is the within-subject backend. They are
  independent, and inner parallelism is capped automatically so `n_jobs=-1` over 30 subjects does
  not spawn N² processes.
- A worker failure raises [`BrainCollectionWorkerError`](../api/data/collection_execution.md#data-collection-execution-braincollectionworkererror)
  with the subject and run in the message and the original exception chained. Survivors' outputs
  stay on disk for inspection; `steps()` tells you where.
- `transform_designs` is not parallel and writes nothing; it maps over the design list in the
  parent process.
- Reductions are not faster with more workers. `concat`, `mean`, and `ttest` stream in the main
  process because pickling would cost more than the arithmetic.

The full design is in [the execution model notes](../development/execution-model.md). Next: the
[BrainCollection tutorial](../tutorials/basics/04_brain_collection.md).
