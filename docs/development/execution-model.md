---
title: Execution model (BrainCollection)
description: How BrainCollection runs per-subject work in parallel — caching, HDF5 bundles, pickling, and write safety.
---

# Execution model — `BrainCollection`

`BrainCollection` saves users from writing for-loops over `BrainData`. It is a
parallel, lazy iterator of `BrainData` whose API mirrors `BrainData`, with first-class
`(BrainData, DesignMatrix)` pairing. This page documents the execution machinery that
makes it memory-efficient. It is design reference, not API documentation — for method
signatures see the [BrainCollection API](../api/data/brain_collection.md).

The machinery lives under `nltools/data/collection/`:

| File | Role |
|---|---|
| `__init__.py` | `BrainCollection` class — facade only |
| `core.py` | metadata coercion, mask resolution, run/step ID helpers |
| `io.py` | constructors (BIDS/glob/paths), `write`, `read`, `load`/`unload`, cache plumbing, `memory_estimate` |
| `execution.py` | parallel `_apply`, materialization, cache/step bookkeeping, HDF5 bundle IO |
| `inference.py` | group reductions (`ttest`, `anova`, `permutation_test`), `isc`, `align` |
| `pipeline.py` | `BrainCollectionPipeline`, used by `cv()` (backed by `pipesteps/`) |

## Path-backed by default

After any parallel op (`fit`, `compute_contrasts`, `smooth`, `standardize`, `detrend`,
`threshold`, `resample`, `predict`, `map`, `apply`), the returned collection is
**path-backed**: workers write each per-subject result to disk and the parent never
accumulates per-subject `BrainData` in RAM. Peak memory stays at roughly
`n_workers × 1 subject`.

Reductions (`concat`, `mean`, `std`, `ttest`, `anova`, `isc(method='loo')`, …) stream
from path-backed inputs and produce a small in-memory `BrainData` (or dict of them). They
never path-back their own output — by construction the result is small. (`concat` and
`median` are the exceptions that must materialize; so does `isc(method='pairwise')` — see
the ISC note below.)

> **Note.** `transform_designs` is *not* a parallel/path-backed op. It maps a function
> over the `DesignMatrix` list synchronously in the parent process and returns an
> in-memory clone; it does not honor `n_jobs`/`progress_bar`/`cache` and writes nothing
> to disk.

## The `cache=` knob

Every collection-returning op accepts `cache: Literal['auto', True, False] = 'auto'`:

| Value | Behavior |
|---|---|
| `'auto'` (default) | Follow source state. All items loaded → return in-memory; any item path-backed → write through. |
| `True` | Force disk write regardless of source state. |
| `False` | Force in-memory result. Loads any path-backed items first. |

The "stay warm" idiom is `bc.load().smooth().standardize().detrend()` — `'auto'` keeps
each step in memory because the source is loaded. `from_bids` returns lazy/path-backed
by default, so the typical "construct → fit" path writes through. `cache=False` is the
escape hatch when you've already paid the load+resample cost and don't want a disk
round-trip.

**Coverage rule:** methods that return a `BrainCollection` accept `cache=`. Methods that
return `BrainData` / `dict` / `tuple` / scalar do not — their output is small by
construction and always lives in memory.

## Cache location

Cache root: `./.nltools_cache/{run_id}/`, where `run_id = "{timestamp}_{token}"` — a
UTC timestamp plus 8 hex characters (`secrets.token_hex(4)`). Visible in `cwd` (not
user-home), so it's discoverable and disposable with `rm -rf`. Each `BrainCollection`
instance owns one cache root; lightweight clones inherit it.

Override at construction with `cache_dir=...`. Use `cache_dir=None` for a tempdir that's
auto-cleaned at process exit (ephemeral analysis or tests).

`cache_dir` resolves in this precedence: explicit arg → `NLTOOLS_CACHE_DIR` env var →
`./.nltools_cache`.

**Capture timing.** The cache root is resolved at *construction* and frozen on the
instance. `os.chdir()` after construction does not relocate it. `cleanup_all(directory)`,
in contrast, resolves at *call* time and only finds caches under that directory's
`.nltools_cache/` — so it can miss collections constructed elsewhere. Use `bc.cleanup()`
for surgical removal.

## Lightweight clones

When a parallel op returns a new collection, it's a thin shallow clone. `BrainCollection`
is a **plain class** (not a dataclass); cloning goes through a hand-written
`_clone(**overrides)` that builds a new instance via `__class__.__new__` and copies
references, overriding `items` and appending to the step lineage. `mask`, `metadata`,
`designs`, and `cache_root` are shared **by reference**. Cost is `O(n_subjects)` paths.
`bc` and `bc.smooth()` coexist; both are valid references into the same cache root.

`bc.cleanup()` removes the shared root and invalidates every clone derived from `bc`.
That's the trade for the cheap clone — one analysis lineage, one cleanup call.

`bc.steps()` returns the accumulated lineage (the step dirs this clone descends
through), not a directory scan of the cache root — so it excludes sibling branches.

## Parallel write safety

Three rules cover the failure modes:

1. **One file per item per step.** Workers write to disjoint paths under a step subdir:
   `step_dir/sub-0001.nii.gz`, `sub-0002.nii.gz`, …. HDF5 single-writer-per-file is
   sufficient; we never share a file across workers.
2. **Atomic rename.** A worker writes to a `.tmp_`-prefixed sibling
   (`.tmp_sub-0001.nii.gz`) then `os.rename`s it into place on success. A crashed worker
   leaves no half-file visible to readers.
3. **Step subdirs are unique per call.** Names are
   `{timestamp}_{seq}_{token}_{op}_{key_kwargs}/` — a UTC timestamp, a process-monotonic
   sequence counter (`itertools.count`, zero-padded, for stable lexical tie-break within
   a process), an 8-char hex token (collision-free across concurrent clones, processes,
   and HPC nodes), then the op and its key params. Same op + same params called twice →
   two subdirs, never overwrite. No idempotent caching, no invalidation logic — disk is
   cheap.

## Shared filesystems / HPC

Defaults are tuned for local disks. On NFS / shared filesystems (typical HPC):

- **Cache location.** Set `NLTOOLS_CACHE_DIR=$SLURM_TMPDIR` (or local scratch) once in
  your shell to keep the working set off the network filesystem.
- **HDF5 locking.** Bundle reads/writes use `h5py.File(..., locking=False)`. POSIX file
  locking is unreliable on NFS, and we already serialize per-subject access via the
  worker model (one writer per file).
- **Worker output collection.** Workers return their output paths explicitly through
  joblib; the parent never lists the cache directory to discover results. This sidesteps
  stale NFS dir caches.
- **Atomic rename.** Tmp+rename is safe within a single directory on NFS — that's the
  only guarantee we depend on.
- **Run / step IDs** carry hex tokens, so multi-node jobs sharing a cache root never
  collide.

Two HPC patterns are explicitly supported:

1. **Local scratch + final write to network storage.** Set
   `NLTOOLS_CACHE_DIR=$SLURM_TMPDIR`; at the end, `bc.write("/network/storage/results")`
   materializes a clean copy outside the cache.
2. **Cache directly on network storage** (slower but persistent across job restarts).
   Default `./.nltools_cache` works when cwd is on the network mount.

## Eager, no fused chains

Each step is eager. `bc.smooth().standardize()` produces *two* on-disk steps (smoothed
BOLD + standardized BOLD). Disk is cheap; intermediates are debuggable; there is no
lazy/fused chain machinery.

## File format rule

| Output shape | Format | Used by |
|---|---|---|
| Bundle (multiple arrays + state per subject) | HDF5 (`sub-XXXX_fit.h5`) | `fit(model='glm')`, `fit(model='ridge')` |
| Single image per subject | NIfTI via `BrainData.write()` (`sub-XXXX.nii.gz`) | `smooth`, `standardize`, `detrend`, `threshold`, `resample`, `compute_contrasts`, `predict(X_new=)`, `align`, `map`, `apply` |

## HDF5 fit bundle

`fit(model='glm')` writes per subject:

```text
{step_dir}/sub-XXXX_fit.h5
├── /betas        (n_regressors, n_voxels)
├── /residuals    (n_obs, n_voxels)
├── /sigma2       (n_voxels,)
├── /r2           (n_voxels,)
├── /X            (n_obs, n_regressors)
├── /mask         (embedded NIfTI bytes — bundle is portable)
└── attrs:
    ├── affine, regressor_names, scale, standardize, model_kwargs
    ├── nltools_version, bundle_schema_version
    └── step_id, parent_step_id, op, kwargs (JSON-encoded)
```

Residuals are always included — disk is cheap, and any downstream `compute_contrasts(...)`
can return a contrast (incl. t/z/p) without re-fitting. The mask is embedded as a dataset
(not a path attr) so bundles survive `mv`, `cp`, archiving, and cross-machine transfer.

`compute_contrasts(name)` reads the bundle, computes the contrast map(s), writes
per-subject NIfTI to a new step subdir, and returns a path-backed collection. Each NIfTI
gets a `sub-XXXX.json` sidecar carrying the same lineage attrs. The contrast string
parser supports coefficients (e.g. `"2*A - B"`), not just `"A - B"`.

`fit(model='ridge')` writes a parallel HDF5 bundle holding `weights`, `cv_scores`,
`predictions`, `scores`, and `intercept`, with the same versioning + lineage attrs.
`predict(X_new=)` reads the bundle and writes per-subject prediction NIfTIs
(`X_new @ weights + intercept`, with JSON sidecars).

**On read:** `bundle_schema_version` mismatch raises with a clear migration message
(schema is currently at version 2). `nltools_version` mismatch logs a warning but does
not refuse — bundles are usually forward-compatible within a minor version.

> **Constraint.** `fit(model='glm')` accepts only OLS. A non-OLS noise model (e.g.
> `noise_model="ar1"`) raises `NotImplementedError`, because the serializable
> closed-form contrast path can't represent AR whitening. Use per-subject `BrainData`
> for AR models.

## Pickling

The trap to design against: a closure that references `self` (the `BrainCollection`)
ships every loaded `BrainData` to every worker. Workers instead receive only a small,
frozen task spec — `_ItemTask` — carrying the item (a `BrainData` if loaded, else a
`Path`), its paired design/confounds/sample-mask, the metadata row, the mask path
(workers load the mask once and cache it process-locally), and the output path when
writing through. The collection itself is never pickled.

For per-item design *builders*, the worker constructs a `_DesignContext` (a frozen
dataclass exposing `bd`, `dm`, confounds, sample mask, metadata, and BIDS entities such
as `subject`/`session`/`run`/`task`/`TR`, with a `__getitem__` fallback into metadata)
and passes it to the user's callable lazily.

## Single execution primitive

All parallel ops route through one function, `_apply(bc, fn, *, op, op_kwargs, step_id,
n_jobs, progress_bar, backend, cache, out_ext, require_design)`, which returns
`(results, step_dir, step_id)`. Every per-subject method on `BrainCollection` is a thin
wrapper that:

1. Resolves `cache=` (auto → on if any input is path-backed).
2. Allocates a step subdir under `bc.cache_root` (when caching).
3. Builds `_ItemTask`s from the parallel slots.
4. Calls `_apply`.
5. Wraps the results into a lightweight clone.

This kills per-method `_map_axis*` proliferation and gives one place to fix pickling,
progress bars, error context, and write atomicity. Joint (all-subjects-at-once) ops like
`align` use a sibling caching path, `_persist_or_keep`, rather than `_apply`.

## Progress bars

A `tqdm_joblib` context manager monkeypatches joblib's batch-completion callback so the
bar advances as workers *complete* (not at submit time). One implementation in
`execution.py`; all methods reuse it.

## Errors

`_apply` fails fast on the first worker exception. Survivors' outputs already in the step
subdir stay there for inspection (the partial step subdir is **not** cleaned up). The
user gets a clear error with subject context; `bc.steps()` shows where to look.

Workers wrap their exceptions in a `BrainCollectionWorkerError(RuntimeError)` with the
original chained via `from e` and the subject/run embedded in the message:

```python
class BrainCollectionWorkerError(RuntimeError):
    pass

try:
    return fn(task)
except Exception as e:
    raise BrainCollectionWorkerError(
        f"[subject={task.metadata_row.get('subject')} "
        f"run={task.metadata_row.get('run')}] {e}"
    ) from e
```

Wrapping (rather than `raise type(e)(...)`) because some exception classes' `__init__`
doesn't accept a single string — that pattern silently drops the original traceback.

## Nested parallelism

Risk: `bc.fit(n_jobs=-1)` over 30 subjects × `BrainData.fit` internally calling
`Ridge(n_jobs=-1)` = N² processes. When `_apply` runs with `n_jobs > 1` it sets
`joblib.parallel_backend("loky", inner_max_num_threads=1)` for the inner scope, capping
thread oversubscription inside each worker.

## When *not* to parallelize

- **Reductions** (`concat()`, `mean()`, `ttest()`) stream over already-loaded or
  path-backed data in the main process — faster than the pickling overhead.
- **ISC / alignment** have their own parallel scheme inside `nltools.algorithms.*`. The
  collection passes `n_jobs`/`device` through but doesn't double-wrap.

> **Note.** `isc(method='loo')` streams: pass 1 accumulates the subject sum (one `T×V`
> array), pass 2 re-streams forming each subject's template `(sum − subject)/(n−1)` and
> per-voxel Pearson r, so peak memory is ~2 subjects regardless of N (see
> `_isc_loo_streaming` in `inference.py`). `isc(method='pairwise')` still materializes all
> subjects — it needs every `C(n,2)` pair, and streaming would cost `O(n²)` subject
> reloads. `isc_test` (bootstrap) also materializes: it needs random subject access across
> draws, so streaming would mean `n_samples × n_subj` reloads.

## GPU

`device="cpu" | "gpu"` is orthogonal to `n_jobs`. `n_jobs` controls subject-level
parallelism; `device` controls the within-subject backend. ISC, permutation tests, and
alignment accept both — pass-through.
