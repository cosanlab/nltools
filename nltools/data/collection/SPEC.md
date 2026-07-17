# BrainCollection — v0.6.0 redesign spec

> Status: implementation substantially complete on branch `collection-impl` (worktree `nltools-collection-impl`). Modules live under `nltools/data/collection/` ({core,execution,inference,io,pipeline}.py + `__init__.py` facade) with tests under `nltools/tests/data/collection/`. This spec remains the source of truth for intended design — keep it in sync as the code evolves. (Collection tests are still skipped on `uv-cleanup` per commit 4ad10050.)

## Goals (in priority order)

1. **Save users from writing for-loops over `BrainData`.** A `BrainCollection` is a parallel/lazy iterator of `BrainData` whose API mirrors `BrainData`.
2. **First-class `(BrainData, DesignMatrix)` pairing**, populated automatically from BIDS or explicitly from matched lists. Operations that need a per-subject design matrix (`fit`, `compute_contrasts`) consume the paired `DesignMatrix` by default.
3. **Memory-efficient by default.** Parallel ops write per-subject results to a visible disk cache and return path-backed collections. Peak RAM stays at roughly `n_workers × 1 subject`. Workflows that want to stay in memory opt in via `bc.load()` or `cache=False`.
4. **Concatenate first-level results without picking a contrast inside the loop.** `bc.fit(model='glm')` returns a `BrainCollection` of per-subject HDF5 fit bundles; `.compute_contrasts(...)` then runs across the collection.
5. **Streaming pipelines (ISC, group classifiers, alignment) are preserved** via `cv()` / `pipeline.py` and the existing extraction machinery in `inference.py`.

## What's gone

- 3D tensor framing: `mean(axis=1|2)`, `to_tensor`, `iter_batches(axis=1)`, multi-dim `bc[i, j, k]`. `mean()`/`std()`/etc. collapse subjects only.
- `fit_glm`, `fit_from_events`, `fit_ridge`, `FittedBrainCollection`. One `.fit()` instead — same surface as `BrainData.fit`.
- `to_stacked` / `from_stacked` (use `concat()` and slicing).
- `to_list` (use `list(bc)`).
- Per-axis private handlers (`_map_axis0/1/2`, `_aggregate_axis0/1/2`) — one parallel primitive (`_apply`) does the work.
- `output=` / `save=` kwargs on `fit()` — disk-backed is the default, residuals are always written into the HDF5 bundle, and downstream collections (e.g. `compute_contrasts`) always have what they need without selecting a stat upfront.
- `checkpoint()` — same reason; writing through is the default behavior, not an opt-in step.

## Module layout

`nltools/data/collection/`

| File | Role |
|---|---|
| `__init__.py` | `BrainCollection` class — facade only |
| `core.py` | metadata coercion, mask resolution, helpers |
| `io.py` | constructors (BIDS/glob/paths), `write`, `read`, `load`/`unload`, cache plumbing, `memory_estimate` |
| `execution.py` | parallel `_apply`, materialization, cache/step bookkeeping, HDF5 bundle IO |
| `inference.py` | group reductions (`ttest`, `anova`, `permutation_test`), `isc`, `align`, internal extraction helpers |
| `pipeline.py` | unchanged — `BrainCollectionPipeline`, used by `cv()` |

12 → 6 modules. `aggregation.py`, `conversions.py`, `constructors.py`, `indexing.py`, `modeling.py`, `prediction.py`, `transforms.py` are absorbed.

## Execution model

This shapes the rest of the API; read it before the public surface.

### Path-backed by default

After any parallel op (`fit`, `compute_contrasts`, `smooth`, `standardize`, `detrend`, `threshold`, `resample`, `predict`, `transform_designs`, `map`, `apply`), the returned `BrainCollection` is **path-backed**: workers write each per-subject result to disk and the parent never accumulates per-subject `BrainData` in RAM. Peak memory stays at roughly `n_workers × 1 subject`.

Reductions (`concat`, `mean`, `std`, `ttest`, `anova`, `isc`, ...) stream from path-backed inputs and produce a small in-memory `BrainData` (or dict of them). They never path-back their own output — by construction, the result is small.

### The `cache=` knob

Every parallel op accepts `cache: Literal['auto', True, False] = 'auto'`:

| Value | Behavior |
|---|---|
| `'auto'` (default) | Follow source state. If all items are loaded → return in-memory; if any item is path-backed → write through. |
| `True` | Force disk write regardless of source state. |
| `False` | Force in-memory result. Loads any path-backed items first. |

The "stay warm" idiom is `bc.load().smooth().standardize().detrend()` — `'auto'` keeps each step in memory because the source is loaded. `from_bids` returns lazy/path-backed by default, so the typical "construct → fit" path writes through. `cache=False` is the escape hatch when you've already paid the load+resample cost and don't want a disk round-trip.

**Coverage rule:** methods that return a `BrainCollection` accept `cache=`. Methods that return `BrainData` / `dict` / `tuple` / scalar do not — their output is small by construction and always lives in memory.

### Cache location

Cache root: `./.nltools_cache/{run_id}/`, where `run_id = "{timestamp}_{uuid8}"`. Visible in `cwd` (not user-home), so it's discoverable and disposable with `rm -rf`. Each `BrainCollection` instance owns one cache root; lightweight clones inherit it (see below).

Override at construction with `cache_dir=...`. Use `cache_dir=None` for a tempdir that's auto-cleaned at process exit (ephemeral analysis or tests).

**Capture timing.** The cache root is resolved at *construction* and frozen on the instance. `os.chdir()` after construction does not relocate it. `cleanup_all(directory)`, in contrast, resolves at *call* time and only finds caches under that directory's `.nltools_cache/` — so it can miss collections constructed elsewhere. Use `bc.cleanup()` for surgical removal.

### Lightweight clones

When a parallel op returns a new collection, it's a thin shallow clone:

```python
new_bc = dataclasses.replace(
    bc,
    items=new_paths,        # or new BrainData list, in the in-memory case
    step_id=bc._next_step_id(),
)
```

`mask`, `metadata`, `designs`, and `cache_root` are shared by reference. Cost is `O(n_subjects)` paths. `bc` and `bc.smooth()` coexist; both are valid references into the same cache root.

`bc.cleanup()` removes the shared root and invalidates every clone derived from `bc`. That's the trade for the cheap clone — one analysis lineage, one cleanup call.

### Parallel write safety

Three rules cover the failure modes:

1. **One file per item per step.** Workers write to disjoint paths under a step subdir: `step_dir/sub-01.h5`, `sub-02.h5`, .... HDF5 single-writer-per-file is sufficient; we never share a file across workers.
2. **Atomic rename.** Worker writes to `sub-01.h5.tmp`, then `os.rename` to `sub-01.h5` on success. A crashed worker leaves no half-file visible to readers.
3. **Step subdirs are unique per call.** Names are `{timestamp}_{uuid8}_{op}_{key_kwargs}/` — e.g. `20260501T143022_a3f8c2d1_smooth_fwhm-6.0/`. Lex-sorted directory listings give chronological order; the 8-char UUID tail makes names collision-free across concurrent clones, processes, and HPC nodes (no atomic counter, no lockfile). Same op + same params called twice → two subdirs, never overwrite. No idempotent caching, no invalidation logic — disk is cheap.

### Shared filesystems / HPC

Defaults are tuned for local disks. On NFS / shared filesystems (typical HPC):

- **Cache location.** `cache_dir` resolves: explicit arg → `NLTOOLS_CACHE_DIR` env var → `./.nltools_cache`. Set `NLTOOLS_CACHE_DIR=$SLURM_TMPDIR` (or local scratch) once in your shell to keep the working set off the network filesystem.
- **HDF5 locking.** Bundle reads/writes use `h5py.File(..., locking=False)`. POSIX file locking is unreliable on NFS, and we already serialize per-subject access via the worker model (one writer per file).
- **Worker output collection.** Workers return their output paths explicitly through joblib; the parent never lists the cache directory to discover results. This sidesteps stale NFS dir caches.
- **Atomic rename.** Tmp+rename is safe within a single directory on NFS — that's the only guarantee we depend on.
- **Run / step IDs.** `run_id = {timestamp}_{uuid8}` and step subdirs add their own UUID tail, so multi-node jobs sharing a cache root never collide.

Two HPC patterns we explicitly support:

1. **Local scratch + final write to network storage.** Set `NLTOOLS_CACHE_DIR=$SLURM_TMPDIR`; at the end of the analysis, `bc.write("/network/storage/results")` materializes a clean copy outside the cache.
2. **Cache directly on network storage** (slower but persistent across job restarts). Default `./.nltools_cache` works when cwd is on the network mount.

### Eager, no fused chains

Each step is eager. `bc.smooth().standardize()` produces *two* on-disk steps (smoothed BOLD + standardized BOLD). Disk is cheap; intermediates are debuggable; no lazy/fused chain machinery in v0.6.0.

### File format rule

| Output shape | Format | Used by |
|---|---|---|
| Bundle (multiple arrays + state per subject) | HDF5 (`sub-XX_fit.h5`) | `fit(model='glm')`, `fit(model='ridge')` |
| Single image per subject | NIfTI via `BrainData.write()` (`sub-XX.nii.gz`) | `smooth`, `standardize`, `detrend`, `threshold`, `resample`, `compute_contrasts`, `predict(X_new)`, `align`, `map`, `apply` |

`transform_designs` writes its outputs via `DesignMatrix.write()` — the class chooses its own serialization format.

### HDF5 fit bundle

`fit(model='glm')` writes per subject:

```
{step_dir}/sub-XX_fit.h5
├── /betas        (n_regressors, n_voxels)
├── /residuals    (n_obs, n_voxels)
├── /sigma2       (n_voxels,)
├── /r2           (n_voxels,)
├── /X            (n_obs, n_regressors)
├── /mask         (embedded NIfTI bytes — bundle is portable)
└── attrs:
    ├── affine, regressor_names, scale, scale_value, model_kwargs
    ├── nltools_version, bundle_schema_version (int, starts at 1)
    └── step_id, parent_step_id, op, kwargs (JSON-encoded)
```

Residuals are always included — disk is cheap, and any downstream `compute_contrasts(...)` can return a contrast (incl. t/z/p) without re-fitting. The mask is embedded as a dataset (not a path attr) so bundles survive `mv`, `cp`, archiving, and cross-machine transfer.

`compute_contrasts(name)` reads the bundle, computes the contrast map(s), writes per-subject NIfTI to a new step subdir, returns a path-backed collection. Each NIfTI gets a `sub-XX.json` sidecar carrying the same lineage attrs (`step_id`, `parent_step_id`, `op`, `kwargs`, `nltools_version`).

`fit(model='ridge')` writes a parallel HDF5 bundle holding `weights`, `cv_scores`, `predictions`, `scores`, with the same versioning + lineage attrs. `predict(X_new)` reads the bundle, writes per-subject prediction NIfTIs (with JSON sidecars).

**On read:** `bundle_schema_version` mismatch raises with a clear migration message. `nltools_version` mismatch logs a warning but does not refuse — bundles are usually forward-compatible within a minor version.

### Pickling

The trap to design against: a closure that references `self` (the `BrainCollection`) ships every loaded `BrainData` to every worker. Workers receive only a small task spec:

```python
@dataclass(frozen=True)
class _ItemTask:
    idx: int
    item: BrainData | Path           # path if lazy, BrainData if loaded
    design: DesignMatrix | Path | None
    confounds: pd.DataFrame | None
    sample_mask: np.ndarray | None
    metadata_row: dict
    mask_path: Path                  # workers load mask once and cache
    out_path: Path | None            # set when writing through cache
```

The worker calls `_materialize(task)` to load any paths, runs the op, writes to `out_path` if set (atomic tmp+rename), and returns either a path or a small in-memory result. The collection itself is never pickled.

### Single execution primitive

```python
def _apply(
    bc: BrainCollection,
    fn: Callable[[_ItemTask], T],
    *,
    n_jobs: int = -1,
    progress_bar: bool = False,
    backend: str = "loky",
    cache: Literal['auto', True, False] = 'auto',
    step_name: str,                   # e.g. "smooth_fwhm-6.0"
    require_design: bool = False,
) -> list[T]:
    """All parallel ops route through here."""
```

Every per-subject method on `BrainCollection` is a thin wrapper that:
1. Resolves `cache=` (auto → on if any path-backed input).
2. Allocates a step subdir under `bc.cache_root` (when caching).
3. Builds `_ItemTask`s from the parallel slots.
4. Calls `_apply`.
5. Wraps the results into a lightweight clone.

This kills the per-method `_map_axis0/1/2` proliferation and gives one place to fix pickling, progress bars, error context, and write atomicity.

### Progress bars

Use a `tqdm_joblib` context manager so the bar updates as workers complete. Today's code wraps the iterator before `Parallel`, which advances at *submit* time, not completion. One implementation in `execution.py`, all methods reuse it.

### Errors

`_apply` fails fast on the first worker exception. Survivors' outputs already in the step subdir stay there for inspection (the partial step subdir is **not** cleaned up). The user gets a clear error with subject context; calling `bc.steps()` shows them where to look.

Workers wrap their exceptions in a `BrainCollectionWorkerError` with the original chained via `from e`:

```python
class BrainCollectionWorkerError(RuntimeError):
    pass

try:
    return fn(task)
except Exception as e:
    raise BrainCollectionWorkerError(
        f"[subject={task.metadata_row.get('subject')} run={task.metadata_row.get('run')}] {e}"
    ) from e
```

Wrapping rather than `raise type(e)(...)` because some exception classes' `__init__` doesn't accept a single string — that pattern silently drops the original traceback. The full chain is preserved through `from e`.

### Nested parallelism

Risk: `bc.fit(n_jobs=-1)` over 30 subjects × `BrainData.fit` internally calling `Ridge(n_jobs=-1)` = N² processes.

Default: when `_apply` runs with `n_jobs > 1`, set `joblib.parallel_backend("loky", inner_max_num_threads=1)` for the inner scope and pass `n_jobs=1` to BrainData methods downstream. Document the override (`bc.fit(..., inner_n_jobs=...)`) for users who really want nested parallelism (e.g. small N, expensive per-subject CV).

### When *not* to parallelize

- `concat()`, `mean()`, `ttest()` — reductions over already-loaded or path-backed data. Streaming over items in the main process is faster than the pickling overhead.
- ISC / alignment — they have their own parallel scheme inside `nltools.algorithms.*`. The collection passes `n_jobs` through but doesn't double-wrap.

### GPU

`device="cpu" | "gpu"` is orthogonal to `n_jobs`. `n_jobs` controls subject-level parallelism; `device` controls within-subject backend. ISC, permutation tests, and alignment already accept both — pass-through.

## Public API

### Construction

```python
class BrainCollection:
    def __init__(
        self,
        brains: list[BrainData | Path | str],
        *,
        mask: nib.Nifti1Image | Path | str,
        designs: list[DesignMatrix | Path | str | None] | None = None,
        metadata: pl.DataFrame | pd.DataFrame | dict | None = None,
        lazy: bool = True,
        cache_dir: Path | str | None = "./.nltools_cache",
    ) -> None: ...

    @classmethod
    def from_bids(
        cls,
        root: Path | str | "BIDSLayout",
        *,
        mask: nib.Nifti1Image | Path | str,
        task: str | None = None,
        space: str | None = None,
        sub_labels: list[str] | None = None,
        img_filters: list[tuple[str, str]] | None = None,
        derivatives_folder: str = "derivatives",
        pair_events: bool = True,
        confounds_strategy: str | tuple[str, ...] | None = None,
        confounds_kwargs: dict | None = None,
        TR: float | str = "infer",       # "infer" reads JSON sidecar
        cache_dir: Path | str | None = "./.nltools_cache",
    ) -> "BrainCollection":
        """Auto-pair BOLD with events.tsv (→ DesignMatrix) and confounds.tsv (→ `_confounds` slot).
        Full design and edge cases: see `from_bids — concrete design` below."""

    @classmethod
    def from_glob(
        cls,
        pattern: str,
        *,
        mask: nib.Nifti1Image | Path | str,
        design_pattern: str | None = None,
        pattern_groups: dict[str, int] | str | None = None,
        sort: bool = True,
        cache_dir: Path | str | None = "./.nltools_cache",
    ) -> "BrainCollection": ...

    @classmethod
    def from_paths(
        cls,
        brain_paths: list[Path | str],
        *,
        mask: nib.Nifti1Image | Path | str,
        design_paths: list[Path | str | None] | None = None,
        metadata: pl.DataFrame | pd.DataFrame | dict | None = None,
        cache_dir: Path | str | None = "./.nltools_cache",
    ) -> "BrainCollection": ...

    @classmethod
    def read(
        cls,
        directory: Path | str,
        *,
        mask: nib.Nifti1Image | Path | str,
        cache_dir: Path | str | None = "./.nltools_cache",
    ) -> "BrainCollection":
        """Inverse of write(): read images + metadata.csv from a directory."""
```

`cache_dir` semantics: explicit arg → `NLTOOLS_CACHE_DIR` env var → `./.nltools_cache`. The resolved directory gets a `{run_id}/` subdir (`{timestamp}_{uuid8}/`). Pass `cache_dir=None` to use a tempdir (auto-cleaned at process exit) for ephemeral analysis.

### Properties

```python
    @property
    def n_subjects(self) -> int: ...
    @property
    def n_voxels(self) -> int: ...
    @property
    def mask(self) -> nib.Nifti1Image: ...
    @property
    def metadata(self) -> pl.DataFrame: ...
    @property
    def designs(self) -> list[DesignMatrix | None]: ...
    @property
    def is_loaded(self) -> list[bool]: ...
    @property
    def shape(self) -> tuple[int, int | None, int]: ...
    @property
    def cache_root(self) -> Path: ...

    def memory_estimate(self) -> str: ...
```

### Indexing and iteration

```python
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[BrainData]: ...
    def __getitem__(self, key) -> BrainData | "BrainCollection":
        """
        bc[i]                 → BrainData
        bc[i:j] / bc[list]    → BrainCollection (subset, shares cache root)
        bc[bool_mask]         → BrainCollection (filter; len == n_subjects)
        bc['sub-01']          → BrainData (looked up via metadata['subject'])
        bc[pl.col(...) == ..] → BrainCollection (polars expression on metadata)
        """

    def iter_pairs(self) -> Iterator[tuple[BrainData, DesignMatrix | None]]: ...

    def filter(
        self,
        predicate: Callable[[BrainData], bool] | list | np.ndarray | pl.Series | pd.Series,
    ) -> "BrainCollection": ...
```

### Per-subject ops — mirror `BrainData`, run in parallel

Every method below has the same signature as the corresponding `BrainData` method, plus `n_jobs`, `progress_bar`, and `cache`. Each returns a new `BrainCollection` (lightweight clone).

```python
    def smooth(self, fwhm: float, *, n_jobs=-1, progress_bar=False,
               cache: Literal['auto', True, False] = 'auto') -> "BrainCollection": ...
    def standardize(self, *, axis=0, method="center", n_jobs=-1,
                    progress_bar=False, cache='auto') -> "BrainCollection": ...
    def detrend(self, *, method="linear", n_jobs=-1,
                progress_bar=False, cache='auto') -> "BrainCollection": ...
    def threshold(self, *, lower=None, upper=None, binarize=False, coerce_nan=True,
                  n_jobs=-1, progress_bar=False, cache='auto') -> "BrainCollection": ...
    def resample(self, target, *, interpolation="continuous",
                 n_jobs=-1, progress_bar=False, cache='auto') -> "BrainCollection": ...

    def transform_designs(
        self,
        fn: Callable[[DesignMatrix], DesignMatrix]
            | Callable[["DesignContext"], DesignMatrix],
        *, n_jobs=-1, progress_bar=False, cache='auto',
    ) -> "BrainCollection":
        """Map a function over paired DesignMatrices."""
```

### Fit / contrasts / predict — mirror `BrainData`

`fit()` returns a `BrainCollection` of per-subject HDF5 fit bundles (when caching). No contrast is picked. Downstream ops read from the bundles on demand.

```python
    def fit(
        self,
        model: str = "glm",
        X: DesignMatrix | list | Callable | None = None,
        *,
        scale: bool = True,
        scale_value: float = 100.0,
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal['auto', True, False] = 'auto',
        **model_kwargs,                # forwarded to BrainData.fit
    ) -> "BrainCollection":
        """
        X resolution priority:
          - None         → use self.designs (must be set per subject)
          - DesignMatrix → shared across all subjects
          - list         → per-subject (len == n_subjects)
          - callable     → fn(ctx: DesignContext) → DesignMatrix
        """

    def compute_contrasts(
        self,
        contrasts: str | list[str] | dict[str, np.ndarray],
        *,
        contrast_type: str = "beta",   # "beta" | "t" | "z" | "p" | "se" | "all"
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal['auto', True, False] = 'auto',
    ) -> "BrainCollection | dict[str, BrainCollection]":
        """
        `contrasts` accepts:
          - a regressor name (e.g. "language")  — identity contrast for that regressor
          - a contrast expression (e.g. "language - string")
          - a list of either
          - a dict {name: contrast_vector}

        Returns:
          single contrast + single contrast_type → BrainCollection
          multiple contrasts                     → dict[str, BrainCollection]
          contrast_type="all"                    → dict["beta"|"t"|"z"|"p"|"se", BrainCollection]
        """

    def predict(
        self,
        y: str | list | np.ndarray | None = None,
        *,
        method: str = "ridge",
        cv: int | None = None,
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal['auto', True, False] = 'auto',
        **kwargs,
    ) -> "BrainCollection | BrainData": ...
```

See "predict and cv() — two distinct prediction paths" below for the cross-subject case.

### Group reductions (collection → `BrainData` or dict of `BrainData`)

`mean`, `std`, `ttest`, ... reduce across subjects (the leading axis), the same way `BrainData.ttest()` reduces across its leading axis (observations). Items must share a shape; mismatched shapes raise. Streaming-friendly: when the source is path-backed, workers stream item-by-item; the parent accumulates with Welford. Output is a small in-memory `BrainData`.

The canonical "pick a contrast and group-test it" pattern:

```python
fitted = bc.fit(model="glm", X=...)
fitted.compute_contrasts("language - string", contrast_type="beta").ttest()
# {'mean': BrainData, 't': BrainData, 'z': BrainData, 'p': BrainData}
```

`.concat()` stays available if you want the intermediate `(n_subjects, n_voxels)` `BrainData`.

```python
    def concat(self) -> BrainData:
        """Stack along axis 0 → BrainData of shape (n_total_obs, n_voxels)."""

    def mean(self) -> BrainData: ...
    def std(self) -> BrainData: ...
    def var(self) -> BrainData: ...
    def median(self) -> BrainData: ...
    def sum(self) -> BrainData: ...
    def min(self) -> BrainData: ...
    def max(self) -> BrainData: ...

    def ttest(self, *, popmean: float = 0.0) -> dict[str, BrainData]:
        """Returns {'mean','t','z','p'} — same shape as BrainData.ttest."""

    def ttest2(self, other: "BrainCollection", *, equal_var: bool = True) -> dict[str, BrainData]: ...
    def anova(self, groups: str | list | np.ndarray) -> dict[str, BrainData]: ...

    def permutation_test(
        self,
        *,
        n_permute: int = 5000,
        tail: int = 2,
        device: str = "cpu",
        return_null: bool = False,
        n_jobs: int = -1,
        random_state: int | None = None,
    ) -> dict: ...

    def permutation_test2(self, other: "BrainCollection", *, n_permute: int = 5000, ...) -> dict: ...
```

### Cross-subject ops (inherently multi-subject; not per-subject)

```python
    def isc(
        self,
        *,
        method: str = "loo",
        roi_mask: nib.Nifti1Image | Path | str | None = None,
        radius_mm: float | None = 6.0,
        metric: str = "median",
        device: str = "cpu",
        n_jobs: int = -1,
        progress_bar: bool = False,
    ) -> dict: ...

    def isc_test(
        self,
        *,
        method: str = "loo",
        roi_mask: ... = None,
        radius_mm: float | None = 6.0,
        n_permute: int = 5000,
        permutation_method: str = "bootstrap",
        ...
    ) -> dict: ...

    def align(
        self,
        *,
        method: str = "procrustes",
        scheme: str = "searchlight",
        radius_mm: float = 10.0,
        parcellation: nib.Nifti1Image | None = None,
        n_features: int | None = None,
        n_iter: int = 3,
        device: str = "cpu",
        return_model: bool = False,
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal['auto', True, False] = 'auto',
    ) -> "BrainCollection | tuple[BrainCollection, LocalAlignment]": ...
```

### Cross-validation pipeline

```python
    def cv(
        self,
        *,
        k: int | None = None,
        method: str = "kfold",          # kfold | loso | loro | bootstrap
        split_by: str | None = None,
        groups: np.ndarray | None = None,
        n: int = 1000,
        random_state: int | None = None,
    ) -> "BrainCollectionPipeline":
        """Unchanged from current. Pipeline machinery in pipeline.py is preserved."""
```

### Composition primitives (used internally; also public)

```python
    def map(
        self,
        fn: Callable[[BrainData], BrainData],
        *,
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal['auto', True, False] = 'auto',
    ) -> "BrainCollection":
        """Apply arbitrary function to each BrainData."""

    def apply(
        self,
        method: str,
        *args,
        n_jobs: int = -1,
        progress_bar: bool = False,
        cache: Literal['auto', True, False] = 'auto',
        **kwargs,
    ) -> "BrainCollection":
        """Call BrainData.<method>(*args, **kwargs) on every item in parallel.
        All per-subject methods reduce to apply()."""
```

### IO / cleanup

```python
    def load(self, indices: list[int] | None = None) -> "BrainCollection":
        """Materialize path-backed items into BrainData. Mutates self in place,
        returns self for chaining (`bc.load().smooth()`)."""

    def unload(self, indices: list[int] | None = None) -> "BrainCollection":
        """Drop in-memory data for items that have backing paths. Mutates self
        in place, returns self for chaining."""

    def steps(self) -> list[Path]:
        """List step subdirs under cache_root, oldest to newest (lex-sorted by
        timestamp prefix). Concrete debugging affordance for inspecting what
        produced this collection's items."""

    def write(
        self,
        directory: Path | str,
        *,
        pattern: str = "image_{i:04d}.nii.gz",
        metadata_file: str | None = "metadata.csv",
    ) -> list[Path]:
        """Write a clean, portable copy outside the cache root (for archiving
        or sharing). Inverse of BrainCollection.read()."""

    def cleanup(self) -> None:
        """Remove the cache root for this collection and all its lightweight clones."""

    @classmethod
    def cleanup_all(cls, directory: Path | str = ".") -> None:
        """Remove every .nltools_cache/{run_id}/ under `directory`. Wide brush —
        this can kill sibling sessions running in the same cwd. Prefer bc.cleanup()
        for surgical removal."""
```

**Mutation semantics.** `load` / `unload` are the *only* methods that mutate `self`. They're cache-management hints, not transformations — they don't allocate a step subdir, don't write to disk, and don't produce a new identity. Every other parallel op returns a lightweight clone. `BrainCollection` is therefore a non-frozen dataclass at the top level (`_items` is a mutable list); `dataclasses.replace` still works for the clone pattern.

**`__del__` is a no-op.** Cache cleanup is always explicit. Garbage collection of a `BrainCollection` does not touch disk.

**`read()` does not recover from cache subdirs.** It expects the `write()` layout (`{dir}/image_{i:04d}.nii.gz` + `metadata.csv`). To make a cache state portable, call `bc.write(...)` first, then `BrainCollection.read(...)` from the written directory.

## Tutorial walkthrough (`docs/tutorials/workflows/01_glm.py`) under the new API

```python
from nltools.data import BrainCollection, DesignMatrix

bc = BrainCollection.from_bids(
    DATASET["data_dir"],
    mask="3mm-MNI152-2009c",
    derivatives=True,
)  # pairs BOLD + events.tsv (→ DesignMatrix) + confounds.tsv (→ metadata); TR from JSON

def make_design(ctx):
    cf = DesignMatrix(ctx.confounds, run_length="infer", TR=ctx.TR)
    return ctx.dm.append(cf, axis=1, as_confounds=True).add_dct_basis(128).add_poly(2).convolve()

result = (
    bc.fit(model="glm", X=make_design, n_jobs=-1, progress_bar=True)  # path-backed: HDF5 fit bundle per subject
      .compute_contrasts("language - string", contrast_type="beta")    # path-backed: contrast nifti per subject
      .ttest()                                                         # streams from disk → small dict[str, BrainData]
)
result["z"].plot(threshold=3.09)
```

Peak RAM through the whole chain: roughly `n_workers × 1 subject`. Intermediates land under `./.nltools_cache/{run_id}/` for inspection or re-use; `bc.cleanup()` removes them.

To test a single regressor's beta (identity contrast), pass its name:

```python
fitted.compute_contrasts("language", contrast_type="beta").ttest()
```

To stay warm in memory (e.g. when load+resample is the bottleneck and the dataset fits):

```python
bc.load().smooth(fwhm=6).standardize().fit(model="glm", X=make_design, n_jobs=-1)
# Each step's items are BrainData in RAM, since cache='auto' follows the loaded source.
```

## `from_bids` — concrete design

Lean on `nilearn` (which already wraps `pybids`) for discovery; we keep the per-subject paths and events/confounds DataFrames it returns and discard everything model-related.

### What we delegate

| Job | Tool | Notes |
|---|---|---|
| Walk the BIDS dataset | `nilearn.glm.first_level.first_level_from_bids` | Returns `(models, imgs, events, confounds)` aligned per subject. We throw away `models`. |
| Pick fmriprep confounds with a strategy | `nilearn.interfaces.fmriprep.load_confounds` | Optional, gated by `confounds_strategy=`. If `None`, we use the raw confounds DataFrames returned by `first_level_from_bids`. |
| Read TR | JSON sidecar (via pybids `get_metadata`, or directly when only `get_bids_files` is available) | When `TR="infer"`. |
| (Fallback) lightweight path discovery | `nilearn.interfaces.bids.get_bids_files` | Used only if `task=None` and the user just wants BOLD images (resting state, decoding without events). |

### What we own

- Mapping discovered paths/DFs onto our `(BrainData, DesignMatrix, metadata)` triplet.
- Building each per-subject `DesignMatrix` from the events DataFrame using `DesignMatrix(events_df, run_length=..., TR=...)`. We do **not** convolve / add drift / merge confounds — that is the user's `transform_designs` step (matches tutorial flow).
- The `metadata` polars DataFrame.

### Signature

```python
    @classmethod
    def from_bids(
        cls,
        root: Path | str | "BIDSLayout",
        *,
        mask: nib.Nifti1Image | Path | str,
        task: str | None = None,
        space: str | None = None,
        sub_labels: list[str] | None = None,
        img_filters: list[tuple[str, str]] | None = None,   # forwarded to nilearn
        derivatives_folder: str = "derivatives",
        pair_events: bool = True,
        confounds_strategy: str | tuple[str, ...] | None = None,  # → load_confounds(strategy=...)
        confounds_kwargs: dict | None = None,                     # extra kwargs for load_confounds
        TR: float | str = "infer",                                # "infer" reads JSON sidecar
        cache_dir: Path | str | None = "./.nltools_cache",
    ) -> "BrainCollection":
        """Build a BrainCollection from a BIDS dataset."""
```

### Item granularity

One item per BOLD file. If a subject has multiple runs, they become multiple items, sharing `subject`/`session` in metadata but differing in `run`. Subject-level concat (one item per subject, runs concatenated) is **not** an option on `from_bids` — do it after construction:

```python
bc.filter(pl.col("subject") == "01").concat()    # one subject's runs stacked
```

### Metadata columns populated

| Column | Source | Notes |
|---|---|---|
| `subject` | BIDS entity | Always present |
| `session` | BIDS entity | Optional |
| `run` | BIDS entity | Optional |
| `task` | BIDS entity | Equals the `task=` arg |
| `space` | BIDS entity | When discovered in derivatives |
| `bold_path` | discovered path | str |
| `events_path` | discovered path | None if no events.tsv |
| `confounds_path` | discovered path | None if no confounds |
| `TR` | JSON sidecar or arg | float seconds |

The confounds DataFrame itself lives in `bc._confounds` (parallel slot), reachable via `ctx.confounds` inside design builders. `metadata` holds polars-friendly types only.

`designs` (parallel list, not a metadata column) holds the per-item `DesignMatrix` (or None when `pair_events=False` or events absent).

### Edge cases / errors

- **Missing events.tsv** for a BOLD when `pair_events=True`: that item's `designs[i] = None`; warn once at the end summarizing how many items lacked events.
- **Missing TR** in JSON sidecar with `TR="infer"`: raise — it's required to build the design matrix.
- **`task=None` and `pair_events=True`**: silently downgrade to `pair_events=False` (no events to pair without a task).
- **fMRIPrep not present** (`derivatives_folder` missing) but `confounds_strategy` set: raise with a clear message.
- **pybids not installed**: raise `ImportError` with `pip install pybids` hint (same as today).

### What we explicitly do NOT do

- We do **not** wrap `nilearn.FirstLevelModel`. Once `first_level_from_bids` returns, we keep only the file paths and DataFrames.
- We do **not** apply `sample_mask` from `load_confounds` automatically. We expose the sample mask via `ctx.sample_mask` so users can opt in to row trimming inside their design builder.

## `fit(X=callable)` — design builder signature

The callable takes one positional argument — a context object the framework constructs per item. The context type is **private** (`_DesignContext`); users access fields by attribute and never need to import or annotate against it. Their builder is just `def make_design(ctx): ...`.

### Storage model (clarifies what `ctx` aggregates)

A `BrainCollection` stores per-item state in four parallel slots, because not everything fits in a polars DataFrame:

| Slot | Type | What's in it |
|---|---|---|
| `bc._items` | `list[BrainData \| Path]` | Lazy/loaded BrainData |
| `bc.designs` | `list[DesignMatrix \| None]` | Paired DesignMatrix (None if events absent or `pair_events=False`) |
| `bc._confounds` | `list[pd.DataFrame \| None]` | Per-item confounds DataFrame (raw or strategy-filtered) |
| `bc._sample_masks` | `list[np.ndarray \| None]` | Set only when `confounds_strategy` includes `'scrub'` |
| `bc.metadata` | `pl.DataFrame` | **Simple-typed columns only** — strings, numbers, paths. Cannot hold DataFrames or arrays. |

The heavy objects (DataFrames, arrays, BrainData) live in parallel lists; the polars `metadata` table holds only what polars can store natively.

### What `ctx` exposes

```python
@dataclass(frozen=True)
class _DesignContext:                    # private — not exported
    bd: BrainData
    dm: DesignMatrix | None
    confounds: pd.DataFrame | None
    sample_mask: np.ndarray | None

    metadata: dict                       # the full row, incl. user-added columns

    # named conveniences for common BIDS columns (read from `metadata`)
    subject: str | None
    session: str | None
    run: int | None
    task: str | None
    TR: float
    bold_path: Path
    events_path: Path | None
    confounds_path: Path | None

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        return self.metadata[key]
```

### Callable contract

```python
DesignBuilder = Callable[[DesignContext], DesignMatrix]
```

- Must return a `DesignMatrix`. The collection passes the result straight to `BrainData.fit(X=...)`.
- Called **lazily, inside the worker**, once per item. Not cached on the collection.
- Pure: same `ctx` should produce the same DM. The collection assumes idempotence for re-runs.
- Errors propagate; the worker fails with the offending `ctx.subject` / `ctx.run` in the message.

### Tutorial pattern

```python
def make_design(ctx):
    cf = DesignMatrix(ctx.confounds, run_length="infer", TR=ctx.TR)
    return (
        ctx.dm
            .append(cf, axis=1, as_confounds=True)
            .add_dct_basis(duration=128)
            .add_poly(order=2, include_lower=True)
            .convolve()
    )

bc.fit(model="glm", X=make_design, n_jobs=-1)
```

### Two related shapes

| Use case | Method | Callable signature | When to use |
|---|---|---|---|
| Build/modify per-subject DM with full subject context | `fit(X=fn)` | `fn(ctx: DesignContext) -> DesignMatrix` | Need `ctx.confounds`, `ctx.TR`, BIDS entities, etc. |
| Apply a uniform pure transform to every paired DM | `transform_designs(fn)` | `fn(dm: DesignMatrix) -> DesignMatrix` | Same operation on every DM, no per-subject info needed |

```python
bc = bc.transform_designs(lambda dm: dm.add_poly(2).convolve())
fitted = bc.fit(model="glm", n_jobs=-1)        # no X — uses self.designs
```

### Other accepted X values

| `X=` value | Resolution |
|---|---|
| `None` | Use `self.designs` (must be set) |
| `DesignMatrix` | Shared across all subjects |
| `list[DesignMatrix \| Path]` | Per-subject (len == n_subjects) |
| `Callable[[DesignContext], DesignMatrix]` | Per-subject builder (above) |

### Why a `DesignContext` dataclass instead of `(bd, dm, meta)` positional args

- **Discoverability.** `ctx.TR` / `ctx.confounds` autocompletes; `meta['confounds']` doesn't.
- **Extensibility.** Adding a new field doesn't break callers — they just don't reference it.
- **Testability.** Users can `DesignContext(bd=..., dm=..., TR=2.0, ...)` directly to unit-test their builder.
- **Stability.** One-arg signature; the alternative locks ordering and forces users to type `_, _, meta` for the common "I just need confounds" case.

## `predict` and `cv()` — two distinct prediction paths

These are *different operations*. Conflating them is the source of much of the current confusion.

| Path | Subjects are... | Train/test boundary | Returns | Mirrors |
|---|---|---|---|---|
| `bc.fit(model='ridge', X=features, cv=5)` then `.predict(...)` | independent fits | within-subject (k-fold over a subject's TRs) | `BrainCollection` of fits / per-subject scores | `BrainData.fit(cv=...)` |
| `bc.cv(method='loso').predict(y)` | samples (one row per subject) | cross-subject (LOSO / LORO / k-fold over the subject axis) | `BrainData` (with CV attrs attached) | The classic group-classifier setup |

### Per-subject path (mirrors BrainData)

```python
fitted = bc.fit(model='ridge', X=features, cv=5, n_jobs=-1)
# Each subject: fits Ridge on (their TRs × voxels) → (their target),
# 5-fold CV WITHIN that subject, exactly like BrainData.fit(cv=5).
# Returns a BrainCollection of HDF5 ridge bundles (path-backed by default).

scores = fitted.map(lambda bd: bd.cv_scores)        # ndarray per subject
new_preds = fitted.predict(X_new, n_jobs=-1)        # per-subject predict, parallel
```

### Cross-subject path (group classification)

Unchanged from current `pipeline.py`. Pipeline is an immutable builder; `predict` is the terminal:

```python
result = (
    bc.cv(method='loso')
      .standardize()                # CV-aware: fit on train fold only
      .reduce(method='pca', n_components=50)
      .predict(y, algorithm='svm')
)
result.mean_score
```

**CV-aware steps vs. plain methods.** `bc.standardize()` (eager, leaks across folds) and `bc.cv(...).standardize()` (refit per fold, no leakage) share a name on purpose — the user's intent is "standardize" in both. The pipeline form is automatically leak-safe; the eager form is for preprocessing where leakage isn't a concern. Pipeline currently calls these `normalize`/`reduce`; we'll rename to `standardize`/`reduce` so they read identically across the two paths.

### `bc.predict` — mirrors `BrainData.predict`, samples are subjects

`BrainData.predict(y=labels)` does MVPA across the rows of `bd.data` (TRs as samples) and returns a `BrainData` with CV attributes attached. `BrainCollection.predict` is the same operation with **subjects as samples**.

```python
def predict(
    self,
    y: np.ndarray | str | None = None,        # str = metadata column name
    *,
    X_new: np.ndarray | None = None,          # for per-subject predict-after-fit
    method: str = "whole_brain",              # 'whole_brain' | 'searchlight' | 'roi'
    estimator: str = "svm",                   # classifier; same names as BrainData.predict
    cv: int | str = "loso",                   # int = k-fold over subjects; 'loso' | 'loro'
    groups: str | np.ndarray | None = None,   # str = metadata column
    roi_mask: ... = None,
    radius_mm: float = 10.0,
    scoring: str = "accuracy",
    standardize: bool = True,
    return_weights: bool = True,              # also fit on full data → weight map
    n_jobs: int = -1,
    progress_bar: bool = False,
    cache: Literal['auto', True, False] = 'auto',
) -> BrainData | BrainCollection: ...
```

Dispatch is by **argument**, not by item state:

| Args | What happens | Returns |
|---|---|---|
| `y=` only | Group MVPA across subjects with subject-level CV. Optional full-data refit for the weight map. | A single `BrainData` (weight map for `whole_brain`, accuracy map for `searchlight`/`roi`) with `.cv_scores`, `.cv_predictions`, `.fold_results` attached. |
| `X_new=` only | Per-subject `predict(X_new)` in parallel; reads each item's fit state from its HDF5 bundle. | `BrainCollection` of per-subject predictions (path-backed by default). |
| both / neither | Raise. They're different operations. | — |

`predict(y=...)` requires each item to already be a single-map-per-subject (e.g. a contrast map from `compute_contrasts`). Items that are GLM/ridge bundles or multi-row BrainData raise with: "call `compute_contrasts(...)` first to get one map per subject."

**No silent recomputation.** Workers load only the per-subject map needed for their fold; MVPA over a path-backed collection of N subjects loads N maps once, never the upstream BOLD or full GLM bundles. Calling `predict(y)` twice on the same collection re-runs the MVPA — consistent with our "no idempotent caching" rule for steps. Hold onto the `BrainData` result if you want to reuse it.

### The canonical workflow (per-subject GLM → LOSO on betas)

```python
bc = BrainCollection.from_bids(root, mask=..., task='localizer')

betas = (
    bc.fit(model='glm', X=make_design, n_jobs=-1)
      .compute_contrasts('faces - houses', contrast_type='beta')
)

result = betas.predict(
    y='group',                      # metadata column with per-subject labels
    estimator='svm',
    cv='loso',
    n_jobs=-1,
)

result                  # → BrainData: SVM weights from a fit on ALL subjects
result.cv_scores        # → ndarray (n_folds,) of LOSO accuracies
result.cv_predictions   # → ndarray of held-out predictions
result.mean_score       # → float
result.plot(...)        # plot the weight map directly — it IS a BrainData
```

The full-data refit is what makes `result` a usable `BrainData` for visualization/inference. CV gives you generalization estimates; the full-data fit gives you the map you actually plot. `return_weights=False` skips the refit if you only want scores.

### With preprocessing — fluent form returns the same type

```python
result = (
    betas.cv(method='loso')
         .standardize()
         .reduce(method='pca', n_components=50)
         .predict(y='group', estimator='svm')
)
# result is a BrainData with the same .cv_scores / .cv_predictions etc. attached.
```

`bc.cv(...).predict(...)` returns the same `BrainData` (with CV attrs) that direct `bc.predict(...)` returns. One result type, one mental model. `BrainCollectionCVResult` goes away.

## Streaming algorithms

Some cross-subject ops can stream from path-backed inputs; some still need everything materialized. Honest accounting:

| Op | Streaming-friendly? | Peak RAM (path-backed input) | Notes |
|---|---|---|---|
| `mean`, `std`, `var`, `sum`, `min`, `max` | yes — Welford one-pass | 1 subject | already implemented; we keep it |
| `ttest`, `ttest2`, `anova` | yes — same Welford pattern | 1 subject | rewrite from current eager impl |
| `permutation_test` | partial | depends on backend | sign-flipping needs all subjects in RAM by design (or memmap) |
| `concat()` | no — output is the full stack | full | the operation *is* materialization |
| `predict(y, cv='loso')` | partial — folds load lazily | training-fold size in RAM | one fold's training subjects in worker |
| `isc(method='loo')` | yes (rewrite) | 1–2 subjects per worker | LOO streamable: total = sum, others = total − this |
| `isc(method='pairwise')` | partial | 2 subjects per worker | pairs only need 2 at a time |
| `align()` (`LocalAlignment`) | no | full | nltools algo takes `list[ndarray]`; v0.6 keeps that |
| `srm()` | no | full | iterative; same constraint |

**Bottom line:** ISC will stream after rewrite; SRM and alignment will not in v0.6.0. They need either a per-subject checkpoint protocol or a minibatch reformulation, both of which are out of scope. We document the cost on those methods, with a one-liner mitigation:

> Memory note: this op materializes all subjects in RAM. For datasets that don't fit, run on a subset (`bc.filter(...)`).

### Streaming ISC contract (the rewrite)

`bc.isc(method='loo', ...)` will be rewritten to stream:

1. **Pass 1**: walk subjects once, accumulate sum-of-timeseries `S` (Welford-style) and sum-of-squares per voxel/feature.
2. **Pass 2**: walk subjects again. For each subject `i` with timeseries `Y_i`, compute `(S − Y_i) / (n − 1)` as the leave-one-out template, correlate against `Y_i`, accumulate per-voxel correlations.
3. Aggregate per the chosen `metric` (`median` or Fisher-z mean).

Peak RAM per worker: 1 subject's timeseries + the running `S` (1 subject's worth). Workers handle disjoint subject ranges in pass 2.

Pairwise ISC is naturally pair-streamable: each pair holds 2 subjects in RAM. Implementation iterates pairs inside `_apply` workers.

The current `extract_for_isc` / `extract_*` / `project_to_brain` private helpers stay but get reused only for the **non-timeseries** path (already-extracted ROI features). The voxelwise/searchlight path goes through the new streaming algorithms in `nltools.algorithms.inference.isc`.

### Why not pass `BrainCollection` straight into the algorithms?

We considered changing `LocalAlignment.fit(data=...)` and `SRM.fit(X=...)` to accept a `BrainCollection`. Decision: not now.

- Those classes are scikit-learn–style estimators and live in `nltools.algorithms` — they are independently usable without our facades. Forcing a `BrainCollection` dependency couples them to data-class internals.
- The collection facade is the right place to handle the memory plumbing. Algorithms consume `list[ndarray]`; the facade decides when to materialize and when to stream.
- The streaming refactor for ISC happens *inside* `nltools.algorithms.inference.isc` — the algorithm gets a `subject_iter` callable instead of a `data` list. That extension is local and doesn't touch unrelated code paths.

### Lifecycle summary (what's in RAM, when)

```text
from_bids / from_paths / read       →  paths only             (cold)
__getitem__(i) / iter                →  load that item          (warm for that index)
fit / map / smooth / ... (cache=auto, source path-backed)
                                     →  worker loads, writes HDF5/nifti, drops item
                                        parent gets paths      (cold result)
fit / map / ... (cache=auto, source loaded)
                                     →  worker computes, returns BrainData
                                        parent keeps in memory (warm result)
.concat / .mean / .ttest             →  reduces to one BrainData; original collection unchanged
.cleanup()                           →  rm -rf the cache root, invalidates clones
```

Invariant: **anything constructed from paths returns to paths after a parallel op (with `cache='auto'`)**. Anything constructed from in-memory `BrainData` stays in memory unless explicitly pushed to disk via `cache=True`.

## Resolved design decisions

1. `__iter__` yields `BrainData`; pairs via `iter_pairs()`.
2. `compute_contrasts` with multiple contrasts → `dict[str, BrainCollection]`. Single regressor names are identity contrasts (matches nilearn / `BrainData`).
3. `fit(X=callable)` is stateless — re-running re-materializes DMs.
4. `mean()` / `ttest()` / etc. collapse subjects only; no `axis` arg. Selecting a contrast for group testing goes through `compute_contrasts(...).ttest()`.
5. `from_bids` reads `TR` from the JSON sidecar by default.
6. `from_bids(derivatives=True)` by default.
7. `cv()` / `BrainCollectionPipeline` API unchanged.
8. **Path-backed by default after parallel ops.** `cache='auto'` follows source state; `True`/`False` force the choice.
9. **Cache lives in `{cache_dir}/{run_id}/`** where `cache_dir` resolves explicit arg → `NLTOOLS_CACHE_DIR` env → `./.nltools_cache`. `cache_dir=None` for ephemeral tempdir.
10. **Lightweight clones share the cache root.** `bc.cleanup()` removes the root and invalidates every clone derived from `bc`.
11. **HDF5 fit bundle, residuals always included.** `compute_contrasts` reads the bundle on demand; no `output=` / `save=` knobs.
12. **File format rule:** bundles (multi-array model state) → HDF5; single-image-per-subject results → NIfTI via `BrainData.write()`. `transform_designs` outputs go via `DesignMatrix.write()`.
13. **No chain fusion.** Each step is eager and writes intermediates.
14. **Parallel write safety:** one file per item per step, atomic tmp+rename, step subdirs named `{timestamp}_{uuid8}_{op}_{kwargs}/`. No idempotent caching, no overwrites, no cross-clone coordination.
15. **HPC / NFS support is first-class.** `h5py.File(..., locking=False)`, workers return paths via joblib (no dir listing), `NLTOOLS_CACHE_DIR=$SLURM_TMPDIR` is the recommended local-scratch pattern.
16. **`cache=` coverage rule:** collection-returning ops accept `cache=`; reduction-returning ops do not.
17. **`bc.predict` dispatch is by argument.** `y=` → group MVPA; `X_new=` → per-subject predict-after-fit; both/neither → raise. `predict(y=...)` requires single-map-per-subject items.
18. **No automatic cleanup.** `bc.cleanup()` and `BrainCollection.cleanup_all()` are explicit. `__del__` is a no-op.
19. **HDF5 bundles are portable.** Mask is embedded as `/mask` (not a path attr); attrs include `nltools_version`, `bundle_schema_version`, and lineage (`step_id`, `parent_step_id`, `op`, `kwargs`). Schema-version mismatch raises; nltools-version mismatch logs a warning. Non-bundle outputs (NIfTI) carry the same lineage in a `sub-XX.json` sidecar.
20. **Errors fail fast.** `_apply` stops at the first worker exception, leaves the partial step subdir for inspection, raises `BrainCollectionWorkerError` with subject context and the original chained via `from e`.
21. **`load` / `unload` mutate-and-return-self.** They're the only methods that mutate; every other op returns a lightweight clone. `BrainCollection` is non-frozen at the top level.
22. **Cache root captured at construction.** `os.chdir()` doesn't relocate it. `cleanup_all` resolves at call time — use `bc.cleanup()` for surgical removal.
23. **`bc.steps()` lists step subdirs** oldest to newest. Concrete debugging affordance.
24. **`read()` does not recover from cache subdirs** in v0.6.0 — use `bc.write(...)` first.
