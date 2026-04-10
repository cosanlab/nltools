# nltools pandas → polars migration: next steps

Working document for the v0.6.0 polars-first refactor. Captures the
goal, what's already landed, what remains, and the order-of-attack for
the outstanding chunks.

## Guiding principle

> **Internal code is polars or numpy. Pandas shows up only at the strict
> pandas ⇄ polars boundary: accept pandas inputs at public entry points
> for user convenience, convert to polars/numpy on ingress, convert back
> to pandas at the exact call site where a dependency requires it
> (seaborn, nilearn), and never store pandas as state.**

Corollaries:

- `pandas` is not in `pyproject.toml` — it's installed transitively via
  `nilearn>=0.12` and `seaborn>=0.13.2`, both of which hard-require
  `pandas>=2.2` / `pandas>=1.2`. We do **not** add pandas to our own
  deps because that would be theater — it's already guaranteed present.
- `polars.DataFrame.to_pandas()` requires `pyarrow` and we do **not**
  depend on pyarrow. The canonical conversion helper is
  `pd.DataFrame(pl_df.to_dict(as_series=False))`. Use this pattern
  anywhere a seaborn or nilearn call needs pandas.
- Accepting pandas input is a **convenience affordance**, not a typing
  contract. Functions should accept `pl.DataFrame | pd.DataFrame |
  np.ndarray | list` where sensible, and always return polars (or
  numpy, if the thing is fundamentally a numeric matrix — e.g. distance
  matrices).
- v0.6.0 is a breaking release, so changing public return types from
  pandas to polars is allowed. Document it in CHANGELOG when ready.

## Done

| Commit    | Scope                                                               |
| --------- | ------------------------------------------------------------------- |
| `bbbfea6` | Lazy-import pandas across ~20 modules — no top-level `import pandas` in `nltools/` source |
| `ef367fe` | Rewrite broken `plotting/adjacency.py` (mean/between label distance + silhouette) as polars-native with numpy math and pandas-only-at-seaborn |
| `3e2a4f3` | `stats.outliers.zscore` accepts pandas/polars, returns polars; `stats.intersubject.isc/isc_group/isps` accept polars DataFrames via a local `_as_ndarray` helper |
| `2ec2eaa` | Scattered small modules: `mask.roi_to_brain`, `cross_validation.KFoldStratified`, `datasets.fetch_neurovault_collection`, `io.file_reader.onsets_to_dm`, `data.roc.Roc`, `data.fitresults` docstring |

## State of play — where pandas remains

All `pd.` usages in `nltools/` source (excluding tests) fall into three
categories:

1. **Legitimate boundary** — lazy imports that build pandas objects
   for seaborn/nilearn at the exact call site, or isinstance checks
   for ingress. Leave these alone. Examples:
   - `designmatrix/__init__.py` — pandas ingress (constructor)
   - `designmatrix/io.py` — `to_pandas()` egress + seaborn call
   - `plotting/adjacency.py` — `_polars_to_pandas` helper feeding sns.violinplot
   - `mask.py::roi_to_brain` — optional pandas acceptance
   - `datasets.py` line 349 — events_df passed to nilearn
   - `io/file_reader.py` — polars → pandas conversion before nilearn
   - `stats/timeseries.py`, `stats/outliers.py`, `stats/intersubject.py` —
     pandas acceptance at boundary
   - `models/glm.py` — docstring only
2. **Internal state coupled to BrainData** — pandas because the
   enclosing class stores pandas. Must migrate together with BrainData.
3. **Internal state coupled to Adjacency / Collection** — same story
   for those classes.

## Remaining chunks, in order

The three big chunks are ordered by dependency: **BrainData first**,
because Collection depends on BrainData and several small modules
(`io/h5.py`, `data/simulator`, `braindata/validation.py::validate_frame`)
can only be finished once BrainData's `.X`/`.Y` contract changes.

Each chunk should be landed as multiple small commits, TDD style, one
sub-piece at a time. Do not attempt to rewrite a whole class in a
single pass.

---

### Chunk A — BrainData

**Goal**: `BrainData.X`, `BrainData.Y`, `BrainData.design_matrix` are
polars DataFrames. All internal operations on them use polars. HDF5
persistence is rewritten to use `h5py` directly (not `pd.HDFStore`).

**Files**

Core:
- `nltools/data/braindata/__init__.py` — class facade; `__getitem__`
  slicing of X/Y, `append`, `find_spikes`, `reset_index`
- `nltools/data/braindata/validation.py` — `validate_frame` (currently
  returns pandas, reads CSV via `pd.read_csv`)
- `nltools/data/braindata/utils.py` — `apply_func` assigns empty
  `pd.DataFrame()` to `.X`/`.Y`; `shallow_copy` deep-copies them
- `nltools/data/braindata/analysis.py` — `find_spikes`,
  `regions_of_interest` and various helpers build pandas frames
- `nltools/data/braindata/modeling.py` — GLM accepts/emits pandas
  design matrices; `regress` calls
- `nltools/data/braindata/prediction.py` — CV splits, predict outputs

Persistence:
- `nltools/io/h5.py` — `save_h5`/`load_brain_data_h5` use `pd.HDFStore`
  to persist X/Y. Replace with `h5py` groups + column-name attrs.
- `nltools/data/braindata/__init__.py::write_h5 / load_h5` — entry points.

Simulator (creates BrainData.Y):
- `nltools/data/simulator/__init__.py` — `create_data` and
  `create_cov_data` assign `self.y = pd.DataFrame(...)` then
  `dat.Y = self.y`; also writes `.to_csv`.

Tests:
- `nltools/tests/data/braindata/test_braindata_core.py` — X/Y shape,
  slicing, append assertions
- `nltools/tests/data/braindata/test_braindata_analysis.py` —
  find_spikes return type, regions_of_interest
- `nltools/tests/data/braindata/test_braindata_modeling.py` — GLM
  design matrix handling
- `nltools/tests/data/braindata/test_braindata_prediction.py` —
  X/Y inputs, prediction outputs
- `nltools/tests/data/braindata/test_braindata_io.py` — h5 round-trips
- Several integration tests (`test_haxby_validation.py`) touch X/Y.

**Suggested sub-chunks**

1. **`validate_frame` returns polars** — rewrite CSV read via
   `pl.read_csv` (or numpy if no header), reject unknown types, return
   `pl.DataFrame`. Update validation tests first (red).
2. **`utils.apply_func` uses `pl.DataFrame()` for empty X/Y** — tiny
   follow-on; fails existing tests if X/Y are still pandas elsewhere,
   so do this only after BrainData's ctor accepts polars X/Y.
3. **Constructor accepts polars X/Y** — `BrainData(data, X=..., Y=...)`
   accepts pandas or polars; stores as polars internally. Update
   `validate_frame` to return polars, update everywhere `.X`/`.Y` is
   read/written to use polars APIs (`.is_empty`, `.with_row_index`,
   slicing). This is the **big commit** — everything downstream rides
   on it.
4. **Slicing / `__getitem__`** — rewrite the `isinstance(new.Y,
   pd.Series)` branch; polars row-select via `.slice()` or `.filter()`.
5. **`append`** — replace `pd.concat` with `pl.concat` (schema
   compatibility check). Handle the `ignore_attrs` branch.
6. **`find_spikes`** — returns `pl.DataFrame` (already does via
   `stats.outliers.find_spikes`) but the BrainData method wraps it.
   Ensure the wrapper doesn't re-wrap in pandas.
7. **`regions_of_interest` and analysis helpers** — audit each for
   `pd.DataFrame` constructions; convert to polars.
8. **`modeling.GLM` integration** — nilearn's GLM wants pandas design
   matrices. Convert `DesignMatrix` → pandas at the exact nilearn
   call, accept polars/DesignMatrix as input. Return results as
   polars or numpy.
9. **`prediction`** — cross-val inputs (`.Y`), output metrics frames.
10. **HDF5 rewrite** — replace `pd.HDFStore` with an `h5py` layout:
    ```
    /data        (ndarray)
    /mask_*      (ndarray)
    /X/values    (ndarray)
    /X/columns   (bytes dataset)
    /Y/values    (ndarray)
    /Y/columns   (bytes dataset)
    ```
    Update `save_h5` / `load_brain_data_h5`. Add a legacy-format
    fallback reader for v0.5 files if we care about backward-compat
    (probably not for v0.6).
11. **Simulator** — `self.y = pl.DataFrame(...)`, use `pl.write_csv`,
    update `dat.Y = self.y` assignment.
12. **Test updates** — rewrite every `assert isinstance(bd.Y,
    pd.DataFrame)` to polars. Update `pd.read_csv` fixture loads in
    tests to polars where sensible (leave alone where they just feed
    BrainData and rely on the ingress conversion).

**Test strategy**

- Red-green per sub-chunk. Do **not** run the full suite each
  iteration; run the specific test file or `-k` expression for the
  method under test.
- After (3) lands, expect broad test breakage. Work through it
  module by module, not all at once.
- `pytest nltools/tests/data/braindata/test_braindata_core.py -xvs`
  is the tight inner loop.
- Gate each commit with: ruff check + the directly-edited test file(s)
  + the single test file that covers the thing you just changed. Defer
  broader regression runs until the whole chunk is green.

**Risks**

- Polars DataFrames are not mutable the way pandas ones are. Code
  like `bd.X["column"] = value` needs to become
  `bd.X = bd.X.with_columns(pl.lit(value).alias("column"))`. Audit
  all writes to `.X`/`.Y` before the constructor change.
- Polars' `.slice()` takes `(offset, length)`, not `[start:stop]`.
  Slicing code needs care.
- `DataFrame.empty` → `.is_empty()` (method, not property).
- `.iloc[...]`  → row-indexing via `.slice()` or `.filter()`.
- `pl.DataFrame` doesn't carry an index. `reset_index(drop=True)` is a
  no-op. Remove any index-reliant logic.
- Polars integer-backed nullable columns don't convert cleanly to
  pandas without pyarrow. If we ever need pandas back, always use the
  `to_dict(as_series=False)` trick.

**Touchpoints outside `braindata/`**

- `nltools/data/collection/__init__.py` creates `BrainData` instances
  and sets `.Y`
- `nltools/data/collection/modeling.py` pulls BrainData X/Y for
  modeling
- Integration tests that mock BrainData from fixtures

---

### Chunk B — Adjacency

**Goal**: `Adjacency.labels`, `Adjacency.Y`, any cached per-matrix
metadata, and all stats helpers use polars or numpy internally.
`Adjacency.stats_label_distance` and friends already use polars for
long-format outputs (landed in `ef367fe`); the class's own internal
state is the remaining work.

**Files**
- `nltools/data/adjacency/__init__.py` — class; `.labels`, `.Y`, IO,
  arithmetic
- `nltools/data/adjacency/io.py` — read_csv, HDF5 persistence
- `nltools/data/adjacency/modeling.py` — regression accepts pandas
  design matrices
- `nltools/data/adjacency/stats.py` — some functions still build
  `pd.DataFrame` internally; `plot_label_distance` is a known candidate
- `nltools/data/adjacency/utils.py` — helper functions

**Sub-chunks**
1. `Adjacency.labels` stored as numpy array (it's fundamentally a
   length-N vector; polars adds nothing; numpy is right).
2. `Adjacency.Y` → polars (or drop if unused — audit usage first).
3. `io.read_csv` variants → `pl.read_csv` with polars DataFrame
   returns.
4. HDF5 — same `pd.HDFStore` → `h5py` rewrite as BrainData.
5. `modeling.regress` — accept `DesignMatrix` / polars / numpy;
   convert at the edge if we hit a pandas-only dependency (unlikely
   here, regression is numpy math).
6. `stats` helpers — audit each `pd.DataFrame` construction; most can
   become polars or numpy.

**Risks**

- `Adjacency` has a lot of arithmetic operator overloading that
  propagates metadata. Make sure any attribute type change is
  visible to `__add__`, `__mul__`, `append`, etc.

---

### Chunk C — Collection (BrainCollection)

**Goal**: `BrainCollection.metadata` and per-subject X/Y frames are
polars. `collection/modeling.py` builds design matrices via nilearn
with pandas **only at the nilearn call**, not as stored state.

**Files**
- `nltools/data/collection/__init__.py` — class; metadata storage
- `nltools/data/collection/io.py`
- `nltools/data/collection/constructors.py`
- `nltools/data/collection/transforms.py`
- `nltools/data/collection/prediction.py`
- `nltools/data/collection/modeling.py` — **the big one**, ~30 `pd.`
  references. `run_first_level_glm` + `run_first_level_glm_group`
  build events, confounds, and design matrices; heavy pandas internal
  assembly. Each call site that feeds nilearn is a boundary; each
  `pd.concat` of design matrices is internal and should become
  polars.

**Depends on**: BrainData (Chunk A) being done first, because
Collection passes BrainData objects around and reads their `.X`/`.Y`.

**Sub-chunks**
1. Metadata → polars
2. `io.read_csv` loaders → polars
3. `constructors` — `from_directory`, etc., build metadata as polars
4. `transforms` — audit each method
5. `prediction` — CV outputs as polars
6. `modeling.run_first_level_glm` — the heavy lift. Break down
   further: events ingress, confounds ingress, design matrix
   assembly, nilearn call, beta extraction, metadata return. Each is
   its own commit.

---

### Chunk D — Follow-up cleanup

After A, B, C land, a final pass for drift and polish:

- Remove `_as_ndarray` helper duplication (if two+ modules grow
  similar helpers, hoist to `nltools/utils.py`).
- Audit test fixtures in `nltools/tests/conftest.py` — the `import
  pandas as pd` there is a test convenience, but most fixtures could
  be polars to exercise the polars path.
- Consider: do any public functions still accept pandas that shouldn't?
  (vs. accepting numpy or polars only). Pandas acceptance is OK but
  each one is a small maintenance burden — document or drop.
- Update CHANGELOG.md with the full list of breaking changes:
  - `BrainData.X` / `.Y` type
  - `Adjacency.labels` type
  - `BrainCollection.metadata` type
  - `fetch_neurovault_collection` return type
  - `plot_silhouette`, `plot_between_label_distance`,
    `plot_mean_label_distance` return types
  - `zscore` return type
- Once everything is migrated, re-run the full `uv run pytest -n auto`
  as a final regression gate, then the slow tests gated by explicit
  user approval.

## Things we deliberately left alone

- **Seaborn's pandas dependency**: seaborn hard-requires pandas, and
  plotting functions that use `sns.violinplot(data=...)` etc. have no
  way to pass polars. The strategy is to build long-format polars
  frames internally and convert to pandas **at the sns call**. No
  attempt to replace seaborn.
- **Nilearn's pandas dependency**: nilearn's
  `make_first_level_design_matrix`, event loading, and GLM APIs
  require pandas DataFrames for events/confounds. Same strategy:
  convert at the call site, don't store pandas.
- **`pd.HDFStore` for BrainData's legacy HDF5 format**: until Chunk A
  step 10, this stays — we don't want a half-migrated persistence
  layer.
- **Docstring examples that use pandas**: where the example is
  demonstrating a function whose input is inherently a
  pandas-DataFrame interface (e.g. nilearn), the docstring stays
  pandas. Where it's optional, we switched to polars.
- **The stats module's pandas acceptance isinstance checks**: these
  are the goal-state boundary — we accept pandas so users don't need
  to convert. Leaving them in place is correct; they don't eagerly
  import pandas and don't propagate pandas internally.

## Testing rhythm (TDD discipline)

Established in the work so far; keep it:

1. **Scope the sub-chunk** to a single function or ~1 class method.
2. **Write or update the test first** — make it fail red.
3. **Make the minimum change** to the code to turn it green.
4. **Run only the targeted test file** or `-k` expression.
5. **Ruff check** the edited files.
6. **Commit** once green + ruff clean. Small commits.
7. Never run the full suite mid-chunk; reserve it for the end of a
   chunk or after a risky cascade.
8. If a subagent is helpful for mechanical work (like the lazy-import
   sweep that became `bbbfea6`), delegate and verify, but do the
   sensitive rewrites yourself.

## Quick reference — useful conversion recipes

```python
# polars → pandas without pyarrow
pd.DataFrame(pl_df.to_dict(as_series=False))

# polars Series → numpy
s.to_numpy()

# polars DataFrame → numpy (2D, row-major)
df.to_numpy()

# accept mixed input, return numpy 2D
def _as_ndarray(x):
    import numpy as np, polars as pl
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, pl.DataFrame):
        return x.to_numpy()
    try:
        import pandas as pd
        if isinstance(x, pd.DataFrame):
            return x.values
    except ImportError:
        pass
    raise ValueError(...)

# z-score columns in polars
df.select([
    ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c)
    for c in df.columns
])

# empty polars frame (replaces pd.DataFrame())
pl.DataFrame()

# polars frame emptiness (method, not attribute)
df.is_empty()

# polars row slice
df.slice(offset, length)        # or
df[offset:offset + length]

# polars column add / overwrite
df.with_columns(pl.lit(value).alias("col"))
df.with_columns(pl.Series("col", values))
```
