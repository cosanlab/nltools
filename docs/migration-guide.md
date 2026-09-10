# Migration Guide: v0.5 → v0.6

Version 0.6.0 is a **breaking release** that refactors nltools to better leverage nilearn and establish cleaner APIs. This guide shows you how to update your code.

---

## Quick Reference: What Changed

| Category | v0.5.1 (Old) | v0.6.0 (New) | Status |
|----------|--------------|--------------|--------|
| **Class names** | `Brain_Data`, `Design_Matrix` | `BrainData`, `DesignMatrix` | **Renamed** |
| **Import paths** | `nltools.file_reader`, `nltools.simulator`, `nltools.external` | `nltools.io`, `nltools.data`, `nltools.algorithms` | **Moved** |
| **GLM regression** | `BrainData.regress()` | `.fit(model='glm', X=…)` | **Removed** |
| **Ridge regression** | Manual | `.fit(model='ridge')` | New |
| **ML prediction** | `.predict(algorithm='svm', cv_dict=…)` returning dict | `.predict(y=…, spatial_scale=…, estimator=…, cv=…)` returning `Predict` dataclass with `.weight_map`, `.scores`, etc. | Unified API |
| **Spatial scale kwarg** | N/A (or `method=` overloaded for both algorithm and spatial scale) | `spatial_scale=` (`'whole_brain' \| 'roi' \| 'searchlight'`) — distinct from `method=` (algorithm); follows the spatial-scale framing of [Jolly & Chang, 2021, *SCAN*](https://doi.org/10.1093/scan/nsab010) | **New canonical kwarg** |
| **RSA workflow** | Manual: per-ROI loop, build Adjacency stack, reduce, paint via `roi_to_brain` | `bd.distance(..., spatial_scale='roi', roi_mask=atlas).similarity(model_rdm)` followed by explicit atlas mapping with `roi_to_brain_from_atlas` | **New** |
| **One-sample t-test** | `BrainData.ttest(threshold_dict=…)` | `BrainData.ttest(popmean=0.0, permutation=False, …)` | **Signature changed** |
| **Two-sample t-test** | N/A | `scipy.stats.ttest_ind` on `.data`, or `nltools.algorithms.inference.two_sample_permutation_test` | New |
| **Method chaining** | `.smooth()` modifies in-place | Returns copy | Changed |
| **Properties** | Method-style shape/empty checks | `.shape`, `.is_empty` | Changed |
| **Cross-validation** | N/A | `.fit(..., cv=5)` | New |
| **HyperAlignment** | Via `align()` only | `HyperAlignment` class | New |
| **Multi-subject** | `Brain_Collection` | Collection orchestration is deferred to 0.6.1; use an explicit per-subject `BrainData` workflow in 0.6.0 | **Deferred** |
| **SRM** | N/A | `SRM` / `DetSRM` classes | **New** |
| **GPU inference** | N/A | `inference` module | **New** |
| **Algorithm kwarg** | `algorithm=`, `scheme=`, `kind=`, `noise_model=`, `extract_type=`, `mode=`, `perm_type=` | `method=` (or `spatial_scale=` for spatial scale; `Adjacency.similarity` keeps the correlation type in the separate `metric=` slot) | **Renamed** |
| **Progress flag** | `show_progress=True` | `progress_bar=False` | **Renamed + default flipped** |
| **Sphere radius** | `radius=` (units implicit) | `radius_mm=` everywhere except `BrainData.predict`, which keeps `radius=` (millimeters, matching nilearn's searchlight) | **Renamed** |
| **Permutation count** | `n_perm=` (Adjacency.generate_permutations) | `n_permute=` | **Renamed** |
| **Similarity diagonal** | `ignore_diagonal=False` | `include_diag=False` (polarity flipped, default now excludes diagonal) | **Changed** |
| **Duplicate columns on append** | `append(axis=1)` accepted value-identical columns | Raises `ValueError` — value-identical columns refused | **Changed** |
| **Cluster summary kwargs** | `cluster_summary(method=…, summary=…)` | `cluster_summary(summary=…, scope='within' \| 'between')` | **Renamed** |
| **ROI extraction kwarg** | `extract_roi(metric=…)` | `extract_roi(method=…)` | **Renamed** |
| **BrainData.plot thresholds** | `thr_upper=`, `thr_lower=`, `kind=` | `upper=`, `lower=`, `method=` | **Renamed** |
| **`DesignMatrix.convolve()` columns** | 1-D kernel: name preserved (`stim` → `stim`); 2-D kernel: `stim_c0`, `stim_c1` | Always suffixed `<col>_c{i}`; source column dropped (`stim` → `stim_c0`) | **Renamed (consistent)** |
| **Generated column names** | `poly_0`, `cosine_1`, `global_spike1`, `0_poly_0` | `.nl_poly_0`, `.nl_cosine_1`, `.nl_global_spike1`, `.nl_r0_poly_0` — the reserved `.nl_` namespace | **Renamed** |
| **Plotting functions** | `surface_plot`, `scatterplot`, `roc_plot`, `heatmap`, … | `plot_surf`, `plot_scatter`, `plot_roc`, `plot_designmatrix`, … | **Renamed** |
| **`nifti_masker` attr** | `brain_data.nifti_masker` | Use `nilearn.masking.apply_mask(img, bd.mask)` | **Removed** |
| **`nltools.prefs`** | Stateful template singleton | `set_brainspace()` / `get_brainspace()` / `with_brainspace()` | **Removed** |
| **Neurovault helpers** | `download_collection`, `get_collection_image_metadata` | `fetch_neurovault_collection` | **Removed** |
| **ICC reliability** | `BrainData.icc()`, `nltools.stats.compute_icc` | None — compute externally (e.g. `pingouin.intraclass_corr`) | **Removed** |

---

## Class Renames

**Status**: **BREAKING** — no backward-compatibility aliases exist

All data classes now follow PEP 8 naming conventions. The old names are **not available** — using them will raise `ImportError`.

| v0.5.1 (Old) | v0.6.0 (New) |
|---------------|--------------|
| `Brain_Data` | `BrainData` |
| `Design_Matrix` | `DesignMatrix` |

**Find and replace in your codebase:**
```bash
# sed/sd commands for bulk rename
sd 'Brain_Data' 'BrainData' **/*.py **/*.ipynb
sd 'Design_Matrix' 'DesignMatrix' **/*.py **/*.ipynb
```

**Import examples:**
```python
# OLD (v0.5.1) — these will raise ImportError in v0.6.0
from nltools.data import Brain_Data, Design_Matrix
from nltools import Brain_Data

# NEW (v0.6.0)
from nltools.data import BrainData, DesignMatrix
from nltools import BrainData
```

---

## Import Path Changes

**Status**: **BREAKING** — old module paths no longer exist

Several modules have been reorganized. The old import paths will raise `ModuleNotFoundError`.

| v0.5.1 Import | v0.6.0 Import | Status |
|----------------|---------------|--------|
| `from nltools.simulator import Simulator` | `from nltools import Simulator` | Moved to `nltools.data.simulator` |
| `from nltools.simulator import SimulateGrid` | `from nltools import SimulateGrid` | Moved to `nltools.data.simulator` |
| `from nltools.file_reader import onsets_to_dm` | **Removed** | Folded into `DesignMatrix.__init__` — `DesignMatrix(events_path, run_length=N, TR=t)` HRF-convolves by default (`hrf_model='glover'`, matches nilearn); pass `hrf_model=None` for raw boxcar |
| `from nltools.external import glover_hrf` | `from nltools.algorithms.hrf import glover_hrf` | Moved to `nltools.algorithms` |
| `from nltools.utils import get_anatomical` | **Removed** | Use `nilearn.datasets.load_mni152_brain_mask()` |
| `from nltools.stats import regress` | `from nltools.algorithms import regress` | Standalone OLS helper: `regress(X, Y)`; only `BrainData.regress()` was removed |

**Example migrations:**
```python
# OLD: glover_hrf
from nltools.external import glover_hrf
# NEW:
from nltools.algorithms.hrf import glover_hrf

# OLD: onsets_to_dm (file path → convolved DM in one call)
from nltools.file_reader import onsets_to_dm
dm = onsets_to_dm(events_path, run_length=200, sampling_freq=0.5)
# NEW: DesignMatrix accepts BIDS events / confounds files directly and
# HRF-convolves by default — same default as nilearn's
# make_first_level_design_matrix(hrf_model='glover'). Columns get the
# canonical `_c0` suffix and .convolved is populated.
from nltools.data import DesignMatrix
dm = DesignMatrix(events_path, run_length=200, TR=2.0)

# Need raw boxcar instead? (PPI / FIR / pedagogy that builds interaction
# terms before convolution.) Opt out:
dm_boxcar = DesignMatrix(events_path, run_length=200, TR=2.0, hrf_model=None)
dm = dm_boxcar.convolve()  # convolve later, after manipulating regressors

# In-memory events DataFrame? Use the helper directly (always boxcar — caller convolves):
from nltools.data.designmatrix.io import events_to_dm
dm_data = events_to_dm(events_frame, run_length=200, sampling_freq=0.5)
dm = DesignMatrix(dm_data, sampling_freq=0.5).convolve()

# OLD: get_anatomical (removed entirely)
from nltools.utils import get_anatomical
anat = get_anatomical()
# NEW: use nilearn directly
from nilearn.datasets import load_mni152_template
anat = load_mni152_template(resolution=2)

# OLD: Simulator / SimulateGrid
from nltools.simulator import Simulator, SimulateGrid
# NEW:
from nltools import Simulator, SimulateGrid
# or: from nltools.data import Simulator, SimulateGrid
```

**`nltools.stats` → `nltools.algorithms`** (see [the stats-module removal](#stats-module-removed) for the full mapping):
- `from nltools.algorithms import fdr, fisher_r_to_z, zscore, find_spikes, threshold`
- `from nltools.algorithms import regress` (standalone OLS helper; distinct from the removed `BrainData.regress()`)
- `from nltools.algorithms import one_sample_permutation_test` (the same function as `nltools.algorithms.inference`'s)

**Unchanged imports** (these still work as before):
- `from nltools.plotting import component_viewer`
- `from nltools.mask import roi_to_brain, expand_mask, create_sphere`
- `from nltools.data import Adjacency` (name unchanged)

---

## Dependency Updates

### nilearn 0.12+ Compatibility

**Status**: ✅ FIXED (v0.6.0)

nltools v0.6.0 now requires **nilearn >= 0.12**, which introduced a breaking change in `NiftiMasker.transform()`:

**What changed in nilearn 0.12:**
- **3D images** now transform to **1D arrays** `(n_voxels,)` instead of 2D arrays `(1, n_voxels)`
- **4D images** still transform to **2D arrays** `(n_timepoints, n_voxels)` (unchanged)

**How nltools adapted:**
- Updated `BrainData._load_from_list()` to use `np.vstack()` instead of `np.concatenate()`
- This ensures correct shape when loading lists of 3D nifti files
- **No user code changes needed** - BrainData API remains identical

**If you're using nilearn directly**, be aware:
```python
from nilearn.maskers import NiftiMasker
import nibabel as nib

masker = NiftiMasker(mask_img=mask)
masker.fit()

# nilearn 0.11 (old)
result = masker.transform(nib.load('image_3d.nii.gz'))
print(result.shape)  # (1, 238955) - 2D array

# nilearn 0.12+ (new)
result = masker.transform(nib.load('image_3d.nii.gz'))
print(result.shape)  # (238955,) - 1D array ⚠️ Breaking change!

# If you need consistent 2D output:
result = masker.transform(nib.load('image_3d.nii.gz'))
if result.ndim == 1:
    result = result.reshape(1, -1)  # Force 2D: (1, n_voxels)
```

**Other dependency updates in v0.6.0:**
- Python >= 3.11 (dropped 3.10 support)
- polars >= 1.18 (from 0.20)
- h5py >= 3.15 (from 3.13)
- pytest >= 8.4 (from 8.3)

---

## Breaking Changes

(find-spikes-dedup)=
### `find_spikes()` no longer emits duplicate regressors

**Status**: ✅ NEW (v0.6.0) — always deduplicated

`find_spikes()` runs two independent detectors (per-TR global signal, and mean
absolute frame-to-frame difference). A single bad volume is routinely caught by
both, and each detection became its own one-hot indicator column — so the same
TR could be flagged twice, producing *exactly identical* regressors and a
rank-deficient design.

```python
spikes = bold.find_spikes(global_spike_cutoff=0.8, diff_spike_cutoff=0.8, TR=2.4)
# before: 23 columns, rank 16  -> rank deficient
# now:    16 columns, rank 16  -> full rank
```

When a TR is flagged by both detectors the `.nl_global_spike` column is kept, so
the result is deterministic rather than dependent on insertion order.

This is deduplication of the function's own output rather than a modeling
decision — the colliding columns are bitwise identical, so only the retained
*name* is at stake and nothing is lost. That is why the interim `clean=` kwarg
was dropped and deduplication is unconditional: an opt-out would only
manufacture straight duplicate columns, which `append(axis=1)` now refuses
(see [](#append-duplicate-columns)).

**Finding no spikes also works properly now.** Polars derives a frame's height
from its columns, so a design matrix with no regressors used to report 0 rows —
which meant a clean subject with no detected spikes broke the whole first-level
build:

```python
spikes = bold.find_spikes(...)      # subject has no spikes
task.append(spikes, axis=1)
# ValueError: All Design Matrices must have the same number of rows!
```

`find_spikes` now hands the row count to `DesignMatrix` explicitly, so the empty
result reports `(n_tr, 0)` and contributes no columns on horizontal append.
Explicitly sized empty inputs must match the design's row count. The default
`DesignMatrix()` with no size or timing is an identity on either side of append.

(append-duplicate-columns)=
### `append(axis=1)` refuses value-identical columns

**Status**: ⚠️ **BREAKING** (v0.6.0)

Appending a column whose values equal an existing column under any name raises
`ValueError`, as duplicate names already did. Comparisons use the final columns
after missing-value filling, so filling can create a duplicate. Numeric equality
is exact across numeric types. Signed zeros match, and distinct large integers
remain distinct. With `fill_na=None`, NaN matches NaN and null matches null at
the same positions, but null differs from NaN.

Duplicate columns make the design rank deficient. Drop or modify one before
appending. Pre-existing duplicates in the base matrix do not prevent an unrelated
append.

(designmatrix-file-round-trip)=
### `DesignMatrix` files read back — `.csv` separator fixed, `.h5` reader added

**Status**: ✅ **FIXED** (v0.6.0) — `.write()` and `DesignMatrix(path)` are now symmetric

Writing a design matrix and reading it back did not work. Two independent
defects:

**A `.csv` was written tab-separated.** `.write()` defaulted to a tab delimiter
whatever the extension, while the file constructor picked the delimiter from
the extension — so a `.csv` round-tripped into a single column named
`'cond_a\tcond_b'`:

```python
dm.write("design.csv")
DesignMatrix("design.csv", sampling_freq=0.5, run_length="infer").columns
# v0.5.1/0.6.0-dev: ['cond_a\tcond_b']   <- one mashed column
# v0.6.0:           ['cond_a', 'cond_b']
```

The delimiter now follows the extension on **both** sides (`.csv` → comma,
everything else → tab). An explicit `sep=` still overrides it. Files already on
disk with the mismatched delimiter are detected and re-parsed, so they load
correctly without intervention.

**There was no `.h5` reader.** `.write("dm.h5")` produced a valid HDF5 file that
nothing could open — the constructor sent every path to the CSV reader, which
failed with `ComputeError: invalid utf-8 sequence`. `DesignMatrix` now reads
its own HDF5 files, and because such a file is a serialized object rather than
a table awaiting interpretation, it needs no `run_length` or `sampling_freq`:

```python
dm.write("design.h5")
back = DesignMatrix("design.h5")     # no other arguments required

back.sampling_freq   # restored
back.confounds       # restored
back.convolved       # restored
back.multi           # restored
```

Passing `sampling_freq=` / `convolved=` / `confounds=` explicitly still
overrides whatever the file recorded. HDF5 files written by earlier 0.6.0
builds (a plain float matrix beside an `S`-typed `columns` dataset) are read
too; new files store the frame as Arrow IPC bytes, so column dtypes survive
exactly — an integer spike indicator comes back an integer instead of a float.
A column-less matrix also records its row count, so `find_spikes()` output for
a subject with no spikes round-trips as `(n_tr, 0)` rather than `(0, 0)`.

(reserved-column-prefix)=
### Generated columns are namespaced with `.nl_`

**Status**: ⚠️ **BREAKING** (v0.6.0) — every column nltools generates was renamed

Column names nltools invents now live in a reserved namespace marked by the
prefix `.nl_`. Nothing about the columns themselves changed — only their names:

| v0.5.1 | v0.6.0 | Produced by |
|---|---|---|
| `poly_0`, `poly_1`, … | `.nl_poly_0`, `.nl_poly_1`, … | `DesignMatrix.add_poly()` |
| `cosine_0`, `cosine_1`, … | `.nl_cosine_0`, `.nl_cosine_1`, … | `DesignMatrix.add_dct_basis()` |
| `global_spike1`, `diff_spike1`, … | `.nl_global_spike1`, `.nl_diff_spike1`, … | `find_spikes()` |
| `0_poly_0`, `1_motion_x`, … | `.nl_r0_poly_0`, `.nl_r1_motion_x`, … | `append(axis=0, keep_separate=True)` |

Note the run-separation form: the run index moved inside the prefix and gained
an `r` (`0_poly_0` → `.nl_r0_poly_0`), and prefixes never stack — a
`.nl_poly_0` separated into run 1 becomes `.nl_r1_poly_0`, not
`.nl_r1_.nl_poly_0`. Run separation applies to your own confound columns too,
so a user column `motion_x` becomes `.nl_r0_motion_x`: the run-prefixed variant
is a name nltools generated, so it belongs to the reserved namespace.

**Why.** nltools has to recognize its own columns — to refuse a global drift
term on a design that already models drift per run, to drop intercepts before
computing VIF, and so on. Those checks used to be heuristics over
user-controlled names, and they were wrong in both directions. `add_poly()`
counted underscores, so a design carrying the standard 24-parameter motion
expansion (`trans_x_sq`, `rot_x_diff_sq`, …) could not have drift terms added
at all:

```python
task.append(motion_24, axis=1, as_confounds=True).add_poly(order=2)
# v0.5.1: ValueError: ...polynomial terms that were kept separate...
# v0.6.0: works — the design has no run-separated drift terms
```

and `vif(exclude_confounds=False)` dropped any column whose name merely
contained `poly_0` while missing the all-ones `cosine_0` it actually needed to
drop. With a namespace nltools controls, both checks key on the prefix and
neither can be fooled: **you can now name your own regressors anything.**

**What to change.** Any code that refers to a generated column by name:

```python
# OLD (v0.5.1)
dm["poly_0"]
dm.columns.get_loc("0_poly_0")
betas = fit.betas[dm.columns.index("cosine_1")]

# NEW (v0.6.0)
dm[".nl_poly_0"]
dm.columns.index(".nl_r0_poly_0")
betas = fit.betas[dm.columns.index(".nl_cosine_1")]
```

Selecting all generated columns is now a prefix test rather than a pattern
guess, and `nltools.utils.RESERVED_PREFIX` holds the token so you never need to
hard-code it:

```python
from nltools.utils import RESERVED_PREFIX, is_reserved_name

generated = [c for c in dm.columns if is_reserved_name(c)]
task_only = [c for c in dm.columns if not c.startswith(RESERVED_PREFIX)]
```

`append(axis=1)` refuses a raw Polars frame whose columns use the reserved
prefix, which identifies generated columns. Rename user columns before
appending. `DesignMatrix` inputs may contain generated columns with the prefix.
Convert pandas inputs with the `DesignMatrix` constructor before appending.

(fit-no-implicit-design-clean)=
### `fit()` no longer cleans the design matrix

**Status**: ⚠️ **BREAKING** (v0.6.0) — the `design_clean*` kwargs were removed

`BrainData.fit(model='glm')` used to run `DesignMatrix.clean()` on `X` before
estimating, silently dropping any column correlating above `0.95` with an
earlier one. That is gone. `fit()` now estimates exactly the design you pass.

```python
# OLD — silently dropped columns, and the kwargs tuned the dropping
bd.fit(model='glm', X=dm)                       # cleaned behind your back
bd.fit(model='glm', X=dm, design_clean=False)   # opt out
bd.fit(model='glm', X=dm, design_clean_thresh=0.8)

# NEW — fit() estimates what you give it; clean explicitly if you want to
bd.fit(model='glm', X=dm)
bd.fit(model='glm', X=dm.clean(thresh=0.8))
```

Removed kwargs: `design_clean`, `design_clean_thresh`,
`design_clean_exclude_confounds`, `design_clean_fill_na`. Passing any of them
now raises `TypeError`.

**Why**: the old behavior applied a *correlation* heuristic, not a rank test,
so it dropped columns from designs that were perfectly estimable. Worse, it
kept the first column of each correlated pair and dropped the second — making
the fitted model depend on the order you happened to build the design in:

```python
base.add_dct_basis(duration=128).add_poly(order=2)   # dropped .nl_poly_1, .nl_poly_2
base.add_poly(order=2).add_dct_basis(duration=128)   # dropped .nl_cosine_1, .nl_cosine_2
```

Same regressors, same data, two different models, no warning either way.
Dropping regressors is a modeling decision, so it belongs to the caller.

**In exchange, `fit()` now warns when the design is genuinely rank deficient.**
That case previously passed silently in *both* modes: with cleaning off, nilearn
falls back to a pseudo-inverse and splits the effect evenly across the linearly
dependent columns, returning finite, plausible-looking betas that are not
uniquely determined.

```text
RankDeficientDesignWarning: Design matrix is rank deficient: rank 2 of 3
columns — 1 column(s) are linear combinations of the others (likely involved:
condA_dup). The OLS betas are not uniquely determined, and contrasts touching
the dependent columns are not interpretable: the fit silently returns one of
infinitely many solutions. Possible fixes: (1) inspect the collinearity with
`DesignMatrix.vif()`; (2) try `DesignMatrix.clean()` to drop redundant columns
before fitting (note: which of a correlated pair survives depends on the order
the design was built in); (3) try regularization — `fit(model='ridge')` keeps
every regressor and has a unique, order-invariant solution.
```

The diagnosis names the likely-involved columns (truncated for wide designs)
rather than dumping the full roster, and a design with more columns than
timepoints — rank deficient by construction — is called out as such instead of
being skipped. The warning has its own category so it can be silenced
surgically: `warnings.filterwarnings("ignore",
category=nltools.data.braindata.modeling.RankDeficientDesignWarning)`.

**Full-rank but near-collinear designs warn too.** The designs the old
`design_clean` used to prune — a column pair correlated at `|r| >= 0.95` — are
technically estimable, so the rank check stays silent on them; `fit()` now
fires a separate `NearCollinearDesignWarning` instead, naming the offending
pair(s). A second signal, a condition number of the column-standardized design
above 30 (Belsley's classic cutoff), catches near-dependence spread across
three or more columns that no pairwise correlation reveals; the message says
which signal fired. Constant (intercept-like) columns are excluded from the
scan, so generated drift and intercept terms don't false-positive. Like the
rank warning this is diagnosis only — nothing is dropped, and the same three
fixes apply (`vif()`, an explicit `clean()`, or ridge). A rank-deficient
design fires only `RankDeficientDesignWarning`, never both.

#### Prefer regularization to dropping columns

If the warning fires, **regularization is usually the better fix**. Ridge has a
unique solution even when `X'X` is singular, because `(X'X + alpha*I)` is always
invertible, and that solution does not depend on the order of the columns:

```python
# Deletion: which regressor survives depends on how you built the design
DesignMatrix({"a": a, "b": b, "c": c}).clean(thresh=0.95).columns   # ['a', 'c']
DesignMatrix({"b": b, "a": a, "c": c}).clean(thresh=0.95).columns   # ['b', 'c']

# Shrinkage: swap the collinear columns and you get the same model back
bd.fit(model="ridge", X=np.column_stack([a, b, c]), ridge_alpha=1.0)  # w = [w_a, w_b, w_c]
bd.fit(model="ridge", X=np.column_stack([b, a, c]), ridge_alpha=1.0)  # w = [w_b, w_a, w_c]
```

Dropping a regressor does not make its variance disappear — it reassigns it to
whichever correlated column happened to survive, which silently changes what the
remaining coefficients mean. Shrinkage instead distributes the shared variance
across the collinear set in a determined way. Pass a sequence of candidate
`alpha` values with a `cv` to choose the penalty by cross-validation rather than
by hand.

The caveat worth stating plainly: regularization fixes the *estimation* problem,
not the *identifiability* one. If two regressors are exactly collinear, no method
can separate their individual contributions — that information is not in the
data. Ridge gives you a stable, reproducible answer instead of an arbitrary one;
it does not recover something that was never measured.

**Migration**: if you relied on the implicit cleaning, decide deliberately —
switch to `fit(model='ridge')`, or add an explicit `.clean()` to your
design-building chain. If you see the new rank warning, your design was already
producing non-unique estimates; inspect it with `.vif()` rather than suppressing
the warning.


(inference-progress-bar-off)=
### Permutation and bootstrap progress bars are off by default

**Status**: ✅ NEW (v0.6.0) — `progress_bar=False` everywhere

Every permutation-test and bootstrap entry point — the `algorithms.inference` engines and the class facades (`BrainData.bootstrap`, `Adjacency.similarity` / `.ttest` / `.bootstrap`) — now takes `progress_bar: bool = False` and stays silent unless asked. Previously most of these functions wrote a tqdm bar to stderr unconditionally, which emitted one bar per call in any loop (a 100-iteration calibration study produced 100 bars).

The one *silent* behavior change: `isc_permutation_test` and `isc_group_permutation_test` previously defaulted to `progress_bar=True` — existing calls will no longer show a bar. Pass `progress_bar=True` to any of these functions to get it back:

```python
# before: bar appeared unasked
stats = isc_permutation_test(data)

# now: opt in explicitly
stats = isc_permutation_test(data, progress_bar=True)
```

`BrainData.fit` follows the same convention: `progress_bar` defaults to `False` and no longer inherits `bd.verbose` when unset (`verbose` is reserved for log-level output only) — pass `progress_bar=True` explicitly if you relied on that coupling.

The mechanism is also unified: all bars go through shared helpers in `nltools.utils` (`maybe_tqdm` / `make_progress_bar`) built on `tqdm.auto`, so notebooks render widget bars and terminals render text bars.


(ttest-popmean-permutation)=
### `BrainData.ttest(popmean=..., permutation=True)` now tests against `popmean`

**Status**: ⚠️ **BREAKING CHANGE** (v0.6.0) — silently wrong p-values fixed

The permutation branch of `BrainData.ttest` previously handed the *raw* data to the sign-flip engine, so with a non-zero `popmean` the returned p-values answered "mean ≠ 0" while the parametric branch (and the docstring) answered "mean ≠ `popmean`". The permutation branch also overwrote the returned `"mean"` map with the raw voxelwise mean instead of the documented effect size. Both are fixed: the engine now sign-flips `images - popmean`, and `"mean"` is `mean(images) - popmean` on both branches.

```python
# v0.6.0-dev (buggy): p tested mean != 0 regardless of popmean
res = bd.ttest(popmean=0.5, permutation=True)

# v0.6.0: p tests mean != 0.5; res["mean"] is mean(images) - 0.5
```

Calls with the default `popmean=0.0` (the overwhelmingly common case) are numerically unchanged. If you recorded permutation p-values from a non-zero `popmean` call, they were wrong — re-run that analysis.

`return_null=True` now works. Earlier in v0.6.0 development it was accepted and silently ignored, and the null was computed and discarded. Combined with `permutation=True` it now adds `"null_dist"` to the result dict: a plain `(n_permute, n_voxels)` array of centered means in the units of `"mean"`. The four map keys are unchanged, and `return_null` still has no effect on the parametric path, which computes no null.

```python
# before: the null was computed and thrown away
res = bd.ttest(permutation=True, n_permute=5000, return_null=True)
set(res)  # {"mean", "t", "z", "p"}

# now
set(res)  # {"mean", "t", "z", "p", "null_dist"}
res["null_dist"].shape  # (5000, n_voxels)
```

(adjacency-ttest-contract)=
### `Adjacency.ttest` matches `BrainData.ttest`

**Status**: ⚠️ **BREAKING CHANGE** (v0.6.0) — mislabeled permutation `t` fixed

`Adjacency.ttest` now returns the same four keys as `BrainData.ttest` — `"mean"`, `"t"`, `"z"`, `"p"` — plus `"null_dist"` with `permutation=True, return_null=True`. It also gains `popmean=0.0`. Its permutation branch previously stored the *mean* under `"t"`; `"t"` is now the observed t-statistic on both branches, and the permutation null holds centered means. Results are independent Adjacency objects that keep the node count, storage kind (directed storage included), and shared node labels, and never alias the input. Every returned map is `float64`, including `"null_dist"` — the shared statistics helper computes in double precision, so a `float32` stack no longer returns `float32` results. There are no compatibility shims.

```python
# before: {"t", "p"}, where permutation "t" was really the edgewise mean
t_map = stacked.ttest(permutation=True)["t"]

# now: the mean is its own key and "t" is the t-statistic
res = stacked.ttest(permutation=True, popmean=0.0)
res["mean"], res["t"], res["z"], res["p"]
```

The permutation path is faster too: it calls the engine once for the whole edge matrix instead of once per edge.


(browser-support-deferred)=
### In-browser (WASM) support removed — returns in 0.6.1

**Status**: ⚠️ **BREAKING CHANGE** (v0.6.0) — deferred, not abandoned

The in-browser stack — the marimo WASM tutorial pages and the library's Pyodide path — is removed from v0.6.0 and will return in 0.6.1. All of it is preserved on the [`0.6.1-browser` branch](https://github.com/cosanlab/nltools/tree/0.6.1-browser) and tracked in [#487](https://github.com/cosanlab/nltools/issues/487). Removed:

- `nltools.templates.seed_resources` and the Pyodide/IDBFS fetch path in `nltools.templates.fetch`
- `nltools.datasets.PAIN_RESOURCES`, `nltools.datasets.EMOTION_METADATA`, and `nltools.datasets.emotion_resources`
- the `docs-wasm` / `test-pyodide` poe tasks, the pyodide CI job, and the node-based Pyodide smoke tests

Tutorials are now plain marimo `.py` notebooks (PEP 723 header: `marimo` + `nltools`) meant for local editing (`uvx marimo edit --sandbox <nb>.py`) or [molab](https://molab.marimo.io); the docs site renders executed previews of them.

(stats-module-removed)=
### `nltools.stats` removed — everything lives in `nltools.algorithms`

**Status**: ⚠️ **BREAKING CHANGE** (v0.6.0)

The `nltools.stats` module is gone. It had become a thin compatibility layer over the functional core, and v0.6.0 consolidates that core into a single entry point: every user-facing statistical function is importable **flat from `nltools.algorithms`**.

```python
# OLD (v0.5.x)
from nltools.stats import fdr, zscore, isc, one_sample_permutation_test

# NEW (v0.6.0)
from nltools.algorithms import fdr, zscore, isc, one_sample_permutation_test
```

The implementations moved into focused submodules (the flat import above is all most code needs):

| Old module | New module | Functions |
|---|---|---|
| `nltools.stats.corrections` | `nltools.algorithms.corrections` | `fdr`, `holm_bonf`, `threshold`, `multi_threshold` |
| `nltools.stats.outliers` | `nltools.algorithms.outliers` | `zscore`, `winsorize`, `trim`, `find_spikes` |
| `nltools.stats.timeseries` | `nltools.algorithms.signal` | `downsample`, `upsample`, `calc_bpm`, `make_cosine_basis` |
| `nltools.stats.correlation` | `nltools.algorithms.similarity` | `fisher_r_to_z`, `fisher_z_to_r`, `compute_similarity`, `compute_multivariate_similarity`, `transform_pairwise` |
| `nltools.stats.regression` | `nltools.algorithms.regression` | `regress` |
| `nltools.stats.alignment` | `nltools.algorithms.alignment` | `align`, `procrustes`, `procrustes_distance`, `align_states` |
| `nltools.stats.intersubject` | `nltools.algorithms.inference.intersubject` | `isc`, `isc_group`, `isfc`, `isps` |
| `nltools.stats.permutation` | *(deleted — the wrappers are gone)* | the `nltools.algorithms` exports **are** the `algorithms.inference` engine functions |

Two kwarg renames rode along, applying the canonical `device=` vocabulary to the inference engine itself (the old `nltools.stats` wrappers used to translate these names at the boundary):

- **`parallel=` → `device=`** on every `algorithms.inference` entry point (`one_sample_permutation_test`, `two_sample_permutation_test`, `correlation_permutation_test`, `timeseries_correlation_permutation_test`, `matrix_permutation_test`, `isc_permutation_test`, `isc_group_permutation_test`). Values are unchanged: `'cpu'` (joblib), `'gpu'` (PyTorch), `None` (single-threaded). Result dicts likewise report a `'device'` key instead of `'parallel'`.
- **`phase_randomize(backend=)` → `phase_randomize(device=)`** with `'cpu' | 'gpu' | 'auto'` replacing `'numpy' | 'torch'`.

The ISC family was canonicalized the same way: `isc_permutation_test` / `isc_group_permutation_test` rename `metric=` (the `'median'|'mean'` central-tendency choice) to **`summary=`** and `sim_metric=` (the similarity metric) to **`metric=`**; `isc_group()` likewise takes `summary=` instead of `metric=`. All ISC results (including wrappers) now expose the null under the engine-standard **`null_dist`** key — the legacy `null_distribution` key is gone — and the `isc` / `isc_group` wrappers expose `progress_bar: bool = False`.

The same `summary` vocabulary reached the two remaining mean/median knobs on the data classes:

- **`Adjacency.cluster_summary(method=, summary=)` → `cluster_summary(summary=, scope=)`** — the `'mean'|'median'|None` central tendency is now `summary=` (was `method=`), and the within/between-cluster choice is now `scope='within'|'between'` (it previously squatted on the `summary=` name).
- **`BrainData.extract_roi(metric=)` → `extract_roi(method=)`** — `'mean'|'median'|'pca'` selects an extraction *variant* (PCA is not a central tendency), so it takes the canonical `method=` name; `metric=` stays reserved for distance/similarity metrics.

Both renames also appear in the [Renamed kwargs](#renamed-kwargs) table.

(tail-vocabulary)=
### One canonical `tail=` vocabulary (v0.6.0)

**Status**: ✅ COMPLETE (v0.6.0)

Every sign-ambiguous p-value in the library now defaults **two-tailed** and speaks one vocabulary: `tail: int | str = 2`, accepting `2 | 'two'` (two-tailed) and `1 | 'one'` (one-tailed in the test's canonical *positive* direction — correlation/ISC/similarity > 0, mean > `popmean`, group1 > group2). The direction is fixed by the test, never chosen from the data (a data-driven direction would silently halve every p-value); to test the negative direction, negate your data, swap the groups, or flip the contrast. Default (`tail=2`) output is numerically unchanged everywhere.

What changed:

- **Removed forms**: the v0.5 `-1` / `'upper'` / `'lower'` arguments now raise a `ValueError` with the negate/swap/flip guidance.
- **Bug fix**: `BrainData.ttest(tail=1)` and `Adjacency.ttest(tail=1)` previously ignored `tail` on the (default) parametric path and always returned two-sided p-values; `tail` now maps onto scipy's `alternative=` so one-tailed parametric tests actually happen. The `"z"` map is derived from the reported p, so it matches the requested tail.
- **New `tail=` options** (default 2 ≡ old behavior): `BrainData.bootstrap` / `Adjacency.bootstrap`, `BrainData.multivariate_similarity`, `regress` / `Adjacency.regress`, and `Roc.calculate`.
- **No knob where only one tail is valid**: `distance_correlation` (dcorr ≥ 0), ANOVA's F, `isps`' Rayleigh test, and SRM variance components keep their statistically forced one-tailed p-values, unchanged.
- **The GLM exception**: GLM contrast inference — `compute_contrasts(..., inference=True)` — reports nilearn's **one-sided** upper-tail p-value, matching the nilearn/SPM directional-contrast convention ("A > B" is the hypothesis; flip the contrast for the other direction). This is the one documented deviation from the two-tailed default.

Code that already imported from `nltools.stats` gets the same signatures it had before — the wrappers' canonical `device=` names are now the engine's. Only code that called the `algorithms.inference` engines directly with `parallel=` needs the kwarg rename.

(gpu-execution-layer)=
### One GPU execution layer — measured budgets, OOM recovery, run-or-raise

**Status**: ⚠️ **BREAKING CHANGE** (v0.6.0)

Every GPU/batched code path now runs through one core layer in `nltools.algorithms.backends` (`device_memory_budget`, `auto_batch_size`, `compute_oom_safe`), replacing five independent batch-size calculators and their hard-coded memory constants. Three things change for users:

- **`max_gpu_memory_gb` defaults to `None` = measured, everywhere.** Previously every GPU entry point assumed a fixed 4 GB budget (and SRM/hyperalignment's internal worker sizing assumed 8 GB) regardless of hardware — a 2 GB card would OOM under the default while a 24 GB card ran at a fraction of capacity. `None` now measures the device at call time (free CUDA memory with headroom; available system RAM for MPS/CPU). When sizing batches, a measured budget is additionally capped at an 8 GB saturation ceiling — larger per-batch working sets add allocation latency without computing any faster, and on unified-memory systems they starve the host. Passing an explicit number behaves exactly as before: it is used verbatim, uncapped. Batch size never affects seeded results, only memory/speed.
- **Device OOM is recovered, not fatal.** If a batch still exhausts device memory, the already-generated batch inputs are split and retried at smaller sizes (`compute_oom_safe`). Because RNG draws happen before the device compute, recovery reuses the exact same permutations; the recovered result matches the uninterrupted one to within float32 reduction order (~1 ulp — backends block reductions differently per batch shape). Only when a *single* item cannot fit does the run fail, with a `MemoryError` naming the fix.
- **Run-or-raise policy**: an explicit `device='gpu'` / `parallel='gpu'` either runs on the GPU or raises — never a silent CPU fallback. `'auto'` remains the one documented graceful path. Concretely:
  - `SRM` / `DetSRM` `fit()` / `transform()` with `parallel='gpu'` now raise `NotImplementedError` (they previously ran on CPU silently). The dead `max_gpu_memory_gb` kwarg on their `fit()` is removed — it controlled nothing.
  - `LocalAlignment` validates `parallel=` (a typo like `'gup'` previously ran single-threaded numpy with no error), raises `NotImplementedError` for `parallel='gpu'` with `method='srm'|'hyperalignment'` (previously a documented silent CPU run), and raises `ImportError` for `parallel='gpu'` without PyTorch (previously a log message + numpy fallback).
  - `correlation_permutation_test(metric='kendall', device='gpu')` no longer warns and falls back to CPU — Kendall now has a real GPU kernel (tie-corrected tau-b via pre-computed pairwise sign tensors, parity-tested against `scipy.stats.kendalltau`).
- **GPU Spearman results over tied data change.** The GPU rank transform mishandled ties — its tie window was off by one on both ends (the first tied element kept its raw rank, the next distinct value was averaged in, and a trailing run was skipped entirely), and its tie scan corrupted the row index for multi-row batches — so any `correlation_permutation_test(metric='spearman', device='gpu')` correlation or null distribution over data with tied values (integer ratings, discrete scores) was numerically wrong. Ranks now match `scipy.stats.rankdata(method='average')` exactly and GPU results match the CPU path; continuous (untied) data was unaffected. Re-run any analysis that recorded GPU Spearman results over tied data.
- **GPU null distributions from `timeseries_correlation_permutation_test` change.** GPU draws now equal the CPU draws for a given seed — the GPU path previously derived circle-shift amounts through a different RNG call, breaking the deterministic cross-backend contract. The batched phase-randomization path also no longer mispairs conjugate frequencies, a bug that made the surrogate spectrum non-Hermitian and silently distorted the surrogates (statistically wrong, not just nondeterministic). Same test, same distribution family, different draws — re-run any analysis that recorded seeded GPU timeseries permutation p-values.

(predict-group)=
### Stored labels and cross-validation in BrainData decoding

`BrainData.predict(y=None)` uses a single-column `.Y`; `y='name'` and
`groups='name'` select columns of a multi-column `.Y`. Assign labels and grouping
variables together with `bd.Y = {"label": labels, "run": runs}`. On a BrainData
carrying both a fitted encoding model and stored `.Y`, a no-argument call
predicts from the fitted model; pass `y=` explicitly to decode the labels
instead.

`predict` takes no cross-validation name strings: `"loso"`, `"loro"`, `"logo"`
and `"loo"` all raise. Pass the splitter itself — `cv=LeaveOneGroupOut()` with
`groups=` for leave-one-group-out, `cv=LeaveOneOut()` for leave-one-out. An
integer selects that many unshuffled folds and ignores `groups`, so pass a group
splitter when groups must stay disjoint across training and test sets. `cv=None`
is a deterministic five-fold split. The standalone
[`resolve_cv`](api/tasks/prediction.md#tasks-prediction-resolve-cv) helper still
accepts the `'loo'`/`'logo'` names and still promotes an integer to a group-aware
splitter, for callers writing their own loops. The legacy fluent `cv()` pipeline
is removed; configure cross-validation through `BrainData.predict`.

Collection decoding and its permutation result fields are deferred to 0.6.1;
see [BrainCollection](#braincollection).

(iplot-autoscale)=
### `iplot()` autoscales robustly; percentile thresholds are shared and zero-aware

**Status**: ⚠️ **BREAKING CHANGE** (v0.6.0)

`iplot()` previously opened with its display window at the raw data min/max, so a couple of outlier voxels set the entire color scale — a single-subject beta map rendered as washed-out noise with a featureless 3D render ([#479](https://github.com/cosanlab/nltools/issues/479)). Three related changes:

- **Robust autoscaling by default** — new kwarg `autoscale: bool = True`. The default window's ceiling is the 98th percentile of the finite **nonzero** magnitudes, so a couple of outlier voxels no longer set the whole scale — 98 is the upper edge of the "robust range" convention used by `fslstats -r` and by [niivue](https://niivue.com) itself. The floor is an epsilon, never above the smallest nonzero magnitude: zeros render transparent, every real voxel stays visible, and you threshold up from there. `autoscale=False` uses the raw magnitude range from zero to the largest absolute value. For a custom percentile window pass `lower`/`upper` (e.g. `lower="60%", upper="98%"`); explicit `threshold`/`lower`/`upper` always override the corresponding edge.
- **Percentile strings everywhere the vocabulary appears**: `iplot(upper="98%")` now works, resolved by the shared `nltools.utils.resolve_threshold` — the same helper `threshold()` uses. In `iplot`, percentiles resolve over the finite nonzero *magnitudes* (its window is a divergent magnitude window); in `threshold()`, over the finite nonzero *signed* values.
- **The slider always shows the rendered window.** `cal_min`/`cal_max` are now always computed in Python and passed explicitly — previously the traits could be `None` ("niivue auto") while the slider handles sat at the raw extremes, so the handles showed one window while niivue rendered another, and the first slider touch destroyed the auto window.

Follow-up plotting consistency changes ([#490](https://github.com/cosanlab/nltools/issues/490)) align default color ranges across renderers with nilearn's sign-dependent behavior:

- `plot(method="glass"|"slices")`, `plot_surf()`, and `plot_flatmap()` now choose a sequential red or blue map for one-sided data and `RdBu_r` for mixed-signed data. One-sided ranges include zero; mixed ranges remain symmetric. Explicit `cmap`, `vmin`, and `vmax` override these defaults.
- `plot(threshold="95%")` is now accepted and resolves over finite, nonzero magnitudes.
- `iplot(symmetric="auto")` mirrors mixed-signed maps and lets each sign in a one-sided map determine its own ceiling. Pass `True` to force symmetry or `False` to scale positive and negative limbs independently.
- The interactive controls now show live values and range endpoints. Ortho views default to 600px tall; other views remain 400px tall, and explicit `height=` wins.

**Behavior change in `threshold()`**: percentile strings (`upper="98%"`) now resolve over finite **nonzero** voxels. On a masked stat map most voxels are exactly zero, which dragged every percentile toward zero — `threshold(upper="98%")` on a sparse map previously produced a near-zero cutoff that thresholded almost nothing. Results change on any map containing zeros; pass a numeric cutoff to reproduce old outputs exactly.


(designmatrix-pandas-polars)=
### DesignMatrix: Pandas → Polars

**Status**: ✅ COMPLETE (v0.6.0)

`DesignMatrix` stores a Polars DataFrame and exposes direct Polars methods,
including expressions such as `dm.select(pl.col("stim").mean())`. Every eager
DataFrame result becomes an independently owned `DesignMatrix`. Series, scalars,
grouping objects and lazy builders retain their native Polars types. Explicit
nltools methods such as `sum()` and `corr()` retain their documented return types.

Known column selections and the row methods `head`, `tail`, `slice` and `filter` preserve
applicable annotations, nominal sampling frequency and multi-run state. Renaming
translates both annotation lists. Replacing a column clears its convolution
annotation while preserving its confound role. Aggregations and operations with
unknown semantics clear metadata whose validity cannot be established.

Constructors, copies and transformations independently own retained mutable data.
Annotation properties return detached lists. Constructor annotations must name
existing columns, timing must be finite and positive, and `n_rows` must be a
nonnegative integer. Zero-column designs retain their observation count through
selection, append, NumPy conversion and HDF5; text export requires a column.

**What's removed:**
- `.loc[]` and `.iloc[]` indexers - Use column/row access instead
- `.assign()` - Use direct column assignment instead

**What's added:**
- `.sum(axis=0)` - Sum along axis (useful for validating onset counts)
- `__eq__()` operator - Pythonic equality: `dm1 == dm2`

**What's also removed in later 0.6.0 cleanup:**
- `.reset_index()` — the pandas-compat no-op was dropped in the `drop pandas-compat shims` refactor. Polars has no row indexes, so there is no equivalent to migrate to; just remove the call.
- pandas inputs to `.downsample()`, `.upsample()`, and the internal outlier transform. Pass Polars or NumPy instead.

**What's changed:**
- Internal storage is Polars (`.data` attribute)
- Faster operations via Polars vectorization
- Column access returns Polars Series (not pandas Series)

**Common API differences** (Polars Series vs pandas Series):
```python
# Getting numpy arrays
dm['column'].to_numpy()   # ✅ Polars way
dm['column'].values       # ❌ Doesn't exist (pandas-only)

# Getting Python lists
dm['column'].to_list()    # ✅ Polars way
dm['column'].tolist()     # ❌ Doesn't exist (pandas-only)

# Computing correlations between columns
import numpy as np
corr = np.corrcoef(dm['col1'].to_numpy(), dm['col2'].to_numpy())[0, 1]  # ✅
dm['col1'].corr(dm['col2'])  # ❌ Polars Series has no .corr() method

# Saving to CSV through DesignMatrix
dm.write('/path/to/file.csv')         # Preserves the zero-column export check
dm.to_csv('/path/to/file.csv')         # ❌ Method doesn't exist

# Loading from CSV
import polars as pl
dm = DesignMatrix(pl.read_csv('/path/to/file.csv'), sampling_freq=0.5)
```

**What's the same:**
- `.shape` and `.columns` work identically; use `.is_empty` to test emptiness
- `.fillna()`, `.drop()`, `.zscore()` methods work identically
- `.convolve()`, `.upsample()`, `.downsample()` retain their nltools interfaces
- `.vif()`, `.clean()` methods work identically

**Migration examples:**
```python
# OLD (pandas-style)
dm.loc[10:15, 'ConditionA'] = 1

# NEW (Polars-style) - use direct column assignment
dm['ConditionA'] = (
    pl.when(pl.arange(0, len(dm)).is_between(10, 15))
    .then(1)
    .otherwise(dm['ConditionA'])
)

# Or for simple cases, convert to numpy and back
arr = dm.to_numpy()
arr[10:15, dm.columns.index('ConditionA')] = 1
dm = DesignMatrix(arr, columns=dm.columns, sampling_freq=dm.sampling_freq)
```

```python
# OLD (pandas .assign())
new_dm = dm.assign(new_col=lambda df: df['col1'] * 2)

# NEW (direct Polars expressions)
new_dm = dm.with_columns(new_col=pl.col('col1') * 2)
```

**New utility methods:**
```python
# Check sum of design matrix columns (useful for onset validation)
dm = DesignMatrix({'stim_a': [1, 0, 1, 0], 'stim_b': [0, 1, 0, 1]})
column_sums = dm.sum()  # Returns Polars Series with sums
column_sums.to_numpy()  # Convert to numpy array: [2, 2]

# Pythonic equality checking
dm1 = DesignMatrix({'a': [1, 2, 3]})
dm2 = DesignMatrix({'a': [1, 2, 3]})
dm1 == dm2  # True
```

**GLM workflows unchanged:**
```python
# Both DesignMatrix and pandas DataFrames work seamlessly
dm = DesignMatrix({'stim': [1, 2, 3, 4]}, sampling_freq=0.5)
brain_data.fit(model='glm', X=dm)  # Automatic conversion to pandas for nilearn
```

**Pandas input and nilearn adapters:**

The constructor immediately converts pandas frames and discards their index.
Convert pandas append inputs explicitly. Horizontal append accepts raw Polars
frames directly and tags their columns as confounds.

```python
motion = DesignMatrix(pandas_motion, sampling_freq=dm.sampling_freq,
                      confounds=list(pandas_motion.columns))
combined = dm.append(motion, axis=1)
```

The dedicated nltools `to_pandas()` method is removed. Generic forwarding still
exposes Polars methods, including Polars' own `to_pandas()`. nltools-maintained
Polars-to-pandas conversions occur only inside the GLM and events adapters where
nilearn requires pandas. Plotting uses arrays.

**Adjacency.regress() compatibility:**
```python
# Works seamlessly with Polars DesignMatrix
from nltools.data import Adjacency, DesignMatrix

adj = Adjacency([...])  # Your adjacency matrices
dm = DesignMatrix({'regressor': [1, 2, 3]})

# Automatic conversion to numpy for regression
stats = adj.regress(dm)  # Works! Converts dm.to_numpy() internally
```

**Timeline**: Complete in v0.6.0. All integration work finished. Tutorials and examples updated.

(designmatrix-from-file)=
### DesignMatrix accepts file paths

**Status**: ⚠️ **BREAKING** (v0.6.0) — replaces standalone `onsets_to_dm`

`DesignMatrix.__init__` now accepts a `.tsv` / `.csv` path (str or `pathlib.Path`) and dispatches based on column inspection:

- **BIDS events** (file has `onset` and `duration` columns) → HRF-convolved regressors aligned to TRs by default (one column per `trial_type`, suffixed `_c0`, `.convolved` populated). Default is `hrf_model='glover'`, matching nilearn's `make_first_level_design_matrix`. Pass `hrf_model=None` for raw boxcar (e.g., PPI / FIR / pedagogical material that introduces convolution as a separate step). No auto `constant` column either way — call `.add_poly(0)` for the intercept.
- **Tabular / confounds** (anything else) → read as-is. `hrf_model` is silently ignored.

```python
from nltools.data import DesignMatrix

# OLD: onsets_to_dm built and HRF-convolved in one call
from nltools.file_reader import onsets_to_dm
dm = onsets_to_dm(events_path, run_length=200, sampling_freq=0.5)

# NEW (default): one-line construct + convolve
dm = DesignMatrix(events_path, run_length=200, TR=2.0)

# Variant: append confounds + drift before convolution (PPI, etc.)
events = DesignMatrix(events_path, run_length=200, TR=2.0, hrf_model=None)
confounds = DesignMatrix(confounds_path, run_length="infer", TR=2.0)
dm = events.append(confounds, axis=1, as_confounds=True).add_poly(2).convolve()
```

Constructor rules for the file-path branch:

- `run_length` is required. `'infer'` is allowed for tabular/confounds files (uses the file's row count); rejected for events files (the row count would be the number of events, not TRs).
- Pass exactly one of `TR` (seconds) or `sampling_freq` (Hz). Passing both raises `ValueError`.
- BIDS events files require lowercase `onset`, `duration`, `trial_type` columns (and optional `modulation`).

**For in-memory events DataFrames** (the path `nltools.datasets.load_haxby_example` and similar takes), use the helper directly:

```python
from nltools.data.designmatrix.io import events_to_dm

dm_data = events_to_dm(events_frame, run_length=200, sampling_freq=0.5)
dm = DesignMatrix(dm_data, sampling_freq=0.5).convolve()
```

#### Worked example: PPI (psycho-physiological interaction) design

The PPI flow exercises most of the v0.6.0 idioms together — boxcar opt-out,
Polars-native column manipulation, mixed-input `.append()` for confounds, and
the `find_spikes` → DesignMatrix interop. The model is

```
Y_voxel = β_task·motor + β_seed·vmpfc + β_PPI·(motor × vmpfc)
        + β_conf·confounds + ε
```

where `motor` is HRF-convolved, `vmpfc` is a measured BOLD timeseries from a
seed ROI (so it is **not** convolved), and the interaction term is the
elementwise product of the two.

```python
import polars as pl
from nltools.data import DesignMatrix

# 1. Load BIDS events as boxcar — PPI needs to combine motor variants BEFORE
#    convolving, so opt out of the constructor's default HRF convolution.
events = DesignMatrix(events_path, run_length=n_tr, TR=tr, hrf_model=None)

# 2. Collapse the four motor variants into one combined regressor with a
#    Polars expression, then convolve everything in one pass.
motor_variables = ["video_left_hand", "audio_left_hand",
                   "video_right_hand", "audio_right_hand"]
task = (
    events
    .with_columns(motor=pl.sum_horizontal(motor_variables))
    .drop(motor_variables)
    .convolve()
)

# 3. Add the seed timeseries (raw — already a BOLD signal) and the PPI
#    interaction. pl.col() expressions let the interaction read like the math.
task = task.with_columns(
    vmpfc=vmpfc_signal,
).with_columns(
    vmpfc_motor=pl.col("vmpfc") * pl.col("motor_c0"),
)

# 4. Convert pandas confounds before appending. find_spikes returns a
#    DesignMatrix whose columns are already tagged as confounds.
csf_dm = DesignMatrix(csf, sampling_freq=task.sampling_freq)
motion_dm = DesignMatrix(mc_cov, sampling_freq=task.sampling_freq)
spikes = bold.find_spikes(global_spike_cutoff=3, diff_spike_cutoff=3, TR=tr)
dm = task.append(
    [csf_dm, motion_dm, spikes], axis=1, as_confounds=True,
).add_poly(order=2, include_lower=True)
```

The metadata stays consistent across the chain — `dm.convolved` lists the
HRF-convolved task regressors (`motor_c0`, …), `dm.confounds` lists the
nuisance regressors (CSF, motion, spike censors, drift), and the
regressors-of-interest (`vmpfc`, `vmpfc_motor`) stay out of both.

(designmatrix-confounds-rename)=
### DesignMatrix `.polys` → `.confounds` (attribute and kwargs)

**Status**: ⚠️ **BREAKING** (v0.6.0) — attribute rename, no compat shim

The DesignMatrix metadata list that tracks nuisance columns (intercept, polynomial drift, DCT cosines, motion regressors, …) was previously called `.polys`. v0.6.0 renames it to `.confounds` to better describe what it actually contains; method names like `add_poly` / `add_dct_basis` are unchanged but their output columns are now registered in `.confounds` instead.

| v0.5.x | v0.6.0 |
|--------|--------|
| `dm.polys` | `dm.confounds` |
| `DesignMatrix(..., polys=[...])` | `DesignMatrix(..., confounds=[...])` |
| `dm.vif(exclude_polys=True)` | `dm.vif(exclude_confounds=True)` |
| `dm.clean(exclude_polys=True)` | `dm.clean(exclude_confounds=True)` |
| (no analogue — pre-existing only on raw-DataFrame inputs) | `dm.append(other_dm, axis=1, as_confounds=True)` (new — promotes appended DM cols to confounds) |

`__repr__` now surfaces the confound list with a count:

```text
DesignMatrix(sampling_freq=0.5, shape=(200, 6))
  convolved (2): ['stim_c0', 'cue_c0']
  confounds (3): ['.nl_poly_0', '.nl_poly_1', '.nl_poly_2']
```

`DesignMatrix.write()` to `.h5` writes the metadata under the key `confounds` (was `polys`), and `DesignMatrix(path)` reads it back — see [](#designmatrix-file-round-trip).

**`BrainData.X = dm` now works.** The `.X` setter previously rejected `DesignMatrix` with `TypeError`; v0.6.0 unwraps it to `dm.data` (the underlying polars DataFrame). DM-specific metadata isn't preserved on `BrainData.X`, but you no longer need the explicit `.data` step.

(designmatrix-confounds-readonly)=
### DesignMatrix `.convolved` and `.confounds` are read-only

**Status**: ⚠️ **BREAKING** (v0.6.0) — direct assignment now raises `AttributeError`

The `.convolved` and `.confounds` properties return detached lists. Mutating a
returned list does not change the design; assigning either property raises.
Set initial annotations in the constructor, or use `.convolve()`, `.append()`,
`.add_poly()` and `.add_dct_basis()` to manage them through transformations.

```python
# OLD (v0.5.1) — silently mutates state, easy to forget when columns later get renamed
combined = DesignMatrix(
    pd.concat([dm_task.to_pandas(), motion, csf, spikes], axis=1),
    sampling_freq=0.5,
)
combined.convolved = list(dm_task.columns)   # manual re-assert after pd.concat
combined.confounds = list(motion.columns) + ["csf"] + list(spikes.columns)

# NEW (v0.6.0): convert pandas inputs and mark appended columns as confounds
motion_dm = DesignMatrix(motion, sampling_freq=dm_task.sampling_freq)
csf_dm = DesignMatrix(csf, sampling_freq=dm_task.sampling_freq)
spikes_dm = DesignMatrix(spikes, sampling_freq=dm_task.sampling_freq)
combined = dm_task.append(
    [motion_dm, csf_dm, spikes_dm], axis=1, as_confounds=True,
).add_poly(order=2)
# combined.convolved → ['stim_c0', ...]
# combined.confounds → ['motion_tx', ..., 'csf', '.nl_global_spike1', ..., '.nl_poly_0', ...]
```

Pass `convolved=` and `confounds=` to the constructor to set initial annotations:

```python
dm = DesignMatrix(arr, sampling_freq=0.5, columns=cols, confounds=["intercept"])
```

The error message points to the canonical replacement:

```text
AttributeError: DesignMatrix.confounds is read-only. Pass `confounds=...` to the
constructor, or use `.append(other_dm, axis=1, as_confounds=True)` /
`.append(raw_frame, axis=1)` (raw frames are auto-marked) to register confound regressors.
```

(designmatrix-copy-constructor)=
### `DesignMatrix(other_dm)` is now a copy-constructor

**Status**: ✅ NEW (v0.6.0) — additive, no migration required

Passing a `DesignMatrix` to the constructor returns an independent copy with `data`, `sampling_freq`, `convolved`, `confounds`, and `multi` carried over. Explicit kwargs override inherited values. This matches the pandas `pd.DataFrame(other_frame)` idiom and short-circuits the v0.5.1 "wrap it again to reset metadata" pattern (which dropped metadata on the floor).

```python
copy = DesignMatrix(dm)                          # full copy, all metadata preserved
copy = DesignMatrix(dm, sampling_freq=1.0)       # override sampling_freq, keep the rest
copy = DesignMatrix(dm, convolved=[])            # clear convolved, keep confounds
```

In v0.5.1 this raised `TypeError: Unsupported data type`.

(designmatrix-convolve-suffix)=
### DesignMatrix.convolve() always suffixes `_c{i}`

**Status**: ⚠️ **BREAKING** (v0.6.0) — column-name policy changed

`dm.convolve()` now renames every convolved column to `<col>_c{i}` regardless of kernel shape, and drops the source column. Previously the 1-D kernel path replaced columns in place (kept the original name) while the 2-D kernel path suffixed `_c0`, `_c1`, ….

The `dm.convolved` metadata list now records the **post-suffix** names that actually exist in the dataframe, so multi-run vertical `.append()` (which renames per run) keeps metadata in sync with the columns.

```python
# OLD (v0.5.1)
dm = DesignMatrix({"face": [1, 0, 1, 0]}, sampling_freq=0.5)
dm_conv = dm.convolve()
dm_conv["face"]            # ✓ existed
dm_conv.convolved          # ['face']  (matched columns)

# NEW (v0.6.0)
dm_conv = dm.convolve()
dm_conv["face"]            # ❌ KeyError — column was dropped
dm_conv["face_c0"]         # ✓
dm_conv.convolved          # ['face_c0']

# Multi-kernel call still produces _c0/_c1/... and now records all three
dm_fir = dm.convolve(conv_func=fir_basis_3kernels)
dm_fir.convolved           # ['face_c0', 'face_c1', 'face_c2']
```

**Migration**: search call sites for column lookups by trial-type name after a `.convolve()` chain (especially in `compute_contrasts(...)` strings) and append `_c0`. For example, `brain.compute_contrasts("language - string")` becomes `brain.compute_contrasts("language_c0 - string_c0")`.

**Why**: deterministic column names regardless of kernel rank, and a fix for a metadata-drift bug where 2-D-kernel `convolve()` recorded pre-suffix names that didn't exist in the dataframe — `.append(..., axis=0)`'s rename map silently skipped them and `dm.convolved` ended up referring to ghost columns.

## BrainData and Adjacency API Changes

(braindata-mask-handling)=
### BrainData Mask Handling

**Status**: ⚠️ Behavior clarification (v0.6.0)

#### How masks work

When you create a `BrainData` without specifying a mask, nltools **auto-detects** the best matching built-in MNI template based on the data's voxel resolution (1mm, 2mm, or 3mm) and resamples the data to fit if necessary. This means most users never need to think about masks at all:

```python
from nltools.data import BrainData

# Just pass a nifti file — mask is auto-detected from resolution
brain = BrainData('sub-01_bold.nii.gz')
# Auto-detects 2mm MNI template, resamples if needed
```

Available built-in templates span three families (`default`, `nilearn`, `fmriprep`) at resolutions of 1mm, 2mm, and 3mm. The default is `2mm-default`.

#### Manual control over templates

You can choose a specific template by name, or pass any nifti file or nibabel object as the mask:

```python
# Pick a specific built-in template by name
brain = BrainData('sub-01_bold.nii.gz', mask='2mm-MNI152-2009c')   # fmriprep 2mm
brain = BrainData('sub-01_bold.nii.gz', mask='3mm-MNI152-2009a')   # nilearn 3mm

# Or change the global default (affects all future BrainData)
import nltools
nltools.set_brainspace(template='fmriprep', resolution=1)

# Scope a change to a block (context manager)
with nltools.with_brainspace(template='nilearn', resolution=2):
    brain = BrainData('sub-01_bold.nii.gz')

# Inspect the current config
print(nltools.get_brainspace())

# Or pass any nifti file / nibabel object as a custom mask
brain = BrainData('sub-01_bold.nii.gz', mask='my_roi_mask.nii.gz')
brain = BrainData('sub-01_bold.nii.gz', mask=nibabel_img)
```

#### Gotcha: custom masks and save/reload

If you use a **custom mask** (not a built-in template), you must pass the same mask when reloading from NIfTI — otherwise auto-detection will pick a built-in template with a different voxel count:

```python
# Custom ROI mask — 50,000 voxels
brain = BrainData(nifti_file, mask='my_roi.nii.gz')
brain.write('/tmp/brain.nii.gz')

# ❌ WRONG: auto-detection picks a built-in template → shape mismatch
reloaded = BrainData('/tmp/brain.nii.gz')

# ✅ CORRECT: pass the same custom mask
reloaded = BrainData('/tmp/brain.nii.gz', mask='my_roi.nii.gz')
```

This is **not** an issue when using the default auto-detected templates, since the same template will be selected on reload.

**Best practice** when using custom masks — save both, or use HDF5:
```python
# Option 1: Save mask separately
brain.write('/tmp/brain.nii.gz')
brain.mask.to_filename('/tmp/mask.nii.gz')

# Option 2: Use HDF5 (preserves mask automatically)
brain.write('/tmp/brain.h5')
reloaded = BrainData('/tmp/brain.h5')  # Mask preserved
```

---

(adjacency-shape-behavior)=
### Adjacency.shape Now Returns Logical Shape

**Status**: ✅ FIXED (v0.6.0)

`Adjacency.shape` now returns the **logical shape** `(n_nodes, n_nodes)` for consistency with `BrainData.shape` and `DesignMatrix.shape`:

```python
from nltools.data import Adjacency
import numpy as np

# Create 10x10 adjacency matrix
matrix = np.random.randn(10, 10)
matrix = (matrix + matrix.T) / 2  # Make symmetric
np.fill_diagonal(matrix, 0)

adj = Adjacency(data=matrix, matrix_type='similarity')

# ✅ shape now returns logical dimensions
print(adj.shape)      # (10, 10) - the logical matrix shape
print(adj.n_nodes)    # 10 - convenience property

# For stacked matrices:
stacked = adj.append(adj)
print(stacked.shape)  # (2, 10, 10) - (n_matrices, n_nodes, n_nodes)

# To get the internal vector representation shape, use vector_shape:
print(adj.vector_shape)      # (45,) - upper triangle as vector
print(stacked.vector_shape)  # (2, 45)
```

**New properties**:
- `.shape` → `(n_nodes, n_nodes)` or `(n_matrices, n_nodes, n_nodes)`
- `.n_nodes` → Number of nodes in the matrix
- `.vector_shape` → Shape of internal vectorized storage

**Removed**:
- The old square-shape helper has no v0.6.0 alias; use `.shape` instead.

**Threshold API**: Uses `lower`/`upper` keywords, not `threshold`:
```python
# ❌ WRONG
adj.threshold(threshold=0.3)  # TypeError: unexpected keyword argument

# ✅ CORRECT
adj.threshold(upper=0.3)       # Keep values >= 0.3
adj.threshold(lower=0.5)       # Keep values <= 0.5
adj.threshold(upper='90%')     # Keep top 10% (percentile threshold)
```

---

### 1. Removed Methods

| Method | Alternative | Migration Effort |
|--------|-------------|------------------|
| `BrainData.regress()` | `.fit(model='glm', X=design_matrix)` — the old method is removed entirely; calling it raises `AttributeError` | **Low** |
| `.predict(algorithm='svm')` | `.predict(y=labels, spatial_scale=…, estimator='linear_svc', cv=…)` returning a `Predict` dataclass (`.weight_map`, `.scores`, `.predictions`, …). Fluent `.cv().predict()` on BrainData removed; pass `estimator=make_pipeline(...)` for custom preprocessing chains. `spatial_scale=` selects ``'whole_brain'``, ``'roi'``, or ``'searchlight'``; `method=` is no longer overloaded. See [Pattern 4](#pattern-4-machine-learning-classification-regression). | **Low** |
| `.decompose(algorithm='ica')` | `.decompose(method='ica', n_components=…, axis=…)` — same `algorithm → method` rename, signature is now keyword-only after `self`; `**kwargs` forwards to the sklearn decomposition estimator | **Low** |
| `BrainData.ttest(threshold_dict=…)` (v0.5.1) | `BrainData.ttest(popmean=0.0, permutation=False, …)` — restored with a new signature. Returns `{"mean", "t", "z", "p"}`, plus `"null_dist"` with `permutation=True, return_null=True`. Threshold the maps afterwards. | **Low** |
| `.randomise()` | Use nilearn permutation testing | Medium |
| `.predict_multi()` | Will return in future Model class | N/A |
| `summarize_bootstrap()` | `BrainData.bootstrap()` or `OnlineBootstrapStats` | **Low** |
| `BrainData.icc()` | Removed — voxelwise intraclass correlation is out of scope for v0.6.0. Compute ICC externally (e.g. `pingouin.intraclass_corr`) on extracted values. The `nltools.stats.compute_icc` helper is also removed. | **Low** |
| `BrainData.iplot(surface=…, anatomical=…)` | `BrainData.iplot(view='ortho'\|'render', threshold=…, autoscale=…, atlas=…, bg_img=…)` — *rebuilt* on [niivue](https://niivue.com) (self-owned `anywidget` driving `@niivue/niivue`, WebGL). Live windowing (right-drag), native 4D frame scrubbing, true 3D render, and atlas overlays. `mode`/`units`/`cut_coords`/`symmetric_cmap` removed; `view='surface'` → `view='render'`. Live kernel (Jupyter, marimo). See [Pattern: interactive viewing (`iplot`)](#interactive-viewing). | **Medium** |

:::{note}
`BrainData.ttest()` was briefly removed earlier in v0.6.0 development, then restored because one-sample voxelwise t-tests across stacked subject-level contrast maps are the 99% group-inference use case. The old `threshold_dict=` kwarg is gone — use the new permutation-based API instead.
:::

### 2. Removed Classes

| Class | Status | Alternative |
|-------|--------|-------------|
| `Brain_Collection` | Deferred | Collection orchestration targets 0.6.1 — see [BrainCollection](#braincollection) |
| `Model` | Removed | Will return in v0.7.0+ |

### 3. Attributes

| Attribute | Status | Alternative |
|-----------|--------|-------------|
| `.X` | Still works | Pass `X=` to `.fit()` directly (preferred) |
| `.Y` | Still works | Manage labels separately (preferred) |
| Old empty-state attribute | Removed | Use `.is_empty` instead |

---

## Migration Patterns

(interactive-viewing)=
### Pattern 0: Interactive viewing (`iplot`) — Rebuilt on niivue

**Status**: 🔧 **REBUILT** — `BrainData.iplot()` is now a WebGL [niivue](https://niivue.com) viewer instead of the nilearn HTML viewer. It is a self-owned `anywidget` (`NiivueViewer`) that drives `@niivue/niivue` (loaded from a CDN) directly through anywidget's standard model API — **not** `ipyniivue`. By default it renders an in-widget **threshold slider** above the viewer and shows the **stat-map colorbar**; niivue also gives live windowing (right-drag), native 4D frame scrubbing, true 3D rendering, and — the headline feature — direct overlays of nltools atlases (colored regions, outlines, hover-to-label).

`iplot` renders in a live kernel (Jupyter, marimo desktop). In-browser (WASM) support is deferred to 0.6.1 — see [](#browser-support-deferred). It does not render in statically-built (plain-Markdown) docs — use `BrainData.plot()` there.

**What changed:**

| | v0.5.1 | v0.6.0 |
|---|---|---|
| Engine | nilearn `view_img` HTML in an iframe | niivue (`@niivue/niivue`, WebGL) via a self-owned `anywidget` |
| Threshold control | Bespoke panel (Value↔Percentile, Symmetric↔Independent) | An in-widget threshold slider (`controls=True`, default) plus niivue's right-drag windowing; `threshold=`/`lower=`/`upper=` set the initial window. Reactive via the `cal_min`/`cal_max` traits |
| Colorbar | On | On by default (`colorbar=False` to hide); only the stat map carries one |
| Return value | `ipyniivue.NiiVue` | `NiivueViewer` (an `anywidget.AnyWidget`); `controls=False` hides the slider. No `ipywidgets` dependency |
| Surface view | `view='surface'` (`view_img_on_surf`) | **removed** — niivue's 3D is volumetric. Use `view='render'`, or `plot_flatmap`/`plot_surf` for a cortical mesh |
| Views | `'ortho'`, `'surface'` | `'ortho'`, `'axial'`, `'coronal'`, `'sagittal'`, `'render'` |
| 4D handling | Re-render per volume; pre-render every frame for static docs | Loaded once; niivue scrubs frames natively |
| Atlas overlay | — | `atlas='aal'` (or an `Atlas`) overlays colored regions / outlines (`outline=`) with hover labels |
| `mode=`, `units=`, `cut_coords=`, `symmetric_cmap=` | supported | **removed** (divergent windowing is implicit) |
| `cmap` default | `'RdBu_r'` | `'warm'` (niivue colormap; matplotlib names auto-mapped with a warning) |
| Static docs | mimebundle + pre-rendered fallback | none — live kernel only |

**Before (v0.5.1):**
```python
bd.iplot()                              # interactive ortho viewer
bd.iplot(surface=True)                  # surface viewer
bd.iplot(anatomical=anat)               # custom background
bd.iplot(units='percentile', upper=2.3) # percentile threshold
bd.iplot(mode='independent', lower=-1.0, upper=2.0)
```

**After (v0.6.0):**
```python
bd.iplot()                              # ortho viewer; right-drag windows live
bd.iplot(view='render')                 # 3D volume render (replaces view='surface')
bd.iplot(bg_img=anat)                   # custom background; bg_img=False disables it
bd.iplot(threshold=2.3)                 # symmetric magnitude floor (sub-threshold → transparent)
bd.iplot(lower=-1.0, upper=2.0)         # explicit divergent window endpoints

# Atlas overlays (deterministic atlases): colored regions, outlines, hover labels
bd.iplot(atlas='aal')                   # filled regions on top of the stat map
bd.iplot(atlas='aal', outline=2)        # region boundaries only (stat map stays visible)

# 4D BrainData: same call — scrub frames with niivue's native 4D controls
stack = BrainData([f1, f2, f3, f4, f5])
stack.iplot()

# Return value: a NiivueViewer widget (threshold slider + viewer, colorbar shown)
ui = bd.iplot()                         # reactive window via ui.cal_min / ui.cal_max
bd.iplot(controls=False)                # hide the slider (right-drag still windows)
bd.iplot(colorbar=False)                # hide the stat-map colorbar
```

By default (`controls=True`) `iplot()` returns a `NiivueViewer` (an `anywidget.AnyWidget`) rendering an in-widget threshold slider above the viewer; the window is reactive through the `cal_min`/`cal_max` traits. No `ipywidgets` dependency is needed either way. Pass `controls=False` to hide the slider (niivue's right-drag windowing still works). Any `new Niivue(opts)` option (e.g. `height=`, `is_colorbar=`) still passes through `iplot(...)`. For surface rendering of a cortical mesh, use `plot_flatmap`/`plot_surf` (static) — niivue's `view='render'` is a 3D volume render, not a mesh projection.



### Pattern 1: GLM Regression

**Status**: ⚠️ **REMOVED** — `BrainData.regress()` is gone entirely in v0.6.0; calling it raises `AttributeError` (`'BrainData' object has no attribute 'regress'`). It does not warn or delegate to another implementation.

Use the unified `.fit(model='glm', X=...)` API instead.

**Before (v0.5.1):**
```python
brain_data.X = design_matrix
results = brain_data.regress()  # Returns dict
betas = results['beta']
t_stats = results['t']
p_vals = results['p']
residuals = results['residual']
```

**After (v0.6.0):**
```python
from nltools.data import DesignMatrix

# X must be a DesignMatrix; fit does not preprocess the response
brain_data.fit(model='glm', X=DesignMatrix(design))
betas = brain_data.glm_betas         # BrainData object, one map per column
residuals = brain_data.glm_residual  # BrainData object

# t and p are per-contrast, not per-regressor attributes
result = brain_data.compute_contrasts('conditionA - conditionB', inference=True)
result.statistic, result.p_value     # BrainData maps
```

`regress()` returned marginal `t` and `p` for every regressor at once. v0.6.0
computes them one contrast at a time instead, because a t-statistic for a
contrast spanning several regressors needs the off-diagonal parameter
covariance that per-regressor maps cannot supply. For the trivial one-regressor
case, ask for that regressor by name.

**With noise model:**
```python
# OLD (removed)
brain_data.X = design_matrix
results = brain_data.regress(noise_model='ar1')

# NEW (v0.6.0) — model-specific options carry a `glm_` prefix
brain_data.fit(model='glm', glm_noise_model='ar1', X=design_matrix)
```

**All available GLM attributes:**
```python
brain_data.fit(model='glm', X=design_matrix)

# Attributes set by fit():
brain_data.glm_betas      # Beta coefficients (BrainData)
brain_data.glm_residual   # Residuals (BrainData)
brain_data.glm_predicted  # Predicted values (BrainData)
brain_data.glm_r2         # R-squared (BrainData)
brain_data.model_         # Fitted Glm model instance
```

| Aspect | Old | New | Benefit |
|--------|-----|-----|---------|
| API style | Dict return | Sklearn-style attributes | Composable, familiar |
| Design matrix | Stored as `.X` | Passed as a `DesignMatrix` argument | Explicit, clearer |
| Results | Dict with keys | BrainData attributes | Type-safe, chainable |
| Inference | `t` / `p` for every regressor | `compute_contrasts(..., inference=True)` per contrast | Correct for multi-regressor contrasts |
| Status | Primary API | Removed; use `.fit(model='glm', X=...)` | Clear migration path |

---

### Pattern 2: Ridge Regression (NEW)

**Before (v0.5.1):**
```python
# No built-in support - used sklearn manually
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model.fit(X, brain_data.data.T)
```

**After (v0.6.0):**
```python
brain_data.fit(model='ridge', ridge_alpha=1.0, X=features)
weights = brain_data.ridge_weights   # (n_features, n_voxels)
r2 = brain_data.ridge_r2             # R² per voxel
predictions = brain_data.predict(X=new_features)
```

Ridge numerics come from [Himalaya](https://github.com/gallantlab/himalaya),
which nltools now depends on; `nltools.models.Ridge` is the estimator behind the
facade. It never adds an intercept — center or standardize before fitting.

| Feature | Before | After | Benefit |
|---------|--------|-------|---------|
| API | Manual sklearn | Integrated `.fit()` | Convenient |
| GPU support | Manual setup | `ridge_device='gpu'` (CUDA/MPS) | Runs or raises, never silent CPU |
| CV support | Manual | `ridge_cv=5` or a splitter | Built-in |
| Alpha selection | Manual grid search | a sequence of `ridge_alpha` plus `ridge_cv` | Per-voxel by default |
| Banded ridge | Not available | a named mapping of feature spaces | Dirichlet search over space weights |

---

### Pattern 3: Cross-Validation (NEW)

**Before (v0.5.1):**
```python
# No built-in CV support
from sklearn.model_selection import cross_val_score
# Complex manual setup required
```

**After (v0.6.0):**
```python
# A sequence of candidate alphas plus a cv selects one per voxel
brain_data.fit(model='ridge', ridge_alpha=[0.1, 1, 10], ridge_cv=5, X=features)
best_alpha = brain_data.model_.alpha_        # (n_voxels,)
selection_scores = brain_data.model_.cv_scores_   # negative MSE at that alpha
```

| Feature | Before | After |
|---------|--------|-------|
| CV splits | Manual sklearn | `ridge_cv=5` (unshuffled K-fold) or a splitter |
| Alpha selection | Manual grid search | a sequence of `ridge_alpha` plus `ridge_cv` |
| Per-voxel alpha | Manual loop | `ridge_per_target_alpha=True` (the default) |
| Selection criterion | Whatever you wrote | Himalaya's negative MSE |

---

(pattern-4-machine-learning-classification-regression)=
### Pattern 4: Machine Learning (Classification/Regression)

**Before (v0.5.1):**
```python
brain_data.Y = labels
results = brain_data.predict(algorithm='svm', cv_dict={'type': 'kfolds', 'n_folds': 5})
weight_map = results['weight_map']
mean_acc = results['mcr_all'].mean()
```

**After (v0.6.0):**
```python
# Unified MVPA API — returns a frozen `Predict` dataclass.
result = brain_data.predict(y=labels, spatial_scale='whole_brain', estimator='linear_svc', cv=5)
result.weight_map        # full-data refit coefficients (BrainData)
result.estimator         # fitted full-data sklearn estimator
result.fold_weight_maps  # per-fold coefs, shape (n_folds, n_voxels)
result.scores          # per-fold scores, shape (n_folds,)
result.mean_score      # mean accuracy across folds (float)
result.predictions     # OOF predictions in original sample order
result.available()     # list non-None fields
```

| Aspect | Old | New | Reason |
|--------|-----|-----|--------|
| API | `algorithm=` | `estimator=` | Names an sklearn object, so it stays distinct from `bd.fit(model=)`, which selects an estimator class |
| Estimator shortcuts | `'svm'`, `'logistic'`, `'ridge'`, `'lda'` | `'linear_svc'`, `'logistic_regression'`, `'linear_discriminant_analysis'`, `'ridge_classifier'` (classification); `'ridge'`, `'lasso'`, `'linear_svr'` (regression) | The abbreviations were ambiguous about kernel and task; each shortcut now names its estimator |
| CV | `cv_dict=` | `cv=` — `None` for a deterministic five folds, an int fold count, or an sklearn splitter (test folds must partition the rows) | Simpler, and follows scikit-learn's grammar |
| Scoring | hardcoded | `scoring=None` uses the estimator's own `score` (accuracy for a classifier, R² for a regressor); any sklearn scoring name or callable overrides it | Follows scikit-learn's single-metric contract |
| Label storage | `.Y` attribute | `y=` argument | Explicit |
| Custom transforms | `brain.cv(k).normalize().reduce().pipe(t).predict()` (fluent) | Pass `estimator=make_pipeline(StandardScaler(), MyXform(), LinearSVC())`, used exactly as given | Standard sklearn pattern, no separate API to learn |
| Return type | dict (`weight_map`, `mcr_all`, …) | `Predict` dataclass | Frozen, introspectable via `.available()` / `.asdict()` |
| Weight map | top-level dict key | `result.weight_map` | Full-data refit coefficients for linear models; per-fold coefficients are in `result.fold_weight_maps` |

**Removed**: `brain.cv(k).predict(y, algorithm=…)` fluent API. The full set of fluent steps (`cv()`, `normalize()`, `reduce()`, `pipe()`) on `BrainData` collapses to kwargs on `bd.predict()`. The standalone `nltools.pipelines.Pipeline` orchestrator was also removed in v0.6.0. Collection orchestration is deferred to 0.6.1. Custom single-dataset preprocessing uses `estimator=make_pipeline(...)` on `bd.predict()`.

---

### Pattern 5: Method Chaining

**Before (v0.5.1):**
```python
brain_data.smooth(5.0)  # Modifies in-place
brain_data.standardize()  # Modifies in-place
```

**After (v0.6.0):**
```python
# Returns new objects (immutable pattern)
smoothed = brain_data.smooth(5.0)
standardized = smoothed.standardize()

# Or chain:
result = brain_data.smooth(5.0).standardize()
```

| Aspect | Old | New | Benefit |
|--------|-----|-----|---------|
| Mutation | In-place | Returns copy | Safer, composable |
| Performance | N/A | ~80% faster (efficient copying) | Optimized |
| Original data | Lost | Preserved | Safer |

---

### Pattern 6: Properties vs Methods

**Before (v0.5.1):**
```python
shape = brain_data.shape()
# Empty-state check used the removed pre-v0.6 accessor.
dtype = brain_data.dtype()
```

**After (v0.6.0):**
```python
shape = brain_data.shape       # No parentheses
is_empty = brain_data.is_empty # No parentheses; the old attribute was removed
dtype = brain_data.dtype       # No parentheses
```

| Method | Old | New | Reason |
|--------|-----|-----|--------|
| `.shape()` | Method call | `.shape` property | No computation |
| Old empty-state accessor | Method call | `.is_empty` property | No computation |
| `.dtype()` | Method call | `.dtype` property | No computation |

---

### Pattern 7: HyperAlignment (NEW)

**Before (v0.5.1):**
```python
# Only available via align() function
aligned = align(data, method='procrustes')
# No access to transformation matrices or reusable model
```

**After (v0.6.0):**
```python
# Option 1: Use align() as before (still works)
aligned = align(data, method='procrustes')

# Option 2: Use HyperAlignment class (NEW)
from nltools.algorithms import HyperAlignment

hyper = HyperAlignment(n_iter=2)
hyper.fit(data)
aligned = hyper.transform(data)

# Access transformations
transforms = hyper.w_
template = hyper.s_

# Align new subject
new_aligned, R, disp, scale = hyper.transform_subject(new_data)
```

| Aspect | Old | New | Benefit |
|--------|-----|-----|---------|
| API | Function only | Class + function | Reusable model |
| Transformations | Not accessible | `.w_` attribute | Inspectable |
| New subjects | Re-run align() | `.transform_subject()` | Efficient |
| sklearn compat | No | Yes | Composable |

---

### Pattern 8: Bootstrap Summary Statistics

**Status**: ⚠️ **BREAKING CHANGE** - `summarize_bootstrap()` has been removed in v0.6.0

The `summarize_bootstrap()` function has been removed and replaced with `BrainData.bootstrap()` and `OnlineBootstrapStats` for more efficient and flexible bootstrap analysis.

**Before (v0.5.1):**
```python
from nltools.stats import summarize_bootstrap

# Create BrainData with multiple bootstrap samples
bootstrap_samples = BrainData(list_of_samples)  # Multiple samples

# Summarize bootstrap samples
result = summarize_bootstrap(bootstrap_samples, save_weights=False)
# Returns: {'mean': BrainData, 'Z': BrainData, 'p': BrainData}

mean_brain = result['mean']
z_brain = result['Z']
p_brain = result['p']
```

**After (v0.6.0) - Option 1: Use BrainData.bootstrap()**
```python
# For generating bootstrap samples and getting statistics
boot = brain.bootstrap(stat='mean', n_samples=1000)
# Returns BrainData with bootstrap mean

# For model statistics (weights, predictions), returns dict with all stats.
# Fitting keeps no copy of the features, so a Ridge bootstrap takes them back
# explicitly and holds the selected hyperparameters fixed across replicates.
brain.fit(X=features, model='ridge', ridge_alpha=1.0)
boot = brain.bootstrap(stat='weights', X=features, n_samples=1000)
# Returns: {'mean': BrainData, 'std': BrainData, 'Z': BrainData, 'p': BrainData,
#           'ci_lower': BrainData, 'ci_upper': BrainData}
```

**After (v0.6.0) - Option 2: Use OnlineBootstrapStats for existing samples**
```python
from nltools.algorithms.inference.bootstrap import OnlineBootstrapStats
from nltools.data import BrainData

# If you already have bootstrap samples (BrainData with multiple images)
bootstrap_samples = BrainData(list_of_samples)

# Initialize OnlineBootstrapStats with shape matching your data
stats = OnlineBootstrapStats(
    shape=(bootstrap_samples.shape[1],),  # Number of voxels/features
    save_samples=False,  # Set True if you need 'samples' key
    percentiles=(2.5, 97.5)  # For confidence intervals
)

# Update with each bootstrap sample
for sample in bootstrap_samples:  # Iterate over samples
    stats.update(sample.data)  # Pass 1D array of voxel values

# Get results (equivalent to summarize_bootstrap output)
result = stats.get_results()
# Returns: {'mean': array, 'std': array, 'Z': array, 'p': array,
#           'ci_lower': array, 'ci_upper': array}

# Convert to BrainData format (reproduce old API format)
mean_brain = bootstrap_samples[0].copy()
mean_brain.data = result['mean']

z_brain = bootstrap_samples[0].copy()
z_brain.data = result['Z']

p_brain = bootstrap_samples[0].copy()
p_brain.data = result['p']

# Result equivalent to old summarize_bootstrap():
equivalent_result = {
    'mean': mean_brain,
    'Z': z_brain,
    'p': p_brain
}
# Optionally include samples if save_samples=True:
if 'samples' in result:
    equivalent_result['samples'] = result['samples']
```

| Aspect | Old | New | Benefit |
|--------|-----|-----|---------|
| API | Single function | Multiple options | More flexible |
| Memory | Stores all samples | Optional online stats | More efficient |
| Additional outputs | mean, Z, p | Plus std, ci_lower, ci_upper | More complete |
| Integration | Standalone | Integrated with BrainData.bootstrap() | Better workflow |

---

(pattern-9-stats-py-inference-module-migration)=
### Pattern 9: Stats.py → Inference Module Migration

**Status**: ✅ Complete — `nltools.stats` is gone; the inference engine is the public API (see [the stats-module removal](#stats-module-removed))

#### ISC Functions (`isc()`, `isc_group()`, `isfc()`, `isps()`)

The familiar intersubject entry points survived the consolidation and import from `nltools.algorithms`:

```python
from nltools.algorithms import isc, isc_group, isfc, isps

result = isc(data, n_samples=1000)
result = isc_group(group1, group2, n_samples=1000)
result = isfc(data)
```

For direct engine access (GPU support, `n_permute` vocabulary, `null_dist` key):

```python
from nltools.algorithms.inference import (
    isc_permutation_test,
    isc_group_permutation_test,
)

# ISC - single group
result = isc_permutation_test(data, n_permute=1000)

# ISC Group - two groups
result = isc_group_permutation_test(group1, group2, n_permute=1000)
```

**Key Changes**:
- `isc()` / `isc_group()` keep `n_samples=` (bootstrap vocabulary); everything else is canonical — `summary='median'|'mean'` for the central tendency (previously `metric=` on `isc_group`), `metric=` for the similarity metric, and a `null_dist` result key (the old `null_distribution` key is gone)
- `isfc()` remains a functional-connectivity calculation and does not perform permutation inference
- GPU acceleration is available on the engine functions with `device="gpu"`; CPU parallelization with `device="cpu"` and `n_jobs=-1`

**Performance**: 4-8× CPU speedup, 10-100× GPU speedup

#### Removed Functions

**Functions Removed** (use alternatives):
- `BrainData.regress()` → Use `BrainData.fit(model='glm', X=...)`. The standalone `regress(X, Y)` OLS helper remains available from `nltools.algorithms`.
- `regress_permutation()` → Use inference module permutation tests
- `correlation()` → Use `correlation_permutation_test()` from inference module
- `pearson()` → Use `scipy.stats.pearsonr` or `correlation_permutation_test()`

**Matrix Utilities** (now in the inference module, also exported flat from `nltools.algorithms`):
- `double_center()` → `nltools.algorithms.inference.double_center()`
- `u_center()` → `nltools.algorithms.inference.u_center()`
- `distance_correlation()` → `nltools.algorithms.inference.distance_correlation()`

---

(pattern-10-fit-dataclass-braindata-fit-inplace-false)=
### Pattern 10: Fit Dataclass (`BrainData.fit(inplace=False)`)

**Status**: ✅ NEW FEATURE (v0.6.0)

**New Feature**: `BrainData.fit()` now supports returning Fit objects instead of mutating attributes.

**Old API** (still works, default behavior):
```python
brain.fit(X=dm, model='ridge', ridge_alpha=1.0)  # Mutates brain, adds attributes
assert hasattr(brain, 'ridge_weights')
```

**New API** (recommended):
```python
from nltools.data import Fit

fit = brain.fit(X=dm, model='ridge', ridge_alpha=1.0, inplace=False)  # Returns Fit object
assert isinstance(fit, Fit)
assert 'weights' in fit.available()
assert not hasattr(brain, 'ridge_weights')  # Data attributes NOT set on brain

# Note: brain.model_ is still set even with inplace=False.
# Only the result attributes (ridge_weights, glm_betas, etc.) are kept off self.

# Serialization
import numpy as np
np.savez('fit_results.npz', **fit.asdict())
loaded = Fit(**{k: np.load('fit_results.npz')[k] for k in np.load('fit_results.npz').files})
```

**Use Cases**:
- Immutable results (no accidental mutation)
- Serialization (save/load fits)
- Multiple fits on same BrainData object
- Functional programming style

**Fit Dataclass Attributes**:
- **Ridge**: `weights`, `scores`, `fitted_values`
- **GLM**: `betas`, `t_stats`, `p_values`, `se`, `residuals`, `fitted_values`, `r2`

---

### Pattern 11: Bootstrap Infrastructure (`OnlineBootstrapStats`)

**Status**: ✅ NEW FEATURE (v0.6.0)

**New Feature**: Memory-efficient online bootstrap statistics.

**Old API** (still works):
```python
boot = brain.bootstrap(stat='mean', n_samples=5000)
```

**New Implementation**:
- Uses `OnlineBootstrapStats` for memory efficiency
- Supports CPU parallelization (`n_jobs=-1`)
- Works with fitted models (ridge, GLM)

**Advanced Usage**:
```python
from nltools.algorithms.inference import OnlineBootstrapStats

# Direct usage (numpy arrays)
stats = OnlineBootstrapStats(shape=samples[0].shape)
for sample in samples:
    stats.update(sample)
result = stats.get_results()
```

---

### Pattern 12: GPU Acceleration

**Status**: ✅ NEW FEATURE (v0.6.0)

**New Feature**: GPU-accelerated permutation tests (10-100× speedup).

**Requirements**:
- PyTorch installed
- CUDA-capable GPU (optional; CPU parallelization available)

**Usage**:
```python
from nltools.algorithms.inference import one_sample_permutation_test

# CPU (default)
result = one_sample_permutation_test(data, n_permute=1000)

# GPU (automatic batching; the memory budget is measured from the device —
# pass max_gpu_memory_gb=<GB> only to cap it explicitly)
result = one_sample_permutation_test(
    data,
    n_permute=1000,
    device='gpu',
)

# CPU parallel (4-8× speedup)
result = one_sample_permutation_test(
    data,
    n_permute=1000,
    device='cpu',
    n_jobs=-1  # Use all cores
)
```

See the [GPU-Accelerated Statistical Inference](#new-feature-gpu-accelerated-statistical-inference) section below for more details.

---

### Pattern 13: Shared Response Model (SRM) (NEW)

**Status**: ✅ NEW (v0.6.0)

**Before (v0.5.1):**
```python
# No built-in SRM support - used brainiak or custom implementations
```

**After (v0.6.0):**
```python
from nltools.algorithms import SRM, DetSRM

# Probabilistic SRM
model = SRM(features=50, n_iter=10)
model.fit(subjects)             # List of (n_voxels, n_timepoints) arrays
aligned = model.transform(subjects)  # Project to shared space

# Deterministic SRM (faster, no noise model)
det_model = DetSRM(features=50, n_iter=10)
det_model.fit(subjects)
aligned = det_model.transform(subjects)

# Align a new subject to existing shared space
rotation = model.transform_subject(new_data)
```

| Aspect | Before | After | Benefit |
|--------|--------|-------|---------|
| Availability | External library | Built-in | No extra dependency |
| API | Varies | sklearn-compatible | Composable pipelines |
| Variants | N/A | SRM + DetSRM | Flexibility |

---

## v0.6.0 Kwarg Standardization (April 2026)

**Status**: ✅ Complete (v0.6.0). No aliases kept for the old spellings — callers using the legacy names will hit a `TypeError: unexpected keyword argument`.

A sweep of the implemented data-class facades (`BrainData`, `Adjacency`, and `DesignMatrix`) landed in a series of `!:` commits on 2026-04-14 and 2026-04-20 to make kwarg names consistent across the public API. The canonical names are documented in `docs/_data/api-vocabulary.yml` (rendered in the [architecture docs](development/index.md)); the table below is the migration mapping for callers.

(renamed-kwargs)=
### Renamed kwargs

| Concept | Old kwarg(s) | New kwarg | Scope |
|---|---|---|---|
| Algorithm / variant choice | `algorithm`, `scheme`, `kind`, `noise_model`, `extract_type`, `mode`, `perm_type` | `method` | Implemented facade methods including `BrainData.decompose`, `Adjacency.cluster`, `Adjacency.similarity`, and the permutation helpers. For `Adjacency.similarity`, `method=` selects the permutation scheme (`'1d'` / `'2d'` / `None`) and the correlation type lives in the separate `metric=` slot (`'spearman'` / `'pearson'` / `'kendall'`). **Note:** `BrainData.predict` and `BrainData.distance` use the new `spatial_scale=` kwarg (not `method=`) for selecting `'whole_brain'`/`'roi'`/`'searchlight'` — see "Spatial scale" row below. |
| Spatial scale (whole-brain / ROI / searchlight) | `method='whole_brain'\|'roi'\|'searchlight'` (predict only — overloaded with the algorithm slot, never canonical elsewhere) | `spatial_scale='whole_brain'\|'roi'\|'searchlight'` | `BrainData.predict` and `BrainData.distance`. Companion kwargs `roi_mask=` and `radius_mm=` are already canonical. Naming follows the spatial-scale framing of [Jolly & Chang, 2021, *SCAN*](https://doi.org/10.1093/scan/nsab010). The `method=` slot is now reserved for algorithm choice everywhere. |
| Classifier / sklearn estimator | `algorithm=` (predict) | `estimator=` | `BrainData.predict`. It names an sklearn object, so it stays distinct from `BrainData.fit(model=…)`, which selects an estimator class. String shortcuts: classification — `'linear_svc'`, `'logistic_regression'`, `'linear_discriminant_analysis'`, `'ridge_classifier'`; regression — `'ridge'`, `'lasso'`, `'linear_svr'`. Or pass any sklearn estimator / `Pipeline` directly, which is used exactly as given. |
| Progress indicator | `show_progress` (defaulted `True`) | `progress_bar` (defaults `False`, matching sklearn) | Implemented facade methods and their submodules. `verbose` is kept only where it controls log-level output (info prints in `DesignMatrix.clean` / `.append`). |
| Warning suppression in `standardize` | `verbose=`, then briefly `suppress_warnings=` | *(removed)* | `BrainData.standardize` no longer delegates to `sklearn.preprocessing.scale`, so there are no numerical warnings to suppress: it computes in float64 (exact on raw float32 BOLD), casts back to the input dtype, and maps constant voxels/observations to 0 instead of NaN. Drop the kwarg. |
| Sphere / searchlight radius | `radius` (millimeters, but units were implicit) | `radius_mm` | `BrainData.plot_flatmap`, `nltools.plotting.plot_surf`, and `plot_flatmap`. `BrainData.predict` keeps `radius` (millimeters, matching nilearn's searchlight), as do the pure-geometry helpers (`create_sphere`, `Simulator`). |
| Permutation count | `n_perm` | `n_permute` | `Adjacency.generate_permutations`. |
| Similarity diagonal | `ignore_diagonal=False` | `include_diag=False` | `Adjacency.similarity`. **Polarity is flipped AND the default changed**: directed matrices now exclude the (trivially 1.0) self-similarity diagonal by default. No-op for symmetric matrices, which never store the diagonal. |
| Threshold arms on `BrainData.plot` | `thr_upper`, `thr_lower`, `kind` | `upper`, `lower`, `method` | The convenience scalar `threshold=` kwarg is unchanged. |
| Contrast output statistic | `contrast_type` | `inference` | `BrainData.compute_contrasts` and `Glm.compute_contrasts` return the contrast effect by default — the right input to a second-level model. There is no statistic to select: `inference=True` returns one `ContrastResult` carrying effect, variance, standard error, t-statistic, z-score, one-sided p-value, and degrees of freedom together. |
| Central tendency + cluster scope | `method=` (the `'mean'\|'median'\|None` choice), `summary=` (the within/between choice) | `summary=`, `scope=` | `Adjacency.cluster_summary` — the central tendency moved to `summary=`, and the within/between-cluster choice it displaced is now `scope='within'\|'between'`. See [the stats-module removal](#stats-module-removed) for the full `summary=` vocabulary sweep (ISC family included). |
| ROI extraction variant | `metric=` | `method=` | `BrainData.extract_roi` — `'mean'\|'median'\|'pca'` selects an extraction *variant* (PCA is not a central tendency), so it takes the canonical `method=` name; `metric=` stays reserved for distance/similarity metrics. |

### Migration examples

```python
# OLD
brain.predict(algorithm='svm', cv_dict={'type': 'kfolds', 'n_folds': 5}, radius=10)
brain.decompose(algorithm='ica', n_components=20, axis='images', whiten=True)
brain.plot(kind='glass', thr_upper=2.3, thr_lower=-2.3)
adj.generate_permutations(n_perm=1000)
adj.similarity(other, ignore_diagonal=True)  # old: include the diagonal

# NEW
brain.predict(y=labels, spatial_scale='searchlight', estimator='linear_svc', cv=5, radius=10)
brain.decompose(method='ica', n_components=20, axis='images', whiten=True)
brain.plot(method='glass', upper=2.3, lower=-2.3)
adj.generate_permutations(n_permute=1000)
adj.similarity(other, include_diag=False)     # explicit + now the default for directed
```

### Algorithm-layer APIs are unchanged

Internal algorithm classes — `CVScheme.scheme`, `Glm.noise_model` — keep their legacy names. The class facades translate at the boundary. You only need to update code that calls the facade methods.

(designmatrix-localalignment-scale)=
### `LocalAlignment`: `scheme` → `spatial_scale`, `parcellation` → `roi_mask`

**Status**: ⚠️ **BREAKING** (v0.6.0) — public class, no compat aliases

`LocalAlignment` (in `nltools.algorithms.alignment`, re-exported from `nltools.algorithms`) now speaks the canonical spatial-scale vocabulary instead of the Bazeille-et-al. "scheme" naming, so it matches `BrainData.align` and the rest of the API:

| v0.5.x / earlier v0.6 dev | v0.6.0 |
|---|---|
| `scheme=` | `spatial_scale=` |
| value `'piecewise'` | value `'roi'` |
| `parcellation=` | `roi_mask=` |

Values are `'searchlight'` (default, overlapping spheres) or `'roi'` (non-overlapping parcels — the "piecewise" scheme of Bazeille et al. 2021). The error/validation strings changed accordingly (`"Unknown scheme"` → `"Unknown spatial_scale"`, `"parcellation is required..."` → `"roi_mask is required for spatial_scale='roi'"`).

```python
from nltools.algorithms import LocalAlignment

# OLD
la = LocalAlignment(scheme='piecewise', parcellation=atlas, method='procrustes')

# NEW
la = LocalAlignment(spatial_scale='roi', roi_mask=atlas, method='procrustes')
```

`BrainData.align` validates `spatial_scale` up front: it supports `'whole_brain'` and `'roi'`; searchlight raises `NotImplementedError`.

---

## Explicit signatures instead of `**kwargs` passthroughs

**Status**: ⚠️ **BREAKING** (v0.6.0, 2026-04-20) — if you relied on forwarding arbitrary unknown kwargs through a facade method, that will now raise `TypeError: unexpected keyword argument`.

Internal `**kwargs` catch-alls have been removed from user-facing methods that delegate to nltools code (they are retained only where the target is a third-party library — sklearn estimator constructors, matplotlib, nilearn, nibabel, seaborn, pandas, scipy).

**Newly-explicit kwargs you can now pass directly** (previously hidden behind `**kwargs`):

- `BrainData.bootstrap`: `X`, `X_test`, `device`, `memory_budget_gb`
- `BrainData.ttest`, `Adjacency.ttest`: `n_permute`, `tail`, `return_null`, `n_jobs`, `random_state`
- `Adjacency.similarity`: `tail`, `return_null`, `n_jobs`, `random_state`

**Dead `*args` / `**kwargs` dropped entirely**:
- `BrainData.align` (never used internally)
- `Adjacency.regress`, `BrainData.regress` deprecation shim
- `Adjacency.__init__` (swallowed unused kwargs)

### Keyword-only (`*`) marker after the primary data arg

The implemented data-class `__init__` methods require keyword arguments after the first positional data arg. Methods with many optional kwargs also enforce keyword-only.

```python
# OLD — these relied on positional order
brain = BrainData(data, Y_vec, design_frame, mask_img)          # positional Y/X/mask
adj = Adjacency(vec, "directed")                                # was actually binding to Y, not matrix_type!

# NEW — positional-only up to the primary data arg; rest must be keyword
brain = BrainData(data, Y=Y_vec, X=design_frame, mask=mask_img)
adj = Adjacency(vec, matrix_type="directed")
```

Affected:
- `BrainData.__init__` — keyword-only after `data` (covers `Y`, `X`, `mask`, `masker`, `h5_compression`, `verbose`, `resample`, `interpolation`)
- `Adjacency.__init__` — keyword-only after `data` (`Y`, `matrix_type`, `labels`); unused `**kwargs` also dropped
- `DesignMatrix.__init__` — already had the `*` marker
- `DesignMatrix.append` — keyword-only after `dm`
- `Adjacency.bootstrap` — keyword-only after `stat`
- `BrainData.predict` — keyword-only after the required positionals
- The seven public inference entry points — `one_sample_permutation_test`, `two_sample_permutation_test`, `correlation_permutation_test`, `matrix_permutation_test`, `timeseries_correlation_permutation_test`, `isc_permutation_test`, `isc_group_permutation_test` — keyword-only after the leading data arguments: `one_sample_permutation_test(data, 5000)` becomes `one_sample_permutation_test(data, n_permute=5000)`
- The remaining public functions the convention sweep caught — `SRM.__init__` / `DetSRM.__init__`, `KFoldStratified.__init__` (matching sklearn's own `KFold(n_splits, *, ...)` shape), `plot_mean_label_distance`, `plot_between_label_distance`, and `plot_interactive_brain`: `KFoldStratified(5, True)` becomes `KFoldStratified(5, shuffle=True)`
- `SphereNeighborhoods.iter_neighborhoods` — `progress_bar` is keyword-only

The `*` marker prevents classes of bug that the old implicit-positional API allowed — e.g. `Adjacency(data, "directed")` used to silently bind `"directed"` to the `Y` parameter, and a parameter inserted mid-signature in the inference layer once silently shifted `single_feature` into `progress_bar` with no error of any kind.

### Canonical trailing-kwarg order

The trailing kwargs on facade methods are now consistently ordered:

```
..., <domain kwargs>, <return_flags>, n_jobs=-1, random_state=None, progress_bar=False
```

This is a **position-only** break — callers passing these as keywords are unaffected. If you were passing them positionally, update to keyword arguments (recommended regardless). Affected signatures:

- `BrainData.bootstrap` — `percentiles`, `X_test` now precede `n_jobs`/`random_state`
- `BrainData.fit` — `progress_bar` now trails `scale`/`scale_value`
- `Adjacency.bootstrap` — `percentiles` now precedes `n_jobs`/`random_state`
- `Adjacency.plot_mds` — `n_jobs` moved to the end (after `ax`)

---

## Plotting: `plot_*` naming convention

**Status**: ⚠️ **BREAKING** (v0.6.0) — module-level plotting functions were renamed to a consistent `plot_*` prefix. Class facade `.plot()` methods are unchanged except `DesignMatrix.heatmap` → `DesignMatrix.plot`.

| Old name | New name |
|---|---|
| `surface_plot` | `plot_surf` |
| `dist_from_hyperplane_plot` | `plot_dist_from_hyperplane` |
| `scatterplot` | `plot_scatter` |
| `probability_plot` | `plot_probability` |
| `roc_plot` | `plot_roc` |
| `nltools.data.adjacency.plotting.plot` (module-level fn) | `plot_adjacency` |
| `nltools.data.designmatrix.io.heatmap` | `plot_designmatrix` |
| `DesignMatrix.heatmap()` (method) | `DesignMatrix.plot()` |
| `nltools.data.braindata.plotting.plot_matplotlib` | `_plot_matplotlib` (now internal — no longer re-exported from the package root) |

```python
# OLD
from nltools.plotting import surface_plot, scatterplot, roc_plot
dm.heatmap()

# NEW
from nltools.plotting import plot_surf, plot_scatter, plot_roc
dm.plot()
```

---

## Removed attributes and modules (other)

**`BrainData.nifti_masker` and `Simulator.nifti_masker`** — the stored `NiftiMasker` wrapper only held a mask image (no standardize/detrend/smoothing/confounds), so `transform` / `inverse_transform` were equivalent to `nilearn.masking.apply_mask` / `unmask` against the stored mask. The attribute is gone; use the functional API directly:

```python
# OLD
vec = brain.nifti_masker.transform(img)
img_out = brain.nifti_masker.inverse_transform(vec)

# NEW
from nilearn.masking import apply_mask, unmask
vec = apply_mask(img, brain.mask)
img_out = unmask(vec, brain.mask)
```

**`nltools.prefs` module** — replaced by `nltools.templates`. The old stateful template singleton is gone; use the functional config API instead.

```python
# OLD
# Stateful configuration through nltools.prefs

# NEW
import nltools
nltools.set_brainspace(template="default", resolution=3)

# Or scope a change to a block
with nltools.with_brainspace(template="nilearn", resolution=2):
    brain = BrainData("img.nii.gz")

# Inspect current state
print(nltools.get_brainspace())
```

Also: `match_resolution()` now returns a frozen `TemplateMatch` dataclass (attribute access: `.template`, `.resolution`, `.mask_path`, …) rather than a dict. Callers using `result["template"]` need to switch to `result.template`.

**Neurovault download shims** — the deprecated `get_collection_image_metadata` and `download_collection` functions were removed. Use `fetch_neurovault_collection` directly.

**Plotting helper re-exports** — `_plot_matplotlib` and other underscore-prefixed plotting helpers are no longer re-exported from `nltools.plotting` / `nltools`. Import them from their actual module if you really need them (internal use only).

(loading-brain-images)=
### Loading canonical brain images

For atlases, parcellations, ROI masks, and templates, prefer
`fetch_resource` from `nltools.templates` over hard-coded external URLs.
Files live in the `nltools/niftis` HF dataset, are cached locally on first
use, and the same returned path drops straight into anything that takes a
NIfTI path — nilearn plotting/masking helpers, `nibabel.load`, and
`BrainData(path)`.

```python
from nltools.templates import fetch_resource, list_resources
from nltools.data import BrainData
from nilearn import plotting

# Discover what's available (one HF API hit per session, cached)
list_resources(prefix="masks/")
# → ['masks/desikan_killiany_mni152nlin6_1mm.nii.gz',
#    'masks/k50_2mm.nii.gz', 'masks/shen_268_2mm.nii.gz', ...]

# Path-string return — works for both consumers without conversion
plotting.plot_roi(fetch_resource("masks/shen_268_2mm.nii.gz"))   # nilearn
mask = BrainData(fetch_resource("masks/k50_2mm.nii.gz"))         # nltools
```

Avoid the v0.5.1-era `BrainData('https://...nii.gz').to_nifti()` round-trip
when the goal is just to feed a remote NIfTI to nilearn — it parses the
file into nltools' internal masked NumPy array and immediately reverses the
process. `fetch_resource(...)` returns a path nilearn accepts directly.

---

## Legacy HDF5 compatibility (restored)

**Status**: ✅ Round-trip support for v0.5.1-and-earlier HDF5 files restored (2026-04-20) after being briefly dropped earlier in v0.6.0 development.

`BrainData` and `Adjacency` files written by older deepdish/PyTables-backed nltools can be loaded directly without re-saving:

```python
brain = BrainData("old_nltools_0.5.1_file.h5")    # works, no migration step needed
adj = Adjacency("old_adjacency_aug2019_vintage.h5", matrix_type="similarity")
```

The reader uses `h5py` + `hdf5plugin` (no PyTables dependency) and handles:
- PyTables-encoded empty lists (groups with `TITLE='list:N'`)
- Missing `mask_file_name` (common in older files)
- Pre-`matrix_type`-field Adjacency files (Aug 2019 vintage) — if you hit a warned default of `'distance_flat'`, pass `matrix_type=` explicitly. Legacy files always store long-form vectors, so user-supplied names are normalized to `*_flat`.

---

## Breaking Changes Summary

| Component | Change | Old API | New API | Migration Path |
|-----------|--------|---------|---------|----------------|
| `BrainData` | Method removed | `BrainData.regress()` | `BrainData.fit(model='glm', X=...)` | Update BrainData call sites; standalone `nltools.algorithms.regress(X, Y)` remains available |
| `stats.py` | Function removed | `correlation()` | `correlation_permutation_test()` | Import from `inference` module |
| `stats.py` | Function removed | `pearson()` | `scipy.stats.pearsonr` | Use scipy or inference module |
| `stats.py` | Function removed | Unsuffixed one-sample permutation wrapper | `one_sample_permutation_test()` | Import from `nltools.algorithms` |
| `stats.py` | Function removed | Unsuffixed two-sample permutation wrapper | `two_sample_permutation_test()` | Import from `nltools.algorithms` |
| `DesignMatrix` | Backend and dataframe return contracts changed | pandas | Polars | Convert pandas append inputs with `DesignMatrix`; eager frame operations return `DesignMatrix` with operation-specific metadata |
| `BrainData.fit()` | New parameter | `fit()` mutates | `fit(inplace=False)` returns Fit | Optional migration |
| `BrainData.predict()` | API + return type changed | `algorithm=`, `cv_dict=`, dict return | `estimator=`, `cv=`, `Predict` dataclass return (`.weight_map`, `.scores`, `.predictions`, …) | Update keywords; `result['weight_map']` → `result.weight_map`. Fluent `.cv().predict()` removed — pass `estimator=Pipeline(...)` for custom transforms |
| `BrainData.decompose()` | Kwarg renamed | `algorithm='ica'` | `method='ica'` | Update keyword (see Algorithm/variant choice row above) |
| Import paths | Module moved | `stats.isc()` | `nltools.algorithms.isc()` (or the `isc_permutation_test()` engine) | Update the import — `nltools.stats` is gone; the permutation `*_test` exports **are** the engine functions, with no wrapper layer |
| Return keys | Unified | `null_distribution` result key | `null_dist` everywhere (engines, `isc`/`isc_group`) | Update key lookups to `null_dist` |

---

## New Features

### Adjacency input, shape, and result contracts

Adjacency now rejects invalid flat lengths, asymmetric matrices declared distance or
similarity, and ambiguous rectangular arrays. Use an explicit flat type for stacks:

```python
adj = Adjacency(vectors, matrix_type="distance_flat")  # (matrices, edges)
single = adj[0]       # shape (nodes, nodes)
stack = adj[[0]]      # shape (1, nodes, nodes), including singleton selections
empty = adj[[]]       # shape (0, nodes, nodes), with the original matrix type
```

Lists of matrices also remain stacks. `None`, `[]`, and a square `(0, 0)` array
are empty; a zero-length symmetric vector represents one node. Symmetric matrices
store only off-diagonal edges and reconstruct with a zero diagonal, including
similarity matrices. Input diagonals are discarded.

Supply shared node labels as a flat list of length `n_nodes`, or per-matrix labels
as a nested `(n_matrices, n_nodes)` grid. `Y` must have one row per matrix.
Append requires matching node counts and matrix types; both inputs must supply
node labels or neither may supply them. Arithmetic requires the same shape,
matrix type, and labeled node order, without broadcasting. Construction, copying,
selection, transformations, and square exports return independent mutable state.

Regression results now follow the axis being estimated. With `DesignMatrix`
predictors, `beta`, `sigma`, `t`, and `p` are coefficient maps with one matrix per
predictor; one predictor returns a single matrix. `sigma` means coefficient
standard error. Residuals retain the observation stack and its `Y`; coefficient
maps clear observation metadata. With `Adjacency` predictors, observations are
edges, the response must be a single matrix, and coefficient fields are native
predictor arrays or scalars. Only residuals are Adjacency. In both cases `df` is
a scalar. Replace accesses such as `result["df"].data` with `result["df"]`, and
for Adjacency predictors use `result["beta"]` directly.

Bootstrap aggregate maps now use single-matrix shape and common node labels.
The broader bootstrap-result and t-test API changes remain separate decisions.
HDF5 preserves matrix kind, single/stack shape, labels, and `Y`; legacy files
remain readable. CSV stores values only: supply a flat type on load, retain
metadata separately, and do not rely on CSV to preserve singleton versus stack
rank for single-column files. Square CSV export supports single matrices only.

### Spatial RSA with explicit mapping

`BrainData.distance` with `spatial_scale="roi"` or `"searchlight"` returns ordinary
Adjacency stacks. `SpatialScale`, `Adjacency.spatial_scale`, `to_brain`, and the
`project` option on `similarity` have been removed without aliases. Keep the atlas
or source mask and mapping order outside Adjacency.

ROI stacks follow sorted nonzero atlas labels present inside the source mask
after nearest-neighbor resampling. Align once and use that same atlas for both
distance and projection:

```python
from nilearn.image import resample_to_img
from nilearn.masking import apply_mask
from nltools.mask import roi_to_brain_from_atlas

aligned_atlas = resample_to_img(
    atlas, brain.mask, interpolation="nearest", force_resample=True, copy_header=True,
)
roi_labels = np.unique(apply_mask(aligned_atlas, brain.mask).astype(int))
roi_labels = roi_labels[roi_labels != 0]
rdms = brain.distance(metric="correlation", spatial_scale="roi", roi_mask=aligned_atlas)
scores = rdms.similarity(model_rdm, metric="spearman", method=None)
brain_map = roi_to_brain_from_atlas(
    np.array([score["correlation"] for score in scores]),
    atlas=aligned_atlas, source_mask=brain.mask, roi_labels=roi_labels,
)
```

When selecting matrices, subset `roi_labels` by the same indices before painting.
Searchlight stacks follow source-mask voxel order; map a complete per-center
vector with `nilearn.masking.unmask(values, brain.mask)`. For a subset, place the
values back at their selected positions in a full mask-length vector before
calling `unmask`.

`BrainData.align(spatial_scale="roi", roi_mask=...)` and
`BrainData.{mean,std,median}(spatial_scale="roi", roi_mask=...)` retain their
per-parcel alignment and reduction behavior.

### Compute Contrasts

```python
# After fitting GLM
brain_data.fit(model='glm', X=design_matrix)

# Compute contrasts
contrast = brain_data.compute_contrasts("conditionA - conditionB")

# Multiple contrasts
contrasts = brain_data.compute_contrasts({
    "main_effect": "conditionA - conditionB",
    "interaction": [1, -1, -1, 1]
})

# Inference: every statistic for one contrast, in one record
result = brain_data.compute_contrasts("conditionA - conditionB", inference=True)
result.statistic, result.p_value
```

The default returns the contrast *effect*, which is what a second-level model
consumes. Ask for `inference=True` when you want the first-level statistics.

### `ContrastResult` for contrast inference

`nltools.models.ContrastResult` is the frozen record for inferential contrast
results: `effect`, `variance`, `standard_error`, `statistic`, `z_score`,
`p_value`, and `degrees_of_freedom` in one object instead of separate maps. It
is generic over its payload — floats or arrays for a `Glm`, `BrainData` maps for
the facade — its fields cannot be rebound, and each result owns its arrays. Its
p-values are one-sided: negate the contrast to test the other direction.

### Automatic Alpha Selection

```python
# Ridge regression with automatic per-voxel alpha selection
brain_data.fit(
    model='ridge',
    ridge_alpha=[0.1, 1.0, 10.0, 100.0],
    ridge_cv=5,
    X=features
)

# Access the selected alpha and its cross-validated selection score
best_alpha = brain_data.model_.alpha_
selection_scores = brain_data.model_.cv_scores_
```

(braincollection)=
### BrainCollection is deferred to 0.6.1

`BrainCollection`, `PredictCollection`, collection execution and fit/predict
bundles are absent from 0.6.0. The development-only `Predict.permutation_scores`
and `Predict.permutation_pvalue` fields are removed with their collection producer.
The [collection specification](development/specs/braincollection.md) and
[execution design](development/execution-model.md) preserve the deferred work.

For 0.6.0, apply `BrainData` methods to each subject and concatenate the resulting
maps for group analysis, as shown in the [GLM workflow](tutorials/workflows/01_glm.md).
BrainData decoding, ROI/searchlight analyses, Glm/Ridge, alignment, GPU inference,
and shared BrainData/DesignMatrix/Adjacency HDF5 persistence remain supported.
NeuroVault dataset collections are unrelated and remain available.

### Niimg-like inputs in analysis functions

**Status**: ✅ NEW (v0.6.0) — additive, no migration required

Analysis entry points that take a "brain-like" argument — `similarity`, `multivariate_similarity`, `apply_mask`, `extract_roi`, `forecast` — now accept anything `BrainData(...)` accepts. This matches nilearn's Niimg-like convention at the API boundary.

Accepted inputs: `BrainData`, `nib.Nifti1Image`, file path (`str` / `Path`), list of paths, URL, `.h5`.

```python
# Previously: only BrainData or Nifti1Image worked
sim = brain_data.similarity(other_brain_data)
sim = brain_data.similarity(nib.load("image.nii.gz"))

# Now: file paths, Path objects, and lists also work
sim = brain_data.similarity("image.nii.gz")
sim = brain_data.similarity(Path("image.nii.gz"))
roi = brain_data.extract_roi(mask="atlas.nii.gz")
```

Unsupported types now raise `TypeError` (with a clearer message) instead of the previous generic `ValueError("Make sure data is a BrainData instance.")`.

---

## Compatibility & Warnings

### Backward Compatibility

| Feature | Status | Action Required |
|---------|--------|-----------------|
| HDF5 files from v0.5.1 (deepdish/PyTables) | ✅ Fully compatible (read path restored via h5py + hdf5plugin; no PyTables dependency) | None |
| `BrainData.regress()` | ❌ Removed | Use `.fit(model='glm', X=...)` |
| `.predict()` | ⚠️ API + return type changed | Update `algorithm=` → `estimator=` and `cv_dict=` → `cv=`; `radius=` keeps its name. Result is a `Predict` dataclass — replace `result['weight_map']` with `result.weight_map`. Fluent `brain.cv(...).predict(...)` removed. |
| `.decompose()` | ⚠️ Kwargs changed | Update `algorithm=` → `method=`; signature is now keyword-only after `self` |
| `BrainData.ttest()` | ⚠️ Signature changed | Old `threshold_dict=` kwarg gone; use `popmean=`, `permutation=`, `tail=`, `n_permute=` |
| `.X` and `.Y` attributes | ✅ Still work | Prefer passing `X=` to `.fit()` directly |
| Old empty-state attribute | ❌ Removed | Use `.is_empty` instead |
| `.smooth()` return value | ⚠️ Changed behavior | Assign to new variable |
| `BrainData.nifti_masker` | ❌ Removed | Use `nilearn.masking.apply_mask(img, bd.mask)` / `unmask(vec, bd.mask)` |
| `nltools.prefs` module | ❌ Removed | Import from `nltools.templates`; use `set_brainspace()` / `with_brainspace()` |

### Deprecation Timeline

| Feature | v0.6.0 Status | v0.7.0 Status |
|---------|---------------|---------------|
| `BrainData.regress()` | ❌ Removed | ❌ Removed |
| Old empty-state attribute | ❌ Removed (use `.is_empty`) | ❌ Removed |
| `.X` and `.Y` | Still works | ⚠️ May be deprecated |
| In-place `.smooth()` | Changed (returns copy) | N/A |
| Legacy data-facade kwarg aliases (`algorithm=`, `show_progress=`, `radius=`, `n_perm=`, `thr_upper=`, `thr_lower=`, `kind=`, `ignore_diagonal=`) | ❌ Removed — no aliases kept | N/A |

---

## Testing Your Migration

### Step 1: Replace Removed Methods
```python
# BrainData.regress() was removed; it does not emit a deprecation warning.
brain_data.fit(model='glm', X=design_matrix)
```

### Step 2: Update Predict API
```python
# OLD (v0.5.1)
results = brain_data.predict(algorithm='svm', cv_dict={'type': 'kfolds', 'n_folds': 5}, radius=10)
weight_map = results['weight_map']

# NEW (v0.6.0) — updated keyword names + Predict dataclass return
# `spatial_scale` selects the prediction mode (whole_brain / roi / searchlight);
# `estimator` selects the sklearn estimator, by shortcut name or as an object.
result = brain_data.predict(
    y=labels, spatial_scale='whole_brain', estimator='linear_svc', cv=5
)
result.weight_map        # full-data refit coefficients (BrainData)
result.estimator         # fitted full-data sklearn estimator
result.fold_weight_maps  # per-fold coefficients
result.scores            # per-fold scores
result.mean_score        # mean across folds

# Searchlight — populates accuracy_map (no weight_map; per-sphere classifiers)
result = brain_data.predict(y=labels, spatial_scale='searchlight',
                            estimator='ridge_classifier', radius=10, cv=5)
result.accuracy_map      # voxel-shaped accuracy

# Note: 'ridge' is regression-only; for classification use 'ridge_classifier'.
# scoring=None (default) uses the estimator's own score method.

# Custom preprocessing chain — pass a sklearn Pipeline as estimator=, and it
# is used exactly as given.
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.svm import LinearSVC
pipe = make_pipeline(StandardScaler(), SelectKBest(k=500), LinearSVC())
result = brain_data.predict(y=labels, estimator=pipe)
```

The fluent API `brain.cv(k=5).normalize().reduce().pipe(t).predict(y, algorithm=…)` has been **removed** from `BrainData`. All four steps fold into kwargs on `bd.predict()` (`cv=`, `estimator=`), with any scaling or reduction expressed as a scikit-learn `Pipeline` passed to `estimator=`. The standalone `nltools.pipelines.Pipeline` API was likewise removed in v0.6.0. Collection orchestration is deferred to 0.6.1; use explicit per-subject calls for multi-subject workflows.

### Step 3: Replace Removed Empty-State Access
```python
is_empty = brain_data.is_empty
```

### Step 4: Update Properties
```python
# Search your codebase for:
# - .shape()
# - the old empty-state method or attribute
# - .dtype()

# Replace with:
# - .shape
# - .is_empty
# - .dtype
```

---

## Migration Checklist

### Must fix (will crash)

- [ ] Rename `Brain_Data` → `BrainData`, `Design_Matrix` → `DesignMatrix` everywhere
- [ ] Replace `onsets_to_dm(events_path, run_length=N, sampling_freq=sf, hrf_model='glover')` → `DesignMatrix(events_path, run_length=N, TR=1/sf)` (HRF-convolved by default; pass `hrf_model=None` for boxcar). For in-memory DataFrames use `events_to_dm(...)` from `nltools.data.designmatrix.io` (always boxcar). The `from nltools.file_reader import ...` / `from nltools.io import onsets_to_dm` paths are both removed.
- [ ] Rename `dm.polys` → `dm.confounds` (attribute), `polys=` → `confounds=` (constructor kwarg), `exclude_polys=` → `exclude_confounds=` (on `.vif()` / `.clean()`)
- [ ] Replace any direct `dm.convolved = …` / `dm.confounds = …` assignments with the constructor kwargs (`convolved=`, `confounds=`) or with `.append(other, axis=1)` — the attributes are now read-only properties. See [DesignMatrix .convolved / .confounds are read-only](#designmatrix-confounds-readonly).
- [ ] Replace pandas concatenation with `dm.append(...)`. Pass raw Polars frames directly; convert pandas frames with `DesignMatrix(frame, sampling_freq=dm.sampling_freq, confounds=list(frame.columns))` first.
- [ ] Update column lookups after `.convolve()`: `dm_conv["stim"]` → `dm_conv["stim_c0"]`. Includes `compute_contrasts("A - B")` strings → `compute_contrasts("A_c0 - B_c0")`. See [DesignMatrix.convolve() always suffixes](#designmatrix-convolve-suffix).
- [ ] Stop introducing value-identical columns via `append(axis=1)` — value-identical columns now raise `ValueError`; drop or modify one copy before appending. See [append(axis=1) refuses value-identical columns](#append-duplicate-columns).
- [ ] Update `from nltools.external import glover_hrf` → `from nltools.algorithms.hrf import glover_hrf`
- [ ] Update `from nltools.simulator import ...` → `from nltools import ...` or `from nltools.data import ...`
- [ ] Replace stateful `nltools.prefs` template configuration with `set_brainspace()` / `get_brainspace()` / `with_brainspace()`
- [ ] Remove `from nltools.utils import get_anatomical` — use `nilearn.datasets.load_mni152_template()`
- [ ] Replace `BrainData.regress()` with `BrainData.fit(model='glm', X=...)`; keep standalone OLS call sites on `nltools.algorithms.regress(X, Y)` if you use that helper
- [ ] Drop the `threshold_dict=` kwarg on `BrainData.ttest()` (signature changed — now `popmean=`, `permutation=`, `tail=`, `n_permute=`)
- [ ] Replace `brain.nifti_masker.transform(img)` → `nilearn.masking.apply_mask(img, brain.mask)` (same for `inverse_transform` → `unmask`)
- [ ] Replace `download_collection` / `get_collection_image_metadata` → `fetch_neurovault_collection`
- [ ] Rename any kwargs still using legacy spellings: `algorithm=` → `method=`/`model=`, `show_progress=` → `progress_bar=`, `radius=` → `radius_mm=`, `n_perm=` → `n_permute=`, `ignore_diagonal=True` → `include_diag=False`, `thr_upper=`/`thr_lower=` → `upper=`/`lower=`, `kind=` → `method=` (see "v0.6.0 Kwarg Standardization" below)
- [ ] Rename any positional-kwarg calls to `__init__`: implemented `BrainData`/`Adjacency`/`DesignMatrix` constructors now require keyword arguments after the first positional data arg
- [ ] Rename module-level plotting callers: `surface_plot` → `plot_surf`, `scatterplot` → `plot_scatter`, `roc_plot` → `plot_roc`, `probability_plot` → `plot_probability`, `dist_from_hyperplane_plot` → `plot_dist_from_hyperplane`, `adjacency.plot(...)` (module fn) → `plot_adjacency`, `DesignMatrix.heatmap()` → `DesignMatrix.plot()`

### Should fix (deprecated or changed behavior)

- [ ] Update `.predict(algorithm=...)` to `.predict(spatial_scale=..., estimator=..., cv=...)` (new keyword API; `spatial_scale=` chooses ``'whole_brain'``/``'roi'``/``'searchlight'``)
- [ ] Update `.decompose(algorithm=...)` to `.decompose(method=...)` (same `algorithm → method` rename; signature is now keyword-only after `self`)
- [ ] Update `.shape()` → `.shape`, old empty-state access → `.is_empty`, and `.dtype()` → `.dtype`
- [ ] Update `.smooth()` to assign return value (returns copy now)
- [ ] Replace `summarize_bootstrap()` with `BrainData.bootstrap()` or `OnlineBootstrapStats`
- [ ] Remove any `DesignMatrix.reset_index()` calls (pandas-compat no-op; removed)
- [ ] `DesignMatrix.add_dct_basis()` now adds a `.nl_cosine_0` constant column by default (parity with `add_poly(0)` → `.nl_poly_0`). If you were chaining `.add_poly(0)` after `.add_dct_basis()` and relied on no intercept from the DCT call, drop the now-redundant `add_poly(0)` or pass `include_constant=False` to restore the old SPM-style (no-constant) behaviour.

### Optional (new features to consider)

- [ ] Consider using new `.fit(model='ridge')` for regression
- [ ] Consider using new CV features (`cv=5`, `alpha='auto'`)
- [ ] Migrate `isc()`, `isc_group()` to `isc_permutation_test()`, `isc_group_permutation_test()` (optional — `isc()` / `isc_group()` remain available from `nltools.algorithms`, with the bootstrap `n_samples=` vocabulary)
- [ ] Replace `stats.correlation()` with `correlation_permutation_test()` from inference module
- [ ] Replace `stats.pearson()` with `scipy.stats.pearsonr` or `correlation_permutation_test()`
- [ ] Consider using `fit(inplace=False)` for immutable results and serialization
- [ ] Consider using `SRM` / `DetSRM` for shared response modeling (new in v0.6.0)
- [ ] Test with `DeprecationWarning` filters to catch remaining issues

---

(new-feature-gpu-accelerated-statistical-inference)=
## New Feature: GPU-Accelerated Statistical Inference

**Status**: ✅ NEW (v0.6.0)

nltools v0.6.0 introduces a comprehensive GPU-accelerated inference module for permutation testing and bootstrap resampling, providing **10-100× speedup** over CPU-only implementations.

### Overview

**New module**: `nltools.algorithms.inference`
- **Focused submodules**: one_sample, two_sample, correlation, timeseries, matrix, isc, intersubject, bootstrap, utils, validation
- **Deterministic across backends**: same seed → identical results on CPU serial, CPU parallel, and GPU
- **GPU-optional**: Works on CPU-only systems with parallel speedup (4-8×)
- **The public API**: these engine functions are exactly what `nltools.algorithms` exports

### Available Functions

| Function | Description | Performance |
|----------|-------------|-------------|
| `one_sample_permutation_test()` | Sign-flipping test (mean ≠ 0) | 10-100× GPU, 4-8× CPU-parallel |
| `two_sample_permutation_test()` | Group comparison (mean₁ ≠ mean₂) | 10-100× GPU, 4-8× CPU-parallel |
| `correlation_permutation_test()` | Correlation significance (Pearson/Spearman/Kendall) | 10-100× GPU, 4-8× CPU-parallel |
| `timeseries_correlation_permutation_test()` | Time-series correlation (preserves autocorrelation) | GPU-batched, 4-8× CPU-parallel |
| `matrix_permutation_test()` | Mantel test for matrix correlation | 6× CPU-parallel |
| `isc_permutation_test()` | Intersubject correlation (LOO/Pairwise) | 15-30× GPU, 4-8× CPU-parallel |
| `circle_shift()` | Circular rotation for time series | - |
| `phase_randomize()` | FFT-based phase shuffling | - |

### Migration from nltools.stats

The old unsuffixed permutation wrappers are removed, and so is `nltools.stats` itself. Add the `_test` suffix and import the resulting names from `nltools.algorithms` (or `nltools.algorithms.inference` — same functions).

**New API** (nltools.algorithms.inference):
```python
from nltools.algorithms.inference import (
    one_sample_permutation_test,
    two_sample_permutation_test,
    correlation_permutation_test,
    matrix_permutation_test,
    isc_permutation_test
)

# One-sample test with GPU acceleration
result = one_sample_permutation_test(
    data,
    n_permute=5000,
    device='gpu',
    random_state=42
)

# Two-sample test
result = two_sample_permutation_test(
    data1, data2,
    n_permute=5000,
    tail=2,  # 2 | 'two' (two-tailed) or 1 | 'one' (one-tailed) — see the tail vocabulary section
    device='gpu'
)

# Correlation test with multiple metrics
result = correlation_permutation_test(
    x, y,
    n_permute=5000,
    metric='spearman',  # 'pearson', 'spearman', or 'kendall'
    device='gpu'
)

# Matrix permutation with extraction modes
result = matrix_permutation_test(
    matrix1, matrix2,
    n_permute=5000,
    how='upper',  # 'upper', 'lower', or 'full'
    metric='pearson'
)

# NEW: Intersubject correlation (ISC)
result = isc_permutation_test(
    data,  # (n_observations, n_subjects) or (n_obs, n_subjects, n_voxels)
    n_permute=5000,
    summary_statistic='pairwise',  # 'pairwise' or 'leave-one-out'
    method='bootstrap',  # 'bootstrap', 'circle_shift', or 'phase_randomize'
    device='gpu'
)
```

### New Features

**1. Time-Series Correlation Tests**
```python
from nltools.algorithms.inference import (
    timeseries_correlation_permutation_test,
    circle_shift,
    phase_randomize
)

# Standard permutation BREAKS autocorrelation (inflates Type I error)
# Use time-series-preserving methods instead:

# Circle shift: Preserves autocorrelation
result = timeseries_correlation_permutation_test(
    x, y,
    n_permute=5000,
    method='circle_shift'
)

# Phase randomize: Preserves power spectrum
result = timeseries_correlation_permutation_test(
    x, y,
    n_permute=5000,
    method='phase_randomize'
)

# Or use the functions directly:
shifted = circle_shift(timeseries, random_state=42)
randomized = phase_randomize(timeseries, random_state=42)
```

**2. Intersubject Correlation (ISC)**
```python
from nltools.algorithms.inference import isc_permutation_test

# Single-feature ISC
data = np.random.randn(100, 20)  # (n_observations, n_subjects)
result = isc_permutation_test(data, n_permute=5000)

# Voxel-wise ISC with GPU
data = np.random.randn(100, 50, 5000)  # (n_obs, n_subjects, n_voxels)
result = isc_permutation_test(
    data,
    n_permute=5000,
    summary_statistic='leave-one-out',  # or 'pairwise'
    method='bootstrap',
    device='gpu'
)

# Direct inference returns:
# - isc: Observed ISC
# - p: P-values
# - null_dist: Null ISC values (if return_null=True)
```

**3. Parallel Options**
```python
# CPU-parallel (default, memory-efficient)
result = one_sample_permutation_test(data, device='cpu')

# GPU-batched (10-100× faster for large problems)
result = one_sample_permutation_test(data, device='gpu')

# Serial execution
result = one_sample_permutation_test(data, device=None)
```

### Key Improvements

**Performance**:
- **GPU acceleration**: 10-100× speedup with PyTorch backend
- **CPU parallelization**: 4-8× speedup with joblib (default)
- **Automatic batching**: Prevents GPU out-of-memory errors
- **Progress bars**: opt-in via `progress_bar=True` for long-running tests (off by default — see [](#inference-progress-bar-off))

**Correctness**:
- **Perfect determinism**: 0.000% cross-backend variance (same seed → identical results)
- **Validated against literature**: Nichols & Holmes 2002, Chen et al. 2016, Theiler et al. 1992
- **Comprehensive testing**: mathematical correctness verified per test, including CPU/GPU draw identity

**Usability**:
- **Comprehensive error messages**: Clear validation and actionable suggestions
- **Full type hints**: Better IDE support and static analysis
- **Extensive documentation**: `docs/development/inference-internals.md` with algorithms, citations, trade-offs
- **Multiple metrics**: Pearson, Spearman, Kendall for correlation/matrix tests

### Migration Checklist

- [ ] Update unsuffixed permutation function names by adding the `_test` suffix; import them from `nltools.algorithms`
- [ ] Add `device='gpu'` for GPU acceleration (optional)
- [ ] Update `metric` parameter for correlation tests
- [ ] Use `method='circle_shift'` or `method='phase_randomize'` for time series
- [ ] Consider using ISC for multi-subject analyses
- [ ] Test with `random_state` for reproducibility

### Deprecation Timeline

**v0.6.0** (current):
- ✅ New inference module available
- ✅ Suffixed `*_permutation_test` functions (including `isc_permutation_test` / `isc_group_permutation_test`) are exported flat from `nltools.algorithms`, alongside the legacy-vocabulary `isc` / `isc_group` wrappers
- ❌ Unsuffixed permutation wrapper names are removed

Future migration guidance will follow the APIs available in those releases.

---

## Getting Help

- **API Documentation**: Check updated API docs for each class/method
- **Tutorials**: See rewritten tutorials for v0.6.0 patterns
- **GitHub Issues**: Report migration problems or unclear docs

---

*Last updated: 2026-08-31 for nltools v0.6.0*
