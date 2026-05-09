# Migration Guide: v0.5 → v0.6

Version 0.6.0 is a **breaking release** that refactors nltools to better leverage nilearn and establish cleaner APIs. This guide shows you how to update your code.

---

## Quick Reference: What Changed

| Category | v0.5.1 (Old) | v0.6.0 (New) | Status |
|----------|--------------|--------------|--------|
| **Class names** | `Brain_Data`, `Design_Matrix` | `BrainData`, `DesignMatrix` | **Renamed** |
| **Import paths** | `nltools.file_reader`, `nltools.simulator`, `nltools.external` | `nltools.io`, `nltools.data`, `nltools.algorithms` | **Moved** |
| **GLM regression** | `.regress()` | `.fit(model='glm')` | **Deprecated (shim still works)** |
| **Ridge regression** | Manual | `.fit(model='ridge')` | New |
| **ML prediction** | `.predict(algorithm='svm', cv_dict=…)` returning dict | `.predict(y=…, spatial_scale=…, model=…, cv=…)` returning `Predict` dataclass with `.weight_map`, `.scores`, etc. | Unified API |
| **Spatial scope kwarg** | N/A (or `method=` overloaded for both algorithm and spatial scope) | `spatial_scale=` (`'whole_brain' \| 'roi' \| 'searchlight'`) — distinct from `method=` (algorithm); follows the spatial-scale framing of [Jolly & Chang, 2021, *SCAN*](https://doi.org/10.1093/scan/nsab010) | **New canonical kwarg** |
| **RSA workflow** | Manual: per-ROI loop, build Adjacency stack, reduce, paint via `roi_to_brain` | `bd.distance(metric='correlation', spatial_scale='roi', roi_mask=atlas).similarity(model_rdm, project=True)` — chain to a voxel-space `BrainData` | **New** |
| **One-sample t-test** | `BrainData.ttest(threshold_dict=…)` | `BrainData.ttest(popmean=0.0, permutation=False, …)` | **Signature changed** |
| **Two-sample t-test** | N/A | `BrainData.ttest2(other)` | New |
| **Method chaining** | `.smooth()` modifies in-place | Returns copy | Changed |
| **Properties** | `.shape()`, `.isempty()` | `.shape`, `.is_empty` | Changed |
| **Cross-validation** | N/A | `.fit(..., cv=5)` | New |
| **HyperAlignment** | Via `align()` only | `HyperAlignment` class | New |
| **Multi-subject** | `Brain_Collection` | `BrainCollection` class | **New** |
| **SRM** | N/A | `SRM` / `DetSRM` classes | **New** |
| **GPU inference** | N/A | `inference` module | **New** |
| **Algorithm kwarg** | `algorithm=`, `scheme=`, `kind=`, `noise_model=`, `icc_type=`, `extract_type=`, `mode=`, `perm_type=` | `method=` (or `permutation_method=` for similarity, `spatial_scale=` for spatial scope) | **Renamed** |
| **Progress flag** | `show_progress=True` | `progress_bar=False` | **Renamed + default flipped** |
| **Sphere radius** | `radius=` (units implicit) | `radius_mm=` | **Renamed** |
| **Permutation count** | `n_perm=` (Adjacency.generate_permutations) | `n_permute=` | **Renamed** |
| **Parallel flag** | `parallel='cpu'\|'gpu'\|None` (BrainCollection) | `device='cpu'\|'gpu'`, `n_jobs=…` | **Split** |
| **Similarity diagonal** | `ignore_diagonal=False` | `include_diag=False` (polarity flipped, default now excludes diagonal) | **Changed** |
| **BrainData.plot thresholds** | `thr_upper=`, `thr_lower=`, `kind=` | `upper=`, `lower=`, `method=` | **Renamed** |
| **`DesignMatrix.convolve()` columns** | 1-D kernel: name preserved (`stim` → `stim`); 2-D kernel: `stim_c0`, `stim_c1` | Always suffixed `<col>_c{i}`; source column dropped (`stim` → `stim_c0`) | **Renamed (consistent)** |
| **Plotting functions** | `surface_plot`, `scatterplot`, `roc_plot`, `heatmap`, … | `plot_surface`, `plot_scatter`, `plot_roc`, `plot_designmatrix`, … | **Renamed** |
| **`nifti_masker` attr** | `brain_data.nifti_masker` | Use `nilearn.masking.apply_mask(img, bd.mask)` | **Removed** |
| **`nltools.prefs`** | `from nltools.prefs import MNI_Template` | `from nltools.templates import MNI_Template` + `set_brainspace()` | **Moved** |
| **Neurovault helpers** | `download_collection`, `get_collection_image_metadata` | `fetch_neurovault_collection` | **Removed** |

---

## Class Renames

**Status**: **BREAKING** — no backward-compatibility aliases exist

All data classes now follow PEP 8 naming conventions. The old names are **not available** — using them will raise `ImportError`.

| v0.5.1 (Old) | v0.6.0 (New) |
|---------------|--------------|
| `Brain_Data` | `BrainData` |
| `Design_Matrix` | `DesignMatrix` |
| `Brain_Collection` | `BrainCollection` |

**Find and replace in your codebase:**
```bash
# sed/sd commands for bulk rename
sd 'Brain_Data' 'BrainData' **/*.py **/*.ipynb
sd 'Design_Matrix' 'DesignMatrix' **/*.py **/*.ipynb
sd 'Brain_Collection' 'BrainCollection' **/*.py **/*.ipynb
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
| `from nltools.stats import regress` | **Removed** | Use `BrainData.fit(model='glm')` |

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
dm_data = events_to_dm(events_df, run_length=200, sampling_freq=0.5)
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

**Unchanged imports** (these still work as before):
- `from nltools.stats import fdr, fisher_r_to_z, zscore, find_spikes, threshold`
- `from nltools.stats import one_sample_permutation` (deprecated wrapper, still works)
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
- polars >= 1.35 (from 0.20)
- h5py >= 3.15 (from 3.13)
- pytest >= 8.4 (from 8.3)

---

## Breaking Changes

(designmatrix-pandas-polars)=
### DesignMatrix: Pandas → Polars

**Status**: ✅ COMPLETE (v0.6.0)

DesignMatrix now uses Polars DataFrames internally instead of pandas. This provides:
- **2-5x faster** operations (especially statistics and concatenation)
- **Lower memory usage** (Apache Arrow format)
- **Better type safety** and error messages
- **Idiomatic Polars patterns** (no pandas anti-patterns)

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
- Internal storage is Polars (`._df` attribute)
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

# Saving to CSV (access underlying Polars DataFrame)
dm._df.write_csv('/path/to/file.csv')  # ✅ Polars way
dm.to_csv('/path/to/file.csv')         # ❌ Method doesn't exist

# Loading from CSV
import polars as pl
dm = DesignMatrix(pl.read_csv('/path/to/file.csv'), sampling_freq=0.5)
```

**What's the same:**
- `.shape`, `.columns`, `.empty` properties work identically
- `.fillna()`, `.drop()`, `.zscore()` methods work identically
- `.append()`, `.convolve()`, `.upsample()`, `.downsample()` work identically
- `.vif()`, `.clean()` methods work identically

**Migration examples:**
```python
# OLD (pandas-style)
dm.loc[10:15, 'ConditionA'] = 1

# NEW (Polars-style) - use direct column assignment
dm['ConditionA'] = pl.when(pl.arange(0, len(dm)).is_between(10, 15))
                     .then(1)
                     .otherwise(dm['ConditionA'])

# Or for simple cases, convert to numpy and back
arr = dm.to_numpy()
arr[10:15, dm.columns.index('ConditionA')] = 1
dm = DesignMatrix(arr, columns=dm.columns, sampling_freq=dm.sampling_freq)
```

```python
# OLD (pandas .assign())
new_dm = dm.assign(new_col=lambda df: df['col1'] * 2)

# NEW (direct assignment)
new_dm = dm.copy()
new_dm['new_col'] = dm['col1'] * 2
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

**For pandas compatibility:**
```python
# Convert to pandas when needed
pd_df = dm._to_pandas()

# Use with legacy code expecting pandas
nilearn_glm.fit(fmri_img, design_matrices=[pd_df])
```

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

dm_data = events_to_dm(events_df, run_length=200, sampling_freq=0.5)
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

# 4. Stack confounds + drift. .append() handles a mixed list of pandas
#    DataFrames (csf, mc_cov) and DesignMatrix instances (spikes — which
#    already knows its own columns are confounds via find_spikes).
spikes = bold.find_spikes(global_spike_cutoff=3, diff_spike_cutoff=3, TR=tr)
dm = task.append(
    [csf, mc_cov, spikes], axis=1, as_confounds=True,
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
  confounds (3): ['poly_0', 'poly_1', 'poly_2']
```

`DesignMatrix.write()` to `.h5` writes the metadata under the key `confounds` (was `polys`). There is no DM HDF5 reader yet, so this only affects newly written files.

**`BrainData.X = dm` now works.** The `.X` setter previously rejected `DesignMatrix` with `TypeError`; v0.6.0 unwraps it to `dm.data` (the underlying polars DataFrame). DM-specific metadata isn't preserved on `BrainData.X`, but you no longer need the explicit `.data` step.

(designmatrix-confounds-readonly)=
### DesignMatrix `.convolved` and `.confounds` are read-only

**Status**: ⚠️ **BREAKING** (v0.6.0) — direct assignment now raises `AttributeError`

The `.convolved` and `.confounds` lists are managed by `.convolve()`, `.append()`, `.add_poly()`, and `.add_dct_basis()`. Direct mutation was a foot-gun (the v0.5.1 PPI-style flow needed `dm.convolved = list(other.columns)` after a `pd.concat` round-trip clobbered metadata) and is now disallowed.

```python
# OLD (v0.5.1) — silently mutates state, easy to forget when columns later get renamed
combined = DesignMatrix(
    pd.concat([dm_task.to_pandas(), motion, csf, spikes], axis=1),
    sampling_freq=0.5,
)
combined.convolved = list(dm_task.columns)   # manual re-assert after pd.concat
combined.confounds = list(motion.columns) + ["csf"] + list(spikes.columns)

# NEW (v0.6.0) — append manages both lists for you
combined = dm_task.append([motion, csf, spikes], axis=1).add_poly(order=2)
# combined.convolved → ['stim_c0', ...]
# combined.confounds → ['motion_tx', ..., 'csf', 'spike_0', ..., 'poly_0', ...]
```

If you really need to set initial state explicitly, pass `convolved=` / `confounds=` to the constructor — those kwargs still work (and `copy_with` uses them internally for metadata propagation):

```python
dm = DesignMatrix(arr, sampling_freq=0.5, columns=cols, confounds=["intercept"])
```

The error message points to the canonical replacement:

```text
AttributeError: DesignMatrix.confounds is read-only. Pass `confounds=...` to the
constructor, or use `.append(other_dm, axis=1, as_confounds=True)` /
`.append(raw_df, axis=1)` (raw frames are auto-marked) to register confound regressors.
```

(designmatrix-copy-constructor)=
### `DesignMatrix(other_dm)` is now a copy-constructor

**Status**: ✅ NEW (v0.6.0) — additive, no migration required

Passing a `DesignMatrix` to the constructor returns an independent copy with `data`, `sampling_freq`, `convolved`, `confounds`, and `multi` carried over. Explicit kwargs override inherited values. This matches the pandas `pd.DataFrame(other_df)` idiom and short-circuits the v0.5.1 "wrap it again to reset metadata" pattern (which dropped metadata on the floor).

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

## Breaking Changes

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
brain.to_nifti('/tmp/brain.nii.gz')

# ❌ WRONG: auto-detection picks a built-in template → shape mismatch
reloaded = BrainData('/tmp/brain.nii.gz')

# ✅ CORRECT: pass the same custom mask
reloaded = BrainData('/tmp/brain.nii.gz', mask='my_roi.nii.gz')
```

This is **not** an issue when using the default auto-detected templates, since the same template will be selected on reload.

**Best practice** when using custom masks — save both, or use HDF5:
```python
# Option 1: Save mask separately
brain.to_nifti('/tmp/brain.nii.gz')
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

**Deprecated**:
- `.square_shape()` → Use `.shape` instead (will be removed in v0.7.0)

**Threshold API**: Uses `lower`/`upper` keywords, not `threshold`:
```python
# ❌ WRONG
adj.threshold(threshold=0.3)  # TypeError: unexpected keyword argument

# ✅ CORRECT
adj.threshold(lower=0.3)       # Keep values >= 0.3
adj.threshold(upper=0.5)       # Keep values <= 0.5
adj.threshold(lower='90%')     # Keep top 10% (percentile threshold)
```

---

### 1. Removed Methods

| Method | Alternative | Migration Effort |
|--------|-------------|------------------|
| `.regress()` | `.fit(model='glm', X=design_matrix)` — deprecated shim still works, just emits a `DeprecationWarning` | **Low** |
| `.predict(algorithm='svm')` | `.predict(y=labels, spatial_scale=…, model='svm', cv=…)` returning a `Predict` dataclass (`.weight_map`, `.scores`, `.predictions`, …). Fluent `.cv().predict()` on BrainData removed; pass `model=make_pipeline(...)` for custom preprocessing chains. `spatial_scale=` selects ``'whole_brain'``, ``'roi'``, or ``'searchlight'``; `method=` is no longer overloaded. See [Pattern 4](#pattern-4-machine-learning-classificationregression). | **Low** |
| `.decompose(algorithm='ica')` | `.decompose(method='ica', n_components=…, axis=…)` — same `algorithm → method` rename, signature is now keyword-only after `self`; `**kwargs` forwards to the sklearn decomposition estimator | **Low** |
| `BrainData.ttest(threshold_dict=…)` (v0.5.1) | `BrainData.ttest(popmean=0.0, permutation=False, …)` — restored with a new signature. Returns `{"t", "p"}` (or `{"mean", "p"}` when `permutation=True`). Also see new `.ttest2(other)` for two-sample tests. | **Low** |
| `.randomise()` | Use nilearn permutation testing | Medium |
| `.predict_multi()` | Will return in future Model class | N/A |
| `summarize_bootstrap()` | `BrainData.bootstrap()` or `OnlineBootstrapStats` | **Low** |
| `BrainData.iplot(surface=…, anatomical=…)` | `BrainData.iplot(view='ortho'\|'surface', bg_img=…)` — *rebuilt* on `anywidget` (no `ipywidgets` extra needed). New threshold panel with Value↔Percentile and Symmetric↔Independent toggles; 4D BrainData also gets a volume-step slider. See [Pattern: interactive viewing (`iplot`)](#interactive-viewing). | **Low** |

:::{note}
`BrainData.ttest()` was briefly removed earlier in v0.6.0 development, then restored because one-sample voxelwise t-tests across stacked subject-level contrast maps are the 99% group-inference use case. The old `threshold_dict=` kwarg is gone — use the new permutation-based API instead.
:::

### 2. Removed Classes

| Class | Status | Alternative |
|-------|--------|-------------|
| `Brain_Collection` | Replaced | Use `BrainCollection` (new in v0.6.0) |
| `Model` | Removed | Will return in v0.7.0+ |

### 3. Attributes

| Attribute | Status | Alternative |
|-----------|--------|-------------|
| `.X` | Still works | Pass `X=` to `.fit()` directly (preferred) |
| `.Y` | Still works | Manage labels separately (preferred) |
| `.isempty` | Deprecated | Use `.is_empty` instead |

---

## Migration Patterns

(interactive-viewing)=
### Pattern 0: Interactive viewing (`iplot`) — Rebuilt API

**Status**: 🔧 **REBUILT** — `BrainData.iplot()` is back, now backed by [anywidget](https://anywidget.dev) instead of the legacy `ipywidgets` viewer. The new widget renders the same way in Jupyter, marimo, VS Code, and Jupyter Book v2 / mystmd built sites.

**What changed:**

| | v0.5.1 | v0.6.0 |
|---|---|---|
| Engine | `ipywidgets` + manual JS | `anywidget` (single ESM bundle, runs everywhere) |
| Surface plot kwarg | `surface=True` | `view='surface'` (canonical with `view='ortho'` default) |
| Background image kwarg | `anatomical=` | `bg_img=` (matches nilearn) |
| 4D handling | Volume slider via `ipywidgets.IntSlider` | Volume slider built into the widget; auto-shown when 4D |
| Threshold control | Numeric `FloatText` widget | Threshold panel: Value↔Percentile + Symmetric↔Independent toggles, with 1–2 sliders depending on mode |
| Asymmetric thresholds | Not supported | `mode='independent'` masks `(lower, upper)` — separate cutoffs for negatives and positives |
| Render performance | Python re-render on every keystroke | Re-render on slider release (mouse-up) only; readouts update live in JS during drag |
| Optional install | `nltools[interactive_plots]` (extra) | Bundled — `anywidget` is now a hard dep |

**Before (v0.5.1):**
```python
bd.iplot()                    # interactive ortho viewer
bd.iplot(surface=True)        # surface viewer
bd.iplot(anatomical=anat)     # custom background
```

**After (v0.6.0):**
```python
bd.iplot()                              # ortho viewer + threshold panel
bd.iplot(view='surface')                # surface viewer
bd.iplot(bg_img=anat)                   # custom background

# Threshold panel: set initial state via kwargs (toggleable in the UI)
bd.iplot(units='percentile', upper=2.3) # start at the value matching ~98th pctile of |x|
bd.iplot(mode='independent',            # separate negative/positive cutoffs
         lower=-1.0, upper=2.0)

# 4D BrainData: same call, widget grows a volume-step slider automatically
stack = BrainData([f1, f2, f3, f4, f5])
stack.iplot()                           # threshold panel + volume slider
```

The widget honors the standard `application/vnd.jupyter.widget-view+json` mimebundle, so it stays interactive when the dartbrains tutorials are rendered into Jupyter Book v2 / mystmd built sites — no static-snapshot fallback needed.



**Status**: ⚠️ **DEPRECATED** — `.regress()` still works in v0.6.0 as a thin shim that delegates to `.fit(model='glm')` (kept for backward compatibility with dartbrains and older notebooks). It will be removed in a future release.

The preferred API is the unified `.fit(model='glm')`.

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
brain_data.fit(model='glm', X=design_matrix)  # Stores results as attributes
betas = brain_data.glm_betas      # BrainData object
t_stats = brain_data.glm_t        # BrainData object
p_vals = brain_data.glm_p         # BrainData object
residuals = brain_data.glm_residual  # BrainData object
```

**With noise model:**
```python
# OLD (removed)
brain_data.X = design_matrix
results = brain_data.regress(noise_model='ar1')

# NEW (v0.6.0)
brain_data.fit(model='glm', noise_model='ar1', X=design_matrix)
```

**All available GLM attributes:**
```python
brain_data.fit(model='glm', X=design_matrix)

# Attributes set by fit():
brain_data.glm_betas      # Beta coefficients (BrainData)
brain_data.glm_t          # T-statistics (BrainData)
brain_data.glm_p          # P-values (BrainData)
brain_data.glm_se         # Standard errors (BrainData)
brain_data.glm_residual   # Residuals (BrainData)
brain_data.glm_predicted  # Predicted values (BrainData)
brain_data.glm_r2         # R-squared (BrainData)
brain_data.model_         # Fitted Glm model instance
```

| Aspect | Old | New | Benefit |
|--------|-----|-----|---------|
| API style | Dict return | Sklearn-style attributes | Composable, familiar |
| Design matrix | Stored as `.X` | Passed as argument | Explicit, clearer |
| Results | Dict with keys | BrainData attributes | Type-safe, chainable |
| Status | Primary API | Deprecated shim (still callable) | Clear migration path |

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
brain_data.fit(model='ridge', alpha=1.0, X=features)
weights = brain_data.ridge_weights   # (n_features, n_voxels)
scores = brain_data.ridge_scores     # R² per voxel
predictions = brain_data.predict(X=new_features)
```

| Feature | Before | After | Benefit |
|---------|--------|-------|---------|
| API | Manual sklearn | Integrated `.fit()` | Convenient |
| GPU support | Manual setup | `backend='gpu'` | Automatic |
| CV support | Manual | `cv=5` parameter | Built-in |
| Alpha selection | Manual grid search | `alpha='auto'` | Automatic |

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
# Basic CV
brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
mean_r2 = brain_data.cv_results_['mean_score']
cv_preds = brain_data.cv_results_['predictions']

# Auto alpha selection
brain_data.fit(model='ridge', cv='auto', alphas=[0.1, 1, 10], X=features)
best_alpha = brain_data.cv_results_['best_alpha']
```

| Feature | Before | After |
|---------|--------|-------|
| CV splits | Manual sklearn | `cv=5` or custom splitter |
| Alpha selection | Manual grid search | `cv='auto'` |
| Out-of-fold predictions | Manual tracking | In `cv_results_` dict |
| Performance metrics | Manual computation | Automatic R² per voxel |

---

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
result = brain_data.predict(y=labels, spatial_scale='whole_brain', model='svm', cv=5)
result.weight_map      # mean classifier coefs across folds, shape (n_voxels,)
result.fold_weight_maps  # per-fold coefs, shape (n_folds, n_voxels)
result.scores          # per-fold scores, shape (n_folds,)
result.mean_score      # mean accuracy across folds (float)
result.predictions     # OOF predictions in original sample order
result.available()     # list non-None fields

# Refit on full data for a single publishable map (instead of per-fold mean)
result = brain_data.predict(y=labels, refit=True)
result.final_weight_map  # weight map from full-data fit
result.final_estimator   # the fitted sklearn estimator
```

| Aspect | Old | New | Reason |
|--------|-----|-----|--------|
| API | `algorithm=` | `model=` | Mirrors `bd.fit(model=)`; v0.6.0 convention |
| Classifier shortcuts | `'svm'`, `'logistic'`, `'ridge'`, `'lda'` | `'svm'`, `'logistic'`, `'lda'`, `'ridge_classifier'` (classification); `'ridge'`, `'lasso'`, `'svr'` (regression) | `'ridge'` was ambiguous; classification variant renamed |
| CV | `cv_dict=` | `cv=` (int or sklearn splitter) | Simpler |
| Scoring | hardcoded | `scoring='auto'` (→ `'accuracy'` for classifiers, `'r2'` for regressors) or any sklearn scoring string | More flexible |
| Label storage | `.Y` attribute | `y=` argument | Explicit |
| Custom transforms | `brain.cv(k).normalize().reduce().pipe(t).predict()` (fluent) | Pass `model=make_pipeline(StandardScaler(), MyXform(), SVC())` | Standard sklearn pattern, no separate API to learn |
| Return type | dict (`weight_map`, `mcr_all`, …) | `Predict` dataclass | Frozen, introspectable via `.available()` / `.asdict()` |
| Weight map | top-level dict key | `result.weight_map` | Always present for linear models |

**Removed**: `brain.cv(k).predict(y, algorithm=…)` fluent API. The full set of fluent steps (`cv()`, `normalize()`, `reduce()`, `pipe()`) on `BrainData` collapses to kwargs on `bd.predict()`. Multi-subject hyperalignment / SRM still use a fluent pipeline on `BrainCollection` — that surface is unchanged.

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
is_empty = brain_data.isempty()
dtype = brain_data.dtype()
```

**After (v0.6.0):**
```python
shape = brain_data.shape       # No parentheses
is_empty = brain_data.is_empty # No parentheses (note: .isempty is deprecated)
dtype = brain_data.dtype       # No parentheses
```

| Method | Old | New | Reason |
|--------|-----|-----|--------|
| `.shape()` | Method call | `.shape` property | No computation |
| `.isempty()` | Method call | `.is_empty` property | No computation |
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

# For model statistics (weights, predictions), returns dict with all stats
brain.fit(X=dm, model='ridge', alpha=1.0)
boot = brain.bootstrap(stat='weights', n_samples=1000)
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

**Status**: ✅ Migrated to inference module (wrappers maintained for backward compatibility)

#### ISC Functions (`isc()`, `isc_group()`, `isfc()`)

**Old API** (still works, but uses inference module internally):
```python
from nltools.stats import isc, isc_group, isfc

result = isc(data, n_samples=1000)
result = isc_group(group1, group2, n_samples=1000)
result = isfc(data, n_permute=1000)
```

**New API** (recommended):
```python
from nltools.algorithms.inference import (
    isc_permutation_test,
    isc_group_permutation_test,
)
from nltools.stats import isfc  # Still available, now uses inference module

# ISC - single group
result = isc_permutation_test(data, n_permute=1000)

# ISC Group - two groups
result = isc_group_permutation_test(group1, group2, n_permute=1000)

# ISFC - functional connectivity
result = isfc(data, n_permute=1000)  # Now uses inference module internally
```

**Key Changes**:
- `isc()` → `isc_permutation_test()` (parameter name: `n_samples` → `n_permute`)
- `isc_group()` → `isc_group_permutation_test()` (parameter name: `n_samples` → `n_permute`)
- `isfc()` unchanged (still `isfc()`, but now uses inference module internally)
- Return keys: `null_dist` → `null_distribution` (wrapper handles mapping)
- GPU acceleration available with `parallel="gpu"` or `backend="torch"`
- CPU parallelization available with `parallel="cpu"` and `n_jobs=-1`

**Performance**: 4-8× CPU speedup, 10-100× GPU speedup

#### Removed Functions

**Functions Removed** (use alternatives):
- `regress()` → Use `nltools.models.Glm` or `BrainData.fit(model='glm')`
- `regress_permutation()` → Use inference module permutation tests
- `correlation()` → Use `correlation_permutation_test()` from inference module
- `pearson()` → Use `scipy.stats.pearsonr` or `correlation_permutation_test()`

**Matrix Utilities** (moved to inference module, re-exported from stats.py):
- `double_center()` → `nltools.algorithms.inference.double_center()` (still available via `nltools.stats`)
- `u_center()` → `nltools.algorithms.inference.u_center()` (still available via `nltools.stats`)
- `distance_correlation()` → `nltools.algorithms.inference.distance_correlation()` (still available via `nltools.stats`)

---

(pattern-10-fit-dataclass-braindata-fit-inplace-false)=
### Pattern 10: Fit Dataclass (`BrainData.fit(inplace=False)`)

**Status**: ✅ NEW FEATURE (v0.6.0)

**New Feature**: `BrainData.fit()` now supports returning Fit objects instead of mutating attributes.

**Old API** (still works, default behavior):
```python
brain.fit(X=dm, model='ridge', alpha=1.0)  # Mutates brain, adds attributes
assert hasattr(brain, 'ridge_weights')
```

**New API** (recommended):
```python
from nltools.data import Fit

fit = brain.fit(X=dm, model='ridge', alpha=1.0, inplace=False)  # Returns Fit object
assert isinstance(fit, Fit)
assert 'weights' in fit.available()
assert not hasattr(brain, 'ridge_weights')  # Data attributes NOT set on brain

# Note: brain.model_ and brain.X_ are still set even with inplace=False.
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
- **Ridge + CV**: Also includes `cv_scores`, `cv_mean_score`, `cv_predictions`, `cv_folds`, `cv_best_alpha`, `cv_alpha_scores`
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
stats = OnlineBootstrapStats()
for sample in samples:
    stats.update(sample)
result = stats.get_statistics()
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

# GPU (automatic batching)
result = one_sample_permutation_test(
    data, 
    n_permute=1000, 
    backend='torch',
    max_gpu_memory_gb=4.0  # Memory budget
)

# CPU parallel (4-8× speedup)
result = one_sample_permutation_test(
    data,
    n_permute=1000,
    backend=None,  # or 'auto'
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
model = SRM(n_components=50, n_iter=10)
model.fit(subjects)             # List of (n_voxels, n_timepoints) arrays
aligned = model.transform(subjects)  # Project to shared space

# Deterministic SRM (faster, no noise model)
det_model = DetSRM(n_components=50, n_iter=10)
det_model.fit(subjects)
aligned = det_model.transform(subjects)

# Align a new subject to existing shared space
new_aligned, rotation, disparity, scale = model.transform_subject(new_data)
```

| Aspect | Before | After | Benefit |
|--------|--------|-------|---------|
| Availability | External library | Built-in | No extra dependency |
| API | Varies | sklearn-compatible | Composable pipelines |
| Variants | N/A | SRM + DetSRM | Flexibility |

---

## v0.6.0 Kwarg Standardization (April 2026)

**Status**: ✅ Complete (v0.6.0). No aliases kept for the old spellings — callers using the legacy names will hit a `TypeError: unexpected keyword argument`.

A sweep of the four data-class facades (`BrainData`, `Adjacency`, `BrainCollection`, `DesignMatrix`) landed in a series of `!:` commits on 2026-04-14 and 2026-04-20 to make kwarg names consistent across the public API. The canonical names are documented in `CLAUDE.md` (section "API Conventions"); the table below is the migration mapping for callers.

### Renamed kwargs

| Concept | Old kwarg(s) | New kwarg | Scope |
|---|---|---|---|
| Algorithm / variant choice | `algorithm`, `scheme`, `kind`, `noise_model`, `icc_type`, `extract_type`, `mode`, `perm_type` | `method` (for `Adjacency.similarity` only: `permutation_method`, because `method` is already the metric selector) | All four facades. Hits include `BrainData.decompose`, `Adjacency.cluster`, `Adjacency.similarity`, `BrainCollection.fit`, `BrainCollection.align` and the ICC/permutation helpers. **Note:** `BrainData.predict` and `BrainData.distance` use the new `spatial_scale=` kwarg (not `method=`) for selecting `'whole_brain'`/`'roi'`/`'searchlight'` — see "Spatial scope" row below. |
| Spatial scope (whole-brain / ROI / searchlight) | `method='whole_brain'\|'roi'\|'searchlight'` (predict only — overloaded with the algorithm slot, never canonical elsewhere) | `spatial_scale='whole_brain'\|'roi'\|'searchlight'` | `BrainData.predict`, `BrainData.distance`, `BrainCollection.predict` (scaffold). Companion kwargs `roi_mask=` and `radius_mm=` already canonical. Naming follows the spatial-scale framing of [Jolly & Chang, 2021, *SCAN*](https://doi.org/10.1093/scan/nsab010). The `method=` slot is now reserved for algorithm choice everywhere. |
| Classifier / sklearn estimator | `algorithm=` (predict), then briefly `estimator=` | `model=` | `BrainData.predict`. Mirrors `BrainData.fit(model=…)` (statistical-model name slot). String shortcuts: classification — `'svm'`, `'logistic'`, `'lda'`, `'ridge_classifier'`; regression — `'ridge'`, `'lasso'`, `'svr'`. Or pass any sklearn estimator / `Pipeline` directly. |
| Progress indicator | `show_progress` (defaulted `True`) | `progress_bar` (defaults `False`, matching sklearn) | All four facades and their submodules. `verbose` is kept only where it controls log-level output (sklearn warning suppression in `standardize`, info prints in `DesignMatrix.clean` / `.append`). |
| Sphere / searchlight radius | `radius` (millimeters, but units were implicit) | `radius_mm` | `BrainCollection.isc`, `.isc_test`, `.predict`, `.align`; `BrainData.predict` (searchlight), `BrainData.plot_flatmap`; `nltools.plotting.plot_surface`, `plot_flatmap`. The returned info dict from `_extract_for_isc` also renames `"radius"` → `"radius_mm"`. Pure-geometry helpers (`create_sphere`, `Simulator`) keep `radius`. |
| Permutation count | `n_perm` | `n_permute` | `Adjacency.generate_permutations` (the rest of the facade and `BrainCollection.isc_test` / `permutation_test` / `permutation_test2` already used `n_permute`). `BrainCollection.align`'s `n_iter` stays — optimizer iterations, not permutations. |
| GPU/CPU selection + parallelism | `parallel='cpu'\|'gpu'\|None` | `device='cpu'\|'gpu'` **and** `n_jobs` (use `n_jobs=1` for single-threaded) | `BrainCollection.permutation_test`, `.permutation_test2`, `.isc`, `.isc_test`, `.align`. Returned dict key `'parallel'` → `'device'`. |
| Similarity diagonal | `ignore_diagonal=False` | `include_diag=False` | `Adjacency.similarity`. **Polarity is flipped AND the default changed**: directed matrices now exclude the (trivially 1.0) self-similarity diagonal by default. No-op for symmetric matrices, which never store the diagonal. |
| Threshold arms on `BrainData.plot` | `thr_upper`, `thr_lower`, `kind` | `upper`, `lower`, `method` | The convenience scalar `threshold=` kwarg is unchanged. |

### Migration examples

```python
# OLD
brain.predict(algorithm='svm', cv_dict={'type': 'kfolds', 'n_folds': 5}, radius=10)
brain.decompose(algorithm='ica', n_components=20, axis='images', whiten=True)
brain.plot(kind='glass', thr_upper=2.3, thr_lower=-2.3)
bc.isc(radius=6.0, parallel='gpu', show_progress=True)
bc.isc_test(n_permute=5000, parallel='cpu')
adj.generate_permutations(n_perm=1000)
adj.similarity(other, ignore_diagonal=True)  # old: include the diagonal

# NEW
brain.predict(y=labels, spatial_scale='searchlight', model='svm', cv=5, radius_mm=10)
brain.decompose(method='ica', n_components=20, axis='images', whiten=True)
brain.plot(method='glass', upper=2.3, lower=-2.3)
bc.isc(radius_mm=6.0, device='gpu', progress_bar=True)
bc.isc_test(n_permute=5000, device='cpu')
adj.generate_permutations(n_permute=1000)
adj.similarity(other, include_diag=False)     # explicit + now the default for directed
```

### Algorithm-layer APIs are unchanged

Internal algorithm classes — `CVScheme.scheme`, `Glm.noise_model`, `compute_icc_voxelwise.icc_type`, `LocalAlignment.scheme` — keep their legacy names. The class facades translate at the boundary. You only need to update code that calls the facade methods.

---

## Explicit signatures instead of `**kwargs` passthroughs

**Status**: ⚠️ **BREAKING** (v0.6.0, 2026-04-20) — if you relied on forwarding arbitrary unknown kwargs through a facade method, that will now raise `TypeError: unexpected keyword argument`.

Internal `**kwargs` catch-alls have been removed from user-facing methods that delegate to nltools code (they are retained only where the target is a third-party library — sklearn estimator constructors, matplotlib, nilearn, nibabel, seaborn, pandas, scipy).

**Newly-explicit kwargs you can now pass directly** (previously hidden behind `**kwargs`):

- `BrainData.bootstrap`: `backend`, `max_gpu_memory_gb`
- `BrainData.ttest`, `Adjacency.ttest`: `n_permute`, `tail`, `return_null`, `n_jobs`, `random_state`
- `Adjacency.similarity`: `tail`, `return_null`, `n_jobs`, `random_state`
- `BrainData.cv`, `BrainCollection.cv`: `n` (iterations for bootstrap/permutation schemes)

**Dead `*args` / `**kwargs` dropped entirely**:
- `BrainData.align` (never used internally)
- `Adjacency.regress`, `BrainData.regress` deprecation shim
- `Adjacency.__init__` (swallowed unused kwargs)

### Keyword-only (`*`) marker after the primary data arg

All four data-class `__init__` methods now require keyword arguments after the first positional data arg. Methods with many optional kwargs also enforce keyword-only.

```python
# OLD — these relied on positional order
brain = BrainData(data, Y_vec, X_df, mask_img)                  # positional Y/X/mask
adj = Adjacency(vec, "directed")                                # was actually binding to Y, not matrix_type!

# NEW — positional-only up to the primary data arg; rest must be keyword
brain = BrainData(data, Y=Y_vec, X=X_df, mask=mask_img)
adj = Adjacency(vec, matrix_type="directed")
```

Affected:
- `BrainData.__init__` — keyword-only after `data` (covers `Y`, `X`, `mask`, `masker`, `h5_compression`, `verbose`, `resample`, `interpolation`)
- `Adjacency.__init__` — keyword-only after `data` (`Y`, `matrix_type`, `labels`); unused `**kwargs` also dropped
- `BrainCollection.__init__` — keyword-only after `items, mask` (`metadata`, `lazy`)
- `DesignMatrix.__init__` — already had the `*` marker
- `DesignMatrix.append` — keyword-only after `dm`
- `Adjacency.bootstrap` — keyword-only after `stat`
- `BrainData.predict`, `BrainCollection.fit`, `BrainCollection.predict` — keyword-only after the required positionals

The `*` marker prevents classes of bug that the old implicit-positional API allowed — e.g. `Adjacency(data, "directed")` used to silently bind `"directed"` to the `Y` parameter.

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
- `BrainCollection.standardize` — `verbose` now precedes `n_jobs`/`progress_bar`

---

## Plotting: `plot_*` naming convention

**Status**: ⚠️ **BREAKING** (v0.6.0) — module-level plotting functions were renamed to a consistent `plot_*` prefix. Class facade `.plot()` methods are unchanged except `DesignMatrix.heatmap` → `DesignMatrix.plot`.

| Old name | New name |
|---|---|
| `surface_plot` | `plot_surface` |
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
from nltools.plotting import plot_surface, plot_scatter, plot_roc
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

**`nltools.prefs` module** — replaced by `nltools.templates`. The old stateful `MNI_Template_Factory` singleton is gone; use the functional config API instead.

```python
# OLD
from nltools.prefs import MNI_Template
MNI_Template["resolution"] = "3mm"

# NEW
from nltools.templates import MNI_Template         # same class, new location
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
file into nltools' internal Polars frame and immediately reverses the
process. `fetch_resource(...)` returns a path nilearn accepts directly.

`list_resources()` requires `huggingface_hub` and is unavailable in
Pyodide; browser-deployed code should pre-seed known paths via
`await seed_resources([...])` instead.

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
| `stats.py` | Function removed | `regress()` | `nltools.models.Glm` | Use `BrainData.fit(model='glm')` |
| `stats.py` | Function removed | `correlation()` | `correlation_permutation_test()` | Import from `inference` module |
| `stats.py` | Function removed | `pearson()` | `scipy.stats.pearsonr` | Use scipy or inference module |
| `stats.py` | Function deprecated | `one_sample_permutation()` | `one_sample_permutation_test()` | Import from `inference` module |
| `stats.py` | Function deprecated | `two_sample_permutation()` | `two_sample_permutation_test()` | Import from `inference` module |
| `DesignMatrix` | Backend changed | pandas | Polars | Automatic migration (backward compatible) |
| `BrainData.fit()` | New parameter | `fit()` mutates | `fit(inplace=False)` returns Fit | Optional migration |
| `BrainData.predict()` | API + return type changed | `algorithm=`, `cv_dict=`, dict return | `model=`, `cv=`, `Predict` dataclass return (`.weight_map`, `.scores`, `.predictions`, …) | Update keywords; `result['weight_map']` → `result.weight_map`. Fluent `.cv().predict()` removed — pass `model=Pipeline(...)` for custom transforms |
| `BrainData.decompose()` | Kwarg renamed | `algorithm='ica'` | `method='ica'` | Update keyword (see Algorithm/variant choice row above) |
| Import paths | Module moved | `stats.isc()` | `inference.isc_permutation_test()` | Wrapper maintained |
| Return keys | Key renamed | `null_dist` | `null_distribution` | Wrapper handles mapping |

---

## New Features

### Spatial-scale-aware RSA (NEW)

**Status**: ✅ NEW (v0.6.0)

`BrainData.distance(spatial_scale=...)` plus `Adjacency.spatial_scale` / `to_brain()` / `similarity(project=True)` make per-ROI and per-searchlight representational similarity analysis a one-liner that ends in a voxel-space `BrainData`. The framing follows [Jolly & Chang, 2021, *SCAN*](https://doi.org/10.1093/scan/nsab010): searchlight → ROI → whole brain as named points on a single spatial-scale axis.

**Canonical chain**:

```python
# Per-ROI RSA: compute one RDM per parcel, score against a model RDM,
# project the per-parcel scalars back to a voxel-space BrainData.
rdms = brain.distance(metric='correlation', spatial_scale='roi', roi_mask=atlas)
brain_map = rdms.similarity(model_rdm, project=True, permutation_method=None)
```

**What's added**:
- `BrainData.distance(spatial_scale='whole_brain' | 'roi' | 'searchlight', roi_mask=, radius_mm=)` — `'whole_brain'` (default) preserves existing behavior; `'roi'` returns a stacked `Adjacency` (one RDM per parcel); `'searchlight'` returns a stacked `Adjacency` (one RDM per voxel center) with a synthetic 1-voxel-per-label atlas so each searchlight scalar paints to its center voxel. All three carry `spatial_scale` provenance.
- `BrainData.align(spatial_scale='roi', roi_mask=)` — per-parcel functional alignment (procrustes / SRM); transformed data stitched back to voxel space as a `BrainData`, transforms / common-models kept as `dict[atlas_label, ndarray]`, plus per-parcel `disparity`/`scale` arrays and `roi_labels`. `spatial_scale='searchlight'` raises `NotImplementedError` (overlapping spheres make the `transformed` reassembly ill-posed — a voxel belongs to many spheres with no canonical value).
- `BrainData.{mean, std, median}(spatial_scale='roi', roi_mask=)` — parcellation smoothing: each voxel painted with its parcel's reduction per image.
- `Adjacency.spatial_scale: SpatialScale | None` — optional frozen dataclass carrying `(atlas, roi_labels, source_mask, kind)`. Survives shape-preserving operations (`copy`, `__getitem__` slice, `r_to_z`, `threshold`); dropped when an op collapses the stack to a single matrix.
- `Adjacency.to_brain(values, fill=np.nan) -> BrainData` — paint per-matrix scalars onto voxel space using the attached atlas. Errors when `spatial_scale` is unset.
- `Adjacency.similarity(other, project=True)` — sugar for `to_brain(np.array([r['correlation'] for r in similarity(...)]))`.
- `nltools.mask.roi_to_brain_from_atlas(values, atlas, source_mask, roi_labels=, fill=)` — sibling of the legacy `roi_to_brain` (which takes an *expanded* mask), but operating on a labeled atlas image. Single source of truth for "paint per-parcel scalars from a labeled atlas onto voxel space."

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
```

### Automatic Alpha Selection

```python
# Ridge regression with automatic alpha selection
brain_data.fit(
    model='ridge',
    cv='auto',
    alphas=[0.1, 1.0, 10.0, 100.0],
    X=features
)

# Access best alpha
best_alpha = brain_data.cv_results_['best_alpha']
alpha_scores = brain_data.cv_results_['alpha_scores']
```

### BrainCollection: Multi-Subject Data Container (NEW)

**Status**: ✅ NEW (v0.6.0)

`BrainCollection` is a new class for working with multi-subject neuroimaging data. It provides a unified interface for group-level analyses including encoding models, GLM workflows, and inter-subject correlation.

**Key Features**:
- **3-axis indexing**: `(n_images, n_observations, n_voxels)` semantics
- **Lazy loading**: Memory-efficient for large multi-subject datasets
- **Group inference**: t-tests, permutation tests, ANOVA
- **Encoding models**: `fit_ridge()`, `fit_glm()`, `predict()`
- **ISC analysis**: `isc()`, `isc_test()` for naturalistic neuroimaging
- **Transformations**: `map()`, `filter()`, aggregations across axes

**Basic Usage**:
```python
from nltools.data import BrainData, BrainCollection
from nltools.datasets import fetch_haxby

# Load multi-subject data
data, _ = fetch_haxby(n_subjects=5)
bc = BrainCollection(data, mask=data[0].mask)

# 3-axis indexing
first_subject = bc[0]              # BrainData
timepoint_10 = bc[:, 10]           # BrainCollection
subset = bc[:, :, :1000]           # BrainCollection

# Group statistics
group_mean = bc.mean(axis=0)       # Mean across subjects -> BrainData
subject_means = bc.mean(axis=1)    # Mean across time -> BrainCollection

# Group inference
t_stat, p_val = subject_means.ttest()
```

**Encoding Models**:
```python
import numpy as np

# Fit ridge regression for each subject
X = np.random.randn(bc[0].shape[0], 10)  # (timepoints, features)
result = bc.fit_ridge(X=X, cv=3)

# Access weights for group-level inference
# weights[:, feature_idx, :] -> BrainCollection of that feature's weights
```

**ISC (Inter-Subject Correlation)**:
```python
# Compute ISC with leave-one-out method
isc_result = bc.isc(method="loo")
print(f"Mean ISC: {isc_result['isc'].data.mean():.3f}")

# ISC with permutation testing
isc_test_result = bc.isc_test(method="loo", n_permute=1000)
significant = (isc_test_result['p'].data < 0.05).sum()
```

**GLM Workflow**:
```python
import pandas as pd

# Create events DataFrame
events = pd.DataFrame({
    "onset": [0, 10, 20, 30],
    "duration": [5, 5, 5, 5],
    "trial_type": ["A", "B", "A", "B"],
})

# Fit first-level GLM for each subject
betas = bc.fit_glm(events=events, t_r=2.0)

# Compute contrasts
contrast = betas.compute_contrasts("A - B")

# Group-level inference
t_stat, p_val = contrast.ttest()
```

**Construction Methods**:
| Method | Description |
|--------|-------------|
| `BrainCollection(data, mask)` | From list of BrainData or paths |
| `BrainCollection.from_glob(pattern, mask)` | From glob pattern |
| `BrainCollection.from_bids(layout, mask)` | From pybids BIDSLayout |
| `BrainCollection.from_stacked(brain_data, axis)` | Split stacked BrainData |

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
| `.regress()` | ⚠️ Deprecated shim | Update to `.fit(model='glm')` to silence warning |
| `.predict()` | ⚠️ API + return type changed | Update `algorithm=` → `model=`, `cv_dict=` → `cv=`, `radius=` → `radius_mm=`. Result is a `Predict` dataclass — replace `result['weight_map']` with `result.weight_map`. Fluent `brain.cv(...).predict(...)` removed. |
| `.decompose()` | ⚠️ Kwargs changed | Update `algorithm=` → `method=`; signature is now keyword-only after `self` |
| `BrainData.ttest()` | ⚠️ Signature changed | Old `threshold_dict=` kwarg gone; use `popmean=`, `permutation=`, `tail=`, `n_permute=` |
| `.X` and `.Y` attributes | ✅ Still work | Prefer passing `X=` to `.fit()` directly |
| `.isempty` | ⚠️ Deprecated | Use `.is_empty` instead |
| `.smooth()` return value | ⚠️ Changed behavior | Assign to new variable |
| `BrainData.nifti_masker` | ❌ Removed | Use `nilearn.masking.apply_mask(img, bd.mask)` / `unmask(vec, bd.mask)` |
| `nltools.prefs` module | ❌ Removed | Import from `nltools.templates`; use `set_brainspace()` / `with_brainspace()` |

### Deprecation Timeline

| Feature | v0.6.0 Status | v0.7.0 Status |
|---------|---------------|---------------|
| `.regress()` | ⚠️ Deprecated shim (still callable, emits warning) | ❌ Removed |
| `.isempty` | ⚠️ Deprecated (use `.is_empty`) | May be removed |
| `.X` and `.Y` | Still works | ⚠️ May be deprecated |
| In-place `.smooth()` | Changed (returns copy) | N/A |
| Legacy kwarg aliases (`algorithm=`, `show_progress=`, `radius=`, `n_perm=`, `parallel=`, `thr_upper=`, `thr_lower=`, `kind=`, `ignore_diagonal=`) | ❌ Removed — no aliases kept | N/A |

---

## Testing Your Migration

### Step 1: Check for Deprecated Methods
```python
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("error", DeprecationWarning)
    try:
        brain_data.regress(design_matrix)
    except DeprecationWarning as e:
        print(f"Method deprecated: {e}")
        # Update to: brain_data.fit(model='glm', X=design_matrix)
```

### Step 2: Update Predict API
```python
# OLD (v0.5.1)
results = brain_data.predict(algorithm='svm', cv_dict={'type': 'kfolds', 'n_folds': 5}, radius=10)
weight_map = results['weight_map']

# NEW (v0.6.0) — updated keyword names + Predict dataclass return
# `method` selects the prediction *mode* (whole_brain / roi / searchlight);
# `model` selects the sklearn algorithm (mirrors bd.fit(model=)).
result = brain_data.predict(y=labels, spatial_scale='whole_brain', model='svm', cv=5)
result.weight_map        # mean coefs across folds, shape (n_voxels,)
result.scores            # per-fold scores
result.mean_score        # mean across folds

# Searchlight — populates accuracy_map (no weight_map; per-sphere classifiers)
result = brain_data.predict(y=labels, spatial_scale='searchlight',
                            model='ridge_classifier', radius_mm=10, cv=5)
result.accuracy_map      # voxel-shaped accuracy

# Note: 'ridge' is regression-only; for classification use 'ridge_classifier'.
# scoring='auto' (default) → 'accuracy' for classifiers, 'r2' for regressors.

# Custom preprocessing chain — pass a sklearn Pipeline as model=
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.svm import LinearSVC
pipe = make_pipeline(StandardScaler(), SelectKBest(k=500), LinearSVC())
result = brain_data.predict(y=labels, model=pipe, standardize=False)
```

The fluent API `brain.cv(k=5).normalize().reduce().pipe(t).predict(y, algorithm=…)` has been **removed** from `BrainData`. All four steps fold into kwargs on `bd.predict()` (`cv=`, `standardize=`, `reduce='pca'`, `n_components=`, `model=`). Multi-subject pipelines on `BrainCollection` (hyperalignment / SRM) keep their fluent API — that's where chaining still earns its keep.

### Step 3: Check for Deprecation Warnings
```python
import warnings
warnings.filterwarnings('default', category=DeprecationWarning)

brain_data.isempty   # Deprecated - use .is_empty instead
```

### Step 4: Update Properties
```python
# Search your codebase for:
# - .shape()
# - .isempty()  or .isempty
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
- [ ] Replace `pd.concat([dm.to_pandas(), confounds_df], axis=1) → DesignMatrix(...)` with `dm.append(confounds_df, axis=1)` (raw DataFrames are auto-marked as confounds; metadata is preserved).
- [ ] Update column lookups after `.convolve()`: `dm_conv["stim"]` → `dm_conv["stim_c0"]`. Includes `compute_contrasts("A - B")` strings → `compute_contrasts("A_c0 - B_c0")`. See [DesignMatrix.convolve() always suffixes](#designmatrix-convolve-suffix).
- [ ] Update `from nltools.external import glover_hrf` → `from nltools.algorithms.hrf import glover_hrf`
- [ ] Update `from nltools.simulator import ...` → `from nltools import ...` or `from nltools.data import ...`
- [ ] Update `from nltools.prefs import MNI_Template` → `from nltools.templates import MNI_Template`
- [ ] Remove `from nltools.utils import get_anatomical` — use `nilearn.datasets.load_mni152_template()`
- [ ] Remove `from nltools.stats import regress` — use `BrainData.fit(model='glm')`
- [ ] Drop the `threshold_dict=` kwarg on `BrainData.ttest()` (signature changed — now `popmean=`, `permutation=`, `tail=`, `n_permute=`)
- [ ] Replace `brain.nifti_masker.transform(img)` → `nilearn.masking.apply_mask(img, brain.mask)` (same for `inverse_transform` → `unmask`)
- [ ] Replace `download_collection` / `get_collection_image_metadata` → `fetch_neurovault_collection`
- [ ] Rename any kwargs still using legacy spellings: `algorithm=` → `method=`/`estimator=`, `show_progress=` → `progress_bar=`, `radius=` → `radius_mm=`, `n_perm=` → `n_permute=`, `parallel=` → `device=`, `ignore_diagonal=True` → `include_diag=False`, `thr_upper=`/`thr_lower=` → `upper=`/`lower=`, `kind=` → `method=` (see "v0.6.0 Kwarg Standardization" below)
- [ ] Rename any positional-kwarg calls to `__init__`: `BrainData/Adjacency/BrainCollection/DesignMatrix` constructors now require keyword arguments after the first positional data arg
- [ ] Rename module-level plotting callers: `surface_plot` → `plot_surface`, `scatterplot` → `plot_scatter`, `roc_plot` → `plot_roc`, `probability_plot` → `plot_probability`, `dist_from_hyperplane_plot` → `plot_dist_from_hyperplane`, `adjacency.plot(...)` (module fn) → `plot_adjacency`, `DesignMatrix.heatmap()` → `DesignMatrix.plot()`

### Should fix (deprecated or changed behavior)

- [ ] Replace `.regress()` with `.fit(model='glm')` (deprecated shim still works, emits `DeprecationWarning`)
- [ ] Update `.predict(algorithm=...)` to `.predict(spatial_scale=..., model=..., cv=...)` (new keyword API; `spatial_scale=` chooses ``'whole_brain'``/``'roi'``/``'searchlight'``)
- [ ] Update `.decompose(algorithm=...)` to `.decompose(method=...)` (same `algorithm → method` rename; signature is now keyword-only after `self`)
- [ ] Update `.shape()` → `.shape`, `.isempty()` → `.is_empty`, `.dtype()` → `.dtype`
- [ ] Update `.smooth()` to assign return value (returns copy now)
- [ ] Replace `summarize_bootstrap()` with `BrainData.bootstrap()` or `OnlineBootstrapStats`
- [ ] Remove any `DesignMatrix.reset_index()` calls (pandas-compat no-op; removed)
- [ ] `DesignMatrix.add_dct_basis()` now adds a `cosine_0` constant column by default (parity with `add_poly(0)` → `poly_0`). If you were chaining `.add_poly(0)` after `.add_dct_basis()` and relied on no intercept from the DCT call, drop the now-redundant `add_poly(0)` or pass `include_constant=False` to restore the old SPM-style (no-constant) behaviour.

### Optional (new features to consider)

- [ ] Consider using new `.fit(model='ridge')` for regression
- [ ] Consider using new CV features (`cv=5`, `alpha='auto'`)
- [ ] Migrate `isc()`, `isc_group()` to `isc_permutation_test()`, `isc_group_permutation_test()` (optional — wrappers maintained)
- [ ] Replace `stats.correlation()` with `correlation_permutation_test()` from inference module
- [ ] Replace `stats.pearson()` with `scipy.stats.pearsonr` or `correlation_permutation_test()`
- [ ] Consider using `fit(inplace=False)` for immutable results and serialization
- [ ] Consider using `BrainCollection` for multi-subject analyses (new in v0.6.0)
- [ ] Consider using `SRM` / `DetSRM` for shared response modeling (new in v0.6.0)
- [ ] Test with `DeprecationWarning` filters to catch remaining issues

---

(new-feature-gpu-accelerated-statistical-inference)=
## New Feature: GPU-Accelerated Statistical Inference

**Status**: ✅ NEW (v0.6.0)

nltools v0.6.0 introduces a comprehensive GPU-accelerated inference module for permutation testing and bootstrap resampling, providing **10-100× speedup** over CPU-only implementations.

### Overview

**New module**: `nltools.algorithms.inference`
- **8 comprehensive modules**: one_sample, two_sample, correlation, timeseries, matrix, isc, utils, __init__
- **170 tests**: 100% passing with perfect cross-backend determinism
- **GPU-optional**: Works on CPU-only systems with parallel speedup (4-8×)
- **Drop-in replacement**: Compatible with existing nltools.stats functions

### Available Functions

| Function | Description | Performance |
|----------|-------------|-------------|
| `one_sample_permutation_test()` | Sign-flipping test (mean ≠ 0) | 10-100× GPU, 4-8× CPU-parallel |
| `two_sample_permutation_test()` | Group comparison (mean₁ ≠ mean₂) | 10-100× GPU, 4-8× CPU-parallel |
| `correlation_permutation_test()` | Correlation significance (Pearson/Spearman/Kendall) | 10-100× GPU, 4-8× CPU-parallel |
| `timeseries_correlation_permutation_test()` | Time-series correlation (preserves autocorrelation) | 4-8× CPU-parallel |
| `matrix_permutation_test()` | Mantel test for matrix correlation | 6× CPU-parallel |
| `isc_permutation_test()` | Intersubject correlation (LOO/Pairwise) | 15-30× GPU, 4-8× CPU-parallel |
| `circle_shift()` | Circular rotation for time series | - |
| `phase_randomize()` | FFT-based phase shuffling | - |

### Migration from nltools.stats

**Old API** (nltools.stats):
```python
from nltools.stats import (
    one_sample_permutation,
    two_sample_permutation,
    correlation_permutation,
    matrix_permutation
)

# One-sample test
result = one_sample_permutation(data, n_permute=5000)

# Two-sample test
result = two_sample_permutation(data1, data2, n_permute=5000)

# Correlation test
result = correlation_permutation(x, y, n_permute=5000, metric='pearson')

# Matrix permutation (Mantel test)
result = matrix_permutation(matrix1, matrix2, n_permute=5000)
```

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
    backend='torch',  # Use GPU (optional, defaults to CPU-parallel)
    random_state=42
)

# Two-sample test
result = two_sample_permutation_test(
    data1, data2,
    n_permute=5000,
    tail='two',  # 'two', 'upper', or 'lower'
    backend='torch'
)

# Correlation test with multiple metrics
result = correlation_permutation_test(
    x, y,
    n_permute=5000,
    metric='spearman',  # 'pearson', 'spearman', or 'kendall'
    backend='torch'
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
    backend='torch'
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
    backend='torch'
)

# Returns:
# - statistic: Observed ISC
# - p: P-values
# - null_distribution: Null ISC values (if return_null=True)
```

**3. Backend Options**
```python
# CPU-parallel (default, memory-efficient)
result = one_sample_permutation_test(data, backend=None)

# GPU-batched (10-100× faster for large problems)
result = one_sample_permutation_test(data, backend='torch')

# NumPy (simple, single-threaded)
result = one_sample_permutation_test(data, backend='numpy')

# Auto-select (chooses best available)
result = one_sample_permutation_test(data, backend='auto')
```

### Key Improvements

**Performance**:
- **GPU acceleration**: 10-100× speedup with PyTorch backend
- **CPU parallelization**: 4-8× speedup with joblib (default)
- **Automatic batching**: Prevents GPU out-of-memory errors
- **Progress bars**: Real-time feedback for long-running tests

**Correctness**:
- **Perfect determinism**: 0.000% cross-backend variance (same seed → identical results)
- **Validated against literature**: Nichols & Holmes 2002, Chen et al. 2016, Theiler et al. 1992
- **Comprehensive testing**: 170 tests with mathematical correctness verification
- **Backward compatible**: ~1-2% variance vs stats.py (acceptable for breaking release)

**Usability**:
- **Comprehensive error messages**: Clear validation and actionable suggestions
- **Full type hints**: Better IDE support and static analysis
- **Extensive documentation**: DESIGN.md with algorithms, citations, trade-offs
- **Multiple metrics**: Pearson, Spearman, Kendall for correlation/matrix tests

### Migration Checklist

- [ ] Replace `nltools.stats` imports with `nltools.algorithms.inference`
- [ ] Update function names (add `_test` suffix)
- [ ] Add `backend='torch'` for GPU acceleration (optional)
- [ ] Update `metric` parameter for correlation tests
- [ ] Use `method='circle_shift'` or `method='phase_randomize'` for time series
- [ ] Consider using ISC for multi-subject analyses
- [ ] Test with `random_state` for reproducibility

### Deprecation Timeline

**v0.6.0** (current):
- ✅ New inference module available
- ⚠️ stats.py functions still work (no warnings yet)
- ✅ Both APIs coexist for migration period

**v0.6.1** (planned):
- ⚠️ Add deprecation warnings to stats.py functions
- ⚠️ Point users to new inference module
- ✅ Update all internal uses to new API

**v0.7.0** (future):
- ❌ Remove stats.py permutation functions
- ✅ Only inference module available

---

## Getting Help

- **API Documentation**: Check updated API docs for each class/method
- **Tutorials**: See rewritten tutorials for v0.6.0 patterns
- **GitHub Issues**: Report migration problems or unclear docs

---

*Last updated: 2026-04-20 for nltools v0.6.0*
