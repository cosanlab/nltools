# Migration Guide: v0.5 → v0.6

Version 0.6.0 is a **breaking release** that refactors nltools to better leverage nilearn and establish cleaner APIs. This guide shows you how to update your code.

---

## Quick Reference: What Changed

| Category | v0.5.1 (Old) | v0.6.0 (New) | Status |
|----------|--------------|--------------|--------|
| **GLM regression** | `.regress()` | `.fit(model='glm')` | **Removed** |
| **Ridge regression** | Manual | `.fit(model='ridge')` | New |
| **ML prediction** | `.predict(algorithm='svm')` | `.predict(X=..., y=...)` | Unified API |
| **t-tests** | `.ttest()` | Use scipy.stats | Removed |
| **Method chaining** | `.smooth()` modifies in-place | Returns copy | Changed |
| **Properties** | `.shape()`, `.isempty()` | `.shape`, `.is_empty` | Changed |
| **Cross-validation** | N/A | `.fit(..., cv=5)` | New |
| **HyperAlignment** | Via `align()` only | `HyperAlignment` class | New |
| **Multi-subject** | `Brain_Collection` | `BrainCollection` class | **New** |
| **SRM** | N/A | `SRM` / `DetSRM` classes | **New** |
| **GPU inference** | N/A | `inference` module | **New** |

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
- `.reset_index(drop=True)` - No-op for pandas compatibility (Polars has no row indexes)

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

# reset_index() is a no-op (for pandas compatibility)
dm_reset = dm.reset_index(drop=True)  # Returns self (no change)
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
from nltools.prefs import MNI_Template
MNI_Template.template = 'fmriprep'  # 'default', 'nilearn', or 'fmriprep'
MNI_Template.resolution = 1          # 1, 2, or 3

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
| `.regress()` | `.fit(model='glm', X=design_matrix)` | **Low** |
| `.predict(algorithm='svm')` | `.predict(y=labels, method='svm')` (updated API) | **Low** |
| `.ttest()` | Use `scipy.stats.ttest_1samp()` | Low |
| `.randomise()` | Use nilearn permutation testing | Medium |
| `.predict_multi()` | Will return in future Model class | N/A |
| `summarize_bootstrap()` | `BrainData.bootstrap()` or `OnlineBootstrapStats` | **Low** |

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

### Pattern 1: regress() Removed → Use fit(model='glm')

**Status**: ⚠️ **BREAKING CHANGE** - `.regress()` has been removed in v0.6.0

The `.regress()` method has been completely removed and replaced with the unified `.fit(model='glm')` API.

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
| Error handling | Raises error in v0.6.0 | Use `.fit()` | Clear migration path |

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
```

**After (v0.6.0):**
```python
# Option 1: Use the unified .predict() API (recommended)
# Timeseries prediction (encoding models)
brain_data.predict(X=features, method='ridge', cv=5)

# MVPA decoding
brain_data.predict(y=labels, method='svm', cv=5)

# Option 2: Use sklearn directly (full flexibility)
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

X = brain_data.data  # (n_samples, n_voxels)
y = labels           # Manage separately
clf = SVC(kernel='linear')
cv_results = cross_validate(clf, X, y, cv=5)
```

| Aspect | Old | New | Reason |
|--------|-----|-----|--------|
| API | `algorithm=` keyword | `method=` or `estimator=` | Clearer naming |
| CV | `cv_dict=` | `cv=` (int or sklearn splitter) | Simpler |
| Label storage | `.Y` attribute | `y=` argument | Explicit |
| Fallback | N/A | Direct sklearn | Full flexibility |

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
| `BrainData.predict()` | API changed | `algorithm=`, `cv_dict=` | `method=`, `cv=` | Update keywords |
| Import paths | Module moved | `stats.isc()` | `inference.isc_permutation_test()` | Wrapper maintained |
| Return keys | Key renamed | `null_dist` | `null_distribution` | Wrapper handles mapping |

---

## New Features

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

---

## Compatibility & Warnings

### Backward Compatibility

| Feature | Status | Action Required |
|---------|--------|-----------------|
| HDF5 files from v0.5.1 | ✅ Fully compatible | None |
| `.regress()` | ❌ **REMOVED** (raises error) | **Must** update to `.fit(model='glm')` |
| `.predict()` | ⚠️ API changed | Update `algorithm=` → `method=`, `cv_dict=` → `cv=` |
| `.X` and `.Y` attributes | ✅ Still work | Prefer passing `X=` to `.fit()` directly |
| `.isempty` | ⚠️ Deprecated | Use `.is_empty` instead |
| `.smooth()` return value | ⚠️ Changed behavior | Assign to new variable |

### Deprecation Timeline

| Feature | v0.6.0 Status | v0.7.0 Status |
|---------|---------------|---------------|
| `.regress()` | ❌ **Removed** (NotImplementedError) | N/A |
| `.isempty` | ⚠️ Deprecated (use `.is_empty`) | May be removed |
| `.X` and `.Y` | Still works | ⚠️ May be deprecated |
| In-place `.smooth()` | Changed (returns copy) | N/A |

---

## Testing Your Migration

### Step 1: Check for Removed Methods
```python
try:
    brain_data.regress()  # Will raise NotImplementedError in v0.6.0
except NotImplementedError as e:
    print(f"Method removed: {e}")
    # Update to: brain_data.fit(model='glm', X=design_matrix)
```

### Step 2: Update Predict API
```python
# OLD (v0.5.1)
brain_data.predict(algorithm='svm', cv_dict={'type': 'kfolds', 'n_folds': 5})

# NEW (v0.6.0) - updated keyword names
brain_data.predict(y=labels, method='svm', cv=5)
```

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

- [ ] Replace `.regress()` with `.fit(model='glm')`
- [ ] Update `.predict(algorithm=...)` to `.predict(method=..., cv=...)` (new keyword API)
- [ ] Replace `.ttest()` with `scipy.stats.ttest_1samp()`
- [ ] Update `.shape()` → `.shape`, `.isempty()` → `.is_empty`, `.dtype()` → `.dtype`
- [ ] Update `.smooth()` to assign return value (returns copy now)
- [ ] Replace `summarize_bootstrap()` with `BrainData.bootstrap()` or `OnlineBootstrapStats`
- [ ] Consider using new `.fit(model='ridge')` for regression
- [ ] Consider using new CV features (`cv=5`, `alpha='auto'`)
- [ ] Migrate `isc()`, `isc_group()` to `isc_permutation_test()`, `isc_group_permutation_test()` (optional - wrappers maintained)
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

*Last updated: 2026-03-10 for nltools v0.6.0*
