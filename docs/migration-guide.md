# Migration Guide: v0.5 → v0.6

Version 0.6.0 is a **breaking release** that refactors nltools to better leverage nilearn and establish cleaner APIs. This guide shows you how to update your code.

---

## Quick Reference: What Changed

| Category | v0.5.1 (Old) | v0.6.0 (New) | Status |
|----------|--------------|--------------|--------|
| **GLM regression** | `.regress()` | `.fit(model='glm')` | Deprecated |
| **Ridge regression** | Manual | `.fit(model='ridge')` | New |
| **ML prediction** | `.predict()` | Use sklearn directly | Removed |
| **t-tests** | `.ttest()` | Use scipy.stats | Removed |
| **Method chaining** | `.smooth()` modifies in-place | Returns copy | Changed |
| **Properties** | `.shape()`, `.isempty()` | `.shape`, `.isempty` | Changed |
| **Cross-validation** | N/A | `.fit(..., cv=5)` | New |
| **HyperAlignment** | Via `align()` only | `HyperAlignment` class | New |

---

## Dependency Updates

### nilearn 0.12+ Compatibility

**Status**: ✅ FIXED (v0.6.0)

nltools v0.6.0 now requires **nilearn >= 0.12**, which introduced a breaking change in `NiftiMasker.transform()`:

**What changed in nilearn 0.12:**
- **3D images** now transform to **1D arrays** `(n_voxels,)` instead of 2D arrays `(1, n_voxels)`
- **4D images** still transform to **2D arrays** `(n_timepoints, n_voxels)` (unchanged)

**How nltools adapted:**
- Updated `Brain_Data._load_from_list()` to use `np.vstack()` instead of `np.concatenate()`
- This ensures correct shape when loading lists of 3D nifti files
- **No user code changes needed** - Brain_Data API remains identical

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

### 1. Removed Methods

| Method | Alternative | Migration Effort |
|--------|-------------|------------------|
| `.predict(algorithm='svm')` | Use sklearn directly | Low |
| `.ttest()` | Use `scipy.stats.ttest_1samp()` | Low |
| `.randomise()` | Use nilearn permutation testing | Medium |
| `.predict_multi()` | Will return in future Model class | N/A |

### 2. Removed Classes

| Class | Status | Alternative |
|-------|--------|-------------|
| `Brain_Collection` | Removed | Will return in v0.7.0+ |
| `Model` | Removed | Will return in v0.7.0+ |

### 3. Deprecated Attributes

| Attribute | Status | Alternative |
|-----------|--------|-------------|
| `.X` | Deprecated (still works) | Pass `X=` to `.fit()` |
| `.Y` | Deprecated (still works) | Manage labels separately |

---

## Migration Patterns

### Pattern 1: GLM Regression

**Before (v0.5.1):**
```python
brain_data.X = design_matrix
results = brain_data.regress()  # Returns dict
betas = results['beta']
t_stats = results['t']
```

**After (v0.6.0):**
```python
brain_data.fit(model='glm', X=design_matrix)  # Stores results as attributes
betas = brain_data.glm_betas      # Brain_Data object
t_stats = brain_data.glm_t        # Brain_Data object
```

| Aspect | Old | New | Benefit |
|--------|-----|-----|---------|
| API style | Dict return | Sklearn-style attributes | Composable, familiar |
| Design matrix | Stored as `.X` | Passed as argument | Explicit, clearer |
| Results | Dict with keys | Brain_Data attributes | Type-safe, chainable |

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
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

X = brain_data.data  # (n_samples, n_voxels)
y = labels           # Manage separately
clf = SVC(kernel='linear')
cv_results = cross_validate(clf, X, y, cv=5)
```

| Aspect | Old | New | Reason |
|--------|-----|-----|--------|
| API | Custom wrapper | Direct sklearn | More flexible |
| Label storage | `.Y` attribute | Separate variable | Explicit |
| ML library | Limited algorithms | Full sklearn | More options |

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
shape = brain_data.shape      # No parentheses
is_empty = brain_data.isempty  # No parentheses
dtype = brain_data.dtype       # No parentheses
```

| Method | Old | New | Reason |
|--------|-----|-----|--------|
| `.shape()` | Method call | Property | No computation |
| `.isempty()` | Method call | Property | No computation |
| `.dtype()` | Method call | Property | No computation |

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

---

## Compatibility & Warnings

### Backward Compatibility

| Feature | Status | Action Required |
|---------|--------|-----------------|
| HDF5 files from v0.5.1 | ✅ Fully compatible | None |
| `.regress()` | ⚠️ Works with `FutureWarning` | Update to `.fit(model='glm')` |
| `.X` and `.Y` attributes | ⚠️ Work but deprecated | Pass `X=` to `.fit()` |
| `.smooth()` return value | ⚠️ Changed behavior | Assign to new variable |

### Deprecation Timeline

| Feature | v0.6.0 Status | v0.7.0 Status |
|---------|---------------|---------------|
| `.regress()` | Deprecated (works) | ❌ Removed (error) |
| `.X` and `.Y` | Deprecated (works) | ⚠️ May be removed |
| In-place `.smooth()` | Changed | N/A |

---

## Testing Your Migration

### Step 1: Check for Removed Methods
```python
import warnings
warnings.filterwarnings('error', category=FutureWarning)

try:
    brain_data.predict()  # Will error if removed
except NotImplementedError:
    print("Method removed - update your code")
```

### Step 2: Check for Deprecation Warnings
```python
warnings.filterwarnings('default', category=FutureWarning)

brain_data.X = design_matrix
brain_data.regress()  # Will show FutureWarning
```

### Step 3: Update Properties
```python
# Search your codebase for:
# - .shape()
# - .isempty()
# - .dtype()

# Replace with:
# - .shape
# - .isempty
# - .dtype
```

---

## Migration Checklist

- [ ] Replace `.regress()` with `.fit(model='glm')`
- [ ] Replace `.predict(algorithm=...)` with sklearn
- [ ] Replace `.ttest()` with `scipy.stats.ttest_1samp()`
- [ ] Update `.shape()` → `.shape` (and `.isempty()`, `.dtype()`)
- [ ] Update `.smooth()` to assign return value
- [ ] Remove `.X` and `.Y` assignments (pass `X=` to `.fit()`)
- [ ] Consider using new `.fit(model='ridge')` for regression
- [ ] Consider using new CV features (`cv=5`, `alpha='auto'`)
- [ ] Test with `FutureWarning` → error to catch remaining issues

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

*Last updated: 2025-10-30 for nltools v0.6.0*
