# API Audit Report: Functional Core Layer
**Audit Date:** 2026-01-24
**Version Target:** v0.6.0
**Scope:** `nltools/stats.py`, `nltools/utils.py`, `nltools/algorithms/` directory

---

## Executive Summary

This audit examines API consistency across the functional core layer of nltools. Overall, the codebase demonstrates **good consistency** with a few areas needing attention:

- **Strengths**: Consistent use of `parallel`, `n_jobs`, `random_state` parameters across the new inference module
- **Areas for improvement**: Legacy wrapper functions in `stats.py` have slightly different parameter naming than underlying inference module functions
- **Deprecations identified**: 1 deprecated function in `stats.py` (`pearson()`)
- **Naming conventions**: Generally consistent but some variations exist

---

## 1. Function Naming Patterns

### 1.1 Pattern Analysis

| Pattern | Module | Examples | Count |
|---------|--------|----------|-------|
| `compute_*` | stats.py | `compute_similarity`, `compute_multivariate_similarity`, `compute_icc` | 3 |
| `_compute_*` (private) | inference | `_compute_pvalue`, `_compute_isc_group`, `_compute_cross_correlation` | 6+ |
| `*_permutation` | stats.py (legacy) | `one_sample_permutation`, `two_sample_permutation`, `correlation_permutation`, `matrix_permutation` | 4 |
| `*_permutation_test` | inference | `one_sample_permutation_test`, `two_sample_permutation_test`, `correlation_permutation_test`, `matrix_permutation_test`, `isc_permutation_test`, `isc_group_permutation_test` | 6 |
| `calc_*` | stats.py | `calc_bpm` | 1 |
| `fisher_*` | stats.py | `fisher_r_to_z`, `fisher_z_to_r` | 2 |
| `find_*` | stats.py | `find_spikes` | 1 |
| `make_*` | stats.py | `make_cosine_basis` | 1 |
| `transform_*` | stats.py | `transform_pairwise` | 1 |
| verb (imperative) | stats.py | `zscore`, `fdr`, `holm_bonf`, `threshold`, `multi_threshold`, `winsorize`, `trim`, `downsample`, `upsample`, `procrustes`, `align`, `align_states` | 12 |
| verb (imperative) | utils.py | `concatenate`, `to_h5`, `load_brain_data_h5` | 3 |
| `*_hrf` | hrf.py | `spm_hrf`, `glover_hrf` | 2 |
| `*_derivative` | hrf.py | `spm_time_derivative`, `glover_time_derivative`, `spm_dispersion_derivative` | 3 |
| `ridge_*` | ridge | `ridge_svd`, `ridge_cv` | 2 |
| `solve_*` | ridge | `solve_ridge_cv`, `solve_banded_ridge_cv` | 2 |

### 1.2 Recommended Naming Conventions

Based on the audit, the following naming conventions should be adopted going forward:

1. **Public computation functions**: Use `compute_*` prefix for pure computation functions (e.g., `compute_similarity`)
2. **Statistical tests**: Use `*_permutation_test` suffix for permutation-based tests (inference module style)
3. **Transformation functions**: Use verb form (e.g., `zscore`, `threshold`, `winsorize`)
4. **Private helpers**: Use `_compute_*` or `_*` prefix
5. **I/O functions**: Use `to_*` / `load_*` / `save_*` pattern

### 1.3 Inconsistencies Noted

| Location | Issue | Recommendation |
|----------|-------|----------------|
| stats.py:455 | `calc_bpm` uses `calc_*` pattern | Consider `compute_bpm` for consistency (minor) |
| stats.py legacy wrappers | `one_sample_permutation` vs `one_sample_permutation_test` | Keep for backward compatibility, document as legacy |

---

## 2. Common Parameter Patterns

### 2.1 Parameter Inventory

#### `parallel` Parameter
Controls execution backend for parallelization.

| Module | Function | Default | Valid Values | Notes |
|--------|----------|---------|--------------|-------|
| inference/one_sample.py:222 | `one_sample_permutation_test` | `"cpu"` | `None`, `"cpu"`, `"gpu"` | Consistent |
| inference/two_sample.py | `two_sample_permutation_test` | `"cpu"` | `None`, `"cpu"`, `"gpu"` | Consistent |
| inference/correlation.py:655 | `correlation_permutation_test` | `"cpu"` | `None`, `"cpu"`, `"gpu"` | Consistent |
| inference/matrix.py | `matrix_permutation_test` | `"cpu"` | `None`, `"cpu"`, `"gpu"` | Consistent |
| inference/isc.py | `isc_permutation_test` | `"cpu"` | `None`, `"cpu"`, `"gpu"` | Consistent |
| algorithms/ridge/_core.py:45 | `ridge_svd` | `None` | `None`, `"cpu"`, `"gpu"` | Different default |
| algorithms/ridge/_core.py:207 | `ridge_cv` | `"cpu"` | `None`, `"cpu"`, `"gpu"` | Consistent |
| algorithms/srm.py:212 | `SRM.fit` | `"cpu"` | `None`, `"cpu"`, `"gpu"` | Consistent |
| algorithms/srm.py:764 | `DetSRM.fit` | `"cpu"` | `None`, `"cpu"`, `"gpu"` | Consistent |
| algorithms/hyperalignment.py:218 | `HyperAlignment.fit` | `"cpu"` | `None`, `"cpu"` | No GPU option |

**Finding**: `ridge_svd` has `parallel=None` as default, inconsistent with other functions that default to `"cpu"`.

#### `n_jobs` Parameter
Controls number of CPU cores for parallel execution.

| Module | Function | Default | Notes |
|--------|----------|---------|-------|
| inference/*.py | All permutation tests | `-1` | Consistent (all cores) |
| algorithms/srm.py | SRM.fit, DetSRM.fit | `-1` | Consistent |
| algorithms/hyperalignment.py | HyperAlignment.fit | `-1` | Consistent |
| stats.py wrappers | Legacy functions | `-1` | Consistent |

**Finding**: Fully consistent across all modules.

#### `random_state` Parameter
Controls random number generation for reproducibility.

| Module | Function | Default | Position | Notes |
|--------|----------|---------|----------|-------|
| inference/*.py | All permutation tests | `None` | Last | Consistent |
| algorithms/ridge/_core.py | `ridge_svd`, `ridge_cv` | `None` | Last | Consistent |
| algorithms/srm.py | SRM, DetSRM | N/A | Uses `rand_seed` | Different naming |
| stats.py | Legacy wrappers | `None` | Last | Consistent |

**Finding**: SRM/DetSRM use `rand_seed` instead of `random_state`. This is inherited from the original BrainIAK implementation.

#### `max_gpu_memory_gb` Parameter
Controls GPU memory budget for batching.

| Module | Function | Default | Notes |
|--------|----------|---------|-------|
| inference/*.py | All GPU-enabled tests | `4.0` | Consistent |
| algorithms/ridge/_core.py | `ridge_svd`, `ridge_cv` | `4.0` | Consistent |
| algorithms/srm.py | SRM.fit, DetSRM.fit | `4.0` | Consistent |

**Finding**: Fully consistent.

#### `tail` Parameter
Controls test directionality for hypothesis tests.

| Module | Function | Default | Valid Values | Notes |
|--------|----------|---------|--------------|-------|
| inference/*.py | All permutation tests | `2` | `int\|str` (`1`, `2`, `-1`, `'upper'`, `'lower'`, `'two'`) | Consistent, flexible |
| stats.py wrappers | Legacy functions | `2` | `int` (`1`, `2`) | Legacy, int only |

**Finding**: Inference module supports more flexible string values; legacy wrappers support only int.

### 2.2 Parameter Order Convention

The established parameter order convention (observed across inference module):

```python
def function_name(
    # Required data parameters
    data,
    # Optional algorithm parameters
    n_permute=5000,
    metric="pearson",  # (if applicable)
    tail=2,
    return_null=False,
    # Backend/parallelization parameters (grouped)
    parallel="cpu",
    n_jobs=-1,
    max_gpu_memory_gb=4.0,
    # Random state (always last optional)
    random_state=None,
) -> dict:
```

### 2.3 Parameter Inconsistencies

| Location | Issue | Severity | Recommendation |
|----------|-------|----------|----------------|
| `ridge/_core.py:45` | `ridge_svd` defaults `parallel=None` | Low | Consider changing to `"cpu"` for consistency |
| `srm.py:200-201` | Uses `rand_seed` instead of `random_state` | Medium | Document difference; keep for backward compat |
| `stats.py` wrappers | Use `return_perms` vs inference module's `return_null` | Low | Documented mapping exists |
| `stats.py` wrappers | `n_samples` vs inference's `n_permute` (isc functions) | Low | Documented mapping exists |

---

## 3. Deprecated Functions

### 3.1 Currently Deprecated

| Function | Location | Deprecated Version | Replacement | Notes |
|----------|----------|-------------------|-------------|-------|
| `pearson()` | stats.py:87-110 | v0.5.2 | `scipy.stats.pearsonr`, `numpy.corrcoef`, or `correlation_permutation_test` | Issues DeprecationWarning |

### 3.2 Candidates for Deprecation

| Function | Location | Reason | Recommendation |
|----------|----------|--------|----------------|
| `get_anatomical()` | utils.py:266-272 | Docstring says "DEPRECATED" but no warning | Add DeprecationWarning or remove |
| `_permute_group()` | stats.py:650-655 | Unused private function, superseded by inference module | Remove in v0.6.0 |

### 3.3 Legacy Wrapper Functions (Not Deprecated, But Documented)

These wrapper functions in `stats.py` delegate to the inference module but are kept for backward compatibility:

| Function | Lines | Delegates To |
|----------|-------|--------------|
| `one_sample_permutation` | 666-710 | `one_sample_permutation_test` |
| `two_sample_permutation` | 713-763 | `two_sample_permutation_test` |
| `correlation_permutation` | 766-847 | `correlation_permutation_test` / `timeseries_correlation_permutation_test` |
| `matrix_permutation` | 850-916 | `matrix_permutation_test` |
| `isc` | 1519-1615 | `isc_permutation_test` |
| `isc_group` | 1733-1846 | `isc_group_permutation_test` |

---

## 4. Error Handling Patterns

### 4.1 Validation Functions

The inference module uses dedicated validation functions from `_validation.py`:

```python
from .._validation import (
    validate_tail_parameter,
    validate_parallel_parameter,
    validate_array_shape_range,
)
```

### 4.2 Error Handling Patterns Observed

| Pattern | Module | Example |
|---------|--------|---------|
| `raise ValueError("descriptive message")` | All | Consistent throughout |
| `raise NotImplementedError(...)` | inference, srm | For unimplemented features |
| `raise TypeError(...)` | utils, srm | For type mismatches |
| `raise NotFittedError(...)` | srm | sklearn-style for unfitted models |
| `warnings.warn(..., DeprecationWarning)` | stats, datasets | For deprecated functions |

### 4.3 Input Validation Consistency

| Function | Validates Shape | Validates Dtype | Validates Params |
|----------|----------------|-----------------|------------------|
| `one_sample_permutation_test` | Yes | Yes (converts to float64) | Yes |
| `correlation_permutation_test` | Yes | Yes | Yes |
| `ridge_svd` | Yes | Yes (converts to float32) | Yes |
| `compute_similarity` | Yes (atleast_2d) | No | Yes |
| `compute_icc` | Yes | Yes | Yes |

---

## 5. Docstring Presence and Quality

### 5.1 Docstring Coverage

| Module | Functions | With Docstrings | Coverage |
|--------|-----------|-----------------|----------|
| stats.py | ~42 | 42 | 100% |
| utils.py | ~18 | 18 | 100% |
| algorithms/ridge/_core.py | 2 | 2 | 100% |
| algorithms/srm.py | ~15 | 15 | 100% |
| algorithms/hyperalignment.py | ~6 | 6 | 100% |
| algorithms/inference/*.py | ~20 | 20 | 100% |
| algorithms/hrf.py | 6 | 6 | 100% |

### 5.2 Docstring Format

All modules use the **Google-style docstring format** with:
- Args section
- Returns section
- Examples section (inconsistent presence)
- Notes section (where applicable)

### 5.3 Areas for Improvement

| Location | Issue | Recommendation |
|----------|-------|----------------|
| stats.py wrappers | Docstrings reference old API | Update to mention they're wrappers |
| utils.py:266 `get_anatomical` | Says "DEPRECATED" in docstring | Either add proper warning or update docstring |

---

## 6. Class API Consistency (sklearn-style)

### 6.1 Class Comparison

| Class | Location | `fit()` | `transform()` | `fit_transform()` | `transform_subject()` |
|-------|----------|---------|---------------|-------------------|-----------------------|
| `SRM` | srm.py:137 | Yes | Yes | No (inherited) | Yes |
| `DetSRM` | srm.py:696 | Yes | No (uses parent) | No | Yes |
| `HyperAlignment` | hyperalignment.py:139 | Yes | Yes | No | Yes |

All classes inherit from `sklearn.base.BaseEstimator, TransformerMixin`.

### 6.2 Fitted Attribute Naming

| Class | Fitted Attributes | Convention |
|-------|-------------------|------------|
| SRM | `w_`, `s_`, `sigma_s_`, `mu_`, `rho2_`, `random_state_` | sklearn trailing underscore |
| DetSRM | `w_`, `s_`, `random_state_` | sklearn trailing underscore |
| HyperAlignment | `w_`, `s_`, `disparity_`, `scale_` | sklearn trailing underscore |

**Finding**: Consistent use of sklearn-style trailing underscore convention.

---

## 7. Recommendations Summary

### 7.1 Critical (Should Fix Before v0.6.0)

1. **Add DeprecationWarning to `get_anatomical()`** (utils.py:266)
   - Currently only has docstring mention
   - File: `/Users/esh/Documents/pypackages/nltools/nltools/utils.py`, line 266

### 7.2 Medium Priority (Consider for v0.6.0)

1. **Consider standardizing `ridge_svd` default**
   - Change `parallel=None` to `parallel="cpu"` for consistency
   - File: `/Users/esh/Documents/pypackages/nltools/nltools/algorithms/ridge/_core.py`, line 45

2. **Remove unused `_permute_group()` function**
   - Unused private function, superseded by inference module
   - File: `/Users/esh/Documents/pypackages/nltools/nltools/stats.py`, lines 650-655

3. **Document parameter mapping for legacy wrappers**
   - Add note to docstrings explaining parameter name differences
   - Affects: `return_perms` vs `return_null`, `n_samples` vs `n_permute`

### 7.3 Low Priority (Future Enhancement)

1. **Consider renaming `calc_bpm` to `compute_bpm`**
   - Minor consistency improvement
   - File: `/Users/esh/Documents/pypackages/nltools/nltools/stats.py`, line 455

2. **Document `rand_seed` vs `random_state` difference in SRM**
   - Add note to SRM/DetSRM docstrings
   - Inherited from original BrainIAK implementation

---

## 8. API Quality Metrics

| Metric | Score | Notes |
|--------|-------|-------|
| Naming Consistency | 8/10 | Minor variations (calc_* vs compute_*) |
| Parameter Consistency | 9/10 | Excellent across inference module |
| Docstring Coverage | 10/10 | 100% coverage |
| Error Handling | 9/10 | Consistent patterns |
| Backward Compatibility | 10/10 | Legacy wrappers preserve old API |
| sklearn Compatibility | 10/10 | Proper BaseEstimator/TransformerMixin usage |

**Overall API Quality: 9.3/10**

---

## Appendix A: Complete Function Signature Index

### stats.py Public Functions

```python
# Deprecated
def pearson(x, y) -> np.ndarray  # DEPRECATED v0.5.2

# Data transformation
def zscore(df) -> pd.DataFrame | pd.Series
def fdr(p, q=0.05) -> float
def holm_bonf(p, alpha=0.05) -> float
def threshold(stat, p, thr=0.05, return_mask=False) -> BrainData | tuple
def multi_threshold(t_map, p_map, thresh) -> BrainData
def winsorize(data, cutoff=None, replace_with_cutoff=True) -> pl.DataFrame | pl.Series
def trim(data, cutoff=None) -> pl.DataFrame | pl.Series
def calc_bpm(beat_interval, sampling_freq) -> float
def downsample(data, sampling_freq=None, target=None, target_type="samples", method="mean") -> pl.DataFrame | pl.Series
def upsample(data, sampling_freq=None, target=None, target_type="samples", method="linear") -> pl.DataFrame | pl.Series
def fisher_r_to_z(r) -> np.ndarray
def fisher_z_to_r(z) -> np.ndarray
def make_cosine_basis(nsamples, sampling_freq, filter_length, unit_scale=True, drop=0) -> np.ndarray
def transform_pairwise(X, y) -> tuple[np.ndarray, np.ndarray]

# Legacy permutation wrappers (delegate to inference module)
def one_sample_permutation(data, n_permute=5000, tail=2, n_jobs=-1, return_perms=False, random_state=None) -> dict
def two_sample_permutation(data1, data2, n_permute=5000, tail=2, n_jobs=-1, return_perms=False, random_state=None) -> dict
def correlation_permutation(data1, data2, method="permute", n_permute=5000, metric="spearman", tail=2, n_jobs=-1, return_perms=False, random_state=None) -> dict
def matrix_permutation(data1, data2, n_permute=5000, metric="spearman", how="upper", include_diag=False, tail=2, n_jobs=-1, return_perms=False, random_state=None) -> dict

# Alignment
def procrustes(data1, data2) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]
def procrustes_distance(mat1, mat2, n_permute=5000, tail=2, n_jobs=-1, random_state=None) -> dict
def align(data, method="deterministic_srm", n_features=None, axis=0, *args, **kwargs) -> dict
def align_states(reference, target, metric="correlation", return_index=False, replace_zero_variance=False) -> np.ndarray | list

# ISC functions
def isc(data, n_samples=5000, metric="median", method="bootstrap", ci_percentile=95, exclude_self_corr=True, return_null=False, tail=2, n_jobs=-1, random_state=None, sim_metric="correlation") -> dict
def isc_group(group1, group2, n_samples=5000, metric="median", method="permute", ci_percentile=95, exclude_self_corr=True, return_null=False, tail=2, n_jobs=-1, random_state=None) -> dict
def isfc(data, method="average", n_jobs=-1) -> list
def isps(data, sampling_freq=0.5, low_cut=0.04, high_cut=0.07, order=5, pairwise=False) -> dict

# Spike detection
def find_spikes(data, global_spike_cutoff=3, diff_spike_cutoff=3) -> pl.DataFrame

# Similarity computation
def compute_similarity(data1, data2, method="correlation") -> np.ndarray
def compute_multivariate_similarity(y, X, method="ols") -> dict
def compute_icc(Y, icc_type="icc2") -> float
```

### algorithms/inference Public Functions

```python
def one_sample_permutation_test(data, n_permute=5000, tail=2, return_null=False, parallel="cpu", n_jobs=-1, max_gpu_memory_gb=4.0, random_state=None) -> dict
def two_sample_permutation_test(data1, data2, n_permute=5000, tail=2, return_null=False, parallel="cpu", n_jobs=-1, max_gpu_memory_gb=4.0, random_state=None) -> dict
def correlation_permutation_test(data1, data2, n_permute=5000, metric="pearson", tail=2, return_null=False, parallel="cpu", n_jobs=-1, max_gpu_memory_gb=4.0, random_state=None) -> dict
def matrix_permutation_test(data1, data2, n_permute=5000, metric="spearman", how="upper", include_diag=False, tail=2, return_null=False, parallel="cpu", n_jobs=-1, max_gpu_memory_gb=4.0, random_state=None) -> dict
def isc_permutation_test(data, n_permute=5000, metric="median", summary_statistic="pairwise", method="bootstrap", ...) -> dict
def isc_group_permutation_test(group1, group2, n_permute=5000, metric="median", method="permute", ...) -> dict
def timeseries_correlation_permutation_test(data1, data2, method="circle_shift", n_permute=5000, metric="pearson", tail=2, ...) -> dict
def circle_shift(data, shift=None, random_state=None) -> np.ndarray
def phase_randomize(data, random_state=None) -> np.ndarray
def double_center(mat) -> np.ndarray
def u_center(mat) -> np.ndarray
def distance_correlation(x, y) -> float
def compute_icc_voxelwise(data, icc_type="icc2") -> np.ndarray
```

### algorithms/ridge Public Functions

```python
def ridge_svd(X, y, alpha=1.0, parallel=None, max_gpu_memory_gb=4.0, random_state=None) -> np.ndarray
def ridge_cv(X, y, alphas=None, cv=5, parallel="cpu", max_gpu_memory_gb=4.0, random_state=None) -> dict
def solve_ridge_cv(X, Y, alphas, cv=5, backend="numpy") -> tuple
def solve_banded_ridge_cv(X, Y, alphas, n_features_per_band, cv=5, backend="numpy") -> tuple
```

### algorithms/hrf Public Functions

```python
def spm_hrf(tr, oversampling=16, time_length=32.0, onset=0.0) -> np.ndarray
def glover_hrf(tr, oversampling=16, time_length=32, onset=0.0) -> np.ndarray
def spm_time_derivative(tr, oversampling=16, time_length=32.0, onset=0.0) -> np.ndarray
def glover_time_derivative(tr, oversampling=16, time_length=32.0, onset=0.0) -> np.ndarray
def spm_dispersion_derivative(tr, oversampling=16, time_length=32.0, onset=0.0) -> np.ndarray
```

---

*Report generated by Claude Code audit on 2026-01-24*
