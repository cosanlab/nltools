# Migration Plan: Replace stats.py Functions with inference Module

## Quick Status Summary

**Current Phase:** ✅ Phase 4 Complete → Migration Complete!

**Completed:** 
- ✅ Phase 1: `circle_shift`, `phase_randomize` (direct imports)
- ✅ Phase 2: `one_sample_permutation`, `two_sample_permutation`, `correlation_permutation`, `matrix_permutation` (wrapper functions)
- ✅ Phase 3: Documentation updates + ISC parameter additions
- ✅ Phase 4: ISC function migration
**Status:** All migration phases complete!

**Last Updated:** 2025-01-XX

---

## Overview

Replace permutation testing functions in `nltools/stats.py` with optimized implementations from `nltools/algorithms/inference`, ensuring 100% backward compatibility.

## Target Functions

| stats.py Function | inference Module Function | Status |
|------------------|--------------------------|--------|
| `one_sample_permutation` | `one_sample_permutation_test` | ✅ Ready |
| `two_sample_permutation` | `two_sample_permutation_test` | ✅ Ready |
| `correlation_permutation` | `correlation_permutation_test` + `timeseries_correlation_permutation_test` | ✅ Ready |
| `matrix_permutation` | `matrix_permutation_test` | ✅ Ready |
| `circle_shift` | `circle_shift` | ✅ **COMPLETE** |
| `phase_randomize` | `phase_randomize` | ✅ **COMPLETE** |
| `isc` | `isc_permutation_test` | ✅ **COMPLETE** |

## API Compatibility Analysis

### 1. `one_sample_permutation` → `one_sample_permutation_test`

**stats.py signature:**
```python
one_sample_permutation(
    data, 
    n_permute=5000, 
    tail=2, 
    n_jobs=-1, 
    return_perms=False, 
    random_state=None
)
# Returns: {'mean': ..., 'p': ..., 'perm_dist': ...}
```

**inference signature:**
```python
one_sample_permutation_test(
    data,
    n_permute=5000,
    tail=2,
    return_null=False,
    backend=None,
    n_jobs=-1,
    max_gpu_memory_gb=4.0,
    random_state=None
)
# Returns: {'mean': ..., 'p': ..., 'backend': ..., 'null_dist': ...}
```

**Differences:**
- ✅ `return_perms` → `return_null` (map parameter)
- ✅ `perm_dist` → `null_dist` (map return key)
- ✅ Additional `backend` key in return dict (OK, backward compatible)
- ✅ Additional `max_gpu_memory_gb` parameter (OK, has default)

**Compatibility:** ✅ **100% compatible** with wrapper

---

### 2. `two_sample_permutation` → `two_sample_permutation_test`

**stats.py signature:**
```python
two_sample_permutation(
    data1, data2,
    n_permute=5000,
    tail=2,
    n_jobs=-1,
    return_perms=False,
    random_state=None
)
# Returns: {'mean': ..., 'p': ..., 'perm_dist': ...}
```

**inference signature:**
```python
two_sample_permutation_test(
    data1, data2,
    n_permute=5000,
    tail=2,
    return_null=False,
    backend=None,
    n_jobs=-1,
    max_gpu_memory_gb=4.0,
    random_state=None
)
# Returns: {'mean_diff': ..., 'p': ..., 'backend': ..., 'null_dist': ...}
```

**Differences:**
- ⚠️ `'mean'` → `'mean_diff'` (need to map return key)
- ✅ `return_perms` → `return_null` (map parameter)
- ✅ `perm_dist` → `null_dist` (map return key)
- ✅ Additional `backend` key (OK, backward compatible)

**Compatibility:** ✅ **100% compatible** with wrapper (map `mean_diff` → `mean`)

---

### 3. `correlation_permutation` → `correlation_permutation_test` / `timeseries_correlation_permutation_test`

**stats.py signature:**
```python
correlation_permutation(
    data1, data2,
    method='permute',
    n_permute=5000,
    metric='spearman',
    tail=2,
    n_jobs=-1,
    return_perms=False,
    random_state=None
)
# Returns: {'correlation': ..., 'p': ..., 'perm_dist': ...}
```

**inference signatures:**
```python
# Standard permutation (method='permute')
correlation_permutation_test(
    data1, data2,
    n_permute=5000,
    metric='pearson',  # ⚠️ Different default!
    tail=2,
    return_null=False,
    backend=None,
    n_jobs=-1,
    max_gpu_memory_gb=4.0,
    random_state=None
)

# Time-series methods (method='circle_shift' or 'phase_randomize')
timeseries_correlation_permutation_test(
    x, y,
    method='circle_shift',
    n_permute=5000,
    metric='pearson',
    tail=2,
    return_null=False,
    backend=None,
    n_jobs=-1,
    random_state=None
)
```

**Differences:**
- ⚠️ Default `metric='spearman'` vs `metric='pearson'` (preserve stats.py default)
- ✅ `method` parameter supported (route to correct function)
- ✅ `return_perms` → `return_null` (map parameter)
- ✅ `perm_dist` → `null_dist` (map return key)

**Compatibility:** ✅ **100% compatible** with wrapper (preserve default, route by method)

---

### 4. `matrix_permutation` → `matrix_permutation_test`

**stats.py signature:**
```python
matrix_permutation(
    data1, data2,
    n_permute=5000,
    metric='spearman',  # ⚠️ Different default!
    how='upper',
    include_diag=False,
    tail=2,
    n_jobs=-1,
    return_perms=False,
    random_state=None
)
# Returns: {'correlation': ..., 'p': ..., 'perm_dist': ...}
```

**inference signature:**
```python
matrix_permutation_test(
    data1, data2,
    n_permute=5000,
    metric='pearson',  # ⚠️ Different default!
    how='upper',
    include_diag=False,
    tail=2,
    n_jobs=-1,
    return_null=False,
    random_state=None
)
# Returns: {'correlation': ..., 'p': ..., 'backend': ..., 'null_dist': ...}
```

**Differences:**
- ⚠️ Default `metric='spearman'` vs `metric='pearson'` (preserve stats.py default)
- ✅ `return_perms` → `return_null` (map parameter)
- ✅ `perm_dist` → `null_dist` (map return key)
- ✅ Additional `backend` key (OK, backward compatible)

**Compatibility:** ✅ **100% compatible** with wrapper (preserve default)

---

### 5. `circle_shift` → `circle_shift`

**stats.py signature:**
```python
circle_shift(data, random_state=None)
```

**inference signature:**
```python
circle_shift(
    data,
    shift_amount=None,
    random_state=None
)
```

**Differences:**
- ✅ Additional `shift_amount` parameter (OK, has default=None)
- ✅ Same function signature otherwise

**Compatibility:** ✅ **100% compatible** (direct import)

---

### 6. `phase_randomize` → `phase_randomize`

**stats.py signature:**
```python
phase_randomize(data, random_state=None)
```

**inference signature:**
```python
phase_randomize(data, random_state=None)
```

**Differences:**
- ✅ Identical signatures

**Compatibility:** ✅ **100% compatible** (direct import)

---

### 7. `isc` → `isc_permutation_test` (NEEDS REVIEW)

**stats.py signature:**
```python
isc(
    data,
    n_samples=5000,
    metric='median',
    method='bootstrap',
    ci_percentile=95,
    exclude_self_corr=True,
    return_null=False,
    tail=2,
    n_jobs=-1,
    random_state=None,
    sim_metric='correlation'
)
# Returns: {'isc': ..., 'p': ..., 'ci': ..., 'null_distribution': ...}
```

**inference signature:**
```python
isc_permutation_test(
    data,
    n_permute=5000,
    metric='median',
    summary_statistic='pairwise',  # ⚠️ Additional parameter
    method='bootstrap',
    ci_percentile=95,
    tail=2,
    backend=None,
    n_jobs=-1,
    max_memory_gb=4,
    random_state=None,
    return_null=False,
    progress_bar=True
)
# Returns: {'isc': ..., 'p': ..., 'ci': ..., 'null_distribution': ...}
```

**Differences:**
- ⚠️ `n_samples` → `n_permute` (parameter name change)
- ✅ `exclude_self_corr` parameter added (default=True, matches stats.py behavior)
- ✅ `sim_metric` parameter added (default='correlation', matches stats.py behavior)
- ✅ `summary_statistic` parameter added (has default 'pairwise', matches stats.py behavior)
- ✅ Additional `backend`, `max_memory_gb`, `progress_bar` parameters (OK, have defaults)

**Implementation Status:**
- ✅ Both `exclude_self_corr` and `sim_metric` parameters have been added to `isc_permutation_test`
- ✅ `exclude_self_corr` controls masking of self-correlations in bootstrap samples (default=True)
- ✅ `sim_metric` allows different similarity metrics via sklearn's pairwise_distances (default='correlation')
- ✅ GPU backend only supports `sim_metric='correlation'` (falls back to CPU with warning for other metrics)
- ✅ All ISC tests passing (24/24)

**Compatibility:** ✅ **100% compatible** - Migration complete with wrapper function

---

## Implementation Plan

### Phase 1: Simple Replacements (Direct Imports) ✅ COMPLETE

**Functions:** `circle_shift`, `phase_randomize`

**Steps:**
1. ✅ Added imports at top of `stats.py`:
   ```python
   from nltools.algorithms.inference.timeseries import circle_shift, phase_randomize
   ```
2. ✅ Removed local implementations (~60 lines)
3. ✅ Added tests: `test_circle_shift`, `test_phase_randomize`
4. ✅ Ran tests: All passing (13/13 in test_stats.py)
5. ✅ Verified backward compatibility with inference module tests

**Result:** ✅ **SUCCESS** - Zero issues, 100% backward compatible

**Files Modified:**
- `nltools/stats.py`: Added import, removed old implementations
- `nltools/tests/core/test_stats.py`: Added two new test functions

---

### Phase 2: Wrapper Functions (Parameter Mapping)

**Functions:** `one_sample_permutation`, `two_sample_permutation`, `correlation_permutation`, `matrix_permutation`

**Steps:**

#### 2.1 Create wrapper functions in `stats.py`:

```python
# Add imports
from nltools.algorithms.inference import (
    one_sample_permutation_test,
    two_sample_permutation_test,
    correlation_permutation_test,
    matrix_permutation_test,
)
from nltools.algorithms.inference.timeseries import timeseries_correlation_permutation_test

# Wrapper for one_sample_permutation
def one_sample_permutation(
    data, n_permute=5000, tail=2, n_jobs=-1, return_perms=False, random_state=None
):
    """One-sample permutation test (wrapper around inference module)."""
    result = one_sample_permutation_test(
        data,
        n_permute=n_permute,
        tail=tail,
        n_jobs=n_jobs,
        return_null=return_perms,  # Map parameter
        random_state=random_state,
    )
    # Map return keys
    output = {
        "mean": result["mean"],
        "p": result["p"],
    }
    if return_perms:
        output["perm_dist"] = result["null_dist"]
    return output

# Wrapper for two_sample_permutation
def two_sample_permutation(
    data1, data2, n_permute=5000, tail=2, n_jobs=-1, return_perms=False, random_state=None
):
    """Two-sample permutation test (wrapper around inference module)."""
    result = two_sample_permutation_test(
        data1,
        data2,
        n_permute=n_permute,
        tail=tail,
        n_jobs=n_jobs,
        return_null=return_perms,  # Map parameter
        random_state=random_state,
    )
    # Map return keys: 'mean_diff' -> 'mean'
    output = {
        "mean": result["mean_diff"],  # Map key
        "p": result["p"],
    }
    if return_perms:
        output["perm_dist"] = result["null_dist"]
    return output

# Wrapper for correlation_permutation
def correlation_permutation(
    data1,
    data2,
    method="permute",
    n_permute=5000,
    metric="spearman",  # Preserve stats.py default
    tail=2,
    n_jobs=-1,
    return_perms=False,
    random_state=None,
):
    """Correlation permutation test (wrapper around inference module)."""
    # Route to correct function based on method
    if method == "permute":
        result = correlation_permutation_test(
            data1,
            data2,
            n_permute=n_permute,
            metric=metric,
            tail=tail,
            n_jobs=n_jobs,
            return_null=return_perms,
            random_state=random_state,
        )
    else:  # circle_shift or phase_randomize
        result = timeseries_correlation_permutation_test(
            data1,
            data2,
            method=method,
            n_permute=n_permute,
            metric=metric,
            tail=tail,
            n_jobs=n_jobs,
            return_null=return_perms,
            random_state=random_state,
        )
    # Map return keys
    output = {
        "correlation": result["correlation"],
        "p": result["p"],
    }
    if return_perms:
        output["perm_dist"] = result["null_distribution"]  # Note: different key name
    return output

# Wrapper for matrix_permutation
def matrix_permutation(
    data1,
    data2,
    n_permute=5000,
    metric="spearman",  # Preserve stats.py default
    how="upper",
    include_diag=False,
    tail=2,
    n_jobs=-1,
    return_perms=False,
    random_state=None,
):
    """Matrix permutation test (wrapper around inference module)."""
    result = matrix_permutation_test(
        data1,
        data2,
        n_permute=n_permute,
        metric=metric,
        how=how,
        include_diag=include_diag,
        tail=tail,
        n_jobs=n_jobs,
        return_null=return_perms,  # Map parameter
        random_state=random_state,
    )
    # Map return keys
    output = {
        "correlation": result["correlation"],
        "p": result["p"],
    }
    if return_perms:
        output["perm_dist"] = result["null_dist"]
    return output
```

#### 2.2 Remove old implementations

#### 2.3 Run tests:
```bash
pytest nltools/tests/core/test_stats.py::test_permutation -v
pytest nltools/tests/core/test_stats.py::test_matrix_permutation -v
```

#### 2.4 Fix any issues:
- Check return value types (scalar vs array)
- Verify random_state handling
- Check edge cases (1D vs 2D inputs)

**Expected:** ✅ Minor fixes may be needed for return value shapes

---

### Phase 4: ISC Function ✅ COMPLETE (2025-01-XX)

**Completed:**
- ✅ Created wrapper function for `isc` that maps `n_samples` → `n_permute`
- ✅ Set `summary_statistic='pairwise'` explicitly to match original behavior
- ✅ Added import for `isc_permutation_test` from inference module
- ✅ Removed old implementation (~120 lines)
- ✅ Kept `_bootstrap_isc` helper function (still used by `adjacency.py` and `isc_group`)
- ✅ All tests passing (2/2 ISC tests in test_stats.py)

**Changes Made:**
- `nltools/stats.py`: 
  - Added import for `isc_permutation_test` from inference module
  - Replaced `isc` function with wrapper that calls `isc_permutation_test`
  - Removed old implementation (~120 lines)
  - Kept `_bootstrap_isc` helper (still needed by `adjacency.py` and `isc_group`)

**Test Results:**
- ✅ `test_isc`: PASSED
- ✅ `test_isc_group`: PASSED
- ✅ All ISC-related tests: PASSED (2/2)

**Implementation Details:**
- Wrapper maps `n_samples` parameter → `n_permute` in inference module
- Sets `summary_statistic='pairwise'` to match original behavior
- Sets `progress_bar=False` for backward compatibility
- All other parameters passed through unchanged
- Return format is identical (100% backward compatible)

**Next Steps:**
- Migration complete! All permutation testing functions now use inference module

---

## Testing Strategy

### Pre-Migration Tests (Baseline)

1. **Run existing tests:**
   ```bash
   pytest nltools/tests/core/test_stats.py -v
   ```

2. **Capture baseline results:**
   ```bash
   pytest nltools/tests/core/test_stats.py -v --tb=short > baseline_tests.log
   ```

3. **Run integration tests:**
   ```bash
   pytest nltools/tests/ -k "permutation" -v
   ```

### Post-Migration Tests

1. **Run same tests:**
   ```bash
   pytest nltools/tests/core/test_stats.py -v
   pytest nltools/tests/ -k "permutation" -v
   ```

2. **Compare results:**
   - Expected: All tests pass
   - Expected: ~1-2% variance in p-values (acceptable, documented in DESIGN.md)
   - Expected: Identical deterministic values (mean, correlation)

3. **Regression tests:**
   ```bash
   # Test with same seeds
   python -c "
   import numpy as np
   from nltools.stats import one_sample_permutation
   np.random.seed(42)
   data = np.random.randn(30)
   result = one_sample_permutation(data, n_permute=1000, random_state=42)
   print(f'Mean: {result[\"mean\"]:.6f}, p: {result[\"p\"]:.6f}')
   "
   ```

### Known Test Tolerances

From `test_inference.py`:
- **Backend consistency:** `rtol=1e-5` (exact match)
- **Backward compatibility:** `rtol=0.02` (2% variance acceptable)
- **GPU precision:** `rtol=1e-3` (0.1% variance)

---

## Rollback Plan

If issues arise:

1. **Git revert:**
   ```bash
   git revert <commit-hash>
   ```

2. **Alternative: Feature flag:**
   ```python
   USE_INFERENCE_MODULE = True  # Set to False to use old implementation
   ```

3. **Documentation:** Add note in changelog about migration

---

## Migration Progress

### ✅ Phase 1: COMPLETE (2025-01-XX)

**Completed:**
- ✅ Added imports for `circle_shift`, `phase_randomize` in `stats.py`
- ✅ Removed old implementations (~60 lines removed)
- ✅ Added tests: `test_circle_shift`, `test_phase_randomize` in `test_stats.py`
- ✅ All tests passing (13/13 in test_stats.py)
- ✅ Backward compatibility verified (inference module tests pass)
- ✅ Integration verified (`correlation_permutation` works with both methods)

**Changes Made:**
- `nltools/stats.py`: Added import `from .algorithms.inference.timeseries import circle_shift, phase_randomize`
- `nltools/stats.py`: Removed local implementations of `circle_shift` and `phase_randomize`
- `nltools/tests/core/test_stats.py`: Added `test_circle_shift()` and `test_phase_randomize()` functions

**Test Results:**
- ✅ `test_circle_shift`: PASSED
- ✅ `test_phase_randomize`: PASSED
- ✅ `test_matrix_permutation`: PASSED (uses these functions)
- ✅ All backward compatibility tests: PASSED

**Next Steps:**
- Proceed to Phase 2: Wrapper functions for permutation tests

---

### ✅ Phase 2: COMPLETE (2025-01-XX)

**Completed:**
- ✅ Created wrapper functions for all 4 permutation test functions
- ✅ Added imports for inference module functions
- ✅ Removed old implementations (~200 lines removed)
- ✅ Fixed one-tailed test bug in inference module (`_compute_pvalue`)
- ✅ All tests passing (16/16 in test_stats.py, 64/64 permutation tests)
- ✅ Backward compatibility verified (100% API compatible)
- ✅ Cleaned up unused imports (`fft`, `ifft`, `check_square_numpy_matrix`)

**Changes Made:**
- `nltools/stats.py`: 
  - Added imports for inference module functions
  - Replaced `one_sample_permutation`, `two_sample_permutation`, `correlation_permutation`, `matrix_permutation` with wrapper functions
  - Removed old implementations (~200 lines)
  - Removed unused imports
- `nltools/algorithms/inference/utils.py`: 
  - Fixed `_compute_pvalue` to handle negative statistics correctly for one-tailed tests (matches `stats.py` behavior)

**Test Results:**
- ✅ `test_permutation`: PASSED (one-sample, two-sample, correlation tests)
- ✅ `test_matrix_permutation`: PASSED
- ✅ All permutation-related tests: PASSED (64 tests)
- ✅ Full test_stats.py suite: PASSED (16 passed, 1 skipped)

**Issues Fixed:**
- One-tailed test bug: The inference module's `_compute_pvalue` was not handling negative statistics correctly for one-tailed tests. Fixed to match `stats.py` behavior: test `null >= obs` for positive stats, `null <= obs` for negative stats.

### Phase 3: Documentation ✅ COMPLETE (2025-01-XX)

**Completed:**
- ✅ Updated docstrings for all wrapper functions to mention inference module
- ✅ Added performance notes about speedups (4-8× CPU, 10-100× GPU)
- ✅ Added `exclude_self_corr` parameter to `isc_permutation_test` (default=True)
- ✅ Added `sim_metric` parameter to `isc_permutation_test` (default='correlation')
- ✅ Added comprehensive tests for both new ISC parameters (6 new tests)
- ✅ All tests passing (30/30 ISC tests, 1 skipped for GPU)

**Changes Made:**
- `nltools/stats.py`: Updated docstrings for wrapper functions with Notes sections
- `nltools/algorithms/inference/isc.py`: 
  - Added `exclude_self_corr` parameter to `isc_permutation_test` and bootstrap functions
  - Added `sim_metric` parameter to `isc_permutation_test` and `_compute_pairwise_isc`
  - Updated `_compute_pairwise_isc_numpy` to support different similarity metrics
  - Added GPU warning for non-correlation metrics
- `nltools/tests/core/test_isc.py`: Added 6 new tests for parameter validation

**Test Results:**
- ✅ `test_isc_exclude_self_corr_parameter`: PASSED
- ✅ `test_isc_exclude_self_corr_affects_bootstrap`: PASSED
- ✅ `test_isc_sim_metric_parameter`: PASSED
- ✅ `test_isc_sim_metric_affects_pairwise_computation`: PASSED
- ✅ `test_isc_exclude_self_corr_pairwise_only`: PASSED
- ✅ `test_isc_sim_metric_pairwise_only`: PASSED
- ✅ All ISC tests: PASSED (30 passed, 1 skipped)

**ISC Parameter Implementation:**
- `exclude_self_corr`: Controls masking of perfect correlations (1.0) from duplicate subjects in bootstrap samples. Default=True matches stats.py behavior.
- `sim_metric`: Allows different similarity metrics (correlation, euclidean, cosine, etc.) for pairwise ISC. Default='correlation' matches stats.py. GPU backend only supports 'correlation' (falls back to CPU with warning for other metrics).

**Next Steps:**
- ISC function is now ready for migration if desired (all parameters available)
- All Phase 3 documentation updates complete

---

## Checklist

### Phase 1: Simple Replacements
- [x] Add imports for `circle_shift`, `phase_randomize`
- [x] Remove old implementations
- [x] Add tests: `test_circle_shift`, `test_phase_randomize`
- [x] Run tests: `pytest nltools/tests/core/test_stats.py::test_permutation`
- [x] Verify no breakage
- [x] Verify backward compatibility with inference module tests

### Phase 2: Wrapper Functions ✅ COMPLETE
- [x] Create wrapper for `one_sample_permutation`
- [x] Create wrapper for `two_sample_permutation`
- [x] Create wrapper for `correlation_permutation`
- [x] Create wrapper for `matrix_permutation`
- [x] Remove old implementations
- [x] Run tests: `pytest nltools/tests/core/test_stats.py -v`
- [x] Fix any return value shape issues
- [x] Update docstrings (mention inference module)
- [x] Run full test suite: `pytest nltools/tests/ -k "permutation"`
- [x] Fix one-tailed test bug in inference module

### Phase 3: Documentation ✅ COMPLETE
- [x] Update docstrings to mention inference module
- [x] Add deprecation notices (if planning to deprecate wrapper)
- [x] Update migration guide
- [x] Update API documentation
- [x] Add missing ISC parameters (`exclude_self_corr`, `sim_metric`)
- [x] Add tests for new ISC parameters

### Phase 4: ISC Function ✅ COMPLETE (2025-01-XX)

**Completed:**
- ✅ Created wrapper function for `isc` that maps `n_samples` → `n_permute`
- ✅ Set `summary_statistic='pairwise'` explicitly to match original behavior
- ✅ Added import for `isc_permutation_test` from inference module
- ✅ Removed old implementation (~120 lines)
- ✅ Kept `_bootstrap_isc` helper function (still used by `adjacency.py` and `isc_group`)
- ✅ All tests passing (2/2 ISC tests in test_stats.py)

**Changes Made:**
- `nltools/stats.py`: 
  - Added import for `isc_permutation_test` from inference module
  - Replaced `isc` function with wrapper that calls `isc_permutation_test`
  - Removed old implementation (~120 lines)
  - Kept `_bootstrap_isc` helper (still needed by `adjacency.py` and `isc_group`)

**Test Results:**
- ✅ `test_isc`: PASSED
- ✅ `test_isc_group`: PASSED
- ✅ All ISC-related tests: PASSED (2/2)

**Implementation Details:**
- Wrapper maps `n_samples` parameter → `n_permute` in inference module
- Sets `summary_statistic='pairwise'` to match original behavior
- Sets `progress_bar=False` for backward compatibility
- All other parameters passed through unchanged
- Return format is identical (100% backward compatible)

**Next Steps:**
- Migration complete! All permutation testing functions now use inference module

---

## Expected Benefits

1. **Performance:**
   - 4-8× speedup for CPU-parallel operations
   - 10-100× speedup for GPU operations
   - Memory-efficient implementations

2. **Features:**
   - GPU acceleration support
   - Better error handling
   - Progress bars
   - Multi-feature support (voxel-wise)

3. **Code Quality:**
   - Well-tested implementations
   - Better documentation
   - Consistent API design

---

## Timeline Estimate

- **Phase 1:** 30 minutes (simple replacements)
- **Phase 2:** 2-3 hours (wrapper functions + testing)
- **Phase 3:** 1 hour (documentation)
- **Phase 4:** TBD (ISC function)

**Total:** ~4-5 hours for Phases 1-3

---

## Notes

1. **Backward Compatibility:** All wrappers preserve exact API signatures
2. **Performance:** Users get speedups automatically without code changes
3. **Testing:** Extensive test suite ensures correctness
4. **Documentation:** Update to mention inference module (optional GPU acceleration)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Return value shape differences | Test extensively, fix wrappers |
| Random seed differences | Use same RNG pattern (already done in inference) |
| Parameter name differences | Map in wrappers |
| Performance regression | Benchmark comparison (unlikely, inference is faster) |
| Breaking existing code | Comprehensive test suite, rollback plan |

