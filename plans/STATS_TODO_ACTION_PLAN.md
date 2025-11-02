# Stats.py TODO Action Plan

## Overview
This document outlines the action plan for addressing all TODO comments in `nltools/stats.py`. The TODOs fall into several categories: deprecation/removal, refactoring to inference module, efficiency improvements, and usage verification.

**Recent Progress:**
- ✅ Phase 1.1: `_calc_pvalue` → `_compute_pvalue` migration completed
- ✅ Phase 2.1: Removed unused functions (`correlation`, `_permute_sign`, `_permute_func`)
- ✅ Phase 2.2: Removed `regress()` and `regress_permutation()` functions completely
- ✅ Phase 3.1: Matrix utilities moved to `inference/matrix.py` (`double_center`, `u_center`, `distance_correlation`)
- ✅ Removed handled TODO comments from code
- ✅ All ruff linting issues resolved
- ✅ Comprehensive test suite added (45+ tests including integration tests)

---

## Phase 1: Code Deduplication & Consolidation (High Priority)

### 1.1 Replace `_calc_pvalue` with `_compute_pvalue` from inference module
**Status**: ✅ Complete  
**Completed Actions**: 
- ✅ Replaced all usages in `stats.py` (`procrustes_distance`, `isc_group`)
- ✅ Updated `adjacency.py` to use `_compute_pvalue`
- ✅ Removed `_calc_pvalue` function definition
- ✅ Updated tests to use new function
- ✅ Added comprehensive test coverage

**Files Modified**:
- ✅ `nltools/stats.py`
- ✅ `nltools/data/adjacency.py`
- ✅ `nltools/tests/core/test_stats.py`

---

## Phase 2: Deprecation & Removal (Medium Priority)

**Status**: ✅ Complete

### 2.1 Remove unused helper functions
**Status**: ✅ Complete  
**Completed Actions**:
- ✅ Removed `correlation()` function - unused, replaced by `correlation_permutation_test`
- ✅ Removed `_permute_sign()` function - unused, replaced by `_generate_sign_flips`
- ✅ Removed `_permute_func()` function - unused, replaced by `matrix_permutation_test`
- ✅ Updated `__all__` exports to remove `correlation`
- ✅ Deprecated `pearson()` function - verified unused, added deprecation warning
- ✅ Removed `pearson` from `__all__` exports
- ✅ Removed `regress()` and `regress_permutation()` functions entirely
- ✅ Removed `_robust_estimator()` function (was only used by `regress()`)
- ✅ Removed ARMA functionality and `_arma_func()` function entirely
- ✅ Removed statsmodels optional dependency
- ✅ Removed `regress` from `__all__` exports
- ✅ Updated `Adjacency.regress()` to use inline OLS implementation

**Details**:

#### 2.1.1 `pearson()` function (Line 91)
- **Status**: ✅ Complete
- **Action**: Verified usage across codebase - function is unused
- **Resolution**: Added deprecation warning, removed from `__all__` exports
- **Migration path**: Use `scipy.stats.pearsonr` or `numpy.corrcoef` instead, or use `correlation_permutation_test` from inference module

#### 2.1.2 `_robust_estimator()` function (Line 897)
- **Status**: ✅ Removed
- **Action**: Removed along with `regress()` function - was only used internally
- **Resolution**: Function completely removed from codebase

#### 2.1.3 `_arma_func()` function (Line 986)
- **Status**: ✅ Removed
- **Action**: Removed ARMA functionality entirely - no longer needed
- **Resolution**: Removed `_arma_func()` function and all ARMA-related code from `regress()`

### 2.2 Remove optional dependencies comment
**Status**: ✅ Complete  
**TODO Location**: Line 87 (`# TODO: remove`)  
**Action**: 
- ✅ Removed statsmodels optional dependency import
- ✅ Removed `attempt_to_import` import (no longer used)
- **Resolution**: ARMA functionality and statsmodels dependency completely removed

---

## Phase 3: Refactor to Inference Module (Medium Priority)

**Status**: ✅ Complete

### 3.1 Matrix utility functions
**Status**: ✅ Complete  
**Completed Actions**:
- ✅ Moved `double_center()` to `nltools/algorithms/inference/matrix.py`
- ✅ Moved `u_center()` to `nltools/algorithms/inference/matrix.py`
- ✅ Moved `distance_correlation()` to `nltools/algorithms/inference/matrix.py`
- ✅ Added functions to `inference/__init__.py` exports
- ✅ Re-exported from `stats.py` for backward compatibility
- ✅ Added comprehensive test suite (18 tests)
- ✅ Enhanced documentation with type hints and examples

### 3.2 Deprecated functions replaced by inference module
**Status**: ✅ Complete  
**Actions**:

#### 3.2.1 `regress()` and `regress_permutation()` (Lines 1034, 1192)
- **Status**: ✅ Removed
- **Action**: Completely removed from `stats.py` module
- **Resolution**: Functions removed, `Adjacency.regress()` updated to use inline OLS implementation
- **Migration path**: Use `nltools.models.Glm` for neuroimaging GLM analysis, `BrainData.fit(model='glm')` for BrainData objects, or future inference module alternatives

#### 3.2.2 `align()` and `procrustes()` (Lines 1246, 1396)
- **Status**: ✅ Complete
- **Action**: Updated docstrings to reference `HyperAlignment` class
- **Resolution**: Documented as wrappers, kept for backward compatibility
- **Migration path**: Use `HyperAlignment` directly for Procrustes-based hyperalignment

### 3.3 Functions to migrate to inference module
**Status**: Needs investigation  
**Actions**:

#### 3.3.1 `isc_group()` (Line 1546)
- **TODO**: `# TODO: update to use inference/ module`
- **Action**: Check if inference module has equivalent functionality
- **Current status**: Function uses `_compute_pvalue` from inference module but may have other optimizations available
- **Recommendation**: Investigate if full migration to inference module is possible

#### 3.3.2 `isfc()` (Line 1701)
- **TODO**: `# TODO: update to use inference/ module`
- **Action**: Check if inference module has ISFC functionality
- **Current status**: Function uses leave-one-out approach for ISFC computation
- **Recommendation**: Investigate if inference module can provide optimized ISFC implementation

#### 3.3.3 `_compute_matrix_correlation()` (Line 1695)
- **TODO**: `# TODO: remove after rewriteing isc_group and isfc`
- **Action**: Remove after `isc_group()` and `isfc()` are migrated to inference module
- **Status**: Helper function used by `isfc()`, should be removed once migration is complete

## Phase 4: Efficiency Improvements (Low-Medium Priority)

### 4.1 Data transformation functions
**Status**: Requires profiling  
**Actions**:

#### 4.1.1 `downsample()` (Line 385)
- **TODO**: `# TODO: ensure efficient`
- **Action**: Profile current implementation
- **Potential improvements**: 
  - Use pandas `resample()` for time-series data
  - Optimize groupby operations
  - Consider using `scipy.signal.resample_poly` for signal processing

#### 4.1.2 `upsample()` (Line 429)
- **TODO**: `# TODO: ensure efficient`
- **Action**: Profile current implementation
- **Potential improvements**:
  - Optimize `interp1d` usage
  - Consider vectorized interpolation
  - Cache interpolation objects if called repeatedly

#### 4.1.3 `make_cosine_basis()` (Line 786)
- **TODO**: `# TODO: make efficient`
- **Action**: Profile current implementation
- **Potential improvements**:
  - Pre-compute constants
  - Consider using `scipy.fft.dct` if available
  - Vectorize loop if possible

#### 4.1.4 `transform_pairwise()` (Line 842)
- **TODO**: `# TODO: make efficient`
- **Action**: Profile current implementation
- **Potential improvements**:
  - Use vectorized operations
  - Consider using `sklearn.preprocessing` alternatives
  - Optimize itertools.combinations usage

#### 4.1.6 `isps()` (Line 1743)
- **TODO**: `# TODO: improve to avoid pandas type conversion use numpy or polars instead`
- **Action**: Refactor to avoid pandas type conversion
- **Potential improvements**:
  - Replace pandas DataFrame operations with numpy arrays
  - Consider using Polars if advanced DataFrame operations are needed
  - Remove unnecessary type conversions

### 4.2 Transform outliers functions
**Status**: Requires review  
**Actions**:

#### 4.2.1 `_transform_outliers()`, `winsorize()`, `trim()` (Lines 293, 260, 279)
- **TODO**: `# TODO: do we need this? can we refactor the function it supports to be more efficient?` (_transform_outliers)
- **TODO**: `# TODO: see related comment on _transform_outliers` (winsorize, trim)
- **Action**: Review implementation for efficiency
- **Potential improvements**:
  - Vectorize operations
  - Optimize pandas operations
  - Consider using `scipy.stats` functions if applicable

---

## Phase 5: Usage Verification (Low Priority)

### 5.1 Check if functions are deprecated by other modules
**Status**: Requires investigation  
**Actions**:

#### 5.1.1 `threshold()` (Line 175)
- **TODO**: `# TODO: check if deprecated given new method in BrainData that makes uses of nilearn + custom code`
- **Action**: Check if BrainData has new thresholding methods
- **Command**: Search `nltools/data/brain_data.py` for threshold methods
- **Recommendation**: Keep for backward compatibility, document BrainData alternatives

#### 5.1.2 `multi_threshold()` (Line 219)
- **TODO**: `# TODO: do we need this or does nilearn offer similar functionality already? Who uses it?`
- **Action**: Check if nilearn offers similar functionality
- **Command**: Search nilearn docs/API for thresholding
- **Recommendation**: If nilearn offers equivalent, add deprecation notice

#### 5.1.3 `summarize_bootstrap()` (Line 901)
- **TODO**: `# TODO: see how best to refactor and where to put this or if covered by other modules`
- **Action**: Check if covered by `inference/bootstrap.py`
- **Status**: `OnlineBootstrapStats` exists in inference module
- **Recommendation**: Compare functionality, potentially migrate or deprecate

---

## Phase 6: Documentation & Cleanup (Ongoing)

### 6.1 Update docstrings
- Add deprecation warnings to functions marked for removal
- Update references to inference module where applicable
- Clarify which functions are wrappers vs. core implementations

### 6.2 Update `__all__` exports
- Remove deprecated functions from `__all__`
- Add new functions from inference module if re-exported

### 6.3 Create migration guide
- Document migration path from old functions to inference module
- Add examples showing before/after usage

---

## Implementation Priority Summary

### Completed ✅
1. ✅ Replace `_calc_pvalue` with `_compute_pvalue` (Phase 1.1)
2. ✅ Verify and remove unused functions (Phase 2.1 - `correlation`, `_permute_sign`, `_permute_func`)
3. ✅ Deprecate `pearson()` function (Phase 2.1.1)
4. ✅ Remove ARMA functionality and statsmodels dependency (Phase 2.1.3, 2.2)
5. ✅ Remove `regress()` and `regress_permutation()` functions (Phase 2.2)
6. ✅ Remove `_robust_estimator()` function (Phase 2.2)
7. ✅ Move matrix utilities to inference module (Phase 3.1)
8. ✅ Document deprecated wrapper functions (Phase 3.2)
9. ✅ Removed handled TODO comments from code
10. ✅ Phase 2: Deprecation & Removal - Complete
11. ✅ Phase 3: Refactor to Inference Module - Complete

### Medium Priority (Do Next) ⚠️
1. ⚠️ Migration to inference module (Phase 3.3)
   - Investigate migrating `isc_group()` to inference module
   - Investigate migrating `isfc()` to inference module
   - Remove `_compute_matrix_correlation()` after migration complete
2. ⚠️ Usage verification for deprecation candidates (Phase 5)
   - Check `threshold()`, `multi_threshold()`, `summarize_bootstrap()`

### Low Priority (Do Later) 📊
1. 📊 Efficiency improvements (Phase 4)
   - Profile and optimize: `downsample()`, `upsample()`, `make_cosine_basis()`, `transform_pairwise()`, `find_spikes()`, `_transform_outliers()`
   - Improve `isps()` to avoid pandas type conversion - use numpy or polars instead
2. 📝 Documentation updates (Phase 6)

---

## Quick Reference: Function Status

| Function | Status | TODO Line | Action |
|----------|--------|-----------|--------|
| `pearson` | ✅ Deprecated | 91 | Deprecated, removed from `__all__`, use `scipy.stats.pearsonr` |
| `correlation` | ✅ Removed | - | Removed (was unused) |
| `_permute_sign` | ✅ Removed | - | Removed (was unused) |
| `_permute_func` | ✅ Removed | - | Removed (was unused) |
| `_calc_pvalue` | ✅ Replaced | - | Now uses `_compute_pvalue` from inference |
| `double_center` | ✅ Moved | - | Now in `inference/matrix.py`, re-exported |
| `u_center` | ✅ Moved | - | Now in `inference/matrix.py`, re-exported |
| `distance_correlation` | ✅ Moved | - | Now in `inference/matrix.py`, re-exported |
| `threshold` | ⚠️ Check | 175 | May be deprecated by BrainData |
| `multi_threshold` | ⚠️ Check | 219 | May be deprecated by nilearn |
| `winsorize` | 📊 Review | 260 | See `_transform_outliers` |
| `trim` | 📊 Review | 279 | See `_transform_outliers` |
| `_transform_outliers` | 📊 Review | 293 | Refactor for efficiency |
| `downsample` | 📊 Optimize | 385 | Profile and improve |
| `upsample` | 📊 Optimize | 429 | Profile and improve |
| `make_cosine_basis` | 📊 Optimize | 786 | Profile and improve |
| `transform_pairwise` | 📊 Optimize | 842 | Profile and improve |
| `summarize_bootstrap` | ⚠️ Check | 901 | Check if covered by inference module |
| `_robust_estimator` | ✅ Removed | - | Removed along with `regress()` function |
| `_arma_func` | ✅ Removed | - | Removed along with all ARMA functionality |
| `regress` | ✅ Removed | - | Completely removed - use `nltools.models.Glm` or `BrainData.fit(model='glm')` |
| `regress_permutation` | ✅ Removed | - | Completely removed - use inference module alternatives |
| `align` | ✅ Documented | 929 | Wrapper for HyperAlignment - see docstring for direct usage |
| `procrustes` | ✅ Documented | 1078 | Lower-level function used by HyperAlignment - see docstring |
| `find_spikes` | 📊 Optimize | 1214 | Profile and improve |
| `isc_group` | ⚠️ Migrate | 1546 | Update to use inference module |
| `_compute_matrix_correlation` | ⚠️ Remove | 1695 | Remove after migrating isc_group and isfc |
| `isfc` | ⚠️ Migrate | 1701 | Update to use inference module |
| `isps` | 📊 Improve | 1743 | Avoid pandas type conversion - use numpy or polars |
| `align_states` | ✅ Documented | 1916 | Uses Hungarian algorithm (different from HyperAlignment) - see docstring |

**Legend**:
- ✅ Complete/Removed
- ⚠️ Needs investigation/documentation
- 📊 Needs optimization/profiling

---

## Next Steps

### Immediate Next Steps (Priority Order)

1. **Investigate inference module migration** (Phase 3.3)
   - Check if `isc_group()` can be migrated to inference module
   - Check if `isfc()` can be migrated to inference module
   - Plan removal of `_compute_matrix_correlation()` helper function

2. **Usage verification for thresholding functions** (Phase 5.1)
   - Check if `threshold()` is deprecated by BrainData methods
   - Check if `multi_threshold()` is covered by nilearn
   - Compare `summarize_bootstrap()` with `OnlineBootstrapStats`

3. **Code quality improvements** (Phase 4)
   - Profile slow functions (`downsample`, `upsample`, `make_cosine_basis`, `transform_pairwise`, `find_spikes`)
   - Refactor `isps()` to avoid pandas type conversion
   - Optimize based on profiling results

4. **Documentation updates** (Phase 6)
   - Create migration guide for deprecated functions
   - Update API documentation
   - Add examples showing new vs. old usage patterns

---

## Notes

- All permutation test functions (`one_sample_permutation`, `two_sample_permutation`, etc.) are already wrappers around inference module - these are correct as-is
- Functions that are wrappers should remain for backward compatibility
- Consider a deprecation cycle (2-3 versions) before removing functions
- Test suite should be updated to reflect any changes
- **Test Coverage**: 45+ tests including integration tests
- **Code Quality**: All ruff linting issues resolved
- **Backward Compatibility**: All moved functions re-exported from `stats.py` to maintain existing imports
