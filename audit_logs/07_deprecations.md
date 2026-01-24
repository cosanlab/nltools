# Deprecation Audit for nltools v0.6.0

**Audit Date:** 2026-01-24
**Scope:** All source files in `nltools/`
**Target Version:** v0.6.0 (breaking release)

---

## Table of Contents

1. [Summary](#summary)
2. [Currently Deprecated Items](#currently-deprecated-items)
3. [Deprecation Candidates for v0.6.0](#deprecation-candidates-for-v060)
4. [Wrapper Functions Analysis](#wrapper-functions-analysis)
5. [Legacy Code Modernization](#legacy-code-modernization)
6. [Migration Recommendations](#migration-recommendations)

---

## Summary

| Category | Count |
|----------|-------|
| Currently deprecated (with warnings) | 5 |
| Candidates for deprecation | 8 |
| Wrapper functions to consolidate | 4 |
| Legacy patterns to modernize | 3 |

**Overall Assessment:** The codebase has a clean deprecation strategy with proper warnings. The main opportunity for v0.6.0 is to complete the migration of wrapper functions in `stats.py` to directly use the optimized `algorithms.inference` module functions.

---

## Currently Deprecated Items

### 1. `nltools.datasets.get_collection_image_metadata()`

**Location:** `/Users/esh/Documents/pypackages/nltools/nltools/datasets.py:446-461`

**Status:** Deprecated with warning, migration path documented

**Replacement:** `fetch_neurovault_collection()`

```python
warnings.warn(
    "get_collection_image_metadata is deprecated and will be removed in a future version. "
    "Please use fetch_neurovault_collection instead.",
    DeprecationWarning,
    stacklevel=2,
)
```

**Recommendation:** Remove in v0.7.0 or later. Keep for v0.6.0 to provide migration period.

---

### 2. `nltools.datasets.download_collection()`

**Location:** `/Users/esh/Documents/pypackages/nltools/nltools/datasets.py:464-480`

**Status:** Deprecated with warning, migration path documented

**Replacement:** `fetch_neurovault_collection()`

```python
warnings.warn(
    "download_collection is deprecated and will be removed in a future version. "
    "Please use fetch_neurovault_collection instead.",
    DeprecationWarning,
    stacklevel=2,
)
```

**Recommendation:** Remove in v0.7.0 or later. Keep for v0.6.0 to provide migration period.

---

### 3. `nltools.stats.pearson()`

**Location:** `/Users/esh/Documents/pypackages/nltools/nltools/stats.py:87-110`

**Status:** Deprecated since v0.5.2

**Replacement:**
- `scipy.stats.pearsonr` or `numpy.corrcoef` for basic correlation
- `correlation_permutation_test` from inference module for permutation testing

```python
warnings.warn(
    "pearson() is deprecated and will be removed in a future version. "
    "Use scipy.stats.pearsonr or numpy.corrcoef instead, or use "
    "correlation_permutation_test from the inference module.",
    DeprecationWarning,
    stacklevel=2,
)
```

**Recommendation:** **Remove in v0.6.0** - This has been deprecated since v0.5.2 and simply duplicates scipy functionality.

---

### 4. `BrainData.predict()` with `method="whole_brain"`

**Location:** `/Users/esh/Documents/pypackages/nltools/nltools/data/brain_data.py:3851-3861`

**Status:** Deprecated, suggests fluent API instead

**Replacement:** Fluent pipeline API

```python
warnings.warn(
    "predict(y=labels, cv=k) is deprecated for whole-brain MVPA. "
    "Use the fluent pipeline API instead:\n"
    "  result = brain.cv(k=5).predict(y=labels, algorithm='svm')\n"
    "The fluent API provides richer results (per-fold scores, predictions) "
    "and supports preprocessing chains (.normalize(), .reduce()).",
    DeprecationWarning,
    stacklevel=3,
)
```

**Recommendation:** Keep deprecation warning for v0.6.0, remove in v0.7.0 after users have migrated to fluent API.

---

### 5. `Adjacency.square_shape()`

**Location:** `/Users/esh/Documents/pypackages/nltools/nltools/data/adjacency/__init__.py:715-734`

**Status:** Deprecated with removal target of v0.7.0

**Replacement:** `.shape` property

```python
warnings.warn(
    "square_shape() is deprecated. Use .shape instead, which now returns "
    "(n_nodes, n_nodes) for single matrices or (n_matrices, n_nodes, n_nodes) "
    "for stacked matrices.",
    DeprecationWarning,
    stacklevel=2,
)
```

**Recommendation:** Keep for v0.6.0, remove in v0.7.0 as documented.

---

## Deprecation Candidates for v0.6.0

### High Priority - Should Deprecate

#### 1. `nltools.stats.one_sample_permutation()`

**Location:** `/Users/esh/Documents/pypackages/nltools/nltools/stats.py:666-710`

**Reason:** This is a wrapper around `nltools.algorithms.inference.one_sample_permutation_test` for backward compatibility. The wrapper adds overhead and parameter name mapping.

**Current Implementation:**
```python
def one_sample_permutation(
    data, n_permute=5000, tail=2, n_jobs=-1, return_perms=False, random_state=None
):
    """This function is a wrapper around `nltools.algorithms.inference.one_sample_permutation_test`
    for backward compatibility..."""
```

**Recommendation:**
- Add `DeprecationWarning` in v0.6.0
- Direct users to `nltools.algorithms.inference.one_sample_permutation_test`
- Remove in v0.7.0

---

#### 2. `nltools.stats.two_sample_permutation()`

**Location:** `/Users/esh/Documents/pypackages/nltools/nltools/stats.py:713-763`

**Reason:** Same as above - wrapper for backward compatibility.

**Recommendation:** Deprecate in v0.6.0, remove in v0.7.0.

---

#### 3. `nltools.stats.correlation_permutation()`

**Location:** `/Users/esh/Documents/pypackages/nltools/nltools/stats.py:766-847`

**Reason:** Wrapper around inference module functions with parameter name translation.

**Recommendation:** Deprecate in v0.6.0, remove in v0.7.0.

---

#### 4. `nltools.stats.matrix_permutation()`

**Location:** `/Users/esh/Documents/pypackages/nltools/nltools/stats.py:850-916`

**Reason:** Wrapper around `matrix_permutation_test` from inference module.

**Recommendation:** Deprecate in v0.6.0, remove in v0.7.0.

---

### Medium Priority - Consider Deprecating

#### 5. `nltools.utils.get_anatomical()`

**Location:** `/Users/esh/Documents/pypackages/nltools/nltools/utils.py:266-273`

**Reason:** Returns hardcoded path to MNI brain. The `MNI_Template` class provides a more flexible solution.

**Recommendation:** Evaluate if still used, consider deprecation.

---

#### 6. `nltools.stats.align()`

**Location:** `/Users/esh/Documents/pypackages/nltools/nltools/stats.py:1051-1250`

**Reason:** Large function that wraps `HyperAlignment` and `SRM` classes. The classes themselves provide cleaner interfaces.

**Recommendation:** Consider deprecating in v0.6.0, directing users to use `HyperAlignment` and `SRM` classes directly for clearer intent.

---

#### 7. ISC Functions in `stats.py`

**Functions:**
- `isc()` (line 1519)
- `isc_group()` (line 1733)
- `isfc()` (line 1849)
- `isps()` (line 1931)
- `_bootstrap_isc()` (line 1468)

**Location:** `/Users/esh/Documents/pypackages/nltools/nltools/stats.py`

**Reason:** These may duplicate functionality in `algorithms.inference.isc` module.

**Recommendation:** Audit for overlap with inference module. If duplicated, deprecate `stats.py` versions.

---

#### 8. `nltools.stats.procrustes()`

**Location:** `/Users/esh/Documents/pypackages/nltools/nltools/stats.py:1253-1337`

**Reason:** Thin wrapper around `scipy.spatial.procrustes` and `scipy.linalg.orthogonal_procrustes`. The docstring already suggests using `HyperAlignment` and `align()` for comprehensive tasks.

**Recommendation:** Consider deprecation in favor of direct scipy usage or `HyperAlignment` class.

---

## Wrapper Functions Analysis

The following functions in `stats.py` are **purely wrapper functions** that delegate to the inference module with minimal value-add:

| Function | Wraps | Lines of Wrapper Code | Migration Path |
|----------|-------|----------------------|----------------|
| `one_sample_permutation` | `one_sample_permutation_test` | ~45 | Direct use of inference module |
| `two_sample_permutation` | `two_sample_permutation_test` | ~50 | Direct use of inference module |
| `correlation_permutation` | `correlation_permutation_test` / `timeseries_correlation_permutation_test` | ~80 | Direct use of inference module |
| `matrix_permutation` | `matrix_permutation_test` | ~66 | Direct use of inference module |

**Total wrapper code:** ~240 lines that could be removed after deprecation period.

### Wrapper Function Value Analysis

These wrappers provide:
1. **Parameter name mapping** (e.g., `return_perms` -> `return_null`)
2. **Return key mapping** (e.g., `perm_dist` vs `null_dist`)
3. **Backward compatibility** for existing code

**Breaking Changes if Removed:**
- Users would need to update parameter names
- Return dictionary key names differ slightly
- Import paths would change

---

## Legacy Code Modernization

### 1. Old String Formatting (% and .format())

**Files with `%` formatting:**
- `nltools/data/brain_data.py:934` - `__repr__` method
- `nltools/data/adjacency/__init__.py:267` - `__repr__` method
- `nltools/stats.py:1108` - Error message
- `nltools/algorithms/srm.py:554, 1035` - Logger messages
- `nltools/plotting.py:197, 299, 301, 303, 305` - Print statements
- `nltools/tests/conftest.py:157, 180, 194` - Test labels

**Files with `.format()`:**
- `nltools/data/adjacency/__init__.py:1345` - Error message
- `nltools/analysis.py:317-326` - Print statements
- `nltools/algorithms/srm.py:248, 257, 800, 809` - Error messages

**Recommendation:** Convert to f-strings for consistency. Low priority for v0.6.0 but good for code quality.

---

### 2. Old Typing Patterns

**Files using `Optional[X]` instead of `X | None`:**
- Extensive use across the codebase (see full list in grep results)
- Modern Python (3.10+) supports `X | None` syntax

**Files using `Union[X, None]`:**
- `nltools/data/design_matrix.py:72, 954`

**Files using `from __future__ import annotations`:**
- `nltools/data/collection.py`
- `nltools/cache.py`
- `nltools/neighborhoods.py`
- `nltools/pipelines/*.py`
- `nltools/algorithms/alignment/_local.py`

**Recommendation:** For v0.6.0, consider:
1. Adding `from __future__ import annotations` to all files
2. Migrating to `X | None` syntax where PEP 604 is supported
3. This is a low-risk modernization that improves readability

---

### 3. Python 2 Compatibility Code

**Finding:** No Python 2 compatibility code detected. The `from __future__ import annotations` imports are for modern PEP 563 (postponed evaluation) support, not Python 2 compatibility.

**Status:** Clean - no action needed.

---

## Migration Recommendations

### For v0.6.0 Release

1. **Add deprecation warnings** to the four wrapper functions in `stats.py`:
   - `one_sample_permutation`
   - `two_sample_permutation`
   - `correlation_permutation`
   - `matrix_permutation`

2. **Remove** `pearson()` function - deprecated since v0.5.2

3. **Keep** existing deprecation warnings for:
   - `get_collection_image_metadata`
   - `download_collection`
   - `Adjacency.square_shape`
   - `BrainData.predict()` with whole_brain method

4. **Document** migration paths in release notes

### For v0.7.0 Release

1. **Remove** deprecated dataset functions
2. **Remove** deprecated stats.py wrapper functions
3. **Remove** `Adjacency.square_shape()`
4. **Remove** deprecated `BrainData.predict()` whole_brain method

### Code Modernization (Ongoing)

1. Convert `%` and `.format()` strings to f-strings
2. Standardize typing annotations with PEP 604 syntax
3. Add `from __future__ import annotations` to remaining files

---

## Appendix: All warnings.warn Calls

For reference, here are all warning sites that are NOT deprecation warnings:

| File | Line | Type | Purpose |
|------|------|------|---------|
| `plotting.py` | 88 | None specified | Percentile threshold warning |
| `backends.py` | 80, 125, 259 | UserWarning | MPS/GPU precision warnings |
| `mask.py` | 180 | None | Collapse warning |
| `utils.py` | 472, 479 | None | Algorithm setup warnings |
| `data/brain_data.py` | 403, 523, 761, 769, 811, 4305 | Various | Data loading warnings |
| `data/adjacency/__init__.py` | 148, 176, 861 | Various | Loading/parsing warnings |
| `algorithms/inference/timeseries.py` | 168 | None | Method warning |
| `algorithms/inference/utils.py` | 349 | None | Batch size warning |
| `algorithms/ridge/backends/_utils.py` | 59, 159 | None | Backend warnings |
| `algorithms/ridge/solvers.py` | 289 | None | Solver warning |
| `algorithms/inference/bootstrap.py` | 51, 79 | None | Bootstrap warnings |

These are operational warnings, not deprecations, and should be preserved.
