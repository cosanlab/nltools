# Critical Findings for nltools v0.6.0

**Audit Date:** 2026-01-24
**Status:** Items that must be addressed before v0.6.0 release

---

## P0 - Blockers (Must Fix)

### 1. Remove `pearson()` Function
**Location:** `nltools/stats.py:87-110`
**Issue:** Deprecated since v0.5.2, simply duplicates scipy functionality
**Action:** Delete function entirely
**Risk:** Low - users already warned, scipy alternatives documented

### 2. Add DeprecationWarning to `get_anatomical()`
**Location:** `nltools/utils.py:266-273`
**Issue:** Docstring says "DEPRECATED" but no runtime warning
**Action:** Add `warnings.warn(..., DeprecationWarning, stacklevel=2)`
**Risk:** None - just completing the deprecation

---

## P1 - High Priority (Should Fix)

### 3. Standardize Empty Check API
**Issue:** Three different patterns across classes

| Class | Current | Recommended |
|-------|---------|-------------|
| BrainData | `empty()` method + `isempty` property | `is_empty` property |
| Adjacency | `isempty` property | `is_empty` property |
| DesignMatrix | `empty` property | `is_empty` property |

**Action:**
1. Add `is_empty` property to all classes
2. Add deprecation warnings to old names
3. Update `BrainData.empty()` method to `create_empty()`

**Risk:** Medium - breaking change, needs migration period

---

### 4. Add `standardize()` to DesignMatrix
**Location:** `nltools/data/design_matrix.py`
**Issue:** Uses `zscore()` while BrainData/Collection use `standardize()`
**Action:** Add `standardize(method='zscore')` method, keep `zscore()` as alias
**Risk:** None - additive change

---

### 5. Add Missing Public Docstrings (7 items)
| File | Line | Item |
|------|------|------|
| `simulator.py` | 27 | `Simulator` class |
| `simulator.py` | 484 | `SimulateGrid` class |
| `brain_data.py` | 4632 | `BrainDataPipeline.cv` property |
| `brain_data.py` | 4636 | `BrainDataPipeline.n_steps` property |
| `utils.py` | 684 | `attempt_to_import` function |
| `utils.py` | 695 | `all_same` function |
| `utils.py` | 790 | `AmbiguityError` class |

**Action:** Add Google-style docstrings with Args/Returns sections
**Risk:** None - documentation only

---

### 6. Add Deprecation Warnings to stats.py Wrappers
**Functions to deprecate:**
- `one_sample_permutation` (line 666)
- `two_sample_permutation` (line 713)
- `correlation_permutation` (line 766)
- `matrix_permutation` (line 850)

**Action:** Add deprecation warnings directing users to inference module:
```python
warnings.warn(
    "one_sample_permutation is deprecated. Use "
    "nltools.algorithms.inference.one_sample_permutation_test instead.",
    DeprecationWarning,
    stacklevel=2
)
```

**Risk:** Low - wrapper functions, underlying functionality unchanged

---

## P2 - Consider for v0.6.0

### 7. Refactor `Adjacency.__init__`
**Location:** `nltools/data/adjacency/__init__.py:60-264`
**Issue:** 205 lines, complexity 35, nesting depth 8
**Action:** Extract format-specific initialization to factory methods
**Effort:** 2-3 hours
**Risk:** Medium - core initialization logic

### 8. Split `social_relations_model`
**Location:** `nltools/data/adjacency/__init__.py:1651-1937`
**Issue:** 287 lines, complexity 37
**Action:** Extract variance partitioning to helper function
**Effort:** 2 hours
**Risk:** Low - well-isolated method

### 9. Add Tests for Untested High-Priority Methods
| Method | Class | Priority |
|--------|-------|----------|
| `multivariate_similarity` | BrainData | High |
| `pipe` | BrainData, Collection | High |
| `pool` | Collection | High |
| `normalize` | BrainData, Collection | Medium |
| `reduce` | BrainData, Collection | Medium |

**Effort:** 4-6 hours for all
**Risk:** None - test additions only

---

## Verification Checklist

Before v0.6.0 release, verify:

- [ ] `pearson()` function removed from stats.py
- [ ] `get_anatomical()` emits DeprecationWarning
- [ ] `is_empty` property added to all data classes
- [ ] Deprecation warnings on old empty-check names
- [ ] `standardize()` added to DesignMatrix
- [ ] 7 missing docstrings added
- [ ] Stats.py wrapper deprecation warnings added
- [ ] CHANGELOG documents all API changes
- [ ] All tests pass

---

## Migration Guide (for CHANGELOG)

### Breaking Changes

1. **Empty Check Standardization**
   ```python
   # Old (deprecated, will be removed in v0.7.0)
   if brain_data.isempty:  # BrainData
   if adj.isempty:         # Adjacency
   if dm.empty:            # DesignMatrix

   # New (preferred)
   if brain_data.is_empty:
   if adj.is_empty:
   if dm.is_empty:
   ```

2. **Removed `pearson()` Function**
   ```python
   # Old
   from nltools.stats import pearson
   r = pearson(x, y)

   # New
   import numpy as np
   r = np.corrcoef(x, y)[0, 1]
   # Or for permutation testing:
   from nltools.algorithms.inference import correlation_permutation_test
   result = correlation_permutation_test(x, y)
   ```

3. **Deprecated Permutation Functions**
   ```python
   # Old (deprecated)
   from nltools.stats import one_sample_permutation
   result = one_sample_permutation(data)

   # New (preferred)
   from nltools.algorithms.inference import one_sample_permutation_test
   result = one_sample_permutation_test(data)
   ```

---

*Generated from codebase audit on 2026-01-24*
