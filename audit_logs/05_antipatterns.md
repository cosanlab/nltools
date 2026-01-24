# Anti-Patterns and Code Smells Audit Report

**Date:** 2026-01-24
**Scope:** Large files (>500 lines) in nltools codebase
**Version Target:** v0.6.0

---

## Executive Summary

| Category | Count | Severity |
|----------|-------|----------|
| Long Methods (>50 lines) | 64 | Medium-High |
| God Methods (complexity >15) | 27 | High |
| Deep Nesting (depth >= 4) | 42 | Medium |
| Magic Numbers | 43 | Low-Medium |
| TODO/FIXME/HACK/XXX Comments | 3 | Low |
| Bare Except Clauses | 0 | N/A |
| Mutable Default Arguments | 0 | N/A |
| Unused Imports (ruff) | 0 | N/A |

**Key Finding:** The codebase is well-maintained with no bare except clauses, mutable defaults, or unused imports. Primary concerns are method length and complexity in core data classes.

---

## 1. Methods Over 50 Lines

Methods exceeding 50 lines are candidates for refactoring. These are ranked by line count.

### 1.1 brain_data.py (4,785 lines total)

| Method | Lines | Location | Severity | Notes |
|--------|-------|----------|----------|-------|
| `bootstrap` | 264 | L3093-3356 | HIGH | Complex statistical bootstrapping with GPU/CPU paths |
| `plot` | 211 | L4124-4334 | HIGH | Visualization with many rendering paths |
| `fit` | 196 | L1549-1744 | HIGH | Model fitting orchestration (Ridge/GLM) |
| `_detect_and_update_mask` | 162 | L282-443 | HIGH | Mask detection with nested image handling |
| `extract_roi` | 145 | L2492-2636 | MEDIUM | ROI extraction with multiple metrics |
| `_compute_ridge_cv` | 138 | L1776-1913 | MEDIUM | Cross-validated Ridge regression |
| `resample_to` | 135 | L1259-1393 | MEDIUM | Image resampling logic |
| `_load_from_list` | 117 | L525-641 | MEDIUM | List-based data loading |
| `compute_contrasts` | 114 | L2078-2191 | MEDIUM | GLM contrast computation |
| `plot_flatmap` | 110 | L4336-4445 | LOW | Specialized visualization |
| `threshold` | 106 | L2927-3032 | LOW | Thresholding logic |
| `_predict_mvpa` | 106 | L3804-3909 | MEDIUM | MVPA prediction pipeline |
| `__init__` | 100 | L113-212 | MEDIUM | Complex initialization |

**Suggested Fixes (High Severity):**
- `bootstrap`: Extract GPU bootstrap logic to `_bootstrap_gpu()` and CPU to `_bootstrap_cpu()`
- `plot`: Extract subplot logic to helper functions
- `fit`: Already delegates to `_fit_ridge()` and `_fit_glm()` - consider further decomposition

### 1.2 collection.py (5,054 lines total)

| Method | Lines | Location | Severity | Notes |
|--------|-------|----------|----------|-------|
| `fit_glm` | 275 | L3077-3351 | HIGH | Multi-subject GLM fitting |
| `fit_ridge` | 206 | L3652-3857 | HIGH | Multi-subject Ridge |
| `predict` | 159 | L3941-4099 | HIGH | Prediction orchestration |
| `_fit_glm` | 148 | L3491-3638 | MEDIUM | Internal GLM helper |
| `isc_test` | 145 | L2755-2899 | MEDIUM | Inter-subject correlation test |
| `_fit_glm_by_run` | 134 | L220-353 | MEDIUM | Run-level GLM fitting |
| `align` | 127 | L2169-2295 | MEDIUM | Hyperalignment orchestration |
| `isc` | 110 | L2643-2752 | LOW | ISC computation |
| `permutation_test2` | 109 | L1741-1849 | LOW | Two-sample permutation test |

### 1.3 adjacency/__init__.py (1,968 lines total)

| Method | Lines | Location | Severity | Notes |
|--------|-------|----------|----------|-------|
| `social_relations_model` | 287 | L1651-1937 | HIGH | Complex SRM analysis |
| `__init__` | 205 | L60-264 | HIGH | Complex multi-format initialization |
| `regress` | 156 | L1494-1649 | MEDIUM | Regression on adjacency data |
| `similarity` | 151 | L798-948 | MEDIUM | Similarity computation |

### 1.4 stats.py (2,426 lines total)

| Method | Lines | Location | Severity | Notes |
|--------|-------|----------|----------|-------|
| `align` | 200 | L1051-1250 | HIGH | Multi-subject alignment |
| `isc_group` | 114 | L1733-1846 | MEDIUM | Group-level ISC |
| `_transform_outliers` | 109 | L344-452 | MEDIUM | Outlier transformation |
| `compute_icc` | 109 | L2272-2380 | MEDIUM | ICC computation |

### 1.5 design_matrix.py (1,410 lines total)

| Method | Lines | Location | Severity | Notes |
|--------|-------|----------|----------|-------|
| `convolve` | 101 | L394-494 | MEDIUM | HRF convolution |
| `clean` | 100 | L952-1051 | MEDIUM | Design matrix cleaning |

---

## 2. God Methods (High Complexity)

Methods with complexity score >15 (branches + loops*2 + try*2 + returns).

### Critical (Complexity > 30)

| File | Method | Line | Complexity | Branches | Loops |
|------|--------|------|------------|----------|-------|
| adjacency | `social_relations_model` | 1651 | 37 | 19 | 4 |
| adjacency | `__init__` | 60 | 35 | 27 | 2 |
| collection | `fit_glm` | 3077 | 35 | 27 | 3 |
| stats | `align` | 1051 | 34 | 21 | 6 |
| adjacency | `similarity` | 798 | 31 | 19 | 2 |
| collection | `_aggregate_axis0` | 1089 | 30 | 15 | 5 |
| collection | `fit_ridge` | 3652 | 30 | 25 | 1 |

### High (Complexity 20-30)

| File | Method | Line | Complexity | Branches | Loops |
|------|--------|------|------------|----------|-------|
| brain_data | `plot` | 4124 | 27 | 19 | 1 |
| collection | `_fit_glm` | 3491 | 26 | 18 | 3 |
| brain_data | `fit` | 1549 | 25 | 19 | 2 |
| brain_data | `_load_from_list` | 525 | 24 | 18 | 3 |
| adjacency | `_import_single_data` | 401 | 23 | 18 | 0 |
| brain_data | `extract_roi` | 2492 | 22 | 19 | 1 |
| collection | `_getitem_multidim` | 748 | 20 | 14 | 0 |

---

## 3. Deep Nesting (>= 4 Levels)

Functions with nesting depth >= 4 indicate complex control flow.

### Critical (Depth >= 6)

| File | Function | Line | Depth |
|------|----------|------|-------|
| brain_data | `_load_from_list` | 525 | 8 |
| collection | `fit_glm` | 3077 | 8 |
| collection | `_fit_glm` | 3491 | 8 |
| adjacency | `__init__` | 60 | 8 |
| adjacency | `_import_single_data` | 401 | 8 |
| adjacency | `cluster_summary` | 1440 | 6 |
| brain_data | `__init__` | 113 | 6 |
| brain_data | `_detect_and_update_mask` | 282 | 6 |
| brain_data | `_shallow_copy_with_data` | 1118 | 6 |
| collection | `_aggregate_axis0` | 1089 | 6 |
| design_matrix | `__init__` | 70 | 6 |

**Suggested Fixes:**
- Extract nested conditionals into helper methods
- Use early returns to reduce nesting
- Consider strategy pattern for type-based dispatching

---

## 4. Magic Numbers

Numbers that should be named constants for clarity.

### High Priority (Domain-Specific Thresholds)

| File | Line | Value | Context | Suggested Constant |
|------|------|-------|---------|-------------------|
| brain_data | 3036 | 1350 | `min_region_size` | `MIN_REGION_SIZE_MM3` |
| brain_data | 3096 | 5000 | Bootstrap iterations | `DEFAULT_BOOTSTRAP_SAMPLES` |
| brain_data | 91 | 1000 | Unique label threshold | `MAX_ATLAS_LABELS` |
| brain_data | 3868 | 42 | `random_state` | (acceptable as convention) |
| brain_data | 3918 | 10000 | SVM `max_iter` | `SVM_MAX_ITER` |
| brain_data | 3919 | 1000 | Logistic `max_iter` | `LOGISTIC_MAX_ITER` |
| collection | 1636 | 5000 | Bootstrap samples | `DEFAULT_BOOTSTRAP_SAMPLES` |
| collection | 2723 | 0.9999 | Correlation threshold | `SINGULAR_CORRELATION_THRESHOLD` |
| stats | 131/160/183 | 0.05 | P-value threshold | `DEFAULT_ALPHA` |
| stats | 667/716/etc | 5000 | Permutation samples | `DEFAULT_PERMUTATION_SAMPLES` |
| design_matrix | 956 | 0.95 | Collinearity threshold | `DEFAULT_COLLINEARITY_THRESHOLD` |

### Medium Priority (Visualization/Formatting)

| File | Line | Value | Context |
|------|------|-------|---------|
| collection | 606-611 | 1e9/1e6/1e3 | Byte formatting (acceptable) |
| brain_data | 4509/4522/4528 | 0.3/0.7 | Plot layout ratios |

---

## 5. Technical Debt Markers

Only 3 TODO/FIXME/HACK/XXX comments found - excellent maintenance.

| File | Line | Type | Content |
|------|------|------|---------|
| pipelines/results.py | 143 | TODO | `Could average across folds or return per-fold results` |
| algorithms/ridge/backends/torch.py | 242 | XXX | `import numpy as np` (lazy import marker) |
| algorithms/ridge/backends/torch.py | 263 | XXX | `import numpy as np` (lazy import marker) |

**Assessment:** The XXX markers are intentional (lazy imports for optional dependency), and the TODO is a minor enhancement suggestion.

---

## 6. Other Findings

### Positive Findings (No Issues)

- **Bare Except Clauses:** 0 found - all exceptions are properly typed
- **Mutable Default Arguments:** 0 found - no `def foo(x=[])` patterns
- **Unused Imports:** ruff reports "All checks passed!" for audited files
- **Formatting:** Code is consistently formatted

### Code Duplication Candidates

Several patterns appear repeatedly that could be consolidated:

1. **Mask space checking:** `_check_space_match` pattern appears multiple times
2. **Progress bar setup:** Similar TQDM configuration in multiple methods
3. **CV integer resolution:** `StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)` pattern

---

## 7. Recommendations by Priority

### P0 - Critical (Address before v0.6.0)

1. **Split `social_relations_model`** (287 lines, complexity 37)
   - Extract `_estimate_srm()`, `_summarize_srm_results()` (already exist as helpers)
   - Move variance partitioning to separate function

2. **Refactor `Adjacency.__init__`** (205 lines, complexity 35)
   - Extract format-specific initialization to factory methods
   - Consider builder pattern for complex initialization

3. **Simplify `fit_glm`** (275 lines, depth 8)
   - Already has `_fit_glm` helper - further decompose
   - Extract subject iteration logic

### P1 - High (Address in v0.6.x)

4. **Extract constants to module-level**
   ```python
   # nltools/constants.py
   DEFAULT_BOOTSTRAP_SAMPLES = 5000
   DEFAULT_ALPHA = 0.05
   MIN_REGION_SIZE_MM3 = 1350
   ```

5. **Reduce nesting in `_load_from_list`** (depth 8)
   - Use early returns
   - Extract type-checking to helper

6. **Split `bootstrap`** method (264 lines)
   - Already has backend-specific code paths
   - Extract `_bootstrap_weights()`, `_bootstrap_predict()`

### P2 - Medium (Address in v0.7.0)

7. Consolidate duplicate patterns (mask checking, CV setup)
8. Add type hints to long methods for better documentation
9. Consider dataclasses for complex return types

---

## 8. Metrics Comparison

| Metric | Current | Target (v0.7.0) |
|--------|---------|-----------------|
| Methods >100 lines | 30 | <15 |
| Methods >50 lines | 64 | <40 |
| God methods (complexity >15) | 27 | <15 |
| Max nesting depth | 8 | 5 |
| Magic numbers | 43 | <20 |

---

## Appendix: Files Audited

| File | Lines | Long Methods | God Methods | Deep Nesting |
|------|-------|--------------|-------------|--------------|
| data/brain_data.py | 4,785 | 20 | 10 | 10 |
| data/collection.py | 5,054 | 18 | 6 | 14 |
| data/adjacency/__init__.py | 1,968 | 11 | 6 | 8 |
| stats.py | 2,426 | 12 | 2 | 6 |
| data/design_matrix.py | 1,410 | 8 | 2 | 4 |
| **Total** | **15,643** | **64** | **27** | **42** |
