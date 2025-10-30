# nltools v0.6.0 Refactoring Tasks

**Purpose**: Task checklist with progress tracking. Minimal prose, easy to scan.

For strategic vision, see `refactor-plan.md`. For context and decisions, see `refactor-progress.md`.

---

## Status Legend
- ✅ Complete
- 🔧 In Progress
- 📋 Planned
- 🔮 Future (post-v0.6.0)
- ⏸️ Deferred

---

## Priority 1: Core Refactoring

| Task | Status | Tests | Commit |
|------|--------|-------|--------|
| Delete Priority 3 files | ✅ | - | Multiple |
| Deprecation stubs (predict, ttest, randomise, predict_multi) | ✅ | - | Multiple |
| `.regress()` nilearn integration | ✅ | 7 | Multiple |
| `.compute_contrasts()` method | ✅ | 3 | c2a0929 |
| `.extract_roi()` refactor (NiftiLabelsMasker) | ✅ | - | Multiple |
| `.smooth()` returns copy | ✅ | - | Multiple |
| `.empty()` returns copy | ✅ | - | Multiple |
| HDF5 backward compatibility | ✅ | - | Multiple |

---

## Priority 1.5: Code Cleanup

| Task | Status | Tests | Commit |
|------|--------|-------|--------|
| `_shallow_copy_with_data()` implementation | ✅ | 14 | 69d4154 |
| Update 10 methods with efficient copying | ✅ | - | Multiple |
| `.shape()` → `@property` | ✅ | - | Multiple |
| `.isempty()` → `@property` | ✅ | - | Multiple |
| `.dtype()` → `@property` | ✅ | - | Multiple |
| Update ~90 property calls across codebase | ✅ | - | Multiple |

---

## Priority 2: Documentation & Tests

| Task | Status | Tests | Commit |
|------|--------|-------|--------|
| Test suite reorganization (shell/core/support) | ✅ | 317 | 5 commits |
| Sphinx → Jupyter Book migration | ✅ | - | Multiple |
| API documentation organization | ✅ | - | 1a4b1d9 |
| Tutorial updates | 📋 | - | - |

---

## Priority 2.5: Nilearn Enhancements

| Task | Status | Tests | Commit |
|------|--------|-------|--------|
| Research R² and effect_variance calculations | ✅ | - | - |
| `.threshold()` clustering enhancement | ✅ | 9 | 327080c |
| `.apply_mask()` nilearn migration | ✅ | 3 | 634eacb |
| `.filter()` docstring enhancement | ✅ | 2 | f004862 |
| Evaluate `.detrend()` and `.standardize()` | ✅ | - | - |

---

## Priority 2.6: fit/predict API

| Task | Status | Tests | Commit |
|------|--------|-------|--------|
| `fit(model='ridge'\|'glm')` implementation | ✅ | 11 | 472259b |
| `predict()` method | ✅ | 5 | 472259b |
| `.regress()` deprecation wrapper | ✅ | 7 | 472259b |
| Model-specific attributes (ridge_*, glm_*) | ✅ | - | 472259b |

---

## Priority 2.6.1: Cross-Validation

| Task | Status | Tests | Commit |
|------|--------|-------|--------|
| `cv=int` for K-fold CV | ✅ | 2 | 187c210 |
| `cv='auto'` for alpha selection | ✅ | 3 | 187c210 |
| `cv=sklearn_splitter` support | ✅ | 1 | 187c210 |
| `cv_results_` dict | ✅ | 2 | 187c210 |
| Out-of-fold predictions | ✅ | 1 | 187c210 |
| Error handling | ✅ | 2 | 187c210 |

---

## Priority 2.7: HyperAlignment

| Task | Status | Tests | Commit |
|------|--------|-------|--------|
| Extract HyperAlignment class | ✅ | 27 | Multiple |
| Sklearn-compatible API | ✅ | 27 | Multiple |
| `fit()`, `transform()`, `transform_subject()` | ✅ | 27 | Multiple |
| Integrate with `align(method='procrustes')` | ✅ | - | Multiple |
| Backward compatibility | ✅ | - | Multiple |

---

## Priority 2.7.1: SRM/DetSRM Testing

| Task | Status | Tests | Commit |
|------|--------|-------|--------|
| Research testing strategies | ✅ | - | - |
| Create test_srm.py | ✅ | 34 | f134854 |
| Initialization tests | ✅ | 4 | f134854 |
| Contract tests | ✅ | 8 | f134854 |
| Mathematical property tests | ✅ | 5 | f134854 |
| Edge case tests | ✅ | 6 | f134854 |
| DetSRM-specific tests | ✅ | 7 | f134854 |
| Comparative tests | ✅ | 4 | f134854 |

---

## Priority 2.8-2.12: Pre-Release Polish

| Task | Status | Tests | Commit |
|------|--------|-------|--------|
| Bootstrap refactoring | 📋 | 26 (planned) | - |
| Brain_Data → BrainData rename | 📋 | - | - |
| Round 1 codebase audit | 🔧 | - | ce3662d (partial) |
| Documentation & tutorials overhaul | 📋 | - | - |
| Migration guide in tabular format | 🔧 | - | - |

---

## Priority 2.13: Polars Migration Follow-up (IMMEDIATE)

| Task | Status | Tests | Commit |
|------|--------|-------|--------|
| Design_Matrix Polars TDD scaffolding | ✅ | 68 | 839f355 |
| Design_Matrix Polars implementation | ✅ | 68 | Pending commit |
| Review Polars code for holistic cleanup | 📋 | - | - |
| Switch design_matrix.py to Polars version | 📋 | - | - |
| Test with real workflows | 📋 | - | - |
| Update migration guide (Polars section) | 📋 | - | - |
| Consider pyarrow dependency | 📋 | - | - |
| Profile and benchmark Polars performance | 📋 | - | - |
| Remove unused _from_polars() method | 📋 | - | - |

---

## Priority 3: Medium Priority (v0.6.0 or v0.6.1)

| Task | Status | Tests | Commit |
|------|--------|-------|--------|
| `fit()` inplace parameter + Fit dataclass | 📋 | TBD | - |
| Adjacency refactoring | 📋 | 30+ | - |
| Plotting integration minimization | 📋 | TBD | - |

---

## Priority 4: Future (v0.7.0+)

| Task | Status | Tests | Commit |
|------|--------|-------|--------|
| Polars GPU Engine | 🔮 | - | - |
| Polars-native resampling (downsample/upsample) | 🔮 | - | - |
| Polars lazy evaluation by default | 🔮 | - | - |
| Replace HDF5 with Parquet for DataFrames | 🔮 | - | - |
| BrainCollection class design | 🔮 | - | - |
| Advanced ML workflows | 🔮 | - | - |
| Model class reimplementation | 🔮 | - | - |
| FastSRM implementation | 🔮 | - | - |
| Banded ridge regression | 🔮 | - | - |

---

## Deferred Items

| Task | Reason | Plan |
|------|--------|------|
| ISC dimension bug | Low usage, not blocker | See `claude-guidelines/align-isc-fix-plan.md` |
| Bootstrap with predict | Needs refactoring | See `claude-guidelines/bootstrap-refactor.md` |

---

## Test Count Summary

**Total**: 385 tests (378+ passing, ~4 skipped)
- **Shell**: 131 tests (Brain_Data: 71+, Adjacency: 54+, Design_Matrix: 10+ old)
- **Shell (New)**: 68 tests (Design_Matrix Polars: 68 passing, 100% complete)
- **Core**: 155 tests (including SRM: 34, HyperAlignment: 27, Ridge: 16, Models: 37)
- **Support**: 31 tests (datasets: 9, efficient_copy: 14, prefs: 5, simulator: 3)

---

*Last updated: 2025-10-29*
*Branch: uv-cleanup*
*Version: v0.6.0*
