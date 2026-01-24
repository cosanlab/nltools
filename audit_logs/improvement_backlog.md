# Improvement Backlog for nltools

**Generated:** 2026-01-24
**Source:** Comprehensive codebase audit

---

## Priority Legend

| Priority | Definition | Target Release |
|----------|------------|----------------|
| P0 | Blocker - must fix before release | v0.6.0 |
| P1 | High - should fix soon | v0.6.x |
| P2 | Medium - address when convenient | v0.7.0 |
| P3 | Low - nice to have | Future |

---

## API Consistency

### P0 - Blockers

| ID | Issue | File | Line | Action |
|----|-------|------|------|--------|
| API-001 | Remove deprecated `pearson()` | stats.py | 87-110 | Delete function |
| API-002 | Add DeprecationWarning to `get_anatomical()` | utils.py | 266 | Add warning |

### P1 - High Priority

| ID | Issue | File | Line | Action |
|----|-------|------|------|--------|
| API-003 | Standardize `is_empty` property | brain_data.py, adjacency, design_matrix.py | Various | Add property, deprecate old |
| API-004 | Add `standardize()` to DesignMatrix | design_matrix.py | - | Add method |
| API-005 | Add deprecation warnings to stats.py wrappers | stats.py | 666, 713, 766, 850 | Add warnings |
| API-006 | Make `_to_pandas()` public | design_matrix.py | 1238 | Rename to `to_pandas()` |

### P2 - Medium Priority

| ID | Issue | File | Line | Action |
|----|-------|------|------|--------|
| API-007 | Add `to_square()` alias | adjacency/__init__.py | 496 | Add alias for `squareform()` |
| API-008 | Add `write()` to DesignMatrix | design_matrix.py | - | Implement method |
| API-009 | Add `write()` to Collection | collection.py | - | Implement method |
| API-010 | Standardize `verbose` to bool | Various | Various | Convert int to bool |

---

## Type Annotations

### P1 - High Priority (Quick Wins)

| ID | File | Functions | Current | Action |
|----|------|-----------|---------|--------|
| TYPE-001 | algorithms/inference/matrix.py | 9 | 89% | Complete 1 function |
| TYPE-002 | algorithms/inference/icc.py | 7 | 86% | Complete 1 function |
| TYPE-003 | algorithms/hrf.py | 6 | 83% | Complete 1 function |
| TYPE-004 | algorithms/_random.py | 4 | 75% | Complete 1 function |
| TYPE-005 | file_reader.py | 1 | 0% | Type 1 function |

### P2 - Medium Priority (Public APIs)

| ID | File | Functions | Current | Action |
|----|------|-----------|---------|--------|
| TYPE-006 | data/adjacency/__init__.py | 61 | 0% | Type all public methods |
| TYPE-007 | stats.py | 42 | 0% | Type all functions |
| TYPE-008 | utils.py | 24 | 0% | Type all functions |
| TYPE-009 | simulator.py | 23 | 0% | Type all methods |
| TYPE-010 | data/brain_data.py | 114 | 4% | Type public methods |

### P3 - Low Priority

| ID | File | Functions | Current | Action |
|----|------|-----------|---------|--------|
| TYPE-011 | models/*.py | 21 | 0% | Type all methods |
| TYPE-012 | plotting.py | 19 | 0% | Type all functions |
| TYPE-013 | algorithms/ridge/backends/*.py | 41 | 0% | Type all functions |

---

## Documentation

### P0 - Missing Docstrings (Public API)

| ID | File | Line | Item | Action |
|----|------|------|------|--------|
| DOC-001 | simulator.py | 27 | `Simulator` class | Add class docstring |
| DOC-002 | simulator.py | 484 | `SimulateGrid` class | Add class docstring |
| DOC-003 | brain_data.py | 4632 | `BrainDataPipeline.cv` | Add property docstring |
| DOC-004 | brain_data.py | 4636 | `BrainDataPipeline.n_steps` | Add property docstring |
| DOC-005 | utils.py | 684 | `attempt_to_import` | Add function docstring |
| DOC-006 | utils.py | 695 | `all_same` | Add function docstring |
| DOC-007 | utils.py | 790 | `AmbiguityError` | Add class docstring |

### P1 - Incomplete Docstrings

| ID | File | Issue | Count | Action |
|----|------|-------|-------|--------|
| DOC-008 | Various | Missing Args sections | 77 | Add Args sections |
| DOC-009 | Various | Missing Returns sections | 171 | Add Returns sections |
| DOC-010 | brain_data.py | `BrainData` class missing example | 1 | Add usage example |
| DOC-011 | adjacency/__init__.py | `Adjacency` class missing example | 1 | Add usage example |

---

## Code Quality (Anti-Patterns)

### P1 - High Severity

| ID | File | Method | Lines | Issue | Action |
|----|------|--------|-------|-------|--------|
| QUAL-001 | adjacency/__init__.py | `social_relations_model` | 287 | Too long, complexity 37 | Extract helpers |
| QUAL-002 | adjacency/__init__.py | `__init__` | 205 | Complexity 35, depth 8 | Extract factory methods |
| QUAL-003 | collection.py | `fit_glm` | 275 | Complexity 35, depth 8 | Further decompose |
| QUAL-004 | brain_data.py | `bootstrap` | 264 | Too long | Extract GPU/CPU helpers |
| QUAL-005 | brain_data.py | `plot` | 211 | Complexity 27 | Extract subplot logic |

### P2 - Medium Severity

| ID | File | Method | Issue | Action |
|----|------|--------|-------|--------|
| QUAL-006 | brain_data.py | `_load_from_list` | Depth 8 | Use early returns |
| QUAL-007 | collection.py | `fit_ridge` | Complexity 30 | Decompose |
| QUAL-008 | Various | Magic numbers | 43 instances | Extract to constants |

### P2 - Consolidation Opportunities

| ID | Issue | Location | Lines Saved | Action |
|----|-------|----------|-------------|--------|
| CONS-001 | Duplicate arithmetic methods | adjacency/__init__.py | 70 | Add `_perform_arithmetic` |
| CONS-002 | Duplicate stat methods | adjacency/__init__.py | 50 | Add `_apply_func` |
| CONS-003 | Duplicate Fisher transforms | brain_data.py, adjacency | 15 | Create mixin |
| CONS-004 | Duplicate H5 detection | Various | 10 | Add `is_h5_path()` |

---

## Test Coverage

### P1 - Untested High-Priority Methods

| ID | Class | Method | Priority |
|----|-------|--------|----------|
| TEST-001 | BrainData | `multivariate_similarity` | High |
| TEST-002 | BrainData | `pipe` | High |
| TEST-003 | Collection | `pipe` | High |
| TEST-004 | Collection | `pool` | High |

### P2 - Untested Medium-Priority Methods

| ID | Class | Method | Priority |
|----|-------|--------|----------|
| TEST-005 | BrainData | `normalize` | Medium |
| TEST-006 | Collection | `normalize` | Medium |
| TEST-007 | BrainData | `reduce` | Medium |
| TEST-008 | Collection | `reduce` | Medium |
| TEST-009 | Collection | `smooth` | Medium |
| TEST-010 | Collection | `unload` | Medium |
| TEST-011 | BrainData | `transform_pairwise` | Medium |

### P3 - Visualization Tests

| ID | Class | Method | Notes |
|----|-------|--------|-------|
| TEST-012 | Adjacency | `plot` | Consider pytest-mpl |
| TEST-013 | Adjacency | `plot_mds` | Consider pytest-mpl |
| TEST-014 | Adjacency | `plot_silhouette` | Consider pytest-mpl |
| TEST-015 | Adjacency | `plot_label_distance` | Consider pytest-mpl |

---

## Deprecations

### P0 - Complete Deprecations

| ID | Item | File | Status | Action |
|----|------|------|--------|--------|
| DEP-001 | `pearson()` | stats.py | Deprecated v0.5.2 | Remove |
| DEP-002 | `get_anatomical()` | utils.py | Docstring only | Add warning |

### P1 - Add Deprecation Warnings

| ID | Item | File | Line | Replacement |
|----|------|------|------|-------------|
| DEP-003 | `one_sample_permutation` | stats.py | 666 | `one_sample_permutation_test` |
| DEP-004 | `two_sample_permutation` | stats.py | 713 | `two_sample_permutation_test` |
| DEP-005 | `correlation_permutation` | stats.py | 766 | `correlation_permutation_test` |
| DEP-006 | `matrix_permutation` | stats.py | 850 | `matrix_permutation_test` |

### P2 - Future Deprecations (v0.7.0)

| ID | Item | File | Replacement |
|----|------|------|-------------|
| DEP-007 | `isempty` property | brain_data.py, adjacency | `is_empty` |
| DEP-008 | `empty` property | design_matrix.py | `is_empty` |
| DEP-009 | `Adjacency.square_shape()` | adjacency/__init__.py | `.shape` |
| DEP-010 | `BrainData.predict()` whole_brain | brain_data.py | fluent API |

---

## Modernization

### P3 - String Formatting

| ID | File | Lines | Issue |
|----|------|-------|-------|
| MOD-001 | brain_data.py | 934 | `%` formatting |
| MOD-002 | adjacency/__init__.py | 267 | `%` formatting |
| MOD-003 | stats.py | 1108 | `%` formatting |
| MOD-004 | plotting.py | 197, 299-305 | `%` formatting |
| MOD-005 | algorithms/srm.py | 554, 1035 | `%` formatting |

### P3 - Type Annotation Syntax

| ID | Issue | Current | Target |
|----|-------|---------|--------|
| MOD-006 | Optional syntax | `Optional[X]` (144 uses) | `X \| None` |
| MOD-007 | Union syntax | `Union[X, None]` (2 uses) | `X \| None` |
| MOD-008 | Missing future import | 52 files | `from __future__ import annotations` |

---

## Summary Counts

| Category | P0 | P1 | P2 | P3 | Total |
|----------|----|----|----|----|-------|
| API Consistency | 2 | 4 | 4 | 0 | 10 |
| Type Annotations | 0 | 5 | 5 | 3 | 13 |
| Documentation | 7 | 4 | 0 | 0 | 11 |
| Code Quality | 0 | 5 | 6 | 0 | 11 |
| Test Coverage | 0 | 4 | 7 | 4 | 15 |
| Deprecations | 2 | 4 | 4 | 0 | 10 |
| Modernization | 0 | 0 | 0 | 8 | 8 |
| **Total** | **11** | **26** | **26** | **15** | **78** |

---

## Effort Estimates

| Priority | Items | Estimated Effort |
|----------|-------|------------------|
| P0 (Blockers) | 11 | 4-6 hours |
| P1 (High) | 26 | 2-3 days |
| P2 (Medium) | 26 | 1-2 weeks |
| P3 (Low) | 15 | Ongoing |

---

*Generated from comprehensive codebase audit on 2026-01-24*
