# nltools v0.6.0 Codebase Audit - Executive Summary

**Audit Date:** 2026-01-24
**Codebase:** 121 Python files, ~83K lines of code
**Target Release:** v0.6.0 (breaking release)

---

## Overall Assessment: B+ (Good with Room for Improvement)

The nltools codebase is well-maintained with excellent practices in some areas and opportunities for improvement in others.

| Category | Grade | Summary |
|----------|-------|---------|
| **API Consistency** | B | Some inconsistencies across shell classes |
| **Documentation** | B+ | 99% docstring presence, incomplete Args/Returns |
| **Type Annotations** | C+ | 41% coverage, inconsistent patterns |
| **Code Quality** | A- | No bare excepts, mutable defaults, or unused imports |
| **Test Coverage** | B+ | 80% method coverage (static analysis) |
| **Deprecation Strategy** | A | Clear deprecation paths, proper warnings |

---

## Key Metrics

### Codebase Overview
| Metric | Value |
|--------|-------|
| Python Files | 121 |
| Total Lines | ~83,000 |
| Functions Analyzed | 852 |
| Test Files | 25+ |

### API Consistency (Imperative Shell)
| Issue Type | Count |
|------------|-------|
| Naming inconsistencies | 10 |
| Property vs method mismatches | 2 |
| Return type inconsistencies | 4 |

### Documentation
| Metric | Value |
|--------|-------|
| Public methods with docstrings | 99.4% |
| Missing Args sections | 77 |
| Missing Returns sections | 171 |
| Classes missing examples | 6 |

### Type Annotations
| Metric | Value |
|--------|-------|
| Functions with return types | 46.6% |
| Fully typed functions | 41.3% |
| `Optional[X]` usages (legacy) | 144 |
| `X \| None` usages (modern) | 78 |

### Code Quality
| Issue Type | Count | Severity |
|------------|-------|----------|
| Methods >50 lines | 64 | Medium |
| God methods (complexity >15) | 27 | High |
| Deep nesting (depth ≥4) | 42 | Medium |
| Magic numbers | 43 | Low |
| Bare except clauses | 0 | N/A |
| Mutable default args | 0 | N/A |
| TODO/FIXME comments | 3 | Low |

### Test Coverage (Static Analysis)
| Class | Coverage |
|-------|----------|
| BrainData | 75.9% |
| Adjacency | 80.6% |
| DesignMatrix | 85.7% |
| Collection | 80.7% |
| **Overall** | **79.8%** |

### Deprecations
| Status | Count |
|--------|-------|
| Currently deprecated | 5 |
| Candidates for deprecation | 8 |
| Legacy patterns to modernize | 3 |

---

## Critical Findings for v0.6.0

### Must Fix (Blockers)
1. **Remove `pearson()` function** - Deprecated since v0.5.2
2. **Add DeprecationWarning to `get_anatomical()`** - Says deprecated in docstring only

### Should Fix (High Priority)
3. **Standardize `is_empty` property** - Currently `empty`, `isempty`, `is_empty` across classes
4. **Add `standardize()` to DesignMatrix** - Currently only has `zscore()`
5. **Add missing docstrings to 7 public items** - `Simulator`, `SimulateGrid`, etc.
6. **Add deprecation warnings to stats.py wrappers** - `one_sample_permutation`, etc.

### Consider for v0.6.0
7. **Refactor `Adjacency.__init__`** (205 lines, complexity 35)
8. **Split `social_relations_model`** (287 lines, complexity 37)
9. **Add tests for `pipe`, `pool`, `multivariate_similarity`**

---

## Consolidation Opportunities

### High Impact, Low Effort (~4 hours total)
| Task | Lines Saved |
|------|-------------|
| Add `_perform_arithmetic` to Adjacency | 70 |
| Add `_apply_func` to Adjacency | 50 |
| Create `FisherTransformMixin` | 15 |
| Add `is_h5_path()` utility | 10 |
| **Total** | **145** |

---

## Recommendations by Release

### v0.6.0 (Current)
1. Remove deprecated `pearson()` function
2. Add deprecation warnings to stats.py wrapper functions
3. Standardize `is_empty` property across classes
4. Add 7 missing public docstrings
5. Document API inconsistencies in changelog

### v0.6.x (Patch Releases)
1. Type quick-win files (14 files, ~2 hours)
2. Add tests for untested high-priority methods
3. Extract magic numbers to constants

### v0.7.0 (Future)
1. Remove deprecated stats.py wrappers
2. Remove deprecated dataset functions
3. Complete type annotation coverage
4. Refactor god methods (complexity >30)

---

## Files Requiring Attention

### Highest Priority
| File | Issues |
|------|--------|
| `data/adjacency/__init__.py` | 0% typed, needs `_perform_arithmetic` |
| `stats.py` | 0% typed, 4 wrapper functions to deprecate |
| `utils.py` | 0% typed, `get_anatomical` needs warning |
| `simulator.py` | 0% typed, 2 classes missing docstrings |

### Well-Maintained Files (Use as Examples)
| File | Coverage |
|------|----------|
| `pipelines/*.py` | 83% typed, modern syntax |
| `data/collection.py` | 80% typed, uses `X \| None` |
| `algorithms/_validation.py` | 100% typed |

---

## Audit Reports Reference

| Report | Location | Size |
|--------|----------|------|
| API Shell | `audit_logs/01_api_shell.md` | 10 KB |
| API Core | `audit_logs/02_api_core.md` | 20 KB |
| Docstrings | `audit_logs/03_docstrings.md` | 19 KB |
| Types | `audit_logs/04_types.md` | 12 KB |
| Anti-patterns | `audit_logs/05_antipatterns.md` | 11 KB |
| Coverage | `audit_logs/06_coverage.md` | 10 KB |
| Deprecations | `audit_logs/07_deprecations.md` | 12 KB |
| Consolidation | `audit_logs/08_consolidation.md` | 15 KB |

---

*Generated by comprehensive codebase audit on 2026-01-24*
