# Type Annotation Audit Report

**Date:** 2026-01-24
**Scope:** nltools v0.6.0 release (all Python source files in `nltools/`, excluding tests)
**Python Version:** 3.11+ (supports native `X | None` union syntax)

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total functions analyzed | 852 |
| Functions with return type | 397 (46.6%) |
| Fully typed functions (return + all params) | 352 (41.3%) |
| Total parameters | 2,187 |
| Typed parameters | 1,027 (47.0%) |
| `Any` usages | 55 |
| `Optional[X]` usages (legacy) | 144 |
| `X \| None` usages (modern) | 78 |

**Overall Assessment:** The codebase has inconsistent type annotation coverage. Newer files (pipelines, collection.py) follow modern practices, while core legacy files (brain_data.py, adjacency, stats.py, utils.py) have minimal to no typing.

---

## Coverage by File

### Well-Typed Files (>80% fully typed)

| File | Functions | Return % | Param % | Fully Typed % | Any | Optional | X\|None |
|------|-----------|----------|---------|---------------|-----|----------|---------|
| nltools/algorithms/_shape_utils.py | 4 | 100% | 100% | 100% | 0 | 0 | 0 |
| nltools/algorithms/_validation.py | 17 | 100% | 100% | 100% | 0 | 3 | 0 |
| nltools/algorithms/ridge/_core.py | 2 | 100% | 100% | 100% | 0 | 5 | 0 |
| nltools/algorithms/ridge/solvers.py | 3 | 100% | 100% | 100% | 3 | 14 | 0 |
| nltools/algorithms/ridge/utils.py | 5 | 100% | 100% | 100% | 0 | 2 | 0 |
| nltools/algorithms/inference/utils.py | 5 | 100% | 100% | 100% | 0 | 1 | 0 |
| nltools/neighborhoods.py | 8 | 100% | 100% | 100% | 0 | 0 | 0 |
| nltools/pipelines/cv.py | 15 | 100% | 100% | 100% | 5 | 10 | 0 |
| nltools/pipelines/results.py | 21 | 100% | 100% | 100% | 2 | 1 | 0 |
| nltools/data/fit_results.py | 2 | 100% | 100% | 100% | 0 | 14 | 0 |
| nltools/pipelines/base.py | 20 | 95% | 100% | 95% | 22 | 2 | 0 |
| nltools/pipelines/steps.py | 18 | 94% | 95% | 94% | 4 | 2 | 2 |
| nltools/algorithms/alignment/_local.py | 14 | 93% | 100% | 93% | 2 | 13 | 0 |
| nltools/cache.py | 11 | 91% | 93% | 82% | 0 | 0 | 2 |
| nltools/data/collection.py | 115 | 90% | 94% | 80% | 1 | 0 | 68 |

### Partially Typed Files (20-80%)

| File | Functions | Return % | Param % | Fully Typed % | Any | Optional | X\|None |
|------|-----------|----------|---------|---------------|-----|----------|---------|
| nltools/algorithms/inference/matrix.py | 9 | 89% | 97% | 89% | 0 | 3 | 0 |
| nltools/data/design_matrix.py | 38 | 89% | 90% | 76% | 0 | 11 | 0 |
| nltools/algorithms/inference/icc.py | 7 | 86% | 96% | 86% | 0 | 2 | 0 |
| nltools/algorithms/inference/correlation.py | 8 | 88% | 95% | 75% | 0 | 3 | 0 |
| nltools/algorithms/hrf.py | 6 | 83% | 69% | 83% | 0 | 0 | 0 |
| nltools/pipelines/pool.py | 19 | 100% | 83% | 79% | 2 | 17 | 0 |
| nltools/algorithms/inference/bootstrap.py | 19 | 79% | 86% | 47% | 0 | 5 | 0 |
| nltools/algorithms/_random.py | 4 | 75% | 100% | 75% | 0 | 4 | 0 |
| nltools/algorithms/inference/one_sample.py | 4 | 75% | 92% | 50% | 0 | 3 | 0 |
| nltools/algorithms/inference/two_sample.py | 4 | 75% | 93% | 50% | 0 | 3 | 0 |
| nltools/algorithms/hyperalignment.py | 8 | 62% | 85% | 62% | 0 | 2 | 0 |
| nltools/pipelines/multi_subject.py | 14 | 64% | 59% | 29% | 2 | 3 | 1 |
| nltools/algorithms/srm.py | 21 | 43% | 58% | 43% | 5 | 11 | 0 |
| nltools/prefs.py | 5 | 40% | 50% | 40% | 0 | 0 | 0 |
| nltools/pipelines/terminals.py | 8 | 50% | 80% | 25% | 4 | 0 | 0 |

### Untyped Files (0-20%)

| File | Functions | Return % | Param % | Fully Typed % | Any | Optional | X\|None |
|------|-----------|----------|---------|---------------|-----|----------|---------|
| nltools/backends.py | 11 | 18% | 31% | 18% | 1 | 0 | 0 |
| nltools/algorithms/inference/isc.py | 19 | 11% | 27% | 11% | 2 | 4 | 0 |
| nltools/data/brain_data.py | 114 | 8% | 10% | 4% | 0 | 0 | 5 |
| **nltools/data/adjacency/__init__.py** | **61** | **0%** | **0%** | **0%** | 0 | 0 | 0 |
| **nltools/stats.py** | **42** | **0%** | **0%** | **0%** | 0 | 0 | 0 |
| **nltools/utils.py** | **24** | **0%** | **0%** | **0%** | 0 | 0 | 0 |
| **nltools/simulator.py** | **23** | **0%** | **0%** | **0%** | 0 | 0 | 0 |
| **nltools/plotting.py** | **19** | **0%** | **0%** | **0%** | 0 | 0 | 0 |
| nltools/algorithms/ridge/backends/torch.py | 23 | 0% | 0% | 0% | 0 | 0 | 0 |
| nltools/algorithms/ridge/backends/numpy.py | 11 | 0% | 0% | 0% | 0 | 0 | 0 |
| nltools/models/glm.py | 9 | 0% | 0% | 0% | 0 | 0 | 0 |
| nltools/datasets.py | 8 | 0% | 0% | 0% | 0 | 0 | 0 |
| nltools/models/base.py | 7 | 0% | 0% | 0% | 0 | 0 | 0 |
| nltools/algorithms/ridge/backends/torch_cuda.py | 7 | 0% | 0% | 0% | 0 | 0 | 0 |
| nltools/data/_validation.py | 7 | 0% | 0% | 0% | 0 | 0 | 0 |
| nltools/mask.py | 5 | 0% | 0% | 0% | 0 | 0 | 0 |
| nltools/models/ridge.py | 5 | 0% | 0% | 0% | 0 | 0 | 0 |
| nltools/algorithms/ridge/backends/_utils.py | 4 | 0% | 0% | 0% | 0 | 0 | 0 |
| nltools/analysis.py | 4 | 0% | 0% | 0% | 0 | 0 | 0 |
| nltools/cross_validation.py | 4 | 0% | 0% | 0% | 0 | 0 | 0 |
| nltools/file_reader.py | 1 | 0% | 0% | 0% | 0 | 0 | 0 |

---

## High-Priority Public APIs Needing Types

### Tier 1: Core Data Classes (most user-facing)

**1. BrainData** (`nltools/data/brain_data.py`) - 114 functions, 4% typed

Critical public methods missing types:
- `__init__`, `__getitem__`, `__setitem__`, `__len__`
- `apply_mask`, `to_nifti`, `write`, `copy`
- `mean`, `std`, `sum`, `zscore`, `append`
- `regress`, `predict`, `fit`, `cv`
- All arithmetic operators (`__add__`, `__sub__`, `__mul__`, etc.)

**2. Adjacency** (`nltools/data/adjacency/__init__.py`) - 61 functions, 0% typed

All public methods missing types:
- `__init__`, `__getitem__`, `__repr__`
- `to_graph`, `threshold`, `cluster_summary`
- `similarity`, `distance`, `bootstrap_samples`
- All arithmetic operators

**3. DesignMatrix** (`nltools/data/design_matrix.py`) - 38 functions, 76% typed

Missing types on:
- `__init__`, `__setitem__`, `__array__`, `__eq__`
- `downsample`, `upsample` (missing `**kwargs`)
- `heatmap`

### Tier 2: Functional Core

**4. stats.py** (`nltools/stats.py`) - 42 functions, 0% typed

All public functions need types:
- `pearson`, `zscore`, `fdr`, `holm_bonf`
- `threshold`, `multi_threshold`
- `downsample`, `upsample`
- `fisher_r_to_z`, `fisher_z_to_r`
- `isc`, `isc_group`, `isfc`, `isps`
- `procrustes`, `align`, `compute_similarity`

**5. utils.py** (`nltools/utils.py`) - 24 functions, 0% typed

Key functions needing types:
- `to_h5`, `load_brain_data_h5`
- `get_resource_path`, `get_anatomical`
- `detect_best_matching_template`
- `set_algorithm`, `set_decomposition_algorithm`

### Tier 3: Models and Algorithms

**6. models/** - All models at 0%
- `Ridge.fit`, `Ridge.predict`, `Ridge.score`
- `Glm.fit`, `Glm.predict`, `Glm.compute_contrast`
- `BaseModel` interface methods

**7. Ridge backends** - All backends at 0%
- `numpy.py`, `torch.py`, `torch_cuda.py`

---

## Style Inconsistencies

### 1. Optional vs Union Syntax

The codebase uses both styles inconsistently:

**Legacy style (`Optional[X]`):** 144 usages
```python
# Found in older/functional modules
from typing import Optional
def foo(x: Optional[str] = None) -> Optional[int]: ...
```

**Modern style (`X | None`):** 78 usages
```python
# Found in newer modules with __future__ annotations
from __future__ import annotations
def foo(x: str | None = None) -> int | None: ...
```

**Files using modern style (with `from __future__ import annotations`):**
- nltools/cache.py
- nltools/data/collection.py (68 usages - best example!)
- nltools/neighborhoods.py
- nltools/pipelines/*.py (base, cv, multi_subject, pool, results, steps, terminals)
- nltools/algorithms/alignment/_local.py

**Files using legacy Optional:** Most algorithm and data modules

### 2. Any Usage (55 instances)

`Any` is used appropriately in most cases for:
- Pipeline transformers/data that can be arbitrary types
- Interop with sklearn estimators
- Avoiding circular imports (`fitted_stack: Any  # FittedStack`)

**Questionable uses:**
- `pipelines/base.py` has 22 `Any` usages - could be tightened with protocols/generics
- `solvers.py` uses `backend: Any` which could be typed as `Backend` protocol

### 3. Missing Typing Imports

Many files that could benefit from typing don't import from `typing`:
- brain_data.py - no typing imports at all
- adjacency/__init__.py - no typing imports
- stats.py - no typing imports
- utils.py - no typing imports

---

## Quick Wins: Files Near 100%

These small files need minimal work to reach 100% coverage:

| File | Functions | Needs Typing | Current % | Effort |
|------|-----------|--------------|-----------|--------|
| nltools/algorithms/inference/matrix.py | 9 | 1 | 89% | Trivial |
| nltools/algorithms/inference/icc.py | 7 | 1 | 86% | Trivial |
| nltools/algorithms/hrf.py | 6 | 1 | 83% | Trivial |
| nltools/algorithms/_random.py | 4 | 1 | 75% | Trivial |
| nltools/file_reader.py | 1 | 1 | 0% | Trivial |
| nltools/algorithms/inference/correlation.py | 8 | 2 | 75% | Low |
| nltools/algorithms/inference/one_sample.py | 4 | 2 | 50% | Low |
| nltools/algorithms/inference/two_sample.py | 4 | 2 | 50% | Low |
| nltools/algorithms/hyperalignment.py | 8 | 3 | 62% | Low |
| nltools/prefs.py | 5 | 3 | 40% | Low |
| nltools/algorithms/ridge/backends/_utils.py | 4 | 4 | 0% | Low |
| nltools/analysis.py | 4 | 4 | 0% | Low |
| nltools/cross_validation.py | 4 | 4 | 0% | Low |
| nltools/mask.py | 5 | 5 | 0% | Low |

**Estimated effort to fully type these 14 files:** ~2 hours

---

## Recommendations

### 1. Standardize on Modern Syntax (Priority: HIGH)

Since Python 3.11+ is required:
- Use `X | None` instead of `Optional[X]`
- Add `from __future__ import annotations` to all modules
- Migrate existing `Optional` imports during regular maintenance

### 2. Type Priority Order (for v0.6.0)

1. **Phase 1 - Quick Wins** (1-2 hours)
   - Complete the 14 "quick win" files listed above
   - Reach 50%+ overall coverage

2. **Phase 2 - Public APIs** (1-2 days)
   - Type all public methods in BrainData, Adjacency, DesignMatrix
   - Focus on `__init__`, data access, and I/O methods first

3. **Phase 3 - Functional Core** (1 day)
   - Type stats.py completely
   - Type utils.py completely
   - These are the most commonly imported modules

4. **Phase 4 - Models** (0.5 days)
   - Type all model fit/predict/score methods
   - Ensure consistent sklearn-compatible signatures

### 3. Reduce Any Usage

Replace `Any` with specific types where possible:
- Use `Protocol` for duck-typed interfaces
- Use `TypeVar` for generic transforms
- Use union types for known alternatives

### 4. Add Type Checking to CI

Consider adding `mypy` or `pyright` to CI:
```toml
# pyproject.toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
```

### 5. Document Type Conventions

Add to CLAUDE.md or CONTRIBUTING.md:
```markdown
## Type Annotation Standards
- Use `X | None` not `Optional[X]`
- Add `from __future__ import annotations` to all files
- Minimize `Any` usage - prefer protocols or generics
- All public functions MUST have type annotations
```

---

## Appendix: Type Coverage by Directory

| Directory | Files | Functions | Fully Typed | Coverage |
|-----------|-------|-----------|-------------|----------|
| nltools/algorithms/inference/ | 10 | 85 | 46 | 54% |
| nltools/pipelines/ | 7 | 115 | 96 | 83% |
| nltools/algorithms/ridge/ | 7 | 28 | 14 | 50% |
| nltools/data/ | 5 | 337 | 100 | 30% |
| nltools/algorithms/ | 6 | 54 | 40 | 74% |
| nltools/models/ | 3 | 21 | 0 | 0% |
| nltools/ (root) | 12 | 212 | 56 | 26% |

**Best practices observed in:** `pipelines/`, `algorithms/inference/`
**Most work needed in:** `data/`, `stats.py`, `utils.py`, `models/`
