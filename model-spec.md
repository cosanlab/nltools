# Model Class Specification - nltools v0.6.0

**Status:** Sprint 1 Complete, Sprint 2 Next
**Last Updated:** 2025-10-28
**Target Release:** v0.6.0 (core), v0.6.1 (GPU optimization)

---

## Executive Summary

This specification outlines the design and implementation of a new Model class system for nltools that:
- Restores deprecated prediction/inference methods (`.predict()`, `.ttest()`, `.randomise()`)
- Provides efficient ridge regression using extracted algorithms from himalaya
- Adds optional GPU acceleration via PyTorch (1.4-2.2x speedup on MPS, 10-30x on CUDA)
- Maintains backward compatibility with v0.5.1 API
- Wraps nilearn GLM for experimental design analysis
- Follows scikit-learn API conventions

**Implementation approach:** Extract core algorithms from himalaya (with attribution) and add cross-platform GPU support via PyTorch.

---

## Background & Motivation

### Current State (v0.6.0)

In the v0.6.0 refactoring, we deprecated several methods from Brain_Data:
- `.predict()` - ML prediction with cross-validation
- `.predict_multi()` - Searchlight/multi-ROI analysis
- `.ttest()` - One-sample t-tests with multiple comparison correction
- `.randomise()` - Permutation-based inference

These methods now raise `NotImplementedError` and point users to a future Model class.

### Why We Need Model Classes

**1. Separation of Concerns**
- Brain_Data should manage neuroimaging data, not implement ML algorithms
- Statistical models deserve their own well-designed classes
- Easier to test, maintain, and extend

**2. Leverage Modern Libraries**
- nilearn provides excellent GLM implementation
- himalaya has elegant ridge regression algorithms
- PyTorch enables cross-platform GPU acceleration

**3. User Experience**
- Restore functionality users depend on
- Provide both simple (Brain_Data methods) and advanced (Model classes) APIs
- Enable efficient processing of large datasets

---

## Architecture Decision

### Selected: Hybrid Base Model with Specialized Subclasses

```
BaseModel (abstract)
├── RegressionModel (ridge with GPU support)
├── GLMModel (wraps nilearn FirstLevelModel/SecondLevelModel)
├── InferenceModel (t-tests, permutation testing)
└── SearchlightModel (future: searchlight/multi-ROI)
```

**Key principles:**
- Each model class has single responsibility
- All inherit sklearn-compatible interface from BaseModel
- Can be used independently or via Brain_Data convenience methods
- GPU support isolated to RegressionModel, optional everywhere else

---

## Ridge Regression: Extract vs Depend

### Decision: Extract Algorithms ✅ DONE

**Rationale:**
1. Core algorithms are elegant and simple (~150 lines total)
2. We only need a subset (SVD solver, efficient CV, batching)
3. Full control for neuroimaging-specific optimizations
4. No hard dependencies - keeps nltools lightweight
5. Can add GPU ourselves via PyTorch backend

**What we extracted from himalaya:**
- SVD-based ridge solver with numerical stability
- Efficient cross-validation with R² scoring
- Multi-target support (vectorized over targets)

**Implementation:** `nltools/algorithms/ridge.py` with BSD-3-Clause attribution

---

## GPU Support Strategy

### Decision: PyTorch Backend ✅ DONE

**Why PyTorch:**
- Cross-platform (CUDA, MPS, ROCm)
- Mature, stable API
- No CUDA-specific code
- Works on consumer GPUs (8-16GB sufficient)

**Implementation:** `nltools/backends.py`

**Backend abstraction:**
```python
Backend('numpy')   # CPU-only
Backend('torch')   # Auto-detects cuda/mps/cpu
Backend('auto')    # Smart selection based on problem size
```

**Auto-selection heuristics:**
- Small problems (< 10M elements): NumPy (avoid GPU overhead)
- Large problems (> 30M elements): GPU if available
- Cross-validation: Prefer GPU even for medium problems

**Actual Performance (MPS on Apple Silicon):**
- Small (100×1k): NumPy 2.5x faster (GPU overhead)
- Medium (300×50k): GPU 2.2x faster
- Large (1000×200k): GPU 1.4x faster
- 5-fold CV (300×100k): GPU 1.6x faster

**Note:** MPS backend shows modest speedups due to SVD CPU fallback. NVIDIA CUDA expected to show 10-30x speedups.

---

## Implementation Status

### ✅ Sprint 1: Core Ridge + Backends (COMPLETE)

**Implemented:**
- `nltools/backends.py` (~290 lines) - Backend abstraction
- `nltools/algorithms/ridge.py` (~286 lines) - Ridge SVD & CV
- `nltools/tests/core/test_ridge.py` - Comprehensive test suite
- Full test coverage with CPU/GPU equivalence tests

**Benchmarking:**
- Initial exploratory benchmarks (19 runs across 5 scenarios)
- Systematic benchmark framework designed (8 conditions)
- Performance guide with real data integrated into docs

**Documentation:**
- `docs/api/backends.md` - Backend API reference
- `docs/api/algorithms.md` - Ridge, HRF, SRM documentation
- `docs/performance.md` - Comprehensive performance guide with benchmark results
- Documentation builds successfully with jupyter-book

**Key Files:**
- `benchmarks/benchmark_ridge.py` - Initial benchmark suite
- `benchmarks/benchmark_ridge_systematic.py` - Systematic grid (ready to run)
- `benchmarks/benchmarking-guide.md` - Methodology documentation
- `benchmarks/results_ridge_performance.csv` - Benchmark data

---

## Next Steps: Sprint 2 - Model Classes

### Sprint 2: Model Classes (2-3 days) - COMPLETE
- [x] Implement `BaseModel` (abstract base with sklearn interface) ✅
- [x] Implement `Ridge` (uses ridge.py algorithms, renamed from RegressionModel) ✅
- [x] Unit tests for BaseModel (11 tests) ✅
- [x] Unit tests for Ridge (17 tests: basic, CV, backends) ✅
- [x] Implement `Glm` (wraps nilearn FirstLevelModel via composition) ✅
- [x] Unit tests for Glm (10 tests: fit, contrast, properties) ✅
- [ ] Implement `InferenceModel` (t-tests, permutation) - FUTURE

### Sprint 3: Integration (1-2 days)
- [ ] Update Brain_Data methods to use Model classes
- [ ] Test backward compatibility
- [ ] Integration tests
- [ ] Update REFACTORING_PLAN.md

### Sprint 4: Documentation (1-2 days)
- [ ] API reference for Model classes
- [ ] GPU support guide updates
- [ ] Migration guide for deprecated methods
- [ ] Example notebooks

### Sprint 5: Polish & Release (1 day)
- [ ] Performance profiling
- [ ] Final testing
- [ ] Update CHANGELOG
- [ ] Release v0.6.0

---

## Future Enhancements (v0.6.1+)

### Phase 2 Features
- [ ] SearchlightModel for searchlight analysis
- [ ] Multi-subject batch processing
- [ ] CuPy backend (NVIDIA-specific optimization)
- [ ] JAX backend (XLA compilation)

### Phase 3 Features
- [ ] Per-voxel alpha selection optimization
- [ ] Sparse data support
- [ ] Advanced permutation testing
- [ ] GPU-accelerated searchlight

---

## Success Criteria

### v0.6.0 Release (Minimum Viable)
- [x] Ridge regression with GPU support ✅
- [x] Backend abstraction tested on CPU and GPU ✅
- [x] Performance benchmarks documenting speedups ✅
- [x] API documentation complete ✅
- [ ] Model classes restore deprecated functionality
- [ ] No breaking changes to existing Brain_Data API
- [ ] All existing tests pass (91/91 currently passing)

### v0.6.1+ (Nice to Have)
- [ ] 10x+ speedup for typical neuroimaging workflows (on CUDA)
- [ ] GPU utilization for searchlight analysis
- [ ] Multi-subject batch processing
- [ ] Advanced permutation testing with GPU acceleration

---

## Dependencies

**Core (already in nltools):**
- numpy
- scipy
- nilearn
- scikit-learn

**Optional (for GPU support):**
- pytorch >= 2.0 (for GPU acceleration)

**Development:**
- pytest
- pytest-cov

**User installation:**
```bash
# CPU only
pip install nltools

# With GPU support
pip install nltools[gpu]  # installs pytorch
```

---

## References

### Libraries
- **himalaya**: https://github.com/gallantlab/himalaya
  - BSD-3-Clause licensed
  - Extracted SVD-based ridge regression algorithms
- **nilearn**: https://nilearn.github.io/
  - For GLM wrapping in future sprints
- **PyTorch**: https://pytorch.org/
  - Cross-platform GPU backend

### Papers
- Huth et al. (2016). "Natural speech reveals the semantic maps that tile human cerebral cortex." *Nature*.
- Chen et al. (2015). "A reduced-dimension fMRI shared response model." *NeurIPS*.

---

## Progress Log

See `model-spec-log.md` for detailed TDD implementation log and progress tracking.

**Latest:** Sprint 2 Complete - BaseModel + Ridge + Glm (2025-10-28)
- BaseModel abstract class with sklearn interface ✅
- Ridge model with CV and GPU support ✅
- Glm wrapping nilearn FirstLevelModel via composition ✅
- 38 comprehensive tests (11 BaseModel + 17 Ridge + 10 Glm) ✅
- Full backend integration for Ridge (numpy/torch/auto) ✅
- Glm uses composition pattern (like Brain_Data + maskers) ✅
- Accurate contrast statistics via nilearn.compute_contrast() ✅

**Previous:** Sprint 1 Complete (2025-10-28)
- Backend abstraction ✅
- Ridge regression algorithms ✅
- Comprehensive benchmarking ✅
- Complete API documentation ✅

**Next:** Sprint 3 - Brain_Data Integration or InferenceModel
