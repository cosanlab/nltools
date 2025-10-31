# Himalaya Library GPU Resampling Research Report

**Date**: 2025-10-31
**Purpose**: Research efficient GPU resampling patterns from himalaya library to inform nltools bootstrap GPU implementation

---

## Executive Summary

**What is himalaya?**

Himalaya is a high-performance Python library from UC Berkeley's Gallant Lab designed for fitting multiple-target linear ridge regression models with GPU acceleration. It's specifically optimized for neuroimaging applications (fMRI encoding models) where models must be fit to extremely large numbers of targets (routinely 100k+ voxels). The library provides CPU (NumPy), GPU (CuPy), and hybrid (PyTorch) backend support with a clean scikit-learn-compatible API.

**Do they have relevant GPU resampling code?**

**No direct bootstrap implementation**, but himalaya has extensively developed **GPU-accelerated cross-validation with sophisticated resampling and batching strategies** that are highly relevant to our bootstrap implementation. Their `solve_ridge_cv_svd` and `solve_group_ridge_random_search` functions implement advanced memory-efficient CV patterns that can directly inform our GPU bootstrap design.

**Key takeaways for our implementation:**

1. **Backend abstraction layer**: Himalaya's clean backend switching (NumPy/CuPy/PyTorch) provides an excellent model for our GPU abstraction
2. **Three-dimensional batching**: They batch across targets, hyperparameters (alphas), AND CV folds - we can adapt this to bootstrap iterations
3. **Y_in_cpu strategy**: Keeping target data on CPU while only moving batches to GPU is a key memory optimization we should adopt
4. **Explicit memory management**: Aggressive `del` statements and immediate CPU transfer after GPU computation prevents memory accumulation
5. **No bootstrap, but patterns transfer**: Their CV resampling (train/test splits) uses the same computational patterns as bootstrap resampling

---

## Key Recommendations for nltools

### Top 5 Patterns to Adopt

1. ✅ **Backend abstraction layer** (HIGH PRIORITY)
   - Create `nltools/algorithms/inference/backends/` module
   - Implement: `numpy.py`, `torch.py`, `torch_cuda.py`, `cupy.py`
   - Provides: `set_backend()`, `get_backend()`, device-agnostic operations

2. ✅ **Y_in_cpu strategy** (HIGH PRIORITY)
   - Keep large data on CPU, batch to GPU as needed
   - Essential for datasets >1GB
   - Prevents GPU OOM errors

3. ✅ **Explicit memory management** (MEDIUM PRIORITY)
   - Aggressive `del` statements after GPU operations
   - Immediate CPU transfer of results
   - Don't accumulate large tensors on GPU

4. ✅ **Three-stage pattern: allocate → compute → aggregate** (MEDIUM PRIORITY)
   - Pre-allocate results array on CPU
   - Compute in batches on GPU
   - Aggregate on CPU (cheap)

5. ✅ **Automatic GPU transfer in user API** (LOW PRIORITY)
   - Users don't manually manage GPU transfers
   - `backend="auto"` detects and uses GPU if available
   - Similar to himalaya's `.fit()` pattern

### Top 5 Things to Avoid

1. ❌ **Don't accumulate large tensors on GPU**
   - Transfer to CPU immediately after computation

2. ❌ **Don't generate all indices upfront for GPU**
   - Generate indices in batches (10k × 100k = 4 GB!)

3. ❌ **Don't use fixed batch sizes**
   - Different GPUs have different memory
   - Use adaptive or user-specified batch sizes

4. ❌ **Don't forget to transfer results to CPU**
   - Public API should return NumPy arrays

5. ❌ **Don't batch only one dimension**
   - Batch both iterations AND targets for large problems

---

## Implementation Priority

**Phase 5a: Backend abstraction** (HIGH)
- Create `backends/` module with NumPy, PyTorch CPU, PyTorch GPU
- Test: Same code runs on CPU and GPU

**Phase 5b: Simple GPU bootstrap** (MEDIUM)
- Implement GPU bootstrap with batching
- Use explicit memory management
- Test: Compare GPU vs CPU results

**Phase 5c: Memory optimizations** (MEDIUM)
- Add `data_in_cpu` support
- Two-dimensional batching (iterations × targets)
- Test: Large dataset (>1GB)

**Phase 5d: Ridge GPU bootstrap** (LOW)
- Extend to Ridge weights/predict on GPU
- Benchmark 10-100× speedup claims

---

## References

- **Repository**: https://github.com/gallantlab/himalaya
- **Documentation**: https://gallantlab.org/himalaya/
- **Paper**: Dupré La Tour et al. (2022), NeuroImage 264, 119728

See full report for detailed code patterns, comparisons, and implementation suggestions.

---

**Conclusion**: Himalaya provides excellent patterns for GPU-accelerated resampling that directly apply to our bootstrap implementation. Their backend abstraction, memory management strategies, and batching patterns should be adopted for Phase 5+ GPU work.
