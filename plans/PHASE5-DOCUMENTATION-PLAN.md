# Phase 5: Documentation Updates Plan

**Goal**: Update migration guide, API docs, and examples for v0.6.0 release

**Estimated Effort**: 8-12 hours

**Priority**: HIGH (must complete before release)

**Status**: 📋 READY FOR SUB-AGENT

---

## Context

v0.6.0 introduces several breaking changes and new features:
1. **Stats.py → Inference Module Migration**: `isc()`, `isc_group()`, `isfc()` now use inference module
2. **Fit Dataclass**: New `BrainData.fit(inplace=False)` returns Fit objects
3. **DesignMatrix Polars Migration**: pandas → Polars (already documented)
4. **Bootstrap Changes**: `OnlineBootstrapStats` infrastructure
5. **GPU Acceleration**: New inference module GPU support

Users need clear migration paths and examples.

---

## Task 1: Update Migration Guide

### Location
- `docs/migration-guide.md`

### Steps

1. **Read existing migration guide**:
   ```bash
   cat docs/migration-guide.md
   ```

2. **Add Stats.py → Inference Module Section**:

   **Subsection 1.1: ISC Functions**
   ```markdown
   ### ISC Functions (`isc()`, `isc_group()`, `isfc()`)
   
   **Status**: ✅ Migrated to inference module (wrappers maintained for backward compatibility)
   
   **Old API** (still works, but deprecated):
   ```python
   from nltools.stats import isc, isc_group, isfc
   
   result = isc(data, n_samples=1000)
   result = isc_group(group1, group2, n_samples=1000)
   result = isfc(data, n_permute=1000)
   ```
   
   **New API** (recommended):
   ```python
   from nltools.algorithms.inference import (
       isc_permutation_test,
       isc_group_permutation_test,
       # isfc uses _compute_cross_correlation internally
   )
   from nltools.stats import isfc  # Still available, now uses inference module
   
   # ISC - single group
   result = isc_permutation_test(data, n_permute=1000)
   
   # ISC Group - two groups
   result = isc_group_permutation_test(group1, group2, n_permute=1000)
   
   # ISFC - functional connectivity
   result = isfc(data, n_permute=1000)  # Now uses inference module internally
   ```
   
   **Key Changes**:
   - `isc()` → `isc_permutation_test()` (parameter name: `n_samples` → `n_permute`)
   - `isc_group()` → `isc_group_permutation_test()` (parameter name: `n_samples` → `n_permute`)
   - `isfc()` unchanged (still `isfc()`, but now uses inference module internally)
   - Return keys: `null_dist` → `null_distribution` (wrapper handles mapping)
   - GPU acceleration available with `parallel="gpu"`
   - CPU parallelization available with `parallel="cpu"` and `n_jobs=-1`
   
   **Performance**: 4-8× CPU speedup, 10-100× GPU speedup
   ```

   **Subsection 1.2: Other Stats Functions**
   ```markdown
   ### Removed Functions
   
   **Functions Removed** (use alternatives):
   - `regress()` → Use `nltools.models.Glm` or `BrainData.fit(model='glm')`
   - `regress_permutation()` → Use inference module permutation tests
   - `correlation()` → Use `correlation_permutation_test()` from inference module
   - `pearson()` → Use `scipy.stats.pearsonr` or `correlation_permutation_test()`
   
   **Matrix Utilities** (moved to inference module):
   - `double_center()` → `nltools.algorithms.inference.double_center()` (re-exported from stats.py)
   - `u_center()` → `nltools.algorithms.inference.u_center()` (re-exported from stats.py)
   - `distance_correlation()` → `nltools.algorithms.inference.distance_correlation()` (re-exported from stats.py)
   ```

3. **Add Fit Dataclass Section**:

   ```markdown
   ### Fit Dataclass (`BrainData.fit(inplace=False)`)
   
   **New Feature**: `BrainData.fit()` now supports returning Fit objects instead of mutating attributes.
   
   **Old API** (still works, default behavior):
   ```python
   brain.fit(X=dm, model='ridge', alpha=1.0)  # Mutates brain, adds attributes
   assert hasattr(brain, 'ridge_weights')
   ```
   
   **New API** (recommended):
   ```python
   fit = brain.fit(X=dm, model='ridge', alpha=1.0, inplace=False)  # Returns Fit object
   assert isinstance(fit, Fit)
   assert 'weights' in fit.available()
   assert not hasattr(brain, 'ridge_weights')  # brain unchanged
   
   # Serialization
   np.savez('fit_results.npz', **fit.asdict())
   loaded = Fit.from_dict(np.load('fit_results.npz'))
   ```
   
   **Use Cases**:
   - Immutable results (no accidental mutation)
   - Serialization (save/load fits)
   - Multiple fits on same BrainData object
   - Functional programming style
   ```

4. **Add Bootstrap Changes Section**:

   ```markdown
   ### Bootstrap Infrastructure (`OnlineBootstrapStats`)
   
   **New Feature**: Memory-efficient online bootstrap statistics.
   
   **Old API** (still works):
   ```python
   boot = brain.bootstrap(stat='mean', n_samples=5000)
   ```
   
   **New Implementation**:
   - Uses `OnlineBootstrapStats` for memory efficiency
   - Supports CPU parallelization (`n_jobs=-1`)
   - Works with fitted models (ridge, GLM)
   
   **Advanced Usage**:
   ```python
   from nltools.algorithms.inference import OnlineBootstrapStats
   
   # Direct usage (numpy arrays)
   stats = OnlineBootstrapStats()
   for sample in samples:
       stats.update(sample)
   result = stats.get_statistics()
   ```
   ```

5. **Add GPU Acceleration Section**:

   ```markdown
   ### GPU Acceleration
   
   **New Feature**: GPU-accelerated permutation tests (10-100× speedup).
   
   **Requirements**:
   - PyTorch installed
   - CUDA-capable GPU
   
   **Usage**:
   ```python
   from nltools.algorithms.inference import one_sample_permutation_test
   
   # CPU (default)
   result = one_sample_permutation_test(data, n_permute=1000)
   
   # GPU (automatic batching)
   result = one_sample_permutation_test(
       data, 
       n_permute=1000, 
       parallel="gpu",
       max_gpu_memory_gb=4.0  # Memory budget
   )
   
   # CPU parallel (4-8× speedup)
   result = one_sample_permutation_test(
       data,
       n_permute=1000,
       parallel="cpu",
       n_jobs=-1  # Use all cores
   )
   ```
   ```

### Success Criteria
- [ ] Migration guide updated with all sections
- [ ] Code examples tested (copy-paste works)
- [ ] Old → new API comparisons clear
- [ ] Performance notes included
- [ ] Links to API documentation added

---

## Task 2: Create Breaking Changes Summary

### Location
- `docs/migration-guide.md` (new section)
- `CHANGELOG.md` (create if doesn't exist)

### Steps

1. **Create Breaking Changes Table**:

   ```markdown
   ## Breaking Changes Summary
   
   | Component | Change | Old API | New API | Migration Path |
   |-----------|--------|---------|---------|----------------|
   | `stats.py` | Function removed | `regress()` | `nltools.models.Glm` | Use `BrainData.fit(model='glm')` |
   | `stats.py` | Function removed | `regress_permutation()` | Inference module | Use `one_sample_permutation_test()` |
   | `stats.py` | Function removed | `correlation()` | `correlation_permutation_test()` | Import from `inference` module |
   | `stats.py` | Function deprecated | `pearson()` | `scipy.stats.pearsonr` | Use scipy or inference module |
   | `DesignMatrix` | Backend changed | pandas | Polars | Automatic migration (backward compatible) |
   | `BrainData.fit()` | New parameter | `fit()` mutates | `fit(inplace=False)` returns Fit | Optional migration |
   | Import paths | Module moved | `stats.isc()` | `inference.isc_permutation_test()` | Wrapper maintained |
   | Return keys | Key renamed | `null_dist` | `null_distribution` | Wrapper handles mapping |
   ```

2. **Create CHANGELOG.md** (if doesn't exist):

   ```markdown
   # Changelog
   
   All notable changes to nltools will be documented in this file.
   
   ## [0.6.0] - 2025-01-XX
   
   ### Added
   - GPU-accelerated inference module (10-100× speedup)
   - CPU parallelization for permutation tests (4-8× speedup)
   - Fit dataclass for immutable results
   - Polars DesignMatrix backend (2-5× speedup)
   - OnlineBootstrapStats for memory-efficient bootstrap
   
   ### Changed
   - `isc()`, `isc_group()`, `isfc()` now use inference module internally
   - `DesignMatrix` uses Polars instead of pandas
   - `BrainData.fit()` supports `inplace=False` parameter
   
   ### Deprecated
   - `stats.pearson()` - Use `scipy.stats.pearsonr` or inference module
   
   ### Removed
   - `stats.regress()` - Use `nltools.models.Glm` or `BrainData.fit(model='glm')`
   - `stats.regress_permutation()` - Use inference module permutation tests
   - `stats.correlation()` - Use `correlation_permutation_test()` from inference module
   
   ### Fixed
   - Return key mapping for `isc()` and `isc_group()` wrappers
   - Nilearn compatibility warnings
   - Int64→int32 conversions for FSL/SPM compatibility
   ```

### Success Criteria
- [ ] Breaking changes table created
- [ ] CHANGELOG.md created/updated
- [ ] All breaking changes documented
- [ ] Migration paths clear for each change

---

## Task 3: Update API Documentation

### Location
- `docs/api/` directory

### Steps

1. **Check existing API docs structure**:
   ```bash
   ls -la docs/api/
   ```

2. **Update inference module API docs**:
   - Document all new functions: `isc_permutation_test()`, `isc_group_permutation_test()`, etc.
   - Add GPU acceleration parameters
   - Add CPU parallelization parameters
   - Include examples

3. **Update BrainData API docs**:
   - Document `fit(inplace=False)` parameter
   - Document Fit dataclass return type
   - Update method signatures

4. **Update stats.py API docs**:
   - Mark deprecated functions
   - Add migration notes
   - Update import paths

5. **Verify API docs build**:
   ```bash
   cd docs && make html  # or appropriate build command
   ```

### Success Criteria
- [ ] All new functions documented
- [ ] Parameter descriptions updated
- [ ] Examples included
- [ ] API docs build successfully
- [ ] Links work correctly

---

## Task 4: Create Examples

### Location
- `docs/examples/` or `docs/tutorials/` (check existing structure)

### Steps

1. **Create Inference Module Quick Start**:

   **File**: `docs/examples/inference_module_quickstart.py` or `.ipynb`
   
   ```python
   """
   Quick Start: Inference Module
   
   This example shows how to use the new inference module for permutation tests.
   """
   import numpy as np
   from nltools.algorithms.inference import (
       one_sample_permutation_test,
       isc_permutation_test,
       correlation_permutation_test,
   )
   
   # Generate example data
   np.random.seed(42)
   data = np.random.randn(30, 100)  # 30 samples, 100 features
   
   # One-sample permutation test
   result = one_sample_permutation_test(data, n_permute=1000)
   print(f"Mean: {result['mean']}")
   print(f"P-value: {result['p']}")
   
   # GPU acceleration (if available)
   result_gpu = one_sample_permutation_test(
       data, 
       n_permute=1000,
       parallel="gpu",
       max_gpu_memory_gb=4.0
   )
   
   # CPU parallelization
   result_cpu = one_sample_permutation_test(
       data,
       n_permute=1000,
       parallel="cpu",
       n_jobs=-1
   )
   ```

2. **Create Fit Dataclass Example**:

   **File**: `docs/examples/fit_dataclass_example.py` or `.ipynb`
   
   ```python
   """
   Fit Dataclass Example
   
   This example shows how to use BrainData.fit(inplace=False) to get Fit objects.
   """
   import numpy as np
   from nltools import BrainData, DesignMatrix
   from nltools.data import Fit
   
   # Create example data
   brain = BrainData(np.random.randn(100, 50))  # 100 timepoints, 50 voxels
   dm = DesignMatrix(np.random.randn(100, 3), columns=['a', 'b', 'c'])
   
   # Old way (mutates brain)
   brain.fit(X=dm, model='ridge', alpha=1.0)
   assert hasattr(brain, 'ridge_weights')
   
   # New way (returns Fit object)
   brain2 = BrainData(np.random.randn(100, 50))
   fit = brain2.fit(X=dm, model='ridge', alpha=1.0, inplace=False)
   assert isinstance(fit, Fit)
   assert 'weights' in fit.available()
   assert not hasattr(brain2, 'ridge_weights')
   
   # Serialization
   np.savez('fit_results.npz', **fit.asdict())
   loaded = Fit.from_dict(np.load('fit_results.npz'))
   assert np.allclose(fit.weights, loaded.weights)
   ```

3. **Create GPU Acceleration Example**:

   **File**: `docs/examples/gpu_acceleration_example.py` or `.ipynb`
   
   ```python
   """
   GPU Acceleration Example
   
   This example demonstrates GPU-accelerated permutation tests.
   """
   import numpy as np
   from nltools.algorithms.inference import one_sample_permutation_test
   from nltools.backends import check_gpu_available
   import time
   
   # Check GPU availability
   gpu_available, device = check_gpu_available()
   print(f"GPU Available: {gpu_available}")
   if gpu_available:
       print(f"Device: {device}")
   
   # Generate large dataset
   np.random.seed(42)
   data = np.random.randn(100, 10000)  # 100 samples, 10K features
   
   # CPU baseline
   start = time.time()
   result_cpu = one_sample_permutation_test(
       data, 
       n_permute=5000,
       parallel=None
   )
   cpu_time = time.time() - start
   print(f"CPU Time: {cpu_time:.2f}s")
   
   # GPU (if available)
   if gpu_available:
       start = time.time()
       result_gpu = one_sample_permutation_test(
           data,
           n_permute=5000,
           parallel="gpu",
           max_gpu_memory_gb=4.0
       )
       gpu_time = time.time() - start
       print(f"GPU Time: {gpu_time:.2f}s")
       print(f"Speedup: {cpu_time/gpu_time:.1f}×")
   ```

4. **Create Migration Example**:

   **File**: `docs/examples/migration_example.py` or `.ipynb`
   
   ```python
   """
   Migration Example: Old → New API
   
   This example shows how to migrate from old stats.py API to new inference module API.
   """
   import numpy as np
   
   # OLD API (still works, but deprecated)
   from nltools.stats import isc, isc_group
   
   data = np.random.randn(10, 100, 50)  # 10 subjects, 100 timepoints, 50 voxels
   result_old = isc(data, n_samples=1000)
   
   # NEW API (recommended)
   from nltools.algorithms.inference import isc_permutation_test
   
   result_new = isc_permutation_test(data, n_permute=1000)
   
   # Results are identical (wrapper handles mapping)
   assert np.allclose(result_old['isc'], result_new['isc'])
   ```

### Success Criteria
- [ ] All example files created
- [ ] Examples tested (run without errors)
- [ ] Examples documented with docstrings
- [ ] Examples added to documentation index/TOC

---

## Task 5: Update DesignMatrix Polars Docs (if not already done)

### Location
- `docs/migration-guide.md` (DesignMatrix section)

### Steps

1. **Verify DesignMatrix migration is documented**
2. **Add Polars-specific examples** if missing:
   ```markdown
   ### DesignMatrix Polars Migration
   
   **Status**: ✅ Complete (automatic migration, backward compatible)
   
   **Performance**: 2-5× speedup for resampling operations
   
   **Usage** (unchanged):
   ```python
   from nltools import DesignMatrix
   
   dm = DesignMatrix(data, columns=['a', 'b', 'c'])
   dm_downsampled = dm.downsample(factor=2)  # Now uses Polars internally
   ```
   ```

### Success Criteria
- [ ] DesignMatrix migration documented
- [ ] Performance notes included
- [ ] Examples provided

---

## Verification Steps

After completing all tasks:

1. **Test all examples**:
   ```bash
   for example in docs/examples/*.py; do
       echo "Testing $example"
       uv run python "$example"
   done
   ```

2. **Build documentation**:
   ```bash
   cd docs && make html  # or appropriate build command
   ```

3. **Review documentation**:
   - Check all links work
   - Verify code examples are correct
   - Ensure formatting is consistent

4. **Update verification checklist**:
   - Update `v0.6.0-VERIFICATION.md` with completed checkboxes

---

## Success Criteria for Phase 5

- [ ] Migration guide updated with all sections
- [ ] Breaking changes summary created
- [ ] CHANGELOG.md created/updated
- [ ] API documentation updated
- [ ] All examples created and tested
- [ ] Documentation builds successfully
- [ ] All links work correctly
- [ ] Code examples are copy-paste ready

---

## Notes

- **Testing**: All code examples must be tested (run without errors)
- **Consistency**: Use consistent formatting and style across all docs
- **Links**: Verify all internal/external links work
- **Examples**: Keep examples simple and focused (one concept per example)

---

## Reference Files

- `docs/migration-guide.md` - Main migration guide
- `docs/api/` - API documentation directory
- `v0.6.0-VERIFICATION.md` - Verification checklist
- `v0.6.0-ACTION-PLAN.md` - Overall action plan

---

**Last Updated**: 2025-01-03  
**Status**: Ready for sub-agent execution

