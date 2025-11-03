# Pre-Release Verification Plan

**Goal**: Run comprehensive smoke tests, integration tests, and performance benchmarks before release

**Estimated Effort**: 2-3 hours

**Priority**: HIGH (must complete before release)

**Status**: 📋 READY FOR SUB-AGENT

---

## Context

Before releasing v0.6.0, we need to verify:
1. **Basic functionality works** (smoke tests)
2. **All tests pass** (integration tests)
3. **Performance hasn't regressed** (benchmarks)
4. **Backward compatibility maintained** (compatibility tests)

This plan provides step-by-step verification procedures.

---

## Task 1: Smoke Tests (Manual)

### Goal
Verify basic functionality works in a clean environment.

### Steps

1. **Import Test**:
   ```bash
   uv run python -c "import nltools; print(f'nltools version: {nltools.__version__}')"
   ```
   **Expected**: No errors, version printed

2. **Load Example Data**:
   ```python
   # Test haxby dataset
   from nltools.datasets import fetch_haxby
   haxby = fetch_haxby()
   print(f"Haxby dataset loaded: {haxby.keys()}")
   
   # Test emotion dataset
   from nltools.datasets import fetch_emotion
   emotion = fetch_emotion()
   print(f"Emotion dataset loaded: {emotion.keys()}")
   ```
   **Expected**: Datasets load without errors

3. **Basic BrainData Workflow**:
   ```python
   import numpy as np
   from nltools import BrainData, DesignMatrix
   
   # Create BrainData
   data = np.random.randn(100, 50)
   brain = BrainData(data)
   print(f"BrainData shape: {brain.shape}")
   
   # Create DesignMatrix
   dm = DesignMatrix(np.random.randn(100, 3), columns=['a', 'b', 'c'])
   print(f"DesignMatrix shape: {dm.shape}")
   
   # Basic operations
   brain_mean = brain.mean()
   print(f"Mean computed: {brain_mean.shape}")
   ```
   **Expected**: All operations work without errors

4. **Inference Module**:
   ```python
   from nltools.algorithms.inference import one_sample_permutation_test
   import numpy as np
   
   data = np.random.randn(30, 100)
   result = one_sample_permutation_test(data, n_permute=100)
   print(f"Permutation test result keys: {result.keys()}")
   ```
   **Expected**: Inference module works, result has expected keys

5. **GPU Acceleration** (if available):
   ```python
   from nltools.backends import check_gpu_available
   gpu_available, device = check_gpu_available()
   print(f"GPU Available: {gpu_available}")
   
   if gpu_available:
       from nltools.algorithms.inference import one_sample_permutation_test
       import numpy as np
       
       data = np.random.randn(30, 100)
       result = one_sample_permutation_test(
           data, 
           n_permute=100,
           parallel="gpu"
       )
       print(f"GPU test passed: {result.keys()}")
   ```
   **Expected**: GPU works if available, graceful fallback if not

6. **Ridge Regression**:
   ```python
   from nltools import BrainData, DesignMatrix
   import numpy as np
   
   brain = BrainData(np.random.randn(100, 50))
   dm = DesignMatrix(np.random.randn(100, 3), columns=['a', 'b', 'c'])
   
   brain.fit(X=dm, model='ridge', alpha=1.0)
   print(f"Ridge weights shape: {brain.ridge_weights.shape}")
   ```
   **Expected**: Ridge regression works

7. **DesignMatrix Operations**:
   ```python
   from nltools import DesignMatrix
   import numpy as np
   
   dm = DesignMatrix(np.random.randn(100, 3), columns=['a', 'b', 'c'])
   dm_down = dm.downsample(factor=2)
   dm_up = dm.upsample(factor=2)
   print(f"Downsample: {dm_down.shape}, Upsample: {dm_up.shape}")
   ```
   **Expected**: Polars operations work correctly

### Success Criteria
- [ ] All imports work without errors
- [ ] Example datasets load successfully
- [ ] Basic BrainData workflow works
- [ ] Inference module works
- [ ] GPU acceleration works (if available)
- [ ] Ridge regression works
- [ ] DesignMatrix operations work

### Deliverable
Create `pre-release-smoke-tests.log` with all test results

---

## Task 2: Integration Tests

### Goal
Run comprehensive test suite and verify all tests pass.

### Steps

1. **Run Tier1 Tests**:
   ```bash
   uv run pytest -m tier1 -n auto --tb=short > tier1_tests.log 2>&1
   ```
   **Expected**: ~36s runtime, 303 passed, 6 skipped
   
   **Check output**:
   ```bash
   tail -20 tier1_tests.log
   grep -E "passed|failed|error" tier1_tests.log | tail -5
   ```

2. **Run Tier2 Tests** (with permission):
   ```bash
   # Ask user permission first - these take ~7 minutes
   uv run pytest -m tier2 -xvs --tb=long > tier2_tests.log 2>&1
   ```
   **Expected**: ~7min runtime, all GPU/benchmark tests pass
   
   **Note**: Only run if user explicitly approves (per CLAUDE.md guidelines)

3. **Run Full Test Suite**:
   ```bash
   uv run pytest nltools/tests/ --tb=short -q > full_tests.log 2>&1
   ```
   **Check summary**:
   ```bash
   tail -10 full_tests.log
   ```

4. **Check for Warnings**:
   ```bash
   uv run pytest -m tier1 -W default::UserWarning 2>&1 | grep -i "warning\|error" | head -30
   ```
   **Expected**: Minimal or no warnings (after Phase 4 fixes)

5. **Verify Backward Compatibility**:
   ```bash
   # Test that old API still works
   uv run python -c "
   from nltools.stats import isc, isc_group, isfc
   import numpy as np
   data = np.random.randn(10, 100, 50)
   result = isc(data, n_samples=100)
   print('Old API works:', 'isc' in result)
   "
   ```
   **Expected**: Old API functions still work (wrappers maintained)

### Success Criteria
- [ ] Tier1 tests pass (~36s, 303 passed, 6 skipped)
- [ ] Tier2 tests pass (if run, ~7min, all pass)
- [ ] Full test suite passes
- [ ] No unexpected warnings
- [ ] Backward compatibility maintained

### Deliverable
Create `pre-release-integration-tests.log` with all test results

---

## Task 3: Performance Benchmarks

### Goal
Verify performance improvements and check for regressions.

### Steps

1. **Inference Module Speedup** (CPU vs GPU):
   ```python
   # Create benchmark script: benchmarks/pre_release_inference_benchmark.py
   import numpy as np
   import time
   from nltools.algorithms.inference import one_sample_permutation_test
   from nltools.backends import check_gpu_available
   
   np.random.seed(42)
   data = np.random.randn(100, 1000)  # 100 samples, 1K features
   n_permute = 5000
   
   # CPU baseline
   start = time.time()
   result_cpu = one_sample_permutation_test(
       data, 
       n_permute=n_permute,
       parallel=None
   )
   cpu_time = time.time() - start
   
   # CPU parallel
   start = time.time()
   result_cpu_parallel = one_sample_permutation_test(
       data,
       n_permute=n_permute,
       parallel="cpu",
       n_jobs=-1
   )
   cpu_parallel_time = time.time() - start
   
   # GPU (if available)
   gpu_available, _ = check_gpu_available()
   if gpu_available:
       start = time.time()
       result_gpu = one_sample_permutation_test(
           data,
           n_permute=n_permute,
           parallel="gpu"
       )
       gpu_time = time.time() - start
       print(f"GPU Speedup: {cpu_time/gpu_time:.1f}×")
   
   print(f"CPU Parallel Speedup: {cpu_time/cpu_parallel_time:.1f}×")
   ```
   **Expected**: CPU parallel 4-8× speedup, GPU 10-100× speedup (if available)

2. **Ridge Regression Speedup**:
   ```python
   # Create benchmark script: benchmarks/pre_release_ridge_benchmark.py
   import numpy as np
   import time
   from nltools import BrainData, DesignMatrix
   
   np.random.seed(42)
   brain = BrainData(np.random.randn(1000, 1000))
   dm = DesignMatrix(np.random.randn(1000, 10))
   
   # Test GPU backend (if available)
   from nltools.backends import check_gpu_available
   gpu_available, _ = check_gpu_available()
   
   if gpu_available:
       start = time.time()
       brain.fit(X=dm, model='ridge', alpha=1.0, backend='torch_cuda')
       gpu_time = time.time() - start
       
       start = time.time()
       brain.fit(X=dm, model='ridge', alpha=1.0, backend='numpy')
       cpu_time = time.time() - start
       
       print(f"Ridge GPU Speedup: {cpu_time/gpu_time:.1f}×")
   ```
   **Expected**: GPU provides speedup if available

3. **DesignMatrix Polars Performance**:
   ```python
   # Create benchmark script: benchmarks/pre_release_designmatrix_benchmark.py
   import numpy as np
   import time
   from nltools import DesignMatrix
   
   np.random.seed(42)
   dm = DesignMatrix(np.random.randn(10000, 10))
   
   # Test downsample
   start = time.time()
   dm_down = dm.downsample(factor=2)
   downsample_time = time.time() - start
   
   # Test upsample
   start = time.time()
   dm_up = dm.upsample(factor=2)
   upsample_time = time.time() - start
   
   print(f"Downsample time: {downsample_time:.3f}s")
   print(f"Upsample time: {upsample_time:.3f}s")
   ```
   **Expected**: Polars operations are fast (2-5× faster than pandas)

4. **Memory Usage Profiling**:
   ```bash
   # Check for memory leaks
   uv run pytest -m tier1 --profile-memory 2>&1 | grep -i "memory\|leak"
   ```
   **Expected**: No memory leaks or excessive memory usage

### Success Criteria
- [ ] Inference module speedup verified (CPU 4-8×, GPU 10-100×)
- [ ] Ridge regression speedup verified (if GPU available)
- [ ] DesignMatrix Polars performance verified (2-5× faster)
- [ ] No memory leaks detected
- [ ] Performance improvements documented

### Deliverable
Create `pre-release-performance-benchmarks.log` with benchmark results

---

## Task 4: Backward Compatibility Verification

### Goal
Verify that old API still works (wrappers maintained).

### Steps

1. **Test Old Stats API**:
   ```python
   # Test script: tests/pre_release_backward_compat.py
   import numpy as np
   from nltools.stats import isc, isc_group, isfc
   
   # ISC test
   data = np.random.randn(10, 100, 50)
   result_isc = isc(data, n_samples=100)
   assert 'isc' in result_isc
   assert 'p' in result_isc
   
   # ISC Group test
   group1 = np.random.randn(5, 100, 50)
   group2 = np.random.randn(5, 100, 50)
   result_isc_group = isc_group(group1, group2, n_samples=100)
   assert 'isc_group_difference' in result_isc_group
   assert 'p' in result_isc_group
   
   # ISFC test
   result_isfc = isfc(data, n_permute=100)
   assert 'isfc' in result_isfc
   assert 'p' in result_isfc
   
   print("All backward compatibility tests passed")
   ```

2. **Test Old BrainData.fit() Behavior**:
   ```python
   from nltools import BrainData, DesignMatrix
   import numpy as np
   
   brain = BrainData(np.random.randn(100, 50))
   dm = DesignMatrix(np.random.randn(100, 3))
   
   # Old behavior (default)
   brain.fit(X=dm, model='ridge', alpha=1.0)
   assert hasattr(brain, 'ridge_weights')
   assert hasattr(brain, 'ridge_')
   
   print("Old BrainData.fit() behavior works")
   ```

3. **Test Deprecated Functions** (should warn but work):
   ```python
   import warnings
   warnings.filterwarnings('error', category=DeprecationWarning)
   
   # These should raise DeprecationWarning or work with warning
   from nltools.stats import pearson
   # Should work but warn
   ```

### Success Criteria
- [ ] Old stats API works (wrappers function correctly)
- [ ] Old BrainData.fit() behavior works (default inplace=True)
- [ ] Deprecated functions work with warnings
- [ ] No breaking changes for existing code

### Deliverable
Create `pre-release-backward-compat.log` with compatibility test results

---

## Task 5: Final Checks

### Steps

1. **Check for Critical TODOs**:
   ```bash
   grep -r "TODO\|FIXME\|XXX" nltools/ --include="*.py" | grep -v "__pycache__" | grep -v ".pyc" | head -20
   ```
   **Expected**: Only non-blocker TODOs (already audited)

2. **Check for Debug Code**:
   ```bash
   grep -r "import pdb\|pdb.set_trace\|print(" nltools/ --include="*.py" | grep -v "__pycache__" | grep -v "test_" | head -20
   ```
   **Expected**: No debug code in production files

3. **Verify Version**:
   ```bash
   grep "version" pyproject.toml
   ```
   **Expected**: Version matches target release (0.6.0)

4. **Check Documentation Builds**:
   ```bash
   cd docs && make html  # or appropriate build command
   ```
   **Expected**: Documentation builds without errors

### Success Criteria
- [ ] No critical TODOs found
- [ ] No debug code found
- [ ] Version correct
- [ ] Documentation builds successfully

---

## Verification Summary Template

Create `pre-release-verification-summary.md`:

```markdown
# Pre-Release Verification Summary

**Date**: YYYY-MM-DD
**Version**: 0.6.0
**Verifier**: [Sub-agent name]

## Smoke Tests
- [ ] Imports work
- [ ] Example datasets load
- [ ] Basic BrainData workflow works
- [ ] Inference module works
- [ ] GPU acceleration works (if available)
- [ ] Ridge regression works
- [ ] DesignMatrix operations work

## Integration Tests
- [ ] Tier1 tests: X passed, Y skipped, Z failed
- [ ] Tier2 tests: X passed, Y skipped, Z failed (if run)
- [ ] Full test suite: X passed, Y skipped, Z failed
- [ ] No unexpected warnings
- [ ] Backward compatibility maintained

## Performance Benchmarks
- [ ] Inference module CPU parallel speedup: X×
- [ ] Inference module GPU speedup: X× (if available)
- [ ] Ridge regression GPU speedup: X× (if available)
- [ ] DesignMatrix Polars performance: X× faster
- [ ] No memory leaks detected

## Backward Compatibility
- [ ] Old stats API works
- [ ] Old BrainData.fit() behavior works
- [ ] Deprecated functions work with warnings

## Final Checks
- [ ] No critical TODOs
- [ ] No debug code
- [ ] Version correct
- [ ] Documentation builds

## Issues Found
[List any issues found during verification]

## Recommendations
[Any recommendations for release]
```

---

## Success Criteria for Pre-Release Verification

- [ ] All smoke tests pass
- [ ] All integration tests pass (tier1 mandatory, tier2 optional)
- [ ] Performance benchmarks show improvements
- [ ] Backward compatibility verified
- [ ] No critical issues found
- [ ] Verification summary document created

---

## Notes

- **Tier2 Tests**: Only run with explicit user permission (per CLAUDE.md guidelines)
- **GPU Tests**: Skip gracefully if GPU not available
- **Documentation**: Fail if documentation doesn't build
- **Performance**: Document any regressions found

---

## Reference Files

- `v0.6.0-VERIFICATION.md` - Full verification checklist
- `v0.6.0-ACTION-PLAN.md` - Overall action plan
- `CLAUDE.md` - Development guidelines (tier2 testing rules)

---

**Last Updated**: 2025-01-03  
**Status**: Ready for sub-agent execution

