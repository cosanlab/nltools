# nltools Testing Strategy Analysis & Tiered Approach

**Date**: 2025-10-29
**Context**: Optimize test suite execution time while maintaining quality
**Goal**: Create fast tier-1 tests for CI/development, comprehensive tier-2 for pre-release

---

## Current State Analysis

### Test Suite Overview

**Total: ~385 tests** organized into three categories:

```
nltools/tests/
├── shell/       131 tests (imperative shell - Brain_Data, Adjacency, Design_Matrix)
├── core/        155 tests (functional core - algorithms, stats, models)
└── support/      31 tests (integration, utilities, performance tests)
```

**Test count by file** (descending):
```
test_brain_data.py          91 tests  (1644 lines) - LARGEST & LIKELY SLOWEST
test_design_matrix_new.py   68 tests  (1497 lines) - NEW POLARS IMPLEMENTATION
test_models.py              41 tests  (926 lines)  - Model fitting operations
test_srm.py                 34 tests  (561 lines)  - Matrix factorization
test_adjacency.py           30 tests  (504 lines)  - Graph operations
test_hyperalignment.py      27 tests  (502 lines)  - Alignment algorithms
test_backends.py            16 tests  (264 lines)  - PyTorch integration
test_ridge.py               15 tests  (308 lines)  - Ridge regression
test_stats.py               15 tests  (741 lines)  - Statistical functions
test_efficient_copy.py      14 tests  (417 lines)  - Performance validation
test_design_matrix.py       10 tests  (175 lines)  - LEGACY PANDAS
test_datasets.py             9 tests  (170 lines)  - Network downloads
[... others <5 tests each ...]
```

### Known Performance Characteristics

**From test_brain_data.py documentation** (91 tests, ~151s total):
- Average: ~3.2s per test
- **Threshold tests**: ~7.2s each (cluster filtering via nilearn)
- **GLM regression**: ~5-6s (FirstLevelModel fitting)
- **Hyperalignment/decomposition**: ~4-5s (matrix operations)
- **Bootstrap/permutation**: ~3-4s (resampling)
- **Math operations**: <1s (fast numpy)

**Estimated full suite timing** (extrapolating from brain_data):
- If all tests average 3.2s: ~385 × 3.2s = **~20 minutes**
- Reality likely worse due to expensive fixtures and I/O

### Current Infrastructure

**Fixtures** (conftest.py):
```python
# Already have multi-tier fixtures!
sim_brain_data          # Module-scoped, full brain (238,955 voxels)
minimal_brain_data      # Function-scoped, 5 voxels, 50 timepoints (~1-2s per test)
small_brain_data_for_cv # Function-scoped, 5 voxels, 24 samples (<0.1s per test)
tiny_brain_data_for_cv  # Function-scoped, 3 voxels, 6 samples (edge cases)
```

**Markers in use**:
```python
@pytest.mark.skip                    # Known bugs or refactoring needed
@pytest.mark.skipif(torch)          # PyTorch-dependent tests (16 tests)
# NO PERFORMANCE MARKERS YET - This is our opportunity!
```

**No pytest configuration** (no pytest.ini or [tool.pytest] in pyproject.toml)

---

## Proposed Tiered Testing Strategy

### Tier 1: Fast Core Tests (Target: <2 minutes)

**Purpose**: Rapid feedback during development, CI on every commit

**Criteria for inclusion**:
- Core API contracts (parameters, return types, shapes)
- Critical correctness checks (mathematical invariants)
- Fast fixtures only (minimal_brain_data, small synthetic data)
- No expensive operations (no GLM fitting, threshold clustering, permutation tests)
- No network I/O

**Marker**: `@pytest.mark.tier1` (default when no tier specified)

**Estimated tests**: ~150-200 tests
**Target time**: <2 minutes (~0.6s avg per test)

**Example test categories**:
```python
# API contracts
def test_brain_data_init_validates_inputs()
def test_regress_returns_correct_output_structure()
def test_ridge_rejects_invalid_cv_type()

# Mathematical invariants
def test_srm_preserves_timepoint_dimensions()
def test_adjacency_symmetry_for_correlation()
def test_design_matrix_rank_after_convolution()

# Fast computations
def test_brain_data_mean_calculation()
def test_adjacency_distance_metrics()
def test_design_matrix_append()
```

### Tier 2: Comprehensive Tests (Target: <15 minutes)

**Purpose**: Exhaustive validation before releases, nightly CI, pre-merge

**Criteria for inclusion**:
- Expensive but realistic operations (GLM, thresholding, bootstrapping)
- Integration tests with real neuroimaging workflows
- Performance benchmarks
- Network-dependent tests (NeuroVault downloads)
- Full brain fixtures (238K voxels)
- Resampling methods (bootstrap, permutation)

**Marker**: `@pytest.mark.tier2`

**Estimated tests**: ~100-150 tests
**Target time**: <15 minutes (~5-9s avg per test)

**Example test categories**:
```python
# Expensive nilearn operations
def test_brain_data_threshold_with_cluster_extent()  # ~7.2s
def test_brain_data_glm_fit_with_firstlevelmodel()   # ~5-6s
def test_brain_data_hyperalignment_full_pipeline()   # ~4-5s

# Integration workflows
def test_complete_glm_analysis_pipeline()
def test_searchlight_analysis_realistic_roi()
def test_cross_validation_with_full_brain_data()

# Performance validation
def test_efficient_copy_vs_deepcopy_timing()
def test_large_design_matrix_operations()

# Network-dependent
def test_fetch_neurovault_collection_real_download()
```

### Tier 3: Development/Optional Tests

**Purpose**: Debugging, hypothesis testing, future features

**Marker**: `@pytest.mark.tier3` or `@pytest.mark.dev`

**Criteria**:
- Experimental features under development
- Extremely slow stress tests (>30s per test)
- Tests requiring special hardware (GPU)
- Manual verification tests

---

## Implementation Plan

### Phase 1: Add pytest configuration (5 minutes)

**Add to pyproject.toml**:
```toml
[tool.pytest.ini_options]
markers = [
    "tier1: Fast core tests for development (default, runs in <2 min)",
    "tier2: Comprehensive tests for releases (runs in <15 min)",
    "tier3: Optional development/stress tests",
    "slow: Tests that take >5s (subset of tier2)",
    "integration: Tests requiring network or external services",
]

# Default: Run tier1 only
addopts = "-m 'not tier2 and not tier3'"
```

### Phase 2: Audit and mark existing tests (2-3 hours)

**Strategy**: Bottom-up approach (start with obviously slow tests)

**Step 1**: Mark obviously slow tests as tier2
```bash
# Search for documented slow operations
grep -r "expensive\|slow\|threshold\|glm\|bootstrap\|permutation" nltools/tests/

# Files to prioritize:
# - test_brain_data.py: threshold tests (9 tests @ ~7.2s each)
# - test_brain_data.py: GLM tests
# - test_models.py: model fitting with CV
# - test_datasets.py: network tests
# - test_efficient_copy.py: performance benchmarks
```

**Step 2**: Mark integration tests as tier2
```bash
# Network-dependent
test_datasets.py::TestIntegration::*

# Full-pipeline workflows
test_brain_data.py tests using sim_brain_data (238K voxels)
```

**Step 3**: Convert remaining tests to tier1-friendly
```python
# Example: Convert to fast fixture
# BEFORE (slow - uses full brain):
def test_regress_with_residuals(sim_brain_data):
    result = sim_brain_data.regress()
    assert result['residual'].shape == sim_brain_data.shape

# AFTER (fast - uses minimal fixture):
def test_regress_with_residuals(minimal_brain_data):
    result = minimal_brain_data.regress()
    assert result['residual'].shape == minimal_brain_data.shape
```

**Step 4**: Add explicit tier1 markers to critical tests
```python
@pytest.mark.tier1
def test_brain_data_init_from_numpy_array():
    """Critical API contract - must always work"""
    ...
```

### Phase 3: Optimize test execution (~1-2 hours)

**3a. Reduce unnecessary fixture scope**
```python
# Example: These can be function-scoped for isolation
@pytest.fixture(scope="function")  # was "module"
def sim_brain_data():
    # Only expensive for tier2 tests
    ...
```

**3b. Create more minimal fixtures**
```python
@pytest.fixture(scope="function")
def micro_brain_data():
    """Ultra-minimal: 3 voxels, 10 timepoints for API tests only"""
    # Even faster than minimal_brain_data for pure API contract tests
    ...

@pytest.fixture(scope="function")
def minimal_adjacency():
    """4x4 matrix for fast adjacency tests"""
    ...
```

**3c. Use pytest-xdist for parallel execution**
```bash
# Add to dev dependencies in pyproject.toml
pytest-xdist>=3.0.0

# Run tier1 in parallel (4 cores)
uv run pytest -m tier1 -n 4

# Even tier2 benefits
uv run pytest -m tier2 -n 4
```

### Phase 4: Update development workflows

**Update CLAUDE.md** with new commands:
```bash
# Fast development loop (default)
uv run pytest                          # tier1 only (~2 min)

# Specific tiers
uv run pytest -m tier1                 # explicit tier1
uv run pytest -m tier2                 # comprehensive (~15 min)
uv run pytest -m "tier1 or tier2"      # both tiers (~17 min)

# Parallel execution
uv run pytest -m tier1 -n auto         # auto-detect cores

# File-specific during development
uv run pytest nltools/tests/shell/test_brain_data.py::TestBrainData::test_init -x

# Before commits: run tier1 + related tier2
uv run pytest -m tier1  # always
uv run pytest nltools/tests/shell/test_brain_data.py -m tier2  # affected module

# Before releases: full suite
uv run pytest -m "tier1 or tier2" -n auto
```

**Update CI/GitHub Actions** (if applicable):
```yaml
# .github/workflows/test.yml
jobs:
  tier1:
    name: Fast Core Tests
    runs-on: ubuntu-latest
    steps:
      - run: uv run pytest -m tier1 -n auto
    # Runs on: every push, every PR

  tier2:
    name: Comprehensive Tests
    runs-on: ubuntu-latest
    steps:
      - run: uv run pytest -m tier2 -n auto
    # Runs on: PR merge, nightly schedule, release tags
```

---

## Parallel Testing Safety & Correctness

**Date Added**: 2025-10-30
**Context**: Ensuring pytest-xdist parallelization is safe and correct

### Overview

Our test suite uses **pytest-xdist** for parallel execution, achieving ~1.42× speedup with 4 workers. This section documents the safety considerations for parallel testing to ensure correctness and avoid race conditions.

**Key Result**: Our current setup is **safe for parallel execution** ✅

---

### Fixture Isolation Between Workers

**Status: ✅ SAFE - No Changes Needed**

pytest-xdist runs each worker in a **completely isolated Python process**. Our fixtures are correctly designed for this:

#### Function-Scoped Fixtures (Perfect Isolation)
```python
# conftest.py lines 25-83, 229-275, 277-316
@pytest.fixture(scope="function")
def minimal_brain_data():
    """Each test gets a fresh instance"""
    # 5 voxels, 50 timepoints
    # Isolated per worker, isolated per test

@pytest.fixture(scope="function")
def small_brain_data_for_cv():
    """CV testing fixture"""
    # Each worker creates its own
    # No sharing, no conflicts
```

**Why Safe**: Each test invocation gets a brand new fixture instance. Workers never share data.

#### Module-Scoped Fixtures (Worker-Local Copies)
```python
# conftest.py lines 10-22, 86-96, 99-138, 140-152
@pytest.fixture(scope="module")
def sim_brain_data():
    """Full brain data (238,955 voxels)"""
    # Expensive to create, but safe:
    # Each worker gets its OWN module scope
    # Not shared across workers!
```

**How pytest-xdist Works:**
- Each worker process maintains **separate module scopes**
- Module-scoped fixtures are created **once per worker** (not once globally)
- Workers never share fixture instances

**Example**: With 4 workers running `test_brain_data.py`:
- Worker 1 creates `sim_brain_data` instance A
- Worker 2 creates `sim_brain_data` instance B
- Worker 3 creates `sim_brain_data` instance C
- Worker 4 creates `sim_brain_data` instance D
- Each worker uses only its own instance

#### Read-Only File Path Fixtures (Safe)
```python
# conftest.py lines 154-199
@pytest.fixture(scope="module")
def old_h5_brain(request):
    """Returns path to shared H5 file"""
    return os.path.join(tests_dir, "data", "old_brain.h5")
```

**Why Safe**: Multiple workers can **read** the same file simultaneously. OS handles file locking for reads.

#### Dependent Fixtures (Safe)
```python
# conftest.py lines 202-227
@pytest.fixture(scope="module")
def regress_result(sim_brain_data):
    """Creates new objects from sim_brain_data"""
    return {
        "beta": sim_brain_data.copy(),
        "residual": fake_timeseries - fake_timeseries.mean(),
        # ...
    }
```

**Why Safe**: Even though it depends on `sim_brain_data`, it creates **new objects**. Each worker's `sim_brain_data` is independent, so derived fixtures are also independent.

---

### GPU/CUDA Parallelization

**Status: ✅ SAFE - Excellent Architecture**

Our backend abstraction (nltools/backends.py) is designed for parallel execution with GPUs.

#### How Our GPU Tests Work

```python
# test_backends.py, test_models.py, test_ridge.py
@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_torch_backend_selection():
    backend = Backend('torch')  # Creates fresh instance
    # Each test/worker has its own Backend instance
```

**Key Safety Features:**

1. **Per-Process Device Selection** (backends.py lines 55-79):
   ```python
   def _init_torch(self):
       if torch.cuda.is_available():
           self._torch_device = torch.device('cuda')  # Uses default GPU
   ```
   - Each worker process independently selects GPU
   - PyTorch/CUDA handle multi-process access automatically
   - GPU 0 is shared safely via CUDA's scheduling

2. **No Global State**:
   - Each test creates its own `Backend` instance
   - No shared backend singleton
   - Workers never conflict

3. **PyTorch/CUDA Multi-Process Safety**:
   - CUDA driver manages concurrent access from multiple processes
   - GPU memory allocation is process-isolated
   - Context switching handled automatically

#### Parallel GPU Performance Characteristics

**Current Results** (from commit 89236cb):
- 4 workers: 35s serial → 24s parallel = **1.42× speedup**
- GPU tests don't bottleneck the suite

**What This Means**:
- GPU isn't the bottleneck yet
- Each worker gets enough GPU time
- Memory isn't constrained

**Potential Future Considerations** (if scaling beyond 4 workers):
- GPU memory limits: Each process needs VRAM
- Context switching overhead: More processes = more switching
- CPU-GPU transfer overhead: Multiple processes uploading data

**Monitoring GPU Usage** (optional):
```bash
# In separate terminal
watch -n 0.5 nvidia-smi  # For CUDA GPUs

# Run tests
uv run pytest -m tier2 -n 4

# Look for:
# - Memory usage per process
# - GPU utilization (should be 80-100%)
# - Temperature (throttling indicator)
```

#### GPU Test Recommendations

**Current Setup: No Changes Needed**

**If GPU tests become bottleneck:**
1. Reduce workers for GPU-heavy tests: `pytest -m tier2 -n 2`
2. Mark GPU tests separately: `@pytest.mark.gpu` and run with different worker count
3. Limit worker count based on GPU memory: `pytest -n $(nvidia-smi --query-gpu=memory.free --format=csv,noheader | awk '{print int($1/2000)}')`

---

### File I/O Safety

**Status: ⚠️ CHECK - Needs Audit**

#### Read Operations (Safe ✅)

Our test data files are **read-only**:
```python
# conftest.py lines 154-199
old_h5_brain(request)  # Returns path to nltools/tests/data/old_brain.h5
```

**Why Safe**: Multiple processes can read the same file simultaneously. OS handles this safely.

#### Write Operations (NEEDS AUDIT ⚠️)

**Concern**: If tests write temporary files without unique names, workers could conflict.

**Example of UNSAFE pattern** (hypothetical):
```python
def test_brain_data_write(sim_brain_data):
    sim_brain_data.write("temp_output.h5")  # ❌ RACE CONDITION!
    # Worker 1 and Worker 2 both try to write temp_output.h5
```

**Example of SAFE pattern**:
```python
def test_brain_data_write(sim_brain_data, tmp_path):
    output_file = tmp_path / "test_output.h5"  # ✅ UNIQUE PER TEST
    sim_brain_data.write(output_file)
    # tmp_path is unique per test invocation
```

#### Action Item: Audit Write Operations

**TODO for team**:
```bash
# Search for potential write operations
cd nltools/tests
grep -r "\.write\|\.to_\|\.save" . | grep -v "tmp_path\|tmpdir"

# Check results - ensure all use tmp_path or unique names
```

**Files to check**:
- `test_brain_data.py` - Brain_Data.write() tests
- `test_adjacency.py` - Adjacency.write() tests
- `test_design_matrix*.py` - CSV/file exports

**Fix pattern if needed**:
```python
# Add tmp_path fixture parameter
def test_write_operation(brain_data, tmp_path):
    output = tmp_path / "unique_name.h5"
    brain_data.write(output)
```

#### Current Cleanup Behavior

From CLAUDE.md:
```bash
rm -f *.log *.csv *.nii.gz  # Manual cleanup recommended
```

This suggests tests may leave artifacts. Consider adding:
```python
# In conftest.py
@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Auto-cleanup test artifacts after each test"""
    yield  # Run test
    # Clean up
    for pattern in ['*.log', '*.csv', '*.nii.gz']:
        for f in Path('.').glob(pattern):
            if f.name not in ['permanent_file.csv']:  # Whitelist
                f.unlink(missing_ok=True)
```

---

### Random Seed Management

**Status: ✅ EXCELLENT**

Our fixtures properly seed random number generators for reproducibility:

```python
# conftest.py examples
@pytest.fixture(scope="function")
def minimal_brain_data():
    np.random.seed(42)  # ✅ Deterministic
    # Each worker gets same seed for same test
    # Results are reproducible
```

**Why This Works**:
- Each process has independent numpy random state
- Seeding at fixture level ensures consistency
- Tests are deterministic regardless of worker count

**Verification**:
```bash
# Run same test multiple times in parallel
for i in {1..5}; do
    uv run pytest -k test_specific_random -n 4 --tb=no -q
done
# Results should be identical every time
```

---

### Brain_Data / Shared Memory Objects

**Status: ✅ SAFE - Well-Designed**

Our core classes (Brain_Data, Adjacency, Design_Matrix) follow the "functional-core, imperative-shell" pattern safely:

**Key Safety Properties**:

1. **Instance-Based, Not Singletons**:
   ```python
   # Each test creates fresh instances
   dat = Brain_Data(nifti_img, mask=mask_img)
   # No global state, no shared objects
   ```

2. **Copy Semantics** (when needed):
   ```python
   # From efficient_copy pattern
   dat_copy = dat.copy()  # Explicit copying
   # Original and copy are independent
   ```

3. **No Mutable Class Variables**:
   - All state is instance-level
   - Workers never share Brain_Data instances

**Verification**:
```python
# Test isolation
def test_brain_data_independence():
    dat1 = Brain_Data(...)
    dat2 = Brain_Data(...)
    dat1.data[0, 0] = 999  # Modify dat1
    assert dat2.data[0, 0] != 999  # dat2 unaffected ✅
```

---

### nilearn Integration & Nested Parallelism

**Status: 📊 MONITOR - Potential Performance Impact**

Many of our tier2 tests use nilearn operations that may spawn their own parallel workers:

**Potentially Parallel nilearn Operations**:
- `FirstLevelModel` (GLM fitting) - may use joblib parallelism
- Connected component labeling (threshold clustering)
- Spatial smoothing with large kernels

**Concern: Thread Pool Explosion**

If both pytest-xdist and nilearn try to parallelize:
- pytest-xdist: 4 worker processes
- nilearn/joblib: 4 threads per process
- Total: 16 threads competing for resources
- Result: CPU thrashing, slower than serial

**Current Evidence**:
Our 1.42× speedup with 4 workers suggests this isn't a major issue yet, but worth monitoring.

**Solution: Limit nilearn's Parallelism**

Add to conftest.py:
```python
@pytest.fixture(scope="session", autouse=True)
def configure_parallel_environment():
    """
    Limit thread pools when using pytest-xdist.

    When pytest-xdist spawns multiple workers, we don't want
    each worker to spawn additional threads for numpy/blas ops.
    This prevents thread pool explosion (4 workers × 4 threads = 16).
    """
    import os
    # Limit BLAS/LAPACK threads (used by numpy, scipy, nilearn)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    # Limit joblib (used by sklearn, nilearn)
    os.environ['JOBLIB_START_METHOD'] = 'forkserver'  # Safer forking
```

**When to Apply**:
- If tier2 parallel tests are slower than expected
- If CPU usage goes above ~400% with 4 workers (indicates thrashing)
- If `top` shows many idle threads

**Testing the Fix**:
```bash
# Before: May thrash with many threads
uv run pytest -m tier2 -n 4

# After: Each worker uses 1 thread, cleaner parallelism
OMP_NUM_THREADS=1 uv run pytest -m tier2 -n 4

# Compare wall-clock time and CPU usage (via `time` command)
```

---

### Performance Monitoring Checklist

Use these commands to diagnose parallel testing issues:

#### CPU Usage
```bash
# Monitor during test run
top -pid $(pgrep -f pytest | tr '\n' ',' | sed 's/,$//')

# Look for:
# - CPU% per worker (should be near 100% if CPU-bound)
# - Total CPU% (should be ~400% with 4 workers, not >800%)
# - Many threads per process (sign of nested parallelism)
```

#### GPU Usage (if using CUDA)
```bash
# Continuous monitoring
watch -n 0.5 nvidia-smi

# Look for:
# - Memory usage per process
# - GPU utilization (80-100% is good)
# - Multiple processes using GPU (expected)
```

#### I/O Wait
```bash
# Check if I/O is bottleneck
iostat -x 1

# Look for:
# - %iowait (should be <10%)
# - High await times (sign of I/O bottleneck)
```

#### Test Timing Comparison
```bash
# Serial timing
time uv run pytest -m tier1

# Parallel timing
time uv run pytest -m tier1 -n 4

# Speedup = serial_time / parallel_time
# Ideal: 4.0× (never achievable due to overhead)
# Good: 2-3× (our current 1.42× is acceptable)
# Bad: <1.2× (parallel overhead too high)
```

---

### Summary: Is Parallel Testing Safe?

**✅ SAFE FOR USE** - Our current setup is well-architected for parallel execution.

**Key Strengths**:
1. Fixture isolation is correct (function + module scopes)
2. GPU/PyTorch backend is designed for multi-process use
3. Random seeding ensures reproducibility
4. No shared mutable state in core classes

**Action Items**:
1. ⚠️ **Audit file writes** - Ensure all use `tmp_path` fixture
2. 📊 **Monitor nilearn parallelism** - Add thread limits if CPU thrashes
3. 🧹 **Consider autouse cleanup fixture** - Reduce manual artifact cleanup

**For Future Reference**:
- This setup scales safely to 4-8 workers on typical hardware
- GPU memory may limit scaling (monitor with `nvidia-smi`)
- If tier2 tests slow down with parallelism, limit nilearn threads
- Always verify correctness: same results parallel vs. serial

---

## Test Classification Guidelines

### Quick Reference: Which tier?

**Use this decision tree**:

```
Is it testing an API contract (params, types, shapes)?
├─ YES → Uses minimal_brain_data → tier1
└─ NO → Is the operation expensive (>2s)?
    ├─ YES → Does it require full brain data?
    │   ├─ YES → sim_brain_data → tier2
    │   └─ NO → Can we make it faster?
    │       ├─ YES → Optimize → tier1
    │       └─ NO → tier2
    └─ NO → Does it require network/external resources?
        ├─ YES → tier2 (or tier3 if truly optional)
        └─ NO → tier1
```

### Detailed Classification Rules

**Tier 1 (Fast Core)**:
```python
✅ Mathematical correctness (invariants, properties)
✅ API contracts (parameter validation, return types)
✅ Shape preservation checks
✅ Small-data computations (<1s)
✅ Unit tests (single function/method, mocked dependencies)
✅ Fast fixtures (minimal_brain_data, small synthetic data)

❌ GLM fitting (FirstLevelModel)
❌ Cluster thresholding (connected components)
❌ Bootstrap/permutation (resampling loops)
❌ Full brain operations (238K voxels)
❌ Network I/O
❌ Integration tests (multi-component pipelines)
```

**Tier 2 (Comprehensive)**:
```python
✅ Expensive nilearn operations (threshold, GLM)
✅ Integration tests (full pipelines)
✅ Resampling methods (bootstrap, permutation)
✅ Full brain data (realistic neuroimaging scale)
✅ Cross-validation with multiple folds
✅ Network-dependent tests
✅ Performance benchmarks
✅ PyTorch operations (if torch is optional dependency)

❌ Stress tests (>30s)
❌ Manual verification tests
❌ Experimental features not yet released
```

**Tier 3 (Optional)**:
```python
✅ Stress tests (huge datasets, extreme parameter values)
✅ Manual inspection tests (visual outputs)
✅ Experimental features under development
✅ GPU-required tests (if GPU is rare in CI)
✅ Debugging-only tests
```

---

## Specific File Recommendations

### HIGH PRIORITY: test_brain_data.py (91 tests, ~151s)

**Estimated breakdown**:
```
Tier 1 (~50 tests, ~30s target):
  - Initialization & I/O (test_load, test_write) - optimize with tmpdir
  - Math operations (test_mean, test_std, test_add)
  - Append/concatenate operations
  - Basic shape/indexing tests
  - Extract ROI (with minimal fixtures)
  - Decompose (PCA/ICA on small data)

Tier 2 (~35 tests, ~120s):
  - Threshold tests (9 tests @ ~7.2s each = ~65s) → tier2
  - GLM regression tests (~5-6s each)
  - Hyperalignment with full data
  - Bootstrap/permutation tests
  - Distance computations on full brain
  - Predict with CV on full data

Skip/Remove (~6 tests):
  - test_distance() - already marked for refactoring
```

**Optimization opportunities**:
```python
# BEFORE: Uses expensive full brain fixture (238K voxels)
def test_regress(sim_brain_data):  # ~5-6s
    result = sim_brain_data.regress()
    assert 'beta' in result

# AFTER: Split into tier1 (contract) + tier2 (integration)
@pytest.mark.tier1
def test_regress_output_structure(minimal_brain_data):  # <1s
    """Test API contract: regress returns correct output structure"""
    result = minimal_brain_data.regress()
    assert 'beta' in result
    assert 'residual' in result
    assert result['beta'].shape == minimal_brain_data.shape

@pytest.mark.tier2
def test_regress_glm_correctness(sim_brain_data):  # ~5-6s
    """Test statistical correctness with realistic brain data"""
    result = sim_brain_data.regress()
    # Test statistical properties with full brain
    ...
```

### HIGH PRIORITY: test_design_matrix_new.py (68 tests, ~?s)

**Current status**: Newly implemented Polars version

**Strategy**:
```
Tier 1 (~55 tests, <30s target):
  - All Polars operations (should be fast!)
  - Convolve operations (synthetic small HRF)
  - Append/concatenate
  - Polynomial/DCT basis functions (small designs)
  - Clean/VIF calculations (small design matrices)

Tier 2 (~10 tests):
  - Large design matrices (500+ TRs)
  - Complex multi-run operations
  - Integration with real HRF files
  - Performance benchmarks vs pandas version
```

**Likely already fast** - most should be tier1!

### MODERATE PRIORITY: test_models.py (41 tests, ~?s)

**Strategy**:
```
Tier 1 (~25 tests):
  - Model initialization
  - Parameter validation
  - Fit/predict on small synthetic data
  - Transform tests with minimal data

Tier 2 (~15 tests):
  - Cross-validation tests (multiple folds = multiple fits)
  - Model fitting on full brain data
  - Searchlight tests (neighborhood computations)
  - ROI aggregation with realistic data
```

### MODERATE PRIORITY: test_srm.py (34 tests, ~?s)

**Current**: Property-based tests (good!)

**Strategy**:
```
Tier 1 (~28 tests):
  - Initialization & parameters
  - Fit/transform on small synthetic data (5 subjects, 100 TPs, 10 features)
  - Mathematical invariants (orthogonality, dimensionality)
  - Edge cases (NotFittedError, shape mismatches)

Tier 2 (~6 tests):
  - Stress tests (many subjects, many features)
  - Convergence tests (many iterations)
  - Integration with Brain_Data objects
```

**Optimization**: Already uses efficient synthetic fixtures - likely fast!

### LOW PRIORITY: test_stats.py (15 tests)

**Strategy**: Most stats functions are fast (numpy operations)
```
Tier 1 (~12 tests):
  - Correlation, ISC (inter-subject correlation)
  - Matrix algebra helpers
  - Threshold functions (without expensive clustering)

Tier 2 (~2 tests):
  - ISC test (currently skipped due to known bugs)
  - Threshold with cluster extent (expensive)
```

### LOW PRIORITY: test_datasets.py (9 tests)

**Strategy**:
```
Tier 1 (~7 tests):
  - All mocked tests (no network)
  - Parameter validation
  - Error handling

Tier 2 (~1 test):
  - test_real_collection_download (requires internet)

Tier 3 (~1 test):
  - Stress test downloading large collections
```

---

## Expected Outcomes

### Before Implementation
```
Full suite: ~385 tests in ~20 minutes (estimated)
No tiers, no parallelization
Development cycle: Run all tests or risk missing breakage
CI: Either slow (run all) or risky (run subset ad-hoc)
```

### After Implementation (Conservative Estimates)

**Tier 1 (Development)**:
```
~200 tests in <2 minutes (avg 0.6s/test)
With pytest-xdist -n4: <1 minute
Development cycle: Fast feedback, run on every save
CI: Run on every commit
```

**Tier 2 (Release)**:
```
~150 tests in <15 minutes (avg 6s/test)
With pytest-xdist -n4: <5 minutes
Development cycle: Run before commits
CI: Run on PR merge, nightly, releases
```

**Combined**:
```
Full suite: ~350 tests in <17 minutes (vs. ~20 min before)
With pytest-xdist: <6 minutes total (3× speedup!)
Plus faster iteration: tier1 in <1 min vs 20 min
```

### Key Improvements

1. **Development velocity**: 20× faster feedback (1 min vs 20 min)
2. **Focused testing**: Run tier1 during development, tier2 before commits
3. **Parallel execution**: 3-4× speedup with pytest-xdist
4. **Clear semantics**: Developers know what each tier means
5. **CI optimization**: Fast tier1 on every commit, comprehensive tier2 on merge
6. **Lower costs**: Faster CI runs = fewer compute minutes

---

## Risk Mitigation

### "What if we miss bugs by not running tier2?"

**Mitigation**:
- Tier1 still tests correctness, just with smaller data
- Enforce tier2 runs before merges (PR requirement)
- Nightly CI runs full suite
- Release checklist: full suite must pass

### "What if marking tests is inconsistent?"

**Mitigation**:
- Clear decision tree (see above)
- Document rationale in test docstrings
- Review markers in PR reviews
- Start with obvious cases (threshold, GLM, network)

### "What if tier1 becomes slow over time?"

**Mitigation**:
- Monitor test duration (pytest --durations=20)
- Set time budget: tier1 must stay <2 min
- Add pre-commit hook: warn if tier1 >2 min
- Regular audits: move slow tests to tier2

### "What if we over-optimize and sacrifice coverage?"

**Mitigation**:
- **Never skip tests** - only organize them
- Tier2 still runs regularly (nightly, pre-release)
- Code coverage tracking: ensure tier1 covers critical paths
- Mathematical correctness tests stay in tier1

---

## Alternative Approaches Considered

### Alternative 1: Fixture-based separation only
```python
# Mark by fixture usage
@pytest.mark.minimal_fixture
@pytest.mark.full_fixture
```

**Pros**: Automatic based on fixture
**Cons**: Doesn't capture other slow operations (GLM, network)
**Decision**: Use tiers + fixtures (complementary)

### Alternative 2: Duration-based automatic marking
```python
# After first run, auto-mark based on timing
pytest --store-durations
pytest -m "duration<2s"  # auto-generated
```

**Pros**: Objective timing data
**Cons**: Requires initial run, machine-dependent, less semantic
**Decision**: Manual marking with clear semantics

### Alternative 3: Directory-based separation
```
tests/fast/
tests/slow/
```

**Pros**: Simple, no markers needed
**Cons**: Duplicates structure, unclear for mixed files
**Decision**: Keep current structure, use markers

### Alternative 4: Tag by functionality
```python
@pytest.mark.integration
@pytest.mark.unit
@pytest.mark.glm
@pytest.mark.threshold
```

**Pros**: Fine-grained control
**Cons**: Too many markers, unclear which to run for fast feedback
**Decision**: Use tiers (coarse) + optional tags (fine) if needed later

---

## Next Steps

### Immediate (This Session)
1. ✅ **Analyze current test suite** (completed above)
2. ⏳ **Get approval for approach** (awaiting Eshin's feedback)

### Phase 1: Infrastructure (30 minutes)
3. Add pytest configuration to pyproject.toml
4. Update CLAUDE.md with new test commands
5. Document tier guidelines in knowledge-base.md

### Phase 2: Mark High-Impact Tests (2 hours)
6. Mark tier2: test_brain_data.py threshold tests (9 tests, ~65s savings)
7. Mark tier2: test_brain_data.py GLM tests
8. Mark tier2: test_datasets.py integration test
9. Mark tier2: test_efficient_copy.py performance benchmarks

### Phase 3: Optimize Fixtures (1 hour)
10. Create micro_brain_data fixture (even smaller than minimal)
11. Convert tier1 tests to use minimal fixtures
12. Audit fixture scope (module vs function)

### Phase 4: Validation (30 minutes)
13. Run tier1, measure time (target: <2 min)
14. Run tier2, measure time (target: <15 min)
15. Verify all tests still pass

### Phase 5: CI Integration (if applicable)
16. Update CI configuration for tiered testing
17. Set up nightly tier2 runs
18. Add PR checks (tier1 required, tier2 on merge)

---

## Questions for Eshin

1. **Approval of approach**: Does this tiered strategy align with your development workflow?

2. **Time budgets**: Are the targets reasonable?
   - Tier 1: <2 minutes (for rapid iteration)
   - Tier 2: <15 minutes (for comprehensive validation)

3. **CI setup**: Do you have GitHub Actions or other CI to configure?

4. **Parallel execution**: Should we add pytest-xdist to dependencies?

5. **Coverage requirements**: Do you want to maintain code coverage metrics?

6. **Migration timeline**:
   - Quick win: Mark obvious slow tests (~1 hour)
   - Full implementation: Mark all tests + optimize fixtures (~4 hours)
   - Which priority for this release?

7. **Breaking changes**: v0.6.0 allows API changes - any test refactoring opportunities?

---

## References

- Test organization: nltools/tests/{shell,core,support}/
- Current fixtures: nltools/tests/conftest.py
- Performance notes: test_brain_data.py lines 7-22
- pytest markers: https://docs.pytest.org/en/stable/example/markers.html
- pytest-xdist: https://pytest-xdist.readthedocs.io/
