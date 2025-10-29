# Plan: Complete Sprint 1 Documentation & Benchmarking

## Overview
Complete the final two tasks from model-spec.md Sprint 1 (line 1702):
- Benchmark performance (comprehensive suite with CSV export)
- Update jupyter-book API documentation pages

## Phase 1: Create Benchmark Suite

**Create:** `benchmarks/benchmark_ridge.py`

**Benchmark scenarios:**
1. **Basic comparison** - CPU (numpy) vs GPU (torch) on medium dataset
2. **Problem size scaling** - Small (100×1k), Medium (300×50k), Large (1000×200k)
3. **Cross-validation impact** - 1-fold vs 5-fold CV overhead
4. **Auto-selection validation** - Verify auto backend chooses correctly
5. **Real-world neuroimaging** - 300×100k voxels, 5-fold CV (typical fMRI analysis)

**Output:** `benchmarks/results_ridge_performance.csv`

**CSV structure:**
```csv
scenario,backend,n_samples,n_features,cv_folds,time_seconds,memory_mb,speedup_vs_numpy
basic_medium,numpy,300,50000,1,0.523,120,1.0
basic_medium,torch-mps,300,50000,1,0.052,250,10.1
...
```

**Metrics tracked:**
- Execution time (seconds)
- Peak memory usage (MB)
- Speedup vs numpy baseline
- Backend selected (for auto mode)

---

## Phase 2: Create API Documentation Pages

### 2a. Create `docs/api/backends.md`

**Content:**
```markdown
# `nltools.backends`

**Backend Abstraction for CPU/GPU Operations**

Provides transparent CPU/GPU acceleration for ridge regression and other algorithms.
Supports NumPy (CPU), PyTorch (CUDA/MPS/CPU), and automatic backend selection.

\```{eval-rst}
.. automodule:: nltools.backends
    :members:
    :undoc-members:
    :show-inheritance:
\```

## Quick Start

\```python
from nltools.backends import Backend, check_gpu_available

# Check GPU availability
available, info = check_gpu_available()
print(f"GPU available: {available}")
print(f"Device: {info['device']}")

# Use backends
backend_cpu = Backend('numpy')
backend_gpu = Backend('torch')  # Auto-detects cuda/mps
backend_auto = Backend('auto')  # Smart selection
\```

## See Also

- [Performance Guide](../performance.md) - When to use GPU acceleration
- [Ridge Regression](algorithms.md#ridge-regression) - Usage in algorithms
```

### 2b. Create `docs/api/algorithms.md`

**Content:**
```markdown
# `nltools.algorithms`

**Optimized Algorithms for Neuroimaging**

High-performance implementations of core algorithms with optional GPU acceleration.

## Ridge Regression

\```{eval-rst}
.. automodule:: nltools.algorithms.ridge
    :members:
    :undoc-members:
    :show-inheritance:
\```

## HRF Functions

\```{eval-rst}
.. automodule:: nltools.algorithms.hrf
    :members:
    :undoc-members:
    :show-inheritance:
\```

## Shared Response Model

\```{eval-rst}
.. automodule:: nltools.algorithms.srm
    :members:
    :undoc-members:
    :show-inheritance:
\```
```

---

## Phase 3: Create Performance Documentation

**Create:** `docs/performance.md`

**Content structure:**
1. **Introduction** - Why performance matters for neuroimaging
2. **Backend Overview** - numpy vs torch, when to use each
3. **Benchmark Results** - Load CSV and render tables
4. **Recommendations** - Problem size thresholds, best practices
5. **Troubleshooting** - Common issues (GPU memory, CUDA errors)

**Key sections:**

### 3a. Backend Comparison Table
```markdown
### CPU vs GPU Performance

Results from comprehensive benchmarking on typical neuroimaging datasets:

| Dataset Size | CPU (numpy) | GPU (torch) | Speedup |
|--------------|-------------|-------------|---------|
| Small: 100×1k | 0.05s | 0.08s | 0.6x (overhead) |
| Medium: 300×50k | 0.52s | 0.05s | 10.4x |
| Large: 1000×200k | 5.2s | 0.18s | 28.9x |
```

### 3b. Cross-Validation Impact
```markdown
### Cross-Validation Overhead

5-fold CV impact on different backends:

| Dataset | numpy (no CV) | numpy (5-fold) | torch (5-fold) | CV Speedup |
|---------|---------------|----------------|----------------|------------|
| 300×100k | 0.50s | 2.5s | 0.13s | 19.2x |
```

### 3c. Auto-Selection Validation
```markdown
### Auto Backend Selection

Verifying `backend='auto'` heuristics:

| Scenario | Expected | Actual | Correct? |
|----------|----------|--------|----------|
| 100×1k, no CV | numpy | numpy | ✅ |
| 300×100k, 5-fold CV | torch | torch-mps | ✅ |
| 1000×200k | torch | torch-mps | ✅ |
```

### 3d. Real-World Scenarios
```markdown
### Neuroimaging Use Cases

Performance on realistic fMRI analysis workflows:

| Workflow | Description | CPU Time | GPU Time | Speedup |
|----------|-------------|----------|----------|---------|
| Whole-brain prediction | 300×100k voxels, 5-fold CV | 2.5s | 0.13s | 19x |
| Searchlight prep | 1000×200k features | 5.2s | 0.18s | 29x |
```

---

## Phase 4: Update Table of Contents

**Modify:** `docs/_toc.yml`

**Changes:**

```yaml
# Around line 27-37 (API Reference section)
  - caption: API Reference
    chapters:
    - file: api/analysis
    - file: api/algorithms        # NEW
    - file: api/backends          # NEW
    - file: api/crossval
    - file: api/data
    - file: api/dataset
    - file: api/filereader
    - file: api/mask
    - file: api/prefs
    - file: api/stats
    - file: api/utils

# Around line 38-40 (Development section)
  - caption: Development
    chapters:
    - file: contributing
    - file: performance           # NEW
```

---

## Phase 5: Implementation Steps

### Step 1: Create Benchmark Script
```bash
# Create benchmarks directory if needed
mkdir -p benchmarks

# Write benchmark_ridge.py
# - Import ridge_svd, ridge_cv, Backend
# - Define benchmark functions for each scenario
# - Run benchmarks with error handling
# - Export to CSV with pandas
```

### Step 2: Run Benchmarks
```bash
# Run comprehensive benchmark suite
uv run python benchmarks/benchmark_ridge.py

# Output: benchmarks/results_ridge_performance.csv
```

### Step 3: Create Documentation Files
```bash
# Create new API docs
touch docs/api/backends.md
touch docs/api/algorithms.md

# Create performance guide
touch docs/performance.md
```

### Step 4: Write Documentation Content
- `backends.md` - API reference with examples
- `algorithms.md` - Combined API for ridge, hrf, srm
- `performance.md` - Load CSV, render tables, add recommendations

### Step 5: Update _toc.yml
- Add `api/algorithms` and `api/backends` to API Reference
- Add `performance` to Development section

### Step 6: Build & Verify
```bash
# Build documentation
uv run jupyter-book build docs/

# Check for warnings/errors
# Verify new pages appear in TOC
# Test links and autodoc rendering
```

---

## Expected Outcomes

**Files created:**
- ✅ `benchmarks/benchmark_ridge.py` (~150 lines)
- ✅ `benchmarks/results_ridge_performance.csv` (benchmark data)
- ✅ `docs/api/backends.md` (~50 lines)
- ✅ `docs/api/algorithms.md` (~80 lines)
- ✅ `docs/performance.md` (~200 lines with tables)

**Files modified:**
- ✅ `docs/_toc.yml` (add 3 new entries)

**Documentation updates:**
- ✅ Complete API reference for backends module
- ✅ Complete API reference for algorithms package
- ✅ Comprehensive performance guide with benchmark results
- ✅ Future-proof for visualizations (CSV can be re-used)

**Deliverables:**
- Detailed performance data for CPU vs GPU
- Clear recommendations for users
- Validation of auto-selection heuristics
- Ready for Sprint 1 completion ✅

---

## Time Estimate
- Phase 1 (Benchmark script): 30-45 min
- Phase 2 (API docs): 15-20 min
- Phase 3 (Performance guide): 30-45 min
- Phase 4 (Update TOC): 5 min
- Phase 5 (Build & verify): 10-15 min

**Total: 1.5-2 hours**
