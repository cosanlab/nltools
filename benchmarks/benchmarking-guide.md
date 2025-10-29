# Ridge Regression Benchmarking Guide

## Overview

This directory contains systematic benchmarks for ridge regression performance across realistic neuroimaging workflows. The benchmarks compare CPU (NumPy) vs GPU (PyTorch) backends across different problem sizes and use cases.

---

## Benchmark Scripts

### 1. `benchmark_ridge.py` (Original - Completed)
**Purpose:** Exploratory benchmarks across various problem sizes
**Status:** ✅ Completed
**Results:** `results_ridge_performance.csv`
**Documentation:** Results integrated into `docs/performance.md`

### 2. `benchmark_ridge_systematic.py` (Current - Ready to run)
**Purpose:** Systematic grid covering realistic neuroimaging scenarios
**Status:** Ready to run
**Results:** Will generate `results_ridge_systematic.csv`

---

## Systematic Benchmark Design

### Benchmark Grid (8 conditions)

**Dimensions:**
1. **Time-series length** (n_samples)
   - 500 timepoints (typical task fMRI study)
   - 1000 timepoints (typical naturalistic/movie fMRI)

2. **Number of voxels** (n_features)
   - 50,000 voxels (typical 3mm resolution whole-brain)
   - 230,000 voxels (typical 2mm resolution whole-brain)

3. **Estimation style**
   - **Estimates only**: Fixed α=1.0, no cross-validation
     - Use case: You only care about coefficient estimates/feature weights
     - Function: `ridge_svd(X, y, alpha=1.0)`
   - **Fit only**: 5-fold CV with hyperparameter search
     - Use case: You care about out-of-sample prediction accuracy
     - Function: `ridge_cv(X, y, cv=5, alphas=10 values)`

**Total:** 2 × 2 × 2 = 8 conditions × 2 backends (NumPy, PyTorch) = **16 benchmark runs**

---

## Runtime Estimates

### Total Estimated Runtime: **~55-60 minutes**

**Breakdown by condition type:**

#### Estimates Only (Fast - 4 conditions, ~2 minutes total)
| Condition | NumPy Time | PyTorch Time | Total |
|-----------|------------|--------------|-------|
| 500×50k   | ~1s        | ~1s          | ~2s   |
| 500×230k  | ~10s       | ~15s         | ~25s  |
| 1000×50k  | ~4s        | ~4s          | ~8s   |
| 1000×230k | ~40s       | ~40s         | ~80s  |

**Subtotal:** ~115 seconds (~2 minutes)

#### Fit Only with 5-Fold CV (Slow - 4 conditions, ~55 minutes total)
| Condition | NumPy Time | PyTorch Time | Total |
|-----------|------------|--------------|-------|
| 500×50k   | ~120s      | ~60s         | ~180s (3 min)   |
| 500×230k  | ~600s      | ~300s        | ~900s (15 min)  |
| 1000×50k  | ~300s      | ~150s        | ~450s (7.5 min) |
| 1000×230k ⚠️ | ~1200s   | ~600s        | ~1800s (30 min) |

**Subtotal:** ~3,330 seconds (~55 minutes)

### Why CV is slow:
- 5 folds × 10 alpha values = 50 model fits per condition
- Each fit involves SVD decomposition on large matrices
- The 1000×230k condition alone = 230M elements per fold

---

## Running the Benchmarks

### Run Systematic Benchmarks
```bash
# From project root
uv run python benchmarks/benchmark_ridge_systematic.py
```

**Output files:**
- `results_ridge_systematic.csv` - Full results table
- Console output with progress and summary tables

**What you'll see:**
1. System information (CPU, GPU, versions)
2. Progress through 8 conditions with timing
3. Summary table with speedups
4. Full results table

### Optional: Faster Version

If you want to run a quicker version (~20 minutes), you could modify the script to:

```python
# In benchmark_ridge_systematic.py

# Option 1: Reduce CV folds (5 → 3)
time_torch, mem_torch = benchmark_fit_only(X, y, backend_torch, cv=3)

# Option 2: Reduce alpha grid (10 → 5 values)
result = ridge_cv(X, y, alphas=np.logspace(-2, 2, 5), cv=5, backend=backend)

# Option 3: Test only one resolution
n_voxels_options = [(50000, "3mm_resolution")]  # Skip 230k
```

---

## Interpreting Results

### Key Metrics

**Speedup vs NumPy:**
- `speedup > 1.0`: GPU is faster
- `speedup < 1.0`: CPU is faster (GPU overhead dominates)
- `speedup ≈ 1.0`: Similar performance

**Memory (MB):**
- Positive values: Memory increased during operation
- Negative values: Memory decreased (garbage collection during run)
- Note: Memory measurements are approximate due to Python GC

### Expected Patterns

Based on previous benchmarks, we expect:

1. **Small problems (500×50k estimates)**: GPU overhead → NumPy faster
2. **Large problems (1000×230k)**: GPU parallelism → PyTorch faster
3. **CV conditions**: GPU advantage increases (more compute per transfer)
4. **MPS limitation**: Modest speedups (1.4-2.2x) due to SVD CPU fallback

### Using Results

The CSV output can be used to:
1. Update `docs/performance.md` with realistic scenarios
2. Inform `auto_select_backend()` heuristics
3. Guide users on when to use GPU vs CPU
4. Document performance on different hardware

---

## Hardware Notes

### Current System (Initial Benchmarks)
- **Platform:** macOS 23.6.0 (Apple Silicon, arm64)
- **GPU:** Apple Metal Performance Shaders (MPS)
- **Python:** 3.10.17
- **NumPy:** 2.2.6
- **PyTorch:** 2.9.0

### MPS Limitation
⚠️ **Important:** PyTorch's SVD operation is not yet optimized for Apple's MPS backend and falls back to CPU. This limits speedup to ~1.4-2.2x.

**On NVIDIA GPUs with CUDA**, expect much higher speedups:
- Small problems: 5-10x
- Large problems: 10-30x
- CV conditions: 20-50x

---

## Future Benchmarking Ideas

### Additional Dimensions to Test

1. **Multi-target regression**
   - Single y vs multiple targets (e.g., 10, 100 targets)
   - Common in encoding models

2. **Batch processing**
   - Single large fit vs many small fits
   - Searchlight scenario: 1000s of small ridge regressions

3. **Data types**
   - Float32 (current) vs float64 precision
   - Memory vs accuracy trade-offs

4. **Different algorithms**
   - Ridge SVD vs Ridge Cholesky
   - Compare with sklearn's RidgeCV

### Hardware Comparisons

Run benchmarks on:
- NVIDIA GPU with CUDA (expect 10-30x speedups)
- AMD GPU with ROCm
- Different CPU architectures (x86 vs ARM)
- High-memory vs standard systems

---

## Benchmark Methodology

### Data Generation
```python
X = np.random.randn(n_samples, n_features).astype(np.float32)
y = np.random.randn(n_samples).astype(np.float32)
```

- Random normal data (mean=0, std=1)
- Float32 precision (memory efficient)
- Ensures reproducible, realistic computational patterns
- Results generalizable to real fMRI data

### Timing
```python
start = time.perf_counter()
# ... operation ...
end = time.perf_counter()
time_seconds = end - start
```

- Wall-clock time (includes all overhead)
- Single-threaded execution
- Excludes data generation time

### Memory
```python
import psutil
process = psutil.Process()
mem_before = process.memory_info().rss / 1024 / 1024  # MB
# ... operation ...
mem_after = process.memory_info().rss / 1024 / 1024
memory_mb = mem_after - mem_before
```

- Process-level memory (RSS = Resident Set Size)
- Includes Python interpreter overhead
- Approximate due to garbage collection

---

## Troubleshooting

### Benchmark Runs Out of Memory

**Solution 1:** Reduce problem size
```python
n_voxels_options = [(50000, "3mm_resolution")]  # Skip 230k
```

**Solution 2:** Skip GPU backend
```python
if est_style == "fit_only" and n_voxels > 100000:
    print("Skipping GPU for memory reasons")
    continue
```

### Benchmark Takes Too Long

**Solution:** Reduce CV complexity
```python
# Fewer folds
benchmark_fit_only(X, y, backend, cv=3)

# Fewer alpha values
ridge_cv(X, y, alphas=np.logspace(-1, 1, 5), cv=5, backend=backend)
```

### GPU Not Detected

Check PyTorch installation:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

---

## Summary

The systematic benchmarks provide realistic performance data for:
- **Task fMRI** (500 timepoints)
- **Naturalistic fMRI** (1000 timepoints)
- **Standard resolution** (50k voxels)
- **High resolution** (230k voxels)
- **Coefficient estimation** (no CV)
- **Prediction modeling** (5-fold CV)

This covers the vast majority of real-world neuroimaging ridge regression use cases and will provide actionable guidance for when to use CPU vs GPU backends.

---

**Last Updated:** 2025-10-28
**Benchmark Version:** Systematic v1.0
