# Benchmarking Guide

## Overview

This directory contains systematic benchmarks for ridge regression and inference algorithm performance across realistic neuroimaging workflows. The benchmarks compare CPU (NumPy), CPU-parallel (joblib), and GPU (PyTorch) backends across different problem sizes and use cases.

---

## Ridge Regression Benchmarking

## Benchmark Script

### `benchmarking.py`
**Purpose:** Systematic ridge regression benchmarks with flexible CLI interface
**Status:** Ready to run
**Features:**
- Configurable problem sizes and CV parameters
- Dry-run mode for time estimation
- Live progress tracking with tqdm
- CPU and GPU backend comparison
**Results:** Generates `results_ridge_systematic.csv`

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
uv run python benchmarks/benchmarking.py

# Or with custom configuration
uv run python benchmarks/benchmarking.py -n 500 -v 50000 -e estimates

# Dry-run to see time estimates
uv run python benchmarks/benchmarking.py --dry-run
```

**Output files:**
- `results_ridge_systematic.csv` - Full results table
- Console output with progress and summary tables

**What you'll see:**
1. System information (CPU, GPU, versions)
2. Progress through 8 conditions with timing
3. Summary table with speedups
4. Full results table

### Faster Versions

Use CLI arguments to run quicker benchmarks:

```bash
# Fast: Estimates only (~2 minutes)
uv run python benchmarks/benchmarking.py -e estimates

# Medium: Reduce CV complexity (~20 minutes)
uv run python benchmarks/benchmarking.py --cv-folds 3 --cv-alphas 5

# Test single problem size
uv run python benchmarks/benchmarking.py -n 500 -v 50000

# See all options
uv run python benchmarks/benchmarking.py --help
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

## Inference Algorithm Benchmarking

### Benchmark Script

### `inference_benchmarking.py`
**Purpose:** Systematic inference algorithm benchmarks with flexible CLI interface  
**Status:** Ready to run  
**Features:**
- Configurable algorithms, problem sizes, and permutation counts
- Device-grouped execution (CPU Single, CPU Parallel, GPU)
- Real-time progress tracking with remaining benchmark counts
- Automatic visualization with seaborn stripplot
- Unique timestamped output files
- Comprehensive summary tables with 4 decimal precision

**Results:** 
- Generates `results_inference_systematic_YYYYMMDD_HHMMSS.csv`
- Generates `results_inference_systematic_YYYYMMDD_HHMMSS.png` (if seaborn/matplotlib available)

---

### Systematic Benchmark Design

### Benchmark Grid

**Algorithms:**
1. **One Sample Permutation Test** (`one_sample`)
   - Tests whether mean differs from zero
   - Supports: CPU Single (parallel=None), CPU Parallel (parallel='cpu'), GPU (parallel='gpu')

2. **Two Sample Permutation Test** (`two_sample`)
   - Tests difference between two groups (equal group sizes from `--n-samples`)
   - Supports: CPU Single, CPU Parallel, GPU

3. **Correlation Permutation Test** (`correlation`)
   - Tests correlation between two datasets
   - Supports: CPU Single, CPU Parallel, GPU

4. **Timeseries Correlation Permutation Test** (`timeseries_correlation`)
   - Two methods: `circle_shift` and `phase_randomize`
   - Tests correlation with temporal structure preservation
   - Supports: CPU Parallel, GPU (no CPU Single baseline)

5. **Matrix Permutation Test** (`matrix`)
   - Mantel test for correlation between matrices
   - Matrix size controlled by `--n-samples` flag
   - Supports: CPU Single, CPU Parallel

6. **ISC Permutation Test** (`isc`)
   - Intersubject Correlation with bootstrap resampling
   - Supports: CPU Parallel, GPU

7. **ISC Group Permutation Test** (`isc_group`)
   - Group-level ISC difference testing
   - Supports: CPU Single, CPU Parallel

**Dimensions:**
- **Sample sizes** (`--n-samples`): Comma-separated integers (default: `25`)
- **Feature/voxel counts** (`--n-features`): Comma-separated integers (default: `100`)
- **Permutation counts** (`--n-permute`): Comma-separated integers (default: `5000`)
- **Timepoints** (`--n-timepoints`): Comma-separated integers for ISC tests (default: `100,500`)

**Backend Options:**
- **CPU Single** (`parallel=None`): Single-threaded NumPy baseline
- **CPU Parallel** (`parallel='cpu'`): Multi-threaded joblib (uses all CPU cores)
- **GPU** (`parallel='gpu'`): PyTorch GPU acceleration (when available)

---

### Running the Benchmarks

### Basic Usage

```bash
# From project root
uv run python benchmarks/inference_benchmarking.py

# Dry-run to see benchmark plan
uv run python benchmarks/inference_benchmarking.py --dry-run

# Single algorithm, small problems
uv run python benchmarks/inference_benchmarking.py --algorithm one_sample --n-features "1,100"

# Custom configuration
uv run python benchmarks/inference_benchmarking.py \
    --algorithm one_sample,two_sample,correlation \
    --n-samples "50,200" \
    --n-features "100,1000" \
    --n-permute "1000,5000"

# ISC tests with custom timepoints
uv run python benchmarks/inference_benchmarking.py \
    --algorithm isc,isc_group \
    --n-timepoints "100,500,1000"

# Quick test (skip large problems)
uv run python benchmarks/inference_benchmarking.py --quick

# Skip GPU (CPU only)
uv run python benchmarks/inference_benchmarking.py --no-gpu

# Minimal output
uv run python benchmarks/inference_benchmarking.py --quiet
```

### Output Structure

The script groups benchmarks by device and runs algorithms sequentially within each device:

1. **Device 1: CPU Single (parallel=None)**
   - Runs all algorithms for CPU single-threaded baseline
   - Shows device summary with average times/memory

2. **Device 2: CPU Parallel (parallel='cpu', n_jobs=N)**
   - Runs all algorithms with CPU parallelization
   - Shows device summary and speedups vs CPU Single

3. **Device 3: GPU (parallel='gpu')** (if available)
   - Runs all algorithms with GPU acceleration
   - Shows device summary and speedups vs both baselines

**Progress Tracking:**
- Shows current test number and total tests: `[X/Total] Algorithm Name: parameters`
- Displays overall progress percentage and remaining count
- Previous device results remain visible while next device runs

**Output Files:**
- `results_inference_systematic_YYYYMMDD_HHMMSS.csv` - Full results table
- `results_inference_systematic_YYYYMMDD_HHMMSS.png` - Visualization (if seaborn/matplotlib available)

---

### Interpreting Results

### Key Metrics

**Speedup Calculations:**
- **speedup_vs_numpy**: CPU Parallel or GPU speed relative to CPU Single baseline
  - `speedup > 1.0`: Faster than baseline
  - `speedup < 1.0`: Slower than baseline (overhead dominates)
  - `speedup ≈ 1.0`: Similar performance
- **speedup_vs_cpu_parallel**: GPU speed relative to CPU Parallel baseline
  - Shows GPU advantage over multi-threaded CPU

**Time (seconds):**
- Wall-clock time measured with `time.perf_counter()`
- Includes all overhead (data transfer, parallelization overhead, etc.)
- Displayed with 4 decimal precision

**Memory (MB):**
- Process memory delta (RSS) measured with `psutil`
- Positive values: Memory increased during operation
- Negative values: Memory decreased (garbage collection)
- Displayed with 4 decimal precision

**Precision:**
- All numeric values displayed with 4 decimal places
- Missing baselines show "N/A" instead of NaN
- Summary tables filter out NaN values before averaging

### Expected Patterns

**CPU Single vs CPU Parallel:**
- Small problems: CPU Parallel overhead may dominate → similar or slower
- Large problems: CPU Parallel provides 4-8× speedup (typical for 8-core machines)
- Permutation-heavy algorithms benefit most from parallelization

**GPU Acceleration:**
- Small problems: GPU overhead dominates → CPU Single/Parallel often faster
- Large problems: GPU provides 10-30× speedup for voxel-wise computations
- Best for: ISC (voxel-wise), correlation (large feature counts), one/two-sample (large permutations)

**Algorithm-Specific Notes:**
- **Timeseries Correlation**: No CPU Single baseline (uses CPU Parallel as baseline)
- **Matrix**: No GPU support (CPU Single and CPU Parallel only)
- **ISC**: No CPU Single baseline (uses CPU Parallel as baseline)
- **ISC Group**: Supports CPU Single and CPU Parallel

---

### Visualization

The script automatically generates a seaborn stripplot visualization showing:
- Individual benchmark runs as points
- Conditional means as diamonds
- Backend comparison (CPU Single, CPU Parallel, GPU)
- Automatic log scale if time range exceeds 100×

The visualization is saved alongside the CSV with matching timestamp.

---

### Troubleshooting

### Benchmark Takes Too Long

**Solution 1:** Reduce permutation count
```bash
uv run python benchmarks/inference_benchmarking.py --n-permute "1000"
```

**Solution 2:** Reduce problem sizes
```bash
uv run python benchmarks/inference_benchmarking.py --n-samples "50" --n-features "100"
```

**Solution 3:** Use quick mode
```bash
uv run python benchmarks/inference_benchmarking.py --quick
```

**Solution 4:** Skip GPU
```bash
uv run python benchmarks/inference_benchmarking.py --no-gpu
```

### GPU Not Detected

Check GPU availability:
```python
from nltools.backends import check_gpu_available
available, info = check_gpu_available()
print(f"GPU Available: {available}")
if available:
    print(f"Device: {info['device']}")
    print(f"Device Name: {info['device_name']}")
```

### Memory Issues

**Solution:** Reduce problem sizes or permutation counts
```bash
uv run python benchmarks/inference_benchmarking.py \
    --n-features "100" \
    --n-permute "1000"
```

---

### Using Results

The CSV output can be used to:
1. Compare backend performance across algorithms
2. Identify when GPU acceleration is beneficial
3. Guide users on `parallel` parameter selection
4. Update performance documentation
5. Inform algorithm development priorities

The visualization provides quick visual comparison of backend performance across problem sizes.

---

**Last Updated:** 2025-11-02  
**Benchmark Version:** Inference Systematic v1.0
