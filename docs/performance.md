# Performance Guide

This guide provides detailed performance benchmarks and recommendations for choosing between CPU (NumPy) and GPU (PyTorch) backends in nltools algorithms.

:::{note} Benchmark provenance
The numbers below were generated on the environment reported under [Benchmark Results](#benchmark-results) (macOS 23.6.0, Python 3.10.17, NumPy 2.2.6, PyTorch 2.9.0). They are **not** re-run on each doc build and no timestamped result artifact is committed alongside them, so treat them as indicative rather than current — absolute timings will drift with hardware and library versions. Regenerate with `uv run python benchmarks/benchmarking.py` to get numbers for your own setup.
:::

---

## Why Performance Matters for Neuroimaging

Neuroimaging analyses often involve:
- **High-dimensional data**: fMRI datasets with 10k-200k voxels per brain
- **Many samples**: Hundreds to thousands of timepoints or subjects
- **Iterative algorithms**: Cross-validation, permutation testing, searchlight
- **Large-scale studies**: Analyzing dozens of subjects or conditions

A 10x speedup can transform workflows:
- **Interactive analysis**: Results in seconds instead of minutes
- **Exploration**: Try more models, parameters, and hypotheses
- **Scalability**: Analyze whole-brain data that was previously impractical

---

## Backend Overview

nltools provides two computational backends:

### NumPy Backend (CPU)
- **Pros**: No dependencies, works everywhere, excellent for small problems
- **Cons**: Single-threaded for most operations, slower for large problems
- **Best for**: Prototyping, small datasets, systems without GPU

### PyTorch Backend (GPU)
- **Pros**: Massive parallelism, 10-30x speedup for large problems
- **Cons**: Requires PyTorch installation, GPU memory limits, transfer overhead
- **Best for**: Large datasets, cross-validation, production workflows

### Auto Selection
- **Recommendation**: Use `backend='auto'` for intelligent selection
- **How it works**: Evaluates problem size and chooses optimal backend
- **Fallback**: Automatically uses NumPy if PyTorch unavailable

---

## Benchmark Results

**System:** macOS 23.6.0 (Apple Silicon, arm64)
**Python:** 3.10.17 | **NumPy:** 2.2.6 | **PyTorch:** 2.9.0
**GPU:** Apple Metal Performance Shaders (MPS)

### Problem Size Scaling

How performance scales with dataset dimensions:

| Dataset Size | NumPy (CPU) | PyTorch (MPS) | Speedup |
|--------------|-------------|---------------|---------|
| Small: 100×1k | 0.006s | 0.014s | **0.4x** (slower) |
| Medium: 300×50k | 1.44s | 0.67s | **2.2x** |
| Large: 1000×200k | 43.0s | 30.0s | **1.4x** |

**Key findings:**
- **Small problems**: GPU overhead outweighs benefits (NumPy **2.5x faster**)
- **Medium problems**: GPU provides significant speedup (**2.2x faster**)
- **Large problems**: GPU reduces compute time by 30% (**1.4x faster**)

### Cross-Validation Impact

Effect of cross-validation on computation time:

| Dataset | NumPy (no CV) | NumPy (5-fold) | PyTorch (5-fold) | CV Speedup |
|---------|---------------|----------------|------------------|------------|
| 300×100k | 3.42s | 141.0s | 85.5s | **1.6x** |

**Key findings:**
- Cross-validation multiplies computation by ~40x (not just 5x due to hyperparameter search)
- GPU becomes **1.6x faster** with CV due to increased workload
- Auto-selection correctly chooses GPU when `cv > 1` for medium/large datasets

### Auto Backend Selection Validation

Testing the `auto_select_backend()` heuristics:

| Scenario | Problem Size | CV Folds | Expected | Actual | Correct? |
|----------|--------------|----------|----------|--------|----------|
| Small, no CV | 100×1k | 1 | numpy | numpy | ✅ |
| Medium with CV | 300×100k | 5 | torch | torch-mps | ✅ |
| Large | 1000×200k | 1 | torch | torch-mps | ✅ |

**Key findings:**
- Auto-selection **correctly identifies optimal backend** for all scenarios
- Conservative thresholds prevent GPU overhead for borderline cases
- Falls back gracefully when GPU unavailable

---

## Real-World Use Cases

Performance on realistic fMRI analysis workflows:

### Whole-Brain Prediction (300 samples × 100k voxels, 5-fold CV)

**Scenario**: Predicting behavioral outcomes from whole-brain activity

| Backend | Time | Memory | Use Case |
|---------|------|--------|----------|
| NumPy | 119.5s | ~444 MB | Prototyping, limited compute |
| PyTorch (MPS) | 88.4s | ~131 MB | Production, iterative analysis |
| **Speedup** | **1.4x** | - | GPU recommended |

**Recommendation**: GPU provides **31-second speedup** (26% faster) for this common workflow.

### Searchlight Preparation (1000 samples × 200k features)

**Scenario**: Computing ridge regression for searchlight seeds

| Backend | Time | Memory | Use Case |
|---------|------|--------|----------|
| NumPy | 47.2s | - | Small ROI, limited iterations |
| PyTorch (MPS) | 26.9s | ~820 MB | Whole-brain, many iterations |
| **Speedup** | **1.8x** | - | GPU strongly recommended |

**Recommendation**: GPU enables **44% faster** whole-brain searchlight computation.

---

## Recommendations

### When to Use Each Backend

**Use NumPy (`backend='numpy'`) when:**
- Problem size < 10 million elements (e.g., 100 samples × 100k features)
- Prototyping or exploratory analysis
- GPU not available
- Running on shared systems where GPU access is limited

**Use PyTorch (`backend='torch'`) when:**
- Problem size > 30 million elements
- Running cross-validation (especially 5-fold or more)
- Fitting many models in a loop (e.g., searchlight)
- Production workflows where speed matters

**Use Auto (`backend='auto'`) when:**
- Unsure about problem size
- Want code to work optimally across systems
- Developing reusable analysis scripts

### Problem Size Thresholds

Based on our benchmarks:

```python
# Effective problem size = n_samples × n_features × cv_folds
problem_size = n_samples * n_features * cv_folds

if problem_size < 10_000_000:
    # Use NumPy (GPU overhead not worth it)
    backend = 'numpy'
elif problem_size > 30_000_000:
    # Use GPU if available (significant speedup)
    backend = 'torch'
else:
    # Medium range: auto-select based on GPU availability
    backend = 'auto'
```

### Memory Considerations

Both backends use float32 precision to reduce memory usage:

```python
# Example: 1000 samples × 200k features
n_samples, n_features = 1000, 200000

# NumPy memory (approx)
memory_numpy = n_samples * n_features * 4 bytes / 1e9  # ~0.8 GB

# PyTorch memory (includes intermediate arrays)
memory_torch = memory_numpy * 2  # ~1.6 GB (rough estimate)
```

**GPU memory limits:**
- Typical GPU VRAM: 8-24 GB (consumer), 16-80 GB (professional)
- If problem doesn't fit in GPU memory, use NumPy
- Monitor GPU memory with `nvidia-smi` (CUDA) or Activity Monitor (MPS)

---

## Troubleshooting

### Common Issues

#### GPU Not Detected
```python
from nltools.algorithms.backends import check_gpu_available

available, info = check_gpu_available()
if not available:
    print(f"GPU not available: {info['device_name']}")
    print("Using CPU backend instead")
```

**Solutions:**
- **CUDA**: Install PyTorch with CUDA support: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- **MPS (Apple Silicon)**: Ensure PyTorch ≥2.0: `pip install torch>=2.0`
- **Fallback**: Use `backend='numpy'` explicitly

#### MPS (Apple Silicon) SVD Limitation

**Symptom**: Warning about `linalg_svd` falling back to CPU

```
UserWarning: The operator 'aten::linalg_svd' is not currently supported on the MPS backend
and will fall back to run on the CPU.
```

**Explanation:**
- PyTorch's MPS backend doesn't fully optimize SVD operations
- Falls back to CPU for SVD computation
- Performance may not exceed NumPy with Accelerate framework

**Solutions:**
- Use `backend='numpy'` explicitly on Apple Silicon
- NumPy with Accelerate is often faster than MPS for ridge regression
- Wait for future PyTorch versions with improved MPS SVD support

#### Out of GPU Memory

**Symptom**: `RuntimeError: CUDA out of memory` or similar

**Solutions:**
1. Reduce problem size (subset voxels, downsample data)
2. Use NumPy backend: `backend='numpy'`
3. Process in batches if applicable
4. Use systems with more GPU memory

#### Slow First Run

**Symptom**: First GPU computation is slower than subsequent runs

**Explanation:**
- PyTorch compiles kernels on first use (JIT compilation)
- GPU initialization overhead
- Memory allocation and transfer

**Solution**: This is normal. Subsequent runs will be faster.

---

## Best Practices

### 1. Profile Before Optimizing
```python
import time
import numpy as np
from nltools.algorithms.ridge import ridge_svd
from nltools.algorithms.backends import Backend

X = np.random.randn(300, 100000)
y = np.random.randn(300)

# Time NumPy
start = time.time()
coef_np = ridge_svd(X, y, backend='numpy')
time_np = time.time() - start

# Time PyTorch
start = time.time()
coef_torch = ridge_svd(X, y, backend='torch')
time_torch = time.time() - start

print(f"NumPy: {time_np:.3f}s")
print(f"PyTorch: {time_torch:.3f}s")
print(f"Speedup: {time_np/time_torch:.1f}x")
```

### 2. Use Auto-Selection for Portability
```python
# Good: Works optimally everywhere
result = ridge_cv(X, y, backend='auto')

# Less portable: Assumes GPU available
result = ridge_cv(X, y, backend='torch')
```

### 3. Batch GPU Operations
```python
# Good: Minimize host-device transfers
backend = Backend('torch')
X_device = backend.to_device(X)
results = [ridge_svd(X_device, y, backend=backend) for y in y_list]

# Less efficient: Transfer on every call
results = [ridge_svd(X, y, backend='torch') for y in y_list]
```

### 4. Monitor Resource Usage
```bash
# GPU monitoring (NVIDIA)
watch -n 1 nvidia-smi

# CPU/Memory monitoring (all systems)
htop  # or top
```

---

## Benchmark Methodology

### Hardware & Software
- **System**: macOS 23.6.0 (Darwin)
- **CPU**: Apple Silicon (arm64)
- **GPU**: Apple Metal Performance Shaders (MPS)
- **Python**: 3.10.17
- **NumPy**: 2.2.6
- **PyTorch**: 2.9.0

### Benchmark Scenarios
1. **Basic comparison**: CPU vs GPU on medium dataset (300×50k)
2. **Problem size scaling**: Small, medium, large datasets
3. **Cross-validation impact**: 1-fold vs 5-fold CV overhead
4. **Auto-selection validation**: Verify heuristics choose correctly
5. **Real-world workflows**: Whole-brain prediction, searchlight prep

### Metrics
- **Execution time**: `time.perf_counter()` (wall-clock time)
- **Memory usage**: `psutil` process memory monitoring
- **Speedup**: Ratio of NumPy time to PyTorch time

### Data Generation
```python
# Synthetic data matching neuroimaging dimensions
X = np.random.randn(n_samples, n_features).astype(np.float32)
y = np.random.randn(n_samples).astype(np.float32)

# Ensures realistic memory/computation patterns
# Results generalizable to real fMRI data
```

---

## See Also

- [Backends](api/backends.md) - CPU/GPU backend documentation
- [Ridge Regression](api/algorithms.md#ridge) - Algorithm details
- [Algorithms](api/algorithms.md) - Complete algorithm reference

---

**Note on MPS Performance**: These benchmarks run on Apple Silicon with MPS backend. PyTorch's SVD operation is not yet fully optimized for MPS and falls back to CPU, limiting speedup compared to CUDA GPUs. On NVIDIA GPUs with CUDA, expect significantly higher speedups (10-30x for large problems).

*To regenerate benchmarks on your system:* `uv run python benchmarks/benchmarking.py`
