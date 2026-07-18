# Performance Guide

This guide provides detailed performance benchmarks and recommendations for choosing between CPU (NumPy) and GPU (PyTorch) backends in nltools algorithms.

:::{note} Benchmark provenance
The numbers under [Benchmark Results](#benchmark-results) are generated from a
committed artifact (`benchmarks/results/*.parquet` + an `.env.json` recording the
host, Python, NumPy, and PyTorch versions) by `uv run python -m benchmarks.build_docs`.
They are **not** re-run on each doc build, so treat them as indicative rather than
current — absolute timings drift with hardware and library versions. Regenerate for
your own setup with `uv run python -m benchmarks.run` (see `benchmarks/benchmarking-guide.md`).
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
- **Recommendation**: Use `device='auto'` for intelligent selection
- **How it works**: Evaluates problem size and chooses optimal backend
- **Fallback**: Automatically uses NumPy if PyTorch unavailable

---

(benchmark-results)=
## Benchmark Results

:::{note} Scale & scope
These reference numbers are from an Apple-Silicon **MPS** run at whole-brain-ish
3mm scale (~20k voxels; `BrainCollection` over 20–50 on-disk subjects). At this
size ridge is dominated by parallel-pool setup, so CPU≈MPS — the large GPU wins
land at 2mm (~230k voxels) and on CUDA, which is a separate run on a GPU host
(the harness produces one artifact per host). The standout here is the
`BrainCollection` **memory** story and CPU parallel scaling, below.
:::

<!-- BENCH:START -->
:::{note} Auto-generated
This block is generated from `benchmarks/results/*.parquet` by `uv run python -m benchmarks.build_docs`. Do not edit by hand.
:::

**Host:** Eshin-M3-Air  
**Platform:** macOS-15.7.4-arm64-arm-64bit  
**Python:** 3.11.14  
**NumPy:** 2.4.4  
**PyTorch:** 2.11.0  
**nltools:** 0.5.1  
**GPU:** MPS

### GPU speedup

#### ridge — GPU speedup (mps)

| Condition | CPU | MPS | Speedup |
|---|--:|--:|--:|
| ridge_cv[1000x20000f100] | 15.15 s | 14.71 s | **1.03×** |
| ridge_cv[500x20000f50] | 12.96 s | 12.89 s | **1.01×** |

#### inference — GPU speedup (mps)

| Condition | CPU | MPS | Speedup |
|---|--:|--:|--:|
| correlation[perm=1000] | 76.7 ms | 68.3 ms | **1.12×** |
| correlation[perm=3000] | 110.6 ms | 193.6 ms | **0.57×** |
| one_sample[perm=1000] | 310.9 ms | 179.3 ms | **1.73×** |
| one_sample[perm=3000] | 576.3 ms | 438.5 ms | **1.31×** |
| two_sample[perm=1000] | 264.9 ms | 173.3 ms | **1.53×** |
| two_sample[perm=3000] | 397.9 ms | 501.5 ms | **0.79×** |

### Memory scaling

#### collection — peak RSS: lazy vs in-memory `.mean()`

| N subjects | lazy | in-memory |
|---:|--:|--:|
| 20 | 0.0 MB | 29.9 MB |
| 50 | 0.0 MB | 103.3 MB |

### Full results

#### ridge

| Condition | Device | Time | Peak RSS | GPU mem |
|---|---|--:|--:|--:|
| BrainData.fit[ridge,200x20000] | cpu | 239.0 ms | 0.1 MB | - |
| ridge_cv[1000x20000f100] | cpu | 15.15 s | 0.0 MB | - |
| ridge_cv[1000x20000f100] | mps | 14.71 s | 0.2 MB | - |
| ridge_cv[500x20000f50] | cpu | 12.96 s | 0.0 MB | - |
| ridge_cv[500x20000f50] | mps | 12.89 s | 0.1 MB | - |

#### predict

| Condition | Device | Time | Peak RSS | GPU mem |
|---|---|--:|--:|--:|
| roi[50parcels] | cpu | 1.27 s | 0.5 MB | - |
| searchlight[60x400] | cpu | 452.0 ms | 0.2 MB | - |
| whole_brain[200x20000] | cpu | 630.2 ms | 0.0 MB | - |

#### inference

| Condition | Device | Time | Peak RSS | GPU mem |
|---|---|--:|--:|--:|
| correlation[perm=1000] | cpu | 76.7 ms | 0.0 MB | - |
| correlation[perm=1000] | mps | 68.3 ms | 0.0 MB | - |
| correlation[perm=3000] | cpu | 110.6 ms | 0.0 MB | - |
| correlation[perm=3000] | mps | 193.6 ms | 0.2 MB | - |
| one_sample[perm=1000] | cpu | 310.9 ms | 0.6 MB | - |
| one_sample[perm=1000] | mps | 179.3 ms | 0.0 MB | - |
| one_sample[perm=3000] | cpu | 576.3 ms | 16.6 MB | - |
| one_sample[perm=3000] | mps | 438.5 ms | 0.0 MB | - |
| two_sample[perm=1000] | cpu | 264.9 ms | 1.5 MB | - |
| two_sample[perm=1000] | mps | 173.3 ms | 1.9 MB | - |
| two_sample[perm=3000] | cpu | 397.9 ms | 10.9 MB | - |
| two_sample[perm=3000] | mps | 501.5 ms | 2.9 MB | - |

#### collection

| Condition | Device | Time | Peak RSS | GPU mem |
|---|---|--:|--:|--:|
| apply[standardize,N=20,n_jobs=-1] | cpu | 520.4 ms | 36.0 MB | - |
| apply[standardize,N=20,n_jobs=1] | cpu | 1.72 s | 12.9 MB | - |
| apply[standardize,N=50,n_jobs=-1] | cpu | 1.30 s | 28.1 MB | - |
| apply[standardize,N=50,n_jobs=1] | cpu | 4.29 s | 69.5 MB | - |
| mean[in_memory,N=20] | cpu | 2.84 s | 29.9 MB | - |
| mean[in_memory,N=50] | cpu | 7.13 s | 103.3 MB | - |
| mean[lazy,N=20] | cpu | 2.86 s | 0.0 MB | - |
| mean[lazy,N=50] | cpu | 7.19 s | 0.0 MB | - |
<!-- BENCH:END -->

---

## Recommendations

### When to Use Each Backend

Set the compute device on the public API with `device=` (`'cpu'` / `'gpu'` /
`'auto'`) — e.g. `Ridge(device='gpu')`, `brain.fit(model='ridge', device='auto')`,
`brain.bootstrap(stat='weights', device='gpu')`.

**Use CPU (`device='cpu'`, NumPy) when:**
- Problem size < 10 million elements (e.g., 100 samples × 100k features)
- Prototyping or exploratory analysis
- GPU not available
- Running on shared systems where GPU access is limited

**Use GPU (`device='gpu'`, PyTorch) when:**
- Problem size > 30 million elements
- Running cross-validation (especially 5-fold or more)
- Fitting many models in a loop (e.g., searchlight)
- Production workflows where speed matters

**Use Auto (`device='auto'`) when:**
- Unsure about problem size
- Want code to work optimally across systems
- Developing reusable analysis scripts

### Problem Size Thresholds

Based on our benchmarks:

```python
# Effective problem size = n_samples × n_features × cv_folds
problem_size = n_samples * n_features * cv_folds

if problem_size < 10_000_000:
    # Use CPU (GPU overhead not worth it)
    device = 'cpu'
elif problem_size > 30_000_000:
    # Use GPU if available (significant speedup)
    device = 'gpu'
else:
    # Medium range: auto-select based on GPU availability
    device = 'auto'
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
- **Fallback**: Use `device='cpu'` explicitly

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
- Use `device='cpu'` explicitly on Apple Silicon
- NumPy with Accelerate is often faster than MPS for ridge regression
- Wait for future PyTorch versions with improved MPS SVD support

#### Out of GPU Memory

**Symptom**: `RuntimeError: CUDA out of memory` or similar

**Solutions:**
1. Reduce problem size (subset voxels, downsample data)
2. Use the CPU device: `device='cpu'`
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

# Time NumPy (algorithm-layer solvers use the internal `parallel=` name)
start = time.time()
coef_np = ridge_svd(X, y, parallel='cpu')
time_np = time.time() - start

# Time PyTorch
start = time.time()
coef_torch = ridge_svd(X, y, parallel='gpu')
time_torch = time.time() - start

print(f"NumPy: {time_np:.3f}s")
print(f"PyTorch: {time_torch:.3f}s")
print(f"Speedup: {time_np/time_torch:.1f}x")
```

### 2. Use Auto-Selection for Portability
```python
# Good: Works optimally everywhere
result = ridge_cv(X, y, parallel='auto')

# Less portable: Assumes GPU available
result = ridge_cv(X, y, parallel='gpu')
```

### 3. Batch GPU Operations
```python
# Good: Minimize host-device transfers
backend = Backend('torch')
X_device = backend.to_device(X)
results = [ridge_svd(X_device, y, parallel=backend) for y in y_list]

# Less efficient: Transfer on every call
results = [ridge_svd(X, y, parallel='gpu') for y in y_list]
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
- [Ridge Regression](api/algorithms.md#algorithms-ridge) - Algorithm details
- [Algorithms](api/algorithms.md) - Complete algorithm reference

---

**Note on MPS Performance**: These benchmarks run on Apple Silicon with MPS backend. PyTorch's SVD operation is not yet fully optimized for MPS and falls back to CPU, limiting speedup compared to CUDA GPUs. On NVIDIA GPUs with CUDA, expect significantly higher speedups (10-30x for large problems).

*To regenerate benchmarks on your system:* `uv run python benchmarks/benchmarking.py`
