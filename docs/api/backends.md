# `nltools.backends`

**Backend Abstraction for CPU/GPU Operations**

Provides transparent CPU/GPU acceleration for ridge regression and other algorithms.
Supports NumPy (CPU), PyTorch (CUDA/MPS/CPU), and automatic backend selection.

```{eval-rst}
.. automodule:: nltools.backends
    :members:
    :undoc-members:
    :show-inheritance:
```

## Quick Start

```python
from nltools.backends import Backend, check_gpu_available

# Check GPU availability
available, info = check_gpu_available()
print(f"GPU available: {available}")
print(f"Device: {info['device']}")

# Use backends
backend_cpu = Backend('numpy')
backend_gpu = Backend('torch')  # Auto-detects cuda/mps
backend_auto = Backend('auto')  # Smart selection
```

## Backend Selection

The `Backend` class provides three modes:

### NumPy Backend (CPU-only)
```python
from nltools.backends import Backend

backend = Backend('numpy')
# Always uses CPU with NumPy arrays
# Best for small to medium problems
```

### PyTorch Backend (GPU-accelerated)
```python
backend = Backend('torch')
# Automatically detects best device:
#   - CUDA (NVIDIA GPUs)
#   - MPS (Apple Silicon)
#   - CPU (fallback)
```

### Auto Selection (Recommended)
```python
backend = Backend('auto')
# Tries PyTorch if available, falls back to NumPy
# Use auto_select_backend() for problem-size heuristics
```

## Advanced: Problem-Size Heuristics

For optimal performance, use `auto_select_backend()` which selects based on problem size:

```python
from nltools.backends import auto_select_backend

# Automatically choose backend based on data dimensions
n_samples, n_features, cv_folds = 300, 100000, 5
backend = auto_select_backend(n_samples, n_features, cv=cv_folds)

print(f"Selected backend: {backend.name}")
# For large problems with CV, likely selects GPU if available
```

**Selection criteria:**
- **Small problems** (< 10M elements): Use NumPy (GPU overhead not worth it)
- **Large problems** (> 30M elements): Use GPU if available
- **Cross-validation**: Prefer GPU even for medium problems

## Usage in Algorithms

The backend system is designed for seamless integration with nltools algorithms:

```python
from nltools.algorithms.ridge import ridge_svd
import numpy as np

X = np.random.randn(300, 50000)
y = np.random.randn(300)

# Explicit backend selection
coef_cpu = ridge_svd(X, y, backend='numpy')
coef_gpu = ridge_svd(X, y, backend='torch')

# Automatic selection
coef_auto = ridge_svd(X, y, backend='auto')
```

## Performance Considerations

### When to Use GPU Acceleration

**Use GPU (`backend='torch'`) when:**
- Problem size > 30M elements (e.g., 300 samples × 100k features)
- Running cross-validation (multiplies effective problem size)
- Fitting many models in a loop

**Use CPU (`backend='numpy'`) when:**
- Problem size < 10M elements
- GPU not available
- Prototyping on small datasets

### Memory Management

Both backends use float32 precision for memory efficiency:

```python
# Data automatically converted to float32
X_float64 = np.random.randn(100, 50000)  # float64
backend = Backend('torch')
X_device = backend.to_device(X_float64)  # Converted to float32
```

### Important Limitations

**MPS (Apple Silicon) Limitation:**
- PyTorch's SVD is not fully optimized for MPS devices
- Falls back to CPU for SVD operations
- You may see warnings about CPU fallback
- Performance may not exceed NumPy on Apple Silicon

**For Apple Silicon users:** NumPy with Accelerate framework may be faster than PyTorch until MPS SVD support improves.

## API Reference

See the full API documentation above for details on:
- `Backend` class and methods
- `check_gpu_available()` function
- `auto_select_backend()` function

## See Also

- [Performance Guide](../performance.md) - Detailed benchmarks and recommendations
- {ref}`Ridge Regression <ridge-regression>` - Usage in algorithms
- {doc}`algorithms` - Complete algorithms documentation
