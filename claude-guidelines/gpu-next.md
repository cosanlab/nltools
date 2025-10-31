# GPU Bootstrap Implementation Plan

**Date**: 2025-10-31
**Status**: Planning (Post-Phase 4)
**Based on**: Himalaya library research (see `himalaya-gpu-resampling-research.md`)

---

## Executive Summary

This document outlines the implementation plan for GPU-accelerated bootstrap inference in nltools, building on the successful CPU implementations (Phases 1-4) and incorporating best practices from the himalaya library research.

**Current state**: CPU-parallelized bootstrap for simple methods and Ridge models (19 tests passing, 10-100× speedup via ridge_svd optimization)

**Goal**: Add GPU backend support with PyTorch for 10-100× additional speedup on large datasets

**Timeline estimate**: 4-6 weeks for full GPU support

---

## Architecture Overview

### Backend Abstraction Layer

Following himalaya's pattern, create a unified backend interface:

```
nltools/algorithms/inference/backends/
├── __init__.py           # set_backend(), get_backend(), Backend class
├── numpy.py              # NumPy backend (CPU, default)
├── torch_cpu.py          # PyTorch CPU backend
├── torch_cuda.py         # PyTorch GPU backend (CUDA)
└── cupy.py               # CuPy GPU backend (optional, future)
```

**Key operations to abstract**:
- Array creation: `zeros()`, `ones()`, `full()`, `asarray()`
- Device management: `to_gpu()`, `to_cpu()`, `get_device()`, `is_in_gpu()`
- Random generation: `randn()`, `rand()`, `randint()`
- Math operations: `mean()`, `std()`, `matmul()`, `svd()`
- Type checking: `is_array()`, `asarray_like()`

### Bootstrap GPU Pattern

Adapt himalaya's 3D batching (targets × alphas × folds) to our 2D needs (iterations × targets):

```python
def _bootstrap_gpu_batched(
    data: np.ndarray,
    method: str,
    n_samples: int = 5000,
    backend: Backend = "torch_cuda",
    data_in_cpu: bool = True,        # Himalaya's Y_in_cpu pattern
    n_iter_batch: int = 100,         # PRIMARY: Batch iterations
    n_targets_batch: int = None,     # SECONDARY: Batch targets
    max_gpu_memory_gb: float = 4.0,
    random_state: Optional[int] = None,
) -> dict:
    """
    Bootstrap using GPU with automatic batching.

    Key patterns from himalaya:
    - data_in_cpu: Keep data on CPU, batch to GPU (memory efficient)
    - Two-dimensional batching: iterations × targets
    - Explicit memory management: del + immediate CPU transfer
    - Pre-allocated results on CPU
    """
```

---

## Implementation Phases

### Phase 8: Backend Infrastructure (Week 1-2)

**Goal**: Create backend abstraction layer (no bootstrap logic yet)

#### Step 1: Core Backend Module

**File**: `nltools/algorithms/inference/backends/__init__.py`

```python
"""Backend abstraction for CPU/GPU operations."""

from typing import Union, Optional
import importlib

# Global backend state
_CURRENT_BACKEND = None


class Backend:
    """
    Abstract backend interface for array operations.

    Provides device-agnostic operations for CPU (NumPy) and GPU (PyTorch/CuPy).
    """

    def __init__(self, name: str):
        self.name = name
        self._backend_module = None
        self._initialize()

    def _initialize(self):
        """Load backend module."""
        if self.name == "numpy":
            from . import numpy as backend
        elif self.name == "torch":
            from . import torch_cpu as backend
        elif self.name == "torch_cuda":
            from . import torch_cuda as backend
        elif self.name == "cupy":
            from . import cupy as backend
        else:
            raise ValueError(f"Unknown backend: {self.name}")

        self._backend_module = backend

    # Delegate all operations to backend module
    def __getattr__(self, name):
        return getattr(self._backend_module, name)


def set_backend(backend: Union[str, Backend]) -> Backend:
    """
    Set global backend for inference operations.

    Parameters
    ----------
    backend : str or Backend
        Backend name: "numpy", "torch", "torch_cuda", "cupy", "auto"
        "auto" selects best available backend (GPU if available, else CPU)

    Returns
    -------
    Backend
        The active backend instance

    Examples
    --------
    >>> from nltools.algorithms.inference.backends import set_backend
    >>> backend = set_backend("torch_cuda")  # Use PyTorch GPU
    >>> # All subsequent operations use GPU
    """
    global _CURRENT_BACKEND

    if isinstance(backend, Backend):
        _CURRENT_BACKEND = backend
        return backend

    if backend == "auto":
        backend = auto_select_backend()

    _CURRENT_BACKEND = Backend(backend)
    return _CURRENT_BACKEND


def get_backend() -> Backend:
    """Get current backend (or default to NumPy)."""
    global _CURRENT_BACKEND

    if _CURRENT_BACKEND is None:
        _CURRENT_BACKEND = Backend("numpy")

    return _CURRENT_BACKEND


def auto_select_backend() -> str:
    """
    Automatically select best available backend.

    Priority:
    1. PyTorch CUDA (if available)
    2. CuPy (if available)
    3. PyTorch CPU
    4. NumPy
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "torch_cuda"
    except ImportError:
        pass

    try:
        import cupy as cp
        # Test if CuPy can access GPU
        cp.zeros(1)
        return "cupy"
    except (ImportError, Exception):
        pass

    try:
        import torch
        return "torch"
    except ImportError:
        pass

    return "numpy"
```

#### Step 2: NumPy Backend (Reference Implementation)

**File**: `nltools/algorithms/inference/backends/numpy.py`

```python
"""NumPy backend for CPU operations."""

import numpy as np
from typing import Optional, Tuple, Any

# Device management (no-ops for NumPy)
def to_gpu(array, device=None):
    """NumPy arrays are always on CPU."""
    return array

def to_cpu(array):
    """NumPy arrays are always on CPU."""
    return array

def to_numpy(array):
    """Already NumPy."""
    return np.asarray(array)

def get_device(array):
    """NumPy always uses CPU."""
    return "cpu"

def is_in_gpu(array):
    """NumPy arrays never on GPU."""
    return False

# Array creation
def zeros(shape, dtype="float32", device=None):
    """Create zero-filled array."""
    return np.zeros(shape, dtype=dtype)

def zeros_like(array, shape=None, dtype=None, device=None):
    """Create zero-filled array matching reference."""
    if shape is None:
        shape = array.shape
    if dtype is None:
        dtype = array.dtype
    return np.zeros(shape, dtype=dtype)

def ones(shape, dtype="float32", device=None):
    """Create ones-filled array."""
    return np.ones(shape, dtype=dtype)

def asarray(x, dtype=None, device=None):
    """Convert to NumPy array."""
    return np.asarray(x, dtype=dtype)

# Random generation
def randn(*args, random_state=None, **kwargs):
    """Generate random normal array."""
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        return rng.randn(*args)
    return np.random.randn(*args)

def rand(*args, random_state=None, **kwargs):
    """Generate random uniform [0,1) array."""
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        return rng.rand(*args)
    return np.random.rand(*args)

def randint(low, high, size, random_state=None):
    """Generate random integers."""
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        return rng.randint(low, high, size=size)
    return np.random.randint(low, high, size=size)

# Math operations
def mean(array, axis=None):
    """Compute mean."""
    return np.mean(array, axis=axis)

def std(array, axis=None, ddof=0):
    """Compute standard deviation."""
    return np.std(array, axis=axis, ddof=ddof)

def sum(array, axis=None):
    """Compute sum."""
    return np.sum(array, axis=axis)

def matmul(a, b):
    """Matrix multiplication."""
    return np.matmul(a, b)

def svd(X, full_matrices=False):
    """Singular value decomposition."""
    return np.linalg.svd(X, full_matrices=full_matrices)

# Type checking
def is_array(x):
    """Check if x is NumPy array."""
    return isinstance(x, np.ndarray)
```

#### Step 3: PyTorch CUDA Backend

**File**: `nltools/algorithms/inference/backends/torch_cuda.py`

```python
"""PyTorch CUDA backend for GPU operations."""

import numpy as np
import torch
from typing import Optional

# Check CUDA availability on import
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch CUDA backend requires CUDA-capable GPU")

# Device management
def to_gpu(array, device=None):
    """Move array to GPU."""
    if device is None:
        device = "cuda"

    if isinstance(array, torch.Tensor):
        return array.to(device)
    else:
        return torch.from_numpy(np.asarray(array)).to(device)

def to_cpu(array):
    """Move array to CPU."""
    if isinstance(array, torch.Tensor):
        return array.cpu()
    return array

def to_numpy(array):
    """Convert to NumPy array (on CPU)."""
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    return np.asarray(array)

def get_device(array):
    """Get device of array."""
    if isinstance(array, torch.Tensor):
        return array.device
    return "cpu"

def is_in_gpu(array):
    """Check if array is on GPU."""
    if isinstance(array, torch.Tensor):
        return array.is_cuda
    return False

# Array creation (default to GPU)
def zeros(shape, dtype="float32", device="cuda"):
    """Create zero-filled tensor on GPU."""
    dtype_map = {"float32": torch.float32, "float64": torch.float64, "int32": torch.int32}
    return torch.zeros(shape, dtype=dtype_map.get(dtype, torch.float32), device=device)

def zeros_like(array, shape=None, dtype=None, device=None):
    """Create zero-filled tensor matching reference."""
    if shape is None:
        shape = array.shape
    if device is None:
        device = array.device if isinstance(array, torch.Tensor) else "cuda"
    if dtype is None and isinstance(array, torch.Tensor):
        return torch.zeros(shape, dtype=array.dtype, device=device)
    return zeros(shape, dtype=dtype, device=device)

def ones(shape, dtype="float32", device="cuda"):
    """Create ones-filled tensor on GPU."""
    dtype_map = {"float32": torch.float32, "float64": torch.float64}
    return torch.ones(shape, dtype=dtype_map.get(dtype, torch.float32), device=device)

def asarray(x, dtype=None, device="cuda"):
    """Convert to PyTorch tensor on GPU."""
    if isinstance(x, torch.Tensor):
        tensor = x
    else:
        tensor = torch.from_numpy(np.asarray(x))

    if dtype is not None:
        dtype_map = {"float32": torch.float32, "float64": torch.float64}
        tensor = tensor.to(dtype=dtype_map.get(dtype, torch.float32))

    return tensor.to(device)

# Random generation (on GPU)
def randn(*args, random_state=None, device="cuda", **kwargs):
    """Generate random normal tensor on GPU."""
    if random_state is not None:
        torch.manual_seed(random_state)
    return torch.randn(*args, device=device, **kwargs)

def rand(*args, random_state=None, device="cuda", **kwargs):
    """Generate random uniform [0,1) tensor on GPU."""
    if random_state is not None:
        torch.manual_seed(random_state)
    return torch.rand(*args, device=device, **kwargs)

def randint(low, high, size, random_state=None, device="cuda"):
    """Generate random integers on GPU."""
    if random_state is not None:
        torch.manual_seed(random_state)
    return torch.randint(low, high, size, device=device)

# Math operations (GPU-accelerated)
def mean(array, axis=None):
    """Compute mean on GPU."""
    if axis is None:
        return torch.mean(array)
    return torch.mean(array, dim=axis)

def std(array, axis=None, ddof=0):
    """Compute standard deviation on GPU."""
    # PyTorch uses Bessel's correction by default (ddof=1)
    # We default to ddof=0 for consistency with NumPy
    unbiased = (ddof == 1)
    if axis is None:
        return torch.std(array, unbiased=unbiased)
    return torch.std(array, dim=axis, unbiased=unbiased)

def sum(array, axis=None):
    """Compute sum on GPU."""
    if axis is None:
        return torch.sum(array)
    return torch.sum(array, dim=axis)

def matmul(a, b):
    """Matrix multiplication on GPU."""
    return torch.matmul(a, b)

def svd(X, full_matrices=False):
    """Singular value decomposition on GPU."""
    U, s, Vt = torch.linalg.svd(X, full_matrices=full_matrices)
    return U, s, Vt

# Type checking
def is_array(x):
    """Check if x is PyTorch tensor."""
    return isinstance(x, torch.Tensor)
```

#### Step 4: Tests for Backend Abstraction

**File**: `nltools/tests/core/test_backends.py` (extend existing)

```python
"""Tests for backend abstraction layer."""

import pytest
import numpy as np
from nltools.algorithms.inference.backends import (
    set_backend, get_backend, auto_select_backend
)

class TestBackendSwitching:
    """Test backend selection and switching."""

    def test_set_backend_numpy(self):
        """Test setting NumPy backend."""
        backend = set_backend("numpy")
        assert backend.name == "numpy"
        assert get_backend().name == "numpy"

    def test_set_backend_auto(self):
        """Test auto backend selection."""
        backend = set_backend("auto")
        # Should select best available
        assert backend.name in ["numpy", "torch", "torch_cuda", "cupy"]

    @pytest.mark.skipif(not _torch_cuda_available(), reason="CUDA not available")
    def test_set_backend_torch_cuda(self):
        """Test setting PyTorch CUDA backend."""
        backend = set_backend("torch_cuda")
        assert backend.name == "torch_cuda"

    def test_backend_operations_numpy(self):
        """Test NumPy backend operations."""
        backend = set_backend("numpy")

        # Array creation
        x = backend.zeros((10, 5))
        assert x.shape == (10, 5)
        assert backend.is_array(x)

        # Device management (no-ops for NumPy)
        assert backend.get_device(x) == "cpu"
        assert not backend.is_in_gpu(x)

    @pytest.mark.skipif(not _torch_cuda_available(), reason="CUDA not available")
    def test_backend_operations_torch_cuda(self):
        """Test PyTorch CUDA backend operations."""
        backend = set_backend("torch_cuda")

        # Array creation (default to GPU)
        x = backend.zeros((10, 5))
        assert x.shape == (10, 5)
        assert backend.is_in_gpu(x)

        # Device transfer
        x_cpu = backend.to_cpu(x)
        assert not backend.is_in_gpu(x_cpu)

        x_numpy = backend.to_numpy(x)
        assert isinstance(x_numpy, np.ndarray)


def _torch_cuda_available():
    """Check if PyTorch with CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
```

**Deliverables for Phase 8**:
- ✅ Backend abstraction module created
- ✅ NumPy backend implemented (reference)
- ✅ PyTorch CUDA backend implemented
- ✅ Backend switching tests pass
- ✅ Documentation for backend usage

**Time estimate**: 1-2 weeks

---

### Phase 9: Simple Methods GPU Bootstrap (Week 3)

**Goal**: Implement GPU bootstrap for simple aggregation methods (mean, std, median)

#### Step 1: GPU Bootstrap Function

**File**: `nltools/algorithms/inference/bootstrap.py`

```python
def _bootstrap_simple_gpu_batched(
    data: np.ndarray,
    method: str,
    n_samples: int = 5000,
    backend: Union[str, Backend] = "torch_cuda",
    data_in_cpu: bool = True,
    n_iter_batch: int = 100,
    n_targets_batch: int = None,
    max_gpu_memory_gb: float = 4.0,
    random_state: Optional[int] = None,
    percentiles: Tuple[float, float] = (2.5, 97.5),
) -> Dict[str, np.ndarray]:
    """
    Bootstrap simple methods using GPU with automatic batching.

    Follows himalaya patterns:
    - data_in_cpu: Keep data on CPU, batch to GPU (memory efficient)
    - Two-dimensional batching: iterations × targets
    - Explicit memory management: del + immediate CPU transfer
    - Pre-allocated results on CPU

    Parameters
    ----------
    data : np.ndarray
        Data to bootstrap, shape (n_samples, n_features)
    method : str
        Aggregation method: 'mean', 'median', 'std', 'sum', 'min', 'max'
    n_samples : int, default=5000
        Number of bootstrap iterations
    backend : str or Backend, default="torch_cuda"
        Backend to use: "torch_cuda", "cupy", or Backend instance
    data_in_cpu : bool, default=True
        If True, keep data on CPU and batch to GPU (memory efficient)
        If False, transfer all data to GPU once (faster but memory intensive)
    n_iter_batch : int, default=100
        Number of bootstrap iterations per batch (primary batching dimension)
    n_targets_batch : int, optional
        Number of targets per batch (secondary batching dimension)
        If None, uses all targets (no target batching)
    max_gpu_memory_gb : float, default=4.0
        Maximum GPU memory to use (for automatic batch sizing)
    random_state : int, optional
        Random seed for reproducibility
    percentiles : tuple, default=(2.5, 97.5)
        Percentiles for confidence intervals

    Returns
    -------
    dict
        Dictionary with mean, std, Z, p, ci_lower, ci_upper, backend

    Examples
    --------
    >>> data = np.random.randn(100, 50)  # 100 samples, 50 features
    >>> result = _bootstrap_simple_gpu_batched(data, 'mean', backend='torch_cuda')
    >>> result['mean'].shape
    (50,)
    >>> result['backend']
    'torch_cuda'
    """
    from .backends import set_backend, get_backend
    from tqdm import tqdm

    # Set backend
    if isinstance(backend, str):
        backend = set_backend(backend)

    # Input validation
    data = np.asarray(data, dtype=np.float32)  # float32 for GPU efficiency
    single_feature = data.ndim == 1
    if single_feature:
        data = data[:, np.newaxis]

    n_obs, n_features = data.shape

    # Determine target batching
    if n_targets_batch is None:
        n_targets_batch = n_features  # No target batching

    # Pre-allocate results on CPU (himalaya pattern)
    results_cpu = np.zeros((n_samples, n_features), dtype=np.float32)

    # Transfer data to GPU once if data_in_cpu=False
    if not data_in_cpu:
        data_gpu = backend.asarray(data, device="cuda")

    # Batch across iterations (primary batching dimension)
    pbar = tqdm(total=n_samples, desc="GPU bootstrap iterations", unit="iter")

    for start_iter in range(0, n_samples, n_iter_batch):
        end_iter = min(start_iter + n_iter_batch, n_samples)
        current_iter_batch = end_iter - start_iter
        iter_slice = slice(start_iter, end_iter)

        # Generate bootstrap indices for this batch (on CPU)
        indices_batch = _generate_bootstrap_indices(
            n_obs, current_iter_batch, random_state=random_state
        )

        # Batch across targets (secondary batching dimension)
        for start_target in range(0, n_features, n_targets_batch):
            end_target = min(start_target + n_targets_batch, n_features)
            target_slice = slice(start_target, end_target)

            # Get data for this target batch
            if data_in_cpu:
                # Transfer only this target batch to GPU
                data_batch = backend.asarray(data[:, target_slice], device="cuda")
            else:
                # Data already on GPU, just slice
                data_batch = data_gpu[:, target_slice]

            # Transfer indices to GPU
            indices_batch_gpu = backend.asarray(indices_batch, device="cuda")

            # Vectorized resampling (across iteration batch)
            # Shape: (current_iter_batch, n_obs, n_targets_batch)
            # This is the key GPU operation - vectorized across iterations
            data_resampled = data_batch[indices_batch_gpu]

            # Compute statistic (vectorized across iteration batch)
            # Shape: (current_iter_batch, n_targets_batch)
            if method == 'mean':
                stats_gpu = backend.mean(data_resampled, axis=1)
            elif method == 'median':
                # PyTorch median returns (values, indices) tuple
                if hasattr(backend, 'median'):
                    stats_gpu = backend.median(data_resampled, axis=1)
                    if isinstance(stats_gpu, tuple):
                        stats_gpu = stats_gpu[0]  # Extract values
                else:
                    # Fallback: use quantile
                    stats_gpu = backend.quantile(data_resampled, 0.5, axis=1)
            elif method == 'std':
                stats_gpu = backend.std(data_resampled, axis=1, ddof=1)
            elif method == 'sum':
                stats_gpu = backend.sum(data_resampled, axis=1)
            elif method == 'min':
                stats_gpu = backend.min(data_resampled, axis=1)
            elif method == 'max':
                stats_gpu = backend.max(data_resampled, axis=1)
            else:
                raise ValueError(f"Unsupported method: {method}")

            # Transfer results to CPU immediately (himalaya pattern)
            stats_cpu = backend.to_numpy(stats_gpu)

            # Store in pre-allocated array
            results_cpu[iter_slice, target_slice] = stats_cpu

            # Explicit cleanup (himalaya pattern - critical for GPU)
            del indices_batch_gpu, data_resampled, stats_gpu, stats_cpu

            if data_in_cpu:
                del data_batch

        pbar.update(current_iter_batch)

    pbar.close()

    # Aggregate on CPU using OnlineBootstrapStats (memory efficient)
    stats_aggregator = OnlineBootstrapStats(
        shape=(n_features,) if not single_feature else (),
        save_samples=False,  # Don't store samples (already in results_cpu)
        percentiles=percentiles,
    )

    # Update with all samples
    for i in range(n_samples):
        sample = results_cpu[i]
        if single_feature:
            sample = sample.flatten()
        stats_aggregator.update(sample)

    # Get final results
    result = stats_aggregator.get_results()
    result['backend'] = backend.name

    # Cleanup
    if not data_in_cpu:
        del data_gpu
    del results_cpu

    return result
```

#### Step 2: Tests for GPU Bootstrap

**File**: `nltools/tests/core/test_bootstrap.py`

```python
@pytest.mark.gpu
@pytest.mark.skipif(not _torch_cuda_available(), reason="CUDA not available")
class TestBootstrapGPU:
    """Test suite for GPU bootstrap."""

    def test_bootstrap_gpu_mean_matches_cpu(self):
        """Test GPU bootstrap matches CPU results."""
        np.random.seed(42)
        data = np.random.randn(100, 50).astype(np.float32)

        # CPU version
        result_cpu = _bootstrap_simple_cpu_parallel(
            data, 'mean', n_samples=100, n_jobs=1, random_state=42
        )

        # GPU version
        result_gpu = _bootstrap_simple_gpu_batched(
            data, 'mean', n_samples=100, backend='torch_cuda',
            data_in_cpu=False, random_state=42
        )

        # Results should be very close (within numerical precision)
        np.testing.assert_allclose(result_cpu['mean'], result_gpu['mean'], rtol=1e-5)
        np.testing.assert_allclose(result_cpu['std'], result_gpu['std'], rtol=1e-5)

    def test_bootstrap_gpu_data_in_cpu(self):
        """Test data_in_cpu strategy."""
        np.random.seed(42)
        data = np.random.randn(100, 50).astype(np.float32)

        # With data_in_cpu=True (memory efficient)
        result = _bootstrap_simple_gpu_batched(
            data, 'mean', n_samples=100, backend='torch_cuda',
            data_in_cpu=True, random_state=42
        )

        assert 'mean' in result
        assert result['mean'].shape == (50,)

    def test_bootstrap_gpu_batching(self):
        """Test two-dimensional batching."""
        np.random.seed(42)
        data = np.random.randn(100, 1000).astype(np.float32)  # Large n_features

        # Batch both iterations and targets
        result = _bootstrap_simple_gpu_batched(
            data, 'mean', n_samples=500, backend='torch_cuda',
            n_iter_batch=50,      # Batch iterations
            n_targets_batch=200,  # Batch targets
            random_state=42
        )

        assert result['mean'].shape == (1000,)
        assert not np.any(np.isnan(result['mean']))
```

**Deliverables for Phase 9**:
- ✅ GPU bootstrap for simple methods implemented
- ✅ data_in_cpu strategy working
- ✅ Two-dimensional batching working
- ✅ GPU tests pass and match CPU results
- ✅ Memory efficiency verified

**Time estimate**: 1 week

---

### Phase 10: Ridge GPU Bootstrap (Week 4-5)

**Goal**: Extend GPU bootstrap to Ridge weights and predictions

#### Key Challenge: Ridge SVD on GPU

**Option 1**: Use PyTorch's `torch.linalg.svd()` on GPU
```python
# Convert ridge_svd to use backend
def ridge_svd_gpu(X, y, alpha, backend):
    """Ridge regression via SVD on GPU."""
    X_gpu = backend.asarray(X, device="cuda")
    y_gpu = backend.asarray(y, device="cuda")

    # SVD on GPU
    U, s, Vt = backend.svd(X_gpu, full_matrices=False)

    # Ridge solution
    d = s / (s**2 + alpha)
    weights = backend.matmul(Vt.T, backend.matmul(U.T, y_gpu) * d[:, None])

    return backend.to_numpy(weights)
```

**Option 2**: Keep ridge_svd on CPU, only batch transfers
```python
# Hybrid: SVD on CPU, batching on GPU
for iter_batch in batches:
    # Generate indices on GPU
    indices_gpu = ...

    # Resample on GPU
    X_resampled = X_gpu[indices_gpu]
    y_resampled = y_gpu[indices_gpu]

    # Transfer to CPU for ridge_svd
    X_cpu = backend.to_cpu(X_resampled)
    y_cpu = backend.to_cpu(y_resampled)

    # Compute weights on CPU (established, tested)
    weights_batch = [ridge_svd(X_cpu[i], y_cpu[i], alpha) for i in range(batch_size)]
```

**Recommendation**: Start with Option 2 (hybrid), benchmark Option 1 later.

**Deliverables for Phase 10**:
- ✅ GPU bootstrap for Ridge weights
- ✅ GPU bootstrap for Ridge predictions
- ✅ Benchmark GPU vs CPU speedup
- ✅ Document when GPU is worth it (data size thresholds)

**Time estimate**: 1-2 weeks

---

### Phase 11: Optimization & Polish (Week 6)

**Goal**: Performance tuning and user experience improvements

#### Automatic Batch Size Tuning

Adapt himalaya's manual `n_targets_batch` to automatic tuning:

```python
def _estimate_batch_sizes(data_shape, n_bootstrap, max_gpu_memory_gb=4.0):
    """
    Estimate optimal batch sizes for GPU bootstrap.

    Parameters
    ----------
    data_shape : tuple
        Shape of data (n_samples, n_features)
    n_bootstrap : int
        Number of bootstrap iterations
    max_gpu_memory_gb : float
        Available GPU memory

    Returns
    -------
    n_iter_batch : int
        Iterations per batch
    n_targets_batch : int
        Targets per batch
    """
    n_samples, n_features = data_shape
    bytes_per_float32 = 4

    # Memory for resampled data: (n_iter_batch, n_samples, n_targets_batch)
    # Conservative estimate: 3× actual (accounts for intermediate tensors)
    safety_factor = 3

    max_memory_bytes = max_gpu_memory_gb * 1e9

    # Try to fit all targets in one batch (common case)
    memory_per_iter = n_samples * n_features * bytes_per_float32 * safety_factor
    max_iter_batch = int(max_memory_bytes / memory_per_iter)

    if max_iter_batch >= n_bootstrap:
        # Can fit all iterations at once
        return n_bootstrap, n_features

    if max_iter_batch >= 10:
        # Can batch iterations, all targets at once
        n_iter_batch = min(max_iter_batch, 100)  # Cap at 100 for reasonable batching
        return n_iter_batch, n_features

    # Need to batch both dimensions
    n_iter_batch = 10  # Minimum reasonable batch
    memory_per_iter_target = n_samples * bytes_per_float32 * safety_factor
    max_targets_per_batch = int(max_memory_bytes / (n_iter_batch * memory_per_iter_target))
    n_targets_batch = max(100, min(max_targets_per_batch, n_features))

    return n_iter_batch, n_targets_batch
```

#### User-Facing API

Make GPU usage seamless:

```python
# Current CPU API (Phases 1-4)
from nltools.algorithms.inference.bootstrap import bootstrap_one_sample

result = bootstrap_one_sample(data, statistic='mean', n_samples=5000)

# Future GPU API (Phase 8+)
# Option 1: Explicit backend
result = bootstrap_one_sample(data, statistic='mean', n_samples=5000,
                               backend='torch_cuda')

# Option 2: Auto backend (detects GPU)
result = bootstrap_one_sample(data, statistic='mean', n_samples=5000,
                               backend='auto')  # Uses GPU if available

# Option 3: BrainData integration
brain.bootstrap('mean', n_samples=5000, backend='auto')
```

**Deliverables for Phase 11**:
- ✅ Automatic batch size estimation
- ✅ Seamless backend='auto' support
- ✅ Documentation and examples
- ✅ Performance benchmarks published

**Time estimate**: 1 week

---

## Testing Strategy

### Test Organization

```
nltools/tests/core/
├── test_backends.py         # Backend abstraction tests
├── test_bootstrap.py         # CPU bootstrap tests (existing)
└── test_bootstrap_gpu.py     # GPU bootstrap tests (new)
```

### GPU Test Markers

```python
# Phases 1-7: All tests run on CPU (no GPU required)
@pytest.mark.tier1
def test_bootstrap_simple_methods_all():
    """CPU test - always runs."""
    pass

# Phases 8+: GPU tests (skipped if CUDA unavailable)
@pytest.mark.gpu
@pytest.mark.skipif(not _torch_cuda_available(), reason="CUDA not available")
def test_bootstrap_gpu_mean_matches_cpu():
    """GPU test - only runs if CUDA available."""
    pass
```

### GPU Test Coverage

**Phase 8 tests** (5 tests):
- Backend switching (numpy, torch_cuda)
- Device management (to_gpu, to_cpu, to_numpy)
- Array operations on GPU
- Random generation on GPU
- Backend='auto' selection

**Phase 9 tests** (6 tests):
- GPU bootstrap matches CPU results
- data_in_cpu=True strategy works
- data_in_cpu=False strategy works
- Two-dimensional batching works
- Large dataset handling (>1GB)
- Memory efficiency verified

**Phase 10 tests** (4 tests):
- Ridge weights GPU vs CPU
- Ridge predict GPU vs CPU
- Speedup benchmark (GPU > CPU for large data)
- Memory usage acceptable

**Total GPU tests**: 15 tests (all marked with @pytest.mark.gpu)

---

## Performance Expectations

### Speedup Targets

**Simple methods (mean, std)**:
- Small data (<100MB): GPU ~1-2× slower than CPU (overhead dominates)
- Medium data (100MB-1GB): GPU ~5-10× faster than CPU
- Large data (>1GB): GPU ~10-50× faster than CPU

**Ridge methods (weights, predict)**:
- Small models (<10k params): GPU ~1× (no benefit)
- Medium models (10k-100k params): GPU ~10-20× faster
- Large models (>100k params): GPU ~50-100× faster

**Target**: GPU provides **10-100× speedup for realistic neuroimaging problems** (100k voxels, 1000-5000 bootstrap samples)

### When GPU is Worth It

**Always worth it**:
- n_voxels > 10,000 AND n_bootstrap > 1,000
- Ridge regression with >10k parameters
- Real-time/interactive applications

**Sometimes worth it**:
- n_voxels > 1,000 AND n_bootstrap > 5,000
- Medium-sized problems where waiting >1 minute

**Not worth it**:
- n_voxels < 1,000 OR n_bootstrap < 500
- CPU parallelization already fast enough (<10 seconds)
- No GPU available

**Recommendation**: Use `backend='auto'` to automatically choose best option.

---

## Documentation Plan

### User Documentation

**Add to `docs/` (or docstrings)**:

1. **GPU Bootstrap Guide** (`docs/gpu-bootstrap-guide.md`)
   - When to use GPU vs CPU
   - Installation requirements (PyTorch/CUDA)
   - Performance expectations
   - Memory considerations

2. **Backend API Reference** (`docs/backend-api.md`)
   - Available backends (numpy, torch_cuda, cupy)
   - Backend switching (`set_backend()`, `backend='auto'`)
   - Creating custom backends

3. **Examples** (`examples/bootstrap_gpu_examples.py`)
   - Simple methods on GPU
   - Ridge weights on GPU
   - Comparison CPU vs GPU
   - Memory-efficient strategies

### Migration Guide Updates

Add to `docs/migration-guide.md`:

```markdown
### GPU Bootstrap Support (v0.7.0)

**New Features**:
- GPU-accelerated bootstrap via PyTorch CUDA
- Backend abstraction: 'numpy', 'torch_cuda', 'auto'
- 10-100× speedup for large datasets
- Memory-efficient batching strategies

**Example**:
```python
# Automatic GPU detection
result = brain.bootstrap('mean', n_samples=5000, backend='auto')

# Explicit GPU
result = brain.bootstrap('mean', n_samples=5000, backend='torch_cuda')

# Memory-efficient for large data
result = brain.bootstrap('mean', n_samples=5000, backend='torch_cuda',
                         data_in_cpu=True)  # Keep data on CPU, batch to GPU
```

**Requirements**:
- PyTorch 1.9+ with CUDA support
- CUDA-capable GPU (compute capability 3.5+)
- Recommended: 4GB+ GPU memory
```

---

## Dependencies

### New Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
gpu = [
    "torch>=1.9.0",  # PyTorch with CUDA support
]

# Alternative GPU backend
gpu-cupy = [
    "cupy-cuda11x>=10.0.0",  # CuPy for CUDA 11.x
]
```

**Installation**:
```bash
# CPU only (default, Phases 1-7)
pip install nltools

# With GPU support (Phases 8+)
pip install nltools[gpu]

# Or with CuPy (alternative)
pip install nltools[gpu-cupy]
```

### Compatibility

**Python versions**: 3.8+
**PyTorch versions**: 1.9+ (CUDA 10.2+)
**CuPy versions**: 10.0+ (CUDA 11.0+)
**NumPy versions**: 1.19+ (unchanged)

---

## Risk Assessment

### Technical Risks

**Risk 1: GPU memory limitations** (HIGH)
- **Impact**: OOM errors for large datasets
- **Mitigation**: data_in_cpu strategy, automatic batching, user-configurable batch sizes
- **Status**: Addressed by himalaya patterns

**Risk 2: Numerical precision differences** (MEDIUM)
- **Impact**: GPU (float32) vs CPU (float64) results may differ slightly
- **Mitigation**: Use float32 consistently, extensive comparison tests, document differences
- **Status**: Acceptable trade-off (float32 sufficient for bootstrap inference)

**Risk 3: Backend compatibility** (MEDIUM)
- **Impact**: Different PyTorch/CUDA versions may behave differently
- **Mitigation**: Test matrix across PyTorch versions, clear documentation
- **Status**: Standard for GPU libraries

**Risk 4: Installation complexity** (LOW)
- **Impact**: Users may struggle with CUDA installation
- **Mitigation**: Clear docs, optional GPU support, backend='auto' fallback to CPU
- **Status**: Optional dependency, not blocking

### Project Risks

**Risk 1: Scope creep** (MEDIUM)
- **Impact**: GPU work could delay CPU integration
- **Mitigation**: GPU is Phase 8+ (after Phases 5-7 complete)
- **Status**: Sequenced correctly

**Risk 2: Testing burden** (LOW)
- **Impact**: Need GPU CI/CD for testing
- **Mitigation**: Use GitHub Actions with GPU runners, or mark GPU tests as optional
- **Status**: Standard practice, manageable

**Risk 3: Maintenance burden** (LOW)
- **Impact**: Multiple backends to maintain
- **Mitigation**: Clean abstraction, CPU backend is reference, GPU backends delegate to PyTorch/CuPy
- **Status**: Well-architected

---

## Success Criteria

### Phase 8 Success
- ✅ Backend abstraction module created and tested
- ✅ NumPy backend works (reference implementation)
- ✅ PyTorch CUDA backend works
- ✅ Backend switching works seamlessly
- ✅ All backend tests pass

### Phase 9 Success
- ✅ GPU bootstrap for simple methods implemented
- ✅ GPU results match CPU results (within numerical precision)
- ✅ data_in_cpu strategy works
- ✅ Two-dimensional batching works
- ✅ Memory efficiency verified (no OOM for reasonable datasets)

### Phase 10 Success
- ✅ Ridge weights bootstrap on GPU works
- ✅ Ridge predict bootstrap on GPU works
- ✅ GPU provides 10-100× speedup for realistic problems
- ✅ Benchmarks documented

### Phase 11 Success
- ✅ Automatic batch sizing works
- ✅ backend='auto' seamlessly selects best backend
- ✅ Documentation complete and clear
- ✅ User-facing API polished

### Overall Success
- ✅ GPU bootstrap available and working
- ✅ 10-100× speedup demonstrated for large datasets
- ✅ Memory-efficient strategies prevent OOM
- ✅ Backward compatible (CPU bootstrap unchanged)
- ✅ Optional dependency (GPU not required for core functionality)

---

## Timeline Summary

| Phase | Description | Duration | Dependencies |
|-------|-------------|----------|--------------|
| 8 | Backend abstraction | 1-2 weeks | None |
| 9 | Simple methods GPU | 1 week | Phase 8 |
| 10 | Ridge GPU bootstrap | 1-2 weeks | Phase 8, 9 |
| 11 | Optimization & polish | 1 week | Phase 8, 9, 10 |

**Total estimate**: 4-6 weeks for full GPU support

**Parallelization opportunity**: Phase 8 (backend) can start immediately after Phase 4 complete. Phases 9-11 depend on Phase 8.

---

## References

- **Himalaya research**: `claude-guidelines/himalaya-gpu-resampling-research.md`
- **Current bootstrap plan**: `claude-guidelines/bootstrap-tdd-plan.md`
- **PyTorch CUDA docs**: https://pytorch.org/docs/stable/notes/cuda.html
- **CuPy docs**: https://docs.cupy.dev/en/stable/

---

**END OF GPU IMPLEMENTATION PLAN**

**Status**: Ready for implementation after Phases 5-7 complete
**Next action**: Complete Phase 6 (error handling) and Phase 7 (performance validation)
