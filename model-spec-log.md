# Test-Driven Development Plan: Backend Abstraction + Ridge Regression
**Phases 1 & 2 from model-spec.md**

**Created:** 2025-10-28
**Status:** Planning → Implementation

---

## Executive Summary

**What we're building:**
- **Phase 1**: Backend abstraction (`nltools/backends.py`) - ~100 lines
- **Phase 2**: Ridge regression algorithms (`nltools/stats/ridge.py`) - ~150 lines

**Integration with new test structure:**
- New tests go in **`nltools/tests/core/`** (functional core pattern)
- Tests will be **function-based** (not class-based) since we're testing pure computational functions
- Follows the pattern established in `test_stats.py` refactoring

**Timeline:** 2-3 days for both phases
**Branch:** Continue on `uv-cleanup`
**Current state:** 130 tests passing, clean slate for new features

---

## Pre-Implementation: Environment Setup

### Step 0: Verify Current State & Create Test Files

**Actions:**
1. Verify all existing tests pass (baseline)
2. Create new test files following core/ pattern
3. Research PyTorch API for optional GPU support

**Commands:**
```bash
# 1. Verify clean baseline
uv run pytest nltools/tests/ -x

# 2. Create new test files in core/
touch nltools/tests/core/test_backends.py
touch nltools/tests/core/test_ridge.py

# 3. Check current structure
ls -la nltools/tests/core/
```

**Deliverables:**
- [ ] Confirm 130/130 tests passing (1 skip)
- [ ] Empty test files created in core/ subdirectory
- [ ] PyTorch API verified via context7 MCP (torch.linalg.svd, device detection)

---

## Phase 1: Backend Abstraction (TDD Cycles)

**Pattern:** All tests go in `nltools/tests/core/test_backends.py` as **functions** (not classes)

### Cycle 1.1: Backend Initialization & Device Detection

**Write Tests First:**

```python
# nltools/tests/core/test_backends.py

"""
Test backend abstraction for CPU/GPU operations.

Part of functional core - tests pure backend selection and device management.
Following model-spec.md Phase 1 implementation.
"""

import numpy as np
import pytest
from nltools.backends import Backend, check_gpu_available


# ============================================================================
# Backend Initialization
# ============================================================================

def test_numpy_backend_default():
    """NumPy backend should work without PyTorch"""
    backend = Backend('numpy')
    assert backend.name == 'numpy'
    assert backend.device == 'cpu'
    assert backend.xp is np


def test_auto_backend_without_torch(monkeypatch):
    """Auto-selection should fall back to numpy if no GPU"""
    # Mock torch unavailable
    import sys
    monkeypatch.setitem(sys.modules, 'torch', None)

    backend = Backend('auto')
    assert backend.name == 'numpy'


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_torch_backend_selection():
    """Torch backend should detect available device"""
    backend = Backend('torch')
    assert backend.name.startswith('torch-')
    assert backend.device in ['cpu', 'cuda', 'mps']


def test_check_gpu_available():
    """GPU availability check should return bool and info dict"""
    available, info = check_gpu_available()
    assert isinstance(available, bool)
    assert 'backend' in info
    assert 'device' in info
    assert 'device_name' in info


# ============================================================================
# Helper Functions
# ============================================================================

def _torch_available():
    """Check if PyTorch is installed"""
    try:
        import torch
        return True
    except ImportError:
        return False
```

**TDD Workflow:**
```bash
# Run tests (expect failures)
uv run pytest nltools/tests/core/test_backends.py::test_numpy_backend_default -xvs 2>&1 | tee test_backend_init.log

# Implement minimal Backend class in nltools/backends.py

# Verify tests pass
uv run pytest nltools/tests/core/test_backends.py -k "test_numpy or test_auto or test_check" -xvs

# Check for regressions
uv run pytest nltools/tests/ -x
```

**Implementation:** Create `nltools/backends.py` with minimal Backend class

---

### Cycle 1.2: Array Operations

**Write Tests:**

```python
# Add to nltools/tests/core/test_backends.py

# ============================================================================
# Array Transfer Operations
# ============================================================================

def test_numpy_to_device():
    """NumPy backend should handle array conversion"""
    backend = Backend('numpy')
    arr = np.random.randn(10, 5)
    result = backend.to_device(arr)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, arr.astype(np.float32))


def test_numpy_to_numpy():
    """NumPy backend to_numpy should be identity"""
    backend = Backend('numpy')
    arr = np.random.randn(10, 5).astype(np.float32)
    result = backend.to_numpy(arr)

    assert result is arr  # Should be same object


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_torch_to_device():
    """Torch backend should convert numpy to torch tensor"""
    import torch
    backend = Backend('torch')
    arr = np.random.randn(10, 5)
    result = backend.to_device(arr)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert result.device.type == backend.device


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_torch_to_numpy():
    """Torch backend should convert tensor back to numpy"""
    import torch
    backend = Backend('torch')
    arr = np.random.randn(10, 5)
    tensor = backend.to_device(arr)
    result = backend.to_numpy(tensor)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, arr.astype(np.float32), rtol=1e-5)
```

**TDD Workflow:**
```bash
# Run tests (expect failures)
uv run pytest nltools/tests/core/test_backends.py -k "to_device or to_numpy" -xvs

# Implement to_device() and to_numpy() methods

# Verify
uv run pytest nltools/tests/core/test_backends.py -k "to_device or to_numpy" -xvs
```

---

### Cycle 1.3: Mathematical Operations

**Write Tests:**

```python
# Add to nltools/tests/core/test_backends.py

# ============================================================================
# Mathematical Operations
# ============================================================================

def test_numpy_svd():
    """NumPy SVD should work correctly"""
    backend = Backend('numpy')
    X = np.random.randn(20, 10).astype(np.float32)

    U, s, Vt = backend.svd(X)

    # Verify shapes
    assert U.shape == (20, 10)
    assert s.shape == (10,)
    assert Vt.shape == (10, 10)

    # Verify reconstruction
    reconstructed = U @ np.diag(s) @ Vt
    np.testing.assert_allclose(reconstructed, X, rtol=1e-4)


def test_numpy_matmul():
    """NumPy matmul should work correctly"""
    backend = Backend('numpy')
    A = np.random.randn(10, 5).astype(np.float32)
    B = np.random.randn(5, 3).astype(np.float32)

    result = backend.matmul(A, B)
    expected = A @ B

    np.testing.assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_torch_svd_equivalence():
    """Torch SVD should match NumPy results"""
    np.random.seed(42)
    X = np.random.randn(20, 10).astype(np.float32)

    # NumPy
    backend_np = Backend('numpy')
    U_np, s_np, Vt_np = backend_np.svd(X)

    # Torch
    backend_torch = Backend('torch')
    X_torch = backend_torch.to_device(X)
    U_torch, s_torch, Vt_torch = backend_torch.svd(X_torch)
    U_torch = backend_torch.to_numpy(U_torch)
    s_torch = backend_torch.to_numpy(s_torch)
    Vt_torch = backend_torch.to_numpy(Vt_torch)

    # Compare singular values
    np.testing.assert_allclose(s_torch, s_np, rtol=1e-4)

    # Check reconstruction (U/Vt may differ by sign)
    recon_np = U_np @ np.diag(s_np) @ Vt_np
    recon_torch = U_torch @ np.diag(s_torch) @ Vt_torch
    np.testing.assert_allclose(recon_torch, recon_np, rtol=1e-4)


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_torch_matmul_equivalence():
    """Torch matmul should match NumPy results"""
    np.random.seed(42)
    A = np.random.randn(10, 5).astype(np.float32)
    B = np.random.randn(5, 3).astype(np.float32)

    # NumPy
    backend_np = Backend('numpy')
    result_np = backend_np.matmul(A, B)

    # Torch
    backend_torch = Backend('torch')
    A_torch = backend_torch.to_device(A)
    B_torch = backend_torch.to_device(B)
    result_torch = backend_torch.matmul(A_torch, B_torch)
    result_torch = backend_torch.to_numpy(result_torch)

    np.testing.assert_allclose(result_torch, result_np, rtol=1e-5)
```

**TDD Workflow:**
```bash
# Run tests
uv run pytest nltools/tests/core/test_backends.py -k "svd or matmul" -xvs

# Implement svd() and matmul() methods

# Verify
uv run pytest nltools/tests/core/test_backends.py -xvs
```

---

### Cycle 1.4: Auto-Selection Logic

**Write Tests:**

```python
# Add to nltools/tests/core/test_backends.py

from nltools.backends import auto_select_backend

# ============================================================================
# Auto-Selection Logic
# ============================================================================

def test_small_dataset_uses_numpy():
    """Small datasets should use NumPy even if GPU available"""
    # Small problem
    backend = auto_select_backend(n_samples=100, n_features=1000)
    # Should use numpy (or torch-cpu) to avoid transfer overhead
    assert backend.name in ['numpy', 'torch-cpu']


def test_large_dataset_considers_gpu():
    """Large datasets should consider GPU if available"""
    backend = auto_select_backend(n_samples=300, n_features=100000)

    # If GPU available, should use torch; otherwise numpy
    assert backend.name in ['numpy', 'torch-cuda', 'torch-mps', 'torch-cpu']


def test_cv_enables_gpu():
    """Cross-validation should prefer GPU even for medium datasets"""
    backend = auto_select_backend(n_samples=200, n_features=30000, cv=5)

    # With CV, should prefer GPU if available
    assert backend.name in ['numpy', 'torch-cuda', 'torch-mps', 'torch-cpu']


def test_auto_selection_without_gpu():
    """Auto-selection should work without GPU"""
    # This should always work
    backend = auto_select_backend(n_samples=1000, n_features=100000)
    assert backend.name in ['numpy', 'torch-cpu', 'torch-cuda', 'torch-mps']
```

**TDD Workflow:**
```bash
# Run tests
uv run pytest nltools/tests/core/test_backends.py -k "auto_select" -xvs

# Implement auto_select_backend() function

# Verify all backend tests pass
uv run pytest nltools/tests/core/test_backends.py -xvs --tb=long 2>&1 | tee test_backends_full.log
```

---

### Phase 1 Completion Checklist

```bash
# Run all backend tests
uv run pytest nltools/tests/core/test_backends.py -xvs

# Verify no regressions in existing tests
uv run pytest nltools/tests/ -x

# Count new tests
uv run pytest nltools/tests/core/test_backends.py --collect-only

# Stage changes (WAIT FOR APPROVAL)
git add nltools/backends.py nltools/tests/core/test_backends.py
git status
```

**Expected outcome:**
- [ ] ~15 new backend tests passing
- [ ] No regressions (130 existing tests still pass)
- [ ] Backend abstraction complete (~100 lines)
- [ ] Ready for Phase 2

---

## Phase 2: Ridge Regression (TDD Cycles)

**Pattern:** All tests go in `nltools/tests/core/test_ridge.py` as **functions**

### Step: Create stats/ directory

```bash
# Create new stats module structure
mkdir -p nltools/stats
touch nltools/stats/__init__.py
touch nltools/stats/ridge.py
```

---

### Cycle 2.1: Basic Ridge SVD Solver

**Write Tests:**

```python
# nltools/tests/core/test_ridge.py

"""
Test ridge regression algorithms.

Part of functional core - tests SVD-based ridge regression inspired by himalaya.
Following model-spec.md Phase 2 implementation.
"""

import numpy as np
import pytest
from nltools.stats.ridge import ridge_svd
from nltools.backends import Backend


# ============================================================================
# Ridge SVD Solver
# ============================================================================

def test_ridge_svd_single_target():
    """Ridge SVD should solve single-target regression"""
    np.random.seed(42)
    n_samples, n_features = 100, 50

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    alpha = 1.0

    beta = ridge_svd(X, y, alpha=alpha)

    # Check shape
    assert beta.shape == (n_features,)

    # Verify it reduces to OLS when alpha≈0
    beta_ols = ridge_svd(X, y, alpha=1e-10)
    beta_expected = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(beta_ols, beta_expected, rtol=1e-3)


def test_ridge_svd_multi_target():
    """Ridge SVD should handle multiple targets"""
    np.random.seed(42)
    n_samples, n_features, n_targets = 100, 50, 5

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    Y = np.random.randn(n_samples, n_targets).astype(np.float32)
    alpha = 1.0

    beta = ridge_svd(X, Y, alpha=alpha)

    # Check shape
    assert beta.shape == (n_features, n_targets)

    # Each column should solve the corresponding target
    for i in range(n_targets):
        beta_single = ridge_svd(X, Y[:, i], alpha=alpha)
        np.testing.assert_allclose(beta[:, i], beta_single, rtol=1e-5)


def test_ridge_vs_sklearn():
    """Ridge SVD should match sklearn Ridge"""
    from sklearn.linear_model import Ridge

    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    alpha = 1.0

    # Our implementation
    beta_ours = ridge_svd(X, y, alpha=alpha)

    # sklearn
    ridge_sklearn = Ridge(alpha=alpha, fit_intercept=False, solver='svd')
    ridge_sklearn.fit(X, y)
    beta_sklearn = ridge_sklearn.coef_

    np.testing.assert_allclose(beta_ours, beta_sklearn, rtol=1e-4)


def test_ridge_regularization_effect():
    """Higher alpha should shrink coefficients"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    beta_small = ridge_svd(X, y, alpha=0.1)
    beta_large = ridge_svd(X, y, alpha=10.0)

    # Higher alpha should give smaller coefficients
    assert np.linalg.norm(beta_large) < np.linalg.norm(beta_small)


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_ridge_cpu_gpu_equivalence():
    """CPU and GPU should give same results"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    alpha = 1.0

    # CPU
    backend_cpu = Backend('numpy')
    beta_cpu = ridge_svd(X, y, alpha=alpha, backend=backend_cpu)

    # GPU
    backend_gpu = Backend('torch')
    beta_gpu = ridge_svd(X, y, alpha=alpha, backend=backend_gpu)

    np.testing.assert_allclose(beta_gpu, beta_cpu, rtol=1e-4)


# ============================================================================
# Helper Functions
# ============================================================================

def _torch_available():
    """Check if PyTorch is installed"""
    try:
        import torch
        return True
    except ImportError:
        return False
```

**TDD Workflow:**
```bash
# Run tests (expect failures)
uv run pytest nltools/tests/core/test_ridge.py -k "ridge_svd" -xvs 2>&1 | tee test_ridge_svd.log

# Implement ridge_svd() in nltools/stats/ridge.py
# Include himalaya attribution in docstring

# Verify
uv run pytest nltools/tests/core/test_ridge.py -k "ridge_svd" -xvs
```

---

### Cycle 2.2: Ridge Cross-Validation

**Write Tests:**

```python
# Add to nltools/tests/core/test_ridge.py

from nltools.stats.ridge import ridge_cv

# ============================================================================
# Ridge Cross-Validation
# ============================================================================

def test_ridge_cv_basic():
    """Ridge CV should select alpha and return results"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    result = ridge_cv(X, y, alphas=[0.1, 1.0, 10.0], cv=3, backend='numpy')

    # Check result structure
    assert 'alpha' in result
    assert 'coef' in result
    assert 'cv_scores' in result
    assert 'backend' in result

    # Check selected alpha
    assert result['alpha'] in [0.1, 1.0, 10.0]

    # Check coefficients shape
    assert result['coef'].shape == (50,)

    # Check CV scores shape: (n_folds, n_alphas, n_targets)
    assert result['cv_scores'].shape == (3, 3, 1)


def test_ridge_cv_multi_target():
    """Ridge CV should handle multiple targets"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    Y = np.random.randn(100, 5).astype(np.float32)

    result = ridge_cv(X, Y, alphas=[0.1, 1.0, 10.0], cv=3, backend='numpy')

    # Check coefficients shape
    assert result['coef'].shape == (50, 5)

    # Check CV scores shape
    assert result['cv_scores'].shape == (3, 3, 5)


def test_ridge_cv_default_alphas():
    """Ridge CV should use default alphas if not provided"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    result = ridge_cv(X, y, cv=3, backend='numpy')

    # Should have selected some alpha
    assert result['alpha'] > 0
    assert result['coef'].shape == (50,)


def test_ridge_cv_reproducibility():
    """Ridge CV should give reproducible results with same seed"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    result1 = ridge_cv(X, y, alphas=[0.1, 1.0], cv=3, backend='numpy')

    np.random.seed(42)
    X2 = np.random.randn(100, 50).astype(np.float32)
    y2 = np.random.randn(100).astype(np.float32)
    result2 = ridge_cv(X2, y2, alphas=[0.1, 1.0], cv=3, backend='numpy')

    assert result1['alpha'] == result2['alpha']
    np.testing.assert_allclose(result1['coef'], result2['coef'], rtol=1e-5)


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_ridge_cv_cpu_gpu_equivalence():
    """CPU and GPU CV should give same results"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    alphas = [0.1, 1.0, 10.0]

    result_cpu = ridge_cv(X, y, alphas=alphas, cv=3, backend='numpy')
    result_gpu = ridge_cv(X, y, alphas=alphas, cv=3, backend='torch')

    assert result_cpu['alpha'] == result_gpu['alpha']
    np.testing.assert_allclose(result_cpu['coef'], result_gpu['coef'], rtol=1e-4)
```

**TDD Workflow:**
```bash
# Run tests
uv run pytest nltools/tests/core/test_ridge.py -k "ridge_cv" -xvs 2>&1 | tee test_ridge_cv.log

# Implement ridge_cv() in nltools/stats/ridge.py

# Verify
uv run pytest nltools/tests/core/test_ridge.py -k "ridge_cv" -xvs
```

---

### Cycle 2.3: Edge Cases & Performance

**Write Tests:**

```python
# Add to nltools/tests/core/test_ridge.py

# ============================================================================
# Performance & Large Datasets
# ============================================================================

def test_large_dataset_completion():
    """Ridge CV should complete on neuroimaging-sized datasets"""
    np.random.seed(42)
    # Neuroimaging-sized problem
    X = np.random.randn(300, 10000).astype(np.float32)
    y = np.random.randn(300).astype(np.float32)

    result = ridge_cv(X, y, alphas=[0.1, 1.0, 10.0], cv=3, backend='auto')

    assert result['coef'].shape == (10000,)
    assert result['alpha'] > 0


def test_auto_backend_selection():
    """Auto backend should select appropriately based on problem size"""
    np.random.seed(42)

    # Small problem
    X_small = np.random.randn(100, 1000).astype(np.float32)
    y_small = np.random.randn(100).astype(np.float32)
    result_small = ridge_cv(X_small, y_small, cv=3, backend='auto')
    assert result_small['coef'].shape == (1000,)

    # Large problem
    X_large = np.random.randn(300, 50000).astype(np.float32)
    y_large = np.random.randn(300).astype(np.float32)
    result_large = ridge_cv(X_large, y_large, alphas=[1.0, 10.0], cv=3, backend='auto')
    assert result_large['coef'].shape == (50000,)
    assert result_large['backend'] in ['numpy', 'torch-cpu', 'torch-cuda', 'torch-mps']


# ============================================================================
# Edge Cases
# ============================================================================

def test_single_alpha():
    """Should work with single alpha value"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    result = ridge_cv(X, y, alphas=[1.0], cv=3, backend='numpy')
    assert result['alpha'] == 1.0


def test_perfect_fit_case():
    """Should handle perfect fit scenarios"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    beta_true = np.random.randn(50).astype(np.float32)
    y = X @ beta_true  # Perfect linear relationship

    result = ridge_cv(X, y, alphas=[1e-6, 0.1, 1.0], cv=3, backend='numpy')

    # Should prefer small alpha for perfect fit
    assert result['alpha'] <= 0.1


def test_noisy_data():
    """Should handle noisy data appropriately"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    beta_true = np.random.randn(50).astype(np.float32)
    y = X @ beta_true + 0.5 * np.random.randn(100).astype(np.float32)

    result = ridge_cv(X, y, alphas=[0.01, 0.1, 1.0, 10.0], cv=3, backend='numpy')

    # Should select some regularization
    assert 0.01 <= result['alpha'] <= 10.0
```

**TDD Workflow:**
```bash
# Run performance tests
uv run pytest nltools/tests/core/test_ridge.py -k "large_dataset or auto_backend" -xvs

# Run edge case tests
uv run pytest nltools/tests/core/test_ridge.py -k "single_alpha or perfect_fit or noisy" -xvs

# Run all ridge tests
uv run pytest nltools/tests/core/test_ridge.py -xvs --tb=long 2>&1 | tee test_ridge_full.log
```

---

### Phase 2 Completion Checklist

```bash
# Run all ridge tests
uv run pytest nltools/tests/core/test_ridge.py -xvs

# Verify no regressions
uv run pytest nltools/tests/ -x

# Count new tests
uv run pytest nltools/tests/core/test_ridge.py --collect-only

# Stage changes (WAIT FOR APPROVAL)
git add nltools/stats/__init__.py nltools/stats/ridge.py nltools/tests/core/test_ridge.py
git status
```

**Expected outcome:**
- [ ] ~16 new ridge tests passing
- [ ] No regressions (130 + ~15 backend tests still pass)
- [ ] Ridge regression complete (~150 lines)
- [ ] Ready for integration

---

## Integration & Documentation

### Step 1: Update stats module exports

```python
# nltools/stats/__init__.py
"""Statistical algorithms for nltools."""

from .ridge import ridge_svd, ridge_cv

__all__ = ['ridge_svd', 'ridge_cv']
```

### Step 2: Update documentation

**Files to update:**
1. **MIGRATION_v0.5_to_v0.6.md** - Add new features section
2. **REFACTORING_PLAN.md** - Mark Phases 1 & 2 as complete
3. **model-spec.md** - Update progress tracker

### Step 3: Final verification

```bash
# Run full test suite
uv run pytest nltools/tests/ -x --tb=short 2>&1 | tee test_final_phases_1_2.log

# Check test count (should be ~161 tests)
# 130 existing + 15 backend + 16 ridge = 161
uv run pytest --collect-only | grep "test session"

# Verify imports work
uv run python -c "from nltools.backends import Backend; from nltools.stats.ridge import ridge_cv; print('✓ Imports successful')"

# Check no regressions on existing core tests
uv run pytest nltools/tests/core/test_stats.py -xvs
```

---

## Summary Checklist

### Phase 1: Backend Abstraction ✅
- [ ] test_backends.py created in core/ (function-based tests)
- [ ] Backend initialization tests (4 tests)
- [ ] Array operation tests (4 tests)
- [ ] Math operation tests (4 tests)
- [ ] Auto-selection tests (3 tests)
- [ ] backends.py implementation (~100 lines)
- [ ] All tests passing
- [ ] No regressions

### Phase 2: Ridge Regression ✅
- [ ] stats/ directory created
- [ ] test_ridge.py created in core/ (function-based tests)
- [ ] Ridge SVD tests (6 tests)
- [ ] Ridge CV tests (5 tests)
- [ ] Performance tests (2 tests)
- [ ] Edge case tests (3 tests)
- [ ] ridge.py implementation (~150 lines)
- [ ] All tests passing
- [ ] No regressions

### Integration ✅
- [ ] stats/__init__.py exports
- [ ] MIGRATION_v0.5_to_v0.6.md updated
- [ ] REFACTORING_PLAN.md updated
- [ ] model-spec.md progress updated
- [ ] All imports work
- [ ] ~161 total tests passing

### Final Actions
- [ ] Stage all changes: `git add .`
- [ ] Review: `git status` + `git diff --staged`
- [ ] Say: "Phases 1 & 2 complete - changes staged and ready for review"
- [ ] **WAIT FOR APPROVAL** before committing

---

## Estimated Timeline

- **Phase 1 (Backend)**: 4-6 hours
  - Cycle 1.1: 1 hour
  - Cycle 1.2: 1 hour
  - Cycle 1.3: 1.5 hours
  - Cycle 1.4: 0.5 hours

- **Phase 2 (Ridge)**: 6-8 hours
  - Cycle 2.1: 2 hours
  - Cycle 2.2: 3 hours
  - Cycle 2.3: 1 hour

- **Integration**: 1-2 hours
  - Documentation updates
  - Final verification

**Total: 1.5-2 days**

---

## Token Efficiency Tips

✅ **Always capture pytest output to log files FIRST**
✅ **Use Read/Grep tools on logs instead of re-running tests**
✅ **Run targeted tests during development** (`-k pattern`)
✅ **Only run full suite at checkpoints**

**Token savings:**
- Each pytest run: 1,000-5,000 tokens
- Each Grep search: ~50 tokens
- Searching 5 patterns: 25,000 tokens (re-running) vs 5,250 tokens (using logs) = **80% savings**

---

## Key Success Factors

1. **Follow new test structure**: Tests in `core/` as functions (not classes)
2. **Pattern consistency**: Match test_stats.py organization style
3. **One test, one concept**: Each test function tests one specific behavior
4. **Clear sections**: Use comment headers to organize related tests
5. **Himalaya attribution**: Include BSD-3-Clause license reference in ridge.py
6. **No regressions**: 130 existing tests must continue passing

---

## Progress Log

### 2025-10-28 (Evening): Sprint 1 Documentation & Benchmarking Completed ✅

**What was completed:**
- ✅ Comprehensive benchmarking suite (`benchmarks/benchmark_ridge.py`)
- ✅ Initial benchmark results (`results_ridge_performance.csv`)
- ✅ Backend API documentation (`docs/api/backends.md`)
- ✅ Algorithms API documentation (`docs/api/algorithms.md` - ridge, hrf, srm)
- ✅ Performance guide with real benchmark data (`docs/performance.md`)
- ✅ Updated documentation TOC (`docs/_toc.yml`)
- ✅ Documentation builds successfully with jupyter-book

**Benchmark highlights:**
- 19 benchmark runs across 5 scenarios
- System: macOS Apple Silicon (MPS backend)
- Key finding: PyTorch MPS shows 1.4-2.2x speedup (limited by SVD CPU fallback)
- Medium problems (300×50k): 2.2x speedup
- Large problems (1000×200k): 1.4x speedup
- 5-fold CV (300×100k): 1.6x speedup

**New systematic benchmark created (not yet run):**
- ✅ `benchmark_ridge_systematic.py` - Realistic neuroimaging workflows
- ✅ `benchmarking-guide.md` - Complete methodology documentation
- Grid: 2 time-series lengths × 2 resolutions × 2 estimation styles = 8 conditions
- Estimated runtime: ~55-60 minutes
- Covers: Task fMRI (500tp), Naturalistic fMRI (1000tp), 3mm (50k voxels), 2mm (230k voxels)
- Estimation styles: Estimates-only (fixed α) vs Fit-only (5-fold CV)

**Files created/modified (6 files):**
- `benchmarks/benchmark_ridge.py` (507 lines)
- `benchmarks/results_ridge_performance.csv` (19 runs)
- `benchmarks/benchmark_ridge_systematic.py` (ready to run)
- `benchmarks/benchmarking-guide.md` (comprehensive methodology)
- `docs/api/backends.md` (complete API reference)
- `docs/api/algorithms.md` (ridge, hrf, srm documentation)
- `docs/performance.md` (340 lines with real data)
- `docs/_toc.yml` (added 3 new pages)

**Status:** Ready to commit after review

**Next steps:**
1. Review and commit Sprint 1 documentation work
2. Run systematic benchmarks (~1 hour)
3. Update performance.md with systematic results
4. Continue with remaining model-spec.md phases (if applicable)

---

### 2025-10-28 (Morning): Plan Created
- Comprehensive TDD plan written
- Aligned with new test suite organization (core/ subdirectory)
- Following function-based pattern from test_stats.py refactoring
- Ready to begin implementation

**Next steps:**
1. Verify baseline (130 tests passing)
2. Create empty test files
3. Begin Cycle 1.1: Backend initialization tests
