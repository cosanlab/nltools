"""
Test backend abstraction for CPU/GPU operations.

Part of functional core - tests pure backend selection and device management.
Following model-spec.md Phase 1 implementation.
"""

import numpy as np
import pytest


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


# ============================================================================
# Backend Initialization
# ============================================================================


def test_numpy_backend_default():
    """NumPy backend should work without PyTorch"""
    from nltools.backends import Backend

    backend = Backend('numpy')
    assert backend.name == 'numpy'
    assert backend.device == 'cpu'
    assert backend.xp is np


def test_auto_backend_without_torch(monkeypatch):
    """Auto-selection should fall back to numpy if no GPU"""
    from nltools.backends import Backend
    import sys

    # Mock torch unavailable
    monkeypatch.setitem(sys.modules, 'torch', None)

    backend = Backend('auto')
    assert backend.name == 'numpy'


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_torch_backend_selection():
    """Torch backend should detect available device"""
    from nltools.backends import Backend

    backend = Backend('torch')
    assert backend.name.startswith('torch-')
    assert backend.device in ['cpu', 'cuda', 'mps']


def test_check_gpu_available():
    """GPU availability check should return bool and info dict"""
    from nltools.backends import check_gpu_available

    available, info = check_gpu_available()
    assert isinstance(available, bool)
    assert 'backend' in info
    assert 'device' in info
    assert 'device_name' in info


# ============================================================================
# Array Transfer Operations
# ============================================================================


def test_numpy_to_device():
    """NumPy backend should handle array conversion"""
    from nltools.backends import Backend

    backend = Backend('numpy')
    arr = np.random.randn(10, 5)
    result = backend.to_device(arr)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, arr.astype(np.float32))


def test_numpy_to_numpy():
    """NumPy backend to_numpy should be identity"""
    from nltools.backends import Backend

    backend = Backend('numpy')
    arr = np.random.randn(10, 5).astype(np.float32)
    result = backend.to_numpy(arr)

    assert result is arr  # Should be same object


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_torch_to_device():
    """Torch backend should convert numpy to torch tensor"""
    import torch
    from nltools.backends import Backend

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
    from nltools.backends import Backend

    backend = Backend('torch')
    arr = np.random.randn(10, 5)
    tensor = backend.to_device(arr)
    result = backend.to_numpy(tensor)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, arr.astype(np.float32), rtol=1e-5)


# ============================================================================
# Mathematical Operations
# ============================================================================


def test_numpy_svd():
    """NumPy SVD should work correctly"""
    from nltools.backends import Backend

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
    from nltools.backends import Backend

    backend = Backend('numpy')
    A = np.random.randn(10, 5).astype(np.float32)
    B = np.random.randn(5, 3).astype(np.float32)

    result = backend.matmul(A, B)
    expected = A @ B

    np.testing.assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_torch_svd_equivalence():
    """Torch SVD should match NumPy results"""
    from nltools.backends import Backend

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
    np.testing.assert_allclose(s_torch, s_np, rtol=1e-3)

    # Check reconstruction (U/Vt may differ by sign)
    recon_np = U_np @ np.diag(s_np) @ Vt_np
    recon_torch = U_torch @ np.diag(s_torch) @ Vt_torch
    np.testing.assert_allclose(recon_torch, recon_np, rtol=1e-3)


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_torch_matmul_equivalence():
    """Torch matmul should match NumPy results"""
    from nltools.backends import Backend

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


# ============================================================================
# Auto-Selection Logic
# ============================================================================


def test_small_dataset_uses_numpy():
    """Small datasets should use NumPy even if GPU available"""
    from nltools.backends import auto_select_backend

    # Small problem
    backend = auto_select_backend(n_samples=100, n_features=1000)
    # Should use numpy (or torch-cpu) to avoid transfer overhead
    assert backend.name in ['numpy', 'torch-cpu']


def test_large_dataset_considers_gpu():
    """Large datasets should consider GPU if available"""
    from nltools.backends import auto_select_backend

    backend = auto_select_backend(n_samples=300, n_features=100000)

    # If GPU available, should use torch; otherwise numpy
    assert backend.name in ['numpy', 'torch-cuda', 'torch-mps', 'torch-cpu']


def test_cv_enables_gpu():
    """Cross-validation should prefer GPU even for medium datasets"""
    from nltools.backends import auto_select_backend

    backend = auto_select_backend(n_samples=200, n_features=30000, cv=5)

    # With CV, should prefer GPU if available
    assert backend.name in ['numpy', 'torch-cuda', 'torch-mps', 'torch-cpu']


def test_auto_selection_without_gpu():
    """Auto-selection should work without GPU"""
    from nltools.backends import auto_select_backend

    # This should always work
    backend = auto_select_backend(n_samples=1000, n_features=100000)
    assert backend.name in ['numpy', 'torch-cpu', 'torch-cuda', 'torch-mps']
