"""Tests for ridge backend abstraction.

Following himalaya's module-based backend pattern.
Tests backend switching, array operations, and device management.
"""

import pytest
import numpy as np
import importlib.util


def _torch_available():
    """Check if PyTorch is available."""
    return importlib.util.find_spec("torch") is not None


def _torch_cuda_available():
    """Check if PyTorch with CUDA is available."""
    if not _torch_available():
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class TestBackendSwitching:
    """Test backend selection and switching."""

    def test_set_backend_numpy(self):
        """NumPy backend should always work."""
        from nltools.algorithms.ridge.backends import set_backend

        backend = set_backend("numpy")
        assert backend.name == "numpy"

    def test_get_backend_default(self):
        """Default backend should be numpy."""
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()
        assert backend.name == "numpy"

    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not available")
    def test_set_backend_torch(self):
        """PyTorch CPU backend should work if PyTorch is installed."""
        from nltools.algorithms.ridge.backends import set_backend

        backend = set_backend("torch")
        assert backend.name == "torch"

        # Reset to numpy
        set_backend("numpy")

    @pytest.mark.skipif(
        not _torch_cuda_available(), reason="PyTorch CUDA not available"
    )
    def test_set_backend_torch_cuda(self):
        """PyTorch CUDA backend should work if CUDA is available."""
        from nltools.algorithms.ridge.backends import set_backend

        backend = set_backend("torch_cuda")
        assert backend.name == "torch_cuda"

        # Reset to numpy
        set_backend("numpy")

    def test_set_backend_invalid(self):
        """Setting invalid backend should raise ValueError."""
        from nltools.algorithms.ridge.backends import set_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("invalid_backend")

    def test_set_backend_warn_on_error(self):
        """Setting unavailable backend with on_error='warn' should warn and keep current."""
        from nltools.algorithms.ridge.backends import set_backend

        # This should warn but not raise
        backend = set_backend("fake_torch_backend", on_error="warn")
        # Should fall back to current backend (numpy)
        assert backend.name == "numpy"

    def test_backend_persistence(self):
        """Backend setting should persist across get_backend calls."""
        from nltools.algorithms.ridge.backends import set_backend, get_backend

        # Set to numpy explicitly
        set_backend("numpy")
        backend1 = get_backend()
        backend2 = get_backend()

        assert backend1.name == backend2.name == "numpy"


class TestNumpyBackend:
    """Test NumPy backend operations."""

    def setup_method(self):
        """Ensure numpy backend is active."""
        from nltools.algorithms.ridge.backends import set_backend

        set_backend("numpy")

    def test_matmul(self):
        """Test matrix multiplication."""
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        A = np.random.randn(10, 5)
        B = np.random.randn(5, 3)
        C = backend.matmul(A, B)

        assert C.shape == (10, 3)
        np.testing.assert_allclose(C, A @ B)

    def test_svd(self):
        """Test SVD decomposition."""
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        X = np.random.randn(10, 5)
        U, s, Vt = backend.svd(X, full_matrices=False)

        assert U.shape == (10, 5)
        assert s.shape == (5,)
        assert Vt.shape == (5, 5)

        # Verify reconstruction
        X_reconstructed = U @ np.diag(s) @ Vt
        np.testing.assert_allclose(X, X_reconstructed, rtol=1e-5)

    def test_asarray(self):
        """Test array conversion."""
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        # From list
        arr = backend.asarray([1, 2, 3])
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [1, 2, 3])

        # With dtype
        arr_f32 = backend.asarray([1, 2, 3], dtype=np.float32)
        assert arr_f32.dtype == np.float32

    def test_zeros_ones(self):
        """Test zeros and ones creation."""
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        z = backend.zeros((3, 4))
        assert z.shape == (3, 4)
        np.testing.assert_array_equal(z, np.zeros((3, 4)))

        # Test with dtype
        z_f32 = backend.zeros((2, 3), dtype="float32")
        assert z_f32.dtype == np.float32

    def test_zeros_like_ones_like(self):
        """Test zeros_like and ones_like."""
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        ref = np.random.randn(3, 4).astype(np.float32)

        z = backend.zeros_like(ref)
        assert z.shape == ref.shape
        assert z.dtype == ref.dtype
        np.testing.assert_array_equal(z, np.zeros_like(ref))

        o = backend.ones_like(ref)
        assert o.shape == ref.shape
        assert o.dtype == ref.dtype
        np.testing.assert_array_equal(o, np.ones_like(ref))

        # Test with custom shape
        z_custom = backend.zeros_like(ref, shape=(5, 6))
        assert z_custom.shape == (5, 6)
        assert z_custom.dtype == ref.dtype

    def test_concatenate(self):
        """Test array concatenation."""
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        a = np.random.randn(3, 4)
        b = np.random.randn(3, 4)
        c = backend.concatenate([a, b], axis=0)

        assert c.shape == (6, 4)
        np.testing.assert_array_equal(c[:3], a)
        np.testing.assert_array_equal(c[3:], b)

    def test_device_management_noop(self):
        """Test device management (no-op for NumPy)."""
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        arr = np.random.randn(3, 4)

        # These should be no-ops for NumPy
        arr_cpu = backend.to_cpu(arr)
        arr_gpu = backend.to_gpu(arr)

        np.testing.assert_array_equal(arr, arr_cpu)
        np.testing.assert_array_equal(arr, arr_gpu)

        # NumPy arrays are never in GPU
        assert not backend.is_in_gpu(arr)

    def test_to_numpy(self):
        """Test conversion to numpy."""
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        arr = np.random.randn(3, 4)
        arr_np = backend.to_numpy(arr)

        assert isinstance(arr_np, np.ndarray)
        np.testing.assert_array_equal(arr, arr_np)

    def test_transpose(self):
        """Test array transposition."""
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        arr = np.random.randn(3, 4)
        arr_t = backend.transpose(arr)

        assert arr_t.shape == (4, 3)
        np.testing.assert_array_equal(arr_t, arr.T)

    def test_sum_sqrt_abs(self):
        """Test basic math operations."""
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        arr = np.array([-2, -1, 0, 1, 2])

        # Sum
        assert backend.sum(arr) == 0

        # Abs
        abs_arr = backend.abs(arr)
        np.testing.assert_array_equal(abs_arr, np.abs(arr))

        # Sqrt
        pos_arr = np.array([1, 4, 9])
        sqrt_arr = backend.sqrt(pos_arr)
        np.testing.assert_array_equal(sqrt_arr, np.sqrt(pos_arr))

    def test_log_exp(self):
        """Test log and exp operations."""
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        arr = np.array([1, 2, 3])

        log_arr = backend.log(arr)
        np.testing.assert_allclose(log_arr, np.log(arr))

        exp_arr = backend.exp(arr)
        np.testing.assert_allclose(exp_arr, np.exp(arr))

    def test_stack(self):
        """Test array stacking."""
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        a = np.random.randn(3, 4)
        b = np.random.randn(3, 4)
        c = backend.stack([a, b], axis=0)

        assert c.shape == (2, 3, 4)
        np.testing.assert_array_equal(c[0], a)
        np.testing.assert_array_equal(c[1], b)


@pytest.mark.tier2
@pytest.mark.skipif(not _torch_available(), reason="PyTorch not available")
class TestTorchBackend:
    """Test PyTorch CPU backend operations."""

    def setup_method(self):
        """Ensure torch backend is active."""
        from nltools.algorithms.ridge.backends import set_backend

        set_backend("torch")

    def teardown_method(self):
        """Reset to numpy backend."""
        from nltools.algorithms.ridge.backends import set_backend

        set_backend("numpy")

    def test_matmul(self):
        """Test matrix multiplication."""
        import torch
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        A = torch.randn(10, 5)
        B = torch.randn(5, 3)
        C = backend.matmul(A, B)

        assert C.shape == (10, 3)
        torch.testing.assert_close(C, A @ B)

    def test_svd(self):
        """Test SVD decomposition."""
        import torch
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        X = torch.randn(10, 5)
        U, s, Vt = backend.svd(X, full_matrices=False)

        assert U.shape == (10, 5)
        assert s.shape == (5,)
        assert Vt.shape == (5, 5)

        # Verify reconstruction
        X_reconstructed = U @ torch.diag(s) @ Vt
        torch.testing.assert_close(X, X_reconstructed, rtol=1e-4, atol=1e-4)

    def test_asarray_from_numpy(self):
        """Test converting numpy array to torch tensor."""
        import torch
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        arr_np = np.array([1, 2, 3])
        arr_torch = backend.asarray(arr_np)

        assert isinstance(arr_torch, torch.Tensor)
        assert arr_torch.device.type == "cpu"
        np.testing.assert_array_equal(arr_torch.numpy(), arr_np)

    def test_asarray_dtype(self):
        """Test array conversion with dtype."""
        import torch
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        arr = backend.asarray([1, 2, 3], dtype="float32")
        assert arr.dtype == torch.float32

    def test_zeros_ones(self):
        """Test zeros and ones creation."""
        import torch
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        z = backend.zeros((3, 4))
        assert z.shape == (3, 4)
        assert isinstance(z, torch.Tensor)
        torch.testing.assert_close(z, torch.zeros((3, 4)))

    def test_device_cpu(self):
        """Test that torch backend creates CPU tensors."""
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        arr = backend.asarray([1, 2, 3])
        assert arr.device.type == "cpu"
        assert not backend.is_in_gpu(arr)

    def test_to_numpy(self):
        """Test conversion to numpy."""
        import torch
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        arr = torch.randn(3, 4)
        arr_np = backend.to_numpy(arr)

        assert isinstance(arr_np, np.ndarray)
        np.testing.assert_array_equal(arr_np, arr.numpy())

    def test_to_cpu_noop(self):
        """Test to_cpu is no-op for CPU tensors."""
        import torch
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        arr = torch.randn(3, 4)
        arr_cpu = backend.to_cpu(arr)

        assert arr_cpu.device.type == "cpu"
        torch.testing.assert_close(arr, arr_cpu)

    def test_to_gpu_noop(self):
        """Test to_gpu is no-op for CPU backend."""
        import torch
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        arr = torch.randn(3, 4)
        arr_gpu = backend.to_gpu(arr)

        # For CPU backend, to_gpu should be a no-op
        assert arr_gpu.device.type == "cpu"
        torch.testing.assert_close(arr, arr_gpu)


@pytest.mark.tier2
@pytest.mark.skipif(not _torch_cuda_available(), reason="PyTorch CUDA not available")
class TestTorchCUDABackend:
    """Test PyTorch CUDA backend operations."""

    def setup_method(self):
        """Ensure torch_cuda backend is active."""
        from nltools.algorithms.ridge.backends import set_backend

        set_backend("torch_cuda")

    def teardown_method(self):
        """Reset to numpy backend and clear GPU memory."""
        import torch
        from nltools.algorithms.ridge.backends import set_backend

        set_backend("numpy")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_asarray_default_gpu(self):
        """Test that asarray creates GPU tensors by default."""
        import torch
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        arr_np = np.array([1, 2, 3])
        arr_gpu = backend.asarray(arr_np)

        assert isinstance(arr_gpu, torch.Tensor)
        assert arr_gpu.device.type == "cuda"
        assert backend.is_in_gpu(arr_gpu)

    def test_zeros_default_gpu(self):
        """Test that zeros creates GPU tensors by default."""
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        z = backend.zeros((3, 4))
        assert z.device.type == "cuda"
        assert backend.is_in_gpu(z)

    def test_to_gpu(self):
        """Test transferring array to GPU."""
        import torch
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        # Create CPU tensor
        arr_cpu = torch.randn(3, 4)
        assert arr_cpu.device.type == "cpu"

        # Move to GPU
        arr_gpu = backend.to_gpu(arr_cpu)
        assert arr_gpu.device.type == "cuda"
        assert backend.is_in_gpu(arr_gpu)

        # Check values are preserved
        torch.testing.assert_close(arr_cpu, arr_gpu.cpu())

    def test_to_cpu(self):
        """Test transferring array to CPU."""
        import torch
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        # Create GPU tensor
        arr_gpu = backend.asarray([1, 2, 3])
        assert arr_gpu.device.type == "cuda"

        # Move to CPU
        arr_cpu = backend.to_cpu(arr_gpu)
        assert arr_cpu.device.type == "cpu"
        assert not backend.is_in_gpu(arr_cpu)

        # Check values are preserved
        torch.testing.assert_close(arr_gpu.cpu(), arr_cpu)

    def test_matmul_gpu(self):
        """Test GPU matrix multiplication."""
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        A = backend.asarray(np.random.randn(10, 5))
        B = backend.asarray(np.random.randn(5, 3))

        assert A.device.type == "cuda"
        assert B.device.type == "cuda"

        C = backend.matmul(A, B)
        assert C.device.type == "cuda"
        assert C.shape == (10, 3)

    def test_svd_gpu(self):
        """Test GPU SVD decomposition."""
        from nltools.algorithms.ridge.backends import get_backend

        backend = get_backend()

        X = backend.asarray(np.random.randn(10, 5))
        assert X.device.type == "cuda"

        U, s, Vt = backend.svd(X, full_matrices=False)

        assert U.device.type == "cuda"
        assert s.device.type == "cuda"
        assert Vt.device.type == "cuda"

        assert U.shape == (10, 5)
        assert s.shape == (5,)
        assert Vt.shape == (5, 5)


class TestDtypeUtils:
    """Test dtype utility functions."""

    def test_dtype_to_str_numpy(self):
        """Test dtype conversion for numpy dtypes."""
        from nltools.algorithms.ridge.backends._utils import _dtype_to_str

        assert _dtype_to_str(np.float32) == "float32"
        assert _dtype_to_str(np.float64) == "float64"
        assert _dtype_to_str("float32") == "float32"
        assert _dtype_to_str(None) is None

    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not available")
    def test_dtype_to_str_torch(self):
        """Test dtype conversion for torch dtypes."""
        import torch
        from nltools.algorithms.ridge.backends._utils import _dtype_to_str

        assert _dtype_to_str(torch.float32) == "float32"
        assert _dtype_to_str(torch.float64) == "float64"


class TestCheckArrays:
    """Test check_arrays function that ensures dtype/device consistency."""

    def test_check_arrays_numpy(self):
        """Test check_arrays with numpy backend."""
        from nltools.algorithms.ridge.backends import set_backend, get_backend

        set_backend("numpy")
        backend = get_backend()

        a = [1, 2, 3]
        b = np.array([4, 5, 6])
        c = None

        a_arr, b_arr, c_arr = backend.check_arrays(a, b, c)

        assert isinstance(a_arr, np.ndarray)
        assert isinstance(b_arr, np.ndarray)
        assert c_arr is None
        assert a_arr.dtype == b_arr.dtype

    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not available")
    def test_check_arrays_torch(self):
        """Test check_arrays with torch backend."""
        import torch
        from nltools.algorithms.ridge.backends import set_backend, get_backend

        set_backend("torch")
        backend = get_backend()

        a = [1, 2, 3]
        b = torch.tensor([4, 5, 6])
        c = None

        a_arr, b_arr, c_arr = backend.check_arrays(a, b, c)

        assert isinstance(a_arr, torch.Tensor)
        assert isinstance(b_arr, torch.Tensor)
        assert c_arr is None
        assert a_arr.dtype == b_arr.dtype
        assert a_arr.device == b_arr.device

        # Reset
        set_backend("numpy")
