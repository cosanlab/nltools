"""
Backend abstraction for CPU/GPU operations.

Supports NumPy (CPU-only) and PyTorch (CPU/CUDA/MPS) backends for
linear algebra operations. Enables transparent acceleration while
maintaining NumPy-first development.
"""

import warnings
import numpy as np
from typing import Tuple, Dict, Any

# Track if we've warned about MPS initialization to avoid spam
_already_warned_mps_init = [False]
# Track if we've warned about float64 conversion to avoid spam
_already_warned_float64 = [False]


class Backend:
    """
    Backend abstraction for numerical operations.

    Provides a unified interface for NumPy and PyTorch operations,
    enabling transparent GPU acceleration when available.

    Args:
        backend (str): Backend type: 'numpy', 'torch', or 'auto'
            - 'numpy': CPU-only using NumPy
            - 'torch': PyTorch with automatic device detection (cuda/mps/cpu)
            - 'auto': Automatically select best available backend

    Attributes:
        name (str): Backend identifier (e.g., 'numpy', 'torch-cuda', 'torch-mps')
        device (str): Device type ('cpu', 'cuda', or 'mps')
        xp (module): Array library module (numpy or torch)
    """

    def __init__(self, backend: str = "numpy"):
        if backend == "numpy":
            self._init_numpy()
        elif backend == "torch":
            self._init_torch()
        elif backend == "auto":
            self._init_auto()
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Use 'numpy', 'torch', or 'auto'"
            )

    def _init_numpy(self):
        """Initialize NumPy backend"""
        self.name = "numpy"
        self.device = "cpu"
        self.xp = np
        self._torch_device = None

    def _init_torch(self):
        """Initialize PyTorch backend with device detection"""
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch not installed. Install with: pip install torch\n"
                "Or use backend='numpy' for CPU-only operations."
            )

        self.xp = torch

        # Detect best available device
        if torch.cuda.is_available():
            self.device = "cuda"
            self._torch_device = torch.device("cuda")
            self.name = "torch-cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self._torch_device = torch.device("mps")
            self.name = "torch-mps"
            # Warn about MPS precision limitations
            if not _already_warned_mps_init[0]:
                warnings.warn(
                    "torch-mps backend uses float32 precision due to MPS framework limitations. "
                    "This may result in reduced numerical precision compared to float64 backends. "
                    "For high-precision requirements, consider using 'torch' (CPU) or 'numpy' backends.",
                    UserWarning,
                    stacklevel=3,
                )
                _already_warned_mps_init[0] = True
        else:
            self.device = "cpu"
            self._torch_device = torch.device("cpu")
            self.name = "torch-cpu"

    @property
    def is_gpu(self):
        """True if backend is using a GPU device (CUDA or MPS)."""
        return self.device in ("cuda", "mps")

    def _init_auto(self):
        """Automatically select best backend"""
        import importlib.util

        # Check if PyTorch is available without importing it
        if importlib.util.find_spec("torch") is not None:
            # PyTorch available, use it
            self._init_torch()
        else:
            # Fall back to NumPy
            self._init_numpy()

    def to_device(self, arr: np.ndarray):
        """
        Transfer array to backend device.

        Args:
            arr (np.ndarray): Input numpy array

        Returns:
            array: Array on device (numpy array or torch tensor)
        """
        if self.name == "numpy":
            # NumPy backend: ensure float32
            return arr.astype(np.float32)
        else:
            # PyTorch backend: convert to tensor and move to device
            import torch

            # Check for float64 conversion and warn if needed
            if arr.dtype == np.float64 and self.device == "mps":
                if not _already_warned_float64[0]:
                    warnings.warn(
                        f"GPU backend {self.name} requires single precision floats (float32), "
                        f"got input in float64. Data will be automatically cast to float32. "
                        "This may result in reduced numerical precision.",
                        UserWarning,
                        stacklevel=2,
                    )
                    _already_warned_float64[0] = True

            tensor = torch.from_numpy(arr.astype(np.float32))
            return tensor.to(self._torch_device)

    def to_numpy(self, arr):
        """
        Convert array back to NumPy.

        Args:
            arr (np.ndarray or torch.Tensor): Array to convert

        Returns:
            np.ndarray: NumPy array
        """
        if self.name == "numpy":
            # NumPy backend: identity operation
            return arr
        else:
            # PyTorch backend: move to CPU and convert
            import torch

            if isinstance(arr, torch.Tensor):
                return arr.cpu().numpy()
            else:
                return arr

    def svd(self, X, full_matrices=False):
        """
        Compute Singular Value Decomposition.

        Args:
            X (array): Input matrix (n_samples, n_features)
            full_matrices (bool, default=False): If False, returns reduced SVD

        Returns:
            tuple: (U, s, Vt) where:
                - U (array): Left singular vectors
                - s (array): Singular values
                - Vt (array): Right singular vectors (transposed)
        """
        if self.name == "numpy":
            try:
                import scipy.linalg as linalg

                use_scipy = True
            except ImportError:
                linalg = np.linalg
                use_scipy = False

            if X.ndim == 2 or not use_scipy:
                return linalg.svd(X, full_matrices=full_matrices)
            elif X.ndim == 3:
                UsV = [linalg.svd(Xi, full_matrices=full_matrices) for Xi in X]
                return tuple(map(np.stack, zip(*UsV)))
            else:
                raise NotImplementedError("SVD only supports 2D and 3D arrays")
        elif self.device == "mps":
            import torch

            X_device = X.device
            X_cpu = X.cpu().to(torch.float64)
            U, s, Vt = torch.linalg.svd(X_cpu, full_matrices=full_matrices)
            U = U.to(dtype=torch.float32, device=X_device)
            s = s.to(dtype=torch.float32, device=X_device)
            Vt = Vt.to(dtype=torch.float32, device=X_device)
            return U, s, Vt
        else:
            import torch

            return torch.linalg.svd(X, full_matrices=full_matrices)

    def matmul(self, A, B):
        """
        Matrix multiplication.

        Args:
            A (array): First matrix
            B (array): Second matrix

        Returns:
            array: Result of A @ B
        """
        if self.name == "numpy":
            return A @ B
        else:
            import torch

            return torch.matmul(A, B)

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def dtype_to_str(dtype):
        """Normalize a dtype (numpy, torch, or string) to its string name.

        Args:
            dtype: Data type to convert (str, numpy dtype, torch dtype, or None).

        Returns:
            str or None: e.g. "float32", "float64", or None if input was None.
        """
        if isinstance(dtype, str):
            return dtype
        if dtype is None:
            return None
        # numpy dtype instances (np.dtype('float32')) and type objects (np.float32)
        if hasattr(dtype, "name"):
            return dtype.name
        # torch dtypes: str(torch.float32) == "torch.float32"
        dtype_str = str(dtype)
        if "torch." in dtype_str:
            return dtype_str.split("torch.")[-1]
        # last resort: try numpy conversion
        try:
            return np.dtype(dtype).name
        except (TypeError, ValueError):
            pass
        raise NotImplementedError(f"Cannot convert dtype {dtype} to string")

    # ------------------------------------------------------------------
    # Array conversion
    # ------------------------------------------------------------------

    def asarray(self, x, dtype=None, device=None):
        """Convert input to a backend array.

        Handles numpy arrays, lists, and torch tensors. Places result on
        the backend's device (or an explicit *device*).

        Args:
            x: Input data (array-like, tensor, list).
            dtype: Desired dtype as string, numpy, or torch dtype. If None,
                inferred from input.
            device: Target device string (e.g. "cpu", "cuda"). Ignored for
                numpy backend. If None, uses the backend's default device.

        Returns:
            Backend array (numpy ndarray or torch Tensor).
        """
        if self.name == "numpy":
            if dtype is not None:
                dtype = self.dtype_to_str(dtype)
            try:
                return np.asarray(x, dtype=dtype)
            except Exception:
                pass
            # torch tensor on CPU
            try:
                return np.asarray(x.cpu().numpy(), dtype=dtype)
            except Exception:
                pass
            return np.asarray(x, dtype=dtype)
        else:
            import torch

            if dtype is None:
                if isinstance(x, torch.Tensor):
                    dtype = x.dtype
                elif hasattr(x, "dtype") and hasattr(x.dtype, "name"):
                    dtype = x.dtype.name
            if dtype is not None:
                dtype_s = self.dtype_to_str(dtype)
                dtype = getattr(torch, dtype_s)
            if device is None:
                device = self._torch_device
                if isinstance(x, torch.Tensor) and device is None:
                    device = x.device
            # MPS doesn't support float64 — enforce float32
            if (
                self.device == "mps"
                and dtype is not None
                and self.dtype_to_str(dtype) == "float64"
            ):
                dtype = torch.float32
            try:
                return torch.as_tensor(x, dtype=dtype, device=device)
            except Exception:
                arr = np.asarray(x, dtype=self.dtype_to_str(dtype))
                return torch.as_tensor(arr, dtype=dtype, device=device)

    def asarray_like(self, x, ref):
        """Convert *x* to an array matching *ref*'s dtype (and device for torch).

        Args:
            x: Input data.
            ref: Reference array whose dtype/device to match.

        Returns:
            Backend array with same dtype/device as ref.
        """
        if self.name == "numpy":
            return np.asarray(x, dtype=ref.dtype)
        else:
            import torch

            return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)

    def check_arrays(self, *inputs):
        """Coerce all inputs to the same dtype (and device) as the first.

        None values are passed through. Lists of arrays are converted
        element-wise.

        Args:
            *inputs: Arrays, lists of arrays, or None.

        Returns:
            list: Converted arrays in the same order as inputs.
        """
        result = []
        first = self.asarray(inputs[0])
        result.append(first)
        dtype = first.dtype
        for item in inputs[1:]:
            if item is None:
                result.append(None)
            elif isinstance(item, list):
                result.append([self.asarray(el, dtype=dtype) for el in item])
            else:
                result.append(self.asarray(item, dtype=dtype))
        return result

    # ------------------------------------------------------------------
    # Array creation with shape override
    # ------------------------------------------------------------------

    def _resolve_torch_device(self, array=None, device=None):
        """Resolve target torch device from explicit arg, array, or backend default."""
        if device is not None:
            return device
        if array is not None and hasattr(array, "device"):
            return array.device
        return self._torch_device

    def zeros_like(self, array, shape=None, dtype=None, device=None):
        """Create zeros array, optionally with a different shape.

        Args:
            array: Reference array for dtype inference.
            shape: Output shape. If None, uses array.shape.
            dtype: Output dtype. If None, uses array.dtype.
            device: Target device (torch only). If None, uses array's device.
        """
        if shape is None:
            shape = array.shape
        if dtype is None:
            dtype = array.dtype
        if self.name == "numpy":
            return np.zeros(shape, dtype=dtype)
        else:
            import torch

            if isinstance(shape, int):
                shape = (shape,)
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)
            return torch.zeros(
                shape, dtype=dtype, device=self._resolve_torch_device(array, device)
            )

    def ones_like(self, array, shape=None, dtype=None, device=None):
        """Create ones array, optionally with a different shape.

        Args:
            array: Reference array for dtype inference.
            shape: Output shape. If None, uses array.shape.
            dtype: Output dtype. If None, uses array.dtype.
            device: Target device (torch only). If None, uses array's device.
        """
        if shape is None:
            shape = array.shape
        if dtype is None:
            dtype = array.dtype
        if self.name == "numpy":
            return np.ones(shape, dtype=dtype)
        else:
            import torch

            if isinstance(shape, int):
                shape = (shape,)
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)
            return torch.ones(
                shape, dtype=dtype, device=self._resolve_torch_device(array, device)
            )

    def full_like(self, array, fill_value, shape=None, dtype=None, device=None):
        """Create array filled with *fill_value*, optionally with a different shape.

        Args:
            array: Reference array for dtype inference.
            fill_value: Scalar fill value.
            shape: Output shape. If None, uses array.shape.
            dtype: Output dtype. If None, uses array.dtype.
            device: Target device (torch only). If None, uses array's device.
        """
        if shape is None:
            shape = array.shape
        if dtype is None:
            dtype = array.dtype
        if self.name == "numpy":
            return np.full(shape, fill_value, dtype=dtype)
        else:
            import torch

            if isinstance(shape, int):
                shape = (shape,)
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)
            return torch.full(
                shape,
                fill_value,
                dtype=dtype,
                device=self._resolve_torch_device(array, device),
            )

    def full(self, shape, fill_value, dtype=None):
        """Create array filled with *fill_value*.

        Args:
            shape: Output shape (int or tuple).
            fill_value: Scalar fill value.
            dtype: Output dtype. If None, inferred by the backend.
        """
        if self.name == "numpy":
            return np.full(shape, fill_value, dtype=dtype)
        else:
            import torch

            if isinstance(shape, int):
                shape = (shape,)
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)
            return torch.full(shape, fill_value, dtype=dtype, device=self._torch_device)

    # ------------------------------------------------------------------
    # Device transfer
    # ------------------------------------------------------------------

    def to_cpu(self, array):
        """Transfer array to CPU. No-op for numpy.

        Args:
            array: Input array or tensor.

        Returns:
            Array on CPU.
        """
        if self.name == "numpy":
            return array
        else:
            return array.cpu()

    def to_gpu(self, array, device=None):
        """Transfer array to GPU. No-op for numpy.

        Args:
            array: Input array or tensor.
            device: Target device (defaults to backend's device).

        Returns:
            Array on GPU device.
        """
        if self.name == "numpy":
            return array
        else:
            target = device or self._torch_device
            return (
                self.asarray(array, dtype=None).to(target)
                if not hasattr(array, "to")
                else array.to(target)
            )

    # ------------------------------------------------------------------
    # Compat ops (differ between numpy and torch)
    # ------------------------------------------------------------------

    def concatenate(self, arrays, axis=0):
        """Concatenate arrays along an axis.

        Args:
            arrays: Sequence of arrays.
            axis: Axis to concatenate along (default 0).
        """
        if self.name == "numpy":
            return np.concatenate(arrays, axis=axis)
        else:
            import torch

            return torch.cat(arrays, dim=axis)

    def expand_dims(self, array, axis):
        """Insert a new axis.

        Args:
            array: Input array.
            axis: Position of the new axis.
        """
        if self.name == "numpy":
            return np.expand_dims(array, axis=axis)
        else:
            import torch

            return torch.unsqueeze(array, dim=axis)

    def copy(self, array):
        """Return an independent copy of the array.

        Args:
            array: Input array.
        """
        if self.name == "numpy":
            return np.copy(array)
        else:
            return array.clone()

    def flatnonzero(self, array):
        """Return indices of non-zero elements in the flattened array.

        Args:
            array: Input array.
        """
        if self.name == "numpy":
            return np.flatnonzero(array)
        else:
            import torch

            return torch.nonzero(torch.flatten(array), as_tuple=True)[0]

    def sort(self, array, axis=-1):
        """Sort along an axis, returning values only.

        Args:
            array: Input array.
            axis: Axis to sort along (default -1).
        """
        if self.name == "numpy":
            return np.sort(array, axis=axis)
        else:
            import torch

            return torch.sort(array, dim=axis).values


def assert_array_almost_equal(x, y, decimal=6, err_msg="", verbose=True, backend=None):
    """Test array equality with automatic precision adjustment for MPS backend.

    This utility automatically reduces precision expectations for torch-mps backend
    due to float32 precision limitations, preventing test failures while maintaining
    realistic precision checks for other backends.

    Args:
        x: First array to compare
        y: Second array to compare
        decimal: Desired decimal precision (default: 6)
        err_msg: Error message prefix
        verbose: Whether to print detailed error messages
        backend: Backend instance (optional). If None, attempts to detect from x/y.

    Returns:
        None (raises AssertionError if arrays don't match)
    """
    # Auto-detect backend from x if possible
    if backend is None:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                if x.device.type == "mps":
                    backend_name = "torch-mps"
                else:
                    backend_name = None
            else:
                backend_name = None
        except (ImportError, AttributeError):
            backend_name = None
    else:
        backend_name = getattr(backend, "name", None)

    # Auto-adjust precision for torch_mps backend
    if backend_name == "torch-mps":
        if decimal > 2:
            import warnings

            warnings.warn(
                f"Reducing precision from decimal={decimal} to decimal=2 for "
                "torch-mps backend due to float32 conversion limitations",
                UserWarning,
            )
            decimal = 2

    # Convert to numpy if needed
    if backend is not None:
        x = backend.to_numpy(x) if hasattr(backend, "to_numpy") else x
        y = backend.to_numpy(y) if hasattr(backend, "to_numpy") else y
    else:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
            if isinstance(y, torch.Tensor):
                y = y.cpu().numpy()
        except (ImportError, AttributeError):
            pass

    return np.testing.assert_array_almost_equal(
        x, y, decimal=decimal, err_msg=err_msg, verbose=verbose
    )


def check_gpu_available() -> Tuple[bool, Dict[str, Any]]:
    """
    Check if GPU acceleration is available.

    Returns:
        tuple: (available, info) where:
            - available (bool): True if GPU (CUDA or MPS) is available
            - info (dict): Dictionary with keys:
                - 'backend': 'torch' or 'numpy'
                - 'device': 'cpu', 'cuda', or 'mps'
                - 'device_name': Human-readable device name
    """
    try:
        import torch

        if torch.cuda.is_available():
            return True, {
                "backend": "torch",
                "device": "cuda",
                "device_name": torch.cuda.get_device_name(0),
            }
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True, {
                "backend": "torch",
                "device": "mps",
                "device_name": "Apple Metal Performance Shaders",
            }
        else:
            return False, {
                "backend": "torch",
                "device": "cpu",
                "device_name": "CPU (PyTorch available)",
            }
    except ImportError:
        return False, {
            "backend": "numpy",
            "device": "cpu",
            "device_name": "CPU (NumPy only)",
        }


def auto_select_backend(n_samples: int, n_features: int, cv: int = 1) -> Backend:
    """
    Automatically select backend based on problem size.

    Uses heuristics to decide between NumPy (CPU) and PyTorch (GPU)
    based on the computational workload. Small problems use NumPy
    to avoid GPU transfer overhead. Large problems prefer GPU when
    available.

    Args:
        n_samples (int): Number of samples in dataset
        n_features (int): Number of features in dataset
        cv (int, default=1): Number of cross-validation folds (multiplies effective size)

    Returns:
        Backend: Selected backend instance

    Notes:
        Selection criteria:
        - Small problems (< 10M elements): Use NumPy
        - Large problems (> 30M elements): Use GPU if available
        - Cross-validation: Prefer GPU even for medium problems
    """
    # Compute effective problem size
    problem_size = n_samples * n_features * cv

    # Thresholds
    SMALL_THRESHOLD = 10_000_000  # 10M elements
    LARGE_THRESHOLD = 30_000_000  # 30M elements

    # Check GPU availability
    gpu_available, _ = check_gpu_available()

    # Decision logic
    if problem_size < SMALL_THRESHOLD:
        # Small problem: NumPy is efficient enough
        return Backend("numpy")
    elif problem_size > LARGE_THRESHOLD and gpu_available:
        # Large problem with GPU: Use PyTorch
        return Backend("torch")
    elif cv > 1 and gpu_available:
        # Cross-validation with GPU: Prefer PyTorch
        return Backend("torch")
    else:
        # Default: Try auto-selection (falls back to NumPy if no GPU)
        return Backend("auto")
