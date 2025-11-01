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
            # NumPy backend
            return np.linalg.svd(X, full_matrices=full_matrices)
        elif self.device == "mps":
            # MPS backend: explicit CPU fallback for better precision
            # PyTorch's MPS backend doesn't natively support SVD and falls back to CPU.
            # We make this explicit to avoid warnings and ensure consistent behavior.
            import torch

            X_device = X.device
            # Move to CPU and use float64 for better precision
            X_cpu = X.cpu().to(torch.float64)
            U, s, Vt = torch.linalg.svd(X_cpu, full_matrices=full_matrices)
            # Move results back to original device (MPS) as float32
            U = U.to(dtype=torch.float32, device=X_device)
            s = s.to(dtype=torch.float32, device=X_device)
            Vt = Vt.to(dtype=torch.float32, device=X_device)
            return U, s, Vt
        else:
            # PyTorch backend (CUDA or CPU)
            import torch

            U, s, Vt = torch.linalg.svd(X, full_matrices=full_matrices)
            return U, s, Vt

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
            # NumPy backend
            return A @ B
        else:
            # PyTorch backend
            import torch

            return torch.matmul(A, B)


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
