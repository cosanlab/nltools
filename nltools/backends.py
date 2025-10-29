"""
Backend abstraction for CPU/GPU operations.

Supports NumPy (CPU-only) and PyTorch (CPU/CUDA/MPS) backends for
linear algebra operations. Enables transparent acceleration while
maintaining NumPy-first development.
"""

import numpy as np
from typing import Tuple, Dict, Any


class Backend:
    """
    Backend abstraction for numerical operations.

    Provides a unified interface for NumPy and PyTorch operations,
    enabling transparent GPU acceleration when available.

    Parameters
    ----------
    backend : str
        Backend type: 'numpy', 'torch', or 'auto'
        - 'numpy': CPU-only using NumPy
        - 'torch': PyTorch with automatic device detection (cuda/mps/cpu)
        - 'auto': Automatically select best available backend

    Attributes
    ----------
    name : str
        Backend identifier (e.g., 'numpy', 'torch-cuda', 'torch-mps')
    device : str
        Device type ('cpu', 'cuda', or 'mps')
    xp : module
        Array library module (numpy or torch)
    """

    def __init__(self, backend: str = 'numpy'):
        if backend == 'numpy':
            self._init_numpy()
        elif backend == 'torch':
            self._init_torch()
        elif backend == 'auto':
            self._init_auto()
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'numpy', 'torch', or 'auto'")

    def _init_numpy(self):
        """Initialize NumPy backend"""
        self.name = 'numpy'
        self.device = 'cpu'
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
            self.device = 'cuda'
            self._torch_device = torch.device('cuda')
            self.name = 'torch-cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
            self._torch_device = torch.device('mps')
            self.name = 'torch-mps'
        else:
            self.device = 'cpu'
            self._torch_device = torch.device('cpu')
            self.name = 'torch-cpu'

    def _init_auto(self):
        """Automatically select best backend"""
        try:
            import torch
            # PyTorch available, use it
            self._init_torch()
        except ImportError:
            # Fall back to NumPy
            self._init_numpy()

    def to_device(self, arr: np.ndarray):
        """
        Transfer array to backend device.

        Parameters
        ----------
        arr : np.ndarray
            Input numpy array

        Returns
        -------
        array
            Array on device (numpy array or torch tensor)
        """
        if self.name == 'numpy':
            # NumPy backend: ensure float32
            return arr.astype(np.float32)
        else:
            # PyTorch backend: convert to tensor and move to device
            import torch
            tensor = torch.from_numpy(arr.astype(np.float32))
            return tensor.to(self._torch_device)

    def to_numpy(self, arr):
        """
        Convert array back to NumPy.

        Parameters
        ----------
        arr : np.ndarray or torch.Tensor
            Array to convert

        Returns
        -------
        np.ndarray
            NumPy array
        """
        if self.name == 'numpy':
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

        Parameters
        ----------
        X : array
            Input matrix (n_samples, n_features)
        full_matrices : bool, default=False
            If False, returns reduced SVD

        Returns
        -------
        U : array
            Left singular vectors
        s : array
            Singular values
        Vt : array
            Right singular vectors (transposed)
        """
        if self.name == 'numpy':
            # NumPy backend
            return np.linalg.svd(X, full_matrices=full_matrices)
        else:
            # PyTorch backend
            import torch
            U, s, Vt = torch.linalg.svd(X, full_matrices=full_matrices)
            return U, s, Vt

    def matmul(self, A, B):
        """
        Matrix multiplication.

        Parameters
        ----------
        A : array
            First matrix
        B : array
            Second matrix

        Returns
        -------
        array
            Result of A @ B
        """
        if self.name == 'numpy':
            # NumPy backend
            return A @ B
        else:
            # PyTorch backend
            import torch
            return torch.matmul(A, B)


def check_gpu_available() -> Tuple[bool, Dict[str, Any]]:
    """
    Check if GPU acceleration is available.

    Returns
    -------
    available : bool
        True if GPU (CUDA or MPS) is available
    info : dict
        Dictionary with keys:
        - 'backend': 'torch' or 'numpy'
        - 'device': 'cpu', 'cuda', or 'mps'
        - 'device_name': Human-readable device name
    """
    try:
        import torch

        if torch.cuda.is_available():
            return True, {
                'backend': 'torch',
                'device': 'cuda',
                'device_name': torch.cuda.get_device_name(0)
            }
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return True, {
                'backend': 'torch',
                'device': 'mps',
                'device_name': 'Apple Metal Performance Shaders'
            }
        else:
            return False, {
                'backend': 'torch',
                'device': 'cpu',
                'device_name': 'CPU (PyTorch available)'
            }
    except ImportError:
        return False, {
            'backend': 'numpy',
            'device': 'cpu',
            'device_name': 'CPU (NumPy only)'
        }


def auto_select_backend(n_samples: int, n_features: int, cv: int = 1) -> Backend:
    """
    Automatically select backend based on problem size.

    Uses heuristics to decide between NumPy (CPU) and PyTorch (GPU)
    based on the computational workload. Small problems use NumPy
    to avoid GPU transfer overhead. Large problems prefer GPU when
    available.

    Parameters
    ----------
    n_samples : int
        Number of samples in dataset
    n_features : int
        Number of features in dataset
    cv : int, default=1
        Number of cross-validation folds (multiplies effective size)

    Returns
    -------
    Backend
        Selected backend instance

    Notes
    -----
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
        return Backend('numpy')
    elif problem_size > LARGE_THRESHOLD and gpu_available:
        # Large problem with GPU: Use PyTorch
        return Backend('torch')
    elif cv > 1 and gpu_available:
        # Cross-validation with GPU: Prefer PyTorch
        return Backend('torch')
    else:
        # Default: Try auto-selection (falls back to NumPy if no GPU)
        return Backend('auto')
