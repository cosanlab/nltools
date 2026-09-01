"""Backend abstraction for CPU/GPU operations.

Supports NumPy (CPU-only) and PyTorch (CPU/CUDA/MPS) backends for
linear algebra operations. Enables transparent acceleration while
maintaining NumPy-first development.
"""

import warnings
import numpy as np
from typing import Any

# Track if we've warned about MPS initialization to avoid spam
_already_warned_mps_init = [False]
# Track if we've warned about float64 conversion to avoid spam
_already_warned_float64 = [False]


class Backend:
    """Backend abstraction for numerical operations.

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
        """Initialize NumPy backend."""
        self.name = "numpy"
        self.device = "cpu"
        self.xp = np
        self._torch_device = None

    def _init_torch(self):
        """Initialize PyTorch backend with device detection."""
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
        """Automatically select best backend."""
        import importlib.util

        # Check if PyTorch is available without importing it
        if importlib.util.find_spec("torch") is not None:
            # PyTorch available, use it
            self._init_torch()
        else:
            # Fall back to NumPy
            self._init_numpy()

    def to_device(self, arr: np.ndarray):
        """Transfer array to backend device.

        Args:
            arr (np.ndarray): Input numpy array

        Returns:
            array: Array on device (numpy array or torch tensor)
        """
        if self.name == "numpy":
            # NumPy backend: ensure float32
            return arr.astype(np.float32)
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
        """Convert array back to NumPy.

        Args:
            arr (np.ndarray or torch.Tensor): Array to convert

        Returns:
            np.ndarray: NumPy array
        """
        if self.name == "numpy":
            # NumPy backend: identity operation
            return arr
        # PyTorch backend: move to CPU and convert
        import torch

        if isinstance(arr, torch.Tensor):
            return arr.cpu().numpy()
        return arr

    def svd(self, X, full_matrices=False):
        """Compute Singular Value Decomposition.

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
            if X.ndim == 3:
                UsV = [linalg.svd(Xi, full_matrices=full_matrices) for Xi in X]
                return tuple(map(np.stack, zip(*UsV)))
            raise NotImplementedError("SVD only supports 2D and 3D arrays")
        if self.device == "mps":
            import torch

            X_device = X.device
            X_cpu = X.cpu().to(torch.float64)
            U, s, Vt = torch.linalg.svd(X_cpu, full_matrices=full_matrices)
            U = U.to(dtype=torch.float32, device=X_device)
            s = s.to(dtype=torch.float32, device=X_device)
            Vt = Vt.to(dtype=torch.float32, device=X_device)
            return U, s, Vt
        import torch

        return torch.linalg.svd(X, full_matrices=full_matrices)

    def matmul(self, A, B):
        """Matrix multiplication.

        Args:
            A (array): First matrix
            B (array): Second matrix

        Returns:
            array: Result of A @ B
        """
        if self.name == "numpy":
            return A @ B
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
        import torch

        return torch.unsqueeze(array, dim=axis)

    def copy(self, array):
        """Return an independent copy of the array.

        Args:
            array: Input array.
        """
        if self.name == "numpy":
            return np.copy(array)
        return array.clone()

    def flatnonzero(self, array):
        """Return indices of non-zero elements in the flattened array.

        Args:
            array: Input array.
        """
        if self.name == "numpy":
            return np.flatnonzero(array)
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
        import torch

        return torch.sort(array, dim=axis).values


def resolve_backend(parallel):
    """Coerce a backend specifier into a `Backend` instance.

    Accepts the values callers typically thread through the algorithms
    package (``None``/``"cpu"`` → numpy, ``"gpu"``/``"torch"`` → torch,
    ``"numpy"``/``"auto"`` → their direct `Backend` constructors).
    Existing `Backend` instances are returned unchanged — this is
    the main reason to prefer ``resolve_backend`` over constructing a new
    ``Backend(...)`` at each call site: it avoids repeated device
    detection/torch imports when a backend has already been chosen upstream.

    Args:
        parallel: Backend specifier. One of:

            - ``None`` or ``"cpu"``: numpy backend.
            - ``"numpy"``, ``"torch"``, ``"auto"``: forwarded to ``Backend(...)``.
            - ``"gpu"``: alias for ``"torch"`` (auto-detects cuda/mps/cpu).
            - An existing `Backend` instance (returned as-is).

    Returns:
        Backend: Resolved backend instance.

    Raises:
        ValueError: If ``parallel`` is a string not in the accepted set.
    """
    if isinstance(parallel, Backend):
        return parallel
    if parallel in (None, "cpu"):
        return Backend("numpy")
    if parallel == "gpu":
        return Backend("torch")
    if parallel in ("numpy", "torch", "auto"):
        return Backend(parallel)
    raise ValueError(
        f"parallel must be None, 'cpu', 'gpu', 'numpy', 'torch', 'auto', "
        f"or a Backend instance; got: {parallel!r}"
    )


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


def check_gpu_available() -> tuple[bool, dict[str, Any]]:
    """Check if GPU acceleration is available.

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
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True, {
                "backend": "torch",
                "device": "mps",
                "device_name": "Apple Metal Performance Shaders",
            }
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
    """Automatically select backend based on problem size.

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
    if problem_size > LARGE_THRESHOLD and gpu_available:
        # Large problem with GPU: Use PyTorch
        return Backend("torch")
    if cv > 1 and gpu_available:
        # Cross-validation with GPU: Prefer PyTorch
        return Backend("torch")
    # Default: Try auto-selection (falls back to NumPy if no GPU)
    return Backend("auto")


# ----------------------------------------------------------------------
# Core memory / batching layer
#
# Single source of truth for how nltools sizes device work: measured
# memory budgets, one batch-size calculator, and reactive OOM recovery.
# Every GPU/batched code path in the package must budget through these
# helpers rather than hard-coding constants or rolling its own math.
# ----------------------------------------------------------------------

_FALLBACK_BUDGET_GB = 4.0
# Fraction of free CUDA memory a computation may claim.
_CUDA_HEADROOM = 0.8
# Fraction of available system RAM for CPU work and MPS (unified memory).
_SYSTEM_HEADROOM = 0.5


def device_memory_budget(
    backend: "Backend | None" = None,
    max_gpu_memory_gb: float | None = None,
) -> float:
    """Usable memory budget in GB for a backend's device.

    An explicit `max_gpu_memory_gb` always wins. Otherwise the budget is
    measured at call time: free CUDA memory (with headroom) on CUDA
    devices; available system RAM (with headroom) for CPU and MPS, which
    share unified/system memory. When nothing can be measured the
    conservative 4 GB fallback applies.

    Args:
        backend: Resolved `Backend` whose device the work runs on. None is
            treated as CPU.
        max_gpu_memory_gb: Explicit budget override in GB. Must be positive.

    Returns:
        float: Budget in GB.

    Raises:
        ValueError: If `max_gpu_memory_gb` is not positive.
    """
    if max_gpu_memory_gb is not None:
        if max_gpu_memory_gb <= 0:
            raise ValueError(
                f"max_gpu_memory_gb must be positive, got {max_gpu_memory_gb!r}"
            )
        return float(max_gpu_memory_gb)
    if getattr(backend, "device", "cpu") == "cuda":
        try:
            import torch

            free_bytes, _ = torch.cuda.mem_get_info()
            return free_bytes * _CUDA_HEADROOM / 1e9
        except Exception:  # pragma: no cover - depends on driver state
            pass
    try:
        import psutil

        return psutil.virtual_memory().available * _SYSTEM_HEADROOM / 1e9
    except ImportError:  # pragma: no cover - psutil ships with the dev env
        return _FALLBACK_BUDGET_GB


def gb_to_bytes(gb: float) -> int:
    """Convert a GB budget to bytes — the package's one GB↔bytes conversion."""
    return int(gb * 1e9)


def auto_batch_size(
    n_items: int,
    bytes_per_item: float,
    *,
    budget_gb: float,
    overhead: float = 1.0,
    min_batch: int = 1,
) -> tuple[int, int]:
    """Split `n_items` into batches that fit a memory budget.

    The one batch calculator for the package. Callers supply only the
    per-item working-set estimate (`bytes_per_item`) and an algorithm's
    allocation `overhead` factor; the clamp/ceil policy lives here.

    Args:
        n_items: Total number of items (permutations, targets, ...).
        bytes_per_item: Dominant working-set size of one item in bytes.
        budget_gb: Memory budget from `device_memory_budget`.
        overhead: Multiplier for intermediate allocations (e.g. 3.0 when
            the computation holds ~3x the input working set).
        min_batch: Smallest batch worth dispatching (amortizes launch and
            transfer overhead). Never exceeds `n_items`.

    Returns:
        tuple[int, int]: `(batch_size, n_batches)` with
        `batch_size * n_batches >= n_items`.
    """
    if n_items <= 0:
        raise ValueError(f"n_items must be positive, got {n_items}")
    per_item = bytes_per_item * overhead
    if per_item <= 0:
        batch_size = n_items
    else:
        batch_size = int(gb_to_bytes(budget_gb) / per_item)
    batch_size = min(max(batch_size, min_batch), n_items)
    n_batches = int(np.ceil(n_items / batch_size))
    return batch_size, n_batches


def is_oom_error(exc: BaseException) -> bool:
    """True if `exc` is a device out-of-memory error (CUDA or MPS)."""
    try:
        import torch

        if hasattr(torch, "OutOfMemoryError") and isinstance(
            exc, torch.OutOfMemoryError
        ):
            return True
    except ImportError:
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def empty_device_cache() -> None:
    """Release cached device memory. No-op without torch or a GPU."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch, "mps")
    ):
        torch.mps.empty_cache()


def compute_oom_safe(fn, *arrays, min_chunk: int = 1):
    """Run `fn(*arrays)` with reactive out-of-memory recovery.

    All `arrays` must share their axis-0 length, and `fn` must map them to
    a numpy array whose axis 0 corresponds row-for-row to its inputs. On a
    device OOM the cache is emptied, the arrays are split in half along
    axis 0, and the halves are retried recursively; partial results are
    concatenated along axis 0.

    Because splitting reuses the *already generated* inputs rather than
    re-drawing them, recovery never changes which permutations a seeded
    result is computed from — RNG-consuming input generation stays outside
    this function. For a row-independent `fn` the recovered output matches
    the unsplit computation to within floating-point reduction order
    (backends may block reductions differently per batch shape; observed
    differences are ~1 float32 ulp).

    Args:
        fn: Callable mapping the arrays to a numpy result (axis-0 aligned).
        *arrays: Input arrays sharing axis-0 length.
        min_chunk: Chunk size below which an OOM is considered fatal.

    Returns:
        np.ndarray: `fn`'s result, possibly assembled from retried chunks.

    Raises:
        MemoryError: If the device OOMs even at `min_chunk` items.
    """
    n = len(arrays[0])
    try:
        return fn(*arrays)
    except Exception as exc:
        if not is_oom_error(exc):
            raise
        empty_device_cache()
        if n <= min_chunk:
            raise MemoryError(
                f"Device out of memory even for a single item (chunk of {n}). "
                "Reduce the problem size, lower max_gpu_memory_gb elsewhere on "
                "the device, or use device='cpu'."
            ) from exc
        mid = n // 2
        left = compute_oom_safe(fn, *(a[:mid] for a in arrays), min_chunk=min_chunk)
        right = compute_oom_safe(fn, *(a[mid:] for a in arrays), min_chunk=min_chunk)
        return np.concatenate([left, right], axis=0)


# ----------------------------------------------------------------------
# CPU worker sizing (joblib) — same budget source as the device batching
# ----------------------------------------------------------------------


def _auto_n_jobs_cpu(
    data_size_mb: float,
    n_permute: int,
    max_memory_gb: float | None = None,
    min_jobs: int = 1,
    max_jobs: int | None = None,
) -> int:
    """Automatically determine optimal number of CPU workers to avoid memory exhaustion.

    Calculates how many parallel workers can safely process permutations given
    available memory. Each worker process needs to serialize (pickle) data,
    which typically requires 2-4× the original data size in memory.

    Args:
        data_size_mb (float): Size of data array in MB (float32: 4 bytes per element)
        n_permute (int): Number of permutations to compute
        max_memory_gb (float, optional): Explicit memory budget in GB. None
            (default) measures available system RAM with headroom via
            `device_memory_budget`.
        min_jobs (int): Minimum number of workers (default: 1)
        max_jobs (int, optional): Maximum number of workers (default: None = all cores)

    Returns:
        int: Optimal number of workers (n_jobs parameter for joblib.Parallel)

    Examples:
        >>> # Small data: Use all cores
        >>> n_jobs = _auto_n_jobs_cpu(1.0, 5000, max_memory_gb=8.0)
        >>> n_jobs >= 4  # Should use multiple cores
        True

        >>> # Large data: Limit workers
        >>> n_jobs = _auto_n_jobs_cpu(100.0, 5000, max_memory_gb=8.0)
        >>> n_jobs < 8  # Should limit workers
        True

    Notes:
        - Accounts for joblib serialization overhead (3× multiplier)
        - Leaves 50% headroom for OS and other processes
        - Minimum 1 worker, maximum all available cores (unless max_jobs specified)
        - Uses available RAM if max_memory_gb is None
    """
    import multiprocessing

    # Get system limits
    if max_jobs is None:
        max_jobs = multiprocessing.cpu_count()

    available_memory_gb = device_memory_budget(None, max_gpu_memory_gb=max_memory_gb)
    available_memory_mb = available_memory_gb * 1024

    # Memory per worker: data serialization overhead (3× is conservative for pickle)
    # Plus small overhead for result arrays (n_permute results per worker)
    serialization_factor = 3.0
    result_overhead_mb = (n_permute * 4 / 1024**2) * 0.1  # ~10% overhead estimate
    memory_per_worker_mb = data_size_mb * serialization_factor + result_overhead_mb

    # How many workers can fit in memory budget?
    if memory_per_worker_mb <= 0:
        return min_jobs

    max_workers_by_memory = int(available_memory_mb / memory_per_worker_mb)
    max_workers_by_memory = max(min_jobs, min(max_workers_by_memory, max_jobs))

    # Use at least min_jobs, but don't exceed memory budget
    optimal_n_jobs = max(min_jobs, min(max_workers_by_memory, max_jobs))

    return optimal_n_jobs


def _estimate_data_size_mb(data: np.ndarray) -> float:
    """Estimate memory size of data array in MB.

    Accounts for numpy array overhead and dtype.

    Args:
        data (np.ndarray): Data array

    Returns:
        float: Estimated size in MB
    """
    if data.size == 0:
        return 0.0

    # Base size: elements × bytes per element
    bytes_per_element = data.dtype.itemsize
    base_size_bytes = data.size * bytes_per_element

    # Add numpy array overhead (typically ~100 bytes)
    overhead_bytes = 100

    total_size_mb = (base_size_bytes + overhead_bytes) / 1024**2

    return total_size_mb


def auto_n_jobs_for_arrays(
    arrays,
    *,
    max_memory_gb: float | None = None,
    min_jobs: int = 1,
) -> int:
    """Memory-aware joblib worker count for a per-item map over arrays.

    Sizes workers by the largest item (each worker pickles its item), using
    the same measured budget as the device batching layer. None entries are
    ignored; an empty list returns ``min_jobs``.

    Args:
        arrays: Iterable of numpy arrays (None entries allowed).
        max_memory_gb: Explicit memory budget in GB. None (default) measures
            available system RAM with headroom via `device_memory_budget`.
        min_jobs: Minimum number of workers (default: 1).

    Returns:
        int: Worker count for ``joblib.Parallel(n_jobs=...)``.
    """
    arrays = [a for a in arrays if a is not None]
    if not arrays:
        return min_jobs
    max_size_mb = max(_estimate_data_size_mb(a) for a in arrays)
    return _auto_n_jobs_cpu(
        data_size_mb=max_size_mb,
        n_permute=len(arrays),
        max_memory_gb=max_memory_gb,
        min_jobs=min_jobs,
    )
