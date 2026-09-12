"""Backend abstraction for CPU/GPU operations.

Supports NumPy (CPU-only) and PyTorch (CPU/CUDA/MPS) backends for linear algebra
operations, so algorithms are written once against NumPy semantics and run on a
GPU when one is available.

This module is also the package's single GPU execution layer: memory budgets
(`device_memory_budget`), batch sizing (`auto_batch_size`), out-of-memory
recovery (`compute_oom_safe`), and CPU worker sizing (`_auto_n_jobs_cpu`)
live only here. Algorithms supply per-item working-set estimates and never do
their own budget math.
"""

import warnings
from copy import deepcopy
import numpy as np
from typing import Any

from nltools.utils import find_stack_level

# Track if we've warned about MPS initialization to avoid spam
_already_warned_mps_init = [False]
# Track if we've warned about float64 conversion to avoid spam


def _array_module_for(name):
    """Return the array module a pickled backend name needs, if it is usable here.

    Args:
        name (str | None): A backend name — `'numpy'`, `'torch-cpu'`,
            `'torch-cuda'`, or `'torch-mps'`.

    Returns:
        module | None: `numpy` or `torch`, or None when the named device is not
            available in this process.
    """
    if name == "numpy":
        return np
    try:
        import torch
    except ImportError:
        return None
    if name == "torch-cuda":
        return torch if torch.cuda.is_available() else None
    if name == "torch-mps":
        available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return torch if available else None
    if name == "torch-cpu":
        return torch
    return None


class Backend:
    """Backend abstraction for numerical operations.

    Provides a unified interface for NumPy and PyTorch operations, enabling
    transparent GPU acceleration when available.

    Args:
        backend (str): Backend type. `'numpy'` is CPU-only NumPy; `'torch'` is
            PyTorch with automatic device detection (cuda, then mps, then cpu);
            `'auto'` picks `'torch'` when PyTorch is installed and `'numpy'`
            otherwise. Defaults to `'numpy'`.

    Attributes:
        name (str): Backend identifier: `'numpy'`, `'torch-cpu'`, `'torch-cuda'`,
            or `'torch-mps'`.
        device (str): Device type: `'cpu'`, `'cuda'`, or `'mps'`.
        xp (module): Array library module (`numpy` or `torch`).
        is_gpu (bool): True when the device is a GPU (`'cuda'` or `'mps'`).
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

    def __deepcopy__(self, memo):
        """Copy backend state without trying to pickle its array module."""
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        for name, value in self.__dict__.items():
            setattr(copied, name, value if name == "xp" else deepcopy(value, memo))
        return copied

    def __getstate__(self):
        """Drop the array module, which is a live module object and unpicklable.

        `backend_` is public fitted state on `Ridge`, so a fitted model has to
        survive `pickle` — `BrainData.copy()` and any process-based `n_jobs`
        worker carry one across. Everything else on a backend is a plain string
        or a `torch.device`, both of which pickle fine; `xp` is recovered by
        name in `__setstate__`.
        """
        state = dict(self.__dict__)
        state.pop("xp", None)
        return state

    def __setstate__(self, state):
        """Restore the descriptor as pickled, and the array module if usable.

        `name` and `device` record the device the model was *fitted* on and are
        restored verbatim: unpickling never re-resolves them, because silently
        turning a CUDA-fitted model into an MPS one would break the
        run-or-raise rule. When that device is not available in this process
        `xp` stays unset, so any attempt to compute through this backend raises
        from `__getattr__` instead of running somewhere else.
        """
        self.__dict__.update(state)
        module = _array_module_for(state.get("name"))
        if module is not None:
            self.xp = module

    def __getattr__(self, name):
        """Explain a missing array module rather than raising a bare AttributeError.

        Only reached when normal lookup fails, which for `xp` means this backend
        was unpickled on a host without the device it was fitted on. Pinned by
        `test_ridge.py::TestSerialization::test_gpu_fitted_model_round_trips_through_pickle`
        and `::test_unpickling_never_switches_device`.

        Args:
            name (str): The attribute being looked up.

        Raises:
            RuntimeError: If `xp` is missing because the device is unavailable.
            AttributeError: For any other missing attribute.
        """
        if name == "xp" and "name" in self.__dict__:
            raise RuntimeError(
                f"This backend was fitted on device "
                f"{self.__dict__.get('device')!r} ({self.__dict__['name']}), "
                "which is not available in this process. Refit on an available "
                "device, or construct a new backend with device='cpu'."
            )
        raise AttributeError(name)

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
                    stacklevel=find_stack_level(),
                )
                _already_warned_mps_init[0] = True
        else:
            self.device = "cpu"
            self._torch_device = torch.device("cpu")
            self.name = "torch-cpu"

    @property
    def is_gpu(self):
        """True when the resolved device is a GPU (`'cuda'` or `'mps'`)."""
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

    def to_numpy(self, arr):
        """Convert an array back to NumPy.

        Args:
            arr (np.ndarray | torch.Tensor): Array to convert.

        Returns:
            np.ndarray: The input as a NumPy array.
        """
        if self.name == "numpy":
            # NumPy backend: identity operation
            return arr
        # PyTorch backend: move to CPU and convert
        import torch

        if isinstance(arr, torch.Tensor):
            return arr.cpu().numpy()
        return arr

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def dtype_to_str(dtype):
        """Normalize a dtype (numpy, torch, or string) to its string name.

        Args:
            dtype (str | np.dtype | type | torch.dtype | None): Data type to convert.

        Returns:
            str | None: The dtype name (e.g. `"float32"`, `"float64"`), or None if
                the input was None.

        Raises:
            NotImplementedError: If the input cannot be interpreted as a dtype.
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

        Handles numpy arrays, lists, and torch tensors. Places the result on
        the backend's device (or an explicit `device`). On MPS a float64 dtype
        is replaced by float32, which is all the device supports.

        Args:
            x (array-like | torch.Tensor | list): Input data.
            dtype (str | np.dtype | torch.dtype | None): Desired dtype. If None,
                inferred from the input.
            device (str | torch.device | None): Target device (e.g. `"cpu"`,
                `"cuda"`). Ignored for the numpy backend. If None, uses the
                backend's default device.

        Returns:
            np.ndarray | torch.Tensor: Backend array.
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

    # ------------------------------------------------------------------
    # Array creation with shape override
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Device transfer
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Compat ops (differ between numpy and torch)
    # ------------------------------------------------------------------


def resolve_backend(parallel):
    """Coerce a backend specifier into a `Backend` instance.

    Accepts the values callers thread through the algorithms package. An
    existing `Backend` is returned unchanged, which is the reason to prefer
    this over constructing `Backend(...)` at each call site: device detection
    and the torch import happen once, upstream.

    Args:
        parallel (str | Backend | None): Backend specifier. `None` or `"cpu"`
            gives the numpy backend; `"gpu"` requires CUDA or MPS; `"numpy"`,
            `"torch"`, and `"auto"` are passed to `Backend(...)`; a `Backend`
            instance is returned as-is.

    Returns:
        Backend: Resolved backend instance.

    Raises:
        ValueError: If `parallel` is a string outside the accepted set.
        RuntimeError: If `parallel="gpu"` and no accelerator is available.
    """
    if isinstance(parallel, Backend):
        return parallel
    if parallel in (None, "cpu"):
        return Backend("numpy")
    if parallel == "gpu":
        backend = Backend("torch")
        if not backend.is_gpu:
            raise RuntimeError(
                "GPU requested explicitly, but no GPU accelerator is available. "
                "Use 'cpu' or 'auto' to allow CPU execution."
            )
        return backend
    if parallel in ("numpy", "torch", "auto"):
        return Backend(parallel)
    raise ValueError(
        f"parallel must be None, 'cpu', 'gpu', 'numpy', 'torch', 'auto', "
        f"or a Backend instance; got: {parallel!r}"
    )


def check_gpu_available() -> tuple[bool, dict[str, Any]]:
    """Check whether GPU acceleration is available.

    Returns:
        tuple[bool, dict[str, Any]]: `(available, info)`. `available` is True when
            a CUDA or MPS device is usable. `info` has keys `'backend'`
            (`'torch'` or `'numpy'`), `'device'` (`'cpu'`, `'cuda'`, or `'mps'`),
            and `'device_name'` (human-readable device name).
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
    """Select a backend from the problem size.

    Small problems stay on NumPy to avoid GPU transfer overhead; large problems
    prefer the GPU when one is available. The effective size is
    `n_samples * n_features * cv`.

    Args:
        n_samples (int): Number of samples in the dataset.
        n_features (int): Number of features in the dataset.
        cv (int): Number of cross-validation folds, which multiplies the effective
            size. Defaults to 1.

    Returns:
        Backend: The selected backend.

    Note:
        Below 10M elements the numpy backend is returned. Above 30M elements the
        torch backend is returned when a GPU is available, as it is for any
        cross-validated problem (`cv > 1`) with a GPU. Everything else falls to
        `Backend('auto')`.
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
BATCH_WORKING_SET_CEILING_GB = 8.0
"""Saturation ceiling for batch sizing, in GB of per-batch working set.

Batches beyond this compute no faster (GPU kernels saturate at moderate working
sets) but add allocation latency and, on unified-memory systems, starve the host:
a measured ~100 GB budget on a 128 GB GB10 sized ~100 GB ISC batches, 4.2s → 16.1s
and a wedged machine. The cap applies only to *measured* budgets — an explicit
`max_gpu_memory_gb` is the documented contract and always wins, uncapped.
Capacity reasoning (OOM recovery, single-item-too-large errors) still uses the
true measured budget. See `device_memory_budget(cap_for_batching=True)`.
"""


def device_memory_budget(
    backend: "Backend | None" = None,
    max_gpu_memory_gb: float | None = None,
    *,
    cap_for_batching: bool = False,
) -> float:
    """Usable memory budget in GB for a backend's device.

    An explicit `max_gpu_memory_gb` always wins, uncapped. Otherwise the
    budget is measured at call time: free CUDA memory (with headroom) on
    CUDA devices; available system RAM (with headroom) for CPU and MPS,
    which share unified/system memory. When nothing can be measured the
    conservative 4 GB fallback applies.

    Args:
        backend (Backend | None): Resolved backend whose device the work runs
            on. None is treated as CPU.
        max_gpu_memory_gb (float | None): Explicit budget override in GB. Must
            be positive.
        cap_for_batching (bool): Pass True when the budget sizes batches — a
            *measured* budget is then capped at `BATCH_WORKING_SET_CEILING_GB`,
            because working sets beyond the saturation ceiling add allocation
            cost without throughput gain and starve unified-memory hosts. Never
            applied to an explicit `max_gpu_memory_gb`; capacity queries (the
            default) stay uncapped.

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
    measured = None
    if getattr(backend, "device", "cpu") == "cuda":
        try:
            import torch

            free_bytes, _ = torch.cuda.mem_get_info()
            measured = free_bytes * _CUDA_HEADROOM / 1e9
        except Exception:  # pragma: no cover - depends on driver state
            pass
    if measured is None:
        try:
            import psutil

            measured = psutil.virtual_memory().available * _SYSTEM_HEADROOM / 1e9
        except ImportError:  # pragma: no cover - psutil ships with the dev env
            measured = _FALLBACK_BUDGET_GB
    if cap_for_batching:
        return min(measured, BATCH_WORKING_SET_CEILING_GB)
    return measured


def gb_to_bytes(gb: float) -> int:
    """Convert a GB budget to bytes — the package's one GB↔bytes conversion."""
    return int(gb * 1e9)


#: Allowance for the transient buffers Himalaya's fixed-hyperparameter solve
#: holds beyond the resampled design and response themselves.
_RIDGE_BOOTSTRAP_SOLVER_OVERHEAD = 3.0


def auto_batch_size(
    n_items: int,
    bytes_per_item: float,
    *,
    budget_gb: float,
    overhead: float = 1.0,
) -> tuple[int, int]:
    """Split `n_items` into batches that fit a memory budget.

    The one batch calculator for the package. Callers supply only the
    per-item working-set estimate (`bytes_per_item`) and an algorithm's
    allocation `overhead` factor; the clamp/ceil policy lives here.

    Args:
        n_items (int): Total number of items (permutations, targets, ...).
        bytes_per_item (float): Dominant working-set size of one item in bytes.
        budget_gb (float): Memory budget from `device_memory_budget`.
        overhead (float): Multiplier for intermediate allocations (e.g. 3.0 when
            the computation holds ~3x the input working set). Defaults to 1.0.

    Returns:
        tuple[int, int]: `(batch_size, n_batches)` with
            `batch_size * n_batches >= n_items`.

    Raises:
        ValueError: If the inputs are invalid or one item exceeds the budget.
    """
    if n_items <= 0:
        raise ValueError(f"n_items must be positive, got {n_items}")
    if budget_gb <= 0:
        raise ValueError(f"budget_gb must be positive, got {budget_gb}")
    per_item = bytes_per_item * overhead
    if per_item <= 0:
        batch_size = n_items
    else:
        capacity = int(gb_to_bytes(budget_gb) / per_item)
        if capacity < 1:
            raise ValueError(
                f"one item requires {per_item / 1e9:.6g} GB, exceeding the "
                f"{budget_gb:.6g} GB memory budget"
            )
        batch_size = min(capacity, n_items)
    n_batches = int(np.ceil(n_items / batch_size))
    return batch_size, n_batches


def ridge_bootstrap_batch_size(
    n_bootstrap: int,
    *,
    n_samples: int,
    n_features: int,
    n_targets: int,
    output_shape: tuple[int, ...],
    device_itemsize: int = 4,
    max_gpu_memory_gb: float | None = None,
    backend=None,
) -> tuple[int, int]:
    """Size a Ridge-bootstrap batch against a memory budget.

    Models what a batch of replicates actually holds. Each replicate solves on
    its own, so device residency is one replicate's resampled design
    `(n_samples, n_features)` and response `(n_samples, n_targets)` with an
    allowance for the solver's own buffers. What accumulates across a batch is
    the host-side result list: `batch_size` float64 arrays of `output_shape`,
    which for a prediction bootstrap is sized by an arbitrary `X_test` row count
    and can dominate everything else. Both terms are charged per replicate so
    the batch cannot outgrow the budget it was given.

    Args:
        n_bootstrap (int): Total number of bootstrap replicates.
        n_samples (int): Observations in the training data.
        n_features (int): Total feature count across all feature spaces.
        n_targets (int): Number of targets (voxels).
        output_shape (tuple[int, ...]): Shape of one retained replicate result.
        device_itemsize (int): Bytes per element of the solver's working dtype
            (4 on MPS, 8 elsewhere). Defaults to 4.
        max_gpu_memory_gb (float | None): Explicit budget in GB, or None to
            measure the device.
        backend (Backend | None): Resolved backend, used only to measure the
            budget when `max_gpu_memory_gb` is None.

    Returns:
        tuple[int, int]: `(batch_size, n_batches)`.
    """
    budget_gb = device_memory_budget(
        backend, max_gpu_memory_gb=max_gpu_memory_gb, cap_for_batching=True
    )
    resident = (
        (n_samples * n_features + n_samples * n_targets)
        * device_itemsize
        * _RIDGE_BOOTSTRAP_SOLVER_OVERHEAD
    )
    retained = int(np.prod(output_shape)) * 8  # host float64 accumulation
    return auto_batch_size(
        n_bootstrap, resident + retained, budget_gb=budget_gb, overhead=1.0
    )


#: Bytes per retained bootstrap output value. Every completed replicate and
#: every summary payload is converted to CPU float64 before it is retained.
_BOOTSTRAP_OUTPUT_ITEMSIZE = 8

#: Output-sized arrays a bootstrap run always holds beyond its retained
#: replicates: the two Welford accumulators (running mean and running sum of
#: squared deviations) and the four `BootstrapResult` summary payloads.
_BOOTSTRAP_FIXED_OUTPUT_ARRAYS = 6

#: Replicates the streaming accumulator buffers before folding them into its
#: bounded tails. Batching the partition keeps the per-replicate cost near a
#: plain comparison, but the buffer and the two temporaries a flush creates are
#: real output-sized allocations — so the constant lives here, with the budget
#: that has to charge for it, and the accumulator imports it. It doubles as the
#: per-worker dispatch window (`bootstrap_replicate_window`).
BOOTSTRAP_TAIL_FLUSH_BLOCK = 64


def bootstrap_replicate_window(n_samples: int, *, n_workers: int = 1) -> int:
    """Replicates a CPU bootstrap may hold in flight before it must aggregate.

    `joblib.Parallel` dispatches eagerly and queues finished results, so neither
    `pre_dispatch` nor `return_as="generator"` bounds how many replicate arrays
    are alive at once. The engines therefore dispatch in windows of this size
    and fold each window into the accumulator before opening the next. The
    window scales with the worker count — enough to keep every worker busy —
    and never with `n_samples`, which is what makes peak memory independent of
    the replicate count.

    Args:
        n_samples (int): Total number of bootstrap replicates.
        n_workers (int): Planned CPU worker count.

    Returns:
        int: Replicates per dispatch window, at least one.

    Examples:
        ```python
        bootstrap_replicate_window(5000, n_workers=4)  # → 256
        bootstrap_replicate_window(50, n_workers=4)  # → 50
        ```
    """
    per_worker = BOOTSTRAP_TAIL_FLUSH_BLOCK * max(1, int(n_workers))
    return max(1, min(int(n_samples), per_worker))


def bootstrap_retained_tail_size(n_samples: int, *, confidence_level: float) -> int:
    """Per-element retained tail size for a streaming percentile interval.

    The streaming accumulator reproduces the complete-distribution percentile
    interval by keeping this many of the smallest and largest values seen for
    each output element:

    ```text
    k = ceil((B - 1) * (1 - c) / 2) + 1
    ```

    That is exactly the number of order statistics NumPy's linear interpolation
    can reach at either end, so nothing the interval needs is discarded.

    Args:
        n_samples (int): Number of bootstrap replicates, `B`.
        confidence_level (float): Interval confidence level, `c`, in `(0, 1)`.

    Returns:
        int: Values retained per element at each end, never more than
            `n_samples`.

    Examples:
        ```python
        bootstrap_retained_tail_size(1000, confidence_level=0.95)  # → 26
        ```
    """
    half_alpha = (1 - confidence_level) / 2
    k = int(np.ceil((n_samples - 1) * half_alpha)) + 1
    return min(k, int(n_samples))


def bootstrap_output_bytes(
    output_shape: tuple[int, ...],
    n_samples: int,
    *,
    confidence_level: float,
    return_samples: bool,
    n_workers: int = 1,
) -> int:
    """Bytes a bootstrap run must hold for its retained output.

    Charges eight bytes for every output-sized array a run holds at once: the
    two bounded tails, the replicates buffered before the next flush and the
    two temporaries that flush builds, one dispatch window of in-flight
    replicates, every replicate when `return_samples=True`, and the two Welford
    accumulators plus the four summary payloads.

    Args:
        output_shape (tuple[int, ...]): Shape of one replicate's output.
        n_samples (int): Number of bootstrap replicates.
        confidence_level (float): Interval confidence level, which sets the
            retained tail size.
        return_samples (bool): Whether the complete distribution is retained.
        n_workers (int): Planned CPU worker count, which sets the dispatch
            window. Defaults to 1 (the GPU driver budgets its own batch through
            `ridge_bootstrap_batch_size` instead).

    Returns:
        int: Required bytes.
    """
    output_size = int(np.prod(output_shape)) if output_shape else 1
    tail_size = bootstrap_retained_tail_size(
        n_samples, confidence_level=confidence_level
    )
    buffered = min(BOOTSTRAP_TAIL_FLUSH_BLOCK, int(n_samples))
    arrays = (
        2 * tail_size  # the two bounded tails
        + buffered  # replicates buffered before the next flush
        + 2 * (tail_size + buffered)  # a flush's concatenation and partition
        + bootstrap_replicate_window(n_samples, n_workers=n_workers)
        + (int(n_samples) if return_samples else 0)
        + _BOOTSTRAP_FIXED_OUTPUT_ARRAYS
    )
    return output_size * arrays * _BOOTSTRAP_OUTPUT_ITEMSIZE


def bootstrap_memory_preflight(
    output_shape: tuple[int, ...],
    n_samples: int,
    *,
    confidence_level: float,
    return_samples: bool,
    n_workers: int = 1,
    memory_budget_gb: float | None = None,
    backend: "Backend | None" = None,
) -> float:
    """Raise before any resampling if the retained output cannot fit the budget.

    The package's one bootstrap memory gate. It runs against the measured
    budget when `memory_budget_gb` is None, so a run that would otherwise die
    part-way through fails immediately and says what it needed. It never
    weakens the interval, reduces `n_samples`, or disables `return_samples`.

    Args:
        output_shape (tuple[int, ...]): Shape of one replicate's output.
        n_samples (int): Number of bootstrap replicates.
        confidence_level (float): Interval confidence level.
        return_samples (bool): Whether the complete distribution is retained.
        n_workers (int): Planned CPU worker count, which sets the dispatch
            window. Defaults to 1.
        memory_budget_gb (float | None): Explicit budget in GB, or None to
            measure the device.
        backend (Backend | None): Resolved backend whose device is measured
            when `memory_budget_gb` is None. None means the CPU.

    Returns:
        float: The required storage in GB.

    Raises:
        ValueError: If the required storage exceeds the budget.
    """
    required_bytes = bootstrap_output_bytes(
        output_shape,
        n_samples,
        confidence_level=confidence_level,
        return_samples=return_samples,
        n_workers=n_workers,
    )
    required_gb = required_bytes / 1e9
    budget_gb = device_memory_budget(backend, max_gpu_memory_gb=memory_budget_gb)
    if required_gb > budget_gb:
        tail_size = bootstrap_retained_tail_size(
            n_samples, confidence_level=confidence_level
        )
        retained = f"{tail_size} values per element at each tail"
        if return_samples:
            retained = (
                f"all {n_samples} replicates (return_samples=True) plus {retained}"
            )
        source = (
            "the explicit memory_budget_gb"
            if memory_budget_gb is not None
            else "the measured device budget"
        )
        raise ValueError(
            f"bootstrap needs {required_gb:.6g} GB to retain output of shape "
            f"{tuple(output_shape)} over {n_samples} replicates — it keeps "
            f"{retained} — which exceeds the {budget_gb:.6g} GB budget "
            f"({source}). Lower n_samples, mask to fewer voxels, turn off "
            f"return_samples, or raise the budget with "
            f"memory_budget_gb=<GB>."
        )
    return required_gb


def bootstrap_n_jobs_cpu(
    data_size_mb: float,
    n_samples: int,
    *,
    memory_budget_gb: float | None = None,
    n_jobs: int = -1,
) -> int:
    """CPU worker count for a bootstrap run, capped by `n_jobs` and by memory.

    `n_jobs` is the ceiling the caller asked for; this planner may return
    fewer when the per-worker copy of the data would not fit the budget. It
    never returns zero, because the preflight — not the worker planner — is
    where a run that cannot fit is refused.

    Args:
        data_size_mb (float): Size of the array each worker pickles, in MB.
        n_samples (int): Number of bootstrap replicates.
        memory_budget_gb (float | None): Explicit budget in GB, or None to
            measure available system memory.
        n_jobs (int): Worker ceiling, with joblib's negative convention
            (`-1` = all cores).

    Returns:
        int: Worker count for `joblib.Parallel(n_jobs=...)`.
    """
    import multiprocessing

    cores = multiprocessing.cpu_count()
    ceiling = cores if n_jobs == -1 else n_jobs
    if ceiling < 0:
        ceiling = cores + 1 + ceiling
    ceiling = max(1, int(ceiling))
    try:
        return _auto_n_jobs_cpu(
            data_size_mb,
            n_samples,
            max_memory_gb=memory_budget_gb,
            max_jobs=ceiling,
        )
    except ValueError:
        return 1


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


def _empty_device_cache() -> None:
    """Release cached device memory.

    No-op without torch or a GPU.
    """
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
        fn (Callable[..., np.ndarray]): Maps the arrays to a numpy result
            (axis-0 aligned).
        *arrays (np.ndarray): Input arrays sharing their axis-0 length.
        min_chunk (int): Chunk size below which an OOM is considered fatal.
            Defaults to 1.

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
        _empty_device_cache()
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
    max_jobs: int | None = None,
) -> int:
    """Choose how many CPU workers fit in memory for a permutation job.

    Each joblib worker pickles its copy of the data, which costs roughly 3× the
    array size, so the worker count is the memory budget divided by that
    per-worker cost, capped at `max_jobs`.

    Args:
        data_size_mb (float): Size of the data array in MB.
        n_permute (int): Number of permutations to compute (adds a small
            per-worker result overhead).
        max_memory_gb (float | None): Explicit memory budget in GB. None
            (default) measures available system RAM with headroom via
            `device_memory_budget`.
        max_jobs (int | None): Maximum number of workers. None (default) means
            all cores.

    Returns:
        int: Worker count for `joblib.Parallel(n_jobs=...)`.

    Examples:
        ```python
        _auto_n_jobs_cpu(1.0, 5000, max_memory_gb=8.0)  # small data → many workers
        _auto_n_jobs_cpu(100.0, 5000, max_memory_gb=8.0)  # large data → fewer workers
        ```
    """
    import multiprocessing

    # Get system limits
    if max_jobs is None:
        max_jobs = multiprocessing.cpu_count()

    available_memory_gb = device_memory_budget(None, max_gpu_memory_gb=max_memory_gb)
    available_memory_bytes = gb_to_bytes(available_memory_gb)

    # Memory per worker: data serialization overhead (3× is conservative for pickle)
    # Plus small overhead for result arrays (n_permute results per worker)
    serialization_factor = 3.0
    memory_per_worker_bytes = (
        data_size_mb * 1024**2 * serialization_factor + n_permute * 4 * 0.1
    )

    # How many workers can fit in memory budget?
    if memory_per_worker_bytes <= 0:
        return 1

    max_workers_by_memory = int(available_memory_bytes / memory_per_worker_bytes)
    if max_workers_by_memory < 1:
        raise ValueError(
            f"one worker requires approximately {memory_per_worker_bytes / 1e9:.6g} "
            f"GB, exceeding the {available_memory_gb:.6g} GB memory budget"
        )

    return max(1, min(max_workers_by_memory, max_jobs))


def _estimate_data_size_mb(data: np.ndarray) -> float:
    """Estimate the memory footprint of an array in MB.

    Accounts for the dtype item size plus a fixed numpy object overhead.

    Args:
        data (np.ndarray): Data array.

    Returns:
        float: Estimated size in MB (0.0 for an empty array).
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
