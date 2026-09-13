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
    import importlib.util

    return importlib.util.find_spec("torch") is not None


# ============================================================================
# Backend Initialization
# ============================================================================


def test_numpy_backend_default():
    """NumPy backend should work without PyTorch"""
    from nltools.algorithms.backends import Backend

    backend = Backend("numpy")
    assert backend.name == "numpy"
    assert backend.device == "cpu"
    assert backend.xp is np


def test_auto_backend_without_torch(monkeypatch):
    """Auto-selection should fall back to numpy if no GPU"""
    from nltools.algorithms.backends import Backend
    import sys

    # Mock torch unavailable
    monkeypatch.setitem(sys.modules, "torch", None)

    backend = Backend("auto")
    assert backend.name == "numpy"


@pytest.mark.slow
@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_torch_backend_selection():
    """Torch backend should detect available device"""
    from nltools.algorithms.backends import Backend

    backend = Backend("torch")
    assert backend.name.startswith("torch-")
    assert backend.device in ["cpu", "cuda", "mps"]


def test_resolve_backend_from_string():
    """resolve_backend should accept string identifiers"""
    from nltools.algorithms.backends import resolve_backend, Backend

    assert isinstance(resolve_backend("numpy"), Backend)
    assert resolve_backend("numpy").name == "numpy"
    assert resolve_backend(None).name == "numpy"
    assert resolve_backend("cpu").name == "numpy"


def test_resolve_backend_passthrough():
    """resolve_backend should return Backend instances unchanged (no re-init)"""
    from nltools.algorithms.backends import resolve_backend, Backend

    existing = Backend("numpy")
    result = resolve_backend(existing)
    assert result is existing  # identity, not a copy


def test_resolve_backend_invalid():
    """resolve_backend should reject unknown strings"""
    from nltools.algorithms.backends import resolve_backend

    with pytest.raises(ValueError, match="parallel"):
        resolve_backend("bogus")


def test_check_gpu_available():
    """GPU availability check should return bool and info dict"""
    from nltools.algorithms.backends import check_gpu_available

    available, info = check_gpu_available()
    assert isinstance(available, bool)
    assert "backend" in info
    assert "device" in info
    assert "device_name" in info


# ============================================================================
# Array Transfer Operations (to_numpy)
# ============================================================================


def test_numpy_to_numpy():
    """NumPy backend to_numpy should be identity"""
    from nltools.algorithms.backends import Backend

    backend = Backend("numpy")
    arr = np.random.randn(10, 5).astype(np.float32)
    result = backend.to_numpy(arr)

    assert result is arr  # Should be same object


# ============================================================================
# Precision warnings
# ============================================================================


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_mps_backend_warning():
    """MPS backend should warn about precision limitations on initialization"""
    import torch
    import warnings
    from nltools.algorithms.backends import Backend

    # Skip if MPS not available
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS not available")

    # Reset warning flag
    import nltools.algorithms.backends

    nltools.algorithms.backends._already_warned_mps_init[0] = False

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = Backend("torch")  # Initialize to trigger warning

        # Check that warning was issued about MPS precision
        mps_warnings = [
            warning
            for warning in w
            if "torch-mps backend uses float32" in str(warning.message)
        ]
        assert len(mps_warnings) > 0, (
            f"Expected MPS precision warning on initialization. Got warnings: {[str(w.message) for w in w]}"
        )


def test_assert_array_almost_equal_precision_adjustment():
    """assert_array_almost_equal should auto-adjust precision for MPS"""
    import torch
    import warnings
    from nltools.algorithms.backends import Backend
    from nltools.tests.support.arrays import assert_array_almost_equal

    # Skip if MPS not available
    if not _torch_available():
        pytest.skip("PyTorch not installed")

    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS not available")

    backend = Backend("torch")
    assert backend.name == "torch-mps"

    # Create arrays that are very close (within float32 precision but might fail at high decimal precision)
    np.random.seed(42)
    x = np.random.randn(10).astype(np.float32)
    y = x.copy()  # Identical arrays

    x_tensor = torch.from_numpy(x).to(backend._torch_device)
    y_tensor = torch.from_numpy(y).to(backend._torch_device)

    # Should auto-adjust precision and issue warning when requesting high precision
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Request high precision (6 decimals) but should auto-adjust to 2
        # Arrays are identical so should pass regardless, but warning should be issued
        assert_array_almost_equal(x_tensor, y_tensor, decimal=6, backend=backend)

        # Check that precision adjustment warning was issued
        precision_warnings = [
            warning
            for warning in w
            if "Reducing precision" in str(warning.message)
            or "decimal=2" in str(warning.message)
        ]
        assert len(precision_warnings) > 0, "Expected precision adjustment warning"


# ============================================================================
# Auto-Selection Logic
# ============================================================================


def test_small_dataset_uses_numpy():
    """Small datasets should use NumPy even if GPU available"""
    from nltools.algorithms.backends import auto_select_backend

    # Small problem
    backend = auto_select_backend(n_samples=100, n_features=1000)
    # Should use numpy (or torch-cpu) to avoid transfer overhead
    assert backend.name in ["numpy", "torch-cpu"]


def test_large_dataset_considers_gpu():
    """Large datasets should consider GPU if available"""
    from nltools.algorithms.backends import auto_select_backend

    backend = auto_select_backend(n_samples=300, n_features=100000)

    # If GPU available, should use torch; otherwise numpy
    assert backend.name in ["numpy", "torch-cuda", "torch-mps", "torch-cpu"]


def test_cv_enables_gpu():
    """Cross-validation should prefer GPU even for medium datasets"""
    from nltools.algorithms.backends import auto_select_backend

    backend = auto_select_backend(n_samples=200, n_features=30000, cv=5)

    # With CV, should prefer GPU if available
    assert backend.name in ["numpy", "torch-cuda", "torch-mps", "torch-cpu"]


def test_auto_selection_without_gpu():
    """Auto-selection should work without GPU"""
    from nltools.algorithms.backends import auto_select_backend

    # This should always work
    backend = auto_select_backend(n_samples=1000, n_features=100000)
    assert backend.name in ["numpy", "torch-cpu", "torch-cuda", "torch-mps"]


# ============================================================================
# dtype_to_str
# ============================================================================


class TestDtypeToStr:
    """Test static dtype normalization method."""

    def test_string_passthrough(self):
        from nltools.algorithms.backends import Backend

        assert Backend.dtype_to_str("float32") == "float32"
        assert Backend.dtype_to_str("float64") == "float64"
        assert Backend.dtype_to_str("int32") == "int32"

    def test_none_passthrough(self):
        from nltools.algorithms.backends import Backend

        assert Backend.dtype_to_str(None) is None

    def test_numpy_dtype(self):
        from nltools.algorithms.backends import Backend

        assert Backend.dtype_to_str(np.float32) == "float32"
        assert Backend.dtype_to_str(np.float64) == "float64"
        assert Backend.dtype_to_str(np.int32) == "int32"

    def test_numpy_dtype_instance(self):
        from nltools.algorithms.backends import Backend

        arr = np.array([1.0], dtype=np.float32)
        assert Backend.dtype_to_str(arr.dtype) == "float32"

    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
    def test_torch_dtype(self):
        import torch
        from nltools.algorithms.backends import Backend

        assert Backend.dtype_to_str(torch.float32) == "float32"
        assert Backend.dtype_to_str(torch.float64) == "float64"
        assert Backend.dtype_to_str(torch.int32) == "int32"


# ============================================================================
# asarray
# ============================================================================


class TestAsarray:
    """Test universal array conversion."""

    def test_from_list(self):
        from nltools.algorithms.backends import Backend

        backend = Backend("numpy")
        result = backend.asarray([1, 2, 3], dtype="float32")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_from_numpy(self):
        from nltools.algorithms.backends import Backend

        backend = Backend("numpy")
        arr = np.array([1, 2, 3], dtype=np.float64)
        result = backend.asarray(arr, dtype="float32")
        assert result.dtype == np.float32

    def test_preserves_dtype_if_none(self):
        from nltools.algorithms.backends import Backend

        backend = Backend("numpy")
        arr = np.array([1, 2, 3], dtype=np.float64)
        result = backend.asarray(arr)
        assert result.dtype == np.float64

    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
    def test_torch_from_numpy(self):
        import torch
        from nltools.algorithms.backends import Backend

        backend = Backend("torch")
        arr = np.array([1, 2, 3], dtype=np.float32)
        result = backend.asarray(arr)
        assert isinstance(result, torch.Tensor)

    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
    def test_torch_from_list(self):
        import torch
        from nltools.algorithms.backends import Backend

        backend = Backend("torch")
        result = backend.asarray([1.0, 2.0, 3.0], dtype="float32")
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32


# ============================================================================
# Device Transfer
# ============================================================================


class TestDeviceTransferOps:
    """Test `to_numpy` brings a device tensor back to the host."""

    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
    def test_torch_to_numpy_from_tensor(self):
        import torch

        from nltools.algorithms.backends import Backend

        backend = Backend("torch")
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = torch.from_numpy(arr).to(backend._torch_device)
        result = backend.to_numpy(tensor)
        assert isinstance(result, np.ndarray)


# ============================================================================
# Core memory / batching layer (v0.6.0 GPU consolidation)
# ============================================================================


class TestDeviceMemoryBudget:
    def test_explicit_budget_wins(self):
        from nltools.algorithms.backends import Backend, device_memory_budget

        assert device_memory_budget(Backend("numpy"), max_gpu_memory_gb=2.5) == 2.5

    def test_measured_budget_is_positive(self):
        from nltools.algorithms.backends import Backend, device_memory_budget

        budget = device_memory_budget(Backend("numpy"))
        assert budget > 0

    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
    def test_torch_backend_budget_is_positive(self):
        from nltools.algorithms.backends import Backend, device_memory_budget

        budget = device_memory_budget(Backend("torch"))
        assert budget > 0

    def test_none_backend_uses_system_memory(self):
        from nltools.algorithms.backends import device_memory_budget

        assert device_memory_budget(None) > 0

    def test_explicit_budget_rejects_nonpositive(self):
        from nltools.algorithms.backends import Backend, device_memory_budget

        with pytest.raises(ValueError, match="max_gpu_memory_gb"):
            device_memory_budget(Backend("numpy"), max_gpu_memory_gb=0)


class TestBatchingSaturationCeiling:
    """Measured budgets are capped at a saturation ceiling for batch sizing only."""

    @staticmethod
    def _mock_measured_ram(monkeypatch, measured_gb):
        """Make the measured system budget come out to exactly `measured_gb`."""
        import psutil

        from nltools.algorithms import backends

        class _VM:
            available = measured_gb / backends._SYSTEM_HEADROOM * 1e9

        monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM())

    def test_measured_budget_capped_for_batching(self, monkeypatch):
        from nltools.algorithms.backends import (
            BATCH_WORKING_SET_CEILING_GB,
            device_memory_budget,
        )

        self._mock_measured_ram(monkeypatch, 100.0)
        capped = device_memory_budget(None, cap_for_batching=True)
        assert capped == BATCH_WORKING_SET_CEILING_GB

    def test_explicit_budget_never_capped(self):
        from nltools.algorithms.backends import device_memory_budget

        budget = device_memory_budget(
            None, max_gpu_memory_gb=100.0, cap_for_batching=True
        )
        assert budget == 100.0

    def test_measured_budget_below_ceiling_used_as_is(self, monkeypatch):
        from nltools.algorithms.backends import device_memory_budget

        self._mock_measured_ram(monkeypatch, 2.0)
        assert device_memory_budget(None, cap_for_batching=True) == pytest.approx(2.0)

    def test_capacity_queries_stay_uncapped(self, monkeypatch):
        from nltools.algorithms.backends import device_memory_budget

        self._mock_measured_ram(monkeypatch, 100.0)
        assert device_memory_budget(None) == pytest.approx(100.0)

    def test_auto_batch_size_working_set_bounded_by_ceiling(self, monkeypatch):
        """A measured 100 GB budget must not produce ~100 GB batches."""
        from nltools.algorithms.backends import (
            BATCH_WORKING_SET_CEILING_GB,
            auto_batch_size,
            device_memory_budget,
            gb_to_bytes,
        )

        self._mock_measured_ram(monkeypatch, 100.0)
        bytes_per_item = 30 * 50000 * 4  # float32
        budget_gb = device_memory_budget(None, cap_for_batching=True)
        batch_size, _ = auto_batch_size(100000, bytes_per_item, budget_gb=budget_gb)
        working_set = batch_size * bytes_per_item
        assert working_set <= gb_to_bytes(BATCH_WORKING_SET_CEILING_GB)


class TestRidgeBootstrapBatchSize:
    """The bootstrap batch planner must model what a batch actually holds."""

    #: The review's measured case: 5000 replicates, 200 training rows, 500
    #: features, 50 000 voxels, against an explicit 8 GB budget.
    CASE = {
        "n_samples": 200,
        "n_features": 500,
        "n_targets": 50_000,
    }

    @staticmethod
    def _held_bytes(batch_size, output_shape, **case):
        """What a batch actually holds.

        One replicate solves at a time, so device residency is one resampled
        design and response plus the solver's buffers; what accumulates across
        the batch is the host list of float64 results.
        """
        import numpy as np

        from nltools.algorithms.backends import _RIDGE_BOOTSTRAP_SOLVER_OVERHEAD

        resident = (
            (
                case["n_samples"] * case["n_features"]
                + case["n_samples"] * case["n_targets"]
            )
            * 8
            * _RIDGE_BOOTSTRAP_SOLVER_OVERHEAD
        )
        retained = int(np.prod(output_shape)) * 8
        return resident + batch_size * retained

    @pytest.mark.parametrize(
        "output_shape",
        [(500, 50_000), (2000, 50_000)],
        ids=["weights", "predict-2000-test-rows"],
    )
    def test_explicit_budget_bounds_the_modelled_batch(self, output_shape):
        from nltools.algorithms.backends import gb_to_bytes, ridge_bootstrap_batch_size

        budget_gb = 8.0
        batch_size, n_batches = ridge_bootstrap_batch_size(
            5000,
            output_shape=output_shape,
            device_itemsize=8,
            max_gpu_memory_gb=budget_gb,
            **self.CASE,
        )

        assert batch_size >= 1
        assert batch_size * n_batches >= 5000
        held = self._held_bytes(batch_size, output_shape, **self.CASE)
        assert held <= gb_to_bytes(budget_gb)

    def test_a_wider_output_shrinks_the_batch(self):
        """The retained result is charged, so a bigger `X_test` costs batch size."""
        from nltools.algorithms.backends import ridge_bootstrap_batch_size

        narrow, _ = ridge_bootstrap_batch_size(
            5000, output_shape=(10, 50_000), max_gpu_memory_gb=8.0, **self.CASE
        )
        wide, _ = ridge_bootstrap_batch_size(
            5000, output_shape=(4000, 50_000), max_gpu_memory_gb=8.0, **self.CASE
        )
        assert wide < narrow

    def test_float32_device_holds_more_per_batch(self):
        from nltools.algorithms.backends import ridge_bootstrap_batch_size

        wide_dtype, _ = ridge_bootstrap_batch_size(
            5000,
            output_shape=(4, 4),
            device_itemsize=8,
            max_gpu_memory_gb=1.0,
            **self.CASE,
        )
        narrow_dtype, _ = ridge_bootstrap_batch_size(
            5000,
            output_shape=(4, 4),
            device_itemsize=4,
            max_gpu_memory_gb=1.0,
            **self.CASE,
        )
        assert narrow_dtype > wide_dtype


class TestAutoBatchSizeCore:
    def test_all_fit_in_one_batch(self):
        from nltools.algorithms.backends import auto_batch_size

        batch, n_batches = auto_batch_size(100, bytes_per_item=1000, budget_gb=1.0)
        assert batch == 100
        assert n_batches == 1

    def test_splits_when_over_budget(self):
        from nltools.algorithms.backends import auto_batch_size

        # 1 GB budget, 100 MB per item -> 10 items per batch
        batch, n_batches = auto_batch_size(100, bytes_per_item=int(1e8), budget_gb=1.0)
        assert batch == 10
        assert n_batches == 10

    def test_batch_size_is_limited_by_budget(self):
        from nltools.algorithms.backends import auto_batch_size

        # Budget fits five items.
        batch, _ = auto_batch_size(1000, bytes_per_item=int(1e9), budget_gb=5.5)
        assert batch == 5

    def test_one_item_too_large_raises(self):
        from nltools.algorithms.backends import auto_batch_size

        with pytest.raises(ValueError, match="one item requires"):
            auto_batch_size(5, bytes_per_item=int(1e9), budget_gb=0.5)

    def test_overhead_shrinks_batch(self):
        from nltools.algorithms.backends import auto_batch_size

        loose, _ = auto_batch_size(10000, bytes_per_item=int(1e6), budget_gb=1.0)
        tight, _ = auto_batch_size(
            10000, bytes_per_item=int(1e6), budget_gb=1.0, overhead=5.0
        )
        assert tight < loose
        assert tight == loose // 5

    def test_zero_bytes_per_item_is_safe(self):
        from nltools.algorithms.backends import auto_batch_size

        batch, n_batches = auto_batch_size(50, bytes_per_item=0, budget_gb=1.0)
        assert batch == 50
        assert n_batches == 1

    def test_batch_count_covers_all_items(self):
        from nltools.algorithms.backends import auto_batch_size

        batch, n_batches = auto_batch_size(1050, bytes_per_item=int(1e7), budget_gb=1.0)
        assert batch * n_batches >= 1050
        assert batch * (n_batches - 1) < 1050


class TestIsOomError:
    def test_mps_oom_runtimeerror(self):
        from nltools.algorithms.backends import is_oom_error

        assert is_oom_error(
            RuntimeError("MPS backend out of memory (MPS allocated ...)")
        )

    def test_cuda_oom_runtimeerror(self):
        from nltools.algorithms.backends import is_oom_error

        assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate ..."))

    def test_ordinary_error_is_not_oom(self):
        from nltools.algorithms.backends import is_oom_error

        assert not is_oom_error(RuntimeError("shape mismatch"))
        assert not is_oom_error(ValueError("out of memory"))  # wrong type

    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
    def test_torch_cuda_oom_class(self):
        import torch
        from nltools.algorithms.backends import is_oom_error

        if hasattr(torch, "OutOfMemoryError"):
            assert is_oom_error(torch.OutOfMemoryError("boom"))


class TestComputeOomSafe:
    def test_no_oom_passthrough(self):
        from nltools.algorithms.backends import compute_oom_safe

        arr = np.arange(20, dtype=np.float64).reshape(10, 2)
        result = compute_oom_safe(lambda a: a * 2, arr)
        np.testing.assert_array_equal(result, arr * 2)

    def test_splits_on_oom_and_matches_unsplit(self):
        from nltools.algorithms.backends import compute_oom_safe

        arr = np.arange(64, dtype=np.float64).reshape(16, 4)
        calls = []

        def flaky(a):
            calls.append(len(a))
            if len(a) > 4:
                raise RuntimeError("MPS backend out of memory")
            return a.sum(axis=1, keepdims=True)

        result = compute_oom_safe(flaky, arr)
        np.testing.assert_array_equal(result, arr.sum(axis=1, keepdims=True))
        # First attempt was the full batch; every successful call was <= threshold
        assert calls[0] == 16
        assert all(c <= 4 for c in calls if c <= 4)  # succeeded chunks
        assert sum(c for c in calls if c <= 4) == 16  # full coverage, no re-draws

    def test_multiple_arrays_split_together(self):
        from nltools.algorithms.backends import compute_oom_safe

        a = np.arange(12, dtype=np.float64).reshape(6, 2)
        b = np.arange(6, dtype=np.float64)

        def flaky(x, y):
            if len(x) > 2:
                raise RuntimeError("CUDA out of memory")
            assert len(x) == len(y)
            return x * y[:, None]

        result = compute_oom_safe(flaky, a, b)
        np.testing.assert_array_equal(result, a * b[:, None])

    def test_non_oom_error_propagates(self):
        from nltools.algorithms.backends import compute_oom_safe

        arr = np.zeros((4, 2))

        def bad(a):
            raise ValueError("genuine bug")

        with pytest.raises(ValueError, match="genuine bug"):
            compute_oom_safe(bad, arr)

    def test_oom_at_single_row_raises_memoryerror(self):
        from nltools.algorithms.backends import compute_oom_safe

        arr = np.zeros((4, 2))

        def always_oom(a):
            raise RuntimeError("CUDA out of memory")

        with pytest.raises(MemoryError, match="single item"):
            compute_oom_safe(always_oom, arr)

    def test_deterministic_result_independent_of_split(self):
        from nltools.algorithms.backends import compute_oom_safe

        rng = np.random.default_rng(0)
        arr = rng.standard_normal((32, 3))
        expected = np.cumsum(arr, axis=1)  # rowwise op -> split-invariant

        thresholds = [64, 16, 5, 1]
        for thresh in thresholds:

            def flaky(a, _t=thresh):
                if len(a) > _t:
                    raise RuntimeError("MPS backend out of memory")
                return np.cumsum(a, axis=1)

            np.testing.assert_array_equal(compute_oom_safe(flaky, arr), expected)


class TestBudgetMathSingleSource:
    """Memory sizing has one home.

    `device_memory_budget`/`auto_batch_size` are the only budget math in the
    package, and the `_auto_n_jobs_cpu`/`_estimate_data_size_mb` worker-sizing
    helpers live only in `backends`.
    """

    def test_no_budget_math_outside_backends(self):
        from pathlib import Path
        import nltools

        root = Path(nltools.__file__).parent
        forbidden = ("* 1e9", "1024**3", "1024 ** 3")
        offenders = []
        for py in sorted(root.rglob("*.py")):
            if py.name == "backends.py" or "tests" in py.parts:
                continue
            for i, line in enumerate(py.read_text().splitlines(), 1):
                if any(tok in line for tok in forbidden):
                    offenders.append(f"{py.relative_to(root)}:{i}: {line.strip()}")
        assert not offenders, (
            "GB->bytes budget math must live only in nltools/algorithms/backends.py "
            "(device_memory_budget / auto_batch_size). Offenders:\n"
            + "\n".join(offenders)
        )

    def test_n_jobs_helpers_single_home(self):
        """The n_jobs memory helpers live only in backends.

        No compat re-export from inference.utils (one import path), and no
        `_verify_n_jobs_memory_constraint` twin — it had no production caller
        and duplicated `_auto_n_jobs_cpu`'s memory math verbatim.
        """
        from nltools.algorithms import backends
        from nltools.algorithms.inference import utils as inf_utils

        assert not hasattr(inf_utils, "_auto_n_jobs_cpu")
        assert not hasattr(inf_utils, "_estimate_data_size_mb")
        assert not hasattr(inf_utils, "_verify_n_jobs_memory_constraint")
        assert not hasattr(backends, "_verify_n_jobs_memory_constraint")
