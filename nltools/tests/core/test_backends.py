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
    from nltools.algorithms.backends import _Backend

    backend = _Backend("numpy")
    assert backend.name == "numpy"
    assert backend.device == "cpu"
    assert backend.xp is np


def test_auto_backend_without_torch(monkeypatch):
    """Auto-selection should fall back to numpy if no GPU"""
    from nltools.algorithms.backends import _Backend
    import sys

    # Mock torch unavailable
    monkeypatch.setitem(sys.modules, "torch", None)

    backend = _Backend("auto")
    assert backend.name == "numpy"


@pytest.mark.slow
@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_torch_backend_selection():
    """Torch backend should detect available device"""
    from nltools.algorithms.backends import _Backend

    backend = _Backend("torch")
    assert backend.name.startswith("torch-")
    assert backend.device in ["cpu", "cuda", "mps"]


def test_resolve_backend_from_string():
    """_resolve_backend should accept string identifiers"""
    from nltools.algorithms.backends import _resolve_backend, _Backend

    assert isinstance(_resolve_backend("numpy"), _Backend)
    assert _resolve_backend("numpy").name == "numpy"
    assert _resolve_backend(None).name == "numpy"
    assert _resolve_backend("cpu").name == "numpy"


def test_resolve_backend_passthrough():
    """_resolve_backend should return Backend instances unchanged (no re-init)"""
    from nltools.algorithms.backends import _resolve_backend, _Backend

    existing = _Backend("numpy")
    result = _resolve_backend(existing)
    assert result is existing  # identity, not a copy


def test_resolve_backend_invalid():
    """_resolve_backend should reject unknown strings"""
    from nltools.algorithms.backends import _resolve_backend

    with pytest.raises(ValueError, match="parallel"):
        _resolve_backend("bogus")


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


# ============================================================================
# Precision warnings
# ============================================================================


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_mps_backend_warning():
    """MPS backend should warn about precision limitations on initialization"""
    import torch
    import warnings
    from nltools.algorithms.backends import _Backend

    # Skip if MPS not available
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS not available")

    # Reset warning flag
    import nltools.algorithms.backends

    nltools.algorithms.backends._already_warned_mps_init[0] = False

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = _Backend("torch")  # Initialize to trigger warning

        # Check that warning was issued about MPS precision
        mps_warnings = [
            warning
            for warning in w
            if "torch-mps backend uses float32" in str(warning.message)
        ]
        assert len(mps_warnings) > 0, (
            f"Expected MPS precision warning on initialization. Got warnings: {[str(w.message) for w in w]}"
        )


# ============================================================================
# Auto-Selection Logic
# ============================================================================


def test_auto_selection_without_gpu():
    """Auto-selection should work without GPU"""
    from nltools.algorithms.backends import _auto_select_backend

    # This should always work
    backend = _auto_select_backend(n_samples=1000, n_features=100000)
    assert backend.name in ["numpy", "torch-cpu", "torch-cuda", "torch-mps"]


# ============================================================================
# dtype_to_str
# ============================================================================


# ============================================================================
# asarray
# ============================================================================


# ============================================================================
# Device Transfer
# ============================================================================


# ============================================================================
# Core memory / batching layer (v0.6.0 GPU consolidation)
# ============================================================================


class TestDeviceMemoryBudget:
    def test_explicit_budget_wins(self):
        from nltools.algorithms.backends import _Backend, _device_memory_budget

        assert _device_memory_budget(_Backend("numpy"), max_gpu_memory_gb=2.5) == 2.5

    def test_explicit_budget_rejects_nonpositive(self):
        from nltools.algorithms.backends import _Backend, _device_memory_budget

        with pytest.raises(ValueError, match="max_gpu_memory_gb"):
            _device_memory_budget(_Backend("numpy"), max_gpu_memory_gb=0)


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
            _device_memory_budget,
        )

        self._mock_measured_ram(monkeypatch, 100.0)
        capped = _device_memory_budget(None, cap_for_batching=True)
        assert capped == BATCH_WORKING_SET_CEILING_GB

    def test_explicit_budget_never_capped(self):
        from nltools.algorithms.backends import _device_memory_budget

        budget = _device_memory_budget(
            None, max_gpu_memory_gb=100.0, cap_for_batching=True
        )
        assert budget == 100.0


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

    def test_a_wider_output_shrinks_the_batch(self):
        """The retained result is charged, so a bigger `X_test` costs batch size."""
        from nltools.algorithms.backends import _ridge_bootstrap_batch_size

        narrow, _ = _ridge_bootstrap_batch_size(
            5000, output_shape=(10, 50_000), max_gpu_memory_gb=8.0, **self.CASE
        )
        wide, _ = _ridge_bootstrap_batch_size(
            5000, output_shape=(4000, 50_000), max_gpu_memory_gb=8.0, **self.CASE
        )
        assert wide < narrow


class TestAutoBatchSizeCore:
    def test_batch_size_is_limited_by_budget(self):
        from nltools.algorithms.backends import _auto_batch_size

        # Budget fits five items.
        batch, _ = _auto_batch_size(1000, bytes_per_item=int(1e9), budget_gb=5.5)
        assert batch == 5

    def test_one_item_too_large_raises(self):
        from nltools.algorithms.backends import _auto_batch_size

        with pytest.raises(ValueError, match="one item requires"):
            _auto_batch_size(5, bytes_per_item=int(1e9), budget_gb=0.5)

    def test_zero_bytes_per_item_is_safe(self):
        from nltools.algorithms.backends import _auto_batch_size

        batch, n_batches = _auto_batch_size(50, bytes_per_item=0, budget_gb=1.0)
        assert batch == 50
        assert n_batches == 1

    def test_batch_count_covers_all_items(self):
        from nltools.algorithms.backends import _auto_batch_size

        batch, n_batches = _auto_batch_size(
            1050, bytes_per_item=int(1e7), budget_gb=1.0
        )
        assert batch * n_batches >= 1050
        assert batch * (n_batches - 1) < 1050


class TestIsOomError:
    def test_cuda_oom_runtimeerror(self):
        from nltools.algorithms.backends import _is_oom_error

        assert _is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate ..."))

    def test_ordinary_error_is_not_oom(self):
        from nltools.algorithms.backends import _is_oom_error

        assert not _is_oom_error(RuntimeError("shape mismatch"))
        assert not _is_oom_error(ValueError("out of memory"))  # wrong type


class TestComputeOomSafe:
    def test_splits_on_oom_and_matches_unsplit(self):
        from nltools.algorithms.backends import _compute_oom_safe

        arr = np.arange(64, dtype=np.float64).reshape(16, 4)
        calls = []

        def flaky(a):
            calls.append(len(a))
            if len(a) > 4:
                raise RuntimeError("MPS backend out of memory")
            return a.sum(axis=1, keepdims=True)

        result = _compute_oom_safe(flaky, arr)
        np.testing.assert_array_equal(result, arr.sum(axis=1, keepdims=True))
        # First attempt was the full batch; every successful call was <= threshold
        assert calls[0] == 16
        assert all(c <= 4 for c in calls if c <= 4)  # succeeded chunks
        assert sum(c for c in calls if c <= 4) == 16  # full coverage, no re-draws

    def test_non_oom_error_propagates(self):
        from nltools.algorithms.backends import _compute_oom_safe

        arr = np.zeros((4, 2))

        def bad(a):
            raise ValueError("genuine bug")

        with pytest.raises(ValueError, match="genuine bug"):
            _compute_oom_safe(bad, arr)

    def test_oom_at_single_row_raises_memoryerror(self):
        from nltools.algorithms.backends import _compute_oom_safe

        arr = np.zeros((4, 2))

        def always_oom(a):
            raise RuntimeError("CUDA out of memory")

        with pytest.raises(MemoryError, match="single item"):
            _compute_oom_safe(always_oom, arr)


class TestBudgetMathSingleSource:
    """Memory sizing has one home.

    `_device_memory_budget`/`_auto_batch_size` are the only budget math in the
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
            "(_device_memory_budget / _auto_batch_size). Offenders:\n"
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
