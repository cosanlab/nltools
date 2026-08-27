"""Reactive OOM recovery in the GPU-batched inference paths.

Every batched device computation routes through
`nltools.algorithms.backends.compute_oom_safe`: on device OOM the
already-generated batch inputs are split and retried, so recovery never
re-draws RNG state and a seeded result is identical with or without OOM.
These tests simulate OOM by making `Backend.to_device` refuse large
batches and assert bit-identical results against the unperturbed run.

The simulated-OOM tests run on any torch device (torch-cpu included), so
they exercise the recovery logic in CI without GPU hardware.
"""

import importlib.util

import numpy as np
import pytest

from nltools.algorithms.backends import Backend

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="PyTorch not installed"
)

# Batches larger than this many rows "OOM" once patched; must exceed
# n_samples so the one-time data transfers still succeed.
THRESHOLD = 64
N_PERMUTE = 300


def _patch_flaky_to_device(monkeypatch, threshold=THRESHOLD):
    """Make Backend.to_device raise a fake device OOM for large batches."""
    orig = Backend.to_device
    counts = {"oom": 0}

    def patched(self, arr):
        shape = getattr(arr, "shape", None)
        if shape and shape[0] > threshold:
            counts["oom"] += 1
            raise RuntimeError("MPS backend out of memory (simulated)")
        return orig(self, arr)

    monkeypatch.setattr(Backend, "to_device", patched)
    return counts


class TestOomRecoveryDeterminism:
    def test_one_sample(self, monkeypatch):
        from nltools.algorithms import one_sample_permutation_test

        rng = np.random.default_rng(11)
        data = rng.standard_normal((20, 30))
        kwargs = {
            "n_permute": N_PERMUTE,
            "device": "gpu",
            "random_state": 7,
            "return_null": True,
        }

        baseline = one_sample_permutation_test(data, **kwargs)
        counts = _patch_flaky_to_device(monkeypatch)
        recovered = one_sample_permutation_test(data, **kwargs)

        assert counts["oom"] > 0, "simulated OOM never triggered"
        np.testing.assert_array_equal(baseline["null_dist"], recovered["null_dist"])
        np.testing.assert_array_equal(baseline["p"], recovered["p"])

    def test_two_sample(self, monkeypatch):
        from nltools.algorithms import two_sample_permutation_test

        rng = np.random.default_rng(12)
        data1 = rng.standard_normal((15, 10))
        data2 = rng.standard_normal((18, 10))
        kwargs = {
            "n_permute": N_PERMUTE,
            "device": "gpu",
            "random_state": 3,
            "return_null": True,
        }

        baseline = two_sample_permutation_test(data1, data2, **kwargs)
        counts = _patch_flaky_to_device(monkeypatch)
        recovered = two_sample_permutation_test(data1, data2, **kwargs)

        assert counts["oom"] > 0, "simulated OOM never triggered"
        np.testing.assert_array_equal(baseline["null_dist"], recovered["null_dist"])
        np.testing.assert_array_equal(baseline["p"], recovered["p"])

    @pytest.mark.parametrize("metric", ["pearson", "spearman", "kendall"])
    def test_correlation(self, monkeypatch, metric):
        from nltools.algorithms import correlation_permutation_test

        rng = np.random.default_rng(13)
        x = rng.standard_normal((20, 5))
        y = 0.4 * x + rng.standard_normal((20, 5))
        kwargs = {
            "metric": metric,
            "n_permute": N_PERMUTE,
            "device": "gpu",
            "random_state": 5,
            "return_null": True,
        }

        baseline = correlation_permutation_test(x, y, **kwargs)
        counts = _patch_flaky_to_device(monkeypatch)
        recovered = correlation_permutation_test(x, y, **kwargs)

        assert counts["oom"] > 0, "simulated OOM never triggered"
        np.testing.assert_array_equal(baseline["null_dist"], recovered["null_dist"])
        np.testing.assert_array_equal(baseline["p"], recovered["p"])

    def test_oom_at_single_item_raises_memoryerror(self, monkeypatch):
        from nltools.algorithms import one_sample_permutation_test

        rng = np.random.default_rng(14)
        data = rng.standard_normal((20, 30))
        # Threshold below n_samples: even a single permutation "OOMs", but the
        # one-time data transfer also fails -> everything device-side OOMs.
        _patch_flaky_to_device(monkeypatch, threshold=0)

        with pytest.raises((MemoryError, RuntimeError)):
            one_sample_permutation_test(
                data, n_permute=50, device="gpu", random_state=1
            )
