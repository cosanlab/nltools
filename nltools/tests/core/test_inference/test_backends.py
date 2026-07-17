"""Tests for backend consistency (NumPy vs PyTorch)."""

import pytest
import numpy as np

from nltools.stats import one_sample_permutation_test
from nltools.tests.core.test_inference import (
    N_PERMUTE_BACKEND,
)
from nltools.algorithms.backends import check_gpu_available


class TestBackends:
    """Test backend consistency (NumPy vs PyTorch)."""

    @pytest.mark.slow
    @pytest.mark.parametrize("n_features", [1, 10])
    def test_backend_consistency(self, backends, n_features):
        """Test that NumPy and PyTorch backends produce same results."""
        if len(backends) < 2:
            pytest.skip("PyTorch not available")

        np.random.seed(42)
        if n_features == 1:
            data = np.random.randn(30)
        else:
            data = np.random.randn(30, n_features)

        results = {}
        for backend in backends:
            # Map backend to device parameter
            if backend == "numpy":
                device = None
            elif backend == "torch":
                device = "gpu"
            else:
                device = "cpu"

            results[backend] = one_sample_permutation_test(
                data, n_permute=N_PERMUTE_BACKEND, device=device, random_state=42
            )

        # Compare results
        np.testing.assert_allclose(
            results["numpy"]["mean"],
            results["torch"]["mean"],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            results["numpy"]["p"],
            results["torch"]["p"],
            rtol=1e-5,
        )

    def test_explicit_numpy_backend(self):
        """Test explicit NumPy backend."""
        np.random.seed(42)
        data = np.random.randn(30)
        result = one_sample_permutation_test(data, device=None, random_state=42)

        assert result["parallel"] is None

    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.skipif(not check_gpu_available()[0], reason="GPU not available")
    def test_explicit_torch_backend(self):
        """Test explicit PyTorch backend."""
        np.random.seed(42)
        data = np.random.randn(30)
        result = one_sample_permutation_test(data, device="gpu", random_state=42)

        assert result["parallel"] == "gpu"
