"""Tests for backend consistency (NumPy vs PyTorch)."""

import pytest
import numpy as np

from nltools.algorithms.inference import one_sample_permutation_test
from nltools.tests.core.test_inference import (
    N_PERMUTE_BACKEND,
)
from nltools.backends import check_gpu_available


class TestBackends:
    """Test backend consistency (NumPy vs PyTorch)."""

    def test_backend_consistency_single_feature(self, backends):
        """Test that NumPy and PyTorch backends produce same results."""
        if len(backends) < 2:
            pytest.skip("PyTorch not available")

        np.random.seed(42)
        data = np.random.randn(30)

        results = {}
        for backend in backends:
            # Map backend to parallel parameter
            if backend == "numpy":
                parallel = None
            elif backend == "torch":
                parallel = "gpu"
            else:
                parallel = "cpu"

            results[backend] = one_sample_permutation_test(
                data, n_permute=N_PERMUTE_BACKEND, parallel=parallel, random_state=42
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

    def test_backend_consistency_multi_feature(self, backends):
        """Test backend consistency for multi-feature data."""
        if len(backends) < 2:
            pytest.skip("PyTorch not available")

        np.random.seed(42)
        data = np.random.randn(30, 10)

        results = {}
        for backend in backends:
            # Map backend to parallel parameter
            if backend == "numpy":
                parallel = None
            elif backend == "torch":
                parallel = "gpu"
            else:
                parallel = "cpu"

            results[backend] = one_sample_permutation_test(
                data, n_permute=N_PERMUTE_BACKEND, parallel=parallel, random_state=42
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
        result = one_sample_permutation_test(data, parallel=None, random_state=42)

        assert result["parallel"] is None

    @pytest.mark.skipif(not check_gpu_available()[0], reason="GPU not available")
    def test_explicit_torch_backend(self):
        """Test explicit PyTorch backend."""
        np.random.seed(42)
        data = np.random.randn(30)
        result = one_sample_permutation_test(data, parallel="gpu", random_state=42)

        assert result["parallel"] == "gpu"
