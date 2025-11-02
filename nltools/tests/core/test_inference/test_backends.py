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
            results[backend] = one_sample_permutation_test(
                data, n_permute=N_PERMUTE_BACKEND, backend=backend, random_state=42
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
            results[backend] = one_sample_permutation_test(
                data, n_permute=N_PERMUTE_BACKEND, backend=backend, random_state=42
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

    def test_auto_backend_selection(self):
        """Test that auto backend selection works."""
        np.random.seed(42)
        data = np.random.randn(30, 10)
        result = one_sample_permutation_test(
            data, n_permute=100, backend="auto", random_state=42
        )

        assert "backend" in result
        assert result["backend"] in ["numpy", "torch-cpu", "torch-cuda", "torch-mps"]

    def test_explicit_numpy_backend(self):
        """Test explicit NumPy backend."""
        np.random.seed(42)
        data = np.random.randn(30)
        result = one_sample_permutation_test(data, backend="numpy", random_state=42)

        assert result["backend"] == "numpy"

    @pytest.mark.skipif(not check_gpu_available()[0], reason="GPU not available")
    def test_explicit_torch_backend(self):
        """Test explicit PyTorch backend."""
        np.random.seed(42)
        data = np.random.randn(30)
        result = one_sample_permutation_test(data, backend="torch", random_state=42)

        assert result["backend"].startswith("torch")
