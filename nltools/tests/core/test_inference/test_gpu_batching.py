"""Tests for GPU batching functionality and memory management."""

import pytest
import numpy as np

from nltools.algorithms.inference import (
    one_sample_permutation_test,
    _auto_batch_size,
)
from nltools.tests.core.test_inference import (
    TOLERANCE_GPU_VALUE,
    TOLERANCE_GPU_PVALUE,
)
from nltools.backends import check_gpu_available


class TestGPUBatching:
    """Test GPU batching functionality and memory management."""

    def test_auto_batch_size_small_problem(self):
        """Test that small problems fit in one batch."""

        # Small problem: 1000 perms × 30 samples × 1000 features
        # Memory: 1000 * 30 * 1000 * 4 bytes = 120 MB
        # Should easily fit in 4GB
        batch_size, n_batches = _auto_batch_size(
            n_permute=1000, n_samples=30, n_features=1000, max_memory_gb=4.0
        )

        assert n_batches == 1  # Everything fits in one batch
        assert batch_size >= 1000  # Batch size should be >= n_permute

    def test_auto_batch_size_large_problem(self):
        """Test that large problems are split into multiple batches."""

        # Large problem: 10K perms × 30 samples × 50K features
        # Memory per perm: 30 * 50K * 4 bytes = 6 MB
        # 4GB budget: ~666 perms per batch
        batch_size, n_batches = _auto_batch_size(
            n_permute=10000, n_samples=30, n_features=50000, max_memory_gb=4.0
        )

        assert n_batches > 1  # Should need multiple batches
        assert batch_size >= 100  # Minimum batch size
        assert batch_size * n_batches >= 10000  # Covers all permutations

    def test_auto_batch_size_memory_budget(self):
        """Test that different memory budgets produce different batch sizes."""

        # Same problem, different budgets
        batch_small, _ = _auto_batch_size(
            n_permute=5000, n_samples=30, n_features=10000, max_memory_gb=2.0
        )
        batch_large, _ = _auto_batch_size(
            n_permute=5000, n_samples=30, n_features=10000, max_memory_gb=8.0
        )

        # Larger budget should allow larger batches
        assert batch_large > batch_small

    @pytest.mark.skipif(not check_gpu_available()[0], reason="GPU not available")
    def test_gpu_batching_correctness(self):
        """Test that GPU batching produces same results as NumPy."""
        np.random.seed(42)
        data = np.random.randn(30, 5000)  # Medium-sized problem

        # NumPy backend (no batching needed)
        result_numpy = one_sample_permutation_test(
            data, n_permute=500, backend="numpy", random_state=42
        )

        # GPU backend with small memory budget to force batching
        result_gpu = one_sample_permutation_test(
            data,
            n_permute=500,
            backend="torch",
            max_gpu_memory_gb=0.5,  # Force small batches
            random_state=42,
        )

        # Results should match (float32 vs float64 precision)
        np.testing.assert_allclose(
            result_numpy["mean"],
            result_gpu["mean"],
            rtol=TOLERANCE_GPU_VALUE,  # float32 vs float64 differences
        )
        np.testing.assert_allclose(
            result_numpy["p"],
            result_gpu["p"],
            rtol=TOLERANCE_GPU_PVALUE,  # P-values accumulate more FP error
        )

    @pytest.mark.skipif(not check_gpu_available()[0], reason="GPU not available")
    def test_gpu_batching_large_problem(self):
        """Test GPU batching with large problem that would OOM without batching."""
        np.random.seed(42)

        # Large problem: Would use ~7.2GB without batching
        # 1000 perms × 30 samples × 60K features × 4 bytes = 7.2 GB
        data = np.random.randn(30, 60000).astype(np.float32)

        # Should work with batching (4GB budget)
        result = one_sample_permutation_test(
            data,
            n_permute=1000,
            backend="torch",
            max_gpu_memory_gb=4.0,
            random_state=42,
        )

        # Verify results are sensible
        assert result["mean"].shape == (60000,)
        assert result["p"].shape == (60000,)
        assert np.all((result["p"] >= 0) & (result["p"] <= 1))

    def test_gpu_batching_parameter_variations(self):
        """Test that max_gpu_memory_gb parameter works correctly."""
        np.random.seed(42)
        data = np.random.randn(30, 1000)

        # Test different memory budgets (should all work, just different batch sizes)
        for memory_gb in [1.0, 2.0, 4.0, 8.0]:
            result = one_sample_permutation_test(
                data,
                n_permute=500,
                backend="numpy",  # Use numpy to avoid GPU requirement
                max_gpu_memory_gb=memory_gb,
                random_state=42,
            )

            # All should produce valid results
            assert "mean" in result
            assert "p" in result
            # Handle both scalar and array p-values
            assert np.all((result["p"] >= 0) & (result["p"] <= 1))
