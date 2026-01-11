"""
Tests for inference utility functions.

Tests helper functions like _generate_sign_flips, _compute_pvalue,
and memory management utilities.
"""

import pytest
import numpy as np
import warnings
import multiprocessing

from nltools.algorithms.inference import _generate_sign_flips, _compute_pvalue
from nltools.algorithms.inference.utils import (
    _auto_n_jobs_cpu,
    _verify_n_jobs_memory_constraint,
    _auto_batch_size,
    _estimate_data_size_mb,
)


class TestHelperFunctions:
    """Test helper functions for correctness."""

    def test_generate_sign_flips_shape(self):
        """Test that sign flips have correct shape."""
        sign_flips = _generate_sign_flips(n_permute=100, n_samples=30, random_state=42)
        assert sign_flips.shape == (100, 30)

    def test_generate_sign_flips_values(self):
        """Test that sign flips only contain +1 and -1."""
        sign_flips = _generate_sign_flips(n_permute=100, n_samples=30, random_state=42)
        assert np.all(np.isin(sign_flips, [-1, 1]))

    def test_generate_sign_flips_deterministic(self):
        """Test that sign flips are deterministic with fixed seed."""
        sf1 = _generate_sign_flips(n_permute=100, n_samples=30, random_state=42)
        sf2 = _generate_sign_flips(n_permute=100, n_samples=30, random_state=42)
        np.testing.assert_array_equal(sf1, sf2)

    def test_generate_sign_flips_random(self):
        """Test that sign flips are different with different seeds."""
        sf1 = _generate_sign_flips(n_permute=100, n_samples=30, random_state=42)
        sf2 = _generate_sign_flips(n_permute=100, n_samples=30, random_state=43)
        assert not np.array_equal(sf1, sf2)

    def test_compute_pvalue_two_tailed(self):
        """Test two-tailed p-value computation with correction factor."""
        # With correction factor: (count + 1) / (n_permute + 1)
        np.random.seed(42)
        null_dist = np.random.randn(10000, 1)
        obs_stat = np.array([np.percentile(null_dist, 90)])
        p = _compute_pvalue(obs_stat, null_dist, tail=2)
        # Should be moderate p-value (not extreme)
        assert 0.1 < p[0] < 0.3

    def test_compute_pvalue_one_tailed(self):
        """Test one-tailed p-value computation with correction factor."""
        np.random.seed(42)
        null_dist = np.random.randn(10000, 1)
        obs_stat = np.array([np.percentile(null_dist, 95)])  # 95th percentile
        p = _compute_pvalue(obs_stat, null_dist, tail=1)
        # With correction factor, should be slightly > 0.05
        assert 0.04 < p[0] < 0.07

    def test_compute_pvalue_extreme(self):
        """Test p-value for extreme observed statistic."""
        # Observed far from null → p-value should be minimum: 1/(n+1)
        null_dist = np.random.randn(1000, 1)
        obs_stat = np.array([10.0])  # Very extreme (essentially no null values exceed)
        p = _compute_pvalue(obs_stat, null_dist, tail=2)
        # Minimum p-value with correction: 1/(1000+1) ≈ 0.001
        assert p[0] == 1.0 / 1001.0

    def test_compute_pvalue_multifeature(self):
        """Test p-value computation for multiple features."""
        np.random.seed(42)
        null_dist = np.random.randn(1000, 10)
        # Use moderate percentile values for testing
        obs_stat = np.percentile(null_dist, 90, axis=0)  # 90th percentile
        p = _compute_pvalue(obs_stat, null_dist, tail=2)
        assert p.shape == (10,)
        # All p-values should be valid (between 0 and 1)
        assert np.all((p > 0) & (p <= 1))

    def test_compute_pvalue_invalid_tail(self):
        """Test that invalid tail raises error."""
        null_dist = np.random.randn(100, 1)
        obs_stat = np.array([0.0])
        with pytest.raises(ValueError, match="tail must be 1 or 2"):
            _compute_pvalue(obs_stat, null_dist, tail=3)


class TestMemoryManagement:
    """Test memory management utilities for CPU and GPU parallelization."""

    # ========================================================================
    # CPU Memory Management Tests
    # ========================================================================

    def test_auto_n_jobs_cpu_small_data(self):
        """Test auto-detection with small data (should use many cores)."""
        n_jobs = _auto_n_jobs_cpu(
            data_size_mb=1.0,  # Small data
            n_permute=1000,
            max_memory_gb=8.0,
        )
        # Should use multiple cores
        assert n_jobs >= 1
        assert n_jobs <= multiprocessing.cpu_count()

    def test_auto_n_jobs_cpu_large_data(self):
        """Test auto-detection with large data (should limit workers)."""
        n_jobs = _auto_n_jobs_cpu(
            data_size_mb=100.0,  # Large data
            n_permute=1000,
            max_memory_gb=8.0,
        )
        # Should limit workers due to memory
        assert n_jobs >= 1
        assert n_jobs <= multiprocessing.cpu_count()

    def test_auto_n_jobs_cpu_respects_max_jobs(self):
        """Test that max_jobs parameter is respected."""
        max_jobs = 2
        n_jobs = _auto_n_jobs_cpu(
            data_size_mb=0.1,  # Small data (would use many cores)
            n_permute=1000,
            max_memory_gb=8.0,
            max_jobs=max_jobs,
        )
        assert n_jobs <= max_jobs

    def test_verify_n_jobs_memory_constraint_allows_requested(self):
        """Test that memory constraint allows requested n_jobs when memory permits."""
        # Small data, plenty of memory
        n_jobs = _verify_n_jobs_memory_constraint(
            requested_n_jobs=4,
            data_size_mb=1.0,  # Small data
            n_permute=1000,
            max_memory_gb=8.0,  # Plenty of memory
        )
        assert n_jobs == 4  # Should use requested value

    def test_verify_n_jobs_memory_constraint_reduces_workers(self):
        """Test that memory constraint reduces workers when memory is limited."""
        # Large data, limited memory
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            n_jobs = _verify_n_jobs_memory_constraint(
                requested_n_jobs=8,
                data_size_mb=100.0,  # Large data per worker
                n_permute=1000,
                max_memory_gb=1.0,  # Very limited memory (1 GB)
            )

            # Should reduce workers
            assert n_jobs < 8
            assert n_jobs >= 1  # But never below min_jobs

            # Should warn if reduction is significant (>20%)
            if (8 - n_jobs) / 8 >= 0.2:
                assert len(w) > 0
                assert any(
                    "exceeds memory limit" in str(warning.message) for warning in w
                )

    def test_verify_n_jobs_memory_constraint_respects_cpu_limit(self):
        """Test that memory constraint respects CPU count limit."""
        max_cpu = multiprocessing.cpu_count()
        # Request more workers than CPU count
        n_jobs = _verify_n_jobs_memory_constraint(
            requested_n_jobs=max_cpu + 10,  # More than CPU count
            data_size_mb=0.1,  # Small data
            n_permute=1000,
            max_memory_gb=8.0,  # Plenty of memory
        )
        # Should be capped at CPU count
        assert n_jobs <= max_cpu

    def test_verify_n_jobs_memory_constraint_min_jobs(self):
        """Test that memory constraint never goes below min_jobs."""
        # Even with very restrictive memory, should use at least min_jobs=1
        n_jobs = _verify_n_jobs_memory_constraint(
            requested_n_jobs=8,
            data_size_mb=10000.0,  # Huge data per worker
            n_permute=1000,
            max_memory_gb=0.01,  # Extremely limited memory (10 MB)
            min_jobs=1,
        )
        assert n_jobs >= 1  # Never below min_jobs

    def test_auto_n_jobs_cpu_vs_verify_consistency(self):
        """Test that auto-detection and verification use same logic."""
        data_size_mb = 2.0
        n_permute = 1000
        max_memory_gb = 4.0

        # Auto-detect optimal n_jobs
        auto_n_jobs = _auto_n_jobs_cpu(
            data_size_mb=data_size_mb,
            n_permute=n_permute,
            max_memory_gb=max_memory_gb,
        )

        # Verify with same parameters (should allow auto-detected value)
        verified_n_jobs = _verify_n_jobs_memory_constraint(
            requested_n_jobs=auto_n_jobs,
            data_size_mb=data_size_mb,
            n_permute=n_permute,
            max_memory_gb=max_memory_gb,
        )

        # Should use same value (no reduction needed)
        assert verified_n_jobs == auto_n_jobs

    def test_verify_n_jobs_memory_constraint_no_warning_small_reduction(self):
        """Test that small reductions don't trigger warnings."""
        # Small reduction (<20% threshold)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            n_jobs = _verify_n_jobs_memory_constraint(
                requested_n_jobs=5,
                data_size_mb=50.0,  # Moderate data
                n_permute=1000,
                max_memory_gb=1.0,  # Limited memory
                warn_threshold=0.2,  # 20% threshold
            )

            # Should reduce but not warn if reduction < 20%
            if (5 - n_jobs) / 5 < 0.2:
                # No warnings should be raised
                assert len(w) == 0 or not any(
                    "exceeds memory limit" in str(warning.message) for warning in w
                )

    # ========================================================================
    # GPU Memory Management Tests
    # ========================================================================

    def test_auto_batch_size_small_problem(self):
        """Test batch size calculation for small problems."""
        # Small problem: All permutations fit in one batch
        batch_size, n_batches = _auto_batch_size(
            n_permute=1000,
            n_samples=30,
            n_features=1000,
            max_memory_gb=4.0,
        )
        assert batch_size >= 100  # Minimum batch size
        assert n_batches >= 1

    def test_auto_batch_size_large_problem(self):
        """Test batch size calculation for large problems."""
        # Large problem: Need multiple batches
        batch_size, n_batches = _auto_batch_size(
            n_permute=10000,
            n_samples=30,
            n_features=50000,
            max_memory_gb=4.0,
        )
        assert batch_size >= 100  # Minimum batch size
        assert n_batches > 1  # Should need multiple batches

    def test_auto_batch_size_memory_budget(self):
        """Test that different memory budgets produce different batch sizes."""
        # Smaller memory budget should produce smaller batches
        batch_small, _ = _auto_batch_size(
            n_permute=5000,
            n_samples=30,
            n_features=10000,
            max_memory_gb=2.0,
        )
        batch_large, _ = _auto_batch_size(
            n_permute=5000,
            n_samples=30,
            n_features=10000,
            max_memory_gb=8.0,
        )
        assert batch_large >= batch_small  # More memory = larger batches

    def test_auto_batch_size_minimum(self):
        """Test that batch size never goes below minimum."""
        # Even with very restrictive memory, should use minimum batch size
        batch_size, n_batches = _auto_batch_size(
            n_permute=5000,
            n_samples=30,
            n_features=100000,  # Huge features
            max_memory_gb=0.01,  # Very small memory
        )
        assert batch_size >= 100  # Minimum batch size

    def test_auto_batch_size_maximum(self):
        """Test that batch size never exceeds n_permute."""
        batch_size, n_batches = _auto_batch_size(
            n_permute=1000,
            n_samples=30,
            n_features=100,
            max_memory_gb=100.0,  # Huge memory budget
        )
        assert batch_size <= 1000  # Never exceeds n_permute
        assert n_batches >= 1

    # ========================================================================
    # Data Size Estimation Tests
    # ========================================================================

    def test_estimate_data_size_mb_float32(self):
        """Test data size estimation for float32 arrays."""
        data = np.random.randn(1000, 100).astype(np.float32)
        size_mb = _estimate_data_size_mb(data)
        # Should be approximately: 1000 * 100 * 4 bytes / 1024^2
        expected_size_mb = (1000 * 100 * 4) / (1024**2)
        assert abs(size_mb - expected_size_mb) < 0.1  # Within 0.1 MB

    def test_estimate_data_size_mb_float64(self):
        """Test data size estimation for float64 arrays."""
        data = np.random.randn(1000, 100).astype(np.float64)
        size_mb = _estimate_data_size_mb(data)
        # Should be approximately: 1000 * 100 * 8 bytes / 1024^2
        expected_size_mb = (1000 * 100 * 8) / (1024**2)
        assert abs(size_mb - expected_size_mb) < 0.1  # Within 0.1 MB

    def test_estimate_data_size_mb_empty(self):
        """Test data size estimation for empty arrays."""
        data = np.array([])
        size_mb = _estimate_data_size_mb(data)
        assert size_mb == 0.0
