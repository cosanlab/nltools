"""
Tests for inference utility functions.

Tests helper functions like _generate_sign_flips, _compute_pvalue,
and memory management utilities.
"""

import pytest
import numpy as np
import multiprocessing

from nltools.algorithms.inference.utils import _generate_sign_flips
from nltools.algorithms.validation import _compute_pvalue
from nltools.algorithms.backends import _auto_n_jobs_cpu, _estimate_data_size_mb


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
        with pytest.raises(ValueError, match="tail must be"):
            _compute_pvalue(obs_stat, null_dist, tail=3)

    def test_compute_pvalue_string_tail_options(self):
        """Test string tail options for MCP compatibility (GH #315)."""
        np.random.seed(42)
        null_dist = np.random.randn(1000, 1)
        obs_stat = np.array([1.5])  # Positive value

        # Test 'two' is equivalent to 2
        p_two_str = _compute_pvalue(obs_stat, null_dist, tail="two")
        p_two_int = _compute_pvalue(obs_stat, null_dist, tail=2)
        assert p_two_str[0] == p_two_int[0]

        # Test 'upper' gives consistent direction (null >= obs)
        p_upper = _compute_pvalue(obs_stat, null_dist, tail="upper")
        assert 0 < p_upper[0] < 0.15  # Should be small for positive obs

        # Test 'lower' gives opposite result (null <= obs)
        p_lower = _compute_pvalue(obs_stat, null_dist, tail="lower")
        assert p_lower[0] > 0.85  # Should be large for positive obs

        # Test -1 is equivalent to 'lower'
        p_lower_int = _compute_pvalue(obs_stat, null_dist, tail=-1)
        assert p_lower[0] == p_lower_int[0]

    def test_compute_pvalue_consistent_direction_for_mcp(self):
        """Test that 'upper'/'lower' give consistent direction across features.

        This is the key fix for GH #315 - when doing MCP correction, all tests
        should use the same direction regardless of observed statistic sign.
        """
        np.random.seed(42)
        null_dist = np.random.randn(1000, 3)
        # Mixed signs: positive, zero, negative
        obs_stat = np.array([2.0, 0.0, -2.0])

        # With 'upper', all tests use same comparison (null >= obs)
        p_upper = _compute_pvalue(obs_stat, null_dist, tail="upper")
        # Positive obs -> small p, negative obs -> large p
        assert p_upper[0] < 0.1  # 2.0 is high, few null values exceed it
        assert p_upper[2] > 0.9  # -2.0 is low, most null values exceed it

        # With 'lower', all tests use same comparison (null <= obs)
        p_lower = _compute_pvalue(obs_stat, null_dist, tail="lower")
        # Positive obs -> large p, negative obs -> small p
        assert p_lower[0] > 0.9
        assert p_lower[2] < 0.1


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

    def test_auto_n_jobs_cpu_caps_at_requested_workers(self):
        """`max_jobs=` covers the old `_verify_n_jobs_memory_constraint` use.

        Capping an explicit request is `min(requested, cpu_count)` passed as
        `max_jobs` — one home for the memory math (the deleted twin duplicated
        it verbatim and had no production caller).
        """
        max_cpu = multiprocessing.cpu_count()
        requested = min(4, max_cpu)
        n_jobs = _auto_n_jobs_cpu(
            data_size_mb=1.0,  # Small data, plenty of memory
            n_permute=1000,
            max_memory_gb=8.0,
            max_jobs=min(requested, max_cpu),
        )
        assert n_jobs == requested

    def test_auto_n_jobs_cpu_raises_when_one_worker_exceeds_budget(self):
        """An explicit budget must fit the estimated working set of one worker."""
        with pytest.raises(ValueError, match="one worker requires"):
            _auto_n_jobs_cpu(
                data_size_mb=10000.0,
                n_permute=1000,
                max_memory_gb=0.01,
            )

    def test_auto_n_jobs_cpu_uses_decimal_gigabyte_budget(self):
        """A 1 GB cap must not be widened to 1 GiB during worker sizing."""
        with pytest.raises(ValueError, match="one worker requires"):
            _auto_n_jobs_cpu(
                data_size_mb=334.0,
                n_permute=100,
                max_memory_gb=1.0,
            )

    # ========================================================================
    # GPU Memory Management Tests
    # ========================================================================

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


class TestMeasuredBudgetDefaults:
    """max_memory_gb=None means: measure the machine, don't assume a fixed budget."""

    def test_auto_n_jobs_cpu_measured_default(self):
        n_jobs = _auto_n_jobs_cpu(1.0, 100, max_memory_gb=None)
        assert n_jobs >= 1
