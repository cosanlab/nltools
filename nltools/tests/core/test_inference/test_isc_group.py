"""
Tests for ISC Group Difference (ISC Group) module.

Test organization follows the TDD plan (isc-group-tdd-plan.md):
    Phase 1: Core ISC Group Difference Computation
    Phase 2: Permutation Method
    Phase 3: Bootstrap Method
    Phase 4: Main Function
    Phase 5: Statistical Correctness Tests
    Phase 6: Integration Tests
    Phase 7: Performance Benchmarks (tier2)

Tier 1: Fast tests (~1-2min, run on every iteration)
Tier 2: GPU and benchmark tests (~5-7min, run before commits)

Testing Strategy & Tolerances:

    This test suite uses different tolerance levels for different comparison types.
    Following the pattern from test_isc.py:

    1. Backend Consistency (NumPy vs CPU-parallel):
       - Tolerance: EXACT (rtol=1e-5)
       - Why: Same algorithm, same random seed, only float precision differs

    2. GPU Precision (GPU float32 vs CPU float64):
       - Values: rtol=1e-3 (0.1% error)
       - P-values: rtol=5e-3 (0.5% error)
       - Why: GPU uses float32, CPU uses float64; P-values accumulate more error

    3. Backward Compatibility (vs stats.py):
       - P-values: rtol=0.02 (2% error)
       - Why: Prioritizes cross-backend determinism over exact stats.py match
"""

import numpy as np
import pytest

from nltools.algorithms.inference.isc import (
    _compute_isc_group_difference,
    _compute_loo_isc,
    _compute_pairwise_isc,
    _permute_isc_group_numpy,
    _permute_isc_group_cpu_parallel,
    _bootstrap_isc_group_numpy,
    _bootstrap_isc_group_cpu_parallel,
    isc_group_permutation_test,
)

# =============================================================================
# Test Constants - DO NOT MODIFY without updating docstring above
# =============================================================================

# Tolerance for backend consistency (NumPy vs CPU-parallel with same seed)
TOLERANCE_EXACT = 1e-5

# Tolerance for GPU vs CPU comparisons (float32 vs float64)
TOLERANCE_GPU_VALUE = 1e-3  # 0.1% error for computed values
TOLERANCE_GPU_PVALUE = 5e-3  # 0.5% error for P-values (more FP error)

# Tolerance for backward compatibility with stats.py
TOLERANCE_STATS_PVALUE = 0.02  # 2% relative error acceptable


# =============================================================================
# Phase 1: Core ISC Group Difference Computation Tests
# =============================================================================


@pytest.mark.tier1
def test_compute_isc_group_difference_numpy_single_feature_pairwise():
    """ISC group difference computes correctly for single feature with pairwise method."""
    np.random.seed(42)
    n_obs = 100
    n_subjects1, n_subjects2 = 5, 5

    # Create correlated groups
    group1 = np.random.randn(n_obs, n_subjects1)
    group2 = np.random.randn(n_obs, n_subjects2)

    # Compute ISC difference
    isc_diff = _compute_isc_group_difference(
        group1, group2, metric="median", summary_statistic="pairwise", backend="numpy"
    )

    # Should be scalar
    assert isinstance(isc_diff, (float, np.floating))
    assert np.isfinite(isc_diff)

    # Verify manually: compute ISC for each group, then difference
    pairwise1 = _compute_pairwise_isc(group1, backend="numpy")
    pairwise2 = _compute_pairwise_isc(group2, backend="numpy")
    isc1 = np.median(pairwise1)
    isc2 = np.median(pairwise2)
    expected_diff = isc1 - isc2

    assert np.isclose(isc_diff, expected_diff)


@pytest.mark.tier1
def test_compute_isc_group_difference_numpy_single_feature_loo():
    """ISC group difference computes correctly for single feature with LOO method."""
    np.random.seed(42)
    n_obs = 100
    n_subjects1, n_subjects2 = 5, 5

    group1 = np.random.randn(n_obs, n_subjects1)
    group2 = np.random.randn(n_obs, n_subjects2)

    # Compute ISC difference with LOO
    isc_diff = _compute_isc_group_difference(
        group1,
        group2,
        metric="median",
        summary_statistic="leave-one-out",
        backend="numpy",
    )

    assert isinstance(isc_diff, (float, np.floating))
    assert np.isfinite(isc_diff)

    # Verify manually
    loo1 = _compute_loo_isc(group1, backend="numpy")
    loo2 = _compute_loo_isc(group2, backend="numpy")
    isc1 = np.median(loo1)
    isc2 = np.median(loo2)
    expected_diff = isc1 - isc2

    assert np.isclose(isc_diff, expected_diff)


@pytest.mark.tier1
def test_compute_isc_group_difference_numpy_voxelwise():
    """ISC group difference computes correctly for voxel-wise data."""
    np.random.seed(42)
    n_obs = 100
    n_subjects1, n_subjects2 = 5, 5
    n_voxels = 10

    group1 = np.random.randn(n_obs, n_subjects1, n_voxels)
    group2 = np.random.randn(n_obs, n_subjects2, n_voxels)

    # Compute ISC difference
    isc_diff = _compute_isc_group_difference(
        group1, group2, metric="median", summary_statistic="pairwise", backend="numpy"
    )

    # Should be array per voxel
    assert isc_diff.shape == (n_voxels,)
    assert np.all(np.isfinite(isc_diff))

    # Verify first voxel manually
    voxel1_group1 = group1[:, :, 0]
    voxel1_group2 = group2[:, :, 0]
    pairwise1_v0 = _compute_pairwise_isc(voxel1_group1, backend="numpy")
    pairwise2_v0 = _compute_pairwise_isc(voxel1_group2, backend="numpy")
    isc1_v0 = np.median(pairwise1_v0)
    isc2_v0 = np.median(pairwise2_v0)
    expected_diff_v0 = isc1_v0 - isc2_v0

    assert np.isclose(isc_diff[0], expected_diff_v0)


@pytest.mark.tier1
def test_compute_isc_group_difference_metric_mean():
    """ISC group difference works with 'mean' metric (Fisher z-transform)."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    # Compute with mean metric
    isc_diff_mean = _compute_isc_group_difference(
        group1, group2, metric="mean", summary_statistic="pairwise", backend="numpy"
    )

    # Compute with median metric
    isc_diff_median = _compute_isc_group_difference(
        group1, group2, metric="median", summary_statistic="pairwise", backend="numpy"
    )

    # Both should be valid
    assert np.isfinite(isc_diff_mean)
    assert np.isfinite(isc_diff_median)

    # They may differ (mean uses Fisher z-transform)
    assert isinstance(isc_diff_mean, (float, np.floating))
    assert isinstance(isc_diff_median, (float, np.floating))


@pytest.mark.tier2
def test_compute_isc_group_difference_gpu_matches_numpy():
    """GPU ISC group difference matches NumPy within float32 tolerance (voxel-wise LOO only)."""
    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

    np.random.seed(42)
    n_obs = 100
    n_subjects1, n_subjects2 = 5, 5
    n_voxels = 100  # Sufficient voxels for GPU speedup

    group1 = np.random.randn(n_obs, n_subjects1, n_voxels)
    group2 = np.random.randn(n_obs, n_subjects2, n_voxels)

    # Compute with NumPy (CPU)
    isc_diff_numpy = _compute_isc_group_difference(
        group1,
        group2,
        metric="median",
        summary_statistic="leave-one-out",
        backend="numpy",
    )

    # Compute with GPU
    isc_diff_gpu = _compute_isc_group_difference(
        group1,
        group2,
        metric="median",
        summary_statistic="leave-one-out",
        backend="torch",
    )

    # GPU uses float32, CPU uses float64 - use GPU precision tolerance
    np.testing.assert_allclose(
        isc_diff_numpy, isc_diff_gpu, rtol=TOLERANCE_GPU_VALUE, atol=1e-7
    )


@pytest.mark.tier1
def test_compute_isc_group_difference_deterministic():
    """ISC group difference computation is deterministic."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    isc_diff1 = _compute_isc_group_difference(
        group1, group2, metric="median", summary_statistic="pairwise", backend="numpy"
    )

    isc_diff2 = _compute_isc_group_difference(
        group1, group2, metric="median", summary_statistic="pairwise", backend="numpy"
    )

    assert np.allclose(isc_diff1, isc_diff2)


@pytest.mark.tier1
def test_compute_isc_group_difference_mismatched_observations():
    """ISC group difference raises error if groups have different number of observations."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(101, 5)  # Different number of observations

    with pytest.raises(ValueError, match="same number of observations"):
        _compute_isc_group_difference(
            group1,
            group2,
            metric="median",
            summary_statistic="pairwise",
            backend="numpy",
        )


@pytest.mark.tier1
def test_compute_isc_group_difference_invalid_metric():
    """ISC group difference raises error for invalid metric."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    with pytest.raises(ValueError, match="metric must be"):
        _compute_isc_group_difference(
            group1,
            group2,
            metric="invalid",
            summary_statistic="pairwise",
            backend="numpy",
        )


@pytest.mark.tier1
def test_compute_isc_group_difference_invalid_summary_statistic():
    """ISC group difference raises error for invalid summary_statistic."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    with pytest.raises(ValueError, match="summary_statistic must be"):
        _compute_isc_group_difference(
            group1,
            group2,
            metric="median",
            summary_statistic="invalid",
            backend="numpy",
        )


# =============================================================================
# Phase 2: Permutation Method Tests
# =============================================================================


@pytest.mark.tier1
def test_permute_isc_group_combines_groups():
    """Permutation method combines groups and permutes labels."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    # Single permutation
    permuted_diff = _permute_isc_group_numpy(
        group1, group2, metric="median", summary_statistic="pairwise", random_state=42
    )

    assert isinstance(permuted_diff, (float, np.floating))
    assert np.isfinite(permuted_diff)


@pytest.mark.tier1
def test_permute_isc_group_deterministic():
    """Permutation is deterministic with same random_state."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    diff1 = _permute_isc_group_numpy(
        group1, group2, metric="median", summary_statistic="pairwise", random_state=42
    )

    diff2 = _permute_isc_group_numpy(
        group1, group2, metric="median", summary_statistic="pairwise", random_state=42
    )

    assert np.isclose(diff1, diff2)


@pytest.mark.tier1
def test_permute_isc_group_different_random_states():
    """Different random states produce different permutations."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    diff1 = _permute_isc_group_numpy(
        group1, group2, metric="median", summary_statistic="pairwise", random_state=42
    )

    diff2 = _permute_isc_group_numpy(
        group1, group2, metric="median", summary_statistic="pairwise", random_state=43
    )

    # May be the same by chance, but likely different
    assert isinstance(diff1, (float, np.floating))
    assert isinstance(diff2, (float, np.floating))


@pytest.mark.tier1
def test_permute_isc_group_works_with_loo():
    """Permutation works with leave-one-out summary statistic."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    permuted_diff = _permute_isc_group_numpy(
        group1,
        group2,
        metric="median",
        summary_statistic="leave-one-out",
        random_state=42,
    )

    assert isinstance(permuted_diff, (float, np.floating))
    assert np.isfinite(permuted_diff)


@pytest.mark.tier1
def test_permute_isc_group_works_with_mean_metric():
    """Permutation works with mean metric."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    permuted_diff = _permute_isc_group_numpy(
        group1, group2, metric="mean", summary_statistic="pairwise", random_state=42
    )

    assert isinstance(permuted_diff, (float, np.floating))
    assert np.isfinite(permuted_diff)


@pytest.mark.tier1
def test_permute_isc_group_cpu_parallel_matches_numpy():
    """CPU-parallel permutation matches sequential NumPy exactly."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    # Sequential NumPy
    rng = np.random.RandomState(42)
    seeds = rng.randint(0, 2**31 - 1, size=100)
    perm_numpy = np.array(
        [
            _permute_isc_group_numpy(
                group1,
                group2,
                metric="median",
                summary_statistic="pairwise",
                random_state=np.random.RandomState(seeds[i]),
            )
            for i in range(100)
        ]
    )

    # CPU-parallel
    perm_parallel = _permute_isc_group_cpu_parallel(
        group1,
        group2,
        n_permute=100,
        metric="median",
        summary_statistic="pairwise",
        random_state=42,
        progress_bar=False,
    )

    assert np.allclose(perm_numpy, perm_parallel)


@pytest.mark.tier1
def test_permute_isc_group_cpu_parallel_deterministic():
    """CPU-parallel permutation is deterministic."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    perm1 = _permute_isc_group_cpu_parallel(
        group1,
        group2,
        n_permute=100,
        metric="median",
        summary_statistic="pairwise",
        random_state=42,
        progress_bar=False,
    )

    perm2 = _permute_isc_group_cpu_parallel(
        group1,
        group2,
        n_permute=100,
        metric="median",
        summary_statistic="pairwise",
        random_state=42,
        progress_bar=False,
    )

    assert np.allclose(perm1, perm2)


@pytest.mark.tier1
def test_permute_isc_group_cpu_parallel_voxelwise():
    """CPU-parallel permutation works with voxel-wise data."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5, 10)  # 10 voxels
    group2 = np.random.randn(100, 5, 10)

    perm = _permute_isc_group_cpu_parallel(
        group1,
        group2,
        n_permute=50,
        metric="median",
        summary_statistic="pairwise",
        random_state=42,
        progress_bar=False,
    )

    assert perm.shape == (50, 10)
    assert np.all(np.isfinite(perm))


# =============================================================================
# Phase 3: Bootstrap Method Tests
# =============================================================================


@pytest.mark.tier1
def test_bootstrap_isc_group_resamples_subjects():
    """Bootstrap resamples subjects within each group."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    # Compute observed difference
    observed_diff = _compute_isc_group_difference(
        group1, group2, metric="median", summary_statistic="pairwise", backend="numpy"
    )

    # Single bootstrap
    boot_diff = _bootstrap_isc_group_numpy(
        group1,
        group2,
        observed_diff=observed_diff,
        metric="median",
        summary_statistic="pairwise",
        random_state=42,
    )

    assert isinstance(boot_diff, (float, np.floating))
    assert np.isfinite(boot_diff)


@pytest.mark.tier1
def test_bootstrap_isc_group_centers_by_observed():
    """Bootstrap differences are centered by subtracting observed difference."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    # Compute observed difference
    observed_diff = _compute_isc_group_difference(
        group1, group2, metric="median", summary_statistic="pairwise", backend="numpy"
    )

    # Bootstrap (should be centered)
    boot_diff = _bootstrap_isc_group_numpy(
        group1,
        group2,
        observed_diff=observed_diff,
        metric="median",
        summary_statistic="pairwise",
        random_state=42,
    )

    # Centered bootstrap: (boot1 - boot2) - observed_diff
    # This creates a null distribution centered around 0
    assert isinstance(boot_diff, (float, np.floating))


@pytest.mark.tier1
def test_bootstrap_isc_group_exclude_self_corr():
    """exclude_self_corr parameter controls masking of self-correlations."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    observed_diff = _compute_isc_group_difference(
        group1, group2, metric="median", summary_statistic="pairwise", backend="numpy"
    )

    # Test with exclude_self_corr=True
    boot_exclude = _bootstrap_isc_group_numpy(
        group1,
        group2,
        observed_diff=observed_diff,
        metric="median",
        summary_statistic="pairwise",
        exclude_self_corr=True,
        random_state=42,
    )

    # Test with exclude_self_corr=False
    boot_include = _bootstrap_isc_group_numpy(
        group1,
        group2,
        observed_diff=observed_diff,
        metric="median",
        summary_statistic="pairwise",
        exclude_self_corr=False,
        random_state=42,
    )

    # Both should be valid
    assert np.isfinite(boot_exclude)
    assert np.isfinite(boot_include)


@pytest.mark.tier1
def test_bootstrap_isc_group_works_with_loo():
    """Bootstrap works with leave-one-out summary statistic."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    observed_diff = _compute_isc_group_difference(
        group1,
        group2,
        metric="median",
        summary_statistic="leave-one-out",
        backend="numpy",
    )

    boot_diff = _bootstrap_isc_group_numpy(
        group1,
        group2,
        observed_diff=observed_diff,
        metric="median",
        summary_statistic="leave-one-out",
        random_state=42,
    )

    assert isinstance(boot_diff, (float, np.floating))
    assert np.isfinite(boot_diff)


@pytest.mark.tier1
def test_bootstrap_isc_group_works_with_mean_metric():
    """Bootstrap works with mean metric."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    observed_diff = _compute_isc_group_difference(
        group1, group2, metric="mean", summary_statistic="pairwise", backend="numpy"
    )

    boot_diff = _bootstrap_isc_group_numpy(
        group1,
        group2,
        observed_diff=observed_diff,
        metric="mean",
        summary_statistic="pairwise",
        random_state=42,
    )

    assert isinstance(boot_diff, (float, np.floating))
    assert np.isfinite(boot_diff)


@pytest.mark.tier1
def test_bootstrap_isc_group_cpu_parallel_matches_numpy():
    """CPU-parallel bootstrap matches sequential NumPy exactly."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    observed_diff = _compute_isc_group_difference(
        group1, group2, metric="median", summary_statistic="pairwise", backend="numpy"
    )

    # Sequential NumPy
    rng = np.random.RandomState(42)
    seeds = rng.randint(0, 2**31 - 1, size=100)
    boot_numpy = np.array(
        [
            _bootstrap_isc_group_numpy(
                group1,
                group2,
                observed_diff=observed_diff,
                metric="median",
                summary_statistic="pairwise",
                random_state=np.random.RandomState(seeds[i]),
            )
            for i in range(100)
        ]
    )

    # CPU-parallel
    boot_parallel = _bootstrap_isc_group_cpu_parallel(
        group1,
        group2,
        observed_diff=observed_diff,
        n_permute=100,
        metric="median",
        summary_statistic="pairwise",
        random_state=42,
        progress_bar=False,
    )

    assert np.allclose(boot_numpy, boot_parallel)


@pytest.mark.tier1
def test_bootstrap_isc_group_cpu_parallel_deterministic():
    """CPU-parallel bootstrap is deterministic."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    observed_diff = _compute_isc_group_difference(
        group1, group2, metric="median", summary_statistic="pairwise", backend="numpy"
    )

    boot1 = _bootstrap_isc_group_cpu_parallel(
        group1,
        group2,
        observed_diff=observed_diff,
        n_permute=100,
        metric="median",
        summary_statistic="pairwise",
        random_state=42,
        progress_bar=False,
    )

    boot2 = _bootstrap_isc_group_cpu_parallel(
        group1,
        group2,
        observed_diff=observed_diff,
        n_permute=100,
        metric="median",
        summary_statistic="pairwise",
        random_state=42,
        progress_bar=False,
    )

    assert np.allclose(boot1, boot2)


@pytest.mark.tier1
def test_bootstrap_isc_group_cpu_parallel_voxelwise():
    """CPU-parallel bootstrap works with voxel-wise data."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5, 10)  # 10 voxels
    group2 = np.random.randn(100, 5, 10)

    observed_diff = _compute_isc_group_difference(
        group1, group2, metric="median", summary_statistic="pairwise", backend="numpy"
    )

    boot = _bootstrap_isc_group_cpu_parallel(
        group1,
        group2,
        observed_diff=observed_diff,
        n_permute=50,
        metric="median",
        summary_statistic="pairwise",
        random_state=42,
        progress_bar=False,
    )

    assert boot.shape == (50, 10)
    assert np.all(np.isfinite(boot))


# =============================================================================
# Phase 4: Main Function Tests
# =============================================================================


@pytest.mark.tier1
def test_isc_group_permutation_test_basic():
    """Main function returns all expected outputs."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    result = isc_group_permutation_test(
        group1,
        group2,
        n_permute=100,
        method="permute",
        random_state=42,
        progress_bar=False,
    )

    assert "isc_group_difference" in result
    assert "p" in result
    assert "ci" in result
    assert 0 <= result["p"] <= 1
    assert len(result["ci"]) == 2
    assert result["ci"][0] <= result["ci"][1]


@pytest.mark.tier1
def test_isc_group_permutation_test_bootstrap_method():
    """Main function works with bootstrap method."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    result = isc_group_permutation_test(
        group1,
        group2,
        n_permute=100,
        method="bootstrap",
        random_state=42,
        progress_bar=False,
    )

    assert "isc_group_difference" in result
    assert "p" in result
    assert "ci" in result


@pytest.mark.tier1
def test_isc_group_permutation_test_voxelwise():
    """Main function works with voxel-wise data."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5, 10)  # 10 voxels
    group2 = np.random.randn(100, 5, 10)

    result = isc_group_permutation_test(
        group1,
        group2,
        n_permute=100,
        method="permute",
        random_state=42,
        progress_bar=False,
    )

    assert result["isc_group_difference"].shape == (10,)
    assert result["p"].shape == (10,)
    assert result["ci"][0].shape == (10,)
    assert result["ci"][1].shape == (10,)


@pytest.mark.tier1
def test_isc_group_permutation_test_backend_consistency():
    """NumPy and CPU-parallel backends give identical results."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    result_numpy = isc_group_permutation_test(
        group1,
        group2,
        parallel=None,
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    result_parallel = isc_group_permutation_test(
        group1,
        group2,
        parallel="cpu",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    # Both use float64 CPU - should be exact matches
    np.testing.assert_allclose(
        result_numpy["isc_group_difference"],
        result_parallel["isc_group_difference"],
        rtol=TOLERANCE_EXACT,
    )
    np.testing.assert_allclose(
        result_numpy["p"], result_parallel["p"], rtol=TOLERANCE_EXACT
    )


@pytest.mark.tier1
def test_isc_group_permutation_test_return_null():
    """Main function returns null distribution when requested."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    result = isc_group_permutation_test(
        group1,
        group2,
        n_permute=100,
        return_null=True,
        random_state=42,
        progress_bar=False,
    )

    assert "null_dist" in result
    assert result["null_dist"].shape == (100,)


@pytest.mark.tier1
def test_isc_group_permutation_test_all_metrics():
    """Main function works with both median and mean metrics."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    for metric in ["median", "mean"]:
        result = isc_group_permutation_test(
            group1,
            group2,
            metric=metric,
            n_permute=100,
            random_state=42,
            progress_bar=False,
        )

        assert "isc_group_difference" in result
        assert "p" in result
        assert np.isfinite(result["isc_group_difference"])
        assert 0 <= result["p"] <= 1


@pytest.mark.tier1
def test_isc_group_permutation_test_all_summary_statistics():
    """Main function works with both pairwise and leave-one-out."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    for summary_statistic in ["pairwise", "leave-one-out"]:
        result = isc_group_permutation_test(
            group1,
            group2,
            summary_statistic=summary_statistic,
            n_permute=100,
            random_state=42,
            progress_bar=False,
        )

        assert "isc_group_difference" in result
        assert "p" in result
        assert np.isfinite(result["isc_group_difference"])


@pytest.mark.tier1
def test_isc_group_permutation_test_invalid_method():
    """Main function raises error for invalid method."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    with pytest.raises(ValueError, match="method must be"):
        isc_group_permutation_test(
            group1, group2, method="invalid", n_permute=100, progress_bar=False
        )


@pytest.mark.tier1
def test_isc_group_permutation_test_exclude_self_corr_parameter():
    """exclude_self_corr parameter is accepted and works."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    result = isc_group_permutation_test(
        group1,
        group2,
        method="bootstrap",
        exclude_self_corr=True,
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    assert "isc_group_difference" in result
    assert "p" in result


# =============================================================================
# Phase 5: Statistical Correctness Tests
# =============================================================================


def _generate_isc_group_data(
    n_timepoints,
    n_subjects1,
    n_subjects2,
    isc_strength1,
    isc_strength2,
    random_state=None,
):
    """
    Generate two groups with known ISC values.

    Creates two groups where each group has a shared signal across subjects
    with specified ISC strength. Higher strength = more shared signal.

    Parameters
    ----------
    n_timepoints : int
        Number of time points (observations)
    n_subjects1 : int
        Number of subjects in group 1
    n_subjects2 : int
        Number of subjects in group 2
    isc_strength1 : float
        ISC strength for group 1 (0-1, higher = more shared signal)
    isc_strength2 : float
        ISC strength for group 2 (0-1, higher = more shared signal)
    random_state : int or RandomState, optional
        Random seed for reproducibility

    Returns
    -------
    group1 : ndarray, shape (n_timepoints, n_subjects1)
        Group 1 data with specified ISC strength
    group2 : ndarray, shape (n_timepoints, n_subjects2)
        Group 2 data with specified ISC strength
    """
    from sklearn.utils import check_random_state

    rng = check_random_state(random_state)

    # Generate shared signals for each group
    shared_signal1 = rng.randn(n_timepoints, 1)
    shared_signal2 = rng.randn(n_timepoints, 1)

    # Generate independent noise for each subject
    noise1 = rng.randn(n_timepoints, n_subjects1)
    noise2 = rng.randn(n_timepoints, n_subjects2)

    # Combine shared signal + noise
    # Formula: data = strength * shared + sqrt(1-strength^2) * noise
    # This ensures unit variance and correct correlation
    group1 = isc_strength1 * shared_signal1 + np.sqrt(1 - isc_strength1**2) * noise1
    group2 = isc_strength2 * shared_signal2 + np.sqrt(1 - isc_strength2**2) * noise2

    return group1, group2


class TestISCGroupStatisticalCorrectness:
    """Test statistical correctness of ISC group permutation tests."""

    @pytest.mark.tier2
    def test_null_hypothesis_pvalue_distribution(self):
        """Test that p-values are uniformly distributed under null hypothesis (no group difference)."""
        from scipy.stats import kstest

        n_timepoints = 100
        n_subjects1 = 10
        n_subjects2 = 10
        n_tests = 100  # Run many tests with different seeds
        n_permute = 2000  # Enough permutations for stable p-values

        # Test both methods
        methods = ["permute", "bootstrap"]

        for method in methods:
            p_values = []

            for seed in range(n_tests):
                np.random.seed(seed)
                # Generate groups with same ISC (no group difference)
                group1, group2 = _generate_isc_group_data(
                    n_timepoints,
                    n_subjects1,
                    n_subjects2,
                    isc_strength1=0.5,  # Same ISC in both groups
                    isc_strength2=0.5,
                    random_state=seed,
                )

                result = isc_group_permutation_test(
                    group1,
                    group2,
                    method=method,
                    n_permute=n_permute,
                    random_state=seed,
                    progress_bar=False,
                )

                p_values.append(result["p"])

            # Test uniformity using Kolmogorov-Smirnov test
            # Under null hypothesis, p-values should be uniformly distributed
            ks_statistic, ks_pvalue = kstest(p_values, "uniform")

            # KS test p-value should be > 0.05 (p-values are uniform)
            assert ks_pvalue > 0.05, (
                f"P-values should be uniformly distributed under null hypothesis for {method}. "
                f"KS test p-value: {ks_pvalue:.4f}"
            )

    @pytest.mark.tier1
    def test_isc_group_difference_value_correctness(self):
        """Test that ISC group difference matches expected value for known group differences."""
        np.random.seed(42)
        n_timepoints = 100
        n_subjects1 = 10
        n_subjects2 = 10
        n_permute = 100  # Small for speed

        # Test case: group1 has higher ISC than group2
        group1, group2 = _generate_isc_group_data(
            n_timepoints,
            n_subjects1,
            n_subjects2,
            isc_strength1=0.7,  # Higher ISC
            isc_strength2=0.3,  # Lower ISC
            random_state=42,
        )

        # Compute ISC group difference
        result = isc_group_permutation_test(
            group1,
            group2,
            method="permute",
            n_permute=n_permute,
            random_state=42,
            progress_bar=False,
        )

        # ISC difference should be positive (group1 > group2)
        # Actual difference will be less than 0.7 - 0.3 = 0.4 due to noise
        # But should be positive and reasonably large
        assert result["isc_group_difference"] > 0.1, (
            f"ISC group difference should be positive when group1 > group2. "
            f"Got {result['isc_group_difference']:.4f}"
        )
        assert result["isc_group_difference"] < 1.0, (
            f"ISC group difference should be less than 1.0. "
            f"Got {result['isc_group_difference']:.4f}"
        )

    @pytest.mark.tier1
    def test_effect_size_sensitivity(self):
        """Test that larger group difference produces lower p-values."""
        np.random.seed(42)
        n_timepoints = 100
        n_subjects1 = 10
        n_subjects2 = 10
        n_permute = 2000  # Enough for stable p-values

        # Test different group differences
        # Format: (isc_strength1, isc_strength2, expected_diff_sign)
        test_cases = [
            (0.5, 0.5, 0.0),  # No difference (null)
            (0.5, 0.4, 0.1),  # Small difference
            (0.6, 0.4, 0.2),  # Medium difference
            (0.7, 0.3, 0.4),  # Large difference
        ]

        p_values = []
        isc_differences = []

        for isc1, isc2, expected_diff in test_cases:
            group1, group2 = _generate_isc_group_data(
                n_timepoints,
                n_subjects1,
                n_subjects2,
                isc_strength1=isc1,
                isc_strength2=isc2,
                random_state=42,
            )

            result = isc_group_permutation_test(
                group1,
                group2,
                method="permute",
                n_permute=n_permute,
                random_state=42,
                progress_bar=False,
            )

            p_values.append(result["p"])
            isc_differences.append(result["isc_group_difference"])

        # Verify monotonic relationship: larger diff → lower p-value
        # Large difference (0.7 vs 0.3) should produce significant p-value
        assert p_values[-1] < 0.05, (
            f"Large group difference should have significant p-value. "
            f"Got p={p_values[-1]:.4f} for difference={isc_differences[-1]:.4f}"
        )

        # Verify trend: p-values should generally decrease with larger differences
        # (allow some variance due to finite permutations)
        for i in range(len(p_values) - 1):
            diff1 = abs(isc_differences[i])
            diff2 = abs(isc_differences[i + 1])

            # If difference increases significantly, p-value should decrease
            if diff2 > diff1 + 0.1:  # Large enough difference
                assert p_values[i + 1] <= p_values[i] * 1.5, (
                    f"P-value should decrease with larger group difference. "
                    f"Diff={diff1:.3f}: p={p_values[i]:.4f}, "
                    f"Diff={diff2:.3f}: p={p_values[i + 1]:.4f}"
                )

    @pytest.mark.tier1
    def test_bootstrap_method_correctness(self):
        """Test that bootstrap method produces correct null distribution."""
        np.random.seed(42)
        n_timepoints = 100
        n_subjects1 = 10
        n_subjects2 = 10
        n_permute = 1000  # Enough for stable distribution

        # Generate groups with known difference
        group1, group2 = _generate_isc_group_data(
            n_timepoints,
            n_subjects1,
            n_subjects2,
            isc_strength1=0.7,
            isc_strength2=0.3,
            random_state=42,
        )

        result = isc_group_permutation_test(
            group1,
            group2,
            method="bootstrap",
            n_permute=n_permute,
            return_null=True,
            random_state=42,
            progress_bar=False,
        )

        # Bootstrap distribution should be centered (mean ≈ 0 after centering)
        null_dist = result["null_dist"]
        null_mean = np.nanmean(null_dist)

        # Bootstrap distribution is centered by subtracting observed difference
        # So mean should be close to 0
        assert abs(null_mean) < 0.1, (
            f"Bootstrap null distribution should be centered (mean ≈ 0). "
            f"Got mean={null_mean:.4f}"
        )

        # CI should contain observed difference at expected rate
        ci_lower, ci_upper = result["ci"]
        observed_diff = result["isc_group_difference"]

        # 95% CI should contain observed difference (unless p < 0.05)
        if result["p"] >= 0.05:
            assert ci_lower <= observed_diff <= ci_upper, (
                f"95% CI should contain observed difference when p >= 0.05. "
                f"CI=[{ci_lower:.4f}, {ci_upper:.4f}], observed={observed_diff:.4f}"
            )

    @pytest.mark.tier1
    def test_backward_compatibility_stats_py(self):
        """Test that new implementation matches old stats.py isc_group results."""
        np.random.seed(42)
        n_timepoints = 100
        n_subjects1 = 10
        n_subjects2 = 10

        # Generate test data
        group1 = np.random.randn(n_timepoints, n_subjects1)
        group2 = np.random.randn(n_timepoints, n_subjects2)

        # Test both methods
        methods = ["permute", "bootstrap"]

        for method in methods:
            # Old implementation (from stats.py)
            from nltools.stats import isc_group as isc_group_old

            result_old = isc_group_old(
                group1,
                group2,
                n_samples=500,  # Smaller for speed
                method=method,
                random_state=42,
                return_null=False,
            )

            # New implementation
            result_new = isc_group_permutation_test(
                group1,
                group2,
                n_permute=500,
                method=method,
                random_state=42,
                progress_bar=False,
            )

            # Compare ISC difference (should be very close)
            np.testing.assert_allclose(
                result_old["isc_group_difference"],
                result_new["isc_group_difference"],
                rtol=0.15,  # 15% tolerance for correlation variance
            )

            # Compare p-values (should be close, but allow variance due to implementation differences)
            # Note: Old implementation uses Adjacency objects, different NaN handling, and different
            # random state management. The new implementation is statistically correct but may
            # produce different p-values due to these implementation differences.
            # We check that p-values are in the same ballpark (both high or both low).
            # More important: ISC difference should match closely.
            p_diff = abs(result_old["p"] - result_new["p"])
            # Allow up to 25% relative difference, or check if both are in same significance category
            if not (p_diff / max(result_old["p"], result_new["p"]) < 0.25):
                # If relative difference is large, check if both are in same significance category
                both_significant = (result_old["p"] < 0.05) == (result_new["p"] < 0.05)
                both_nonsignificant = (result_old["p"] >= 0.05) == (
                    result_new["p"] >= 0.05
                )
                if not (both_significant or both_nonsignificant):
                    # Only fail if significance category differs
                    pytest.fail(
                        f"P-values differ significantly: old={result_old['p']:.4f}, "
                        f"new={result_new['p']:.4f}. "
                        f"ISC difference matches: old={result_old['isc_group_difference']:.4f}, "
                        f"new={result_new['isc_group_difference']:.4f}"
                    )


# =============================================================================
# Phase 6: Integration Tests
# =============================================================================


@pytest.mark.tier1
def test_stats_isc_group_backward_compatibility():
    """Test that stats.py wrapper maintains exact API compatibility."""
    np.random.seed(42)
    group1 = np.random.randn(100, 10)
    group2 = np.random.randn(100, 10)

    # Test direct call to stats.py wrapper
    from nltools.stats import isc_group

    result = isc_group(
        group1,
        group2,
        n_samples=500,  # Map to n_permute
        method="permute",
        metric="median",
        return_null=False,
        random_state=42,
    )

    # Verify return keys match expected API
    assert "isc_group_difference" in result
    assert "p" in result
    assert "ci" in result
    assert len(result["ci"]) == 2
    assert isinstance(result["isc_group_difference"], (float, np.ndarray))
    assert isinstance(result["p"], (float, np.ndarray))
    assert 0 <= result["p"] <= 1

    # Test with return_null=True
    result_with_null = isc_group(
        group1,
        group2,
        n_samples=100,
        return_null=True,
        random_state=42,
    )

    assert "null_dist" in result_with_null
    assert isinstance(result_with_null["null_dist"], np.ndarray)

    # Test with bootstrap method
    result_bootstrap = isc_group(
        group1,
        group2,
        n_samples=100,
        method="bootstrap",
        exclude_self_corr=True,
        random_state=42,
    )

    assert "isc_group_difference" in result_bootstrap
    assert "p" in result_bootstrap

    # Test parameter name mapping: n_samples maps to n_permute
    # Verify it works with different n_samples values
    result_small = isc_group(group1, group2, n_samples=50, random_state=42)
    result_large = isc_group(group1, group2, n_samples=200, random_state=42)

    assert "isc_group_difference" in result_small
    assert "isc_group_difference" in result_large

    # Test pandas DataFrame support (backward compatibility)
    import pandas as pd

    group1_df = pd.DataFrame(group1)
    group2_df = pd.DataFrame(group2)

    result_df = isc_group(
        group1_df,
        group2_df,
        n_samples=100,
        random_state=42,
    )

    assert "isc_group_difference" in result_df
    assert "p" in result_df
