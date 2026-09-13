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
Tier 2: Benchmark tests (~5-7min, run before commits)

Testing Strategy & Tolerances:

    1. Worker-count consistency (n_jobs=1 vs n_jobs=-1):
       - Tolerance: EXACT (rtol=1e-5)
       - Why: Same algorithm, same seeds, worker count only changes scheduling

    2. Backward Compatibility (vs stats.py):
       - P-values: rtol=0.02 (2% error)
       - Why: Prioritizes seeded determinism over exact stats.py match
"""

import numpy as np
import pytest

from nltools.algorithms.inference.isc import (
    _compute_isc_group_difference,
    _compute_pairwise_isc,
    _isc_group_permutation_test,
)

# =============================================================================
# Test Constants - DO NOT MODIFY without updating docstring above
# =============================================================================

# Tolerance for worker-count consistency (same seeds, different n_jobs)
TOLERANCE_EXACT = 1e-5

# Tolerance for backward compatibility with stats.py
TOLERANCE_STATS_PVALUE = 0.02  # 2% relative error acceptable


# =============================================================================
# Phase 1: Core ISC Group Difference Computation Tests
# =============================================================================


def test_compute_isc_group_difference_numpy_voxelwise():
    """ISC group difference computes correctly for voxel-wise data."""
    np.random.seed(42)
    n_obs = 50  # Reduced from 100 for tier1 speed
    n_subjects1, n_subjects2 = 5, 5
    n_voxels = 5  # Reduced from 10 for tier1 speed

    group1 = np.random.randn(n_obs, n_subjects1, n_voxels)
    group2 = np.random.randn(n_obs, n_subjects2, n_voxels)

    # Compute ISC difference
    isc_diff = _compute_isc_group_difference(
        group1, group2, summary="median", summary_statistic="pairwise"
    )

    # Should be array per voxel
    assert isc_diff.shape == (n_voxels,)
    assert np.all(np.isfinite(isc_diff))

    # Verify first voxel manually
    voxel1_group1 = group1[:, :, 0]
    voxel1_group2 = group2[:, :, 0]
    pairwise1_v0 = _compute_pairwise_isc(voxel1_group1)
    pairwise2_v0 = _compute_pairwise_isc(voxel1_group2)
    isc1_v0 = np.median(pairwise1_v0)
    isc2_v0 = np.median(pairwise2_v0)
    expected_diff_v0 = isc1_v0 - isc2_v0

    assert np.isclose(isc_diff[0], expected_diff_v0)


def test_compute_isc_group_difference_mismatched_observations():
    """ISC group difference raises error if groups have different number of observations."""
    np.random.seed(42)
    group1 = np.random.randn(50, 5)  # Reduced from 100, 5 for tier1 speed
    group2 = np.random.randn(
        51, 5
    )  # Reduced from 101, 5 for tier1 speed - Different number of observations

    with pytest.raises(ValueError, match="same number of observations"):
        _compute_isc_group_difference(
            group1,
            group2,
            summary="median",
            summary_statistic="pairwise",
        )


def test_compute_isc_group_difference_invalid_metric():
    """ISC group difference raises error for invalid summary."""
    np.random.seed(42)
    group1 = np.random.randn(50, 5)  # Reduced from 100, 5 for tier1 speed
    group2 = np.random.randn(50, 5)  # Reduced from 100, 5 for tier1 speed

    with pytest.raises(ValueError, match="summary must be"):
        _compute_isc_group_difference(
            group1,
            group2,
            summary="invalid",
            summary_statistic="pairwise",
        )


def test_compute_isc_group_difference_invalid_summary_statistic():
    """ISC group difference raises error for invalid summary_statistic."""
    np.random.seed(42)
    group1 = np.random.randn(50, 5)  # Reduced from 100, 5 for tier1 speed
    group2 = np.random.randn(50, 5)  # Reduced from 100, 5 for tier1 speed

    with pytest.raises(ValueError, match="summary_statistic must be"):
        _compute_isc_group_difference(
            group1,
            group2,
            summary="median",
            summary_statistic="invalid",
        )


# =============================================================================
# Phase 2: Permutation Method Tests
# =============================================================================


# =============================================================================
# Phase 3: Bootstrap Method Tests
# =============================================================================


# =============================================================================
# Phase 4: Main Function Tests
# =============================================================================


def test_isc_group_permutation_test_voxelwise():
    """Main function works with voxel-wise data."""
    np.random.seed(42)
    group1 = np.random.randn(50, 5, 5)  # Reduced from 100, 5, 10 for tier1 speed
    group2 = np.random.randn(50, 5, 5)  # Reduced from 100, 5, 10 for tier1 speed

    result = _isc_group_permutation_test(
        group1,
        group2,
        n_permute=100,
        method="permute",
        random_state=42,
        progress_bar=False,
    )

    assert result["isc_group_difference"].shape == (5,)  # Reduced from 10
    assert result["p"].shape == (5,)  # Reduced from 10
    assert result["ci"][0].shape == (5,)  # Reduced from 10
    assert result["ci"][1].shape == (5,)  # Reduced from 10


def test_isc_group_permutation_test_invalid_method():
    """Main function raises error for invalid method."""
    np.random.seed(42)
    group1 = np.random.randn(100, 5)
    group2 = np.random.randn(100, 5)

    with pytest.raises(ValueError, match="method must be"):
        _isc_group_permutation_test(
            group1, group2, method="invalid", n_permute=100, progress_bar=False
        )


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

    @pytest.mark.slow
    def test_isc_group_difference_value_correctness(self):
        """Test that ISC group difference matches expected value for known group differences."""
        np.random.seed(42)
        n_timepoints = 100
        n_subjects1 = 10
        n_subjects2 = 10
        n_permute = 2000  # Increased from 100 for tier2 statistical correctness

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
        result = _isc_group_permutation_test(
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


# =============================================================================
# Phase 6: Integration Tests
# =============================================================================


def test_isc_group_bootstrap_ci_brackets_estimate():
    """F013 regression: bootstrap CI must bracket the observed difference.

    The bootstrap null is centered (``boot - observed_diff``) for the p-value,
    but the reported CI must be built from the UNCENTERED draws so it brackets
    the estimate — not zero. Groups are built with a large true ISC difference
    so a CI centered on 0 would fail to contain the estimate.
    """
    rng = np.random.RandomState(0)
    n_tp, n_subs = 40, 15  # data is (n_observations, n_subjects)
    common = rng.randn(n_tp)
    # group1: strong shared signal across subjects -> high ISC (~1)
    group1 = np.column_stack([common + 0.1 * rng.randn(n_tp) for _ in range(n_subs)])
    # group2: pure noise -> ISC ~ 0
    group2 = np.column_stack([rng.randn(n_tp) for _ in range(n_subs)])

    result = _isc_group_permutation_test(
        group1,
        group2,
        n_permute=200,
        method="bootstrap",
        random_state=42,
        progress_bar=False,
    )

    obs = float(result["isc_group_difference"])
    ci_lower, ci_upper = float(result["ci"][0]), float(result["ci"][1])
    assert obs > 0.5, f"expected a large positive difference, got {obs}"
    assert ci_lower <= obs <= ci_upper, (
        f"CI ({ci_lower:.3f}, {ci_upper:.3f}) does not bracket estimate {obs:.3f}"
    )
