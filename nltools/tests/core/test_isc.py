"""
Tests for the Intersubject Correlation (ISC) module.

Test organization:
    Phase 1: Leave-One-Out (LOO) Computation
    Phase 2: Pairwise Computation
    Phase 3: LOO Bootstrap
    Phase 4: Pairwise Bootstrap
    Phase 5: Main Function
    Integration Tests

Testing Strategy & Tolerances:

    Worker-count consistency (n_jobs=1 vs n_jobs=-1):
        Tolerance: EXACT (rtol=1e-5). Seeds are drawn before the joblib block
        and consumed in index order, so the worker count only changes
        scheduling, never arithmetic.
"""

import numpy as np
import pytest
from scipy.spatial.distance import squareform

from nltools.algorithms.inference.isc import (
    _bootstrap_loo_numpy,
    _compute_loo_isc,
    _compute_pairwise_isc,
    _isc_permutation_test,
)


# =============================================================================
# Test Constants - DO NOT MODIFY without updating docstring above
# =============================================================================

# Tolerance for worker-count consistency (same seeds, different n_jobs)
TOLERANCE_EXACT = 1e-5


# =============================================================================
# Phase 1: Leave-One-Out (LOO) Computation Tests
# =============================================================================


def test_compute_loo_isc_single_feature_basic():
    """LOO ISC computes correlation of each subject with mean of others."""
    np.random.seed(42)
    data = np.random.randn(100, 5)  # 100 timepoints, 5 subjects

    loo_values = _compute_loo_isc(data)

    assert loo_values.shape == (5,)

    # Verify each value manually
    for i in range(5):
        others_mean = data[:, np.arange(5) != i].mean(axis=1)
        expected = np.corrcoef(data[:, i], others_mean)[0, 1]
        assert np.isclose(loo_values[i], expected)


def test_compute_loo_isc_voxelwise_shape():
    """Voxel-wise LOO returns (n_subjects, n_voxels) array."""
    np.random.seed(42)
    data = np.random.randn(100, 5, 10)  # 10 voxels

    loo_values = _compute_loo_isc(data)

    assert loo_values.shape == (5, 10)

    # Each voxel computed independently
    for v in range(10):
        voxel_loo = _compute_loo_isc(data[:, :, v])
        assert np.allclose(loo_values[:, v], voxel_loo)


# =============================================================================
# Phase 2: Pairwise Computation Tests
# =============================================================================


def test_compute_pairwise_isc_single_feature_condensed():
    """Pairwise ISC returns condensed correlation matrix."""
    np.random.seed(42)
    data = np.random.randn(100, 5)

    pairwise = _compute_pairwise_isc(data)

    # 5 subjects → 10 pairs
    assert pairwise.shape == (10,)

    # Verify matches np.corrcoef
    corr_matrix = np.corrcoef(data.T)
    expected = squareform(corr_matrix, checks=False)
    assert np.allclose(pairwise, expected)


def test_compute_pairwise_isc_voxelwise_shape():
    """Voxel-wise pairwise returns (n_pairs, n_voxels)."""
    np.random.seed(42)
    data = np.random.randn(100, 5, 10)

    pairwise = _compute_pairwise_isc(data)

    assert pairwise.shape == (10, 10)  # 10 pairs × 10 voxels

    # Verify each voxel independently
    for v in range(10):
        voxel_pair = _compute_pairwise_isc(data[:, :, v])
        assert np.allclose(pairwise[:, v], voxel_pair)


# =============================================================================
# Phase 3: LOO Bootstrap Tests
# =============================================================================


def test_bootstrap_loo_fisher_z_transform():
    """LOO bootstrap with mean applies Fisher z-transform."""
    loo_values = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    rng = np.random.RandomState(42)

    boot_mean = _bootstrap_loo_numpy(loo_values, summary="mean", random_state=rng)

    # Manual Fisher z
    rng2 = np.random.RandomState(42)
    indices = rng2.choice(5, size=5, replace=True)
    boot_sample = loo_values[indices]
    z = np.arctanh(boot_sample)
    expected = np.tanh(np.mean(z))

    assert np.isclose(boot_mean, expected)


# =============================================================================
# Phase 4: Pairwise Bootstrap Tests
# =============================================================================


def test_bootstrap_pairwise_condensed_to_square_roundtrip():
    """Condensed storage roundtrip preserves correlation values."""
    # Create correlation matrix
    corr_matrix = np.array([[1.0, 0.8, 0.7], [0.8, 1.0, 0.6], [0.7, 0.6, 1.0]])

    # Extract condensed form (upper triangle, excluding diagonal)
    condensed = squareform(corr_matrix, checks=False)

    # Reconstruct to square
    reconstructed = squareform(condensed, force="tomatrix", checks=False)
    # Fill diagonal with 1.0 (correlation matrices have 1 on diagonal)
    np.fill_diagonal(reconstructed, 1.0)

    # Back to condensed
    condensed_again = squareform(reconstructed, checks=False)

    assert np.allclose(condensed, condensed_again)


# =============================================================================
# Phase 5: Main Function Tests
# =============================================================================


def test_isc_voxelwise_shape():
    """Voxel-wise ISC returns arrays per voxel."""
    np.random.seed(42)
    data = np.random.randn(100, 10, 50)  # 50 voxels

    result = _isc_permutation_test(
        data,
        summary_statistic="pairwise",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    assert result["isc"].shape == (50,)
    assert result["p"].shape == (50,)
    assert result["ci"][0].shape == (50,)
    assert result["ci"][1].shape == (50,)


def test_isc_return_null_dist():
    """ISC with return_null=True includes bootstrap distribution."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    result = _isc_permutation_test(
        data, n_permute=100, return_null=True, random_state=42, progress_bar=False
    )

    assert "null_dist" in result
    assert result["null_dist"].shape == (100,)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.slow
def test_isc_matches_brainiak_loo_logic():
    """LOO ISC matches Brainiak computation pattern."""
    # Create controlled data
    np.random.seed(42)
    data = np.random.randn(100, 5)

    # Our implementation
    result = _isc_permutation_test(
        data,
        summary_statistic="leave-one-out",
        summary="median",
        n_permute=1000,
        random_state=42,
        progress_bar=False,
    )

    # Manual Brainiak-style computation
    loo_values = []
    for i in range(5):
        others = data[:, np.arange(5) != i].mean(axis=1)
        loo_values.append(np.corrcoef(data[:, i], others)[0, 1])

    expected_isc = np.median(loo_values)

    assert np.isclose(result["isc"], expected_isc)


# =============================================================================
# Edge Cases and Input Validation
# =============================================================================


def test_isc_invalid_summary_statistic():
    """Invalid summary_statistic raises ValueError."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    with pytest.raises(ValueError, match="summary_statistic must be"):
        _isc_permutation_test(
            data, summary_statistic="invalid", n_permute=100, progress_bar=False
        )


def test_isc_invalid_method():
    """Invalid method raises ValueError."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    with pytest.raises(ValueError, match="method must be"):
        _isc_permutation_test(data, method="invalid", n_permute=100, progress_bar=False)


def test_isc_invalid_data_dimensions():
    """Data with wrong dimensions raises ValueError."""
    data_1d = np.random.randn(100)

    with pytest.raises(ValueError, match="data must be 2D or 3D"):
        _isc_permutation_test(data_1d, n_permute=100, progress_bar=False)


def test_isc_default_is_pairwise():
    """Default summary_statistic is 'pairwise'."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    # Call without specifying summary_statistic
    result = _isc_permutation_test(
        data, n_permute=100, random_state=42, progress_bar=False
    )

    # Verify it used pairwise by comparing with explicit pairwise call
    result_explicit = _isc_permutation_test(
        data,
        summary_statistic="pairwise",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    assert np.isclose(result["isc"], result_explicit["isc"])


def test_isc_metric_parameter():
    """metric parameter allows different similarity metrics."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    # Test with correlation (default)
    result_corr = _isc_permutation_test(
        data,
        summary_statistic="pairwise",
        metric="correlation",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    # Test with euclidean distance (converted to similarity)
    result_eucl = _isc_permutation_test(
        data,
        summary_statistic="pairwise",
        metric="euclidean",
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    # Both should complete successfully
    assert "isc" in result_corr
    assert "isc" in result_eucl
    assert "p" in result_corr
    assert "p" in result_eucl

    # Different metrics should produce different ISC values
    # Correlation ISC is bounded [-1, 1], euclidean similarity can be negative
    assert isinstance(result_corr["isc"], (float, np.floating))
    assert isinstance(result_eucl["isc"], (float, np.floating))

    # Correlation should be in reasonable range
    assert -1 <= result_corr["isc"] <= 1


def test_isc_exclude_self_corr_pairwise_only():
    """exclude_self_corr only applies to pairwise bootstrap."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    # exclude_self_corr should not affect LOO (which doesn't use pairwise matrix)
    result_loo = _isc_permutation_test(
        data,
        summary_statistic="leave-one-out",
        method="bootstrap",
        exclude_self_corr=True,  # Should be ignored for LOO
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    result_loo_default = _isc_permutation_test(
        data,
        summary_statistic="leave-one-out",
        method="bootstrap",
        exclude_self_corr=False,  # Should be ignored for LOO
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    # LOO results should be identical regardless of exclude_self_corr
    assert np.isclose(result_loo["isc"], result_loo_default["isc"])


def test_isc_metric_pairwise_only():
    """metric only applies to pairwise summary_statistic."""
    np.random.seed(42)
    data = np.random.randn(100, 10)

    # metric should not affect LOO (which computes correlations directly)
    result_loo_corr = _isc_permutation_test(
        data,
        summary_statistic="leave-one-out",
        metric="correlation",  # Should be ignored for LOO
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    result_loo_eucl = _isc_permutation_test(
        data,
        summary_statistic="leave-one-out",
        metric="euclidean",  # Should be ignored for LOO
        n_permute=100,
        random_state=42,
        progress_bar=False,
    )

    # LOO results should be identical regardless of metric
    assert np.isclose(result_loo_corr["isc"], result_loo_eucl["isc"])


def test_compute_pairwise_isc_cosine_handles_zero_norm():
    """Cosine similarity handles zero-norm vectors gracefully."""
    np.random.seed(42)
    data = np.random.randn(100, 5)

    # Add a zero vector (all zeros)
    data_zero = np.copy(data)
    data_zero[:, 0] = 0.0  # First subject has zero norm

    # Should not raise error
    result = _compute_pairwise_isc(data_zero, metric="cosine")

    # Should produce valid results (may have NaN or 0 for zero-norm pairs)
    assert result.shape == (10,)  # 5*4/2 = 10 pairs
    # Values should be finite or NaN (for zero-norm cases)
    assert np.all(np.isfinite(result) | np.isnan(result))


# =============================================================================
# Statistical Correctness Tests
# =============================================================================


def _generate_shared_signal_isc(
    n_timepoints, n_subjects, isc_strength, random_state=None
):
    """
    Generate time series data with known ISC.

    Creates data where all subjects share a common signal with strength
    controlled by isc_strength. Higher isc_strength → higher ISC.

    Parameters
    ----------
    n_timepoints : int
        Number of time points
    n_subjects : int
        Number of subjects
    isc_strength : float
        Strength of shared signal (0.0 = no ISC, 1.0 = perfect ISC)
        Higher values → higher ISC
    random_state : int or RandomState, optional
        Random seed

    Returns
    -------
    data : ndarray, shape (n_timepoints, n_subjects)
        Time series data with known ISC structure
    """
    from sklearn.utils import check_random_state

    rng = check_random_state(random_state)

    # Generate shared signal
    shared_signal = rng.randn(n_timepoints)

    # Generate data for each subject: shared_signal * strength + noise * (1 - strength)
    data = np.zeros((n_timepoints, n_subjects))
    for i in range(n_subjects):
        noise = rng.randn(n_timepoints)
        data[:, i] = shared_signal * isc_strength + noise * np.sqrt(1 - isc_strength**2)

    return data


class TestISCStatisticalCorrectness:
    """Test statistical correctness of ISC permutation tests."""

    @pytest.mark.slow
    @pytest.mark.parametrize("method", ["circle_shift"])
    @pytest.mark.parametrize("summary_statistic", ["pairwise"])
    def test_null_hypothesis_pvalue_distribution(self, method, summary_statistic):
        """Surrogate-null p-values are uniform under the null hypothesis (ISC = 0).

        Only the surrogate methods (``circle_shift``/``phase_randomize``) are
        genuine null-hypothesis significance tests: they generate a null by
        destroying inter-subject temporal alignment while preserving each
        subject's marginal structure, so under H0 (independent time series,
        true ISC = 0) their p-values are uniform on [0, 1].

        ``method='bootstrap'`` is deliberately excluded. The subject-wise
        bootstrap (Chen et al. 2016) is a confidence-interval / effect-magnitude
        tool; its recentered-exceedance p-value is only asymptotically valid and
        is biased in finite samples (empirically anti-conservative for LOO,
        conservative for pairwise here), so it does NOT yield uniform p-values
        under the null. Asserting uniformity for bootstrap would be testing the
        wrong contract.
        """
        from scipy.stats import kstest

        n_timepoints = 100
        n_subjects = 15
        n_tests = 100  # Run many tests with different seeds
        n_permute = 1000  # Enough permutations for stable p-values

        p_values = []
        for seed in range(n_tests):
            np.random.seed(seed)
            # Generate independent time series (ISC = 0)
            data = np.random.randn(n_timepoints, n_subjects)

            result = _isc_permutation_test(
                data,
                summary_statistic=summary_statistic,
                method=method,
                n_permute=n_permute,
                random_state=seed,
                progress_bar=False,
            )

            # result["p"] is a length-1 array for single-feature input; take scalar
            p_values.append(np.asarray(result["p"]).item())

        # Under H0, p-values should be uniform: a KS test should fail to reject.
        # Everything above is seeded, so this KS p-value is deterministic.
        ks_statistic, ks_pvalue = kstest(p_values, "uniform")

        assert ks_pvalue > 0.05, (
            f"P-values should be uniformly distributed under the null hypothesis "
            f"for method={method!r}, summary_statistic={summary_statistic!r}. "
            f"KS test p-value: {float(ks_pvalue):.4f}"
        )
