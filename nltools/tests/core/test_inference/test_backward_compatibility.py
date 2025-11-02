"""Tests for backward compatibility with stats.py."""

import numpy as np

from nltools.algorithms.inference import one_sample_permutation_test
from nltools.tests.core.test_inference import (
    N_PERMUTE_STATS_COMPARISON,
    TOLERANCE_STATS_DETERMINISTIC,
    TOLERANCE_STATS_PVALUE,
    TOLERANCE_STATS_PVALUE_ONE_TAILED,
)
from nltools.stats import one_sample_permutation as stats_one_sample


class TestBackwardCompatibility:
    """Test compatibility with existing stats.py implementation."""

    def test_matches_stats_single_feature(self):
        """Test that results match stats.py for single feature."""
        np.random.seed(42)
        data = np.random.randn(30)

        # New implementation
        result_new = one_sample_permutation_test(
            data, n_permute=N_PERMUTE_STATS_COMPARISON, parallel=None, random_state=42
        )

        # Old implementation
        result_old = stats_one_sample(
            data,
            n_permute=N_PERMUTE_STATS_COMPARISON,
            tail=2,
            n_jobs=1,
            random_state=42,
        )

        # Compare results
        # Mean should be identical (it's just np.mean)
        np.testing.assert_allclose(
            result_new["mean"], result_old["mean"], rtol=TOLERANCE_STATS_DETERMINISTIC
        )
        # P-values will differ slightly due to different random sampling
        # but should be in the same ballpark (within ~15% relative error)
        np.testing.assert_allclose(
            result_new["p"], result_old["p"], rtol=TOLERANCE_STATS_PVALUE
        )

    def test_new_multi_feature_support(self):
        """Test that new implementation supports multi-feature data.

        Note: Old stats.py implementation doesn't support multi-feature data
        (it has a broadcasting bug in _permute_sign). Our new implementation
        fixes this limitation.
        """
        np.random.seed(42)
        data = np.random.randn(30, 10)

        # New implementation should work
        result_new = one_sample_permutation_test(
            data, n_permute=N_PERMUTE_STATS_COMPARISON, parallel=None, random_state=42
        )

        # Verify results are sensible
        assert result_new["mean"].shape == (10,)
        assert result_new["p"].shape == (10,)
        assert np.all((result_new["p"] > 0) & (result_new["p"] <= 1))

    def test_matches_stats_one_tailed(self):
        """Test one-tailed test matches stats.py."""
        np.random.seed(42)
        data = np.random.randn(30) + 0.5  # Positive mean

        # New implementation
        result_new = one_sample_permutation_test(
            data,
            n_permute=N_PERMUTE_STATS_COMPARISON,
            parallel=None,
            tail=1,
            random_state=42,
        )

        # Old implementation
        result_old = stats_one_sample(
            data,
            n_permute=N_PERMUTE_STATS_COMPARISON,
            tail=1,
            n_jobs=1,
            random_state=42,
        )

        # Compare results
        np.testing.assert_allclose(
            result_new["mean"], result_old["mean"], rtol=TOLERANCE_STATS_DETERMINISTIC
        )
        # P-values should match exactly now (same RNG pattern)
        np.testing.assert_allclose(
            result_new["p"], result_old["p"], rtol=TOLERANCE_STATS_PVALUE_ONE_TAILED
        )
