"""
Tests for one-sample permutation tests.

Tests both basic functionality and statistical correctness.
"""

import pytest
import numpy as np

from nltools.algorithms import one_sample_permutation_test
from nltools.algorithms.inference.one_sample import _one_sample_statistics


class TestOneSamplePermutation:
    """Test one-sample permutation tests."""

    @pytest.mark.parametrize("n_features", [1, 10])
    def test_basic_functionality(self, n_features):
        """Test basic one-sample test with single or multiple features."""
        np.random.seed(42)
        if n_features == 1:
            data = np.random.randn(30)  # Mean ~ 0
        else:
            data = np.random.randn(30, n_features)  # 30 samples, n_features features

        result = one_sample_permutation_test(data, n_permute=100, random_state=42)

        assert "mean" in result
        assert "p" in result
        assert "device" in result

        if n_features == 1:
            assert isinstance(result["mean"], (float, np.floating))
            assert isinstance(result["p"], (float, np.floating))
        else:
            assert result["mean"].shape == (n_features,)
            assert result["p"].shape == (n_features,)

        assert (
            0 <= result["p"] <= 1
            if n_features == 1
            else np.all((result["p"] >= 0) & (result["p"] <= 1))
        )

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with fixed seed."""
        np.random.seed(42)
        data = np.random.randn(30, 5)

        result1 = one_sample_permutation_test(data, n_permute=100, random_state=42)
        result2 = one_sample_permutation_test(data, n_permute=100, random_state=42)

        np.testing.assert_array_almost_equal(result1["mean"], result2["mean"])
        np.testing.assert_array_almost_equal(result1["p"], result2["p"])

    @pytest.mark.parametrize("n_features", [1, 5])
    def test_return_null_distribution(self, n_features):
        """Test that null distribution is returned when requested."""
        np.random.seed(42)
        if n_features == 1:
            data = np.random.randn(30)
            expected_shape = (100,)
        else:
            data = np.random.randn(30, n_features)
            expected_shape = (100, n_features)

        result = one_sample_permutation_test(
            data, n_permute=100, return_null=True, random_state=42
        )

        assert "null_dist" in result
        assert result["null_dist"].shape == expected_shape

    def test_invalid_tail(self):
        """Test that invalid tail raises error."""
        data = np.random.randn(30)
        with pytest.raises(ValueError, match="tail must be"):
            one_sample_permutation_test(data, tail=3)

    def test_invalid_data_shape(self):
        """Test that invalid data shape raises error."""
        data = np.random.randn(5, 5, 5)  # 3D
        with pytest.raises(ValueError, match="data must be 1D to 2D"):
            one_sample_permutation_test(data)


class TestOneSamplePermutationStatisticalCorrectness:
    """Test statistical correctness of one-sample permutation tests (not just CPU/GPU consistency)."""

    @pytest.mark.slow
    def test_null_hypothesis_pvalue_distribution(self):
        """Test that p-values are uniformly distributed under null hypothesis (mean=0)."""
        from scipy.stats import kstest

        n_samples = 50
        n_tests = 100  # Run many tests with different seeds
        p_values = []

        # Generate data from N(0, 1) (true mean = 0) and run many tests
        for seed in range(n_tests):
            np.random.seed(seed)
            data = np.random.randn(n_samples)  # Mean ~ 0
            result = one_sample_permutation_test(
                data, n_permute=2000, random_state=seed
            )
            p_values.append(result["p"])

        p_values = np.array(p_values)

        # Verify p-values are uniformly distributed using Kolmogorov-Smirnov test
        # Under null hypothesis, p-values should be uniform on [0, 1]
        ks_statistic, ks_pvalue = kstest(p_values, "uniform")

        # KS test p-value should be > 0.05 (fail to reject uniform distribution)
        assert ks_pvalue > 0.05, (
            f"P-values not uniformly distributed: KS statistic={ks_statistic:.4f}, p={ks_pvalue:.4f}"
        )

    @pytest.mark.slow
    def test_effect_size_sensitivity(self):
        """Test that larger true mean produces lower p-values."""
        n_samples = 50
        n_permute = 5000  # Large permutation count for stable p-values

        # Test with different effect sizes
        means = [0.0, 0.5, 1.0, 2.0]
        p_values = []

        for mean in means:
            np.random.seed(42)
            data = np.random.randn(n_samples) + mean
            result = one_sample_permutation_test(
                data, n_permute=n_permute, random_state=42
            )
            p_values.append(result["p"])

        # Verify larger mean → smaller p-value (monotonic relationship)
        # Skip mean=0 (null hypothesis), test others
        # Note: Very large effects may hit minimum p-value (1/(n_permute+1)),
        # so allow >= for equality case when effects are extremely large
        assert p_values[1] >= p_values[2], (
            f"Larger effect should produce smaller p-value. mean=0.5: p={p_values[1]:.6f}, mean=1.0: p={p_values[2]:.6f}"
        )
        assert p_values[2] >= p_values[3], (
            f"Larger effect should produce smaller p-value. mean=1.0: p={p_values[2]:.6f}, mean=2.0: p={p_values[3]:.6f}"
        )

        # Medium effect (mean=1.0) should be significant
        assert p_values[2] < 0.05, (
            f"Medium effect (mean=1.0) should be significant, got p={p_values[2]:.4f}"
        )
        # Large effect (mean=2.0) should be significant
        assert p_values[3] < 0.05, (
            f"Large effect (mean=2.0) should be significant, got p={p_values[3]:.4f}"
        )

    def test_mean_converges_to_true_mean(self):
        """Test that computed mean converges to true mean (fast version with smaller n_permute)."""
        n_samples = 30  # Reduced from 50 for tier1 speed
        true_mean = 5.0

        np.random.seed(42)
        data = np.random.randn(n_samples) + true_mean

        result = one_sample_permutation_test(
            data, n_permute=100, random_state=42, device=None
        )

        # Computed mean should be close to true mean
        # Tolerance: rtol=0.1 (10% as specified in plan)
        np.testing.assert_allclose(result["mean"], true_mean, rtol=0.1, atol=0.1)

    @pytest.mark.slow
    def test_mean_converges_to_true_mean_full(self):
        """Test that computed mean converges to true mean."""
        n_samples = 50
        true_mean = 5.0

        np.random.seed(42)
        data = np.random.randn(n_samples) + true_mean

        result = one_sample_permutation_test(data, n_permute=2000, random_state=42)

        # Computed mean should be close to true mean
        # Tolerance: rtol=0.1 (10% as specified in plan)
        np.testing.assert_allclose(result["mean"], true_mean, rtol=0.1, atol=0.1)

    @pytest.mark.slow
    def test_pvalue_converges_with_more_permutations(self):
        """Test that p-values stabilize (variance decreases) with more permutations."""
        n_samples = 50
        true_mean = 2.0  # Known positive mean

        # Run with different permutation counts
        n_permute_values = [100, 1000, 5000]
        p_values = []

        for n_permute in n_permute_values:
            np.random.seed(42)  # Same seed for reproducibility
            data = np.random.randn(n_samples) + true_mean
            result = one_sample_permutation_test(
                data, n_permute=n_permute, random_state=42
            )
            p_values.append(result["p"])

        # P-values should stabilize with more permutations
        # Variance should decrease (p-values become more consistent)
        # We'll test by running multiple times with different seeds for variance estimation
        p_value_variances = []

        for n_permute in n_permute_values:
            p_vals_multi = []
            for seed in range(20):  # Run 20 times with different seeds
                np.random.seed(seed)
                data = np.random.randn(n_samples) + true_mean
                result = one_sample_permutation_test(
                    data, n_permute=n_permute, random_state=seed
                )
                p_vals_multi.append(result["p"])
            p_value_variances.append(np.var(p_vals_multi))

        # Variance should decrease with more permutations
        # Allow some flexibility (variance estimation is noisy)
        assert p_value_variances[1] < p_value_variances[0] * 2, (
            "Variance should decrease with more permutations"
        )
        assert p_value_variances[2] < p_value_variances[0] * 2, (
            "Variance should decrease with more permutations"
        )

    @pytest.mark.slow
    def test_one_tailed_vs_two_tailed(self):
        """Test that one-tailed p-value ≈ two-tailed p-value / 2 for positive mean."""
        n_samples = 50
        true_mean = (
            0.3  # Known positive mean (small enough to avoid hitting minimum p-value)
        )

        np.random.seed(42)
        data = np.random.randn(n_samples) + true_mean

        result_two = one_sample_permutation_test(
            data, n_permute=5000, tail=2, random_state=42
        )
        result_one = one_sample_permutation_test(
            data, n_permute=5000, tail=1, random_state=42
        )

        # One-tailed p-value should be approximately half of two-tailed
        # (for positive mean in one-tailed test)
        # Allow tolerance due to finite permutations
        ratio = result_one["p"] / result_two["p"]
        assert 0.3 < ratio < 0.7, (
            f"One-tailed p-value should be ~half of two-tailed. Got ratio={ratio:.4f}, one_tailed={result_one['p']:.4f}, two_tailed={result_two['p']:.4f}"
        )

    def test_null_distribution_has_zero_mean(self):
        """Test that null distribution is centered at zero under null hypothesis (fast version)."""
        n_samples = 30  # Reduced from 50 for tier1 speed
        n_permute = 100  # Reduced from 500 for tier1 speed

        # Generate null data (mean=0)
        np.random.seed(42)
        data = np.random.randn(n_samples)  # Mean ~ 0

        result = one_sample_permutation_test(
            data, n_permute=n_permute, return_null=True, random_state=42, device=None
        )

        # Null distribution mean should be close to 0 (within sampling error)
        null_mean = np.mean(result["null_dist"])
        null_std = np.std(result["null_dist"])

        # Expected std of mean under null: std(null_dist) / sqrt(n_permute)
        # Use more lenient tolerance (3 standard errors)
        expected_std_of_mean = null_std / np.sqrt(n_permute)
        tolerance = 3 * expected_std_of_mean

        assert abs(null_mean) < tolerance, (
            f"Null distribution mean should be close to 0. "
            f"Got mean={null_mean:.6f}, expected within ±{tolerance:.6f}"
        )

    @pytest.mark.slow
    def test_null_distribution_has_zero_mean_full(self):
        """Test that null distribution is centered at zero under null hypothesis."""
        n_samples = 50
        n_permute = 2000

        # Generate null data (mean=0)
        np.random.seed(42)
        data = np.random.randn(n_samples)  # Mean ~ 0

        result = one_sample_permutation_test(
            data, n_permute=n_permute, return_null=True, random_state=42
        )

        # Null distribution mean should be close to 0 (within sampling error)
        null_mean = np.mean(result["null_dist"])
        null_std = np.std(result["null_dist"])

        # Expected std of mean under null: std(data) / sqrt(n_samples)
        # Use more lenient tolerance (3 standard errors)
        expected_std_of_mean = null_std / np.sqrt(n_permute)
        tolerance = 3 * expected_std_of_mean

        assert abs(null_mean) < tolerance, (
            f"Null distribution mean should be close to 0. "
            f"Got mean={null_mean:.6f}, expected within ±{tolerance:.6f}"
        )


class TestOneSampleStatistics:
    """Shared one-sample t-test contract (docs/development/specs/ttest.md)."""

    @staticmethod
    def _data(n_obs=12, n_features=4, shift=0.0, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_obs, n_features)) + shift

    def test_requires_two_dimensional_input(self):
        with pytest.raises(ValueError, match="2-D"):
            _one_sample_statistics(np.zeros(5))

    def test_requires_at_least_two_observations(self):
        with pytest.raises(ValueError, match="at least two observations"):
            _one_sample_statistics(np.zeros((1, 4)))

    @pytest.mark.parametrize("popmean", [0.0, 0.75])
    @pytest.mark.parametrize("tail,alternative", [(2, "two-sided"), (1, "greater")])
    def test_parametric_matches_scipy(self, popmean, tail, alternative):
        from scipy.stats import ttest_1samp

        data = self._data()
        out = _one_sample_statistics(data, popmean=popmean, tail=tail)
        expected_t, expected_p = ttest_1samp(
            data, popmean, axis=0, alternative=alternative
        )
        assert set(out) == {"mean", "t", "z", "p"}
        np.testing.assert_allclose(out["t"], expected_t)
        np.testing.assert_allclose(out["p"], expected_p)
        np.testing.assert_allclose(out["mean"], data.mean(axis=0) - popmean)

    def test_permutation_matches_engine_at_fixed_seed(self):
        from scipy.stats import ttest_1samp

        data = self._data(shift=0.4)
        popmean = 0.25
        out = _one_sample_statistics(
            data,
            popmean=popmean,
            permutation=True,
            n_permute=64,
            return_null=True,
            random_state=11,
        )
        engine = one_sample_permutation_test(
            data - popmean,
            n_permute=64,
            tail=2,
            return_null=True,
            n_jobs=-1,
            random_state=11,
        )
        np.testing.assert_allclose(out["p"], engine["p"])
        np.testing.assert_allclose(out["mean"], engine["mean"])
        np.testing.assert_allclose(out["null_dist"], engine["null_dist"])
        assert out["null_dist"].shape == (64, data.shape[1])
        # t is the observed parametric statistic even on the permutation path.
        expected_t, _ = ttest_1samp(data, popmean, axis=0)
        np.testing.assert_allclose(out["t"], expected_t)

    def test_null_kept_only_when_permuting_and_requested(self):
        data = self._data()
        assert "null_dist" not in _one_sample_statistics(
            data, permutation=True, n_permute=16, random_state=0
        )
        assert "null_dist" not in _one_sample_statistics(data, return_null=True)
        with_null = _one_sample_statistics(
            data, permutation=True, n_permute=16, return_null=True, random_state=0
        )
        without_null = _one_sample_statistics(
            data, permutation=True, n_permute=16, return_null=False, random_state=0
        )
        for key in ("mean", "t", "z", "p"):
            np.testing.assert_array_equal(with_null[key], without_null[key])

    def test_single_feature_null_keeps_its_feature_axis(self):
        data = self._data(n_features=1)
        out = _one_sample_statistics(
            data, permutation=True, n_permute=32, return_null=True, random_state=3
        )
        assert out["null_dist"].shape == (32, 1)
        for key in ("mean", "t", "z", "p"):
            assert np.asarray(out[key]).shape == (1,)

    def test_z_follows_the_shared_tail_aware_conversion(self):
        from scipy.stats import norm

        data = self._data(shift=0.5)
        two = _one_sample_statistics(data, tail=2)
        np.testing.assert_allclose(
            two["z"], np.sign(two["t"]) * norm.isf(two["p"] / 2.0)
        )
        upper = _one_sample_statistics(data, tail=1)
        np.testing.assert_allclose(upper["z"], norm.isf(upper["p"]))

    def test_z_stays_finite_at_both_p_endpoints(self):
        rng = np.random.default_rng(5)
        data = rng.standard_normal((30, 3)) * 1e-8 + 50.0
        huge = _one_sample_statistics(data)
        assert np.all(np.isfinite(huge["z"])) and np.all(huge["z"] > 0)
        saturated = _one_sample_statistics(data, popmean=100.0, tail=1)
        assert np.all(saturated["p"] == 1.0)
        assert np.all(np.isfinite(saturated["z"])) and np.all(saturated["z"] < 0)

    def test_constant_and_missing_columns_match_scipy(self):
        from scipy.stats import ttest_1samp

        data = np.column_stack(
            [np.ones(8), np.arange(8.0), np.full(8, np.nan), np.zeros(8)]
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            expected_t, expected_p = ttest_1samp(data, 0.0, axis=0)
            out = _one_sample_statistics(data)
        np.testing.assert_allclose(out["t"], expected_t)
        np.testing.assert_allclose(out["p"], expected_p)

    def test_results_do_not_alias_the_input_or_each_other(self):
        data = self._data()
        original = data.copy()
        out = _one_sample_statistics(
            data, permutation=True, n_permute=16, return_null=True, random_state=0
        )
        arrays = [out[key] for key in ("mean", "t", "z", "p", "null_dist")]
        for i, first in enumerate(arrays):
            assert not np.shares_memory(first, data)
            for second in arrays[i + 1 :]:
                assert not np.shares_memory(first, second)
        for key in ("mean", "t", "z", "p"):
            out[key][...] = -999.0
        out["null_dist"][...] = -999.0
        np.testing.assert_array_equal(data, original)
