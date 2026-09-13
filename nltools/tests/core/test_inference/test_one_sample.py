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

    @pytest.mark.parametrize("n_features", [10])
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

    @pytest.mark.parametrize("n_features", [5])
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

    def test_single_nan_matches_dropping_that_entry(self):
        """A NaN observation drops out of the mean and every permuted mean."""
        data_with_nan = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
        data_dropped = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = one_sample_permutation_test(
            data_with_nan, n_permute=100, random_state=0
        )
        expected = one_sample_permutation_test(
            data_dropped, n_permute=100, random_state=0
        )

        assert np.isfinite(result["mean"])
        assert result["mean"] == pytest.approx(3.0)
        assert result["mean"] == pytest.approx(expected["mean"])
        assert np.isfinite(result["p"])
        assert 0 < result["p"] <= 1


class TestOneSampleStatistics:
    """Shared one-sample t-test contract (development/specs/ttest.md)."""

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

    @pytest.mark.parametrize("popmean", [0.0])
    @pytest.mark.parametrize("tail,alternative", [(2, "two-sided")])
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

    def test_z_stays_finite_at_both_p_endpoints(self):
        rng = np.random.default_rng(5)
        data = rng.standard_normal((30, 3)) * 1e-8 + 50.0
        huge = _one_sample_statistics(data)
        assert np.all(np.isfinite(huge["z"])) and np.all(huge["z"] > 0)
        saturated = _one_sample_statistics(data, popmean=100.0, tail=1)
        assert np.all(saturated["p"] == 1.0)
        assert np.all(np.isfinite(saturated["z"])) and np.all(saturated["z"] < 0)

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
