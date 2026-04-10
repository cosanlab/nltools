"""Tests for nltools.stats.correlation — similarity, correlation metrics, and ICC."""

import numpy as np
import pytest

from nltools.stats.correlation import (
    fisher_r_to_z,
    fisher_z_to_r,
    compute_similarity,
    compute_multivariate_similarity,
    compute_icc,
    transform_pairwise,
)


class TestFisherTransform:
    """Test Fisher r-to-z transformation and its inverse."""

    def test_roundtrip(self):
        """Fisher r→z→r should recover original value."""
        for r in np.arange(0, 1, 0.05):
            np.testing.assert_almost_equal(
                r, fisher_z_to_r(fisher_r_to_z(r)), decimal=3
            )


class TestComputeSimilarity:
    """Test compute_similarity with various metrics."""

    @pytest.fixture
    def similarity_data(self):
        np.random.seed(42)
        data1 = np.random.randn(10, 100)
        data2 = np.random.randn(5, 100)
        return data1, data2

    def test_correlation(self, similarity_data):
        data1, data2 = similarity_data
        result = compute_similarity(data1, data2, method="correlation")
        assert result.shape == (10, 5)
        assert np.all(-1 <= result) and np.all(result <= 1)

    def test_pearson_alias(self, similarity_data):
        data1, data2 = similarity_data
        r1 = compute_similarity(data1, data2, method="correlation")
        r2 = compute_similarity(data1, data2, method="pearson")
        np.testing.assert_allclose(r1, r2)

    def test_single_image(self, similarity_data):
        data1, data2 = similarity_data
        result = compute_similarity(data1, data2[0:1], method="correlation")
        assert result.shape == (10,)

    def test_self_similarity(self, similarity_data):
        data1, _ = similarity_data
        result = compute_similarity(data1, data1, method="correlation")
        assert result.shape == (10, 10)
        np.testing.assert_allclose(np.diag(result), 1.0, rtol=1e-10)

    def test_spearman(self, similarity_data):
        data1, data2 = similarity_data
        result = compute_similarity(data1, data2, method="spearman")
        assert result.shape == (10, 5)
        assert np.all(-1 <= result) and np.all(result <= 1)

    def test_rank_correlation_alias(self, similarity_data):
        data1, data2 = similarity_data
        r1 = compute_similarity(data1, data2, method="spearman")
        r2 = compute_similarity(data1, data2, method="rank_correlation")
        np.testing.assert_allclose(r1, r2)

    def test_dot_product(self, similarity_data):
        data1, data2 = similarity_data
        result = compute_similarity(data1, data2, method="dot_product")
        assert result.shape == (10, 5)

    def test_dot_product_single(self, similarity_data):
        data1, data2 = similarity_data
        result = compute_similarity(data1, data2[0:1], method="dot_product")
        assert result.shape == (10,)

    def test_cosine(self, similarity_data):
        data1, data2 = similarity_data
        result = compute_similarity(data1, data2, method="cosine")
        assert result.shape == (10, 5)
        assert np.all(-1 <= result) and np.all(result <= 1)

    def test_cosine_self_similarity(self, similarity_data):
        data1, _ = similarity_data
        result = compute_similarity(data1, data1, method="cosine")
        np.testing.assert_allclose(np.diag(result), 1.0, rtol=1e-5)

    def test_invalid_method(self, similarity_data):
        data1, data2 = similarity_data
        with pytest.raises(ValueError, match="method must be one of"):
            compute_similarity(data1, data2, method="invalid")


class TestComputeMultivariateSimilarity:
    """Test OLS regression-based multivariate similarity."""

    def test_ols_basic(self):
        np.random.seed(42)
        y = np.random.randn(100)
        X = np.random.randn(100, 5)
        result = compute_multivariate_similarity(y, X, method="ols")

        required_keys = ["beta", "t", "p", "df", "sigma", "residual"]
        for key in required_keys:
            assert key in result

        assert result["beta"].shape == (6,)  # +1 for intercept
        assert result["t"].shape == (6,)
        assert result["p"].shape == (6,)
        assert isinstance(result["df"], (int, np.integer))
        assert isinstance(result["sigma"], (float, np.floating))
        assert result["residual"].shape == (100,)
        assert np.all(result["p"] >= 0) and np.all(result["p"] <= 1)
        assert result["df"] > 0
        assert result["sigma"] >= 0

    def test_ols_residuals(self):
        np.random.seed(42)
        y = np.random.randn(100)
        X = np.random.randn(100, 5)
        result = compute_multivariate_similarity(y, X, method="ols")
        expected = y - (result["beta"][0] + np.dot(X, result["beta"][1:]))
        np.testing.assert_allclose(result["residual"], expected, rtol=1e-10)

    def test_ols_transposed_input(self):
        np.random.seed(42)
        y = np.random.randn(100)
        X = np.random.randn(100, 5)
        r1 = compute_multivariate_similarity(y, X, method="ols")
        r2 = compute_multivariate_similarity(y, X.T, method="ols")
        np.testing.assert_allclose(r1["beta"], r2["beta"], rtol=1e-10)

    def test_invalid_method(self):
        np.random.seed(42)
        with pytest.raises(NotImplementedError):
            compute_multivariate_similarity(
                np.random.randn(100), np.random.randn(100, 5), method="ridge"
            )


class TestComputeICC:
    """Test intraclass correlation coefficient."""

    def test_icc2(self, icc_data):
        icc = compute_icc(icc_data, icc_type="icc2")
        assert isinstance(icc, (float, np.floating))
        assert -1 <= icc <= 1

    def test_icc3(self, icc_data):
        icc = compute_icc(icc_data, icc_type="icc3")
        assert isinstance(icc, (float, np.floating))
        assert -1 <= icc <= 1

    def test_icc1(self, icc_data):
        icc = compute_icc(icc_data, icc_type="icc1")
        assert isinstance(icc, (float, np.floating))
        assert -1 <= icc <= 1

    def test_icc1_equals_icc3(self, icc_data):
        """ICC1 and ICC3 use the same formula (different assumptions)."""
        icc1 = compute_icc(icc_data, icc_type="icc1")
        icc3 = compute_icc(icc_data, icc_type="icc3")
        np.testing.assert_almost_equal(icc1, icc3, decimal=10)

    def test_icc1_known_high_reliability(self):
        """High between-subject variance should produce high ICC."""
        np.random.seed(42)
        subject_effects = np.linspace(-2, 2, 10)
        Y = np.zeros((10, 5))
        for i in range(10):
            Y[i, :] = subject_effects[i] + np.random.randn(5) * 0.1
        icc1 = compute_icc(Y, icc_type="icc1")
        assert icc1 > 0.5

    def test_icc_zero_reliability(self):
        """Pure noise should produce ICC near 0 or negative."""
        np.random.seed(42)
        Y = np.random.randn(10, 5)
        icc1 = compute_icc(Y, icc_type="icc1")
        assert icc1 < 0.5

    def test_icc_perfect_reliability(self):
        """Identical values across sessions should produce ICC ~1."""
        subject_effects = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        Y = np.zeros((5, 3))
        for i in range(5):
            Y[i, :] = subject_effects[i]
        icc1 = compute_icc(Y, icc_type="icc1")
        icc2 = compute_icc(Y, icc_type="icc2")
        assert icc1 > 0.99
        assert icc2 > 0.99

    def test_icc2_le_icc1_with_session_effects(self):
        """ICC2 should be <= ICC1 when session effects exist."""
        np.random.seed(42)
        subject_effects = np.linspace(-1, 1, 10)
        session_effects = np.array([2.0, -2.0, 1.0, -1.0, 0.0])
        Y = np.zeros((10, 5))
        for i in range(10):
            for j in range(5):
                Y[i, j] = (
                    subject_effects[i] + session_effects[j] + np.random.randn() * 0.1
                )
        icc1 = compute_icc(Y, icc_type="icc1")
        icc2 = compute_icc(Y, icc_type="icc2")
        assert icc2 <= icc1 + 1e-10

    def test_icc_formula_manual(self):
        """Verify against manual Shrout & Fleiss (1979) calculation."""
        Y = np.array([[1.0, 1.1, 0.9], [2.0, 2.1, 1.9], [3.0, 3.1, 2.9]])
        grand_mean = np.mean(Y)
        n, k = Y.shape
        SSR = ((np.mean(Y, axis=1) - grand_mean) ** 2).sum() * k
        SSC = ((np.mean(Y, axis=0) - grand_mean) ** 2).sum() * n
        SST = ((Y - grand_mean) ** 2).sum()
        SSE = SST - SSR - SSC
        MSR = SSR / (n - 1)
        MSC = SSC / (k - 1)
        MSE = SSE / ((n - 1) * (k - 1))
        icc1_manual = (MSR - MSE) / (MSR + (k - 1) * MSE)
        icc2_manual = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)

        np.testing.assert_almost_equal(
            compute_icc(Y, icc_type="icc1"), icc1_manual, decimal=10
        )
        np.testing.assert_almost_equal(
            compute_icc(Y, icc_type="icc2"), icc2_manual, decimal=10
        )

    def test_icc_effect_size_sensitivity(self):
        """Higher reliability should produce higher ICC values."""
        reliability_levels = [
            (1.0, 0.5),  # Low: high noise, low signal
            (0.5, 0.5),  # Medium-low
            (0.5, 1.0),  # Medium
            (0.5, 2.0),  # High: low noise, high signal
        ]
        icc_values = []
        for idx, (noise_level, signal_level) in enumerate(reliability_levels):
            np.random.seed(42 + idx)
            subject_effects = np.linspace(-signal_level, signal_level, 10)
            Y = np.zeros((10, 5))
            for i in range(10):
                Y[i, :] = subject_effects[i] + np.random.randn(5) * noise_level
            icc_values.append(compute_icc(Y, icc_type="icc1"))

        for i in range(len(icc_values) - 1):
            assert icc_values[i] <= icc_values[i + 1] + 0.05

    def test_invalid_type(self, icc_data):
        with pytest.raises(ValueError, match="icc_type must be"):
            compute_icc(icc_data, icc_type="invalid")


class TestTransformPairwise:
    """Test pairwise distance transformations."""

    def test_without_groups(self):
        n_features, n_samples = 50, 100
        new_n = int(n_samples * (n_samples - 1) / 2)
        X = np.random.rand(n_samples, n_features)
        y = np.random.rand(n_samples)
        x_new, y_new = transform_pairwise(X, y)
        assert x_new.shape == (new_n, n_features)
        assert y_new.shape == (new_n,)
        assert y_new.ndim == 1

    def test_with_groups(self):
        n_features, n_samples, n_subs = 50, 100, 4
        new_n = int(n_subs * ((n_samples / n_subs) * (n_samples / n_subs - 1)) / 2)
        X = np.random.rand(n_samples, n_features)
        y = np.random.rand(n_samples)
        groups = np.repeat(np.arange(1, 1 + n_subs), n_samples / n_subs)
        y = np.vstack((y, groups)).T
        x_new, y_new = transform_pairwise(X, y)
        assert x_new.shape == (new_n, n_features)
        assert y_new.shape == (new_n, 2)
        a = y_new[:, 1] == np.repeat(
            np.arange(1, 1 + n_subs),
            ((n_samples / n_subs) * (n_samples / n_subs - 1)) / 2,
        )
        assert a.all()
