"""Tests for nltools.stats.correlation — similarity and correlation metrics."""

import numpy as np
import pytest

from nltools.stats.correlation import (
    fisher_r_to_z,
    fisher_z_to_r,
    compute_similarity,
    compute_multivariate_similarity,
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
        result = compute_similarity(data1, data2, metric="correlation")
        assert result.shape == (10, 5)
        assert np.all(result >= -1) and np.all(result <= 1)

    def test_pearson_alias(self, similarity_data):
        data1, data2 = similarity_data
        r1 = compute_similarity(data1, data2, metric="correlation")
        r2 = compute_similarity(data1, data2, metric="pearson")
        np.testing.assert_allclose(r1, r2, rtol=1e-7, atol=1e-10)

    def test_single_image(self, similarity_data):
        data1, data2 = similarity_data
        result = compute_similarity(data1, data2[0:1], metric="correlation")
        assert result.shape == (10,)

    def test_self_similarity(self, similarity_data):
        data1, _ = similarity_data
        result = compute_similarity(data1, data1, metric="correlation")
        assert result.shape == (10, 10)
        np.testing.assert_allclose(np.diag(result), 1.0, rtol=1e-10)

    def test_spearman(self, similarity_data):
        data1, data2 = similarity_data
        result = compute_similarity(data1, data2, metric="spearman")
        assert result.shape == (10, 5)
        assert np.all(result >= -1) and np.all(result <= 1)

    def test_rank_correlation_alias(self, similarity_data):
        data1, data2 = similarity_data
        r1 = compute_similarity(data1, data2, metric="spearman")
        r2 = compute_similarity(data1, data2, metric="rank_correlation")
        np.testing.assert_allclose(r1, r2, rtol=1e-7, atol=1e-10)

    def test_dot_product(self, similarity_data):
        data1, data2 = similarity_data
        result = compute_similarity(data1, data2, metric="dot_product")
        assert result.shape == (10, 5)

    def test_dot_product_single(self, similarity_data):
        data1, data2 = similarity_data
        result = compute_similarity(data1, data2[0:1], metric="dot_product")
        assert result.shape == (10,)

    def test_cosine(self, similarity_data):
        data1, data2 = similarity_data
        result = compute_similarity(data1, data2, metric="cosine")
        assert result.shape == (10, 5)
        assert np.all(result >= -1) and np.all(result <= 1)

    def test_cosine_self_similarity(self, similarity_data):
        data1, _ = similarity_data
        result = compute_similarity(data1, data1, metric="cosine")
        np.testing.assert_allclose(np.diag(result), 1.0, rtol=1e-5)

    def test_invalid_metric(self, similarity_data):
        data1, data2 = similarity_data
        with pytest.raises(ValueError, match="metric must be one of"):
            compute_similarity(data1, data2, metric="invalid")


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
