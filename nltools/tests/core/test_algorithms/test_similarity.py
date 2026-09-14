"""Tests for nltools.algorithms.similarity — similarity and correlation metrics."""

import numpy as np
import pytest

from nltools.algorithms.similarity import (
    fisher_r_to_z,
    fisher_z_to_r,
    compute_similarity,
    _compute_multivariate_similarity,
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

    def test_single_image(self, similarity_data):
        data1, data2 = similarity_data
        result = compute_similarity(data1, data2[0:1], metric="correlation")
        assert result.shape == (10,)

    def test_dot_product(self, similarity_data):
        data1, data2 = similarity_data
        result = compute_similarity(data1, data2, metric="dot_product")
        assert result.shape == (10, 5)

    def test_dot_product_single_row_is_1d(self):
        """A single-row `data1` squeezes like every other metric."""
        result = compute_similarity(
            [[1.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]], metric="dot_product"
        )
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [11.0, 17.0])

    def test_cosine(self, similarity_data):
        data1, data2 = similarity_data
        result = compute_similarity(data1, data2, metric="cosine")
        assert result.shape == (10, 5)
        assert np.all(result >= -1) and np.all(result <= 1)

    def test_invalid_metric(self, similarity_data):
        data1, data2 = similarity_data
        with pytest.raises(ValueError, match="metric must be one of"):
            compute_similarity(data1, data2, metric="invalid")


class TestComputeMultivariateSimilarity:
    """Test OLS regression-based multivariate similarity."""

    def test_matches_the_pinned_ols_reference(self):
        """The returned keys, shapes, types and numbers are pinned."""
        np.random.seed(42)
        y = np.random.randn(100)
        X = np.random.randn(100, 5)
        result = _compute_multivariate_similarity(y, X)

        assert set(result) == {"beta", "t", "p", "df", "sigma", "residual"}
        assert result["beta"].shape == (6,)  # +1 for intercept
        assert result["t"].shape == (6,)
        assert result["p"].shape == (6,)
        assert result["residual"].shape == (100,)
        assert isinstance(result["df"], (int, np.integer))
        assert isinstance(result["sigma"], (float, np.floating))
        assert result["df"] == 94

        np.testing.assert_allclose(
            result["beta"],
            [
                -0.10849785827695631,
                0.09364741258617441,
                0.08596014163021565,
                0.042854585427877456,
                0.09116099699604027,
                -0.08188422884282083,
            ],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            result["t"],
            [
                -1.1430139753361808,
                0.9173498174692328,
                0.9504989127675603,
                0.44499909916504043,
                0.9349766579090509,
                -0.8864999687308583,
            ],
            rtol=1e-12,
        )
        assert result["p"].shape == (6,)
        np.testing.assert_allclose(result["sigma"], 0.9151390832701242, rtol=1e-12)
        np.testing.assert_allclose(
            result["residual"] @ result["residual"], 78.72307692247742, rtol=1e-12
        )

    def test_reported_df_reproduces_the_reported_p(self):
        """A rank-deficient design must not report a df its own p disagrees with.

        `regress` scores t and p against `n - rank(X)`, so recomputing the df
        from the column count would describe a different test than the one the
        returned p-values came from.
        """
        from scipy.stats import t as t_dist

        rng = np.random.default_rng(0)
        x0 = rng.standard_normal(50)
        X = np.column_stack([x0, x0])  # duplicated column: rank 1, not 2
        y = rng.standard_normal(50)

        result = _compute_multivariate_similarity(y, X)

        assert result["df"] == 48  # 50 - rank([1, x0, x0]) = 50 - 2
        np.testing.assert_allclose(
            result["p"],
            2 * (1 - t_dist.cdf(np.abs(result["t"]), result["df"])),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            result["sigma"],
            np.sqrt(result["residual"] @ result["residual"] / result["df"]),
            rtol=1e-12,
        )

    def test_ols_transposed_input(self):
        """A (n_predictors, n_features) X is transposed into place."""
        np.random.seed(42)
        y = np.random.randn(100)
        X = np.random.randn(100, 5)
        r1 = _compute_multivariate_similarity(y, X)
        r2 = _compute_multivariate_similarity(y, X.T)
        np.testing.assert_allclose(r1["beta"], r2["beta"], rtol=1e-10)


class TestTransformPairwise:
    """Test pairwise distance transformations."""

    def test_retained_pairs_are_sign_balanced(self):
        """Signs alternate over retained pairs, not over candidate pairs."""
        X = np.arange(5.0)[:, None]
        y = np.array([0, 0, 0, 1, 0])
        x_new, y_new = transform_pairwise(X, y)
        assert (y_new == 1).sum() == 2
        assert (y_new == -1).sum() == 2
        # Retained pairs are (0,3), (1,3), (2,3), (3,4); each row stays the
        # feature difference flipped together with its label.
        np.testing.assert_allclose(x_new.ravel(), [3.0, -2.0, 1.0, 1.0])
        np.testing.assert_allclose(y_new, [1.0, -1.0, 1.0, -1.0])

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
