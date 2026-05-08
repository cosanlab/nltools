"""Tests for pipeline transform steps (used by MultiSubjectPipeline).

Tests cover:
- NormalizeStep: zscore and minmax normalization
- ReduceStep: PCA and ICA dimensionality reduction
- PipeStep: sklearn transformer wrapper
"""

import numpy as np
import pytest

from nltools.pipelines.steps import (
    FittedNormalize,
    FittedPipe,
    FittedReduce,
    NormalizeStep,
    PipeStep,
    ReduceStep,
)


class TestNormalizeStep:
    """Tests for NormalizeStep transform."""

    def test_zscore_fit_transform(self):
        """Test z-score normalization."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        step = NormalizeStep(method="zscore")

        fitted = step.fit(data)
        transformed = fitted.transform(data)

        # Should have mean ~0 and std ~1 per column
        np.testing.assert_array_almost_equal(transformed.mean(axis=0), [0, 0, 0])
        np.testing.assert_array_almost_equal(transformed.std(axis=0), [1, 1, 1])

    def test_zscore_inverse(self):
        """Test z-score inverse transform."""
        data = np.random.randn(20, 5) * 10 + 50
        step = NormalizeStep(method="zscore")

        fitted = step.fit(data)
        transformed = fitted.transform(data)
        reconstructed = fitted.inverse_transform(transformed)

        np.testing.assert_array_almost_equal(data, reconstructed)

    def test_minmax_fit_transform(self):
        """Test min-max normalization."""
        data = np.array([[0, 10], [5, 20], [10, 30]], dtype=float)
        step = NormalizeStep(method="minmax")

        fitted = step.fit(data)
        transformed = fitted.transform(data)

        # Should be scaled to [0, 1]
        assert transformed.min() >= 0
        assert transformed.max() <= 1
        np.testing.assert_array_almost_equal(transformed.min(axis=0), [0, 0])
        np.testing.assert_array_almost_equal(transformed.max(axis=0), [1, 1])

    def test_minmax_inverse(self):
        """Test min-max inverse transform."""
        data = np.random.randn(20, 5) * 10 + 50
        step = NormalizeStep(method="minmax")

        fitted = step.fit(data)
        transformed = fitted.transform(data)
        reconstructed = fitted.inverse_transform(transformed)

        np.testing.assert_array_almost_equal(data, reconstructed)

    def test_zero_std_handling(self):
        """Test handling of zero standard deviation (constant column)."""
        data = np.array([[5, 1], [5, 2], [5, 3]], dtype=float)
        step = NormalizeStep(method="zscore")

        fitted = step.fit(data)
        transformed = fitted.transform(data)

        # Constant column should become 0, not NaN/Inf
        assert not np.any(np.isnan(transformed))
        assert not np.any(np.isinf(transformed))
        np.testing.assert_array_almost_equal(transformed[:, 0], [0, 0, 0])

    def test_invertible_property(self):
        """Test invertible property is True."""
        step = NormalizeStep()
        assert step.invertible is True

    def test_unknown_method_raises(self):
        """Test unknown method raises error."""
        step = NormalizeStep(method="unknown")
        with pytest.raises(ValueError, match="Unknown normalization method"):
            step.fit(np.random.randn(10, 5))


class TestReduceStep:
    """Tests for ReduceStep dimensionality reduction."""

    def test_pca_fit_transform(self):
        """Test PCA dimensionality reduction."""
        data = np.random.randn(50, 20)
        step = ReduceStep(method="pca", n_components=5, random_state=42)

        fitted = step.fit(data)
        transformed = fitted.transform(data)

        assert transformed.shape == (50, 5)

    def test_pca_inverse(self):
        """Test PCA inverse transform."""
        data = np.random.randn(50, 20)
        step = ReduceStep(method="pca", n_components=10, random_state=42)

        fitted = step.fit(data)
        transformed = fitted.transform(data)
        reconstructed = fitted.inverse_transform(transformed)

        # Reconstruction won't be exact due to dimensionality reduction
        assert reconstructed.shape == data.shape
        # But correlation should be high for first few components
        corr = np.corrcoef(data.ravel(), reconstructed.ravel())[0, 1]
        assert corr > 0.5

    def test_ica_fit_transform(self):
        """Test ICA dimensionality reduction."""
        # ICA needs more samples than components
        data = np.random.randn(100, 20)
        step = ReduceStep(method="ica", n_components=5, random_state=42)

        fitted = step.fit(data)
        transformed = fitted.transform(data)

        assert transformed.shape == (100, 5)

    def test_ica_inverse_exists(self):
        """Test ICA can inverse transform (uses mixing matrix)."""
        # Note: FastICA does support inverse_transform via mixing matrix
        data = np.random.randn(100, 20)
        step = ReduceStep(method="ica", n_components=5, random_state=42)

        fitted = step.fit(data)
        transformed = fitted.transform(data)
        reconstructed = fitted.inverse_transform(transformed)

        # Should get back same shape
        assert reconstructed.shape == data.shape

    def test_invertible_property(self):
        """Test invertible property based on method."""
        # Both PCA and ICA support inverse_transform
        assert ReduceStep(method="pca").invertible is True
        # ICA also has inverse_transform (via mixing matrix)
        # But our implementation marks it as False for safety
        # since the reconstruction may not be exact
        assert ReduceStep(method="ica").invertible is False

    def test_unknown_method_raises(self):
        """Test unknown method raises error."""
        step = ReduceStep(method="unknown")
        with pytest.raises(ValueError, match="Unknown reduction method"):
            step.fit(np.random.randn(50, 20))


class TestPipeStep:
    """Tests for PipeStep sklearn wrapper."""

    def test_sklearn_transformer(self):
        """Test wrapping sklearn transformer."""
        from sklearn.preprocessing import StandardScaler

        data = np.random.randn(30, 10) * 5 + 10
        step = PipeStep(transformer=StandardScaler())

        fitted = step.fit(data)
        transformed = fitted.transform(data)

        # StandardScaler should z-score
        np.testing.assert_array_almost_equal(transformed.mean(axis=0), np.zeros(10))
        np.testing.assert_array_almost_equal(transformed.std(axis=0), np.ones(10))

    def test_inverse_transform(self):
        """Test inverse transform with sklearn."""
        from sklearn.preprocessing import StandardScaler

        data = np.random.randn(30, 10) * 5 + 10
        step = PipeStep(transformer=StandardScaler())

        fitted = step.fit(data)
        transformed = fitted.transform(data)
        reconstructed = fitted.inverse_transform(transformed)

        np.testing.assert_array_almost_equal(data, reconstructed)

    def test_no_inverse_raises(self):
        """Test error when transformer lacks inverse."""
        from sklearn.cluster import KMeans

        # KMeans doesn't have inverse_transform
        step = PipeStep(transformer=KMeans(n_clusters=3))
        assert step.invertible is False

    def test_invertible_property(self):
        """Test invertible property checks transformer."""
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        assert PipeStep(transformer=PCA()).invertible is True
        assert PipeStep(transformer=KMeans()).invertible is False

    def test_clone_preserves_original(self):
        """Test original transformer is not modified."""
        from sklearn.preprocessing import StandardScaler

        original = StandardScaler()
        step = PipeStep(transformer=original)

        data = np.random.randn(30, 10)
        step.fit(data)

        # Original should not be fitted
        assert not hasattr(original, "mean_")


class TestFittedTransforms:
    """Tests for fitted transform dataclasses."""

    def test_fitted_normalize_attrs(self):
        """Test FittedNormalize has expected attributes."""
        fitted = FittedNormalize(
            mean=np.array([0, 0]), std=np.array([1, 1]), method="zscore"
        )
        assert hasattr(fitted, "transform")
        assert hasattr(fitted, "inverse_transform")

    def test_fitted_reduce_attrs(self):
        """Test FittedReduce has expected attributes."""
        from sklearn.decomposition import PCA

        model = PCA(n_components=2).fit(np.random.randn(20, 5))
        fitted = FittedReduce(model=model, method="pca")
        assert hasattr(fitted, "transform")
        assert hasattr(fitted, "inverse_transform")

    def test_fitted_pipe_attrs(self):
        """Test FittedPipe has expected attributes."""
        from sklearn.preprocessing import StandardScaler

        transformer = StandardScaler().fit(np.random.randn(20, 5))
        fitted = FittedPipe(transformer=transformer)
        assert hasattr(fitted, "transform")
        assert hasattr(fitted, "inverse_transform")
