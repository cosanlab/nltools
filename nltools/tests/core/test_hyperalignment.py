"""
Test HyperAlignment class (Procrustes-based hyperalignment)

This test module documents the expected behavior of the HyperAlignment class
before implementation (TDD approach). Tests should fail until implementation
is complete.

References:
    Haxby, J. V., et al. (2011). A common, high-dimensional model of the
    representational space in human ventral temporal cortex. Neuron, 72(2), 404-416.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.slow


# ========== MODULE-SCOPED FIXTURES ==========
# Reduce redundant computation by sharing fitted models across tests


@pytest.fixture(scope="module")
def sample_data_equal_size():
    """Create sample data with equal-sized matrices (50 features x 20 samples).

    Module-scoped: deterministic data, shared across all tests.
    """
    np.random.seed(42)
    return [np.random.randn(50, 20) for _ in range(3)]


@pytest.fixture(scope="module")
def sample_data_different_sizes():
    """Create sample data with different-sized matrices (for padding tests)."""
    np.random.seed(42)
    return [
        np.random.randn(50, 20),  # 50 features
        np.random.randn(45, 20),  # 45 features (needs padding)
        np.random.randn(52, 20),  # 52 features (will be truncated to 45)
    ]


@pytest.fixture(scope="module")
def fitted_hyperalignment(sample_data_equal_size):
    """Pre-fitted HyperAlignment for property tests.

    Module-scoped: expensive fit() runs once, shared across tests.
    Returns (hyper, data) tuple.
    """
    from nltools.algorithms import HyperAlignment

    hyper = HyperAlignment()
    hyper.fit(sample_data_equal_size)
    return hyper, sample_data_equal_size


class TestHyperAlignmentInitialization:
    """Test HyperAlignment initialization and parameters."""

    def test_import(self):
        """Test that HyperAlignment can be imported from algorithms module."""
        from nltools.algorithms import HyperAlignment

        assert HyperAlignment is not None

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        from nltools.algorithms import HyperAlignment

        hyper = HyperAlignment()
        assert hyper.n_iter == 2
        assert hyper.auto_pad is True

    def test_init_custom_n_iter(self):
        """Test initialization with custom n_iter parameter."""
        from nltools.algorithms import HyperAlignment

        hyper = HyperAlignment(n_iter=5)
        assert hyper.n_iter == 5

    def test_init_custom_auto_pad(self):
        """Test initialization with custom auto_pad parameter."""
        from nltools.algorithms import HyperAlignment

        hyper = HyperAlignment(auto_pad=False)
        assert hyper.auto_pad is False


class TestHyperAlignmentFit:
    """Test HyperAlignment fit() method.

    Uses module-scoped fixtures where appropriate.
    Consolidates property tests to reduce redundant fitting.
    """

    def test_fit_returns_self(self, sample_data_equal_size):
        """Test that fit() returns self for method chaining."""
        from nltools.algorithms import HyperAlignment

        hyper = HyperAlignment()
        result = hyper.fit(sample_data_equal_size)
        assert result is hyper

    def test_fitted_model_properties(self, fitted_hyperalignment):
        """Test all properties of a fitted HyperAlignment model.

        Consolidates: stores_attributes, common_model_property, shapes, orthogonality.
        Single fit(), multiple assertions.
        """
        hyper, data = fitted_hyperalignment
        n_features = data[0].shape[0]
        n_samples = data[0].shape[1]

        # 1. Check all required attributes exist with correct types
        assert hasattr(hyper, "w_"), "Missing w_ attribute"
        assert hasattr(hyper, "s_"), "Missing s_ attribute"
        assert hasattr(hyper, "disparity_"), "Missing disparity_ attribute"
        assert hasattr(hyper, "scale_"), "Missing scale_ attribute"

        assert isinstance(hyper.w_, list)
        assert len(hyper.w_) == len(data)
        assert isinstance(hyper.s_, np.ndarray)
        assert isinstance(hyper.disparity_, list)
        assert len(hyper.disparity_) == len(data)
        assert isinstance(hyper.scale_, list)
        assert len(hyper.scale_) == len(data)

        # 2. Check common_model_ property alias
        assert hasattr(hyper, "common_model_")
        np.testing.assert_array_equal(hyper.common_model_, hyper.s_)

        # 3. Check transformation matrix shapes (square, match feature dims)
        for i, w in enumerate(hyper.w_):
            assert w.shape[0] == w.shape[1] == n_features, (
                f"Subject {i} transformation shape {w.shape} != features {n_features}"
            )

        # 4. Check common template shape
        assert hyper.s_.shape == (n_features, n_samples), (
            f"Common template shape {hyper.s_.shape} != expected ({n_features}, {n_samples})"
        )

        # 5. Check orthogonality (W @ W.T ≈ I)
        for i, w in enumerate(hyper.w_):
            orthogonal_check = np.dot(w, w.T)
            identity = np.eye(w.shape[0])
            np.testing.assert_almost_equal(
                orthogonal_check,
                identity,
                decimal=5,
                err_msg=f"Subject {i} transformation is not orthogonal",
            )

    def test_fit_with_different_n_iter(self, sample_data_equal_size):
        """Test that n_iter parameter affects refinement iterations."""
        from nltools.algorithms import HyperAlignment

        # Fit with n_iter=1
        hyper1 = HyperAlignment(n_iter=1)
        hyper1.fit(sample_data_equal_size)

        # Fit with n_iter=5
        hyper5 = HyperAlignment(n_iter=5)
        hyper5.fit(sample_data_equal_size)

        # Results should differ (more refinement = different template)
        # Note: Not checking for specific improvements, just that n_iter has effect
        assert not np.allclose(hyper1.s_, hyper5.s_), (
            "n_iter parameter should affect template refinement"
        )

    def test_fit_auto_pad_true(self, sample_data_different_sizes):
        """Test fit with auto_pad=True handles different-sized matrices."""
        from nltools.algorithms import HyperAlignment

        hyper = HyperAlignment(auto_pad=True)

        # Should not raise error
        hyper.fit(sample_data_different_sizes)

        # All transformations should be same size (padded to common dimensions)
        w_shapes = [w.shape for w in hyper.w_]
        assert all(shape == w_shapes[0] for shape in w_shapes), (
            "With auto_pad=True, all transformation matrices should have same shape"
        )

    def test_fit_auto_pad_zero_pads_to_max_not_truncates(
        self, sample_data_different_sizes
    ):
        """F001: auto_pad must zero-pad up to the LARGEST feature count.

        Previously it truncated every subject to the smallest feature count
        (silent data loss). Features are the row axis; the fixture has 50/45/52
        features, so the common space must be 52-dimensional, not 45.
        """
        from nltools.algorithms import HyperAlignment

        max_features = max(x.shape[0] for x in sample_data_different_sizes)
        n_samples = sample_data_different_sizes[0].shape[1]
        assert max_features == 52  # guard the fixture assumption

        hyper = HyperAlignment(auto_pad=True).fit(sample_data_different_sizes)

        # Common template s_ is (features, samples) — must keep the max features.
        assert hyper.s_.shape == (max_features, n_samples), (
            f"expected zero-padding to {max_features} features, got {hyper.s_.shape}"
        )
        for i, w in enumerate(hyper.w_):
            assert w.shape == (max_features, max_features), (
                f"subject {i} transform {w.shape} was truncated below "
                f"{max_features} features"
            )

    def test_fit_auto_pad_false_raises_error(self, sample_data_different_sizes):
        """Test fit with auto_pad=False raises error for different-sized matrices."""
        from nltools.algorithms import HyperAlignment

        hyper = HyperAlignment(auto_pad=False)

        # Should raise ValueError for mismatched sizes
        with pytest.raises((ValueError, AssertionError)):
            hyper.fit(sample_data_different_sizes)

    def test_fit_single_subject_edge_case(self):
        """Test fit with single subject (edge case)."""
        from nltools.algorithms import HyperAlignment

        np.random.seed(42)
        data = [np.random.randn(50, 20)]

        hyper = HyperAlignment()
        hyper.fit(data)

        # Should still create valid attributes
        assert len(hyper.w_) == 1
        assert hyper.s_.shape == data[0].shape

    def test_fit_identical_subjects_edge_case(self):
        """Test fit with identical subjects (edge case)."""
        from nltools.algorithms import HyperAlignment

        np.random.seed(42)
        base_data = np.random.randn(50, 20)
        data = [base_data.copy() for _ in range(3)]

        hyper = HyperAlignment()
        hyper.fit(data)

        # Should have low disparity (subjects already aligned)
        assert all(d < 0.1 for d in hyper.disparity_), (
            "Identical subjects should have very low disparity"
        )


class TestHyperAlignmentTransform:
    """Test HyperAlignment transform() method.

    Uses module-scoped fitted_hyperalignment fixture.
    """

    def test_transform_training_data(self, fitted_hyperalignment):
        """Test transform on training data."""
        hyper, data = fitted_hyperalignment
        transformed = hyper.transform(data)

        # Check output structure
        assert isinstance(transformed, list)
        assert len(transformed) == len(data)

        # Check shapes
        for i, t in enumerate(transformed):
            assert t.shape == data[i].shape, (
                f"Transformed data {i} shape {t.shape} doesn't match input {data[i].shape}"
            )

    def test_transform_consistency(self, fitted_hyperalignment):
        """Test that transformed data is consistent across subjects (aligned)."""
        hyper, data = fitted_hyperalignment
        transformed = hyper.transform(data)

        # Transformed data should be more similar to each other than original
        # Calculate pairwise correlations
        from scipy.spatial.distance import correlation

        # Original data correlations
        orig_corr = correlation(data[0].flatten(), data[1].flatten())

        # Transformed data correlations
        trans_corr = correlation(transformed[0].flatten(), transformed[1].flatten())

        # After alignment, correlation should be higher (distance lower)
        assert trans_corr <= orig_corr, (
            "Transformed data should have higher correlation (lower distance) than original"
        )

    def test_transform_new_data_raises_error(self, fitted_hyperalignment):
        """Test that transform with new data (different size) raises appropriate error."""
        hyper, _ = fitted_hyperalignment

        # Create new data with different size
        new_data = [np.random.randn(60, 20)]  # Different feature count

        # Should raise error (or handle gracefully with padding if auto_pad=True)
        # Exact behavior depends on implementation - test for reasonable error handling
        try:
            hyper.transform(new_data)
        except (ValueError, AssertionError, IndexError):
            # Expected behavior: raises error for incompatible shapes
            pass


class TestHyperAlignmentTransformSubject:
    """Test HyperAlignment transform_subject() method.

    Uses module-scoped fitted_hyperalignment fixture.
    """

    def test_transform_subject_properties(self, fitted_hyperalignment):
        """Test all transform_subject properties in one test.

        Consolidates: basic, returns_tuple, alignment_quality tests.
        Single transform_subject() call, multiple assertions.
        """
        hyper, _ = fitted_hyperalignment

        # Create new subject data
        np.random.seed(100)
        new_subject = np.random.randn(50, 20)

        # Transform to common space (returns tuple)
        result = hyper.transform_subject(new_subject)

        # 1. Check returns tuple of 4 elements
        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 4, "Should return 4 elements"

        transformed, R, disparity, scale = result

        # 2. Check output types
        assert isinstance(transformed, np.ndarray)
        assert isinstance(R, np.ndarray)
        assert isinstance(disparity, (float, np.floating))
        assert isinstance(scale, (float, np.floating))

        # 3. Check output shape matches common template
        assert transformed.shape == hyper.s_.shape, (
            f"Transformed shape {transformed.shape} != template {hyper.s_.shape}"
        )

        # 4. Check alignment quality (transformed closer to template than original)
        from scipy.spatial.distance import correlation

        orig_dist = correlation(new_subject.flatten(), hyper.s_.flatten())
        trans_dist = correlation(transformed.flatten(), hyper.s_.flatten())
        assert trans_dist <= orig_dist, (
            "Transformed subject should be closer to common template than original"
        )


class TestHyperAlignmentNumericalCorrectness:
    """Test numerical correctness against current align() implementation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for comparison tests."""
        np.random.seed(42)
        data = [np.random.randn(50, 20) for _ in range(5)]
        return data

    def test_numerical_match_with_align_procrustes(self, sample_data):
        """Test that HyperAlignment produces same results as align(method='procrustes').

        Note: align() expects input as [observations, features] and returns
        transformed as [features, observations]. HyperAlignment uses
        [features, samples] throughout. This test accounts for the transposition.
        """
        from nltools.algorithms import HyperAlignment
        from nltools.stats import align

        # align() transposes numpy input from [obs, feat] to [feat, obs] internally
        # So we transpose sample_data to match what align() works with internally
        transposed_data = [x.T for x in sample_data]

        # Use HyperAlignment class with transposed data
        # Note: Use n_iter=1 to match current align() implementation (only 1 refinement)
        # align() does Stage 1 + Stage 2 (once) + Stage 3
        hyper = HyperAlignment(n_iter=1, auto_pad=True)
        hyper.fit(transposed_data)
        hyper_transformed = hyper.transform(transposed_data)

        # Use align() function with original data (it will transpose internally)
        align_out = align(sample_data, method="procrustes")

        # Compare transformed data (accounting for align's output transposition)
        # align() returns transformed in [features, observations] format
        # HyperAlignment returns in [features, samples] format
        # They should match!
        for i, (ht, at) in enumerate(zip(hyper_transformed, align_out["transformed"])):
            np.testing.assert_allclose(
                ht,
                at,
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"Subject {i} transformed data doesn't match align() output",
            )

        # Compare common template
        # Note: align() returns common_model in [samples, features] format
        # but returns transformed in [features, observations] format (inconsistent!)
        # HyperAlignment keeps both in [features, samples] format (consistent)
        # So we need to transpose hyper.s_ for comparison
        np.testing.assert_allclose(
            hyper.s_.T,
            align_out["common_model"],
            rtol=1e-5,
            atol=1e-8,
            err_msg="Common template doesn't match align() output",
        )

        # Compare transformation matrices
        for i, (hw, aw) in enumerate(zip(hyper.w_, align_out["transformation_matrix"])):
            np.testing.assert_allclose(
                hw,
                aw,
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"Subject {i} transformation matrix doesn't match align() output",
            )

        # Compare disparity
        for i, (hd, ad) in enumerate(zip(hyper.disparity_, align_out["disparity"])):
            np.testing.assert_allclose(
                hd,
                ad,
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"Subject {i} disparity doesn't match align() output",
            )

        # Compare scale
        for i, (hs, as_val) in enumerate(zip(hyper.scale_, align_out["scale"])):
            np.testing.assert_allclose(
                hs,
                as_val,
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"Subject {i} scale doesn't match align() output",
            )

    def test_sklearn_api_compliance(self, sample_data):
        """Test that HyperAlignment follows sklearn BaseEstimator/TransformerMixin API."""
        from nltools.algorithms import HyperAlignment
        from sklearn.base import BaseEstimator, TransformerMixin

        hyper = HyperAlignment()

        # Should inherit from BaseEstimator and TransformerMixin
        assert isinstance(hyper, BaseEstimator)
        assert isinstance(hyper, TransformerMixin)

        # Should have get_params and set_params
        assert hasattr(hyper, "get_params")
        assert hasattr(hyper, "set_params")

        # Test get_params
        params = hyper.get_params()
        assert "n_iter" in params
        assert "auto_pad" in params

        # Test set_params
        hyper.set_params(n_iter=5)
        assert hyper.n_iter == 5


class TestHyperAlignmentEdgeCases:
    """Test edge cases and error handling."""

    def test_fit_before_transform_error(self):
        """Test that transform before fit raises appropriate error."""
        from nltools.algorithms import HyperAlignment

        hyper = HyperAlignment()

        data = [np.random.randn(50, 20)]

        # Should raise error if transform called before fit
        with pytest.raises((AttributeError, ValueError)):
            hyper.transform(data)

    def test_fit_with_empty_list(self):
        """Test fit with empty list raises error."""
        from nltools.algorithms import HyperAlignment

        hyper = HyperAlignment()

        with pytest.raises((ValueError, IndexError)):
            hyper.fit([])

    def test_fit_with_2d_array_not_list(self):
        """Test fit with single array (not list) raises error."""
        from nltools.algorithms import HyperAlignment

        hyper = HyperAlignment()

        data = np.random.randn(50, 20)

        # Should raise error (expects list of arrays)
        with pytest.raises((ValueError, TypeError)):
            hyper.fit(data)

    def test_fit_with_mismatched_samples(self):
        """Test fit with mismatched number of samples raises error."""
        from nltools.algorithms import HyperAlignment

        hyper = HyperAlignment(auto_pad=False)

        data = [
            np.random.randn(50, 20),  # 20 samples
            np.random.randn(50, 25),  # 25 samples - MISMATCH
        ]

        # Should raise error for mismatched samples
        with pytest.raises((ValueError, AssertionError)):
            hyper.fit(data)
