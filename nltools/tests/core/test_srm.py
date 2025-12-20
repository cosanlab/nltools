"""
Tests for Shared Response Model (SRM) algorithms.

Testing philosophy: Property-based tests with mathematical invariants
rather than golden outputs (following BrainIAK, PyMVPA, Hypertools best practices).

Based on research documented in claude-guidelines/srm-hyperalignment-testing-strategy.md
"""

import pytest
import numpy as np
from nltools.algorithms.srm import SRM, DetSRM
from sklearn.exceptions import NotFittedError


# ========== FIXTURES ==========
# Module-scoped fixtures reduce redundant computation across tests


@pytest.fixture(scope="module")
def multi_subject_data():
    """Generate synthetic multi-subject data with known shared structure.

    Creates data where subjects share common latent structure but have
    different observation spaces (different voxel counts).

    Module-scoped: deterministic data, shared across all tests in module.
    """
    n_timepoints = 100
    n_features = 10  # True latent dimensionality

    # True shared response (ground truth)
    np.random.seed(42)
    shared = np.random.randn(n_features, n_timepoints)

    # Generate subject-specific data with variable voxel counts
    subjects = []
    true_transforms = []
    voxel_counts = [200, 180, 220, 190, 210]  # 5 subjects with variable sizes

    for voxels in voxel_counts:
        # Random orthogonal projection (ground truth)
        w = np.linalg.qr(np.random.randn(voxels, n_features))[0]
        true_transforms.append(w)

        # Subject data = projection @ shared + small noise
        data = w @ shared + 0.01 * np.random.randn(voxels, n_timepoints)
        subjects.append(data)

    return {
        "data": subjects,
        "shared": shared,
        "transforms": true_transforms,
        "voxels": voxel_counts,
        "timepoints": n_timepoints,
        "features": n_features,
    }


@pytest.fixture(scope="module")
def fitted_srm(multi_subject_data):
    """Pre-fitted SRM model for property tests.

    Module-scoped: expensive fit() runs once, shared across tests.
    Uses n_iter=10 for good convergence.
    """
    srm = SRM(features=10, n_iter=10, rand_seed=42)
    srm.fit(multi_subject_data["data"])
    return srm


@pytest.fixture(scope="module")
def fitted_detsrm(multi_subject_data):
    """Pre-fitted DetSRM model for property tests.

    Module-scoped: expensive fit() runs once, shared across tests.
    """
    detsrm = DetSRM(features=10, n_iter=10, rand_seed=42)
    detsrm.fit(multi_subject_data["data"])
    return detsrm


@pytest.fixture(scope="module")
def identical_subjects():
    """Data where all subjects are identical (edge case)."""
    n_subjects = 3
    np.random.seed(123)
    base_data = np.random.randn(100, 50)
    return [base_data.copy() for _ in range(n_subjects)]


@pytest.fixture(scope="module")
def single_subject():
    """Single subject data (should error)."""
    np.random.seed(456)
    return [np.random.randn(100, 50)]


@pytest.fixture(scope="module")
def minimal_brain_data():
    """Minimal synthetic data for quick tests."""
    np.random.seed(789)
    n_subjects = 3
    n_voxels = 50
    n_timepoints = 30

    # Shared structure with noise
    shared = np.random.randn(10, n_timepoints)
    subjects = []
    for _ in range(n_subjects):
        w = np.linalg.qr(np.random.randn(n_voxels, 10))[0]
        data = w @ shared + 0.05 * np.random.randn(n_voxels, n_timepoints)
        subjects.append(data)

    return subjects


# ========== INITIALIZATION TESTS ==========


class TestSRMInitialization:
    """Test SRM initialization and parameter validation."""

    def test_srm_init_defaults(self):
        """Test SRM initializes with correct defaults."""
        srm = SRM()
        assert srm.n_iter == 10
        assert srm.features == 50
        assert srm.rand_seed == 0

    def test_srm_init_custom_params(self):
        """Test SRM accepts custom parameters."""
        srm = SRM(n_iter=20, features=30, rand_seed=123)
        assert srm.n_iter == 20
        assert srm.features == 30
        assert srm.rand_seed == 123

    def test_detsrm_init_defaults(self):
        """Test DetSRM initializes with correct defaults."""
        detsrm = DetSRM()
        assert detsrm.n_iter == 10
        assert detsrm.features == 50
        assert detsrm.rand_seed == 0

    def test_detsrm_init_custom_params(self):
        """Test DetSRM accepts custom parameters."""
        detsrm = DetSRM(n_iter=15, features=25, rand_seed=999)
        assert detsrm.n_iter == 15
        assert detsrm.features == 25
        assert detsrm.rand_seed == 999


# ========== CONTRACT TESTS (Interface/API) ==========


class TestSRMContract:
    """Test SRM API contracts and error handling."""

    def test_fit_before_transform_error(self, multi_subject_data):
        """Test that transform raises error before fit."""
        srm = SRM()
        with pytest.raises(NotFittedError, match="model fit has not been run"):
            srm.transform(multi_subject_data["data"])

    def test_fit_before_transform_subject_error(self, multi_subject_data):
        """Test that transform_subject raises error before fit."""
        srm = SRM()
        with pytest.raises(NotFittedError, match="model fit has not been run"):
            srm.transform_subject(multi_subject_data["data"][0])

    def test_fit_single_subject_error(self, single_subject):
        """Test error with only 1 subject (need multiple)."""
        srm = SRM()
        with pytest.raises(ValueError, match="not enough subjects"):
            srm.fit(single_subject)

    def test_fit_mismatched_timepoints(self):
        """Test error when subjects have different timepoints."""
        np.random.seed(111)
        data = [
            np.random.randn(100, 50),  # 50 timepoints
            np.random.randn(100, 60),  # 60 timepoints
        ]
        srm = SRM()
        with pytest.raises(ValueError, match="Different number of samples"):
            srm.fit(data)

    def test_fit_insufficient_samples(self):
        """Test error when samples < features."""
        np.random.seed(222)
        data = [
            np.random.randn(100, 40),  # 40 samples
            np.random.randn(100, 40),
        ]
        srm = SRM(features=50)  # More features than samples
        with pytest.raises(ValueError, match="not enough samples"):
            srm.fit(data)

    def test_fit_sets_attributes(self, multi_subject_data):
        """Test that fit() creates required attributes."""
        srm = SRM(features=10, n_iter=2)
        srm.fit(multi_subject_data["data"])

        # Check fitted attributes exist
        assert hasattr(srm, "w_")
        assert hasattr(srm, "s_")
        assert hasattr(srm, "sigma_s_")
        assert hasattr(srm, "mu_")
        assert hasattr(srm, "rho2_")

        # Check correct types and shapes
        assert isinstance(srm.w_, list)
        assert len(srm.w_) == len(multi_subject_data["data"])
        assert srm.s_.shape == (10, multi_subject_data["timepoints"])

    def test_transform_wrong_subject_count(self, multi_subject_data):
        """Test error when transforming different number of subjects."""
        srm = SRM(features=10, n_iter=2)
        srm.fit(multi_subject_data["data"])

        # Try to transform different number of subjects
        wrong_data = multi_subject_data["data"][:3]  # Only 3 instead of 5
        with pytest.raises(ValueError, match="number of subjects does not match"):
            srm.transform(wrong_data)

    def test_transform_subject_wrong_timepoints(self, multi_subject_data):
        """Test error when new subject has different timepoints."""
        srm = SRM(features=10, n_iter=2)
        srm.fit(multi_subject_data["data"])

        # New subject with wrong timepoint count
        np.random.seed(333)
        wrong_subject = np.random.randn(100, 60)  # 60 instead of 100
        with pytest.raises(ValueError, match="number of timepoints.*does not match"):
            srm.transform_subject(wrong_subject)


# ========== MATHEMATICAL PROPERTY TESTS ==========


class TestSRMMathematicalProperties:
    """Test mathematical properties that must hold for correct SRM.

    Uses module-scoped fitted_srm fixture to avoid redundant fitting.
    All properties tested with multiple assertions per computation.
    """

    def test_fitted_model_properties(self, fitted_srm, multi_subject_data):
        """Test all mathematical invariants of a fitted SRM model.

        Consolidates: orthogonality, reconstruction, shape, variance tests.
        Single fit(), multiple assertions.
        """
        # 1. Check shared response shape
        expected_shape = (10, multi_subject_data["timepoints"])
        assert fitted_srm.s_.shape == expected_shape, (
            f"Shared response shape {fitted_srm.s_.shape} != expected {expected_shape}"
        )

        # 2. Check orthogonality of all W_i matrices (W.T @ W ≈ I)
        for i, w in enumerate(fitted_srm.w_):
            gram = w.T @ w
            identity = np.eye(w.shape[1])
            ortho_error = np.linalg.norm(gram - identity, "fro")
            assert ortho_error < 1e-5, (
                f"Subject {i}: W.T @ W not orthogonal (error={ortho_error:.2e})"
            )

        # 3. Check reconstruction quality (X_i ≈ W_i @ S)
        for i, (x, w) in enumerate(zip(multi_subject_data["data"], fitted_srm.w_)):
            x_centered = x - x.mean(axis=1, keepdims=True)
            reconstruction = w @ fitted_srm.s_
            error = np.linalg.norm(x_centered - reconstruction, "fro")
            data_norm = np.linalg.norm(x_centered, "fro")
            relative_error = error / data_norm
            assert relative_error < 0.5, (
                f"Subject {i}: Poor reconstruction (error={relative_error:.2%})"
            )

    def test_transform_properties(self, fitted_srm, multi_subject_data):
        """Test transform output properties.

        Single transform(), multiple assertions on shape and variance.
        """
        # Transform once
        transformed = fitted_srm.transform(multi_subject_data["data"])

        # 1. Check shape preservation for each subject
        for i, s in enumerate(transformed):
            expected_shape = (10, multi_subject_data["timepoints"])
            assert s.shape == expected_shape, (
                f"Subject {i}: Wrong shape {s.shape}, expected {expected_shape}"
            )

        # 2. Check variance preservation
        original_var = np.mean([np.var(x) for x in multi_subject_data["data"]])
        transformed_var = np.mean([np.var(s) for s in transformed])
        assert transformed_var > 0.3 * original_var, (
            f"Variance not preserved: {transformed_var:.4f} < 0.3 * {original_var:.4f}"
        )


# ========== EDGE CASES ==========


class TestSRMEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_identical_subjects(self, identical_subjects):
        """Test SRM with identical subjects.

        Note: Even with identical subjects, perfect reconstruction is not guaranteed
        due to dimensionality reduction (voxels > features) and iterative optimization.
        We just verify the algorithm runs without error.
        """
        srm = SRM(features=10, n_iter=5)
        srm.fit(identical_subjects)

        # Algorithm should complete without error
        # Check that basic properties hold
        for i, w in enumerate(srm.w_):
            # Orthogonality should still hold
            gram = w.T @ w
            identity = np.eye(w.shape[1])
            ortho_error = np.linalg.norm(gram - identity, "fro")
            assert ortho_error < 1e-5

    def test_deterministic_with_seed(self, multi_subject_data):
        """Test reproducibility with same random seed."""
        srm1 = SRM(features=10, n_iter=5, rand_seed=42)
        srm1.fit(multi_subject_data["data"])

        srm2 = SRM(features=10, n_iter=5, rand_seed=42)
        srm2.fit(multi_subject_data["data"])

        # Should produce identical results
        np.testing.assert_array_almost_equal(srm1.s_, srm2.s_, decimal=10)

        for w1, w2 in zip(srm1.w_, srm2.w_):
            np.testing.assert_array_almost_equal(w1, w2, decimal=10)

    def test_different_seed_different_results(self, multi_subject_data):
        """Test that different seeds produce different initializations."""
        srm1 = SRM(features=10, n_iter=1, rand_seed=42)
        srm1.fit(multi_subject_data["data"])

        srm2 = SRM(features=10, n_iter=1, rand_seed=123)
        srm2.fit(multi_subject_data["data"])

        # Should produce different results (due to random init)
        with pytest.raises(AssertionError):
            np.testing.assert_array_almost_equal(srm1.s_, srm2.s_, decimal=5)

    def test_transform_subject_new_data(self, multi_subject_data):
        """Test transform_subject() with new subject data."""
        srm = SRM(features=10, n_iter=5)
        srm.fit(multi_subject_data["data"])

        # Create new subject with same shared response but different projection
        np.random.seed(999)
        new_voxels = 150
        new_w = np.linalg.qr(np.random.randn(new_voxels, 10))[0]
        new_subject_data = new_w @ multi_subject_data[
            "shared"
        ] + 0.01 * np.random.randn(new_voxels, multi_subject_data["timepoints"])

        # Transform new subject
        new_w_learned = srm.transform_subject(new_subject_data)

        # Check that learned transform has orthonormal columns
        gram = new_w_learned.T @ new_w_learned
        identity = np.eye(new_w_learned.shape[1])  # features x features
        ortho_error = np.linalg.norm(gram - identity, "fro")

        assert ortho_error < 1e-5, (
            f"New subject transform not orthogonal (error={ortho_error:.2e})"
        )

    def test_minimal_features(self, minimal_brain_data):
        """Test SRM with very small number of features."""
        srm = SRM(features=3, n_iter=5)
        srm.fit(minimal_brain_data)

        # Should still produce valid orthogonal transforms
        for i, w in enumerate(srm.w_):
            gram = w.T @ w
            identity = np.eye(w.shape[1])  # features x features
            ortho_error = np.linalg.norm(gram - identity, "fro")
            assert ortho_error < 1e-5

    def test_many_iterations(self, minimal_brain_data):
        """Test SRM with many iterations converges."""
        srm = SRM(features=10, n_iter=50)
        srm.fit(minimal_brain_data)

        # Should still maintain orthogonality
        for w in srm.w_:
            gram = w.T @ w
            identity = np.eye(w.shape[1])  # features x features
            ortho_error = np.linalg.norm(gram - identity, "fro")
            assert ortho_error < 1e-5


# ========== DETSRM TESTS ==========
# Note: Integration with align() function is already tested in
# test_stats.py::test_align_without_isc() which covers both SRM methods


class TestDetSRMMathematicalProperties:
    """Test mathematical properties for deterministic SRM.

    Uses module-scoped fitted_detsrm fixture to avoid redundant fitting.
    """

    def test_fitted_detsrm_properties(self, fitted_detsrm, multi_subject_data):
        """Test all mathematical invariants of a fitted DetSRM model.

        Consolidates: orthogonality, reconstruction, shape tests.
        Single fit(), multiple assertions.
        """
        # 1. Check shared response shape
        expected_shape = (10, multi_subject_data["timepoints"])
        assert fitted_detsrm.s_.shape == expected_shape, (
            f"DetSRM shared response shape {fitted_detsrm.s_.shape} != {expected_shape}"
        )

        # 2. Check orthogonality of all W_i matrices
        for i, w in enumerate(fitted_detsrm.w_):
            gram = w.T @ w
            identity = np.eye(w.shape[1])
            ortho_error = np.linalg.norm(gram - identity, "fro")
            assert ortho_error < 1e-5, (
                f"DetSRM Subject {i}: W.T @ W not orthogonal (error={ortho_error:.2e})"
            )

        # 3. Check reconstruction quality
        for i, (x, w) in enumerate(zip(multi_subject_data["data"], fitted_detsrm.w_)):
            x_centered = x - x.mean(axis=1, keepdims=True)
            reconstruction = w @ fitted_detsrm.s_
            error = np.linalg.norm(x_centered - reconstruction, "fro")
            data_norm = np.linalg.norm(x_centered, "fro")
            assert error / data_norm < 0.5, (
                f"DetSRM Subject {i}: Poor reconstruction"
            )

    def test_detsrm_transform_properties(self, fitted_detsrm, multi_subject_data):
        """Test DetSRM transform output properties."""
        transformed = fitted_detsrm.transform(multi_subject_data["data"])

        for i, s in enumerate(transformed):
            expected_shape = (10, multi_subject_data["timepoints"])
            assert s.shape == expected_shape, (
                f"DetSRM Subject {i}: Wrong shape {s.shape}"
            )

    def test_srm_vs_detsrm_similar_results(self, fitted_srm, fitted_detsrm, multi_subject_data):
        """Test that SRM and DetSRM produce similar alignments.

        Uses pre-fitted fixtures - no additional fitting needed.
        """
        srm_transformed = fitted_srm.transform(multi_subject_data["data"])
        detsrm_transformed = fitted_detsrm.transform(multi_subject_data["data"])

        # Shared responses should be highly correlated
        for s1, s2 in zip(srm_transformed, detsrm_transformed):
            corr = np.corrcoef(s1.flatten(), s2.flatten())[0, 1]
            assert abs(corr) > 0.8, "SRM and DetSRM should produce similar alignments"

    def test_detsrm_deterministic_with_seed(self, multi_subject_data):
        """Test DetSRM reproducibility with same random seed.

        Note: Requires two fits to compare - cannot use fixture.
        """
        detsrm1 = DetSRM(features=10, n_iter=5, rand_seed=42)
        detsrm1.fit(multi_subject_data["data"])

        detsrm2 = DetSRM(features=10, n_iter=5, rand_seed=42)
        detsrm2.fit(multi_subject_data["data"])

        np.testing.assert_array_almost_equal(detsrm1.s_, detsrm2.s_, decimal=10)
        for w1, w2 in zip(detsrm1.w_, detsrm2.w_):
            np.testing.assert_array_almost_equal(w1, w2, decimal=10)

    def test_detsrm_transform_subject(self, fitted_detsrm, multi_subject_data):
        """Test DetSRM transform_subject() with new data."""
        np.random.seed(888)
        new_voxels = 175
        new_w = np.linalg.qr(np.random.randn(new_voxels, 10))[0]
        new_subject_data = new_w @ multi_subject_data[
            "shared"
        ] + 0.01 * np.random.randn(new_voxels, multi_subject_data["timepoints"])

        new_w_learned = fitted_detsrm.transform_subject(new_subject_data)

        # Check orthogonality
        gram = new_w_learned.T @ new_w_learned
        identity = np.eye(new_w_learned.shape[1])
        ortho_error = np.linalg.norm(gram - identity, "fro")
        assert ortho_error < 1e-5


# ========== CONTRACT TESTS FOR DETSRM ==========


class TestDetSRMContract:
    """Test DetSRM API contracts and error handling."""

    def test_detsrm_fit_before_transform_error(self, multi_subject_data):
        """Test that transform raises error before fit."""
        detsrm = DetSRM()
        with pytest.raises(NotFittedError, match="model fit has not been run"):
            detsrm.transform(multi_subject_data["data"])

    def test_detsrm_fit_before_transform_subject_error(self, multi_subject_data):
        """Test that transform_subject raises error before fit."""
        detsrm = DetSRM()
        with pytest.raises(NotFittedError, match="model fit has not been run"):
            detsrm.transform_subject(multi_subject_data["data"][0])

    def test_detsrm_single_subject_error(self, single_subject):
        """Test error with only 1 subject."""
        detsrm = DetSRM()
        with pytest.raises(ValueError, match="not enough subjects"):
            detsrm.fit(single_subject)

    def test_detsrm_fit_sets_attributes(self, multi_subject_data):
        """Test that DetSRM fit() creates required attributes."""
        detsrm = DetSRM(features=10, n_iter=2)
        detsrm.fit(multi_subject_data["data"])

        # Check fitted attributes exist
        assert hasattr(detsrm, "w_")
        assert hasattr(detsrm, "s_")

        # Check correct types and shapes
        assert isinstance(detsrm.w_, list)
        assert len(detsrm.w_) == len(multi_subject_data["data"])
        assert detsrm.s_.shape == (10, multi_subject_data["timepoints"])
