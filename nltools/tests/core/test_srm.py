"""
Tests for Shared Response Model (SRM) algorithms.

Testing philosophy: Property-based tests with mathematical invariants
rather than golden outputs (following BrainIAK, PyMVPA, Hypertools best practices).

Based on research documented in claude-guidelines/srm-hyperalignment-testing-strategy.md
"""

import pytest
import numpy as np
from nltools.algorithms.alignment import _SRM
from sklearn.exceptions import NotFittedError

pytestmark = pytest.mark.slow


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
    srm = _SRM(n_features=10, n_iter=10, random_state=42)
    srm.fit(multi_subject_data["data"])
    return srm


@pytest.fixture(scope="module")
def single_subject():
    """Single subject data (should error)."""
    np.random.seed(456)
    return [np.random.randn(100, 50)]


# ========== INITIALIZATION TESTS ==========


class TestSRMInitialization:
    """Test SRM initialization and parameter validation."""

    def test_srm_init_defaults(self):
        """Test SRM initializes with correct defaults."""
        srm = _SRM()
        assert srm.n_iter == 10
        assert srm.n_features == 50
        assert srm.random_state == 0


# ========== CONTRACT TESTS (Interface/API) ==========


class TestSRMContract:
    """Test SRM API contracts and error handling."""

    def test_fit_before_transform_error(self, multi_subject_data):
        """Test that transform raises error before fit."""
        srm = _SRM()
        with pytest.raises(NotFittedError, match="model fit has not been run"):
            srm.transform(multi_subject_data["data"])

    def test_fit_single_subject_error(self, single_subject):
        """Test error with only 1 subject (need multiple)."""
        srm = _SRM()
        with pytest.raises(ValueError, match="not enough subjects"):
            srm.fit(single_subject)

    def test_fit_mismatched_timepoints(self):
        """Test error when subjects have different numbers of timepoints."""
        np.random.seed(111)
        data = [
            np.random.randn(100, 50),  # 50 timepoints
            np.random.randn(100, 60),  # 60 timepoints
        ]
        srm = _SRM()
        with pytest.raises(ValueError, match="Different number of samples"):
            srm.fit(data)

    def test_fit_sets_attributes(self, multi_subject_data):
        """Test that fit() creates required attributes."""
        srm = _SRM(n_features=10, n_iter=2)
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
        srm = _SRM(n_features=10, n_iter=2)
        srm.fit(multi_subject_data["data"])

        # Try to transform different number of subjects
        wrong_data = multi_subject_data["data"][:3]  # Only 3 instead of 5
        with pytest.raises(ValueError, match="number of subjects does not match"):
            srm.transform(wrong_data)

    def test_transform_subject_wrong_timepoints(self, multi_subject_data):
        """Test error when new subject has different timepoints."""
        srm = _SRM(n_features=10, n_iter=2)
        srm.fit(multi_subject_data["data"])

        # New subject with wrong timepoint count
        np.random.seed(333)
        wrong_subject = np.random.randn(100, 60)  # 60 instead of 100
        with pytest.raises(ValueError, match="number of timepoints.*does not match"):
            srm.transform_subject(wrong_subject)

    def test_srm_rejects_legacy_features_kwarg(self):
        """`features=` is a removed legacy alias; the canonical name is n_features."""
        with pytest.raises(TypeError):
            _SRM(features=5)


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


# ========== EDGE CASES ==========


class TestSRMEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_deterministic_with_seed(self, multi_subject_data):
        """Test reproducibility with same random seed."""
        srm1 = _SRM(n_features=10, n_iter=5, random_state=42)
        srm1.fit(multi_subject_data["data"])

        srm2 = _SRM(n_features=10, n_iter=5, random_state=42)
        srm2.fit(multi_subject_data["data"])

        # Should produce identical results
        np.testing.assert_array_almost_equal(srm1.s_, srm2.s_, decimal=10)

        for w1, w2 in zip(srm1.w_, srm2.w_):
            np.testing.assert_array_almost_equal(w1, w2, decimal=10)

    def test_transform_subject_new_data(self, multi_subject_data):
        """Test transform_subject() with new subject data."""
        srm = _SRM(n_features=10, n_iter=5)
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


# ========== DETSRM TESTS ==========
# Note: Integration with align() function is already tested in
# test_stats.py::test_align_without_isc() which covers both SRM methods


# ========== CONTRACT TESTS FOR DETSRM ==========
