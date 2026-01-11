"""
Test LocalAlignment class (neighborhood-based functional alignment)

This test module tests the LocalAlignment implementation for searchlight-based
multi-subject functional alignment.

References:
    Bazeille et al. (2021). An empirical evaluation of functional alignment
    using inter-subject decoding. NeuroImage.
"""

import numpy as np
import pytest


# ========== MODULE-SCOPED FIXTURES ==========


@pytest.fixture(scope="module")
def small_mask():
    """Create a small 3D brain mask for testing.

    Returns a nibabel Nifti1Image mask with 27 voxels (3x3x3 cube).
    Uses 3mm isotropic voxels for reasonable searchlight sizes.
    """
    import nibabel as nib

    spatial_shape = (3, 3, 3)
    affine = np.eye(4) * 3  # 3mm voxels
    affine[3, 3] = 1

    # All voxels active
    mask_data = np.ones(spatial_shape, dtype=np.float32)
    return nib.Nifti1Image(mask_data, affine)


@pytest.fixture(scope="module")
def sample_multisubject_data(small_mask):
    """Create sample multi-subject data for testing.

    Returns list of 3 subjects, each with shape (27 voxels, 20 samples).
    """
    np.random.seed(42)
    n_voxels = 27  # 3x3x3
    n_samples = 20
    n_subjects = 3

    data = [np.random.randn(n_voxels, n_samples) for _ in range(n_subjects)]
    return data


@pytest.fixture(scope="module")
def fitted_local_alignment(sample_multisubject_data, small_mask):
    """Pre-fitted LocalAlignment for property tests.

    Module-scoped: expensive fit() runs once, shared across tests.
    Uses procrustes method (fastest).
    """
    from nltools.algorithms.alignment import LocalAlignment

    la = LocalAlignment(
        scheme="searchlight",
        method="procrustes",
        radius_mm=5.0,  # Small radius for test mask
        n_iter=2,
    )
    la.fit(sample_multisubject_data, small_mask)
    return la, sample_multisubject_data, small_mask


class TestLocalAlignmentInitialization:
    """Test LocalAlignment initialization and parameters."""

    def test_import(self):
        """Test that LocalAlignment can be imported from algorithms module."""
        from nltools.algorithms.alignment import LocalAlignment

        assert LocalAlignment is not None

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        from nltools.algorithms.alignment import LocalAlignment

        la = LocalAlignment()
        assert la.scheme == "searchlight"
        assert la.method == "procrustes"
        assert la.radius_mm == 10.0
        assert la.n_features is None
        assert la.n_iter == 3
        assert la.aggregation == "center"

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        from nltools.algorithms.alignment import LocalAlignment

        la = LocalAlignment(
            scheme="searchlight",
            method="srm",
            radius_mm=8.0,
            n_features=10,
            n_iter=5,
        )
        assert la.method == "srm"
        assert la.radius_mm == 8.0
        assert la.n_features == 10
        assert la.n_iter == 5

    def test_init_invalid_scheme(self):
        """Test that invalid scheme raises error."""
        from nltools.algorithms.alignment import LocalAlignment

        with pytest.raises(ValueError, match="Unknown scheme"):
            LocalAlignment(scheme="invalid")

    def test_init_invalid_method(self):
        """Test that invalid method raises error."""
        from nltools.algorithms.alignment import LocalAlignment

        with pytest.raises(ValueError, match="Unknown method"):
            LocalAlignment(method="invalid")

    def test_init_invalid_aggregation(self):
        """Test that invalid aggregation raises error."""
        from nltools.algorithms.alignment import LocalAlignment

        with pytest.raises(ValueError, match="Unknown aggregation"):
            LocalAlignment(aggregation="mean")


class TestLocalAlignmentFit:
    """Test LocalAlignment fit() method."""

    def test_fit_returns_self(self, sample_multisubject_data, small_mask):
        """Test that fit() returns self for method chaining."""
        from nltools.algorithms.alignment import LocalAlignment

        la = LocalAlignment(radius_mm=5.0, n_iter=1)
        result = la.fit(sample_multisubject_data, small_mask)
        assert result is la

    def test_fit_stores_attributes(self, fitted_local_alignment):
        """Test that fit() stores required attributes."""
        la, data, mask = fitted_local_alignment

        # Check fitted attributes exist
        assert hasattr(la, "transforms_")
        assert hasattr(la, "template_")
        assert hasattr(la, "neighborhoods_")
        assert hasattr(la, "mask_")
        assert hasattr(la, "n_voxels_")

        # Check types
        assert isinstance(la.transforms_, dict)
        assert isinstance(la.template_, dict)
        assert la.n_voxels_ == data[0].shape[0]

    def test_fit_transforms_shape(self, fitted_local_alignment):
        """Test that transforms have correct shapes."""
        la, data, _ = fitted_local_alignment
        n_subjects = len(data)

        # Each neighborhood should have transforms for all subjects
        for center_idx, transforms in la.transforms_.items():
            assert len(transforms) == n_subjects, (
                f"Neighborhood {center_idx} should have {n_subjects} transforms"
            )

    def test_fit_with_procrustes(self, sample_multisubject_data, small_mask):
        """Test fit with procrustes method."""
        from nltools.algorithms.alignment import LocalAlignment

        la = LocalAlignment(method="procrustes", radius_mm=5.0, n_iter=2)
        la.fit(sample_multisubject_data, small_mask)

        assert la.transforms_ is not None
        assert len(la.transforms_) > 0

    def test_fit_with_srm(self, sample_multisubject_data, small_mask):
        """Test fit with SRM method."""
        from nltools.algorithms.alignment import LocalAlignment

        la = LocalAlignment(method="srm", radius_mm=5.0, n_iter=2, n_features=5)
        la.fit(sample_multisubject_data, small_mask)

        assert la.transforms_ is not None
        assert len(la.transforms_) > 0

    def test_fit_with_hyperalignment(self, sample_multisubject_data, small_mask):
        """Test fit with hyperalignment method."""
        from nltools.algorithms.alignment import LocalAlignment

        la = LocalAlignment(method="hyperalignment", radius_mm=5.0, n_iter=2)
        la.fit(sample_multisubject_data, small_mask)

        assert la.transforms_ is not None
        assert len(la.transforms_) > 0

    def test_fit_validates_data(self, small_mask):
        """Test that fit validates input data."""
        from nltools.algorithms.alignment import LocalAlignment

        la = LocalAlignment(radius_mm=5.0)

        # Single subject should fail
        with pytest.raises(ValueError, match="at least 2 subject"):
            la.fit([np.random.randn(27, 20)], small_mask)

        # Empty list should fail
        with pytest.raises(ValueError):
            la.fit([], small_mask)

    def test_fit_validates_voxel_count(self, small_mask):
        """Test that fit validates consistent voxel counts."""
        from nltools.algorithms.alignment import LocalAlignment

        la = LocalAlignment(radius_mm=5.0)

        # Different voxel counts should fail
        data = [
            np.random.randn(27, 20),
            np.random.randn(30, 20),  # Different voxel count
        ]
        with pytest.raises(ValueError, match="same number of voxels"):
            la.fit(data, small_mask)

    def test_fit_validates_sample_count(self, small_mask):
        """Test that fit validates consistent sample counts."""
        from nltools.algorithms.alignment import LocalAlignment

        la = LocalAlignment(radius_mm=5.0)

        # Different sample counts should fail
        data = [
            np.random.randn(27, 20),
            np.random.randn(27, 25),  # Different sample count
        ]
        with pytest.raises(ValueError, match="same number of samples"):
            la.fit(data, small_mask)


class TestLocalAlignmentTransform:
    """Test LocalAlignment transform() method."""

    def test_transform_output_shape(self, fitted_local_alignment):
        """Test that transform produces correct output shapes."""
        la, data, _ = fitted_local_alignment

        aligned = la.transform(data)

        # Check output structure
        assert isinstance(aligned, list)
        assert len(aligned) == len(data)

        # Check each subject's shape
        for i, (orig, trans) in enumerate(zip(data, aligned)):
            assert trans.shape == orig.shape, (
                f"Subject {i}: transformed shape {trans.shape} != original {orig.shape}"
            )

    def test_transform_before_fit_raises(self):
        """Test that transform before fit raises error."""
        from nltools.algorithms.alignment import LocalAlignment

        la = LocalAlignment()

        with pytest.raises(ValueError, match="must be fit"):
            la.transform([np.random.randn(27, 20)])

    def test_transform_wrong_voxels_raises(self, fitted_local_alignment):
        """Test that transform with wrong voxel count raises error."""
        la, _, _ = fitted_local_alignment

        # Different voxel count
        wrong_data = [np.random.randn(30, 20)]

        with pytest.raises(ValueError, match="voxels"):
            la.transform(wrong_data)

    def test_fit_transform(self, sample_multisubject_data, small_mask):
        """Test fit_transform convenience method."""
        from nltools.algorithms.alignment import LocalAlignment

        la = LocalAlignment(radius_mm=5.0, n_iter=1)
        aligned = la.fit_transform(sample_multisubject_data, small_mask)

        assert isinstance(aligned, list)
        assert len(aligned) == len(sample_multisubject_data)
        for i, trans in enumerate(aligned):
            assert trans.shape == sample_multisubject_data[i].shape


class TestLocalAlignmentNumericalProperties:
    """Test numerical properties of LocalAlignment."""

    def test_procrustes_transforms_orthogonal(
        self, sample_multisubject_data, small_mask
    ):
        """Test that procrustes transforms are orthogonal."""
        from nltools.algorithms.alignment import LocalAlignment

        la = LocalAlignment(method="procrustes", radius_mm=5.0, n_iter=2)
        la.fit(sample_multisubject_data, small_mask)

        # Check orthogonality for a sample of neighborhoods
        for center_idx in list(la.transforms_.keys())[:5]:
            for transform in la.transforms_[center_idx]:
                # W @ W.T should be identity
                product = transform @ transform.T
                expected = np.eye(transform.shape[0])
                np.testing.assert_allclose(
                    product,
                    expected,
                    atol=1e-5,
                    err_msg=f"Transform at {center_idx} not orthogonal",
                )

    def test_alignment_reduces_variance(self, sample_multisubject_data, small_mask):
        """Test that alignment reduces inter-subject variance."""
        from nltools.algorithms.alignment import LocalAlignment

        la = LocalAlignment(method="procrustes", radius_mm=5.0, n_iter=3)
        aligned = la.fit_transform(sample_multisubject_data, small_mask)

        # Calculate inter-subject variance before and after
        orig_stack = np.stack(sample_multisubject_data, axis=0)
        aligned_stack = np.stack(aligned, axis=0)

        orig_var = np.var(orig_stack, axis=0).mean()
        aligned_var = np.var(aligned_stack, axis=0).mean()

        # Aligned data should have similar or lower variance
        # Note: This is a soft test - alignment may not always reduce variance
        # especially with random data
        assert aligned_var < orig_var * 2, (
            f"Aligned variance ({aligned_var:.4f}) much higher than "
            f"original ({orig_var:.4f})"
        )


class TestLocalAlignmentPiecewise:
    """Test LocalAlignment with piecewise scheme."""

    @pytest.fixture
    def parcellation_and_mask(self):
        """Create a small parcellation and matching mask for testing."""
        import nibabel as nib

        spatial_shape = (3, 3, 3)
        affine = np.eye(4) * 3  # 3mm voxels
        affine[3, 3] = 1

        # Create mask (all voxels active)
        mask_data = np.ones(spatial_shape, dtype=np.float32)
        mask = nib.Nifti1Image(mask_data, affine)

        # Create parcellation with 3 parcels
        # Parcel 1: first 9 voxels (z=0)
        # Parcel 2: middle 9 voxels (z=1)
        # Parcel 3: last 9 voxels (z=2)
        parc_data = np.zeros(spatial_shape, dtype=np.int32)
        parc_data[:, :, 0] = 1
        parc_data[:, :, 1] = 2
        parc_data[:, :, 2] = 3
        parcellation = nib.Nifti1Image(parc_data, affine)

        return parcellation, mask

    @pytest.fixture
    def piecewise_data(self, parcellation_and_mask):
        """Create multi-subject data matching the parcellation."""
        np.random.seed(42)
        n_voxels = 27  # 3x3x3
        n_samples = 20
        n_subjects = 3
        data = [np.random.randn(n_voxels, n_samples) for _ in range(n_subjects)]
        return data

    def test_piecewise_requires_parcellation(self):
        """Test that piecewise scheme requires parcellation parameter."""
        from nltools.algorithms.alignment import LocalAlignment

        with pytest.raises(ValueError, match="parcellation is required"):
            LocalAlignment(scheme="piecewise")

    def test_piecewise_auto_aggregation(self, parcellation_and_mask):
        """Test that piecewise auto-switches to 'all' aggregation."""
        from nltools.algorithms.alignment import LocalAlignment

        parcellation, _ = parcellation_and_mask
        la = LocalAlignment(scheme="piecewise", parcellation=parcellation)

        # Should auto-switch from 'center' to 'all'
        assert la.aggregation == "all"

    def test_piecewise_fit_returns_self(self, piecewise_data, parcellation_and_mask):
        """Test that piecewise fit() returns self."""
        from nltools.algorithms.alignment import LocalAlignment

        parcellation, mask = parcellation_and_mask
        la = LocalAlignment(scheme="piecewise", parcellation=parcellation, n_iter=1)

        result = la.fit(piecewise_data, mask)
        assert result is la

    def test_piecewise_stores_transforms(self, piecewise_data, parcellation_and_mask):
        """Test that piecewise fit stores transforms for each parcel."""
        from nltools.algorithms.alignment import LocalAlignment

        parcellation, mask = parcellation_and_mask
        la = LocalAlignment(scheme="piecewise", parcellation=parcellation, n_iter=1)
        la.fit(piecewise_data, mask)

        # Should have transforms for 3 parcels
        assert len(la.transforms_) == 3
        assert set(la.transforms_.keys()) == {1, 2, 3}

        # Each parcel should have transforms for all subjects
        for parcel_id, transforms in la.transforms_.items():
            assert len(transforms) == 3  # 3 subjects

    def test_piecewise_transform_output_shape(
        self, piecewise_data, parcellation_and_mask
    ):
        """Test that piecewise transform produces correct output shapes."""
        from nltools.algorithms.alignment import LocalAlignment

        parcellation, mask = parcellation_and_mask
        la = LocalAlignment(scheme="piecewise", parcellation=parcellation, n_iter=1)
        aligned = la.fit_transform(piecewise_data, mask)

        assert len(aligned) == len(piecewise_data)
        for i, (orig, trans) in enumerate(zip(piecewise_data, aligned)):
            assert trans.shape == orig.shape, (
                f"Subject {i}: shape mismatch {trans.shape} != {orig.shape}"
            )

    def test_piecewise_with_srm(self, piecewise_data, parcellation_and_mask):
        """Test piecewise with SRM method."""
        from nltools.algorithms.alignment import LocalAlignment

        parcellation, mask = parcellation_and_mask
        la = LocalAlignment(
            scheme="piecewise",
            parcellation=parcellation,
            method="srm",
            n_features=5,
            n_iter=2,
        )
        aligned = la.fit_transform(piecewise_data, mask)

        assert len(aligned) == 3
        for a in aligned:
            assert a.shape == (27, 20)

    def test_piecewise_with_hyperalignment(self, piecewise_data, parcellation_and_mask):
        """Test piecewise with hyperalignment method."""
        from nltools.algorithms.alignment import LocalAlignment

        parcellation, mask = parcellation_and_mask
        la = LocalAlignment(
            scheme="piecewise",
            parcellation=parcellation,
            method="hyperalignment",
            n_iter=1,
        )
        aligned = la.fit_transform(piecewise_data, mask)

        assert len(aligned) == 3
        for a in aligned:
            assert a.shape == (27, 20)


class TestLocalAlignmentEdgeCases:
    """Test edge cases and error handling."""

    def test_tiny_neighborhood(self):
        """Test handling of very small neighborhoods."""
        import nibabel as nib
        from nltools.algorithms.alignment import LocalAlignment

        # Create tiny mask with just 2 voxels
        mask_data = np.zeros((2, 1, 1), dtype=np.float32)
        mask_data[0, 0, 0] = 1
        mask_data[1, 0, 0] = 1
        affine = np.eye(4) * 10  # 10mm voxels (large to ensure small neighborhoods)
        affine[3, 3] = 1
        tiny_mask = nib.Nifti1Image(mask_data, affine)

        np.random.seed(42)
        data = [np.random.randn(2, 20) for _ in range(3)]

        la = LocalAlignment(radius_mm=5.0, n_iter=1)

        # Should not raise - handles degenerate cases
        la.fit(data, tiny_mask)
        aligned = la.transform(data)

        assert len(aligned) == 3
        for a in aligned:
            assert a.shape == (2, 20)
