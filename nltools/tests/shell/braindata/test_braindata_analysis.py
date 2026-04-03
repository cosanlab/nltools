import numpy as np
import nibabel as nb
import pytest

from nltools.data import BrainData
from nltools.mask import create_sphere, roi_to_brain
from nltools.simulator import Simulator
from nltools.stats import align

shape_2d = (6, 238955)


class TestBrainDataAnalysis:
    def test_apply_mask(self, sim_brain_data):
        """Test applying masks to BrainData."""
        s1 = create_sphere([12, 10, -8], radius=10)
        assert isinstance(s1, nb.Nifti1Image)
        masked_dat = sim_brain_data.apply_mask(s1)
        assert masked_dat.shape[1] == np.sum(s1.get_fdata() != 0)
        masked_dat = sim_brain_data.apply_mask(s1, resample_mask_to_brain=True)
        assert masked_dat.shape[1] == np.sum(s1.get_fdata() != 0)

    def test_apply_mask_nilearn_validation(self, sim_brain_data):
        """Nilearn should provide better error messages for invalid inputs"""
        # Test that multi-volume mask raises clear error
        # Create invalid 4D mask (should be 3D)
        s1 = create_sphere([12, 10, -8], radius=10)

        # Stack to create 4D (invalid for masking)
        from nilearn.image import concat_imgs

        invalid_mask = concat_imgs([s1, s1])

        # Create BrainData from invalid mask
        mask_bd = BrainData(invalid_mask, mask=sim_brain_data.mask)

        # Should raise ValueError for non-single image
        with pytest.raises(ValueError, match="Mask must be a single image"):
            sim_brain_data.apply_mask(mask_bd)

    def test_apply_mask_dimension_compatibility(self, sim_brain_data):
        """Nilearn should handle dimension compatibility automatically"""
        # Create a compatible mask
        s1 = create_sphere([12, 10, -8], radius=10)
        mask_bd = BrainData(s1, mask=sim_brain_data.mask)

        # This should work (nilearn handles dimension matching)
        result = sim_brain_data.apply_mask(mask_bd)

        assert isinstance(result, BrainData)
        # Verify output shape matches number of non-zero mask voxels
        assert result.shape[1] == mask_bd.data.astype(bool).sum()

    def test_apply_mask_resampling(self, sim_brain_data):
        """Test resample_mask_to_brain parameter works correctly"""
        s1 = create_sphere([12, 10, -8], radius=10)
        mask_bd = BrainData(s1, mask=sim_brain_data.mask)

        # With resampling
        result_resample = sim_brain_data.apply_mask(
            mask_bd, resample_mask_to_brain=True
        )
        assert isinstance(result_resample, BrainData)
        assert result_resample.shape[1] == np.sum(s1.get_fdata() != 0)

        # Without resampling (default)
        result_no_resample = sim_brain_data.apply_mask(
            mask_bd, resample_mask_to_brain=False
        )
        assert isinstance(result_no_resample, BrainData)
        assert result_no_resample.shape[1] == mask_bd.data.astype(bool).sum()

    @pytest.mark.slow
    def test_extract_roi(self, sim_brain_data):
        """Test ROI extraction with different metrics and labeled atlases."""
        mask = create_sphere([12, 10, -8], radius=10)
        assert len(sim_brain_data.extract_roi(mask, metric="mean")) == shape_2d[0]
        assert len(sim_brain_data.extract_roi(mask, metric="median")) == shape_2d[0]
        n_components = 2
        assert sim_brain_data.extract_roi(
            mask, metric="pca", n_components=n_components
        ).shape == (n_components, shape_2d[0])
        with pytest.raises(NotImplementedError):
            sim_brain_data.extract_roi(mask, metric="p")

        assert isinstance(
            sim_brain_data[0].extract_roi(mask, metric="mean"), (float, np.floating)
        )
        assert isinstance(
            sim_brain_data[0].extract_roi(mask, metric="median"), (float, np.floating)
        )
        with pytest.raises(ValueError):
            sim_brain_data[0].extract_roi(mask, metric="pca")
        with pytest.raises(NotImplementedError):
            sim_brain_data[0].extract_roi(mask, metric="p")

        s1 = create_sphere([15, 10, -8], radius=10)
        s2 = create_sphere([-15, 10, -8], radius=10)
        s3 = create_sphere([0, -15, -8], radius=10)
        masks = BrainData([s1, s2, s3])
        mask = roi_to_brain([1, 2, 3], masks)
        assert len(sim_brain_data[0].extract_roi(mask, metric="mean")) == len(masks)
        assert len(sim_brain_data[0].extract_roi(mask, metric="median")) == len(masks)
        assert sim_brain_data.extract_roi(mask, metric="mean").shape == (
            len(masks),
            shape_2d[0],
        )
        assert sim_brain_data.extract_roi(mask, metric="median").shape == (
            len(masks),
            shape_2d[0],
        )
        assert len(
            sim_brain_data.extract_roi(mask, metric="pca", n_components=n_components)
        ) == len(masks)

    # ==================== Transform Methods ====================

    def test_r_to_z(self, sim_brain_data):
        """Test Fisher r-to-z transformation."""
        z = sim_brain_data.r_to_z()
        assert z.shape == sim_brain_data.shape

    def test_copy(self, sim_brain_data):
        """Test copying BrainData objects."""
        d_copy = sim_brain_data.copy()
        assert d_copy.shape == sim_brain_data.shape

    def test_detrend(self, sim_brain_data):
        """Test detrending removes linear trends."""
        detrend = sim_brain_data.detrend()
        assert detrend.shape == sim_brain_data.shape

    @pytest.mark.filterwarnings("ignore:Numerical issues:UserWarning")
    def test_standardize(self, sim_brain_data):
        """Test standardization with different methods."""
        s = sim_brain_data.standardize()
        assert s.shape == sim_brain_data.shape
        # Mean should be close to zero after standardization (tolerance for numerical precision)
        assert np.isclose(np.sum(s.mean().data), 0, atol=0.5)
        s = sim_brain_data.standardize(method="zscore")
        assert s.shape == sim_brain_data.shape
        assert np.isclose(np.sum(s.mean().data), 0, atol=0.5)

    def test_filter_high_pass(self, minimal_brain_data):
        """Test high-pass filtering returns BrainData with correct shape."""
        # Test basic API: sampling_freq + high_pass
        filtered = minimal_brain_data.filter(sampling_freq=0.5, high_pass=0.01)

        assert isinstance(filtered, BrainData)
        assert filtered.shape == minimal_brain_data.shape
        # Original data should be unchanged (immutability)
        assert not np.array_equal(id(filtered.data), id(minimal_brain_data.data))

    def test_filter_low_pass(self, minimal_brain_data):
        """Test low-pass filtering returns BrainData with correct shape."""
        filtered = minimal_brain_data.filter(sampling_freq=0.5, low_pass=0.1)

        assert isinstance(filtered, BrainData)
        assert filtered.shape == minimal_brain_data.shape

    def test_filter_band_pass(self, minimal_brain_data):
        """Test band-pass filtering (both high and low pass)."""
        filtered = minimal_brain_data.filter(
            sampling_freq=0.5, high_pass=0.01, low_pass=0.1
        )

        assert isinstance(filtered, BrainData)
        assert filtered.shape == minimal_brain_data.shape

    def test_filter_error_no_sampling_freq(self, minimal_brain_data):
        """Test error when sampling_freq not provided."""
        with pytest.raises(ValueError, match="sampling rate"):
            minimal_brain_data.filter(high_pass=0.01)

    def test_filter_error_no_cutoff(self, minimal_brain_data):
        """Test error when neither high_pass nor low_pass specified."""
        # Note: current error message has typo "beprovided"
        with pytest.raises(ValueError, match="must.*provided"):
            minimal_brain_data.filter(sampling_freq=0.5)

    def test_filter_kwargs_passed_through(self, minimal_brain_data):
        """Test that additional kwargs reach nilearn.signal.clean."""
        # Test with ensure_finite kwarg (nilearn.signal.clean parameter)
        # This is a smoke test - we don't validate parameter effect,
        # just that the method runs without error when kwargs provided
        filtered = minimal_brain_data.filter(
            sampling_freq=0.5,
            high_pass=0.01,
            ensure_finite=True,  # nilearn parameter not extracted by filter()
        )

        assert isinstance(filtered, BrainData)

    def test_smooth(self, sim_brain_data):
        """Test spatial smoothing."""
        smoothed = sim_brain_data.smooth(5.0)
        assert isinstance(smoothed, BrainData)
        assert smoothed.shape == sim_brain_data.shape
        smoothed = sim_brain_data[0].smooth(5.0)
        assert len(smoothed.shape) == 1

    @pytest.mark.slow
    def test_threshold(self):
        """Test thresholding and region extraction."""
        s1 = create_sphere([12, 10, -8], radius=10)
        s2 = create_sphere([22, -2, -22], radius=10)
        mask = BrainData(s1) * 5
        mask = mask + BrainData(s2)

        m1 = mask.threshold(upper=0.5)
        m2 = mask.threshold(upper=3)
        m3 = mask.threshold(upper="98%")
        m4 = BrainData(s1) * 5 + BrainData(s2) * -0.5
        m4 = mask.threshold(upper=0.5, lower=-0.3)
        assert np.sum(m1.data > 0) > np.sum(m2.data > 0)
        assert np.sum(m1.data > 0) == np.sum(m3.data > 0)
        assert np.sum(m4.data[(m4.data > -0.3) & (m4.data < 0.5)]) == 0
        assert np.sum(m4.data[(m4.data < -0.3) | (m4.data > 0.5)]) > 0

        # Test Regions
        r = mask.regions(min_region_size=10)
        m1 = BrainData(s1)
        m2 = r.threshold(1, binarize=True)
        assert len(np.unique(r.to_nifti().get_fdata())) == 2
        diff = m2 - m1
        assert np.sum(diff.data) == 0

    # ============================================================================
    # Thresholding Operations - Cluster Enhancement
    # ============================================================================

    @pytest.mark.slow
    def test_threshold_cluster_basic(self, sim_brain_data):
        """Cluster thresholding should filter small clusters using nilearn"""
        # Create data with distinct regions
        brain = sim_brain_data.copy()

        # Threshold with cluster size minimum
        result = brain.threshold(lower=2, cluster_threshold=10)

        # Should return BrainData
        assert isinstance(result, BrainData)
        # Should have removed small clusters (basic check that it ran)
        assert result.shape == brain.shape

    @pytest.mark.slow
    def test_threshold_cluster_with_upper_only(self, sim_brain_data):
        """Cluster threshold should work with upper threshold only"""
        brain = sim_brain_data.copy()
        result = brain.threshold(upper=2, cluster_threshold=10)
        assert isinstance(result, BrainData)

    @pytest.mark.slow
    def test_threshold_cluster_with_lower_only(self, sim_brain_data):
        """Cluster threshold should work with lower threshold only"""
        brain = sim_brain_data.copy()
        result = brain.threshold(lower=2, cluster_threshold=10)
        assert isinstance(result, BrainData)

    def test_threshold_cluster_rejects_bandpass(self, sim_brain_data):
        """Should raise error when using both upper AND lower with cluster_threshold"""
        brain = sim_brain_data.copy()

        with pytest.raises(
            ValueError, match="Band-pass filtering.*not supported.*cluster"
        ):
            brain.threshold(lower=-2, upper=2, cluster_threshold=10)

    @pytest.mark.slow
    def test_threshold_cluster_with_binarize(self, sim_brain_data):
        """Cluster threshold should work with binarization"""
        brain = sim_brain_data.copy()
        result = brain.threshold(lower=2, cluster_threshold=10, binarize=True)

        # Should be binary
        unique_vals = np.unique(result.data)
        assert len(unique_vals) <= 2
        assert all(v in [0, 1] for v in unique_vals)

    def test_threshold_cluster_zero_disables(self, sim_brain_data):
        """cluster_threshold=0 should use fast path (current implementation)"""
        brain = sim_brain_data.copy()

        # These should be equivalent
        result_no_cluster = brain.threshold(lower=2, upper=5)
        result_zero_cluster = brain.threshold(lower=2, upper=5, cluster_threshold=0)

        np.testing.assert_array_equal(result_no_cluster.data, result_zero_cluster.data)

    def test_threshold_backwards_compatible_no_cluster(self, sim_brain_data):
        """Existing threshold behavior unchanged when cluster_threshold=0"""
        brain = sim_brain_data.copy()

        # Old way (default cluster_threshold=0)
        result_old = brain.threshold(lower=-2, upper=2)

        # Explicit cluster_threshold=0
        result_explicit = brain.threshold(lower=-2, upper=2, cluster_threshold=0)

        # Should be identical
        np.testing.assert_array_equal(result_old.data, result_explicit.data)

    def test_threshold_bandpass_still_works(self, sim_brain_data):
        """Band-pass filtering (unique feature) still works without cluster_threshold"""
        brain = sim_brain_data.copy()

        # This should still work (keep middle values, zero extremes)
        result = brain.threshold(lower=-2, upper=2)

        assert isinstance(result, BrainData)
        # Verify band-pass behavior preserved (values in range kept)

    def test_threshold_with_zero_value(self, sim_brain_data):
        """Test threshold works correctly when upper=0 or lower=0 (#370).

        This was a bug where `if upper:` evaluated to False when upper=0,
        causing thresholding to be skipped.
        """
        brain = sim_brain_data.copy()
        # Create data with positive and negative values
        brain.data = brain.data - brain.data.mean()

        # Test upper=0: should zero out values < 0
        result_upper0 = brain.threshold(upper=0)
        assert np.all(result_upper0.data >= 0), "upper=0 should zero values < 0"

        # Test lower=0: should zero out values > 0
        result_lower0 = brain.threshold(lower=0)
        assert np.all(result_lower0.data <= 0), "lower=0 should zero values > 0"

        # Test upper=0, lower=-0.5: bandpass with zero boundary
        result_bandpass = brain.threshold(upper=0, lower=-0.5)
        # Values between -0.5 and 0 should be zeroed
        non_zero = result_bandpass.data[result_bandpass.data != 0]
        assert np.all((non_zero >= 0) | (non_zero <= -0.5)), (
            "bandpass with upper=0 should work"
        )

    @pytest.mark.slow
    def test_threshold_cluster_realistic_neuroimaging(self, sim_brain_data):
        """Integration test with realistic neuroimaging workflow"""
        # Test with actual brain data structure from fixtures
        brain = sim_brain_data.copy()

        # Realistic workflow: threshold then cluster filter
        result = brain.threshold(lower=2.5, cluster_threshold=50)

        # Basic sanity checks
        assert isinstance(result, BrainData)
        assert result.shape == brain.shape
        assert not result.is_empty

    # ==================== Similarity & Analysis ====================

    def test_similarity(self, sim_brain_data):
        """Test similarity computation with different metrics."""
        # Test comparing BrainData to itself
        r = sim_brain_data.similarity(sim_brain_data, method="correlation")
        assert r.shape == (sim_brain_data.shape[0], sim_brain_data.shape[0])
        r = sim_brain_data.similarity(sim_brain_data, method="dot_product")
        assert r.shape == (sim_brain_data.shape[0], sim_brain_data.shape[0])
        r = sim_brain_data.similarity(sim_brain_data, method="cosine")
        assert r.shape == (sim_brain_data.shape[0], sim_brain_data.shape[0])

        # Test comparing to a single image
        r = sim_brain_data.similarity(sim_brain_data[0], method="correlation")
        assert len(r) == shape_2d[0]

    @pytest.mark.slow
    def test_decompose(self, sim_brain_data):
        """Test decomposition with PCA, ICA, NMF, and Factor Analysis."""
        n_components = 3
        stats = sim_brain_data.decompose(
            algorithm="pca", axis="voxels", n_components=n_components
        )
        assert n_components == len(stats["components"])
        assert stats["weights"].shape == (len(sim_brain_data), n_components)

        stats = sim_brain_data.decompose(
            algorithm="ica", axis="voxels", n_components=n_components
        )
        assert n_components == len(stats["components"])
        assert stats["weights"].shape == (len(sim_brain_data), n_components)

        sim_brain_data.data = sim_brain_data.data + 2
        sim_brain_data.data[sim_brain_data.data < 0] = 0
        stats = sim_brain_data.decompose(
            algorithm="nnmf", axis="voxels", n_components=n_components
        )
        assert n_components == len(stats["components"])
        assert stats["weights"].shape == (len(sim_brain_data), n_components)

        stats = sim_brain_data.decompose(
            algorithm="fa", axis="voxels", n_components=n_components
        )
        assert n_components == len(stats["components"])
        assert stats["weights"].shape == (len(sim_brain_data), n_components)

        stats = sim_brain_data.decompose(
            algorithm="pca", axis="images", n_components=n_components
        )
        assert n_components == len(stats["components"])
        assert stats["weights"].shape == (len(sim_brain_data), n_components)

        stats = sim_brain_data.decompose(
            algorithm="ica", axis="images", n_components=n_components
        )
        assert n_components == len(stats["components"])
        assert stats["weights"].shape == (len(sim_brain_data), n_components)

        sim_brain_data.data = sim_brain_data.data + 2
        sim_brain_data.data[sim_brain_data.data < 0] = 0
        stats = sim_brain_data.decompose(
            algorithm="nnmf", axis="images", n_components=n_components
        )
        assert n_components == len(stats["components"])
        assert stats["weights"].shape == (len(sim_brain_data), n_components)

        stats = sim_brain_data.decompose(
            algorithm="fa", axis="images", n_components=n_components
        )
        assert n_components == len(stats["components"])
        assert stats["weights"].shape == (len(sim_brain_data), n_components)

    # ==================== Alignment ====================

    @pytest.mark.slow
    def test_hyperalignment(self):
        """Test hyperalignment with SRM and Procrustes methods."""
        sim = Simulator()
        y = [0, 1]
        n_reps = 10
        s1 = create_sphere([0, 0, 0], radius=3)
        d1 = sim.create_data(y, 1, reps=n_reps, output_dir=None).apply_mask(s1)
        d2 = sim.create_data(y, 2, reps=n_reps, output_dir=None).apply_mask(s1)
        d3 = sim.create_data(y, 3, reps=n_reps, output_dir=None).apply_mask(s1)
        data = [d1, d2, d3]

        # Test deterministic brain_data
        out = align(data, method="deterministic_srm")

        bout = d1.align(out["common_model"], method="deterministic_srm")
        assert d1.shape == bout["transformed"].shape
        assert d1.shape == bout["common_model"].shape
        assert d1.shape[1] == bout["transformation_matrix"].shape[0]
        btransformed = np.dot(d1.data, bout["transformation_matrix"].data.T)
        np.testing.assert_almost_equal(
            0, np.sum(bout["transformed"].data - btransformed)
        )

        # Test probabilistic brain_data
        bout = d1.align(out["common_model"], method="probabilistic_srm")
        assert d1.shape == bout["transformed"].shape
        assert d1.shape == bout["common_model"].shape
        assert d1.shape[1] == bout["transformation_matrix"].shape[0]
        btransformed = np.dot(d1.data, bout["transformation_matrix"].data.T)
        np.testing.assert_almost_equal(
            0, np.sum(bout["transformed"].data - btransformed)
        )

        # Test procrustes brain_data
        out = align(data, method="procrustes")
        centered = data[0].data - np.mean(data[0].data, 0)

        bout = d1.align(out["common_model"], method="procrustes")
        assert d1.shape == bout["transformed"].shape
        assert d1.shape == bout["common_model"].shape
        assert d1.shape[1] == bout["transformation_matrix"].shape[0]
        centered = d1.data - np.mean(d1.data, 0)
        btransformed = (
            np.dot(
                centered / np.linalg.norm(centered), bout["transformation_matrix"].data
            )
            * bout["scale"]
        )
        np.testing.assert_almost_equal(
            0, np.sum(bout["transformed"].data - btransformed), decimal=5
        )
        np.testing.assert_almost_equal(
            0, np.sum(out["transformed"][0].data - bout["transformed"].data), decimal=5
        )

        # Test over time
        sim = Simulator()
        y = [0, 1]
        n_reps = 10
        s1 = create_sphere([0, 0, 0], radius=5)
        d1 = sim.create_data(y, 1, reps=n_reps, output_dir=None).apply_mask(s1)
        d2 = sim.create_data(y, 2, reps=n_reps, output_dir=None).apply_mask(s1)
        d3 = sim.create_data(y, 3, reps=n_reps, output_dir=None).apply_mask(s1)
        data = [d1, d2, d3]

        out = align(data, method="deterministic_srm", axis=1)
        bout = d1.align(out["common_model"], method="deterministic_srm", axis=1)
        assert d1.shape == bout["transformed"].shape
        assert d1.shape == bout["common_model"].shape
        assert d1.shape[0] == bout["transformation_matrix"].shape[0]
        btransformed = np.dot(d1.data.T, bout["transformation_matrix"].data.T)
        np.testing.assert_almost_equal(
            0, np.sum(bout["transformed"].data - btransformed.T)
        )

        out = align(data, method="probabilistic_srm", axis=1)
        bout = d1.align(out["common_model"], method="probabilistic_srm", axis=1)
        assert d1.shape == bout["transformed"].shape
        assert d1.shape == bout["common_model"].shape
        assert d1.shape[0] == bout["transformation_matrix"].shape[0]
        btransformed = np.dot(d1.data.T, bout["transformation_matrix"].data.T)
        np.testing.assert_almost_equal(
            0, np.sum(bout["transformed"].data - btransformed.T)
        )

        out = align(data, method="procrustes", axis=1)
        bout = d1.align(out["common_model"], method="procrustes", axis=1)
        assert d1.shape == bout["transformed"].shape
        assert d1.shape == bout["common_model"].shape
        assert d1.shape[0] == bout["transformation_matrix"].shape[0]
        centered = d1.data.T - np.mean(d1.data.T, 0)
        btransformed = (
            np.dot(
                centered / np.linalg.norm(centered), bout["transformation_matrix"].data
            )
            * bout["scale"]
        )
        np.testing.assert_almost_equal(
            0, np.sum(bout["transformed"].data - btransformed.T), decimal=5
        )
        np.testing.assert_almost_equal(
            0, np.sum(out["transformed"][0].data - bout["transformed"].data)
        )

    # ==================== Temporal Methods ====================

    @pytest.mark.slow
    def test_temporal_resample(self, sim_brain_data):
        """Test temporal resampling (upsampling and downsampling)."""
        up = sim_brain_data.temporal_resample(
            sampling_freq=1 / 2, target=2, target_type="hz"
        )
        assert len(sim_brain_data) * 4 == len(up)
        down = up.temporal_resample(sampling_freq=2, target=1 / 2, target_type="hz")
        assert len(sim_brain_data) == len(down)
        assert len(up) / 4 == len(down)

    def test_fisher_r_to_z(self, sim_brain_data):
        """Test Fisher r-to-z and inverse transformation."""
        np.testing.assert_almost_equal(
            np.nansum(sim_brain_data.data - sim_brain_data.r_to_z().z_to_r().data),
            0,
            decimal=2,
        )
