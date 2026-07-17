import numpy as np
import nibabel as nb
import pytest

from nltools.data import BrainData
from nltools.mask import create_sphere, roi_to_brain
from nltools.data.simulator import Simulator
from nltools.stats import align


class TestBrainDataAnalysis:
    @pytest.mark.slow
    def test_apply_mask(self, sim_brain_data):
        """Test applying masks to BrainData."""
        s1 = create_sphere([12, 10, -8], radius=10)
        assert isinstance(s1, nb.Nifti1Image)
        masked_dat = sim_brain_data.apply_mask(s1)
        assert masked_dat.shape[1] == np.sum(s1.get_fdata() != 0)
        masked_dat = sim_brain_data.apply_mask(s1, resample_mask_to_brain=True)
        assert masked_dat.shape[1] == np.sum(s1.get_fdata() != 0)

    def test_apply_mask_dimension_compatibility(self, sim_brain_data):
        """Test mask as BrainData with dimension handling."""
        s1 = create_sphere([12, 10, -8], radius=10)
        mask_bd = BrainData(s1, mask=sim_brain_data.mask)
        result = sim_brain_data.apply_mask(mask_bd)
        assert isinstance(result, BrainData)
        assert result.shape[1] == mask_bd.data.astype(bool).sum()

        # With resampling
        result_resample = sim_brain_data.apply_mask(
            mask_bd, resample_mask_to_brain=True
        )
        assert isinstance(result_resample, BrainData)
        assert result_resample.shape[1] == np.sum(s1.get_fdata() != 0)

        # Without resampling
        result_no_resample = sim_brain_data.apply_mask(
            mask_bd, resample_mask_to_brain=False
        )
        assert isinstance(result_no_resample, BrainData)
        assert result_no_resample.shape[1] == mask_bd.data.astype(bool).sum()

    def test_apply_mask_invalid_4d(self, sim_brain_data):
        """Multi-volume mask should raise clear error."""
        s1 = create_sphere([12, 10, -8], radius=10)
        from nilearn.image import concat_imgs

        invalid_mask = concat_imgs([s1, s1])
        mask_bd = BrainData(invalid_mask, mask=sim_brain_data.mask)

        with pytest.raises(ValueError, match="Mask must be a single image"):
            sim_brain_data.apply_mask(mask_bd)

    @pytest.mark.slow
    def test_extract_roi(self, sim_brain_data):
        """Test ROI extraction with different metrics and labeled atlases."""
        n_images = sim_brain_data.shape[0]
        mask = create_sphere([12, 10, -8], radius=10)
        assert len(sim_brain_data.extract_roi(mask, metric="mean")) == n_images
        assert len(sim_brain_data.extract_roi(mask, metric="median")) == n_images
        n_components = 2
        assert sim_brain_data.extract_roi(
            mask, metric="pca", n_components=n_components
        ).shape == (n_components, n_images)
        with pytest.raises(NotImplementedError):
            sim_brain_data.extract_roi(mask, metric="p")

        assert isinstance(
            sim_brain_data[0].extract_roi(mask, metric="mean"), (float, np.floating)
        )
        with pytest.raises(ValueError):
            sim_brain_data[0].extract_roi(mask, metric="pca")

        s1 = create_sphere([15, 10, -8], radius=10)
        s2 = create_sphere([-15, 10, -8], radius=10)
        s3 = create_sphere([0, -15, -8], radius=10)
        masks = BrainData([s1, s2, s3])
        mask = roi_to_brain([1, 2, 3], masks)
        assert len(sim_brain_data[0].extract_roi(mask, metric="mean")) == len(masks)
        assert sim_brain_data.extract_roi(mask, metric="mean").shape == (
            len(masks),
            n_images,
        )

        # PCA on labeled atlas: n_components > 1 → list of per-ROI arrays
        pca_multi = sim_brain_data.extract_roi(mask, metric="pca", n_components=2)
        assert isinstance(pca_multi, list)
        assert len(pca_multi) == len(masks)
        assert all(comp.shape == (2, n_images) for comp in pca_multi)

        # n_components == 1 → stacked ndarray
        pca_single = sim_brain_data.extract_roi(mask, metric="pca", n_components=1)
        assert isinstance(pca_single, np.ndarray)
        assert pca_single.shape == (len(masks), 1, n_images)

    # ==================== Transform Methods ====================

    def test_r_to_z(self, minimal_brain_data):
        """Test Fisher r-to-z transformation."""
        z = minimal_brain_data.r_to_z()
        assert z.shape == minimal_brain_data.shape

    def test_copy(self, minimal_brain_data):
        """Test copying BrainData objects."""
        d_copy = minimal_brain_data.copy()
        assert d_copy.shape == minimal_brain_data.shape

    def test_detrend(self, minimal_brain_data):
        """Test detrending removes linear trends."""
        detrend = minimal_brain_data.detrend()
        assert detrend.shape == minimal_brain_data.shape

    @pytest.mark.filterwarnings("ignore:Numerical issues:UserWarning")
    def test_standardize(self, minimal_brain_data):
        """Test standardization with different methods."""
        s = minimal_brain_data.standardize()
        assert s.shape == minimal_brain_data.shape
        assert np.isclose(np.sum(s.mean().data), 0, atol=0.5)
        s = minimal_brain_data.standardize(method="zscore")
        assert s.shape == minimal_brain_data.shape
        assert np.isclose(np.sum(s.mean().data), 0, atol=0.5)

    def test_filter_high_pass(self, minimal_brain_data):
        """Test high-pass filtering returns BrainData with correct shape."""
        filtered = minimal_brain_data.filter(sampling_freq=0.5, high_pass=0.01)
        assert isinstance(filtered, BrainData)
        assert filtered.shape == minimal_brain_data.shape
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
        with pytest.raises(ValueError, match="must.*provided"):
            minimal_brain_data.filter(sampling_freq=0.5)

    def test_filter_kwargs_passed_through(self, minimal_brain_data):
        """Test that additional kwargs reach nilearn.signal.clean."""
        filtered = minimal_brain_data.filter(
            sampling_freq=0.5,
            high_pass=0.01,
            ensure_finite=True,
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

        # Regions
        r = mask.regions(min_region_size=10)
        m1 = BrainData(s1)
        m2 = r.threshold(upper=1, binarize=True)
        assert len(np.unique(r.to_nifti().get_fdata())) == 2
        diff = m2 - m1
        assert np.sum(diff.data) == 0

    # ============================================================================
    # Thresholding Operations - Cluster Enhancement
    # ============================================================================

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "lower,upper,cluster_threshold,binarize",
        [
            (2, None, 10, False),  # lower only
            (None, 2, 10, False),  # upper only
            (2.5, None, 50, False),  # realistic workflow
            (2, None, 10, True),  # with binarization
        ],
        ids=["lower_only", "upper_only", "realistic", "binarize"],
    )
    def test_threshold_cluster(
        self, sim_brain_data, lower, upper, cluster_threshold, binarize
    ):
        """Cluster thresholding with various parameter combinations."""
        brain = sim_brain_data.copy()
        result = brain.threshold(
            lower=lower,
            upper=upper,
            cluster_threshold=cluster_threshold,
            binarize=binarize,
        )
        assert isinstance(result, BrainData)
        assert result.shape == brain.shape

        if binarize:
            unique_vals = np.unique(result.data)
            assert len(unique_vals) <= 2
            assert all(v in [0, 1] for v in unique_vals)

    def test_threshold_cluster_rejects_bandpass(self, sim_brain_data):
        """Should raise error when using both upper AND lower with cluster_threshold."""
        brain = sim_brain_data.copy()
        with pytest.raises(
            ValueError, match="Band-pass filtering.*not supported.*cluster"
        ):
            brain.threshold(lower=-2, upper=2, cluster_threshold=10)

    def test_threshold_cluster_zero_disables(self, sim_brain_data):
        """cluster_threshold=0 should match no-cluster behavior."""
        brain = sim_brain_data.copy()
        result_no_cluster = brain.threshold(lower=-2, upper=2)
        result_zero_cluster = brain.threshold(lower=-2, upper=2, cluster_threshold=0)
        np.testing.assert_array_equal(result_no_cluster.data, result_zero_cluster.data)

    def test_threshold_with_zero_value(self, sim_brain_data):
        """Test threshold works correctly when upper=0 or lower=0 (#370)."""
        brain = sim_brain_data.copy()
        brain.data = brain.data - brain.data.mean()

        result_upper0 = brain.threshold(upper=0)
        assert np.all(result_upper0.data >= 0), "upper=0 should zero values < 0"

        result_lower0 = brain.threshold(lower=0)
        assert np.all(result_lower0.data <= 0), "lower=0 should zero values > 0"

        result_bandpass = brain.threshold(upper=0, lower=-0.5)
        non_zero = result_bandpass.data[result_bandpass.data != 0]
        assert np.all((non_zero >= 0) | (non_zero <= -0.5))

    # ==================== Similarity & Analysis ====================

    def test_similarity(self, minimal_brain_data):
        """Test similarity computation with different metrics."""
        r = minimal_brain_data.similarity(minimal_brain_data, metric="correlation")
        assert r.shape == (minimal_brain_data.shape[0], minimal_brain_data.shape[0])
        r = minimal_brain_data.similarity(minimal_brain_data, metric="dot_product")
        assert r.shape == (minimal_brain_data.shape[0], minimal_brain_data.shape[0])
        r = minimal_brain_data.similarity(minimal_brain_data, metric="cosine")
        assert r.shape == (minimal_brain_data.shape[0], minimal_brain_data.shape[0])

        r = minimal_brain_data.similarity(minimal_brain_data[0], metric="correlation")
        assert len(r) == minimal_brain_data.shape[0]

    @pytest.mark.slow
    def test_decompose(self, sim_brain_data):
        """Test decomposition with PCA, ICA, NMF, and Factor Analysis."""
        n_components = 3
        for method in ["pca", "ica", "fa"]:
            stats = sim_brain_data.decompose(
                method=method, axis="voxels", n_components=n_components
            )
            assert n_components == len(stats["components"])
            assert stats["weights"].shape == (len(sim_brain_data), n_components)

        # NMF needs non-negative data
        bd = sim_brain_data.copy()
        bd.data = bd.data + 2
        bd.data[bd.data < 0] = 0
        stats = bd.decompose(method="nnmf", axis="voxels", n_components=n_components)
        assert n_components == len(stats["components"])

        # Test axis="images"
        stats = sim_brain_data.decompose(
            method="pca", axis="images", n_components=n_components
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

        # Deterministic SRM
        out = align(data, method="deterministic_srm")
        bout = d1.align(out["common_model"], method="deterministic_srm")
        assert d1.shape == bout["transformed"].shape
        assert d1.shape == bout["common_model"].shape
        assert d1.shape[1] == bout["transformation_matrix"].shape[0]
        btransformed = np.dot(d1.data, bout["transformation_matrix"].data.T)
        np.testing.assert_almost_equal(
            0, np.sum(bout["transformed"].data - btransformed)
        )

        # Probabilistic SRM
        bout = d1.align(out["common_model"], method="probabilistic_srm")
        assert d1.shape == bout["transformed"].shape
        btransformed = np.dot(d1.data, bout["transformation_matrix"].data.T)
        np.testing.assert_almost_equal(
            0, np.sum(bout["transformed"].data - btransformed)
        )

        # Procrustes
        out = align(data, method="procrustes")
        bout = d1.align(out["common_model"], method="procrustes")
        assert d1.shape == bout["transformed"].shape
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

    def test_fisher_r_to_z(self, minimal_brain_data):
        """Test Fisher r-to-z and inverse transformation."""
        np.testing.assert_almost_equal(
            np.nansum(
                minimal_brain_data.data - minimal_brain_data.r_to_z().z_to_r().data
            ),
            0,
            decimal=2,
        )
