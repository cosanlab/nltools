import warnings

import numpy as np
import nibabel as nb
import pytest

from nltools.data import BrainData
from nltools.mask import create_sphere, roi_to_brain
from nltools.data.simulator import Simulator
from nltools.algorithms import align


class TestBrainDataAnalysis:
    @pytest.mark.slow
    def test_extract_roi(self, sim_brain_data):
        """Test ROI extraction with different methods and labeled atlases."""
        n_images = sim_brain_data.shape[0]
        mask = create_sphere([12, 10, -8], radius=10)
        assert len(sim_brain_data.extract_roi(mask, method="mean")) == n_images
        assert len(sim_brain_data.extract_roi(mask, method="median")) == n_images
        n_components = 2
        assert sim_brain_data.extract_roi(
            mask, method="pca", n_components=n_components
        ).shape == (n_components, n_images)
        with pytest.raises(NotImplementedError):
            sim_brain_data.extract_roi(mask, method="p")

        assert isinstance(
            sim_brain_data[0].extract_roi(mask, method="mean"), (float, np.floating)
        )
        with pytest.raises(ValueError):
            sim_brain_data[0].extract_roi(mask, method="pca")

        s1 = create_sphere([15, 10, -8], radius=10)
        s2 = create_sphere([-15, 10, -8], radius=10)
        s3 = create_sphere([0, -15, -8], radius=10)
        masks = BrainData([s1, s2, s3])
        mask = roi_to_brain([1, 2, 3], masks)
        assert len(sim_brain_data[0].extract_roi(mask, method="mean")) == len(masks)
        assert sim_brain_data.extract_roi(mask, method="mean").shape == (
            len(masks),
            n_images,
        )

        # PCA on labeled atlas: n_components > 1 → list of per-ROI arrays
        pca_multi = sim_brain_data.extract_roi(mask, method="pca", n_components=2)
        assert isinstance(pca_multi, list)
        assert len(pca_multi) == len(masks)
        assert all(comp.shape == (2, n_images) for comp in pca_multi)

        # n_components == 1 → stacked ndarray
        pca_single = sim_brain_data.extract_roi(mask, method="pca", n_components=1)
        assert isinstance(pca_single, np.ndarray)
        assert pca_single.shape == (len(masks), 1, n_images)

    def test_extract_roi_nifti_atlas_with_many_labels_keeps_integer_labels(
        self, sim_brain_data
    ):
        """A raw NIfTI atlas with 1000+ labels is coerced with nearest-neighbour.

        BrainData's automatic interpolation switches to continuous above 1000
        unique values, which would blend integer labels; extract_roi must force
        nearest-neighbour on the NIfTI path so the result matches the explicit
        BrainData path exactly.
        """
        bd = BrainData(sim_brain_data.to_nifti(), mask="3mm-MNI152-2009fsl")
        template = BrainData(create_sphere([0, 0, 0], radius=40))
        n_voxels = template.data.shape[-1]
        labels = (np.arange(n_voxels) % 1200 + 1).astype(float)
        atlas = BrainData(np.tile(labels, (1, 1)), mask=template.mask)
        atlas.data = atlas.data.ravel()

        via_braindata = bd.extract_roi(atlas, method="mean")
        via_nifti = bd.extract_roi(atlas.to_nifti(), method="mean")

        assert via_braindata.shape == via_nifti.shape
        np.testing.assert_allclose(via_braindata, via_nifti)

    def test_extract_roi_does_not_mutate_a_same_grid_atlas(self, sim_brain_data):
        """The labeled branch rounds labels on a copy, never on the caller's mask."""
        bd = sim_brain_data
        centers = [[15, 10, -8], [-15, 10, -8], [0, -15, -8]]
        spheres = [create_sphere(c, radius=10) for c in centers]
        atlas = roi_to_brain([1.2, 2.2, 3.2], BrainData(spheres))
        before = atlas.data.copy()

        bd.extract_roi(atlas, method="mean")

        assert atlas.data.dtype == before.dtype
        np.testing.assert_array_equal(atlas.data, before)

    @pytest.mark.slow
    def test_extract_roi_resamples_mask_onto_object_grid(self, sim_brain_data):
        """extract_roi coerces a foreign-grid mask onto the object's own grid.

        A binary mask and a labeled atlas are supplied to a BrainData on a
        non-default (3 mm) grid three ways: as a BrainData already on the
        object's own grid, as a BrainData on the default (2 mm) grid, and as
        a raw NIfTI on the default grid. All three forms succeed and agree
        for the same ROI within nearest-neighbor resampling tolerance.
        """
        bd = BrainData(sim_brain_data.to_nifti(), mask="3mm-MNI152-2009fsl")
        center = [12, 10, -8]
        radius = 10

        binary_on_object_grid = create_sphere(center, radius=radius, mask=bd.mask)
        binary_on_default_grid = create_sphere(center, radius=radius)

        mask_a = BrainData(binary_on_object_grid, mask=bd.mask)
        mask_b = BrainData(binary_on_default_grid)
        mask_c = binary_on_default_grid

        value_a = bd.extract_roi(mask_a, method="mean")
        value_b = bd.extract_roi(mask_b, method="mean")
        value_c = bd.extract_roi(mask_c, method="mean")

        # The underlying signal is per-voxel gaussian noise (sigma=1), so ROI
        # means from slightly different (but heavily overlapping) voxel
        # selections agree only up to the resulting sampling noise, not
        # bit-for-bit; the absolute tolerance below is sized to that noise
        # floor rather than to the (near-zero) ROI means themselves.
        assert value_a.shape == value_b.shape == value_c.shape
        np.testing.assert_allclose(value_a, value_b, atol=0.3)
        np.testing.assert_allclose(value_a, value_c, atol=0.3)

        centers = [[15, 10, -8], [-15, 10, -8], [0, -15, -8]]
        spheres_on_object_grid = [
            create_sphere(c, radius=radius, mask=bd.mask) for c in centers
        ]
        spheres_on_default_grid = [create_sphere(c, radius=radius) for c in centers]

        atlas_a = roi_to_brain(
            [1, 2, 3], BrainData(spheres_on_object_grid, mask=bd.mask)
        )
        atlas_b = roi_to_brain([1, 2, 3], BrainData(spheres_on_default_grid))
        atlas_c = atlas_b.to_nifti()

        labels_a = bd.extract_roi(atlas_a, method="mean")
        labels_b = bd.extract_roi(atlas_b, method="mean")
        labels_c = bd.extract_roi(atlas_c, method="mean")

        assert labels_a.shape == labels_b.shape == labels_c.shape
        np.testing.assert_allclose(labels_a, labels_b, atol=0.3)
        np.testing.assert_allclose(labels_a, labels_c, atol=0.3)

    def test_extract_roi_raises_when_mask_has_no_overlap_after_coercion(
        self, sim_brain_data
    ):
        """A mask with no overlap after coercion raises the no-voxels error."""
        bd = BrainData(sim_brain_data.to_nifti(), mask="3mm-MNI152-2009fsl")

        # A single-voxel mask placed far outside the template's field of view:
        # resampling it onto bd's grid leaves an all-zero result.
        far_away_affine = np.eye(4)
        far_away_affine[:3, 3] = [500, 500, 500]
        far_away_mask = nb.Nifti1Image(
            np.ones((2, 2, 2), dtype=np.uint8), far_away_affine
        )

        with pytest.raises(ValueError, match="No voxels remain"):
            bd.extract_roi(far_away_mask, method="mean")

    def test_extract_roi_signature_is_canonical(self):
        """extract_roi selects an extraction variant via method= (metric is reserved)."""
        import inspect

        params = inspect.signature(BrainData.extract_roi).parameters
        assert params["method"].default == "mean"
        assert "metric" not in params

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

    def test_scale_divides_by_voxel_mean_or_grand_mean(self, minimal_brain_data):
        """#285: `axis=0` divides by each voxel's own mean, `axis=None` by the grand mean.

        Voxel j holds the constant value j + 1, so its temporal mean is j + 1 and
        the grand mean is 3.
        """
        bd = minimal_brain_data.copy()
        bd.data = np.tile(np.arange(1.0, 6.0), (50, 1))

        np.testing.assert_allclose(bd.scale(100.0, axis=0).data, 100.0)
        np.testing.assert_allclose(
            bd.scale(100.0).data, np.tile(np.arange(1.0, 6.0) / 3 * 100, (50, 1))
        )

    def test_standardize(self, minimal_brain_data):
        """Test standardization with different methods."""
        voxel_std = minimal_brain_data.data.std(axis=0)

        s = minimal_brain_data.standardize()
        assert s.shape == minimal_brain_data.shape
        assert np.isclose(np.sum(s.mean().data), 0, atol=0.5)
        np.testing.assert_allclose(s.data.std(axis=0), voxel_std)

        s = minimal_brain_data.standardize(method="zscore")
        assert s.shape == minimal_brain_data.shape
        assert np.isclose(np.sum(s.mean().data), 0, atol=0.5)
        np.testing.assert_allclose(s.data.std(axis=0), 1.0)

    def test_standardize_rejects_unknown_method(self, minimal_brain_data):
        """An unsupported `method` raises `ValueError` naming both choices."""
        with pytest.raises(ValueError, match="'center'.*'zscore'"):
            minimal_brain_data.standardize(method="rescale")

    def test_standardize_zscore_is_exact_and_silent_on_raw_bold(
        self, minimal_brain_data
    ):
        """Large-offset float32 data with constant voxels: no warnings, exact z-scores.

        Raw BOLD (values ~1e4, float32) tripped sklearn's precision warnings and
        constant voxels its near-zero-std warning; z-scoring must instead be
        computed in float64 and map constant voxels to 0.
        """
        bd = minimal_brain_data.copy()
        rng = np.random.default_rng(0)
        data = (12_000 + 50 * rng.standard_normal(bd.shape)).astype(np.float32)
        data[:, 0] = 12_345.0  # constant voxel
        bd.data = data

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            z = bd.standardize(method="zscore")

        assert z.data.dtype == np.float32
        np.testing.assert_allclose(z.data[:, 0], 0.0)
        np.testing.assert_allclose(z.data[:, 1:].mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(z.data[:, 1:].std(axis=0), 1.0, atol=1e-5)

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
        # "98%" resolves over the finite NONZERO values (v0.6.0 zero-aware
        # percentile change): the 98th percentile of the two-sphere values
        # {1, 5} is 5, so only the high-valued sphere survives — unlike
        # upper=0.5, which keeps both.
        assert np.sum(m3.data > 0) == np.sum(mask.data == 5)
        assert np.sum(m3.data > 0) < np.sum(m1.data > 0)
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

    def test_similarity_operand_is_named_data(self, minimal_brain_data):
        """The compared image is `data=`; v0.5.1's `image=` spelling is gone."""
        r = minimal_brain_data.similarity(data=minimal_brain_data[0])
        assert len(r) == minimal_brain_data.shape[0]
        with pytest.raises(TypeError):
            minimal_brain_data.similarity(image=minimal_brain_data[0])

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


class TestThresholdPercentileNonzero:
    """#479: percentile thresholds resolve over finite NONZERO voxels.

    On a masked stat map most voxels are exactly zero; including them
    dragged every percentile toward zero.
    """

    def test_percentile_excludes_zeros(self):
        import nibabel as nib

        from nltools.data import BrainData

        mask = nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.int8), np.eye(4))
        data = np.zeros((1, 27), dtype=np.float32)
        data[0, :4] = [1.0, 2.0, 3.0, 4.0]
        bd = BrainData(data, mask=mask)

        out = bd.threshold(upper="50%")
        cutoff = float(np.percentile([1.0, 2.0, 3.0, 4.0], 50))  # 2.5, not ~0
        surviving = out.data[out.data != 0]
        np.testing.assert_array_equal(np.sort(surviving), [3.0, 4.0])
        assert (np.abs(out.data) >= cutoff).sum() == 2


class TestFilterDetrendStandardize:
    """F047: `_filter_data` double-passed detrend/standardize to nilearn's clean().

    It read them with ``kwargs.get()`` (leaving them in ``kwargs``) then
    forwarded them both explicitly and again via ``**kwargs``, so the documented
    ``filter(..., detrend=True)`` usage raised "got multiple values".
    """

    def test_filter_with_detrend_via_kwargs(self, minimal_brain_data):
        """The documented `filter(..., detrend=True)` usage must not crash."""
        out = minimal_brain_data.filter(sampling_freq=2.0, high_pass=0.01, detrend=True)
        assert out.data.shape == minimal_brain_data.data.shape

    def test_filter_with_standardize_via_kwargs(self, minimal_brain_data):
        """`standardize` passed via kwargs must reach clean() exactly once."""
        out = minimal_brain_data.filter(
            sampling_freq=2.0, high_pass=0.01, standardize="zscore_sample"
        )
        # Standardized output should be roughly zero-mean per voxel.
        np.testing.assert_allclose(out.data.mean(axis=0), 0.0, atol=1e-6)


class TestStandardizeIsNotABool:
    """nilearn 0.15 drops boolean ``standardize``; never hand it one."""

    def test_filter_default_emits_no_future_warning(self, minimal_brain_data):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            minimal_brain_data.filter(sampling_freq=2.0, high_pass=0.01)

    def test_filter_maps_true_to_zscore_sample(self, minimal_brain_data):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            out = minimal_brain_data.filter(
                sampling_freq=2.0, high_pass=0.01, standardize=True
            )
        np.testing.assert_allclose(out.data.mean(axis=0), 0.0, atol=1e-6)

    def test_filter_false_means_off(self, minimal_brain_data):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            out = minimal_brain_data.filter(
                sampling_freq=2.0, high_pass=0.01, standardize=False
            )
        default = minimal_brain_data.filter(sampling_freq=2.0, high_pass=0.01)
        np.testing.assert_array_equal(out.data, default.data)

    def test_extract_roi_labels_emits_no_future_warning(self):
        shape, affine = (6, 6, 6), np.eye(4)
        mask = nb.Nifti1Image(np.ones(shape, dtype=np.int8), affine)
        labels = np.zeros(shape)  # 0 = background, as nilearn requires
        labels[:2] = 1
        labels[2:4] = 2
        atlas = BrainData(nb.Nifti1Image(labels, affine), mask=mask)
        rng = np.random.default_rng(0)
        brain = BrainData(
            nb.Nifti1Image(rng.standard_normal(shape + (4,)), affine), mask=mask
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            out = brain.extract_roi(atlas, method="mean")
        assert out.shape == (2, 4)
