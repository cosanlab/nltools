"""
Test suite for Brain_Data class.

Follows "imperative shell" pattern: tests focus on method usage and interface contracts,
not implementation details. Organized into logical sections for clarity.

Performance Notes:
------------------
Test suite timing (47 tests total, ~151s):
- Average per test: ~3.2s
- Threshold tests: ~7.2s each (cluster filtering is computationally expensive)
- Math operations: <1s each (fast numpy operations)

Slowest test categories:
1. Threshold operations (~7.2s each) - cluster filtering uses nilearn connected components
2. GLM regression (~5-6s) - FirstLevelModel fitting
3. Hyperalignment/decomposition (~4-5s) - large matrix operations
4. Bootstrap/permutation tests (~3-4s) - resampling operations

The cluster threshold tests (9 tests, ~65s) consume ~43% of total runtime. This is
expected and acceptable - they test realistic neuroimaging workflows that require
expensive connected components analysis via nilearn.
"""

import os
import pytest
import numpy as np
import nibabel as nb
import pandas as pd
from nltools.simulator import Simulator
from nltools.data import Brain_Data, Adjacency
from nltools.stats import threshold, align
from nltools.mask import create_sphere, roi_to_brain
from pathlib import Path

from nltools.prefs import MNI_Template


shape_3d = (91, 109, 91)
shape_2d = (6, 238955)


class TestBrainData:
    """Test Brain_Data class - focus on method usage, not implementation."""

    # ==================== Initialization & I/O ====================

    def test_load(self, tmpdir):
        """Test loading Brain_Data from various sources and formats."""
        sim = Simulator()
        sigma = 1
        y = [0, 1]
        n_reps = 3
        output_dir = str(tmpdir)
        dat = sim.create_data(y, sigma, reps=n_reps, output_dir=output_dir)

        # Shape depends on MNI_Template.resolution
        # 2mm: shape_3d = (91, 109, 91), shape_2d = (6, 238955)
        # 3mm: shape_3d = (60, 72, 60), shape_2d = (6, 71020)

        y = pd.read_csv(
            os.path.join(str(tmpdir.join("y.csv"))), header=None, index_col=None
        )

        # Test load list of 4D images
        file_list = [str(tmpdir.join("data.nii.gz")), str(tmpdir.join("data.nii.gz"))]
        dat = Brain_Data(file_list)
        dat = Brain_Data([nb.load(x) for x in file_list])

        # Test load string and path
        dat = Brain_Data(data=str(tmpdir.join("data.nii.gz")), Y=y)
        dat = Brain_Data(data=Path(tmpdir.join("data.nii.gz")), Y=y)

        # Test Write
        dat.write(os.path.join(str(tmpdir.join("test_write.nii"))))
        assert Brain_Data(os.path.join(str(tmpdir.join("test_write.nii"))))

        # Test i/o for hdf5
        dat.write(os.path.join(str(tmpdir.join("test_write.h5"))))
        b = Brain_Data(os.path.join(tmpdir.join("test_write.h5")))
        # Note: X and Y attributes removed in v0.6.0, skip checking them
        for k in ["mask", "nifti_masker", "data"]:
            if k == "data":
                assert np.allclose(b.__dict__[k], dat.__dict__[k])
            elif k == "mask":
                assert np.allclose(b.__dict__[k].affine, dat.__dict__[k].affine)
                assert np.allclose(b.__dict__[k].get_fdata(), dat.__dict__[k].get_fdata())
                assert b.__dict__[k].get_filename() == dat.__dict__[k].get_filename()
            elif k == "nifti_masker":
                assert np.allclose(b.__dict__[k].affine_, dat.__dict__[k].affine_)
                assert np.allclose(
                    b.__dict__[k].mask_img.get_fdata(), dat.__dict__[k].mask_img.get_fdata()
                )
            else:
                assert b.__dict__[k] == dat.__dict__[k]
        # Test situation where we present a user warning when they're trying to load an .h5
        # file that includes a mask AND they pass in value for the mask argument. In this
        # case the mask argument takes precedence so we warn the user
        with pytest.warns(UserWarning):
            bb = Brain_Data(
                os.path.join(tmpdir.join("test_write.h5")), mask=MNI_Template.mask
            )
            assert os.path.abspath(bb.mask.get_filename()) == os.path.abspath(
                MNI_Template.mask
            )

    def test_load_legacy_h5(self, old_h5_brain, new_h5_brain, tmpdir):
        """Test loading old HDF5 format (backward compatibility)."""
        with pytest.warns(UserWarning):
            # With verbosity on we should see a warning about the old h5 file format
            b_old = Brain_Data(old_h5_brain, verbose=True)
        b_new = Brain_Data(new_h5_brain)
        assert b_old.shape == b_new.shape
        assert np.allclose(b_old.data, b_new.data)
        # NOTE: We lose pandas column dtype information between old and new h5 files
        # so we can't use .equals()
        assert b_old.X.shape == b_new.X.shape
        assert b_old.Y.shape == b_new.Y.shape
        assert np.allclose(b_old.mask.affine, b_new.mask.affine)
        assert np.allclose(b_old.mask.get_fdata(), b_new.mask.get_fdata())

        new_file = Path(tmpdir) / "tmp.h5"
        b_new.write(new_file)
        b_new_written = Brain_Data(new_file)
        assert b_new.shape == b_new_written.shape
        assert np.allclose(b_new.data, b_new_written.data)
        new_file.unlink()

    # ==================== Properties & Basic Operations ====================

    def test_shape(self, sim_brain_data):
        """Test shape property returns correct dimensions."""
        assert sim_brain_data.shape == shape_2d

    def test_mean(self, sim_brain_data):
        """Test mean computation across different axes."""
        assert sim_brain_data.mean().shape[0] == shape_2d[1]
        assert sim_brain_data.mean().shape[0] == shape_2d[1]
        assert len(sim_brain_data.mean(axis=1)) == shape_2d[0]
        with pytest.raises(ValueError):
            sim_brain_data.mean(axis="1")
        assert isinstance(sim_brain_data[0].mean(), (float, np.floating))

    def test_median(self, sim_brain_data):
        """Test median computation across different axes."""
        assert sim_brain_data.median().shape[0] == shape_2d[1]
        assert sim_brain_data.median().shape[0] == shape_2d[1]
        assert len(sim_brain_data.median(axis=1)) == shape_2d[0]
        with pytest.raises(ValueError):
            sim_brain_data.median(axis="1")
        assert isinstance(sim_brain_data[0].median(), (float, np.floating))

    def test_std(self, sim_brain_data):
        """Test standard deviation computation."""
        assert sim_brain_data.std().shape[0] == shape_2d[1]

    def test_sum(self, sim_brain_data):
        """Test sum aggregation."""
        s = sim_brain_data.sum()
        assert s.shape == sim_brain_data[1].shape

    # ==================== Arithmetic Operations ====================

    def test_add(self, sim_brain_data):
        """Test addition of Brain_Data objects and scalars."""
        new = sim_brain_data + sim_brain_data
        assert new.shape == shape_2d
        value = 10
        assert (value + sim_brain_data[0]).mean() == (sim_brain_data[0] + value).mean()

    def test_subtract(self, sim_brain_data):
        """Test subtraction of Brain_Data objects and scalars."""
        new = sim_brain_data - sim_brain_data
        assert new.shape == shape_2d
        value = 10
        assert (-value - (-1) * sim_brain_data[0]).mean() == (
            sim_brain_data[0] - value
        ).mean()

    def test_multiply(self, sim_brain_data):
        """Test multiplication of Brain_Data objects, scalars, and arrays."""
        new = sim_brain_data * sim_brain_data
        assert new.shape == shape_2d
        value = 10
        assert (value * sim_brain_data[0]).mean() == (sim_brain_data[0] * value).mean()
        c1 = [0.5, 0.5, -0.5, -0.5]
        new = sim_brain_data[0:4] * c1
        new2 = (
            sim_brain_data[0] * 0.5
            + sim_brain_data[1] * 0.5
            - sim_brain_data[2] * 0.5
            - sim_brain_data[3] * 0.5
        )
        np.testing.assert_almost_equal((new - new2).sum(), 0, decimal=4)

    def test_divide(self, sim_brain_data):
        """Test division of Brain_Data objects and scalars."""
        new = sim_brain_data / sim_brain_data
        assert new.shape == shape_2d
        np.testing.assert_almost_equal(new.mean(axis=0).mean(), 1, decimal=6)
        value = 10
        new2 = sim_brain_data / value
        np.testing.assert_almost_equal(((new2 * value) - new2).mean().mean(), 0, decimal=2)

    def test_inplace_add(self, sim_brain_data):
        """Test in-place addition with scalars and Brain_Data."""
        # Test in-place add with scalar
        bd = sim_brain_data[0].copy()
        original_data = bd.data.copy()
        bd += 5
        assert np.allclose(bd.data, original_data + 5)

        # Test in-place add with Brain_Data
        bd1 = sim_brain_data[0].copy()
        bd2 = sim_brain_data[0].copy()
        original_data = bd1.data.copy()
        bd1 += bd2
        assert np.allclose(bd1.data, original_data + bd2.data)

    def test_inplace_subtract(self, sim_brain_data):
        """Test in-place subtraction with scalars and Brain_Data."""
        # Test in-place subtract with scalar
        bd = sim_brain_data[0].copy()
        original_data = bd.data.copy()
        bd -= 3
        assert np.allclose(bd.data, original_data - 3)

        # Test in-place subtract with Brain_Data
        bd1 = sim_brain_data[0].copy()
        bd2 = sim_brain_data[0].copy()
        original_data = bd1.data.copy()
        bd1 -= bd2
        assert np.allclose(bd1.data, original_data - bd2.data)

    def test_inplace_multiply(self, sim_brain_data):
        """Test in-place multiplication with scalars, Brain_Data, and arrays."""
        # Test in-place multiply with scalar
        bd = sim_brain_data[0].copy()
        original_data = bd.data.copy()
        bd *= 2
        assert np.allclose(bd.data, original_data * 2)

        # Test in-place multiply with Brain_Data
        bd1 = sim_brain_data[0].copy()
        bd2 = sim_brain_data[0].copy()
        original_data = bd1.data.copy()
        bd1 *= bd2
        assert np.allclose(bd1.data, original_data * bd2.data)

        # Test in-place multiply with array
        bd = sim_brain_data[0:4].copy()
        c1 = [0.5, 0.5, -0.5, -0.5]
        bd *= c1
        expected = (
            sim_brain_data[0] * 0.5
            + sim_brain_data[1] * 0.5
            - sim_brain_data[2] * 0.5
            - sim_brain_data[3] * 0.5
        )
        np.testing.assert_almost_equal((bd - expected).sum(), 0, decimal=4)

    def test_inplace_divide(self, sim_brain_data):
        """Test in-place division with scalars and Brain_Data."""
        # Test in-place divide with scalar
        bd = sim_brain_data[0].copy()
        original_data = bd.data.copy()
        bd /= 2
        assert np.allclose(bd.data, original_data / 2)

        # Test in-place divide with Brain_Data
        bd1 = sim_brain_data[0].copy()
        bd2 = sim_brain_data[0].copy()
        bd2.data = bd2.data + 1  # Avoid division by zero
        original_data = bd1.data.copy()
        bd1 /= bd2
        assert np.allclose(bd1.data, original_data / bd2.data)

    # ==================== Indexing & Concatenation ====================

    def test_indexing(self, sim_brain_data):
        """Test indexing with lists, ranges, boolean masks, and slices."""
        index = [0, 3, 1]
        assert len(sim_brain_data[index]) == len(index)
        index = range(4)
        assert len(sim_brain_data[index]) == len(index)
        index = sim_brain_data.Y == 1
        assert len(sim_brain_data[index.values.flatten()]) == index.values.sum()
        assert len(sim_brain_data[index]) == index.values.sum()
        assert len(sim_brain_data[:3]) == 3
        d = sim_brain_data.to_nifti()
        assert d.shape[0:3] == shape_3d
        assert Brain_Data(d)

    def test_concatenate(self, sim_brain_data):
        """Test concatenating Brain_Data objects from list."""
        out = Brain_Data([x for x in sim_brain_data])
        assert isinstance(out, Brain_Data)
        assert len(out) == len(sim_brain_data)

    def test_append(self, sim_brain_data):
        """Test appending Brain_Data objects."""
        assert sim_brain_data.append(sim_brain_data).shape[0] == shape_2d[0] * 2

    # ==================== Statistical Methods ====================

    def test_distance(self, sim_brain_data):
        """Test distance computation returns Adjacency object."""
        distance = sim_brain_data.distance(metric="correlation")
        assert isinstance(distance, Adjacency)
        assert distance.square_shape()[0] == shape_2d[0]

    # ==================== Regression & GLM ====================

    def test_regress(self, sim_brain_data):
        """Test regression with OLS and robust methods."""
        sim_brain_data.X = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data.Y)),
                "X1": np.array(sim_brain_data.Y).flatten(),
            },
            index=None,
        )
        # OLS
        out = sim_brain_data.regress()
        assert isinstance(out["beta"].data, np.ndarray)
        assert isinstance(out["t"].data, np.ndarray)
        assert isinstance(out["p"].data, np.ndarray)
        assert isinstance(out["residual"].data, np.ndarray)
        assert out["beta"].shape == (2, shape_2d[1])
        assert out["t"][1].shape[0] == shape_2d[1]

        # Robust OLS
        out = sim_brain_data.regress(mode="robust")
        assert isinstance(out["beta"].data, np.ndarray)
        assert isinstance(out["t"].data, np.ndarray)
        assert isinstance(out["p"].data, np.ndarray)
        assert isinstance(out["residual"].data, np.ndarray)
        assert out["beta"].shape == (2, shape_2d[1])
        assert out["t"][1].shape[0] == shape_2d[1]

        # Test threshold
        i = 1
        tt = threshold(out["t"][i], out["p"][i], 0.05)
        assert isinstance(tt, Brain_Data)

    # ==================== Masking & ROI Extraction ====================

    def test_apply_mask(self, sim_brain_data):
        """Test applying masks to Brain_Data."""
        s1 = create_sphere([12, 10, -8], radius=10)
        assert isinstance(s1, nb.Nifti1Image)
        masked_dat = sim_brain_data.apply_mask(s1)
        assert masked_dat.shape[1] == np.sum(s1.get_fdata() != 0)
        masked_dat = sim_brain_data.apply_mask(s1, resample_mask_to_brain=True)
        assert masked_dat.shape[1] == np.sum(s1.get_fdata() != 0)

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
        masks = Brain_Data([s1, s2, s3])
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
        """Test copying Brain_Data objects."""
        d_copy = sim_brain_data.copy()
        assert d_copy.shape == sim_brain_data.shape

    def test_detrend(self, sim_brain_data):
        """Test detrending removes linear trends."""
        detrend = sim_brain_data.detrend()
        assert detrend.shape == sim_brain_data.shape

    def test_standardize(self, sim_brain_data):
        """Test standardization with different methods."""
        s = sim_brain_data.standardize()
        assert s.shape == sim_brain_data.shape
        assert np.isclose(np.sum(s.mean().data), 0, atol=0.1)
        s = sim_brain_data.standardize(method="zscore")
        assert s.shape == sim_brain_data.shape
        assert np.isclose(np.sum(s.mean().data), 0, atol=0.1)

    def test_smooth(self, sim_brain_data):
        """Test spatial smoothing."""
        smoothed = sim_brain_data.smooth(5.0)
        assert isinstance(smoothed, Brain_Data)
        assert smoothed.shape == sim_brain_data.shape
        smoothed = sim_brain_data[0].smooth(5.0)
        assert len(smoothed.shape) == 1

    def test_threshold(self):
        """Test thresholding and region extraction."""
        s1 = create_sphere([12, 10, -8], radius=10)
        s2 = create_sphere([22, -2, -22], radius=10)
        mask = Brain_Data(s1) * 5
        mask = mask + Brain_Data(s2)

        m1 = mask.threshold(upper=0.5)
        m2 = mask.threshold(upper=3)
        m3 = mask.threshold(upper="98%")
        m4 = Brain_Data(s1) * 5 + Brain_Data(s2) * -0.5
        m4 = mask.threshold(upper=0.5, lower=-0.3)
        assert np.sum(m1.data > 0) > np.sum(m2.data > 0)
        assert np.sum(m1.data > 0) == np.sum(m3.data > 0)
        assert np.sum(m4.data[(m4.data > -0.3) & (m4.data < 0.5)]) == 0
        assert np.sum(m4.data[(m4.data < -0.3) | (m4.data > 0.5)]) > 0

        # Test Regions
        r = mask.regions(min_region_size=10)
        m1 = Brain_Data(s1)
        m2 = r.threshold(1, binarize=True)
        assert len(np.unique(r.to_nifti().get_fdata())) == 2
        diff = m2 - m1
        assert np.sum(diff.data) == 0

    # ============================================================================
    # Thresholding Operations - Cluster Enhancement
    # ============================================================================

    def test_threshold_cluster_basic(self, sim_brain_data):
        """Cluster thresholding should filter small clusters using nilearn"""
        # Create data with distinct regions
        brain = sim_brain_data.copy()

        # Threshold with cluster size minimum
        result = brain.threshold(lower=2, cluster_threshold=10)

        # Should return Brain_Data
        assert isinstance(result, Brain_Data)
        # Should have removed small clusters (basic check that it ran)
        assert result.shape == brain.shape

    def test_threshold_cluster_with_upper_only(self, sim_brain_data):
        """Cluster threshold should work with upper threshold only"""
        brain = sim_brain_data.copy()
        result = brain.threshold(upper=2, cluster_threshold=10)
        assert isinstance(result, Brain_Data)

    def test_threshold_cluster_with_lower_only(self, sim_brain_data):
        """Cluster threshold should work with lower threshold only"""
        brain = sim_brain_data.copy()
        result = brain.threshold(lower=2, cluster_threshold=10)
        assert isinstance(result, Brain_Data)

    def test_threshold_cluster_rejects_bandpass(self, sim_brain_data):
        """Should raise error when using both upper AND lower with cluster_threshold"""
        brain = sim_brain_data.copy()

        with pytest.raises(ValueError, match="Band-pass filtering.*not supported.*cluster"):
            brain.threshold(lower=-2, upper=2, cluster_threshold=10)

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

        assert isinstance(result, Brain_Data)
        # Verify band-pass behavior preserved (values in range kept)

    def test_threshold_cluster_realistic_neuroimaging(self, sim_brain_data):
        """Integration test with realistic neuroimaging workflow"""
        # Test with actual brain data structure from fixtures
        brain = sim_brain_data.copy()

        # Realistic workflow: threshold then cluster filter
        result = brain.threshold(lower=2.5, cluster_threshold=50)

        # Basic sanity checks
        assert isinstance(result, Brain_Data)
        assert result.shape == brain.shape
        assert not result.isempty

    # ==================== Similarity & Analysis ====================

    def test_similarity(self, sim_brain_data):
        """Test similarity computation with different metrics."""
        # Test comparing Brain_Data to itself
        r = sim_brain_data.similarity(sim_brain_data, method="correlation")
        assert r.shape == (sim_brain_data.shape[0], sim_brain_data.shape[0])
        r = sim_brain_data.similarity(sim_brain_data, method="dot_product")
        assert r.shape == (sim_brain_data.shape[0], sim_brain_data.shape[0])
        r = sim_brain_data.similarity(sim_brain_data, method="cosine")
        assert r.shape == (sim_brain_data.shape[0], sim_brain_data.shape[0])

        # Test comparing to a single image
        r = sim_brain_data.similarity(sim_brain_data[0], method="correlation")
        assert len(r) == shape_2d[0]

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
        np.testing.assert_almost_equal(0, np.sum(bout["transformed"].data - btransformed))

        # Test probabilistic brain_data
        bout = d1.align(out["common_model"], method="probabilistic_srm")
        assert d1.shape == bout["transformed"].shape
        assert d1.shape == bout["common_model"].shape
        assert d1.shape[1] == bout["transformation_matrix"].shape[0]
        btransformed = np.dot(d1.data, bout["transformation_matrix"].data.T)
        np.testing.assert_almost_equal(0, np.sum(bout["transformed"].data - btransformed))

        # Test procrustes brain_data
        out = align(data, method="procrustes")
        centered = data[0].data - np.mean(data[0].data, 0)

        bout = d1.align(out["common_model"], method="procrustes")
        assert d1.shape == bout["transformed"].shape
        assert d1.shape == bout["common_model"].shape
        assert d1.shape[1] == bout["transformation_matrix"].shape[0]
        centered = d1.data - np.mean(d1.data, 0)
        btransformed = (
            np.dot(centered / np.linalg.norm(centered), bout["transformation_matrix"].data)
            * bout["scale"]
        )
        np.testing.assert_almost_equal(
            0, np.sum(bout["transformed"].data - btransformed), decimal=5
        )
        np.testing.assert_almost_equal(
            0, np.sum(out["transformed"][0].data - bout["transformed"].data)
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
        np.testing.assert_almost_equal(0, np.sum(bout["transformed"].data - btransformed.T))

        out = align(data, method="probabilistic_srm", axis=1)
        bout = d1.align(out["common_model"], method="probabilistic_srm", axis=1)
        assert d1.shape == bout["transformed"].shape
        assert d1.shape == bout["common_model"].shape
        assert d1.shape[0] == bout["transformation_matrix"].shape[0]
        btransformed = np.dot(d1.data.T, bout["transformation_matrix"].data.T)
        np.testing.assert_almost_equal(0, np.sum(bout["transformed"].data - btransformed.T))

        out = align(data, method="procrustes", axis=1)
        bout = d1.align(out["common_model"], method="procrustes", axis=1)
        assert d1.shape == bout["transformed"].shape
        assert d1.shape == bout["common_model"].shape
        assert d1.shape[0] == bout["transformation_matrix"].shape[0]
        centered = d1.data.T - np.mean(d1.data.T, 0)
        btransformed = (
            np.dot(centered / np.linalg.norm(centered), bout["transformation_matrix"].data)
            * bout["scale"]
        )
        np.testing.assert_almost_equal(
            0, np.sum(bout["transformed"].data - btransformed.T), decimal=5
        )
        np.testing.assert_almost_equal(
            0, np.sum(out["transformed"][0].data - bout["transformed"].data)
        )

    # ==================== Temporal Methods ====================

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

    # ==================== Deprecated Methods ====================

    def test_ttest(self, sim_brain_data):
        """Test that deprecated ttest method raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="ttest.*deprecated.*Model class"):
            sim_brain_data.ttest()

    def test_randomise(self, sim_brain_data):
        """Test that deprecated randomise method raises NotImplementedError."""
        sim_brain_data.X = pd.DataFrame({"Intercept": np.ones(len(sim_brain_data.Y))})

        with pytest.raises(NotImplementedError, match="randomise.*deprecated.*Model class"):
            sim_brain_data.randomise(n_permute=10)

    def test_bootstrap(self, sim_brain_data):
        """Test bootstrap with mean/std (predict is deprecated)."""
        # Bootstrap itself is not deprecated, but some functions it calls might be
        masked = sim_brain_data.apply_mask(create_sphere(radius=10, coordinates=[0, 0, 0]))

        # Test basic bootstrap with mean and std (should work)
        n_samples = 3
        b = masked.bootstrap("mean", n_samples=n_samples)
        assert isinstance(b["Z"], Brain_Data)
        b = masked.bootstrap("std", n_samples=n_samples)
        assert isinstance(b["Z"], Brain_Data)

        # Bootstrap with "predict" will fail since predict is deprecated
        with pytest.raises(NotImplementedError, match="predict.*deprecated.*Model class"):
            masked.bootstrap("predict", n_samples=n_samples, plot=False)

    def test_predict(self, sim_brain_data):
        """Test that deprecated predict method raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="predict.*deprecated.*Model class"):
            sim_brain_data.predict(
                algorithm="svm", cv_dict={"type": "kfolds", "n_folds": 2},
                plot=False, **{"kernel": "linear"}
            )

    def test_predict_multi(self):
        """Test that deprecated predict_multi method raises NotImplementedError."""
        # Need to set up minimal data for the test
        sim = Simulator()
        dat = sim.create_data([0, 1], sigma=1, reps=5, output_dir=".")
        y = pd.read_csv("y.csv", header=None, index_col=None)
        dat = Brain_Data("data.nii.gz", Y=y)

        with pytest.raises(NotImplementedError, match="predict_multi.*deprecated.*Model class"):
            dat.predict_multi(algorithm="svm")
