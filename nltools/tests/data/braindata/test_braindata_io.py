import os

import numpy as np
import nibabel as nib
import polars as pl
import pytest

from nltools.data import BrainData
from nltools.templates import get_brainspace


class TestBrainDataIO:
    @pytest.mark.slow
    def test_load(self, tmpdir):
        """Test loading BrainData from various sources and formats."""
        from nltools.data.simulator import Simulator
        from pathlib import Path
        import nibabel as nb

        sim = Simulator()
        sigma = 1
        y = [0, 1]
        n_reps = 3
        output_dir = str(tmpdir)
        dat = sim.create_data(y, sigma, reps=n_reps, output_dir=output_dir)

        # Shape depends on current brain-space resolution
        # 2mm: shape_3d = (91, 109, 91), shape_2d = (6, 238955)
        # 3mm: shape_3d = (60, 72, 60), shape_2d = (6, 71020)

        y = pl.read_csv(os.path.join(str(tmpdir.join("y.csv"))), has_header=False)

        # Test load list of 4D images
        file_list = [str(tmpdir.join("data.nii.gz")), str(tmpdir.join("data.nii.gz"))]
        dat = BrainData(file_list)
        dat = BrainData([nb.load(x) for x in file_list])

        # Test load string and path
        dat = BrainData(data=str(tmpdir.join("data.nii.gz")), Y=y)
        dat = BrainData(data=Path(tmpdir.join("data.nii.gz")), Y=y)

        # Test Write
        dat.write(os.path.join(str(tmpdir.join("test_write.nii"))))
        assert BrainData(os.path.join(str(tmpdir.join("test_write.nii"))))

        # Test i/o for hdf5
        dat.write(os.path.join(str(tmpdir.join("test_write.h5"))))
        b = BrainData(os.path.join(tmpdir.join("test_write.h5")))
        # Note: X and Y attributes removed in v0.6.0, skip checking them
        for k in ["mask", "data"]:
            if k == "data":
                assert np.allclose(b.__dict__[k], dat.__dict__[k])
            elif k == "mask":
                assert np.allclose(b.__dict__[k].affine, dat.__dict__[k].affine)
                assert np.allclose(
                    b.__dict__[k].get_fdata(), dat.__dict__[k].get_fdata()
                )
                assert b.__dict__[k].get_filename() == dat.__dict__[k].get_filename()
        # Test situation where we present a user warning when they're trying to load an .h5
        # file that includes a mask AND they pass in value for the mask argument. In this
        # case the mask argument takes precedence so we warn the user
        with pytest.warns(UserWarning):
            bb = BrainData(
                os.path.join(tmpdir.join("test_write.h5")), mask=get_brainspace().mask
            )
            assert os.path.abspath(bb.mask.get_filename()) == os.path.abspath(
                get_brainspace().mask
            )

    def test_h5_roundtrip_x_y(self, tmpdir):
        """X and Y polars frames survive h5 round-trip."""
        mask_img = nib.load(get_brainspace().mask)
        source_data = nib.Nifti1Image(
            np.random.randn(*mask_img.shape + (4,)), affine=mask_img.affine
        )
        X = pl.DataFrame({"cond": [1.0, 2.0, 3.0, 4.0], "drift": [0.0, 0.1, 0.2, 0.3]})
        Y = pl.DataFrame({"outcome": [0.0, 1.0, 0.0, 1.0]})
        brain = BrainData(source_data, X=X, Y=Y)

        path = str(tmpdir.join("rt.h5"))
        brain.write(path)
        loaded = BrainData(path)

        assert loaded.X.columns == ["cond", "drift"]
        assert loaded.Y.columns == ["outcome"]
        assert np.allclose(loaded.X.to_numpy(), X.to_numpy())
        assert np.allclose(loaded.Y.to_numpy(), Y.to_numpy())

    # ==================== Resampling Methods ====================

    def test_resample_to_img_nibabel(self):
        """Test resampling to target nibabel image."""

        # Create source BrainData (3mm) - need custom mask in 3mm space
        source_data = nib.Nifti1Image(
            np.random.randn(60, 72, 60, 10), affine=np.eye(4) * 3
        )
        # Create 3mm mask matching the data
        mask_3mm = nib.Nifti1Image(
            np.ones((60, 72, 60), dtype=np.float32), affine=np.eye(4) * 3
        )
        brain_source = BrainData(source_data, mask=mask_3mm, resample=False)

        # Create target image (2mm, same as MNI template)
        mask_img = nib.load(get_brainspace().mask)

        # Resample
        brain_resampled = brain_source.resample_to(img=mask_img)

        # Should match target space
        assert brain_resampled.shape[1] == 238955  # 2mm voxel count
        assert np.allclose(brain_resampled.mask.affine, mask_img.affine, rtol=1e-2)
        assert (
            brain_resampled.shape[0] == brain_source.shape[0]
        )  # Same number of images

    def test_resample_to_img_filepath(self, tmpdir):
        """Test resampling to target image from file path."""

        # Create source BrainData (3mm) - need custom mask in 3mm space
        source_data = nib.Nifti1Image(
            np.random.randn(60, 72, 60, 10), affine=np.eye(4) * 3
        )
        # Create 3mm mask matching the data
        mask_3mm = nib.Nifti1Image(
            np.ones((60, 72, 60), dtype=np.float32), affine=np.eye(4) * 3
        )
        brain_source = BrainData(source_data, mask=mask_3mm, resample=False)

        # Save target image to file
        target_path = str(tmpdir.join("target.nii.gz"))
        mask_img = nib.load(get_brainspace().mask)
        mask_img.to_filename(target_path)

        # Resample
        brain_resampled = brain_source.resample_to(img=target_path)

        # Should match target space
        assert brain_resampled.shape[1] == 238955
        assert brain_resampled.shape[0] == brain_source.shape[0]

    def test_resample_to_resolution_isotropic(self):
        """Test resampling to specified isotropic resolution."""

        # Create source BrainData (2mm)
        mask_img = nib.load(get_brainspace().mask)
        source_data = nib.Nifti1Image(
            np.random.randn(*mask_img.shape + (10,)), affine=mask_img.affine
        )
        brain_source = BrainData(source_data, resample=False)

        # Resample to 3mm
        brain_resampled = brain_source.resample_to(resolution=3.0)

        # Should have different voxel count
        assert brain_resampled.shape[1] != brain_source.shape[1]
        # Check resolution is approximately 3mm
        resampled_nifti = brain_resampled.to_nifti()
        zooms = resampled_nifti.header.get_zooms()[:3]
        assert np.allclose(zooms, 3.0, rtol=0.1)
        assert (
            brain_resampled.shape[0] == brain_source.shape[0]
        )  # Same number of images

    def test_resample_to_both_params_error(self):
        """Test error when both img and resolution are provided."""

        # Create BrainData with matching mask
        mask_img = nib.load(get_brainspace().mask)
        source_data = nib.Nifti1Image(
            np.random.randn(*mask_img.shape), affine=mask_img.affine
        )
        brain = BrainData(source_data, resample=False)

        with pytest.raises(ValueError, match="both.*img.*and.*resolution"):
            brain.resample_to(img=mask_img, resolution=2.0)

    def test_resample_to_no_params_error(self):
        """Test error when neither img nor resolution is provided."""

        # Create BrainData with matching mask
        mask_img = nib.load(get_brainspace().mask)
        source_data = nib.Nifti1Image(
            np.random.randn(*mask_img.shape), affine=mask_img.affine
        )
        brain = BrainData(source_data, resample=False)

        with pytest.raises(ValueError, match="either.*img.*or.*resolution"):
            brain.resample_to()

    def test_resample_to_invalid_img_type(self):
        """Test error with invalid img type."""

        # Create BrainData with matching mask
        mask_img = nib.load(get_brainspace().mask)
        source_data = nib.Nifti1Image(
            np.random.randn(*mask_img.shape), affine=mask_img.affine
        )
        brain = BrainData(source_data, resample=False)

        with pytest.raises(TypeError, match="img.*must be"):
            brain.resample_to(img=123)  # Invalid type

    def test_resample_to_preserves_metadata(self):
        """Test that X and Y metadata are preserved after resampling."""

        # Create BrainData with matching mask
        mask_img = nib.load(get_brainspace().mask)
        source_data = nib.Nifti1Image(
            np.random.randn(*mask_img.shape + (5,)), affine=mask_img.affine
        )
        X = pl.DataFrame({"cond1": [1, 2, 3, 4, 5]})
        Y = pl.DataFrame({"outcome": [0, 1, 0, 1, 0]})

        brain_source = BrainData(source_data, X=X, Y=Y, resample=False)
        brain_resampled = brain_source.resample_to(resolution=3.0)

        # Metadata should be preserved
        assert brain_resampled.X.equals(brain_source.X)
        assert brain_resampled.Y.equals(brain_source.Y)
        assert (
            brain_resampled.shape[0] == brain_source.shape[0]
        )  # Same number of images

    def test_resample_to_does_not_mutate_caller_headers(self):
        """resample_to must not touch bd.mask or a caller-supplied target img."""
        # Build a source with sform_code=0 on its mask
        mask_3mm = nib.Nifti1Image(
            np.ones((60, 72, 60), dtype=np.float32), affine=np.eye(4) * 3
        )
        mask_3mm.header.set_sform(mask_3mm.affine, code=0)
        source_data = nib.Nifti1Image(
            np.random.randn(60, 72, 60, 4), affine=np.eye(4) * 3
        )
        brain = BrainData(source_data, mask=mask_3mm, resample=False)
        assert brain.mask.header.get_sform(coded=True)[1] == 0

        # Resample by resolution — must not mutate brain.mask
        brain.resample_to(resolution=4.0)
        assert brain.mask.header.get_sform(coded=True)[1] == 0

        # Resample by target img — must not mutate the caller's target
        target = nib.Nifti1Image(
            np.ones((45, 54, 45), dtype=np.float32), affine=np.eye(4) * 4
        )
        target.header.set_sform(target.affine, code=0)
        brain.resample_to(img=target)
        assert target.header.get_sform(coded=True)[1] == 0

    def test_resample_to_same_space_identity(self):
        """Test resampling to same space produces similar results."""

        # Create BrainData
        mask_img = nib.load(get_brainspace().mask)
        source_data = nib.Nifti1Image(
            np.random.randn(*mask_img.shape + (5,)), affine=mask_img.affine
        )
        brain_source = BrainData(source_data, resample=False)

        # Resample to same space (should be near-identical)
        brain_resampled = brain_source.resample_to(img=mask_img)

        # Data should be very similar (within interpolation tolerance)
        assert np.allclose(
            brain_source.data, brain_resampled.data, rtol=1e-3, atol=1e-3
        )

    # ==================== Interpolation Tests ====================

    def test_interpolation_auto_default(self):
        """Test that interpolation='auto' is the default."""
        brain = BrainData()
        assert brain._interpolation == "auto"

    def test_interpolation_explicit_nearest(self):
        """Test explicit interpolation='nearest' is stored."""
        brain = BrainData(interpolation="nearest")
        assert brain._interpolation == "nearest"

    def test_interpolation_explicit_continuous(self):
        """Test explicit interpolation='continuous' is stored."""
        brain = BrainData(interpolation="continuous")
        assert brain._interpolation == "continuous"

    def test_interpolation_invalid_raises_error(self):
        """Test invalid interpolation value raises ValueError."""
        with pytest.raises(ValueError, match="interpolation must be one of"):
            BrainData(interpolation="invalid")

    def test_interpolation_auto_detects_atlas(self, tmpdir):
        """Test auto-detection uses nearest for atlas/label data."""

        # Create atlas with discrete integer labels
        atlas_data = np.zeros((20, 20, 20))
        atlas_data[5:10, 5:10, 5:10] = 1
        atlas_data[10:15, 10:15, 10:15] = 2
        atlas_data[15:18, 15:18, 15:18] = 3

        # 3mm voxels to force resampling to 2mm template
        affine = np.eye(4) * 3
        affine[3, 3] = 1
        atlas_img = nib.Nifti1Image(atlas_data, affine)

        # Load with auto interpolation
        brain = BrainData(atlas_img, interpolation="auto")

        # Values should remain discrete (no interpolation artifacts)
        unique_vals = np.unique(brain.data)
        # With nearest interpolation, we should have only a few discrete values
        # (some may be 0 due to masking, but shouldn't have many intermediate values)
        assert len(unique_vals) < 10, (
            f"Expected discrete values, got {len(unique_vals)} unique values"
        )

    def test_interpolation_auto_detects_continuous(self, tmpdir):
        """Test auto-detection uses continuous for statistical maps."""

        # Create continuous statistical data
        stat_data = np.random.randn(20, 20, 20)

        # 3mm voxels
        affine = np.eye(4) * 3
        affine[3, 3] = 1
        stat_img = nib.Nifti1Image(stat_data, affine)

        # Load with auto interpolation
        brain = BrainData(stat_img, interpolation="auto")

        # Continuous data should have many unique values
        unique_vals = np.unique(brain.data)
        assert len(unique_vals) > 100, (
            f"Expected many unique values, got {len(unique_vals)}"
        )

    def test_resample_to_respects_interpolation(self):
        """Test resample_to uses instance interpolation setting."""

        # Create atlas-like source data with explicit nearest
        atlas_data = np.zeros((60, 72, 60))
        atlas_data[20:40, 20:50, 20:40] = 1
        atlas_data[30:50, 30:60, 30:50] = 2

        affine = np.eye(4) * 3
        affine[3, 3] = 1
        atlas_img = nib.Nifti1Image(atlas_data, affine)
        mask_3mm = nib.Nifti1Image(
            np.ones((60, 72, 60), dtype=np.float32), affine=affine
        )

        # Load with explicit nearest interpolation
        brain = BrainData(
            atlas_img, mask=mask_3mm, interpolation="nearest", resample=False
        )

        # Resample to 2mm template
        mask_2mm = nib.load(get_brainspace().mask)
        brain_resampled = brain.resample_to(img=mask_2mm)

        # Values should still be discrete after resampling
        unique_vals = np.unique(brain_resampled.data)
        assert len(unique_vals) < 10, (
            f"Expected discrete values after resample, got {len(unique_vals)}"
        )
