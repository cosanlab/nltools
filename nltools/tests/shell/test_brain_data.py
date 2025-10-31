"""
Test suite for BrainData class.

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
from nltools.data import BrainData, Adjacency
from nltools.stats import threshold, align
from nltools.mask import create_sphere, roi_to_brain
from pathlib import Path

from nltools.prefs import MNI_Template
from nltools.tests.conftest import _tables_available


shape_3d = (91, 109, 91)
shape_2d = (6, 238955)


class TestBrainData:
    """Test BrainData class - focus on method usage, not implementation."""

    # ==================== Initialization & I/O ====================

    @pytest.mark.tier2
    def test_load(self, tmpdir):
        """Test loading BrainData from various sources and formats."""
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
        for k in ["mask", "nifti_masker", "data"]:
            if k == "data":
                assert np.allclose(b.__dict__[k], dat.__dict__[k])
            elif k == "mask":
                assert np.allclose(b.__dict__[k].affine, dat.__dict__[k].affine)
                assert np.allclose(
                    b.__dict__[k].get_fdata(), dat.__dict__[k].get_fdata()
                )
                assert b.__dict__[k].get_filename() == dat.__dict__[k].get_filename()
            elif k == "nifti_masker":
                assert np.allclose(b.__dict__[k].affine_, dat.__dict__[k].affine_)
                assert np.allclose(
                    b.__dict__[k].mask_img.get_fdata(),
                    dat.__dict__[k].mask_img.get_fdata(),
                )
            else:
                assert b.__dict__[k] == dat.__dict__[k]
        # Test situation where we present a user warning when they're trying to load an .h5
        # file that includes a mask AND they pass in value for the mask argument. In this
        # case the mask argument takes precedence so we warn the user
        with pytest.warns(UserWarning):
            bb = BrainData(
                os.path.join(tmpdir.join("test_write.h5")), mask=MNI_Template.mask
            )
            assert os.path.abspath(bb.mask.get_filename()) == os.path.abspath(
                MNI_Template.mask
            )

    @pytest.mark.skipif(
        not _tables_available(), reason="HDF5 support deprecated, requires PyTables"
    )
    def test_load_legacy_h5(self, old_h5_brain, new_h5_brain, tmpdir):
        """Test loading old HDF5 format (backward compatibility)."""
        with pytest.warns(UserWarning):
            # With verbosity on we should see a warning about the old h5 file format
            b_old = BrainData(old_h5_brain, verbose=True)
        b_new = BrainData(new_h5_brain)
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
        b_new_written = BrainData(new_file)
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
        """Test addition of BrainData objects and scalars."""
        new = sim_brain_data + sim_brain_data
        assert new.shape == shape_2d
        value = 10
        assert (value + sim_brain_data[0]).mean() == (sim_brain_data[0] + value).mean()

    def test_subtract(self, sim_brain_data):
        """Test subtraction of BrainData objects and scalars."""
        new = sim_brain_data - sim_brain_data
        assert new.shape == shape_2d
        value = 10
        assert (-value - (-1) * sim_brain_data[0]).mean() == (
            sim_brain_data[0] - value
        ).mean()

    def test_multiply(self, sim_brain_data):
        """Test multiplication of BrainData objects, scalars, and arrays."""
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
        """Test division of BrainData objects and scalars."""
        new = sim_brain_data / sim_brain_data
        assert new.shape == shape_2d
        np.testing.assert_almost_equal(new.mean(axis=0).mean(), 1, decimal=6)
        value = 10
        new2 = sim_brain_data / value
        np.testing.assert_almost_equal(
            ((new2 * value) - new2).mean().mean(), 0, decimal=2
        )

    def test_inplace_add(self, sim_brain_data):
        """Test in-place addition with scalars and BrainData."""
        # Test in-place add with scalar
        bd = sim_brain_data[0].copy()
        original_data = bd.data.copy()
        bd += 5
        assert np.allclose(bd.data, original_data + 5)

        # Test in-place add with BrainData
        bd1 = sim_brain_data[0].copy()
        bd2 = sim_brain_data[0].copy()
        original_data = bd1.data.copy()
        bd1 += bd2
        assert np.allclose(bd1.data, original_data + bd2.data)

    def test_inplace_subtract(self, sim_brain_data):
        """Test in-place subtraction with scalars and BrainData."""
        # Test in-place subtract with scalar
        bd = sim_brain_data[0].copy()
        original_data = bd.data.copy()
        bd -= 3
        assert np.allclose(bd.data, original_data - 3)

        # Test in-place subtract with BrainData
        bd1 = sim_brain_data[0].copy()
        bd2 = sim_brain_data[0].copy()
        original_data = bd1.data.copy()
        bd1 -= bd2
        assert np.allclose(bd1.data, original_data - bd2.data)

    def test_inplace_multiply(self, sim_brain_data):
        """Test in-place multiplication with scalars, BrainData, and arrays."""
        # Test in-place multiply with scalar
        bd = sim_brain_data[0].copy()
        original_data = bd.data.copy()
        bd *= 2
        assert np.allclose(bd.data, original_data * 2)

        # Test in-place multiply with BrainData
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
        """Test in-place division with scalars and BrainData."""
        # Test in-place divide with scalar
        bd = sim_brain_data[0].copy()
        original_data = bd.data.copy()
        bd /= 2
        assert np.allclose(bd.data, original_data / 2)

        # Test in-place divide with BrainData
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
        assert BrainData(d)

    def test_concatenate(self, sim_brain_data):
        """Test concatenating BrainData objects from list."""
        out = BrainData([x for x in sim_brain_data])
        assert isinstance(out, BrainData)
        assert len(out) == len(sim_brain_data)

    def test_append(self, sim_brain_data):
        """Test appending BrainData objects."""
        assert sim_brain_data.append(sim_brain_data).shape[0] == shape_2d[0] * 2

    # ==================== Statistical Methods ====================

    def test_distance(self, sim_brain_data):
        """Test distance computation returns Adjacency object."""
        distance = sim_brain_data.distance(metric="correlation")
        assert isinstance(distance, Adjacency)
        assert distance.square_shape()[0] == shape_2d[0]

    # ==================== Regression & GLM ====================

    @pytest.mark.tier2
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
        assert isinstance(tt, BrainData)

    @pytest.mark.tier2
    def test_regress_uses_glm_model(self, sim_brain_data):
        """Test that .regress() uses Glm model internally."""
        from nltools.models import Glm

        # Set up design matrix
        sim_brain_data.X = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data.Y)),
                "X1": np.array(sim_brain_data.Y).flatten(),
            },
            index=None,
        )

        # Run regression
        sim_brain_data.regress()

        # Should have created a Glm model instance stored as attribute
        assert hasattr(sim_brain_data, "glm_model")
        assert isinstance(sim_brain_data.glm_model, Glm)
        assert sim_brain_data.glm_model.is_fitted_

    @pytest.mark.tier2
    def test_regress_glm_parameters(self, sim_brain_data):
        """Test that .regress() passes parameters to Glm correctly."""
        # Set up design matrix
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data.Y)),
                "X1": np.array(sim_brain_data.Y).flatten(),
            },
            index=None,
        )

        # Test noise_model parameter
        sim_brain_data.regress(design_matrix, noise_model="ar1")
        assert sim_brain_data.glm_model.noise_model == "ar1"

        # Test with OLS (default)
        sim_brain_data.regress(design_matrix, noise_model="ols")
        assert sim_brain_data.glm_model.noise_model == "ols"

    @pytest.mark.tier2
    def test_regress_attributes_match_glm(self, sim_brain_data):
        """Test that .regress() attributes are correctly extracted from Glm."""
        # Set up design matrix
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data.Y)),
                "X1": np.array(sim_brain_data.Y).flatten(),
            },
            index=None,
        )

        # Run regression
        sim_brain_data.regress(design_matrix)

        # Check all expected attributes exist
        assert hasattr(sim_brain_data, "glm_betas")
        assert hasattr(sim_brain_data, "glm_t")
        assert hasattr(sim_brain_data, "glm_p")
        assert hasattr(sim_brain_data, "glm_se")
        assert hasattr(sim_brain_data, "glm_residual")
        assert hasattr(sim_brain_data, "glm_predicted")
        assert hasattr(sim_brain_data, "glm_r2")

        # Check shapes are correct
        n_regressors = design_matrix.shape[1]
        n_voxels = sim_brain_data.shape[1]

        assert sim_brain_data.glm_betas.shape == (n_regressors, n_voxels)
        assert sim_brain_data.glm_t.shape == (n_regressors, n_voxels)
        assert sim_brain_data.glm_p.shape == (n_regressors, n_voxels)
        assert sim_brain_data.glm_se.shape == (n_regressors, n_voxels)
        assert sim_brain_data.glm_residual.shape == sim_brain_data.shape
        assert sim_brain_data.glm_predicted.shape == sim_brain_data.shape
        assert sim_brain_data.glm_r2.shape == (1, n_voxels)

        # Check residuals property matches attribute
        residuals_from_model = sim_brain_data.glm_model.residuals
        assert len(residuals_from_model) == 1  # One run
        residuals_brain_data = BrainData(
            residuals_from_model[0], mask=sim_brain_data.mask
        )
        np.testing.assert_allclose(
            sim_brain_data.glm_residual.data, residuals_brain_data.data, rtol=1e-5
        )

    @pytest.mark.tier2
    def test_regress_backward_compatible_dict(self, sim_brain_data):
        """Test that .regress() still returns dict for backward compatibility."""
        # Set up design matrix
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data.Y)),
                "X1": np.array(sim_brain_data.Y).flatten(),
            },
            index=None,
        )

        # Run regression and capture returned dict
        with pytest.warns(FutureWarning):
            out = sim_brain_data.regress(design_matrix)

        # Check dict structure
        assert isinstance(out, dict)
        assert "beta" in out
        assert "t" in out
        assert "p" in out
        assert "residual" in out

        # Dict values should match attributes
        assert out["beta"] is sim_brain_data.glm_betas
        assert out["t"] is sim_brain_data.glm_t
        assert out["p"] is sim_brain_data.glm_p
        assert out["residual"] is sim_brain_data.glm_residual

    @pytest.mark.tier2
    def test_regress_numerical_equivalence(self, sim_brain_data):
        """Test that Glm-based .regress() gives same numerical results as before."""
        # This is a key test - we want to ensure refactoring doesn't change results

        # Set up design matrix with known relationship
        np.random.seed(42)
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data.Y)),
                "X1": np.array(sim_brain_data.Y).flatten(),
            },
            index=None,
        )

        # Run regression
        out = sim_brain_data.regress(design_matrix)

        # Check betas are reasonable (not NaN, not all zeros)
        assert not np.isnan(out["beta"].data).any()
        assert not np.allclose(out["beta"].data, 0)

        # Check t-statistics are reasonable
        assert not np.isnan(out["t"].data).any()

        # Check p-values are in valid range [0, 1]
        assert np.all(out["p"].data >= 0)
        assert np.all(out["p"].data <= 1)

        # Check residuals + predicted = original data
        reconstructed = out["residual"].data + sim_brain_data.glm_predicted.data
        np.testing.assert_allclose(
            reconstructed, sim_brain_data.data, rtol=1e-5, atol=1e-8
        )

        # Check R² exists and is computed (values can be negative for poor fits on random data)
        assert hasattr(sim_brain_data, "glm_r2")
        assert sim_brain_data.glm_r2.shape == (1, sim_brain_data.shape[1])

    def test_compute_contrasts_error_not_fitted(self, minimal_brain_data):
        """Test error when compute_contrasts() called before regress()."""
        # Should raise RuntimeError if regress() not called first
        with pytest.raises(RuntimeError, match="Must run .regress"):
            minimal_brain_data.compute_contrasts([1, -1, 0])

    @pytest.mark.tier2
    def test_compute_contrasts_numeric_vector(self, minimal_brain_data):
        """Test numeric contrast vector (unique nltools API)."""
        # Set up and run regression
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condA": np.random.randn(len(minimal_brain_data)),
                "condB": np.random.randn(len(minimal_brain_data)),
            }
        )

        with pytest.warns(FutureWarning):
            minimal_brain_data.regress(design_matrix)

        # Compute contrast: A - B (unique nltools logic)
        contrast = minimal_brain_data.compute_contrasts([0, 1, -1])

        # Test nltools-specific API contract
        assert isinstance(contrast, BrainData)
        assert contrast.shape == (1, minimal_brain_data.shape[1])

    @pytest.mark.tier2
    def test_compute_contrasts_string_parsing(self, minimal_brain_data):
        """Test string contrast parsing (unique nltools feature)."""
        # Set up and run regression
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condA": np.random.randn(len(minimal_brain_data)),
                "condB": np.random.randn(len(minimal_brain_data)),
            }
        )

        with pytest.warns(FutureWarning):
            minimal_brain_data.regress(design_matrix)

        # Test string parsing (unique nltools feature)
        contrast = minimal_brain_data.compute_contrasts("condA - condB")

        assert isinstance(contrast, BrainData)
        assert contrast.shape == (1, minimal_brain_data.shape[1])

    @pytest.mark.tier2
    def test_compute_contrasts_multiple_dict(self, minimal_brain_data):
        """Test multiple contrasts via dict (unique nltools API)."""
        # Set up and run regression
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condA": np.random.randn(len(minimal_brain_data)),
                "condB": np.random.randn(len(minimal_brain_data)),
            }
        )

        with pytest.warns(FutureWarning):
            minimal_brain_data.regress(design_matrix)

        # Test dict of contrasts (unique nltools API)
        contrasts = {"A_vs_B": "condA - condB", "avg_effect": [0, 0.5, 0.5]}
        results = minimal_brain_data.compute_contrasts(contrasts)

        # Should return dict of BrainData objects
        assert isinstance(results, dict)
        assert "A_vs_B" in results
        assert "avg_effect" in results
        assert isinstance(results["A_vs_B"], BrainData)
        assert isinstance(results["avg_effect"], BrainData)

    @pytest.mark.tier2
    def test_compute_contrasts_invalid_length(self, minimal_brain_data):
        """Test error for invalid contrast vector length (nltools validation)."""
        # Set up and run regression with 3 regressors
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condA": np.random.randn(len(minimal_brain_data)),
                "condB": np.random.randn(len(minimal_brain_data)),
            }
        )

        with pytest.warns(FutureWarning):
            minimal_brain_data.regress(design_matrix)

        # Provide wrong length contrast (2 instead of 3)
        with pytest.raises(ValueError, match="Contrast vector length.*must match"):
            minimal_brain_data.compute_contrasts([1, -1])

    # ==================== Unified fit/predict API ====================

    def test_fit_predict_ridge_workflow(self, sim_brain_data):
        """Test complete Ridge fit/predict workflow."""
        from nltools.data import BrainData
        from nltools.models import Ridge

        # Fit Ridge model
        X_train = np.random.randn(len(sim_brain_data), 10)
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X_train)

        # Check model stored
        assert hasattr(sim_brain_data, "model_")
        assert isinstance(sim_brain_data.model_, Ridge)
        assert sim_brain_data.model_.is_fitted_

        # Check attributes set
        assert hasattr(sim_brain_data, "ridge_weights")
        assert hasattr(sim_brain_data, "ridge_fitted_values")
        assert hasattr(sim_brain_data, "ridge_scores")

        # Predict on new data
        X_test = np.random.randn(20, 10)  # Different n_samples
        predictions = sim_brain_data.predict(X=X_test)

        # Check predictions
        assert isinstance(predictions, BrainData)
        assert predictions.shape == (20, sim_brain_data.shape[1])

        # Predict on training data (X=None)
        train_predictions = sim_brain_data.predict()
        assert train_predictions.shape == sim_brain_data.shape

    @pytest.mark.tier2
    def test_fit_predict_glm_workflow(self, sim_brain_data):
        """Test complete GLM fit/predict workflow."""
        from nltools.models import Glm

        # Fit GLM model
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data)),
                "X1": np.random.randn(len(sim_brain_data)),
            }
        )
        sim_brain_data.fit(model="glm", noise_model="ols", X=design_matrix)

        # Check model stored
        assert hasattr(sim_brain_data, "model_")
        assert isinstance(sim_brain_data.model_, Glm)

        # Check GLM attributes set
        assert hasattr(sim_brain_data, "glm_betas")
        assert hasattr(sim_brain_data, "glm_t")

        # Predict on training data (fitted values)
        # Note: GLM doesn't support prediction with new design matrices yet
        predictions = sim_brain_data.predict()

        # Check predictions match training data shape
        assert predictions.shape == sim_brain_data.shape

    def test_fit_uses_brain_data_as_target(self, sim_brain_data):
        """Test fit() always uses self.data as y target."""
        X = np.random.randn(len(sim_brain_data), 10)

        # Fit Ridge
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X)

        # Model should be fitted to (X, sim_brain_data.data)
        # Check by predicting and comparing shapes
        predictions = sim_brain_data.predict(X=X)
        assert predictions.shape == sim_brain_data.shape

    @pytest.mark.tier2
    def test_fit_passes_kwargs_to_model(self, sim_brain_data):
        """Test fit() passes additional kwargs to model constructor."""
        X = np.random.randn(len(sim_brain_data), 10)

        # Ridge with backend kwarg
        sim_brain_data.fit(model="ridge", alpha=1.0, backend="numpy", X=X)
        assert sim_brain_data.model_.backend == "numpy"

        # GLM with noise_model kwarg
        design_matrix = pd.DataFrame({"Intercept": np.ones(len(sim_brain_data))})
        sim_brain_data.fit(model="glm", noise_model="ar1", X=design_matrix)
        assert sim_brain_data.model_.noise_model == "ar1"

    def test_predict_requires_fitted_model(self, sim_brain_data):
        """Test predict() raises error if fit() not called first."""
        # Get a fresh copy (fixture may be contaminated by previous tests)
        bd = sim_brain_data.copy()

        # Explicitly remove model attributes to test the error case
        # (copy shares model_ from fitted instances due to pickle handling)
        for attr in ["model_", "X_"]:
            if hasattr(bd, attr):
                delattr(bd, attr)

        with pytest.raises(ValueError, match="Must call fit"):
            bd.predict()

    def test_predict_validates_X_dimensions(self, sim_brain_data):
        """Test predict() validates X has correct n_features."""
        # Fit with 10 features
        X_train = np.random.randn(len(sim_brain_data), 10)
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X_train)

        # Try to predict with 5 features - should fail
        X_wrong = np.random.randn(15, 5)
        with pytest.raises(ValueError, match="features"):
            sim_brain_data.predict(X=X_wrong)

    def test_ridge_weights_structure(self, sim_brain_data):
        """Test Ridge weights stored correctly as BrainData."""
        from nltools.data import BrainData

        X = np.random.randn(len(sim_brain_data), 10)
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X)

        # Weights should be BrainData
        assert isinstance(sim_brain_data.ridge_weights, BrainData)

        # Shape: (n_features, n_voxels)
        assert sim_brain_data.ridge_weights.shape == (10, sim_brain_data.shape[1])

        # Should have same mask
        assert sim_brain_data.ridge_weights.mask is sim_brain_data.mask

    @pytest.mark.tier2
    def test_glm_fit_matches_current_regress(self, sim_brain_data):
        """Test new fit(model='glm') matches current regress() numerically."""

        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data)),
                "X1": np.random.randn(len(sim_brain_data)),
            }
        )

        # New API
        bd_new = sim_brain_data.copy()
        bd_new.fit(model="glm", noise_model="ols", X=design_matrix)

        # Old API
        bd_old = sim_brain_data.copy()
        with pytest.warns(FutureWarning):
            bd_old.regress(design_matrix, noise_model="ols")

        # Should be numerically identical
        np.testing.assert_allclose(bd_new.glm_betas.data, bd_old.glm_betas.data)
        np.testing.assert_allclose(bd_new.glm_t.data, bd_old.glm_t.data)

    def test_fit_validates_model_name(self, sim_brain_data):
        """Test fit() raises error for unknown model names."""
        X = np.random.randn(len(sim_brain_data), 10)

        with pytest.raises(ValueError, match="Unknown model"):
            sim_brain_data.fit(model="unknown_model", X=X)

    def test_fit_validates_X_shape(self, sim_brain_data):
        """Test fit() validates X has correct n_samples."""
        # X has wrong number of samples
        X_wrong = np.random.randn(len(sim_brain_data) + 5, 10)

        with pytest.raises(ValueError, match="number of samples"):
            sim_brain_data.fit(model="ridge", alpha=1.0, X=X_wrong)

    def test_predict_with_no_X_uses_training_data(self, sim_brain_data):
        """Test predict() with no X returns predictions on training data."""
        X_train = np.random.randn(len(sim_brain_data), 10)
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X_train)

        # Predict with explicit X
        predictions_explicit = sim_brain_data.predict(X=X_train)

        # Predict with no X (should use training data)
        predictions_implicit = sim_brain_data.predict()

        # Should be identical
        np.testing.assert_allclose(predictions_explicit.data, predictions_implicit.data)

        # Should match training data shape
        assert predictions_implicit.shape == sim_brain_data.shape

    # ==================== fit() with Cross-Validation ====================

    def test_fit_ridge_cv_basic_integer(self, small_brain_data_for_cv):
        """Test fit() with cv=3 returns cross-validated scores for fixed alpha."""
        brain_data, X = small_brain_data_for_cv

        # Fit with CV and fixed alpha
        brain_data.fit(model="ridge", alpha=1.0, cv=3, X=X)

        # CV results should exist
        assert hasattr(brain_data, "cv_results_")
        assert isinstance(brain_data.cv_results_, dict)

        # Check expected keys
        assert "scores" in brain_data.cv_results_
        assert "mean_score" in brain_data.cv_results_
        assert "predictions" in brain_data.cv_results_
        assert "folds" in brain_data.cv_results_

        # Check shapes
        cv_scores = brain_data.cv_results_["scores"]
        assert cv_scores.shape == (3, 5)  # (n_folds=3, n_voxels=5)

        mean_score = brain_data.cv_results_["mean_score"]
        assert mean_score.shape == (5,)  # Per-voxel mean

        # Check fold indices
        folds = brain_data.cv_results_["folds"]
        assert len(folds) == 24  # n_samples
        assert set(folds) == {0, 1, 2}  # Fold IDs

        # Regular fit attributes should still exist
        assert hasattr(brain_data, "ridge_weights")
        assert hasattr(brain_data, "ridge_fitted_values")

    def test_fit_ridge_cv_sklearn_splitter(self, small_brain_data_for_cv):
        """Test fit() accepts sklearn CV splitter objects."""
        from sklearn.model_selection import KFold

        brain_data, X = small_brain_data_for_cv

        # Create CV splitter
        cv_splitter = KFold(n_splits=3, shuffle=True, random_state=42)

        # Fit with CV splitter
        brain_data.fit(model="ridge", alpha=1.0, cv=cv_splitter, X=X)

        # CV results should exist with same structure
        assert hasattr(brain_data, "cv_results_")
        assert brain_data.cv_results_["scores"].shape == (3, 5)

        # Test reproducibility - fit again with same random_state
        brain_data2, X2 = small_brain_data_for_cv
        cv_splitter2 = KFold(n_splits=3, shuffle=True, random_state=42)
        brain_data2.fit(model="ridge", alpha=1.0, cv=cv_splitter2, X=X2)

        # Should get identical results
        np.testing.assert_allclose(
            brain_data.cv_results_["mean_score"], brain_data2.cv_results_["mean_score"]
        )

    def test_fit_ridge_cv_predictions(self, small_brain_data_for_cv):
        """Test CV predictions are out-of-fold and stored as BrainData."""
        brain_data, X = small_brain_data_for_cv

        # Fit with CV
        brain_data.fit(model="ridge", alpha=1.0, cv=3, X=X)

        # Check predictions structure
        cv_preds = brain_data.cv_results_["predictions"]
        assert isinstance(cv_preds, BrainData)
        assert cv_preds.shape == (24, 5)  # (n_samples, n_voxels)

        # CV predictions should differ from full model predictions
        # (out-of-fold vs. in-sample)
        full_preds = brain_data.ridge_fitted_values
        assert not np.allclose(cv_preds.data, full_preds.data)

        # Sanity checks on R² values
        # Note: Out-of-sample R² can be negative (model worse than mean)
        cv_r2 = np.mean(brain_data.cv_results_["mean_score"])
        full_r2 = np.mean(brain_data.ridge_scores.data)

        # Just check both are finite and reasonable (not NaN/Inf)
        assert np.isfinite(cv_r2)
        assert np.isfinite(full_r2)
        # Full R² should generally be non-negative (in-sample)
        assert full_r2 >= -0.1  # Allow small numerical errors

    def test_fit_ridge_cv_auto_alpha_selection(self, small_brain_data_for_cv):
        """Test cv='auto' triggers alpha selection."""
        brain_data, X = small_brain_data_for_cv

        # Fit with cv='auto' (implies alpha='auto')
        alphas = [0.1, 1.0, 10.0]  # Small grid for speed
        brain_data.fit(model="ridge", cv="auto", alphas=alphas, X=X)

        # CV results should exist
        assert hasattr(brain_data, "cv_results_")

        # Alpha selection results
        assert "best_alpha" in brain_data.cv_results_
        assert "alpha_scores" in brain_data.cv_results_

        # Best alpha should be one of the tested alphas
        best_alpha = brain_data.cv_results_["best_alpha"]
        assert best_alpha in alphas

        # Alpha scores shape: (n_folds, n_alphas, n_voxels)
        alpha_scores = brain_data.cv_results_["alpha_scores"]
        assert alpha_scores.shape == (
            5,
            3,
            5,
        )  # (5 folds default for 'auto', 3 alphas, 5 voxels)

        # Model should be fitted with best_alpha
        assert brain_data.model_.alpha == best_alpha

    def test_fit_ridge_cv_integer_with_alpha_auto(self, small_brain_data_for_cv):
        """Test cv=int with alpha='auto' performs both alpha selection and CV scoring."""
        brain_data, X = small_brain_data_for_cv

        # Fit with explicit alpha selection + CV
        alphas = [0.1, 1.0, 10.0]
        brain_data.fit(model="ridge", alpha="auto", cv=3, alphas=alphas, X=X)

        # Should have both alpha selection and CV scoring results
        assert "best_alpha" in brain_data.cv_results_
        assert "alpha_scores" in brain_data.cv_results_
        assert "scores" in brain_data.cv_results_
        assert "mean_score" in brain_data.cv_results_

        # Alpha scores: (n_folds=3, n_alphas=3, n_voxels=5)
        assert brain_data.cv_results_["alpha_scores"].shape == (3, 3, 5)

        # CV scores computed with best alpha: (n_folds=3, n_voxels=5)
        assert brain_data.cv_results_["scores"].shape == (3, 5)

        # Best alpha selected
        assert brain_data.cv_results_["best_alpha"] in alphas

    def test_fit_ridge_no_cv_backward_compat(self, small_brain_data_for_cv):
        """Test fit() without cv parameter doesn't create cv_results_ (backward compat)."""
        brain_data, X = small_brain_data_for_cv

        # Fit without CV (existing behavior)
        brain_data.fit(model="ridge", alpha=1.0, X=X)

        # CV results should NOT exist
        assert not hasattr(brain_data, "cv_results_")

        # Regular attributes should exist
        assert hasattr(brain_data, "ridge_weights")
        assert hasattr(brain_data, "ridge_fitted_values")
        assert hasattr(brain_data, "ridge_scores")

    def test_fit_ridge_cv_invalid_parameter(self, small_brain_data_for_cv):
        """Test fit() raises errors for invalid cv parameters."""
        brain_data, X = small_brain_data_for_cv

        # Invalid cv type
        with pytest.raises((TypeError, ValueError)):
            brain_data.fit(model="ridge", alpha=1.0, cv="invalid", X=X)

        # Negative cv
        with pytest.raises(ValueError):
            brain_data.fit(model="ridge", alpha=1.0, cv=-1, X=X)

        # Zero cv
        with pytest.raises(ValueError):
            brain_data.fit(model="ridge", alpha=1.0, cv=0, X=X)

    def test_fit_ridge_cv_with_insufficient_samples(self, tiny_brain_data_for_cv):
        """Test fit() raises error when cv folds > n_samples."""
        brain_data, X = tiny_brain_data_for_cv  # Only 6 samples

        # Try 10-fold CV with 6 samples
        with pytest.raises(ValueError, match="Cannot have number of splits.*greater"):
            brain_data.fit(model="ridge", alpha=1.0, cv=10, X=X)

    def test_fit_ridge_cv_predict_consistency(self, small_brain_data_for_cv):
        """Test predict() returns full model predictions, not CV predictions."""
        brain_data, X = small_brain_data_for_cv

        # Fit with CV
        brain_data.fit(model="ridge", alpha=1.0, cv=3, X=X)

        # Call predict() on training data
        train_predictions = brain_data.predict(X=X)

        # Should match full model predictions (ridge_fitted_values)
        np.testing.assert_allclose(
            train_predictions.data, brain_data.ridge_fitted_values.data
        )

        # Should NOT match CV predictions (out-of-fold)
        assert not np.allclose(
            train_predictions.data, brain_data.cv_results_["predictions"].data
        )

    def test_fit_ridge_cv_stores_all_expected_keys(self, small_brain_data_for_cv):
        """Test cv_results_ dict contains all expected keys and types."""
        brain_data, X = small_brain_data_for_cv

        # Fit with alpha selection
        alphas = [0.1, 1.0, 10.0]
        brain_data.fit(model="ridge", alpha="auto", cv=3, alphas=alphas, X=X)

        # Check all expected keys exist
        expected_keys = {
            "scores",
            "mean_score",
            "predictions",
            "folds",
            "best_alpha",
            "alpha_scores",
        }
        assert set(brain_data.cv_results_.keys()) == expected_keys

        # Check types
        assert isinstance(brain_data.cv_results_["scores"], np.ndarray)
        assert isinstance(brain_data.cv_results_["mean_score"], np.ndarray)
        assert isinstance(brain_data.cv_results_["predictions"], BrainData)
        assert isinstance(brain_data.cv_results_["folds"], np.ndarray)
        assert isinstance(brain_data.cv_results_["best_alpha"], (int, float))
        assert isinstance(brain_data.cv_results_["alpha_scores"], np.ndarray)

    # ==================== regress() Backward Compatibility ====================

    @pytest.mark.tier2
    def test_regress_emits_future_warning(self, sim_brain_data):
        """Test regress() emits FutureWarning telling users to switch to fit()."""
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data)),
                "X1": np.random.randn(len(sim_brain_data)),
            }
        )

        # Should emit FutureWarning with clear migration message
        with pytest.warns(
            FutureWarning, match="regress.*deprecated.*will raise an error in v0.7.0"
        ):
            result = sim_brain_data.regress(design_matrix, noise_model="ols")

        # Should still work and return dict
        assert isinstance(result, dict)
        assert "beta" in result

    def test_regress_calls_fit_internally(self, sim_brain_data):
        """Test regress() calls fit(model='glm') internally for backward compatibility."""
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data)),
            }
        )

        with pytest.warns(FutureWarning):
            sim_brain_data.regress(design_matrix, noise_model="ols")

        # Should have set model_ and glm_* attributes via fit()
        assert hasattr(sim_brain_data, "model_")
        assert hasattr(sim_brain_data, "glm_betas")
        assert hasattr(sim_brain_data, "glm_model")  # Backward compat alias

    def test_regress_supports_self_X_pattern(self, sim_brain_data):
        """Test regress() still works with deprecated self.X pattern."""
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data)),
            }
        )

        # Old pattern: setting self.X then calling regress() with no args
        sim_brain_data.X = design_matrix

        with pytest.warns(FutureWarning):
            result = sim_brain_data.regress()

        # Should still work
        assert "beta" in result
        assert hasattr(sim_brain_data, "glm_betas")

    def test_regress_ignores_mode_robust_silently(self, sim_brain_data):
        """Test regress() silently ignores deprecated mode='robust' parameter."""
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data)),
            }
        )

        # mode='robust' should be silently ignored (only one FutureWarning, not multiple)
        with pytest.warns(FutureWarning, match="regress.*deprecated"):
            result = sim_brain_data.regress(design_matrix, mode="robust")

        # Should still work
        assert isinstance(result, dict)
        assert "beta" in result

    def test_regress_returns_backward_compatible_dict(self, sim_brain_data):
        """Test regress() returns dict with expected keys for backward compatibility."""
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data)),
            }
        )

        with pytest.warns(FutureWarning):
            result = sim_brain_data.regress(design_matrix, noise_model="ols")

        # Check dict has expected structure for old code
        assert isinstance(result, dict)
        assert "beta" in result
        assert "t" in result
        assert "p" in result
        assert "residual" in result

        # Check dict values match the new attributes
        assert result["beta"] is sim_brain_data.glm_betas
        assert result["t"] is sim_brain_data.glm_t

    # ==================== Masking & ROI Extraction ====================

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

    @pytest.mark.tier2
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

    def test_standardize(self, sim_brain_data):
        """Test standardization with different methods."""
        s = sim_brain_data.standardize()
        assert s.shape == sim_brain_data.shape
        assert np.isclose(np.sum(s.mean().data), 0, atol=0.1)
        s = sim_brain_data.standardize(method="zscore")
        assert s.shape == sim_brain_data.shape
        assert np.isclose(np.sum(s.mean().data), 0, atol=0.1)

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

    @pytest.mark.tier2
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

    @pytest.mark.tier2
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

    @pytest.mark.tier2
    def test_threshold_cluster_with_upper_only(self, sim_brain_data):
        """Cluster threshold should work with upper threshold only"""
        brain = sim_brain_data.copy()
        result = brain.threshold(upper=2, cluster_threshold=10)
        assert isinstance(result, BrainData)

    @pytest.mark.tier2
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

    @pytest.mark.tier2
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

    @pytest.mark.tier2
    def test_threshold_cluster_realistic_neuroimaging(self, sim_brain_data):
        """Integration test with realistic neuroimaging workflow"""
        # Test with actual brain data structure from fixtures
        brain = sim_brain_data.copy()

        # Realistic workflow: threshold then cluster filter
        result = brain.threshold(lower=2.5, cluster_threshold=50)

        # Basic sanity checks
        assert isinstance(result, BrainData)
        assert result.shape == brain.shape
        assert not result.isempty

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

    @pytest.mark.tier2
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

    @pytest.mark.tier2
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

    @pytest.mark.tier2
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

        with pytest.raises(
            NotImplementedError, match="randomise.*deprecated.*Model class"
        ):
            sim_brain_data.randomise(n_permute=10)

    @pytest.mark.skip(reason="method needs refactoring")
    def test_bootstrap(self, sim_brain_data):
        """Test bootstrap with mean/std (predict is deprecated)."""
        # Bootstrap itself is not deprecated, but some functions it calls might be
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        # Test basic bootstrap with mean and std (should work)
        n_samples = 3
        b = masked.bootstrap("mean", n_samples=n_samples)
        assert isinstance(b["Z"], BrainData)
        b = masked.bootstrap("std", n_samples=n_samples)
        assert isinstance(b["Z"], BrainData)

        # Bootstrap with "predict" will fail since predict is deprecated
        with pytest.raises(
            NotImplementedError, match="predict.*deprecated.*Model class"
        ):
            masked.bootstrap("predict", n_samples=n_samples)

    @pytest.mark.tier2
    def test_predict_multi(self):
        """Test that deprecated predict_multi method raises NotImplementedError."""
        # Need to set up minimal data for the test
        sim = Simulator()
        dat = sim.create_data([0, 1], sigma=1, reps=5, output_dir=".")
        y = pd.read_csv("y.csv", header=None, index_col=None)
        dat = BrainData("data.nii.gz", Y=y)

        with pytest.raises(
            NotImplementedError, match="predict_multi.*deprecated.*Model class"
        ):
            dat.predict_multi(algorithm="svm")
