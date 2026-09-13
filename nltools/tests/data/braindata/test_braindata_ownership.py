"""Mutation and numerical contracts for BrainData result ownership."""

from copy import copy, deepcopy

import nibabel as nib
import numpy as np
import pandas as pd
import polars as pl
import pytest
from nilearn.maskers import NiftiMasker

from nltools.data import BrainData, DesignMatrix
from nltools.data.braindata.utils import _copy_for_fit
from nltools.data.ownership import _copy_frame, _copy_object_frames
from nltools.mask import create_sphere


@pytest.fixture
def brain():
    mask = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))
    x = np.column_stack([np.ones(12), np.linspace(-1, 1, 12)])
    y = x @ np.arange(16).reshape(2, 8) + np.sin(np.arange(96).reshape(12, 8))
    b = BrainData(y, mask=mask, X=pl.DataFrame(x), Y=pl.DataFrame({"row": range(12)}))
    b.masker = NiftiMasker(mask_img=b.mask).fit()
    return b


@pytest.mark.parametrize("copier", [copy])
def test_complete_graph_copy(brain, copier):
    brain.fit(model="ridge", X=brain.X.to_numpy(), ridge_alpha=2)
    brain.alias = brain.data
    brain.cycle = brain
    other = copier(brain)
    assert other.alias is other.data
    assert other.cycle is other
    other.masker.mask_img_.get_fdata()[0, 0, 0] = 77
    other.data[0, 0] = 888
    other.model_.coef_[0, 0] = 999
    other.ridge_weights.data[0, 0] = 333
    assert brain.masker.mask_img_.get_fdata()[0, 0, 0] == 1
    assert brain.data[0, 0] != 888
    assert brain.model_.coef_[0, 0] != 999
    assert brain.ridge_weights.data[0, 0] != 333


@pytest.mark.parametrize("model", ["glm"])
def test_fit_maps_predictions_and_numerics(brain, model):
    x = brain.X.to_numpy()
    y = brain.data.copy()
    design = DesignMatrix(x) if model == "glm" else x
    fitted = brain.fit(
        model=model,
        X=design,
        inplace=False,
        **({"glm_noise_model": "ols"} if model == "glm" else {"ridge_alpha": 2}),
    )
    expected = (
        np.linalg.lstsq(x, y, rcond=None)[0]
        if model == "glm"
        else np.linalg.solve(x.T @ x + 2 * np.eye(2), x.T @ y)
    )
    weights = fitted.glm_betas if model == "glm" else fitted.ridge_weights
    np.testing.assert_allclose(weights.data, expected, atol=2e-6, rtol=2e-6)
    from nltools.data.braindata.prediction import _predict_timeseries

    predicted = _predict_timeseries(fitted)
    new = fitted.predict(
        X=DesignMatrix(x[:3], columns=design.columns) if model == "glm" else x[:3]
    )
    np.testing.assert_allclose(predicted.data, x @ expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(new.data, x[:3] @ expected, atol=2e-6, rtol=2e-6)
    assert weights.X.is_empty() and weights.Y.is_empty()
    assert new.X.is_empty() and new.Y.is_empty()
    assert predicted.X.equals(brain.X) and predicted.Y.equals(brain.Y)
    if model == "glm":
        assert fitted.glm_residual.Y.equals(brain.Y)
        np.testing.assert_allclose(
            fitted.glm_residual.data, y - x @ expected, atol=2e-6, rtol=2e-6
        )
    weights.mask.get_fdata()[0, 0, 0] = 9
    new.masker.mask_img_.get_fdata()[0, 0, 0] = 8
    assert fitted.mask.get_fdata()[0, 0, 0] == 1
    assert brain.mask.get_fdata()[0, 0, 0] == 1
    assert predicted.masker.mask_img_.get_fdata()[0, 0, 0] == 1


def test_contrast_results_clear_rows_and_own_their_maps(brain):
    """Contrast maps leave the source's row metadata alone and carry none."""
    from nltools.models import ContrastResult

    design = DesignMatrix(brain.X.to_numpy(), columns=["intercept", "slope"])
    fitted = brain.fit(model="glm", X=design, inplace=False)

    effect = fitted.compute_contrasts("intercept - slope")
    result = fitted.compute_contrasts("intercept - slope", inference=True)

    assert isinstance(result, ContrastResult)
    payloads = [effect] + [
        getattr(result, name)
        for name in (
            "effect",
            "variance",
            "standard_error",
            "statistic",
            "z_score",
            "p_value",
        )
    ]
    for payload in payloads:
        assert payload.X.is_empty() and payload.Y.is_empty()
        assert not np.shares_memory(payload.data, fitted.glm_betas.data)
    assert not fitted.X.is_empty() and not fitted.Y.is_empty()
    assert fitted.X.equals(brain.X) and fitted.Y.equals(brain.Y)

    effect.data[0] = 4242.0
    result.effect.data[0] = 4243.0
    assert fitted.glm_betas.data[0, 0] not in (4242.0, 4243.0)
    assert brain.X["column_0"].to_list() == [1.0] * 12


def test_transforms_selection_and_metadata(brain):
    for result in [brain.scale(), brain + 2, brain[[3, 1]], brain.mean()]:
        result.masker.mask_img_.get_fdata()[0, 0, 0] = 8
        assert brain.masker.mask_img_.get_fdata()[0, 0, 0] == 1
    assert brain[[3, 1]].Y["row"].to_list() == [3, 1]
    assert (brain + 2).Y.equals(brain.Y)
    assert brain.mean().Y.is_empty()


def test_inplace_weighted_reduction_clears_rows(brain):
    expected = brain.data.T @ np.ones(12)
    brain *= np.ones(12)
    np.testing.assert_allclose(brain.data, expected)
    assert brain.X.is_empty() and brain.Y.is_empty()


def test_replacement_mask_does_not_copy_old_masker(brain):
    from nltools.data.braindata.utils import _result_with_mask

    class ObsoleteMasker:
        def __deepcopy__(self, memo):
            raise AssertionError("obsolete spatial state was copied")

    brain.masker = ObsoleteMasker()
    mask = nib.Nifti1Image(np.ones((1, 1, 2), dtype=np.uint8), np.eye(4))
    result = _result_with_mask(brain, brain.data[:, :2], mask, rows="preserve")
    assert result.masker is None
    assert result.Y.equals(brain.Y)
    result.mask.get_fdata()[0, 0, 0] = 9
    assert mask.get_fdata()[0, 0, 0] == 1


def test_replacement_rows_validate_before_return(brain):
    from nltools.data.braindata.utils import _result_from_rows

    with pytest.raises(ValueError, match="rows"):
        _result_from_rows(brain, brain.data[:2], X=brain.X, Y=None)


def test_fit_copy_skips_obsolete_fitted_state(brain):
    from nltools.data.braindata.utils import _copy_for_fit, _result_from_array

    class ObsoleteFit:
        def __deepcopy__(self, memo):
            raise AssertionError("obsolete fitted state was copied")

    brain.model_ = ObsoleteFit()
    for result in [
        _copy_for_fit(brain),
        _result_from_array(brain, brain.data, rows="preserve"),
    ]:
        assert not hasattr(result, "model_")
        result.data[0, 0] = 999
        assert brain.data[0, 0] != 999


@pytest.mark.parametrize("model", ["ridge"])
def test_source_and_sibling_mutation_after_fitting(brain, model):
    x = brain.X.to_numpy()
    fitted = brain.fit(
        model=model,
        X=DesignMatrix(x) if model == "glm" else x,
        inplace=False,
    )
    training = fitted.glm_predicted if model == "glm" else fitted.ridge_fitted_values
    expected = training.data.copy()
    brain.data[:] = -999
    np.asarray(brain.mask.dataobj)[:] = 0
    brain.X.replace_column(0, pl.Series("column_0", np.zeros(12)))
    np.testing.assert_array_equal(training.data, expected)
    assert np.all(np.asarray(fitted.mask.dataobj) == 1)
    assert fitted.X["column_0"].to_list() == [1.0] * 12
    training.data[:] = 123
    assert not np.all(fitted.data == 123)
    training.X.replace_column(0, pl.Series("column_0", np.zeros(12)))
    assert fitted.X["column_0"].to_list() == [1.0] * 12
    fitted.Y = None
    public_training = fitted.predict()
    assert public_training.X.equals(fitted.X)
    assert not hasattr(public_training, "model_")


def test_refit_and_transforms_clear_old_fit_family(brain):
    x = brain.X.to_numpy()
    brain.fit(model="ridge", X=x)
    old = brain.copy()
    for transformed in [brain.scale(), brain + 2, brain[:2]]:
        assert not hasattr(transformed, "model_")
        assert not hasattr(transformed, "ridge_weights")
    brain.fit(model="glm", X=DesignMatrix(x))
    assert not hasattr(brain, "ridge_weights")
    assert hasattr(brain, "glm_betas")
    assert hasattr(old, "ridge_weights")
    assert not hasattr(old, "glm_betas")


def test_append_metadata_requires_matching_operands(brain):
    left, right = brain[:3], brain[3:]
    assert left.append(right).Y.equals(brain.Y)
    right.Y = None
    with pytest.raises(ValueError, match="Y"):
        left.append(right)
    cleared = left.append(right, ignore_attrs=True)
    assert cleared.X.is_empty() and cleared.Y.is_empty()


def test_masked_result_remains_spatially_usable(brain):
    values = np.ones((2, 2, 2), dtype=np.uint8)
    values[0] = 0
    mask = nib.Nifti1Image(values, np.eye(4))
    result = brain.apply_mask(mask)
    np.testing.assert_array_equal(result.data, brain.data[:, 4:])
    assert result.Y.equals(brain.Y)
    assert result.masker is None
    assert result.to_nifti().shape == (2, 2, 2, 12)
    np.asarray(result.mask.dataobj)[1, 0, 0] = 0
    assert np.asarray(brain.mask.dataobj)[1, 0, 0] == 1
    assert np.asarray(mask.dataobj)[1, 0, 0] == 1


@pytest.mark.parametrize("copier", [deepcopy])
def test_object_metadata_has_independent_mutable_cells(brain, copier):
    shared = {"nested": [1]}
    brain.X = pl.DataFrame({"object": pl.Series([shared] * 12, dtype=pl.Object)})
    brain.Y = pl.DataFrame({"object": pl.Series([shared] * 12, dtype=pl.Object)})
    brain.nested_frames = {"frames": [brain.X, brain.Y]}
    result = copier(brain)
    assert result.nested_frames["frames"][0] is result.X
    assert result.X["object"][0] is result.Y["object"][0]
    result.X["object"][0]["nested"].append(2)
    assert shared["nested"] == [1]
    shared["nested"].append(3)
    assert result.Y["object"][0]["nested"] == [1, 2]


def test_append_keeps_compatible_numeric_metadata(brain):
    left, right = brain[:3], brain[3:]
    right.Y = right.Y.cast({"row": pl.Float64})
    result = left.append(right)
    assert result.Y["row"].to_list() == list(range(12))


# ---------------------------------------------------------------------------
# Ownership of derived arrays, fit copies and appends
# ---------------------------------------------------------------------------


def test_derived_array_owns_input_data(sim_brain_data):
    from nltools.data.braindata.utils import _result_from_array

    result = _result_from_array(sim_brain_data, sim_brain_data.data, rows="preserve")
    result.data[0, 0] = 999
    assert sim_brain_data.data[0, 0] != 999
    assert result.mask is not sim_brain_data.mask


def test_copy_for_fit_owns_spatial_structure(sim_brain_data):
    """The default derived copy owns mutable data and metadata."""

    # Add DataFrame attributes to test
    sim_brain_data.test_X = pd.DataFrame({"col1": [1, 2, 3]})

    copied = _copy_for_fit(sim_brain_data)

    assert id(copied.mask) != id(sim_brain_data.mask), "mask should be independent"
    assert id(copied.data) != id(sim_brain_data.data), "data should be copied"

    # These should be COPIED (different objects)
    if hasattr(sim_brain_data, "X") and sim_brain_data.X is not None:
        assert id(copied.X) != id(sim_brain_data.X), "X DataFrame should be copied"

    # Clean up
    del sim_brain_data.test_X


def test_data_mutation_safety(sim_brain_data):
    """Test that operations don't accidentally mutate the original data"""

    # Store original data
    original_data_copy = sim_brain_data.data.copy()
    original_mean = sim_brain_data.data.mean()

    # Perform operations that should NOT mutate original
    scaled = sim_brain_data.scale(100.0)
    added = sim_brain_data + 5
    subtracted = sim_brain_data - 2
    multiplied = sim_brain_data * 3

    # Original data should be completely unchanged
    assert np.array_equal(sim_brain_data.data, original_data_copy), (
        "Original data was mutated!"
    )
    assert sim_brain_data.data.mean() == original_mean, "Original mean changed!"

    # Each result should have different data
    assert not np.array_equal(scaled.data, sim_brain_data.data)
    assert not np.array_equal(added.data, sim_brain_data.data)
    assert not np.array_equal(subtracted.data, sim_brain_data.data)
    assert not np.array_equal(multiplied.data, sim_brain_data.data)


def test_append_correctness():
    """Test that append works correctly with independent copying"""

    # Create test data with multiple images for clearer testing
    s1 = create_sphere([12, 10, -8], radius=10)
    brain1 = BrainData([s1] * 3)  # 3 images
    brain2 = BrainData([s1] * 2)  # 2 images

    # Store original data for verification
    brain1_data_copy = brain1.data.copy()
    brain2_data_copy = brain2.data.copy()
    n_images_1 = len(brain1)
    n_images_2 = len(brain2)

    # Perform append
    appended = brain1.append(brain2)

    # Verify correctness: should have combined number of images
    assert len(appended) == n_images_1 + n_images_2, (
        f"Appended should have {n_images_1 + n_images_2} images, got {len(appended)}"
    )
    assert appended.shape[0] == brain1.shape[0] + brain2.shape[0], (
        "Appended data should have combined number of images"
    )
    assert appended.shape[1] == brain1.shape[1], (
        "Appended data should preserve number of voxels"
    )

    # Verify data values are preserved
    assert np.array_equal(appended.data[0], brain1_data_copy[0]), (
        "First image should match brain1 first image"
    )
    assert np.array_equal(appended.data[n_images_1 - 1], brain1_data_copy[-1]), (
        "Last brain1 image should be preserved"
    )
    assert np.array_equal(appended.data[n_images_1], brain2_data_copy[0]), (
        "First brain2 image should follow brain1 data"
    )
    assert np.array_equal(appended.data[-1], brain2_data_copy[-1]), (
        "Last image should match brain2 last image"
    )

    # Verify independent copying: should own mask
    assert id(appended.mask) != id(brain1.mask), (
        "Append should own mask object independently"
    )

    # Verify data independence: new data array
    assert id(appended.data) != id(brain1.data), (
        "Appended data should be independent from brain1"
    )
    assert id(appended.data) != id(brain2.data), (
        "Appended data should be independent from brain2"
    )

    # Verify originals are unchanged
    assert np.array_equal(brain1.data, brain1_data_copy), (
        "Original brain1 should be unchanged"
    )
    assert np.array_equal(brain2.data, brain2_data_copy), (
        "Original brain2 should be unchanged"
    )


def test_copy_frame_owns_buffers_that_object_frame_copying_leaves_shared():
    """`_copy_frame` detaches numeric buffers; `_copy_object_frames` does not.

    Polars wraps a NumPy array zero-copy, so a plain clone still reads the
    caller's memory. The two copiers answer that differently and both answers
    are load-bearing, so they stay separate functions.
    """
    values = np.arange(5, dtype=np.float64)
    frame = pl.DataFrame({"a": values})

    detached = _copy_frame(frame)
    memo = {}
    _copy_object_frames({"frame": frame}, memo)
    shared = memo[id(frame)]

    values[0] = 99.0

    assert detached["a"].to_list() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert shared["a"].to_list() == [99.0, 1.0, 2.0, 3.0, 4.0]
