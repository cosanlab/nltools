"""Ownership and numerical regressions for common BrainData transforms."""

import pytest
import numpy as np
from nltools.data import BrainData
from nltools.data.braindata.utils import _copy_for_fit
from nltools.mask import create_sphere
import pandas as pd


def test_copy_for_fit_with_data(sim_brain_data):
    """The fit preparation copy independently owns retained structure."""

    # Derived copies independently own both data and spatial structure.
    copied = _copy_for_fit(sim_brain_data)

    # Should own mask
    assert copied.mask is not sim_brain_data.mask

    assert copied.data is not sim_brain_data.data
    np.testing.assert_array_equal(copied.data, sim_brain_data.data)

    # Should have copied X and Y if present
    if hasattr(sim_brain_data, "X") and sim_brain_data.X is not None:
        assert copied.X is not sim_brain_data.X
        assert copied.X.equals(sim_brain_data.X)

    if hasattr(sim_brain_data, "Y") and sim_brain_data.Y is not None:
        assert copied.Y is not sim_brain_data.Y
        assert copied.Y.equals(sim_brain_data.Y)


def test_scale_efficient(sim_brain_data):
    """Test that scale() uses independent copying"""

    # Test scale method
    scaled = sim_brain_data.scale(100.0)

    # Verify the scaling worked
    assert np.isclose(scaled.data.mean(), 100.0, rtol=1e-5)

    # Verify data independence
    assert scaled.data is not sim_brain_data.data

    # Verify independent mask state
    assert scaled.mask is not sim_brain_data.mask

    # Original should be unchanged
    original_mean = sim_brain_data.data.mean()
    assert not np.isclose(original_mean, 100.0)


@pytest.mark.filterwarnings("ignore:Numerical issues:UserWarning")
def test_method_chaining_efficiency(sim_brain_data):
    """Chained transforms leave the original data and mask independent."""

    result = sim_brain_data.scale(100.0).standardize()

    # Both mask state and the transformed data buffer are independently owned.
    assert result.mask is not sim_brain_data.mask
    assert result.data is not sim_brain_data.data
    assert result.shape == sim_brain_data.shape

    # Should have mean close to 0 (after standardization)
    assert np.abs(result.data.mean()) < 0.1


def test_scale_preserves_attributes(sim_brain_data):
    """Test that scale() preserves all necessary attributes"""

    # Add some test attributes
    sim_brain_data._test_attr = "test_value"

    scaled = sim_brain_data.scale(100.0)

    # Check that attributes are preserved
    assert hasattr(scaled, "_test_attr")
    assert scaled._test_attr == "test_value"

    # Clean up
    del sim_brain_data._test_attr


def test_scale_with_different_values(sim_brain_data):
    """Test scale with different scale values"""

    # Test with different scale values
    for scale_val in [1.0, 100.0, 10000.0]:
        scaled = sim_brain_data.scale(scale_val)
        assert np.isclose(scaled.data.mean(), scale_val, rtol=1e-5)

        # Verify independence each time
        assert scaled.data is not sim_brain_data.data


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


def test_all_arithmetic_methods_efficient(sim_brain_data):
    """Test that all arithmetic operations use independent copying"""

    # Verify that arithmetic operations preserve object independence
    result = sim_brain_data + 1
    assert id(result.mask) != id(sim_brain_data.mask), "Addition should own mask"

    result = sim_brain_data - 1
    assert id(result.mask) != id(sim_brain_data.mask), "Subtraction should own mask"

    result = sim_brain_data * 2
    assert id(result.mask) != id(sim_brain_data.mask), "Multiplication should own mask"

    result = sim_brain_data / 2
    assert id(result.mask) != id(sim_brain_data.mask), "Division should own mask"

    # Test chaining preserves independence
    result = ((sim_brain_data + 1) * 2 - 0.5) / 2
    assert id(result.mask) != id(sim_brain_data.mask), "Chain should own mask"

    # Verify data independence
    assert id(result.data) != id(sim_brain_data.data), "Should have new data"


def test_transform_methods_efficient():
    """Transform methods own the mask and leave the source data untouched."""

    s1 = create_sphere([12, 10, -8], radius=10)
    brain = BrainData([s1] * 20)  # 20 images
    original_data = brain.data.copy()

    result = brain.scale(100.0).standardize()

    # Mask state and data are independent; the source remains unchanged.
    assert result.mask is not brain.mask
    assert result.data is not brain.data
    assert np.array_equal(brain.data, original_data)


def test_getitem_efficiency(sim_brain_data):
    """Test that indexing operations use independent copying"""

    # Test that indexing preserves object independence
    indexed = sim_brain_data[0]

    # Should own mask
    assert id(indexed.mask) != id(sim_brain_data.mask), "Indexing should own mask"

    # Data should be different (it's a slice/subset)
    assert indexed.data.shape != sim_brain_data.data.shape, (
        "Should have different shape"
    )

    # Test slicing
    sliced = sim_brain_data[0:2]
    assert id(sliced.mask) != id(sim_brain_data.mask), "Slicing should own mask"

    # Test fancy indexing
    if len(sim_brain_data) >= 3:
        fancy = sim_brain_data[[0, 2]]
        assert id(fancy.mask) != id(sim_brain_data.mask), (
            "Fancy indexing should own mask"
        )


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


@pytest.mark.slow
def test_no_accidental_deep_copies():
    """Transforms independently copy custom mutable state."""

    # Create a BrainData with a reasonable size
    s1 = create_sphere([12, 10, -8], radius=10)
    brain = BrainData([s1] * 20)  # 20 images

    # Add a custom attribute to track if it gets deep copied
    brain._custom_tracking_attribute = "original"
    brain._custom_list = [1, 2, 3]

    # Operations return independent derived objects
    scaled = brain.scale(100.0)

    # Verify object independence behavior
    assert id(scaled) != id(brain), "Should be a new object"
    assert id(scaled.mask) != id(brain.mask), "Should own mask object"

    # Verify custom attributes were copied correctly
    assert hasattr(scaled, "_custom_tracking_attribute"), (
        "Should preserve custom attributes"
    )
    assert scaled._custom_tracking_attribute == "original"

    # Lists should be deep copied (to prevent mutation issues)
    assert id(scaled._custom_list) != id(brain._custom_list), "Lists should be copied"
    assert scaled._custom_list == brain._custom_list, "List contents should match"

    # Verify data independence
    assert id(scaled.data) != id(brain.data), "Data should be independent"

    # Test mutation safety
    scaled._custom_tracking_attribute = "modified"
    assert brain._custom_tracking_attribute == "original", (
        "Modifying copy shouldn't affect original"
    )


@pytest.mark.slow
def test_chained_operations_preserve_efficiency():
    """Test that chaining multiple operations maintains independence"""

    # Create test data
    s1 = create_sphere([12, 10, -8], radius=10)
    brain = BrainData([s1] * 10)  # 10 images

    # Test a long chain of operations
    result = brain.scale(100.0).r_to_z().z_to_r().scale(50.0)

    # Verify that mask remains independent after chaining
    assert id(result.mask) != id(brain.mask), "Chain should preserve mask independence"

    # Verify data is independent
    assert id(result.data) != id(brain.data), "Data should be new after chain"

    # Verify the operations actually worked
    assert not np.array_equal(result.data, brain.data), "Data should be transformed"

    # Test mutation safety through chain
    original_data = brain.data.copy()
    result.data[0, 0] = 999999
    assert np.array_equal(brain.data, original_data), (
        "Original should be unaffected by mutation"
    )
