"""Tests for efficient copying in Brain_Data

These tests verify that:
1. Shallow copying is actually shallow (not deep)
2. Data independence is maintained (no accidental mutations)
3. Performance improvements are real and measurable
4. All updated methods use efficient copying
"""

import pytest
import numpy as np
import time
from copy import deepcopy
from nltools.data import Brain_Data
from nltools.mask import create_sphere
import pandas as pd


def test_shallow_copy_with_data(sim_brain_data):
    """Test that _shallow_copy_with_data works correctly"""

    # Create shallow copy
    copied = sim_brain_data._shallow_copy_with_data()

    # Should share mask and nifti_masker
    assert copied.mask is sim_brain_data.mask
    assert copied.nifti_masker is sim_brain_data.nifti_masker

    # Should initially share data array
    assert copied.data is sim_brain_data.data

    # Should have copied X and Y if present
    if hasattr(sim_brain_data, 'X') and sim_brain_data.X is not None:
        assert copied.X is not sim_brain_data.X
        assert copied.X.equals(sim_brain_data.X)

    if hasattr(sim_brain_data, 'Y') and sim_brain_data.Y is not None:
        assert copied.Y is not sim_brain_data.Y
        assert copied.Y.equals(sim_brain_data.Y)


def test_scale_efficient(sim_brain_data):
    """Test that scale() uses efficient copying"""

    # Test scale method
    scaled = sim_brain_data.scale(100.0)

    # Verify the scaling worked
    assert np.isclose(scaled.data.mean(), 100.0, rtol=1e-5)

    # Verify data independence
    assert scaled.data is not sim_brain_data.data

    # Verify mask sharing (efficient)
    assert scaled.mask is sim_brain_data.mask
    assert scaled.nifti_masker is sim_brain_data.nifti_masker

    # Original should be unchanged
    original_mean = sim_brain_data.data.mean()
    assert not np.isclose(original_mean, 100.0)


def test_method_chaining_efficiency(sim_brain_data):
    """Test that method chaining is efficient"""

    # Measure time for chained operations
    start = time.time()
    result = sim_brain_data.scale(100.0).standardize()
    elapsed = time.time() - start

    # Should be reasonably fast (under 1 second for small data)
    assert elapsed < 1.0, f"Chained operations took {elapsed:.3f} seconds"

    # Verify result is independent
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
    assert hasattr(scaled, '_test_attr')
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


def test_comparison_with_deepcopy():
    """Compare performance of shallow copy vs deep copy"""

    # Create a larger Brain_Data for performance testing
    s1 = create_sphere([12, 10, -8], radius=10)
    brain = Brain_Data([s1] * 10)  # 10 images

    # Measure deep copy time
    start = time.time()
    deep_copied = deepcopy(brain)
    deep_time = time.time() - start

    # Measure shallow copy time
    start = time.time()
    shallow_copied = brain._shallow_copy_with_data()
    shallow_time = time.time() - start

    # Shallow copy should be much faster
    assert shallow_time < deep_time * 0.5, (
        f"Shallow copy ({shallow_time:.4f}s) should be much faster than "
        f"deep copy ({deep_time:.4f}s)"
    )

    # Both should have the same data initially
    assert np.array_equal(shallow_copied.data, deep_copied.data)


def test_shallow_copy_is_truly_shallow(sim_brain_data):
    """Verify that shallow copy shares the right objects and copies the right ones"""

    # Add DataFrame attributes to test
    sim_brain_data.test_X = pd.DataFrame({'col1': [1, 2, 3]})

    # Create shallow copy
    copied = sim_brain_data._shallow_copy_with_data()

    # These should be SHARED (same object in memory)
    assert id(copied.mask) == id(sim_brain_data.mask), "mask should be shared"
    assert id(copied.nifti_masker) == id(sim_brain_data.nifti_masker), "nifti_masker should be shared"
    assert id(copied.data) == id(sim_brain_data.data), "data should initially be shared"

    # These should be COPIED (different objects)
    if hasattr(sim_brain_data, 'X') and sim_brain_data.X is not None:
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
    assert np.array_equal(sim_brain_data.data, original_data_copy), "Original data was mutated!"
    assert sim_brain_data.data.mean() == original_mean, "Original mean changed!"

    # Each result should have different data
    assert not np.array_equal(scaled.data, sim_brain_data.data)
    assert not np.array_equal(added.data, sim_brain_data.data)
    assert not np.array_equal(subtracted.data, sim_brain_data.data)
    assert not np.array_equal(multiplied.data, sim_brain_data.data)


def test_all_arithmetic_methods_efficient(sim_brain_data):
    """Test that all arithmetic operations use efficient copying"""

    # Verify that arithmetic operations preserve object sharing
    result = sim_brain_data + 1
    assert id(result.mask) == id(sim_brain_data.mask), "Addition should share mask"

    result = sim_brain_data - 1
    assert id(result.mask) == id(sim_brain_data.mask), "Subtraction should share mask"

    result = sim_brain_data * 2
    assert id(result.mask) == id(sim_brain_data.mask), "Multiplication should share mask"

    result = sim_brain_data / 2
    assert id(result.mask) == id(sim_brain_data.mask), "Division should share mask"

    # Test chaining preserves efficiency
    result = ((sim_brain_data + 1) * 2 - 0.5) / 2
    assert id(result.mask) == id(sim_brain_data.mask), "Chain should share mask"

    # Verify data independence
    assert id(result.data) != id(sim_brain_data.data), "Should have new data"


def test_transform_methods_efficient():
    """Test that transform methods (scale, standardize, etc.) are efficient"""

    # Create a larger dataset for meaningful timing
    s1 = create_sphere([12, 10, -8], radius=10)
    brain = Brain_Data([s1] * 20)  # 20 images

    # Time transform chain
    start = time.time()
    result = brain.scale(100.0).standardize()
    transform_time = time.time() - start

    # Time equivalent deep copy operations
    start = time.time()
    copy1 = deepcopy(brain)
    copy1.data = copy1.data / copy1.data.mean() * 100.0
    copy2 = deepcopy(copy1)
    from sklearn.preprocessing import scale
    copy2.data = scale(copy2.data, axis=0, with_std=True)
    deep_copy_time = time.time() - start

    # Transform chain should be significantly faster
    assert transform_time < deep_copy_time * 0.75, (
        f"Transform chain ({transform_time:.4f}s) should be faster than "
        f"deep copy approach ({deep_copy_time:.4f}s)"
    )


def test_getitem_efficiency(sim_brain_data):
    """Test that indexing operations use efficient copying"""

    # Add some data to make it more realistic
    if len(sim_brain_data) == 1:
        # Skip if single image
        pytest.skip("Need multiple images for indexing test")

    # Test that indexing preserves object sharing
    indexed = sim_brain_data[0]

    # Should share mask and nifti_masker
    assert id(indexed.mask) == id(sim_brain_data.mask), "Indexing should share mask"
    assert id(indexed.nifti_masker) == id(sim_brain_data.nifti_masker), "Indexing should share masker"

    # Data should be different (it's a slice/subset)
    assert indexed.data.shape != sim_brain_data.data.shape, "Should have different shape"

    # Test slicing
    sliced = sim_brain_data[0:2]
    assert id(sliced.mask) == id(sim_brain_data.mask), "Slicing should share mask"
    assert id(sliced.nifti_masker) == id(sim_brain_data.nifti_masker), "Slicing should share masker"

    # Test fancy indexing
    if len(sim_brain_data) >= 3:
        fancy = sim_brain_data[[0, 2]]
        assert id(fancy.mask) == id(sim_brain_data.mask), "Fancy indexing should share mask"
        assert id(fancy.nifti_masker) == id(sim_brain_data.nifti_masker), "Fancy indexing should share masker"


def test_append_efficiency():
    """Test that append is efficient"""

    # Create test data
    s1 = create_sphere([12, 10, -8], radius=10)
    brain1 = Brain_Data(s1)
    brain2 = Brain_Data(s1)

    # Time append operation
    start = time.time()
    appended = brain1.append(brain2)
    append_time = time.time() - start

    # Time deep copy equivalent
    start = time.time()
    copy1 = deepcopy(brain1)
    copy2 = deepcopy(brain2)
    deep_copy_time = time.time() - start

    # Append should be faster than two deep copies
    assert append_time < deep_copy_time * 0.8, (
        f"Append ({append_time:.4f}s) should be faster than "
        f"deep copies ({deep_copy_time:.4f}s)"
    )


def test_no_accidental_deep_copies():
    """Ensure methods aren't secretly doing deep copies internally"""

    # Create a Brain_Data with a reasonable size
    s1 = create_sphere([12, 10, -8], radius=10)
    brain = Brain_Data([s1] * 20)  # 20 images

    # Add a custom attribute to track if it gets deep copied
    brain._custom_tracking_attribute = "original"
    brain._custom_list = [1, 2, 3]

    # Operations that should NOT deep copy the Brain_Data object
    scaled = brain.scale(100.0)

    # Verify object sharing behavior
    assert id(scaled) != id(brain), "Should be a new object"
    assert id(scaled.mask) == id(brain.mask), "Should share mask object"
    assert id(scaled.nifti_masker) == id(brain.nifti_masker), "Should share nifti_masker"

    # Verify custom attributes were copied correctly
    assert hasattr(scaled, '_custom_tracking_attribute'), "Should preserve custom attributes"
    assert scaled._custom_tracking_attribute == "original"

    # Lists should be deep copied (to prevent mutation issues)
    assert id(scaled._custom_list) != id(brain._custom_list), "Lists should be copied"
    assert scaled._custom_list == brain._custom_list, "List contents should match"

    # Verify data independence
    assert id(scaled.data) != id(brain.data), "Data should be independent"

    # Test mutation safety
    scaled._custom_tracking_attribute = "modified"
    assert brain._custom_tracking_attribute == "original", "Modifying copy shouldn't affect original"


def test_chained_operations_preserve_efficiency():
    """Test that chaining multiple operations maintains object sharing"""

    # Create test data
    s1 = create_sphere([12, 10, -8], radius=10)
    brain = Brain_Data([s1] * 10)  # 10 images

    # Test a long chain of operations
    result = brain.scale(100.0).r_to_z().z_to_r().scale(50.0)

    # Verify that mask and nifti_masker are still shared after chain
    assert id(result.mask) == id(brain.mask), "Chain should preserve mask sharing"
    assert id(result.nifti_masker) == id(brain.nifti_masker), "Chain should preserve masker sharing"

    # Verify data is independent
    assert id(result.data) != id(brain.data), "Data should be new after chain"

    # Verify the operations actually worked
    assert not np.array_equal(result.data, brain.data), "Data should be transformed"

    # Test mutation safety through chain
    original_data = brain.data.copy()
    result.data[0, 0] = 999999
    assert np.array_equal(brain.data, original_data), "Original should be unaffected by mutation"