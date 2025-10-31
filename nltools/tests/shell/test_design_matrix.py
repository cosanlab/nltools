"""
Comprehensive test suite for DesignMatrix (Polars-based implementation)

These tests specify BEHAVIOR, not implementation details. Each test validates
what DesignMatrix should DO, allowing implementation flexibility.

Test organization:
1. Construction - All ways to create a DesignMatrix
2. Data Access - Column access and manipulation
3. Transformations - Simple data operations
4. Stats Operations - Statistical transformations
5. Convolution - HRF convolution and custom kernels
6. Polynomials - Legendre and DCT basis functions
7. Concatenation - Multi-run append logic (most complex)
8. Diagnostics - VIF and collinearity detection
9. Utilities - Misc helper methods
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from nltools.data.design_matrix import DesignMatrix


# ============================================================================
# 1. Construction Tests
# ============================================================================


class TestDesignMatrixConstruction:
    """
    Test all supported ways to create a DesignMatrix.

    Behavioral contract:
    - Accept numpy arrays, dicts, Polars/pandas DataFrames
    - Auto-generate column names if not provided
    - Store metadata: sampling_freq, convolved, polys, multi
    - Ensure all column names are strings (critical for consistency)
    """

    def test_from_numpy_with_columns(self):
        """
        Create from numpy array with explicit column names.

        Expected behavior:
        - Shape matches input array
        - Column names match provided list
        - Sampling frequency stored correctly
        """
        data = np.random.randn(100, 3)
        dm = DesignMatrix(data, sampling_freq=2, columns=["a", "b", "c"])

        assert dm.shape == (100, 3), "Shape should match input array"
        assert dm.columns == ["a", "b", "c"], "Column names should match provided list"
        assert dm.sampling_freq == 2, "Sampling frequency should be stored"

    def test_from_numpy_auto_columns(self):
        """
        Create from numpy array without column names.

        Expected behavior:
        - Auto-generate column names as strings: '0', '1', '2', ...
        - This ensures consistent string column names
        """
        data = np.random.randn(50, 2)
        dm = DesignMatrix(data, sampling_freq=1)

        assert dm.columns == ["0", "1"], "Should auto-generate string column names"

    def test_from_dict(self):
        """
        Create from dictionary (keys=columns, values=data).

        Expected behavior:
        - Converts dict to DataFrame naturally
        - Shape and column names inferred from dict
        """
        dm = DesignMatrix({"a": [1, 2, 3], "b": [4, 5, 6]}, sampling_freq=2)

        assert dm.shape == (3, 2)
        assert set(dm.columns) == {"a", "b"}
        assert dm.sampling_freq == 2

    def test_from_polars_dataframe(self):
        """
        Create from existing Polars DataFrame.

        Expected behavior:
        - Accept Polars DataFrame directly (zero-copy)
        - Preserve shape and columns
        """
        df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        dm = DesignMatrix(df, sampling_freq=1)

        assert dm.shape == (3, 2)
        assert set(dm.columns) == {"x", "y"}

    def test_from_pandas_dataframe(self):
        """
        Create from pandas DataFrame (backward compatibility).

        Expected behavior:
        - Accept pandas DataFrame for legacy code
        - Convert to Polars internally (users shouldn't know)
        - Preserve all data and column names
        """
        pdf = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        dm = DesignMatrix(pdf, sampling_freq=1)

        assert dm.shape == (3, 2)
        assert set(dm.columns) == {"x", "y"}
        # Implementation detail: internally should be Polars
        assert isinstance(dm._df, pl.DataFrame), (
            "Should convert pandas to Polars internally"
        )

    def test_empty_initialization(self):
        """
        Create empty DesignMatrix with only metadata.

        Expected behavior:
        - Shape (0, 0) for empty matrix
        - .empty property returns True
        - Metadata still accessible

        Use case: Initialize empty, then add columns iteratively
        """
        dm = DesignMatrix(sampling_freq=2)

        assert dm.shape == (0, 0)
        assert dm.empty is True
        assert dm.sampling_freq == 2

    def test_column_names_are_always_strings(self):
        """
        Ensure all column names are strings, even if input has numeric names.

        Expected behavior:
        - Convert all column names to strings automatically
        - This prevents type confusion bugs

        Rationale: Consistent string column names simplify logic throughout
        """
        # Numpy auto-generates numeric columns, should be stringified
        data = np.zeros((10, 3))
        dm = DesignMatrix(data, sampling_freq=1)

        for col in dm.columns:
            assert isinstance(col, str), (
                f"Column name {col} should be string, got {type(col)}"
            )

    def test_metadata_initialization(self):
        """
        Verify metadata attributes are initialized correctly.

        Expected behavior:
        - convolved starts as empty list
        - polys starts as empty list
        - multi starts as False
        - sampling_freq can be None
        """
        dm = DesignMatrix({"a": [1, 2, 3]}, sampling_freq=None)

        assert dm.sampling_freq is None
        assert dm.convolved == []
        assert dm.polys == []
        assert dm.multi is False


# ============================================================================
# 2. Data Access Tests
# ============================================================================


# %%
# NOTE: we should only support patterns that polars recommends so we don't build bad habits by encouraging old pandas-patters in polars. We should retain a very usable API, but not at the cost of behind-the-scenes polars complexity/efficiency. It's ok re-teach users a new API in these situations. So think carefully!
class TestDesignMatrixDataAccess:
    """
    Test column access and manipulation patterns.

    Behavioral contract:
    - dm['col'] returns Polars Series
    - dm[['col1', 'col2']] returns DesignMatrix subset
    - dm['col'] = value sets/creates column
    - Metadata preserved when subsetting
    """

    # NOTE: What does polars recommend when you just want a column back? Is it more efficient to stay as a dataframe? pros vs cons?
    def test_getitem_single_column_returns_series(self):
        """
        Access single column should return Polars Series.

        Expected behavior:
        - dm['col'] returns Series (not DesignMatrix)
        - Data matches original values

        Use case: Plotting, analysis on single column
        """
        dm = DesignMatrix({"a": [1, 2, 3], "b": [4, 5, 6]}, sampling_freq=1)
        col = dm["a"]

        assert isinstance(col, pl.Series), "Single column access should return Series"
        assert col.to_list() == [1, 2, 3], "Data should match original"

    def test_getitem_multiple_columns_returns_designmatrix(self):
        """
        Access multiple columns should return DesignMatrix subset.

        Expected behavior:
        - dm[['a', 'c']] returns DesignMatrix (not raw DataFrame)
        - Metadata preserved (sampling_freq, polys, etc.)
        - Only requested columns included

        Use case: Select subset of regressors for analysis
        """
        dm = DesignMatrix({"a": [1, 2], "b": [3, 4], "c": [5, 6]}, sampling_freq=2)
        dm.polys = ["a"]  # Mark 'a' as polynomial for metadata test

        subset = dm[["a", "c"]]

        assert isinstance(subset, DesignMatrix), (
            "Multi-column should return DesignMatrix"
        )
        assert subset.columns == ["a", "c"], "Should only include requested columns"
        assert subset.sampling_freq == 2, "Metadata should be preserved"
        assert subset.polys == ["a"], "Polynomial metadata should be preserved"

    def test_setitem_scalar_broadcasts(self):
        """
        Setting column to scalar should broadcast to all rows.

        Expected behavior:
        - dm['new_col'] = 0 creates column of zeros
        - Length matches number of rows

        Use case: Initialize column before filling specific values
        """
        dm = DesignMatrix({"a": [1, 2, 3]}, sampling_freq=1)
        dm["b"] = 0

        assert dm["b"].to_list() == [0, 0, 0], "Scalar should broadcast to all rows"

    def test_setitem_array_matches_length(self):
        """
        Setting column to array should use array values.

        Expected behavior:
        - Array length must match number of rows
        - Values assigned element-wise

        Use case: Add covariate column from external array
        """
        dm = DesignMatrix({"a": [1, 2, 3]}, sampling_freq=1)
        dm["b"] = [10, 20, 30]

        assert dm["b"].to_list() == [10, 20, 30], "Array values should be assigned"

    def test_setitem_replaces_existing_column(self):
        """
        Setting existing column should replace values.

        Expected behavior:
        - Overwrite existing data
        - Column order preserved
        """
        dm = DesignMatrix({"a": [1, 2, 3], "b": [4, 5, 6]}, sampling_freq=1)
        dm["a"] = [7, 8, 9]

        assert dm["a"].to_list() == [7, 8, 9], "Should replace existing column"
        assert dm.columns == ["a", "b"], "Column order should be preserved"

    def test_shape_property(self):
        """
        .shape should return (n_rows, n_cols) tuple like pandas/numpy.

        Expected behavior:
        - Returns tuple (not DataFrame.shape which is also tuple in Polars)
        - Matches dimensions of underlying data
        """
        dm = DesignMatrix(np.zeros((10, 3)), sampling_freq=1)

        assert dm.shape == (10, 3)
        assert isinstance(dm.shape, tuple)

    def test_columns_property_getter(self):
        """
        .columns should return list of column names.

        Expected behavior:
        - Returns list (or list-like) of strings
        - Order matches DataFrame column order
        """
        dm = DesignMatrix({"a": [1], "b": [2], "c": [3]}, sampling_freq=1)

        assert dm.columns == ["a", "b", "c"]

    def test_columns_property_setter(self):
        """
        Setting .columns should rename all columns.

        Expected behavior:
        - Rename columns in-place (mutates object)
        - Number of new names must match number of columns

        Use case: Rename columns after construction
        """
        dm = DesignMatrix({"a": [1], "b": [2]}, sampling_freq=1)
        dm.columns = ["x", "y"]

        assert dm.columns == ["x", "y"]

    def test_empty_property(self):
        """
        .empty should return True if no data, False otherwise.

        Expected behavior:
        - Empty DesignMatrix returns True
        - Non-empty returns False
        """
        dm_empty = DesignMatrix(sampling_freq=1)
        dm_full = DesignMatrix({"a": [1]}, sampling_freq=1)

        assert dm_empty.empty is True
        assert dm_full.empty is False

    def test_len_returns_number_of_rows(self):
        """
        len(dm) should return number of rows (like pandas).

        Expected behavior:
        - Returns int
        - Matches first element of .shape

        Use case: Quick check of number of timepoints/observations
        """
        dm = DesignMatrix(np.zeros((42, 5)), sampling_freq=1)

        assert len(dm) == 42
        assert len(dm) == dm.shape[0]


# ============================================================================
# 3. Simple Transformation Tests
# ============================================================================


class TestDesignMatrixTransformations:
    """
    Test simple data transformation methods.

    Behavioral contract:
    - All transformations return NEW DesignMatrix (immutable pattern)
    - Original DesignMatrix unchanged
    - Metadata preserved appropriately
    """

    # NOTE: ensure we do what users expect and take into consideration how polars might handle NaN vs Null vs np.nan vs None, etc
    def test_fillna_replaces_missing_values(self):
        """
        .fillna() should replace NaN/None with specified value.

        Expected behavior:
        - Returns new DesignMatrix
        - Original unchanged
        - All NaN replaced with fill value

        Use case: Fill missing motion covariates with 0
        """
        dm = DesignMatrix(
            {"a": [1.0, None, 3.0], "b": [None, 2.0, None]}, sampling_freq=1
        )
        dm_filled = dm.fillna(0)

        # New DesignMatrix has no NaNs
        assert dm_filled["a"].to_list() == [1.0, 0.0, 3.0]
        assert dm_filled["b"].to_list() == [0.0, 2.0, 0.0]

        # Original unchanged (immutability check)
        assert dm["a"][1] is None

    def test_drop_columns_removes_specified_columns(self):
        """
        .drop() should remove specified columns.

        Expected behavior:
        - Returns new DesignMatrix without dropped columns
        - Original unchanged
        - Other columns preserved

        Use case: Remove redundant or problematic regressors
        """
        dm = DesignMatrix({"a": [1], "b": [2], "c": [3]}, sampling_freq=1)
        dm_dropped = dm.drop(columns=["b"])

        assert dm_dropped.columns == ["a", "c"]
        assert dm.columns == ["a", "b", "c"], "Original should be unchanged"

    def test_drop_multiple_columns(self):
        """
        .drop() should handle multiple columns at once.
        """
        dm = DesignMatrix({"a": [1], "b": [2], "c": [3], "d": [4]}, sampling_freq=1)
        dm_dropped = dm.drop(columns=["b", "d"])

        assert dm_dropped.columns == ["a", "c"]

    def test_transformations_preserve_metadata(self):
        """
        Verify that transformations preserve metadata attributes.

        Expected behavior:
        - sampling_freq preserved
        - polys list preserved
        - convolved list preserved
        - multi flag preserved
        """
        dm = DesignMatrix({"a": [1, 2, 3]}, sampling_freq=2)
        dm.polys = ["poly_0"]
        dm.convolved = ["stim"]
        dm.multi = True

        dm_filled = dm.fillna(0)

        assert dm_filled.sampling_freq == 2
        assert dm_filled.polys == ["poly_0"]
        assert dm_filled.convolved == ["stim"]
        assert dm_filled.multi is True


# ============================================================================
# 4. Statistical Operations Tests
# ============================================================================


# NOTE: Make sure these as all re-implemented efficiently to take-advantage of polars vectorized operations as their documentation suggests!
class TestDesignMatrixStatisticalOperations:
    """
    Test statistical transformation methods.

    Behavioral contract:
    - zscore() standardizes to mean=0, std=1
    - downsample() reduces temporal resolution
    - upsample() increases temporal resolution
    """

    def test_zscore_standardizes_single_column(self):
        """
        .zscore() on single column should produce mean=0, std=1.

        Expected behavior:
        - Specified column has mean ≈ 0, std ≈ 1
        - Other columns unchanged
        - Returns new DesignMatrix

        Use case: Standardize predictors for regularization
        """
        dm = DesignMatrix(
            {"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]}, sampling_freq=1
        )
        dm_z = dm.zscore(columns=["a"])

        assert dm_z["a"].mean() == pytest.approx(0.0, abs=1e-10), "Mean should be 0"
        assert dm_z["a"].std() == pytest.approx(1.0, abs=1e-10), "Std should be 1"
        assert dm_z["b"].to_list() == dm["b"].to_list(), "Unspecified columns unchanged"

    def test_zscore_all_columns_by_default(self):
        """
        .zscore() without arguments should standardize all columns.

        Expected behavior:
        - All columns standardized
        - Each has mean=0, std=1
        """
        dm = DesignMatrix({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}, sampling_freq=1)
        dm_z = dm.zscore()

        assert dm_z["a"].mean() == pytest.approx(0.0, abs=1e-10)
        assert dm_z["a"].std() == pytest.approx(1.0, abs=1e-10)
        assert dm_z["b"].mean() == pytest.approx(0.0, abs=1e-10)
        assert dm_z["b"].std() == pytest.approx(1.0, abs=1e-10)

    def test_zscore_excludes_polynomial_columns(self):
        """
        .zscore() should NOT standardize polynomial columns by default.

        Expected behavior:
        - Columns in .polys list are skipped
        - Only non-polynomial columns standardized

        Rationale: Polynomials (intercept, trends) should not be standardized
        """
        dm = DesignMatrix(
            {"stim": [1, 2, 3, 4], "poly_0": [1, 1, 1, 1]}, sampling_freq=1
        )
        dm.polys = ["poly_0"]

        dm_z = dm.zscore()

        assert dm_z["stim"].mean() == pytest.approx(0.0, abs=1e-10), (
            "Stim should be standardized"
        )
        assert dm_z["poly_0"].to_list() == [1, 1, 1, 1], (
            "Polynomial should be unchanged"
        )

    def test_downsample_reduces_sampling_rate(self):
        """
        .downsample() should reduce temporal resolution.

        Expected behavior:
        - Fewer rows in output
        - sampling_freq updated to target
        - Data appropriately aggregated (e.g., mean)

        Use case: Match design matrix to lower TR acquisition
        """
        # Create 100 samples at 1 Hz
        dm = DesignMatrix({"a": list(range(100))}, sampling_freq=1.0)

        # Downsample to 0.5 Hz (slower sampling = fewer samples)
        dm_down = dm.downsample(target=0.5)

        assert dm_down.shape[0] < dm.shape[0], "Should have fewer rows"
        assert dm_down.sampling_freq == 0.5, "Sampling freq should be updated"

        # Verify exact values (Polars-native implementation)
        # Downsample groups every 2 samples and takes mean: [0,1]→0.5, [2,3]→2.5, etc.
        expected_first_10 = [0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 18.5]
        assert dm_down["a"].to_list()[:10] == expected_first_10, (
            "Polars downsample should produce correct aggregated values"
        )

    def test_downsample_polars_native_correctness(self):
        """
        Verify Polars-native downsample() produces correct results.

        Tests:
        1. Multiple columns handled correctly
        2. Different downsample ratios work
        3. Median aggregation method works
        4. Edge case: non-integer n_samples
        """
        # Test 1: Multiple columns
        dm = DesignMatrix(
            {"x": list(range(10)), "y": list(range(10, 20))}, sampling_freq=1.0
        )
        dm_down = dm.downsample(target=0.5)
        assert dm_down["x"].to_list() == [0.5, 2.5, 4.5, 6.5, 8.5]
        assert dm_down["y"].to_list() == [10.5, 12.5, 14.5, 16.5, 18.5]

        # Test 2: Different ratio (1 Hz → 0.25 Hz, groups of 4)
        dm2 = DesignMatrix({"b": list(range(100))}, sampling_freq=1.0)
        dm2_down = dm2.downsample(target=0.25)
        assert dm2_down.shape == (25, 1)
        assert dm2_down["b"].to_list()[:5] == [1.5, 5.5, 9.5, 13.5, 17.5]

        # Test 3: Median method
        dm3 = DesignMatrix({"a": list(range(10))}, sampling_freq=1.0)
        dm3_median = dm3.downsample(target=0.5, method="median")
        assert dm3_median["a"].to_list() == [0.5, 2.5, 4.5, 6.5, 8.5]

        # Test 4: Non-integer n_samples (e.g., 10 samples, 1.0 Hz → 0.3 Hz)
        # n_samples = 1.0 / 0.3 ≈ 3.33, should still work
        dm4 = DesignMatrix({"c": list(range(10))}, sampling_freq=1.0)
        dm4_down = dm4.downsample(target=0.3)
        assert dm4_down.sampling_freq == 0.3
        assert dm4_down.shape[0] < dm4.shape[0]

    def test_upsample_increases_sampling_rate(self):
        """
        .upsample() should increase temporal resolution.

        Expected behavior:
        - More rows in output
        - sampling_freq updated to target
        - Data appropriately interpolated

        Use case: Match design matrix to faster TR acquisition
        """
        # Create 10 samples at 1 Hz
        dm = DesignMatrix({"a": list(range(10))}, sampling_freq=1.0)

        # Upsample to 2 Hz (faster sampling = more samples)
        dm_up = dm.upsample(target=2.0)

        assert dm_up.shape[0] > dm.shape[0], "Should have more rows"
        assert dm_up.sampling_freq == 2.0, "Sampling freq should be updated"

    def test_upsample_linear_interpolation_correctness(self):
        """
        .upsample() with linear interpolation should produce correct values.

        Tests Polars-native implementation against expected behavior:
        - Correct number of samples calculated
        - Linear interpolation produces mathematically correct values
        - Multiple columns handled correctly
        """
        # Create simple linear data: [0, 1, 2, 3, 4] at 1 Hz
        dm = DesignMatrix({"a": [0.0, 1.0, 2.0, 3.0, 4.0]}, sampling_freq=1.0)

        # Upsample to 2 Hz (2x sampling rate)
        # Original: 0, 1, 2, 3, 4 (5 samples spanning 0-4)
        # Step size: 1.0/2.0 = 0.5
        # New spacing: np.arange(0, 4, 0.5) = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5] (8 samples)
        dm_up = dm.upsample(target=2.0)

        # Check shape (8 samples, not 9 - arange excludes endpoint)
        assert dm_up.shape[0] == 8, f"Expected 8 samples, got {dm_up.shape[0]}"
        assert dm_up.sampling_freq == 2.0

        # Check interpolated values (linear interpolation)
        expected = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        np.testing.assert_allclose(dm_up["a"], expected, rtol=1e-10)

    def test_upsample_method_parameter(self):
        """
        .upsample() should support method='nearest' for nearest-neighbor interpolation.
        """
        dm = DesignMatrix({"a": [0.0, 1.0, 2.0, 3.0, 4.0]}, sampling_freq=1.0)

        # Test linear method (default)
        dm_linear = dm.upsample(target=2.0, method="linear")
        expected_linear = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        np.testing.assert_allclose(dm_linear["a"], expected_linear, rtol=1e-10)

        # Test nearest method
        dm_nearest = dm.upsample(target=2.0, method="nearest")
        # Nearest neighbor: values should snap to closest original value
        expected_nearest = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        np.testing.assert_allclose(dm_nearest["a"], expected_nearest, rtol=1e-10)


# ============================================================================
# 5. Convolution Tests
# ============================================================================


class TestDesignMatrixConvolution:
    """
    Test HRF convolution functionality.

    Behavioral contract:
    - Default convolution uses Glover HRF
    - Can provide custom kernels
    - Polynomial columns excluded from convolution
    - .convolved metadata tracks which columns were convolved
    """

    def test_convolve_with_default_hrf_delays_response(self):
        """
        Default HRF convolution should delay and smooth response.

        Expected behavior:
        - Peak shifts later in time (HRF peaks ~5-6s after stimulus)
        - Signal is smoothed (convolution blurs sharp edges)

        Use case: Model hemodynamic response in fMRI
        """
        # Box-car stimulus: on at TRs 2-4, off otherwise
        dm = DesignMatrix(
            {"stim": [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]},
            sampling_freq=0.5,  # 2s TR
        )
        dm_conv = dm.convolve()

        # Peak should shift later due to HRF delay
        original_peak_idx = dm["stim"].arg_max()
        convolved_peak_idx = dm_conv["stim"].arg_max()

        assert convolved_peak_idx > original_peak_idx, (
            "HRF convolution should delay peak response"
        )

    # NOTE: DesignMatrix may not currently have this feature, but we've always wanted to support multiple kernels as well passed in a list of inputs that automatically creates columns * kernels new convovled columns. See if it's easy to support this
    def test_convolve_with_custom_kernel(self):
        """
        Convolution with custom kernel should use provided function.

        Expected behavior:
        - Custom kernel applied via convolution
        - Results differ from default HRF

        Use case: Model non-canonical HRF, or other response functions (SCR, pupil)
        """
        dm = DesignMatrix({"stim": [1, 0, 0, 0, 0]}, sampling_freq=1)

        # Custom kernel: simple 3-point average (box-car smoothing)
        kernel = np.array([0.33, 0.33, 0.33])
        dm_conv = dm.convolve(conv_func=kernel)

        # First value should be smoothed
        assert dm_conv["stim"].to_list()[0] == pytest.approx(0.33, abs=0.01)

    def test_convolve_ignores_polynomial_columns(self):
        """
        Convolution should skip columns marked as polynomials.

        Expected behavior:
        - Columns in .polys list are NOT convolved
        - Only stimulus columns are convolved

        Rationale: Polynomials (intercept, drift) represent baseline, not stimulus
        """
        dm = DesignMatrix(
            {"stim": [1, 0, 0, 0], "intercept": [1, 1, 1, 1]}, sampling_freq=1
        )
        dm.polys = ["intercept"]

        dm_conv = dm.convolve()

        # Intercept should be unchanged
        assert dm_conv["intercept"].to_list() == [1, 1, 1, 1], (
            "Polynomial columns should not be convolved"
        )

    def test_convolve_specific_columns_only(self):
        """
        Can specify which columns to convolve.

        Expected behavior:
        - Only specified columns are convolved
        - Other columns unchanged

        Use case: Convolve task regressors but not parametric modulators
        """
        dm = DesignMatrix(
            {"stim_A": [1, 0, 0, 0], "stim_B": [0, 1, 0, 0], "baseline": [1, 1, 1, 1]},
            sampling_freq=1,
        )

        dm_conv = dm.convolve(columns=["stim_A"])

        # Only stim_A should be different
        assert dm_conv["stim_A"].to_list() != dm["stim_A"].to_list(), (
            "Specified column should be convolved"
        )
        assert dm_conv["stim_B"].to_list() == dm["stim_B"].to_list(), (
            "Unspecified column should be unchanged"
        )
        assert dm_conv["baseline"].to_list() == dm["baseline"].to_list()

    def test_convolve_updates_metadata(self):
        """
        Convolution should update .convolved metadata.

        Expected behavior:
        - .convolved list contains names of convolved columns
        - Metadata persists in returned DesignMatrix

        Use case: Track which regressors have been convolved
        """
        dm = DesignMatrix({"stim": [1, 0, 0, 0]}, sampling_freq=1)
        dm_conv = dm.convolve(columns=["stim"])

        assert "stim" in dm_conv.convolved, (
            "Convolved columns should be tracked in metadata"
        )

    # NOTE: see earlier note
    def test_convolve_with_multiple_kernels(self):
        """
        Support convolution with multiple kernels (2D array).

        Expected behavior:
        - Each column convolved with multiple kernels
        - New columns created for each kernel variant
        - Column names suffixed with kernel index (e.g., 'stim_c0', 'stim_c1')

        Use case: FIR models, temporal derivatives
        """
        dm = DesignMatrix({"stim": [1, 0, 0, 0, 0, 0]}, sampling_freq=1)

        # Two simple kernels
        kernels = np.array(
            [
                [1.0, 0.5, 0.0],  # Kernel 0: quick rise
                [0.0, 0.5, 1.0],  # Kernel 1: delayed rise
            ]
        ).T  # Shape: (3, 2) - samples x kernels

        dm_conv = dm.convolve(conv_func=kernels)

        # Should create stim_c0 and stim_c1
        assert "stim_c0" in dm_conv.columns
        assert "stim_c1" in dm_conv.columns


# ============================================================================
# 6. Polynomial Tests
# ============================================================================


class TestDesignMatrixPolynomials:
    """
    Test polynomial and DCT basis function addition.

    Behavioral contract:
    - add_poly() adds Legendre polynomials (orthogonal on [-1, 1])
    - add_dct_basis() adds discrete cosine transform basis (high-pass filter)
    - .polys metadata tracks polynomial columns
    - Polynomials are NOT duplicated if added twice
    """

    def test_add_poly_creates_legendre_polynomials(self):
        """
        .add_poly(order=2) should add polynomials of order 0, 1, 2.

        Expected behavior:
        - Creates poly_0 (intercept), poly_1 (linear), poly_2 (quadratic)
        - All columns present in output
        - .polys metadata updated

        Use case: Model baseline and slow drift in fMRI
        """
        dm = DesignMatrix({"stim": [1, 0, 0, 0] * 10}, sampling_freq=1)
        dm_poly = dm.add_poly(order=2, include_lower=True)

        # Should add 3 polynomial columns
        assert dm_poly.shape[1] == 4, "Should have stim + 3 polynomials"
        assert "poly_0" in dm_poly.columns
        assert "poly_1" in dm_poly.columns
        assert "poly_2" in dm_poly.columns

        # Metadata should track polynomials
        assert set(dm_poly.polys) == {"poly_0", "poly_1", "poly_2"}

    def test_add_poly_intercept_is_constant(self):
        """
        poly_0 (order=0) should be constant intercept term.

        Expected behavior:
        - Mean ≈ 1.0 (or some constant)
        - Variance ≈ 0 (constant across rows)

        Rationale: Legendre polynomial of order 0 is constant
        """
        dm = DesignMatrix({"stim": [1, 2, 3, 4]}, sampling_freq=1)
        dm_poly = dm.add_poly(order=0)

        # Intercept should be constant (very low variance)
        poly_0 = dm_poly["poly_0"]
        assert poly_0.std() < 1e-10, "Intercept should have near-zero variance"

    def test_add_poly_linear_trend(self):
        """
        poly_1 (order=1) should be linear trend.

        Expected behavior:
        - Monotonic increase or decrease
        - First and last values have opposite signs (scaled -1 to 1)

        Use case: Model linear drift in signal
        """
        dm = DesignMatrix(np.zeros((20, 1)), sampling_freq=1, columns=["stim"])
        dm_poly = dm.add_poly(order=1, include_lower=False)

        poly_1 = dm_poly["poly_1"]

        # Should be monotonic (always increasing or decreasing)
        diffs = np.diff(poly_1.to_numpy())
        assert np.all(diffs > 0) or np.all(diffs < 0), "Should be monotonic"

    def test_add_poly_without_lower_terms(self):
        """
        include_lower=False should add only specified order.

        Expected behavior:
        - Only poly_2 added, not poly_0 or poly_1

        Use case: Add specific polynomial without lower orders
        """
        dm = DesignMatrix({"stim": [1, 2, 3, 4]}, sampling_freq=1)
        dm_poly = dm.add_poly(order=2, include_lower=False)

        assert dm_poly.shape[1] == 2, "Should have stim + poly_2 only"
        assert "poly_2" in dm_poly.columns
        assert "poly_0" not in dm_poly.columns
        assert "poly_1" not in dm_poly.columns

    def test_add_poly_idempotent(self):
        """
        Adding same polynomial twice should skip (no duplicates).

        Expected behavior:
        - Second call to add_poly(order=1) does nothing
        - Column count unchanged
        - Warning message printed (optional)

        Rationale: Prevents accidental duplication
        """
        dm = DesignMatrix({"stim": [1, 2, 3, 4]}, sampling_freq=1)
        dm1 = dm.add_poly(order=1)
        dm2 = dm1.add_poly(order=1)

        assert dm1.shape == dm2.shape, "Should not duplicate polynomials"
        assert dm1.columns == dm2.columns

    def test_add_dct_basis_creates_cosine_filters(self):
        """
        .add_dct_basis() should add discrete cosine basis functions.

        Expected behavior:
        - Multiple cosine_* columns added
        - Number of bases depends on duration and sampling_freq
        - .polys metadata updated

        Use case: High-pass filtering (SPM-style)
        """
        dm = DesignMatrix(
            np.zeros((100, 1)),
            sampling_freq=0.5,  # 2s TR
            columns=["stim"],
        )

        dm_dct = dm.add_dct_basis(duration=60)  # 60s filter

        # Should add multiple cosine basis functions
        cosine_cols = [c for c in dm_dct.columns if "cosine" in c]
        assert len(cosine_cols) > 1, "Should add multiple DCT bases"

        # Metadata should track
        assert "cosine_1" in dm_dct.polys

    def test_add_dct_basis_drop_parameter(self):
        """
        drop parameter should exclude low-frequency bases.

        Expected behavior:
        - drop=2 skips first 2 basis functions (constant and slowest)
        - Remaining bases start from index 3

        Use case: Remove very slow drifts beyond typical DCT filtering
        """
        dm = DesignMatrix(np.zeros((100, 1)), sampling_freq=0.5, columns=["stim"])

        # Drop first 2 bases (including constant, like SPM)
        dm_dct = dm.add_dct_basis(duration=60, drop=2)

        # Should not have cosine_1 or cosine_2
        assert "cosine_1" not in dm_dct.columns
        assert "cosine_2" not in dm_dct.columns
        # Should have higher-order bases
        assert (
            "cosine_3" in dm_dct.columns or "cosine_1" in dm_dct.columns
        )  # Depends on numbering convention


# ============================================================================
# 7. Concatenation Tests (Most Complex and Critical)
# ============================================================================


class TestDesignMatrixConcatenation:
    """
    Test multi-run concatenation logic - the most complex functionality.

    Behavioral contract:
    - Horizontal (axis=1): Combine columns from multiple DesignMatrix
    - Vertical (axis=0): Stack rows, with intelligent polynomial separation
    - keep_separate=True: Automatically separate polynomial columns across runs
    - unique_cols: User-specified columns to keep separated
    - Wildcard support: 'house*' matches house_A, house_B
    - Auto-numbering: 0_poly_0, 1_poly_0, 2_poly_0 for multi-run
    """

    def test_horizontal_append_adds_columns(self):
        """
        Horizontal concatenation (axis=1) should combine columns.

        Expected behavior:
        - Columns from both DesignMatrix combined
        - Row count unchanged
        - Metadata from both preserved

        Use case: Add motion covariates to design matrix
        """
        dm1 = DesignMatrix({"a": [1, 2]}, sampling_freq=1)
        dm2 = DesignMatrix({"b": [3, 4]}, sampling_freq=1)

        dm_combined = dm1.append(dm2, axis=1)

        assert dm_combined.shape == (2, 2), "Should have 2 rows, 2 columns"
        assert set(dm_combined.columns) == {"a", "b"}

    def test_horizontal_append_multiple_columns(self):
        """
        Horizontal append can add multiple columns at once.
        """
        dm1 = DesignMatrix({"a": [1]}, sampling_freq=1)
        dm2 = DesignMatrix({"b": [2], "c": [3]}, sampling_freq=1)

        dm_combined = dm1.append(dm2, axis=1)

        assert dm_combined.shape == (1, 3)
        assert set(dm_combined.columns) == {"a", "b", "c"}

    def test_vertical_append_stacks_rows_simple(self):
        """
        Vertical concatenation (axis=0) without separation stacks rows.

        Expected behavior:
        - Same columns, more rows
        - Data concatenated vertically
        - keep_separate=False means no column renaming

        Use case: Combine data from same run split into chunks
        """
        dm1 = DesignMatrix({"a": [1, 2]}, sampling_freq=1)
        dm2 = DesignMatrix({"a": [3, 4]}, sampling_freq=1)

        dm_combined = dm1.append(dm2, axis=0, keep_separate=False)

        assert dm_combined.shape == (4, 1), "Should have 4 rows, 1 column"
        assert dm_combined["a"].to_list() == [1, 2, 3, 4]

    def test_vertical_append_separates_polynomials_automatically(self):
        """
        CRITICAL: Multi-run appending with keep_separate=True.

        This is the core feature that makes nltools DesignMatrix special.
        When appending multiple runs vertically, polynomial columns (intercepts,
        trends) should be automatically separated because each run needs its
        own baseline.

        Expected behavior:
        - Stimulus columns shared across runs (NOT duplicated)
        - Polynomial columns separated with run prefix: 0_poly_0, 1_poly_0
        - .multi flag set to True
        - Run 1 polynomials active only in run 1 rows (others filled with 0)
        - Run 2 polynomials active only in run 2 rows

        Use case: Standard multi-run fMRI analysis
        """
        # Run 1: 4 TRs with stimulus and intercept
        dm1 = DesignMatrix({"stim": [1, 0, 0, 0]}, sampling_freq=1)
        dm1 = dm1.add_poly(order=0)  # Adds 'poly_0' (intercept)

        # Run 2: 4 TRs with different stimulus timing, same intercept
        dm2 = DesignMatrix({"stim": [0, 1, 0, 0]}, sampling_freq=1)
        dm2 = dm2.add_poly(order=0)

        # Append with automatic polynomial separation
        dm_runs = dm1.append(dm2, axis=0, keep_separate=True)

        # Verify structure
        assert dm_runs.shape == (8, 3), (
            "Should have 8 rows (4+4), 3 columns (stim, 0_poly_0, 1_poly_0)"
        )
        assert "stim" in dm_runs.columns, "Stimulus should be shared"
        assert "0_poly_0" in dm_runs.columns, "Run 1 intercept should be separated"
        assert "1_poly_0" in dm_runs.columns, "Run 2 intercept should be separated"
        assert dm_runs.multi is True, "Multi-run flag should be set"

        # Verify separation: run 1 intercept active only in first 4 rows
        run1_intercept = dm_runs["0_poly_0"].to_list()
        run2_intercept = dm_runs["1_poly_0"].to_list()

        assert sum(run1_intercept[:4]) > 0, (
            "Run 1 intercept should be active in first 4 rows"
        )
        assert sum(run1_intercept[4:]) == 0, (
            "Run 1 intercept should be 0 in last 4 rows"
        )
        assert sum(run2_intercept[:4]) == 0, (
            "Run 2 intercept should be 0 in first 4 rows"
        )
        assert sum(run2_intercept[4:]) > 0, (
            "Run 2 intercept should be active in last 4 rows"
        )

    def test_vertical_append_unique_cols_exact_match(self):
        """
        unique_cols parameter specifies additional columns to separate.

        Expected behavior:
        - Columns in unique_cols are separated like polynomials
        - Other columns remain shared

        Use case: Separate run-specific motion covariates but share task regressors
        """
        dm1 = DesignMatrix(
            {"motion_x": [1, 2], "motion_y": [3, 4], "stim": [1, 0]}, sampling_freq=1
        )

        dm2 = DesignMatrix(
            {"motion_x": [5, 6], "motion_y": [7, 8], "stim": [0, 1]}, sampling_freq=1
        )

        dm_runs = dm1.append(dm2, axis=0, unique_cols=["motion_x", "motion_y"])

        # Motion columns should be separated
        assert "0_motion_x" in dm_runs.columns
        assert "0_motion_y" in dm_runs.columns
        assert "1_motion_x" in dm_runs.columns
        assert "1_motion_y" in dm_runs.columns

        # Stimulus should be shared (not separated)
        assert "stim" in dm_runs.columns
        assert "0_stim" not in dm_runs.columns

    def test_vertical_append_unique_cols_wildcard_prefix(self):
        """
        Wildcard 'house*' should match all columns starting with 'house'.

        Expected behavior:
        - house_A and house_B both matched by 'house*'
        - Both separated across runs
        - Other columns (face_A) not affected

        Use case: Separate all regressors for a category (e.g., all house stimuli)
        """
        dm1 = DesignMatrix(
            {"house_A": [1, 0], "house_B": [0, 1], "face_A": [1, 1]}, sampling_freq=1
        )

        dm2 = DesignMatrix(
            {"house_A": [0, 1], "house_B": [1, 0], "face_A": [1, 1]}, sampling_freq=1
        )

        dm_runs = dm1.append(dm2, axis=0, unique_cols=["house*"])

        # House columns separated
        assert "0_house_A" in dm_runs.columns
        assert "0_house_B" in dm_runs.columns
        assert "1_house_A" in dm_runs.columns
        assert "1_house_B" in dm_runs.columns

        # Face column shared
        assert "face_A" in dm_runs.columns
        assert "0_face_A" not in dm_runs.columns

    def test_vertical_append_unique_cols_wildcard_suffix(self):
        """
        Wildcard '*_motion' should match all columns ending with '_motion'.

        Expected behavior:
        - x_motion and y_motion matched by '*_motion'
        - Both separated across runs

        Use case: Separate all motion-related regressors
        """
        dm1 = DesignMatrix(
            {"x_motion": [1, 2], "y_motion": [3, 4], "stim": [1, 0]}, sampling_freq=1
        )

        dm2 = DesignMatrix(
            {"x_motion": [5, 6], "y_motion": [7, 8], "stim": [0, 1]}, sampling_freq=1
        )

        dm_runs = dm1.append(dm2, axis=0, unique_cols=["*_motion"])

        assert "0_x_motion" in dm_runs.columns
        assert "1_y_motion" in dm_runs.columns
        assert "stim" in dm_runs.columns

    def test_vertical_append_multiple_runs_increments_numbering(self):
        """
        Appending 3+ runs should correctly increment run numbering.

        Expected behavior:
        - First append creates 0_* and 1_*
        - Second append creates 2_*
        - Numbering continues sequentially

        Use case: Realistic multi-run experiment (e.g., 4-6 runs)
        """
        dm1 = DesignMatrix({"s": [1]}, sampling_freq=1).add_poly(0)
        dm2 = DesignMatrix({"s": [2]}, sampling_freq=1).add_poly(0)
        dm3 = DesignMatrix({"s": [3]}, sampling_freq=1).add_poly(0)

        # Chain appends
        dm_runs = dm1.append(dm2, axis=0).append(dm3, axis=0)

        assert "0_poly_0" in dm_runs.columns, "Run 1 intercept"
        assert "1_poly_0" in dm_runs.columns, "Run 2 intercept"
        assert "2_poly_0" in dm_runs.columns, "Run 3 intercept"

    def test_vertical_append_fill_na_fills_missing_columns(self):
        """
        Mismatched columns should be filled with fill_na value.

        Expected behavior:
        - Run 1 has column 'a', run 2 has column 'b'
        - Result has both columns
        - Missing values filled with fill_na (default=0)

        Use case: Different covariates across runs
        """
        dm1 = DesignMatrix({"a": [1, 2]}, sampling_freq=1)
        dm2 = DesignMatrix({"b": [3, 4]}, sampling_freq=1)

        dm_combined = dm1.append(dm2, axis=0, fill_na=0)

        assert dm_combined.shape == (4, 2), "Should have both columns"

        # Check fill pattern
        assert dm_combined["a"].to_list() == [1, 2, 0, 0], "a should be 0 in run 2"
        assert dm_combined["b"].to_list() == [0, 0, 3, 4], "b should be 0 in run 1"

    def test_vertical_append_list_of_design_matrices(self):
        """
        .append() can take a list of DesignMatrix objects.

        Expected behavior:
        - Accept list [dm2, dm3] and append all at once
        - Same result as chaining .append() calls

        Use case: Convenience for multi-run loops
        """
        dm1 = DesignMatrix({"a": [1]}, sampling_freq=1)
        dm2 = DesignMatrix({"a": [2]}, sampling_freq=1)
        dm3 = DesignMatrix({"a": [3]}, sampling_freq=1)

        # Append list of DesignMatrix
        dm_combined = dm1.append([dm2, dm3], axis=0, keep_separate=False)

        assert dm_combined.shape == (3, 1)
        assert dm_combined["a"].to_list() == [1, 2, 3]

    def test_vertical_append_preserves_polynomial_metadata(self):
        """
        After separation, .polys metadata should contain all separated poly names.

        Expected behavior:
        - .polys list contains '0_poly_0', '1_poly_0', etc.
        - Original 'poly_0' not in metadata (replaced by separated versions)

        Use case: Track which columns are polynomials for later operations
        """
        dm1 = DesignMatrix({"s": [1]}, sampling_freq=1).add_poly(0)
        dm2 = DesignMatrix({"s": [2]}, sampling_freq=1).add_poly(0)

        dm_runs = dm1.append(dm2, axis=0, keep_separate=True)

        assert "0_poly_0" in dm_runs.polys
        assert "1_poly_0" in dm_runs.polys
        assert "poly_0" not in dm_runs.polys  # Original name replaced


# ============================================================================
# 8. Diagnostic Tests
# ============================================================================


class TestDesignMatrixDiagnostics:
    """
    Test collinearity diagnostics and cleaning.

    Behavioral contract:
    - vif() computes variance inflation factor
    - clean() removes highly correlated columns
    - Correlation threshold configurable
    """

    # NOTE: make sure this implementation is also as efficient as possible using built-in polars functionality
    def test_vif_computes_variance_inflation_factor(self):
        """
        VIF measures collinearity: VIF > 10 indicates problematic correlation.

        Expected behavior:
        - Returns array of VIF values (one per column)
        - With perfect collinearity, returns None and prints error

        Use case: Detect multicollinearity before regression
        """
        # Create intentionally collinear columns (perfect collinearity)
        dm_perfect = DesignMatrix(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [2, 4, 6, 8, 10],  # b = 2*a, perfect collinearity
                "c": [1, 1, 2, 2, 3],  # Moderately correlated
            },
            sampling_freq=1,
        )

        # Perfect collinearity should return None with error message
        vifs_perfect = dm_perfect.vif(exclude_polys=True)
        assert vifs_perfect is None, "Perfect collinearity should return None"

        # Create data with high but not perfect correlation
        dm_high = DesignMatrix(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [1.9, 4.1, 5.8, 8.2, 9.9],  # High but not perfect correlation
                "c": [1, 1, 2, 2, 3],
            },
            sampling_freq=1,
        )

        vifs_high = dm_high.vif(exclude_polys=True)
        assert vifs_high is not None, (
            "Should return VIF values for non-perfect correlation"
        )
        assert len(vifs_high) == 3, "Should have VIF for each column"
        # VIF should be elevated for highly correlated columns
        assert any(v > 5 for v in vifs_high), "Should detect high collinearity"

    def test_vif_excludes_polynomial_columns_by_default(self):
        """
        exclude_polys=True should not compute VIF for polynomial columns.

        Expected behavior:
        - Polynomials skipped in VIF calculation
        - Only data columns included

        Rationale: Polynomials are expected to be correlated (e.g., linear and quadratic)
        """
        dm = DesignMatrix({"a": [1, 2, 3, 4, 5]}, sampling_freq=1)
        dm = dm.add_poly(order=2)  # Adds poly_0, poly_1, poly_2

        vifs = dm.vif(exclude_polys=True)

        # Should only compute VIF for 'a', not polynomials
        assert len(vifs) == 1, "Should only compute VIF for non-polynomial columns"

    def test_clean_removes_highly_correlated_columns(self):
        """
        .clean() drops columns correlated above threshold.

        Expected behavior:
        - Columns with r >= thresh are dropped
        - First instance of correlated pair is kept
        - Returns new DesignMatrix

        Use case: Remove redundant regressors before fitting
        """
        dm = DesignMatrix(
            {
                "a": [1, 2, 3, 4],
                "b": [1.01, 2.01, 3.01, 4.01],  # r ≈ 1.0 with 'a'
                "c": [4, 3, 2, 1],  # Uncorrelated with a, b
            },
            sampling_freq=1,
        )

        dm_clean = dm.clean(thresh=0.95)

        # Should keep 'a' (first instance), drop 'b' (correlated duplicate)
        assert "a" in dm_clean.columns, "First instance should be kept"
        assert "b" not in dm_clean.columns, "Highly correlated column should be dropped"
        assert "c" in dm_clean.columns, "Uncorrelated column should be kept"

    def test_clean_with_lower_threshold(self):
        """
        Lower threshold should drop more columns.

        Expected behavior:
        - thresh=0.8 more aggressive than thresh=0.95
        - Moderately correlated columns also dropped
        """
        dm = DesignMatrix(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [1.1, 2.2, 3.1, 3.9, 5.1],  # r ≈ 0.99 with 'a'
                "c": [2, 3, 4, 5, 6],  # r ≈ 1.0 with 'a'
            },
            sampling_freq=1,
        )

        dm_clean = dm.clean(thresh=0.8)

        # Both b and c should be dropped (highly correlated with a)
        assert "a" in dm_clean.columns
        assert dm_clean.shape[1] < dm.shape[1], "Should drop correlated columns"

    def test_clean_excludes_polys_from_collinearity_check(self):
        """
        exclude_polys=True should keep polynomial columns even if correlated.

        Expected behavior:
        - Polynomials not checked for collinearity
        - Preserved in output even if correlated with each other

        Rationale: Polynomial trends naturally correlated, but needed for modeling
        """
        dm = DesignMatrix({"a": [1, 2, 3, 4]}, sampling_freq=1)
        dm = dm.add_poly(order=2)  # poly_1 and poly_2 might be correlated

        dm_clean = dm.clean(exclude_polys=True)

        # All polynomials should be retained
        assert "poly_0" in dm_clean.columns
        assert "poly_1" in dm_clean.columns
        assert "poly_2" in dm_clean.columns

    def test_clean_fillna_before_checking(self):
        """
        fill_na parameter should fill NaNs before correlation check.

        Expected behavior:
        - NaNs filled with specified value
        - Correlation computed on filled data

        Use case: Handle missing motion data before diagnostics
        """
        dm = DesignMatrix(
            {"a": [1.0, None, 3.0, 4.0], "b": [2.0, None, 6.0, 8.0]}, sampling_freq=1
        )

        dm_clean = dm.clean(fill_na=0, thresh=0.95)

        # Should successfully compute correlations after filling
        assert dm_clean.shape[0] == 4, "All rows should be preserved"


# ============================================================================
# 9. Utility Tests
# ============================================================================


class TestDesignMatrixUtilities:
    """
    Test miscellaneous utility methods.

    Behavioral contract:
    - details() returns human-readable metadata string
    - replace_data() swaps data while preserving polynomials
    - heatmap() creates visualization (tested separately)
    """

    def test_details_shows_metadata(self):
        """
        .details() should return string with metadata summary.

        Expected behavior:
        - Contains sampling_freq
        - Contains shape
        - Lists convolved columns
        - Lists polynomial columns

        Use case: Quick inspection of DesignMatrix state
        """
        dm = DesignMatrix({"a": [1, 2, 3]}, sampling_freq=2)
        dm = dm.add_poly(0)
        dm = dm.convolve(columns=["a"])

        details = dm.details()

        assert "sampling_freq=2" in details
        assert "shape=(3, 2)" in details or "(3, 2)" in details
        assert "poly_0" in details or "polys" in details
        assert "convolved" in details

    def test_replace_data_keeps_metadata_and_polys(self):
        """
        .replace_data() swaps data columns but preserves polynomial columns.

        Expected behavior:
        - Old data columns removed
        - New data columns added (with provided names)
        - Polynomial columns unchanged
        - Metadata preserved

        Use case: Substitute stimulus regressors while keeping drift terms
        """
        dm = DesignMatrix({"a": [1, 2, 3], "b": [4, 5, 6]}, sampling_freq=2)
        dm = dm.add_poly(order=0)  # Adds poly_0

        # Replace data with new columns
        new_data = np.array([[10, 20], [30, 40], [50, 60]])
        dm_replaced = dm.replace_data(new_data, column_names=["x", "y"])

        # New data columns should be present
        assert "x" in dm_replaced.columns
        assert "y" in dm_replaced.columns
        assert dm_replaced["x"].to_list() == [10, 30, 50]

        # Old data columns should be gone
        assert "a" not in dm_replaced.columns
        assert "b" not in dm_replaced.columns

        # Polynomials should be preserved
        assert "poly_0" in dm_replaced.columns
        assert dm_replaced["poly_0"].to_list() == dm["poly_0"].to_list()

        # Metadata should be preserved
        assert dm_replaced.sampling_freq == 2

    def test_replace_data_validates_row_count(self):
        """
        .replace_data() should error if new data has different number of rows.

        Expected behavior:
        - Raise ValueError if row count mismatch
        - Prevents invalid state

        Rationale: Polynomial columns have specific length, can't change
        """
        dm = DesignMatrix({"a": [1, 2, 3]}, sampling_freq=1)
        dm = dm.add_poly(0)

        # Try to replace with different row count
        new_data = np.array([[10], [20]])  # Only 2 rows, original has 3

        with pytest.raises(ValueError):
            _ = dm.replace_data(new_data, column_names=["x"])

    # NOTE: since polars dataframes don't have comprehensive plotting abilities like pandas and we don't want additional dependencies, the method should use seaborn heatmap + matplotlib under-the-hood
    def test_heatmap_visualization(self):
        """
        .heatmap() should create matplotlib visualization.

        Expected behavior:
        - Creates plot without error
        - Returns matplotlib axes object (optional)

        Note: We don't test visual output, just that it doesn't crash
        """
        dm = DesignMatrix(
            np.random.randn(20, 3), sampling_freq=1, columns=["a", "b", "c"]
        )
        dm = dm.add_poly(order=1)

        # Should not raise error
        try:
            dm.heatmap()
            import matplotlib.pyplot as plt

            plt.close("all")  # Clean up
        except Exception as e:
            pytest.fail(f"heatmap() raised unexpected error: {e}")


# ============================================================================
# 10. Edge Cases and Error Handling
# ============================================================================


class TestDesignMatrixEdgeCases:
    """
    Test edge cases and error conditions.

    Ensures robustness of implementation.
    """

    def test_single_column_design_matrix(self):
        """Single column should work (edge case for VIF, etc.)"""
        dm = DesignMatrix({"a": [1, 2, 3]}, sampling_freq=1)

        assert dm.shape == (3, 1)
        assert dm["a"].to_list() == [1, 2, 3]

    def test_single_row_design_matrix(self):
        """Single row should work (unusual but valid)"""
        dm = DesignMatrix({"a": [1], "b": [2]}, sampling_freq=1)

        assert dm.shape == (1, 2)

    def test_vif_requires_multiple_columns(self):
        """VIF should error with only 1 column"""
        dm = DesignMatrix({"a": [1, 2, 3]}, sampling_freq=1)

        with pytest.raises(ValueError):
            dm.vif()

    def test_append_requires_matching_sampling_freq(self):
        """Appending DesignMatrix with different sampling_freq should error"""
        dm1 = DesignMatrix({"a": [1]}, sampling_freq=1)
        dm2 = DesignMatrix({"b": [2]}, sampling_freq=2)

        with pytest.raises(ValueError):
            dm1.append(dm2, axis=0)

    def test_convolve_requires_sampling_freq(self):
        """Convolution needs sampling_freq to be set"""
        dm = DesignMatrix({"a": [1, 0, 0, 0]})  # No sampling_freq

        with pytest.raises(ValueError):
            dm.convolve()

    def test_downsample_target_must_be_lower(self):
        """Downsample target must be < current sampling_freq"""
        dm = DesignMatrix({"a": list(range(10))}, sampling_freq=1.0)

        with pytest.raises(ValueError):
            dm.downsample(target=2.0)  # Target higher than current

    def test_upsample_target_must_be_higher(self):
        """Upsample target must be > current sampling_freq"""
        dm = DesignMatrix({"a": list(range(10))}, sampling_freq=1.0)

        with pytest.raises(ValueError):
            dm.upsample(target=0.5)  # Target lower than current
