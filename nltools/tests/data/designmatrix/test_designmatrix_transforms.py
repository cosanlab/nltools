import pytest
import numpy as np
from nltools.data.designmatrix import DesignMatrix
from nltools.data.designmatrix.append import _check_dtype_compatibility
import polars as pl


class TestDownsampleNonIntegerRatio:
    """F083: non-integer sampling ratios must not produce an oversized final group."""

    def test_non_integer_ratio_no_oversized_final_group(self):
        """2 Hz -> 0.7 Hz (ratio ~2.857) must spread rows evenly, not lump them.

        The buggy implementation lumped rows 70-99 into a single trailing group
        (mean ~84.5); a balanced grouping puts only the last 2-3 rows there
        (mean ~98.5).
        """
        dm = DesignMatrix({"a": [float(i) for i in range(100)]}, sampling_freq=2.0)
        down = dm.downsample(target=0.7)
        values = down.data["a"].to_list()
        assert len(values) == 35
        assert values[-1] > 95

    def test_integer_ratio_unchanged(self):
        """Exact integer ratios keep their original balanced grouping."""
        dm = DesignMatrix({"a": [float(i) for i in range(100)]}, sampling_freq=2.0)
        down = dm.downsample(target=1.0)  # ratio == 2
        values = down.data["a"].to_list()
        assert len(values) == 50
        assert values[0] == 0.5  # mean of rows 0, 1


class TestCheckDtypeCompatibility:
    """F082: dtype mismatches among later frames must be detected, not just vs dfs[0]."""

    def test_mismatch_among_later_frames_flagged(self):
        """A column absent from dfs[0] but conflicting between dfs[1]/dfs[2] is flagged."""
        dfs = [
            pl.DataFrame({"base": pl.Series([1], dtype=pl.Int64)}),
            pl.DataFrame({"later": pl.Series([1], dtype=pl.Int64)}),
            pl.DataFrame({"later": pl.Series([1.0], dtype=pl.Float64)}),
        ]
        with pytest.raises(ValueError, match="later"):
            _check_dtype_compatibility(dfs)


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
        - confounds list preserved
        - convolved list preserved
        - multi flag preserved
        """
        dm = DesignMatrix(
            {"stim": [1, 2, 3], "poly_0": [1, 1, 1]},
            sampling_freq=2,
            confounds=["poly_0"],
            convolved=["stim"],
        )
        dm.multi = True

        dm_filled = dm.fillna(0)

        assert dm_filled.sampling_freq == 2
        assert dm_filled.confounds == ["poly_0"]
        assert dm_filled.convolved == ["stim"]
        assert dm_filled.multi is True


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
        - Columns in .confounds list are skipped
        - Only non-confound columns standardized

        Rationale: Confounds (intercept, trends, motion, …) should not be standardized
        """
        dm = DesignMatrix(
            {"stim": [1, 2, 3, 4], "poly_0": [1, 1, 1, 1]},
            sampling_freq=1,
            confounds=["poly_0"],
        )

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
