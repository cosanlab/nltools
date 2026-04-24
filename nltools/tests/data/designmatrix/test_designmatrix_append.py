import pytest

from nltools.data.designmatrix import DesignMatrix


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

    def test_horizontal_append_pandas_dataframe_as_confounds(self):
        """
        Horizontal append accepts pandas DataFrames; new columns are tracked
        as confounds (added to .confounds) so vertical multi-run appends
        separate them per run. Typical use: adding motion confounds from a
        BIDS confounds.tsv in one call instead of a per-column loop.
        """
        import pandas as pd

        dm = DesignMatrix({"task": [0.0, 1.0, 0.0]}, sampling_freq=0.5)
        confounds = pd.DataFrame(
            {"trans_x": [0.1, 0.2, -0.1], "rot_y": [0.01, -0.02, 0.03]}
        )

        combined = dm.append(confounds, axis=1)

        assert combined.shape == (3, 3)
        assert set(combined.columns) == {"task", "trans_x", "rot_y"}
        # The confound columns should be tracked in .confounds so they get
        # per-run separation on vertical append.
        assert "trans_x" in combined.confounds
        assert "rot_y" in combined.confounds
        assert "task" not in combined.confounds

    def test_horizontal_append_polars_dataframe_as_confounds(self):
        """Same behavior for a polars DataFrame input."""
        import polars as pl

        dm = DesignMatrix({"task": [0.0, 1.0]}, sampling_freq=0.5)
        confounds = pl.DataFrame({"x": [0.1, 0.2], "y": [0.3, 0.4]})

        combined = dm.append(confounds, axis=1)

        assert combined.shape == (2, 3)
        assert set(combined.columns) == {"task", "x", "y"}
        assert set(combined.confounds) >= {"x", "y"}

    def test_horizontal_append_dataframe_rejects_unsupported_type(self):
        """A non-DesignMatrix, non-DataFrame input raises a clear error."""
        dm = DesignMatrix({"task": [0.0, 1.0]}, sampling_freq=0.5)
        with pytest.raises(TypeError, match="pandas DataFrame, or polars DataFrame"):
            dm.append([[1, 2], [3, 4]], axis=1)

    def test_horizontal_append_dataframe_then_vertical_separates_nuisance(self):
        """End-to-end: DataFrame confounds survive multi-run separation."""
        import pandas as pd

        run1 = DesignMatrix({"task": [0.0, 1.0, 0.0]}, sampling_freq=0.5)
        run1 = run1.append(pd.DataFrame({"mot_x": [0.1, 0.2, 0.3]}), axis=1)
        run2 = DesignMatrix({"task": [1.0, 0.0, 1.0]}, sampling_freq=0.5)
        run2 = run2.append(pd.DataFrame({"mot_x": [-0.1, 0.0, 0.2]}), axis=1)

        combined = run1.append(run2, axis=0)

        # Task stacks into a single column; motion confound gets per-run split.
        assert "task" in combined.columns
        assert any(c.endswith("mot_x") and c != "mot_x" for c in combined.columns)

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
        After separation, .confounds metadata should contain all separated poly names.

        Expected behavior:
        - .confounds list contains '0_poly_0', '1_poly_0', etc.
        - Original 'poly_0' not in metadata (replaced by separated versions)

        Use case: Track which columns are confounds for later operations
        """
        dm1 = DesignMatrix({"s": [1]}, sampling_freq=1).add_poly(0)
        dm2 = DesignMatrix({"s": [2]}, sampling_freq=1).add_poly(0)

        dm_runs = dm1.append(dm2, axis=0, keep_separate=True)

        assert "0_poly_0" in dm_runs.confounds
        assert "1_poly_0" in dm_runs.confounds
        assert "poly_0" not in dm_runs.confounds  # Original name replaced


class TestDesignMatrixAppendMetadata:
    """Audit: convolved + confounds metadata survives all append paths."""

    def test_horizontal_append_merges_convolved(self):
        """convolved from both DMs should be preserved on horizontal append."""
        dm1 = DesignMatrix({"a": [1, 2]}, sampling_freq=1)
        dm2 = DesignMatrix({"b": [3, 4]}, sampling_freq=1)
        dm1.convolved = ["a"]
        dm2.convolved = ["b"]

        out = dm1.append(dm2, axis=1)
        assert set(out.convolved) == {"a", "b"}

    def test_vertical_simple_append_merges_convolved(self):
        """keep_separate=False should preserve shared convolved entries."""
        dm1 = DesignMatrix({"stim": [1, 2, 3]}, sampling_freq=1)
        dm2 = DesignMatrix({"stim": [4, 5, 6]}, sampling_freq=1)
        dm1.convolved = ["stim"]
        dm2.convolved = ["stim"]

        out = dm1.append(dm2, axis=0, keep_separate=False)
        assert out.convolved == ["stim"]

    def test_vertical_separation_merges_convolved_differing_names(self):
        """keep_separate=True with different convolved columns per run preserves both."""
        dm1 = DesignMatrix({"house": [1, 0, 0]}, sampling_freq=1).add_poly(0)
        dm1.convolved = ["house"]
        dm2 = DesignMatrix({"face": [0, 1, 0]}, sampling_freq=1).add_poly(0)
        dm2.convolved = ["face"]

        out = dm1.append(dm2, axis=0, keep_separate=True)
        assert set(out.convolved) == {"house", "face"}

    def test_vertical_separation_renames_convolved_via_unique_cols(self):
        """When unique_cols renames a convolved column, convolved should track renames."""
        dm1 = DesignMatrix({"motion_x": [1, 2], "stim": [1, 0]}, sampling_freq=1)
        dm1.convolved = ["motion_x"]
        dm2 = DesignMatrix({"motion_x": [3, 4], "stim": [0, 1]}, sampling_freq=1)
        dm2.convolved = ["motion_x"]

        out = dm1.append(dm2, axis=0, unique_cols=["motion_x"])
        assert set(out.convolved) == {"0_motion_x", "1_motion_x"}


class TestDesignMatrixAppendErrors:
    """Audit: helpful errors for user-fixable issues."""

    def test_non_multi_base_with_multi_to_append_raises(self):
        """Base non-multi + appended multi-run DM would cause silent collision — should raise."""
        multi = (
            DesignMatrix({"s": [1, 2]}, sampling_freq=1)
            .add_poly(0)
            .append(DesignMatrix({"s": [3, 4]}, sampling_freq=1).add_poly(0), axis=0)
        )
        simple = DesignMatrix({"s": [99]}, sampling_freq=1).add_poly(0)

        with pytest.raises(ValueError, match="multi-run"):
            simple.append(multi, axis=0)

    def test_dtype_mismatch_raises_clear_error(self):
        """Vertical append of int vs float column should name column and dtypes."""
        dm1 = DesignMatrix({"a": [1, 2]}, sampling_freq=1)
        dm2 = DesignMatrix({"a": [1.5, 2.5]}, sampling_freq=1)

        with pytest.raises(ValueError, match="dtype.*'a'"):
            dm1.append(dm2, axis=0, keep_separate=False)

    def test_horizontal_append_duplicate_cols_raises(self):
        """Duplicate columns on horizontal append should raise with a helpful message."""
        dm1 = DesignMatrix({"a": [1]}, sampling_freq=1)
        dm2 = DesignMatrix({"a": [2]}, sampling_freq=1)

        with pytest.raises(ValueError, match="[Dd]uplicate.*'a'"):
            dm1.append(dm2, axis=1)


class TestDesignMatrixAppendFillNa:
    """Audit: fill_na=None should mean 'don't fill' across all paths."""

    def test_fill_na_none_preserves_nulls_in_simple_vertical(self):
        """keep_separate=False with fill_na=None keeps nulls."""
        dm1 = DesignMatrix({"a": [1, 2]}, sampling_freq=1)
        dm2 = DesignMatrix({"b": [3, 4]}, sampling_freq=1)
        out = dm1.append(dm2, axis=0, keep_separate=False, fill_na=None)
        # Missing cells stay as null (not filled with 0)
        assert out["a"].to_list() == [1, 2, None, None]
        assert out["b"].to_list() == [None, None, 3, 4]

    def test_fill_na_none_preserves_nulls_in_separation(self):
        """keep_separate=True with fill_na=None keeps nulls for separated confounds."""
        dm1 = DesignMatrix({"s": [1, 2]}, sampling_freq=1).add_poly(0)
        dm2 = DesignMatrix({"s": [3, 4]}, sampling_freq=1).add_poly(0)
        out = dm1.append(dm2, axis=0, keep_separate=True, fill_na=None)
        # Separated poly columns: null in the other run, not 0
        assert out["0_poly_0"].to_list()[2:] == [None, None]
        assert out["1_poly_0"].to_list()[:2] == [None, None]

    def test_fill_na_none_preserves_nulls_in_horizontal(self):
        """Horizontal append with fill_na=None keeps nulls (when shapes differ would fail, but equal shapes no nulls)."""
        # Same shape so no nulls introduced, but verify no error when fill_na=None
        dm1 = DesignMatrix({"a": [1, 2]}, sampling_freq=1)
        dm2 = DesignMatrix({"b": [3, 4]}, sampling_freq=1)
        out = dm1.append(dm2, axis=1, fill_na=None)
        assert out.shape == (2, 2)
