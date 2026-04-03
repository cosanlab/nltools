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
