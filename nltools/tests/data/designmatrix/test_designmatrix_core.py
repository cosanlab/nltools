import numpy as np
import pandas as pd
import polars as pl
from nltools.data.designmatrix import DesignMatrix


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
        - .is_empty property returns True
        - Metadata still accessible

        Use case: Initialize empty, then add columns iteratively
        """
        dm = DesignMatrix(sampling_freq=2)

        assert dm.shape == (0, 0)
        assert dm.is_empty is True
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

    def test_is_empty_property(self):
        """
        .is_empty should return True if no data, False otherwise.

        Expected behavior:
        - Empty DesignMatrix returns True
        - Non-empty returns False
        """
        dm_empty = DesignMatrix(sampling_freq=1)
        dm_full = DesignMatrix({"a": [1]}, sampling_freq=1)

        assert dm_empty.is_empty is True
        assert dm_full.is_empty is False

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
