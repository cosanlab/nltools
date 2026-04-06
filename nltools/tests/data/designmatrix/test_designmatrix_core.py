import numpy as np
import pandas as pd
import polars as pl
import pytest
from nltools.data.designmatrix import DesignMatrix


class TestDesignMatrixConstruction:
    """Test all supported ways to create a DesignMatrix."""

    @pytest.mark.parametrize(
        "input_data,kwargs,expected_cols",
        [
            (np.random.randn(100, 3), {"columns": ["a", "b", "c"]}, ["a", "b", "c"]),
            (np.random.randn(50, 2), {}, ["0", "1"]),
            ({"a": [1, 2, 3], "b": [4, 5, 6]}, {}, ["a", "b"]),
            (pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}), {}, ["x", "y"]),
            (pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}), {}, ["x", "y"]),
        ],
        ids=["numpy_explicit", "numpy_auto", "dict", "polars", "pandas"],
    )
    def test_construction(self, input_data, kwargs, expected_cols):
        """Test construction from numpy, dict, Polars, and pandas."""
        dm = DesignMatrix(input_data, sampling_freq=1, **kwargs)
        assert set(dm.columns) == set(expected_cols)
        assert isinstance(dm._df, pl.DataFrame)

    def test_empty_initialization(self):
        """Create empty DesignMatrix with only metadata."""
        dm = DesignMatrix(sampling_freq=2)
        assert dm.shape == (0, 0)
        assert dm.is_empty is True
        assert dm.sampling_freq == 2

    def test_column_names_are_always_strings(self):
        """Ensure all column names are strings."""
        data = np.zeros((10, 3))
        dm = DesignMatrix(data, sampling_freq=1)
        for col in dm.columns:
            assert isinstance(col, str)

    def test_metadata_initialization(self):
        """Verify metadata attributes are initialized correctly."""
        dm = DesignMatrix({"a": [1, 2, 3]}, sampling_freq=None)
        assert dm.sampling_freq is None
        assert dm.convolved == []
        assert dm.polys == []
        assert dm.multi is False


class TestDesignMatrixDataAccess:
    """Test column access, manipulation, and properties."""

    def test_getitem(self):
        """Test single column returns Series, multiple returns DesignMatrix."""
        dm = DesignMatrix(
            {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, sampling_freq=2
        )
        dm.polys = ["a"]

        # Single column
        col = dm["a"]
        assert isinstance(col, pl.Series)
        assert col.to_list() == [1, 2, 3]

        # Multiple columns
        subset = dm[["a", "c"]]
        assert isinstance(subset, DesignMatrix)
        assert subset.columns == ["a", "c"]
        assert subset.sampling_freq == 2
        assert subset.polys == ["a"]

    def test_setitem(self):
        """Test scalar broadcast, array assignment, and column replacement."""
        dm = DesignMatrix({"a": [1, 2, 3], "b": [4, 5, 6]}, sampling_freq=1)

        # Scalar broadcast
        dm["c"] = 0
        assert dm["c"].to_list() == [0, 0, 0]

        # Array assignment
        dm["d"] = [10, 20, 30]
        assert dm["d"].to_list() == [10, 20, 30]

        # Replace existing
        dm["a"] = [7, 8, 9]
        assert dm["a"].to_list() == [7, 8, 9]
        assert dm.columns == ["a", "b", "c", "d"]

    def test_properties(self):
        """Test shape, columns, is_empty, len."""
        dm = DesignMatrix(
            np.zeros((42, 5)), sampling_freq=1, columns=["a", "b", "c", "d", "e"]
        )
        assert dm.shape == (42, 5)
        assert isinstance(dm.shape, tuple)
        assert dm.columns == ["a", "b", "c", "d", "e"]
        assert len(dm) == 42
        assert dm.is_empty is False

        # Rename columns
        dm.columns = ["v", "w", "x", "y", "z"]
        assert dm.columns == ["v", "w", "x", "y", "z"]

        # Empty
        dm_empty = DesignMatrix(sampling_freq=1)
        assert dm_empty.is_empty is True
