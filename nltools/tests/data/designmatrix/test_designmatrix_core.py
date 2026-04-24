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
        assert isinstance(dm.data, pl.DataFrame)

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


class TestDesignMatrixPassthrough:
    """Test polars DataFrame passthrough via __getattr__."""

    def _dm(self):
        dm = DesignMatrix(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [10, 20, 30, 40, 50],
                "poly_0": [1, 1, 1, 1, 1],
            },
            sampling_freq=2,
        )
        dm.polys = ["poly_0"]
        dm.convolved = ["a"]
        return dm

    def test_slice_returns_designmatrix(self):
        """slice() is allowlisted and preserves DesignMatrix + metadata."""
        dm = self._dm()
        result = dm.slice(1, 3)
        assert isinstance(result, DesignMatrix)
        assert result.sampling_freq == 2
        assert result.polys == ["poly_0"]
        assert result.convolved == ["a"]
        assert result.columns == ["a", "b", "poly_0"]

    def test_filter_returns_designmatrix(self):
        """filter() returns DesignMatrix with metadata preserved."""
        dm = self._dm()
        result = dm.filter(pl.col("a") > 2)
        assert isinstance(result, DesignMatrix)
        assert result.sampling_freq == 2
        assert result.polys == ["poly_0"]
        assert len(result) == 3

    @pytest.mark.parametrize(
        "method,args,kwargs",
        [
            ("head", (2,), {}),
            ("tail", (2,), {}),
            ("sample", (), {"n": 3, "seed": 0}),
        ],
    )
    def test_unwrapping_row_methods_return_polars(self, method, args, kwargs):
        """head/tail/sample unwrap to polars — they're not on the allowlist.

        Keeping these unwrapped is the v0.6 design: only methods that clearly
        preserve DesignMatrix semantics (row-aligned subsetting by the user's
        own slice/filter/select predicate) are re-wrapped. Inspection helpers
        like head/tail/sample hand back the raw polars DataFrame so users can
        chain directly into polars idioms.
        """
        dm = self._dm()
        result = getattr(dm, method)(*args, **kwargs)
        assert isinstance(result, pl.DataFrame)
        assert not isinstance(result, DesignMatrix)
        assert list(result.columns) == ["a", "b", "poly_0"]

    def test_raw_passthrough_for_informational_attrs(self):
        """Attrs that don't return DataFrames pass through raw (dtypes, schema)."""
        dm = self._dm()
        assert dm.dtypes == dm.data.dtypes
        assert dm.schema == dm.data.schema

    def test_raw_passthrough_for_methods_outside_allowlist(self):
        """Methods not in the allowlist return raw polars results."""
        dm = self._dm()
        # describe() returns a pl.DataFrame but isn't in allowlist — stays raw
        result = dm.describe()
        assert isinstance(result, pl.DataFrame)
        assert not isinstance(result, DesignMatrix)

    def test_unknown_attribute_raises(self):
        """Unknown attrs raise AttributeError (not silently forwarded)."""
        dm = self._dm()
        with pytest.raises(AttributeError, match="no attribute 'not_a_real_method'"):
            dm.not_a_real_method

    def test_select_returns_designmatrix_and_prunes_metadata(self):
        """select() returns DesignMatrix; polys/convolved entries for dropped cols are removed."""
        dm = self._dm()
        result = dm.select(["a", "b"])
        assert isinstance(result, DesignMatrix)
        assert result.columns == ["a", "b"]
        assert result.sampling_freq == 2
        # poly_0 was dropped — should be gone from polys
        assert result.polys == []
        # a was kept — convolved should still include it
        assert result.convolved == ["a"]

    def test_select_keeps_polys_entry_when_column_kept(self):
        """select() preserves polys entries for columns it keeps."""
        dm = self._dm()
        result = dm.select(["a", "poly_0"])
        assert result.polys == ["poly_0"]
        assert result.convolved == ["a"]

    def test_getitem_list_prunes_stale_metadata(self):
        """dm[[cols]] likewise prunes polys/convolved entries for dropped cols."""
        dm = self._dm()
        subset = dm[["b"]]
        assert subset.polys == []
        assert subset.convolved == []

    def test_dir_exposes_polars_methods(self):
        """__dir__ surfaces polars methods for REPL/IDE autocomplete."""
        dm = self._dm()
        names = dir(dm)
        assert "head" in names
        assert "filter" in names
        assert "describe" in names
        # Explicit DesignMatrix methods still present
        assert "convolve" in names
        assert "add_poly" in names
