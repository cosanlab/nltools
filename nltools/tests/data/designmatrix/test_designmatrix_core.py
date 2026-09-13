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
            (np.random.randn(50, 2), {}, ["0", "1"]),
            (pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}), {}, ["x", "y"]),
            (pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}), {}, ["x", "y"]),
        ],
        ids=["numpy_auto", "polars", "pandas"],
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
        assert dm.confounds == []
        assert dm.multi is False


class TestDesignMatrixDataAccess:
    """Test column access, manipulation, and properties."""

    def test_getitem(self):
        """Test single column returns Series, multiple returns DesignMatrix."""
        dm = DesignMatrix(
            {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
            sampling_freq=2,
            confounds=["a"],
        )

        # Single column
        col = dm["a"]
        assert isinstance(col, pl.Series)
        assert col.to_list() == [1, 2, 3]

        # Multiple columns
        subset = dm[["a", "c"]]
        assert isinstance(subset, DesignMatrix)
        assert subset.columns == ["a", "c"]
        assert subset.sampling_freq == 2
        assert subset.confounds == ["a"]

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

    def test_setitem_polars_expr(self):
        """`dm['col'] = pl.Expr` evaluates the expression in the DM's frame."""
        dm = DesignMatrix({"a": [1, 2, 3], "b": [10, 20, 30]}, sampling_freq=1)
        dm["c"] = pl.col("a") + pl.col("b")
        assert dm["c"].to_list() == [11, 22, 33]

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
        return DesignMatrix(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [10, 20, 30, 40, 50],
                "poly_0": [1, 1, 1, 1, 1],
            },
            sampling_freq=2,
            confounds=["poly_0"],
            convolved=["a"],
        )

    def test_slice_returns_designmatrix(self):
        """slice() is allowlisted and preserves DesignMatrix + metadata."""
        dm = self._dm()
        result = dm.slice(1, 3)
        assert isinstance(result, DesignMatrix)
        assert result.sampling_freq == 2
        assert result.confounds == ["poly_0"]
        assert result.convolved == ["a"]
        assert result.columns == ["a", "b", "poly_0"]

    def test_filter_returns_designmatrix(self):
        """filter() returns DesignMatrix with metadata preserved."""
        dm = self._dm()
        result = dm.filter(pl.col("a") > 2)
        assert isinstance(result, DesignMatrix)
        assert result.sampling_freq == 2
        assert result.confounds == ["poly_0"]
        assert len(result) == 3

    def test_raw_passthrough_for_methods_outside_allowlist(self):
        """Unknown eager operations return a DesignMatrix with cleared metadata."""
        dm = self._dm()
        # Statistical summaries have different row meaning.
        result = dm.describe()
        assert isinstance(result, DesignMatrix)

    def test_unknown_attribute_raises(self):
        """Unknown attrs raise AttributeError (not silently forwarded)."""
        dm = self._dm()
        with pytest.raises(AttributeError, match="no attribute 'not_a_real_method'"):
            dm.not_a_real_method

    def test_select_returns_designmatrix_and_prunes_metadata(self):
        """select() returns DesignMatrix; confounds/convolved entries for dropped cols are removed."""
        dm = self._dm()
        result = dm.select(["a", "b"])
        assert isinstance(result, DesignMatrix)
        assert result.columns == ["a", "b"]
        assert result.sampling_freq == 2
        # poly_0 was dropped — should be gone from confounds
        assert result.confounds == []
        # a was kept — convolved should still include it
        assert result.convolved == ["a"]

    def test_select_keeps_confounds_entry_when_column_kept(self):
        """select() preserves confounds entries for columns it keeps."""
        dm = self._dm()
        result = dm.select(["a", "poly_0"])
        assert result.confounds == ["poly_0"]
        assert result.convolved == ["a"]

    def test_getitem_list_prunes_stale_metadata(self):
        """dm[[cols]] likewise prunes confounds/convolved entries for dropped cols."""
        dm = self._dm()
        subset = dm[["b"]]
        assert subset.confounds == []
        assert subset.convolved == []


class TestDesignMatrixWithColumns:
    """Polars-style `with_columns(**named_exprs)` returns a new DM with cols added/replaced."""

    def _dm(self):
        return DesignMatrix(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [10.0, 20.0, 30.0, 40.0, 50.0],
                "poly_0": [1, 1, 1, 1, 1],
            },
            sampling_freq=2,
            confounds=["poly_0"],
            convolved=["a"],
        )

    def test_named_kwarg_with_polars_expr(self):
        """`with_columns(c=pl.col('a') + pl.col('b'))` adds a new column from an expression."""
        dm = self._dm()
        result = dm.with_columns(c=pl.col("a") + pl.col("b"))
        assert isinstance(result, DesignMatrix)
        assert "c" in result.columns
        assert result["c"].to_list() == [11.0, 22.0, 33.0, 44.0, 55.0]

    def test_returns_new_dm_original_unchanged(self):
        dm = self._dm()
        _ = dm.with_columns(c=pl.col("a"))
        assert "c" not in dm.columns

    def test_preserves_metadata(self):
        """sampling_freq, convolved, confounds carry over."""
        dm = self._dm()
        result = dm.with_columns(c=pl.col("a") * pl.col("b"))
        assert result.sampling_freq == 2
        assert result.convolved == ["a"]
        assert result.confounds == ["poly_0"]
