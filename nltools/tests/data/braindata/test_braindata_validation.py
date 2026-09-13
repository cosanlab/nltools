"""Tests for the shared frame validation in nltools.data.validation."""

import numpy as np
import polars as pl
import pytest

from nltools.data.validation import _validate_frame


class TestValidateFrame:
    def test_none_returns_empty_polars(self):
        out = _validate_frame(None)
        assert isinstance(out, pl.DataFrame)
        assert out.is_empty()

    def test_polars_passthrough(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        out = _validate_frame(df)
        assert isinstance(out, pl.DataFrame)
        assert out.equals(df)

    def test_accepts_pandas_converts_to_polars(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        out = _validate_frame(df)
        assert isinstance(out, pl.DataFrame)
        assert out.shape == (3, 2)
        assert out.columns == ["a", "b"]

    def test_accepts_designmatrix_unwraps_to_polars(self):
        """Handing a DesignMatrix to BrainData.X should work — unwraps to .data."""
        from nltools.data import DesignMatrix

        dm = DesignMatrix(
            {"stim": [1.0, 0.0, 1.0], "drift": [0.0, 0.5, 1.0]},
            sampling_freq=0.5,
        ).add_poly(0)
        out = _validate_frame(dm)
        assert isinstance(out, pl.DataFrame)
        assert out.shape == (3, 3)
        assert set(out.columns) == {"stim", "drift", ".nl_poly_0"}

    def test_accepts_numpy_2d(self):
        arr = np.arange(6, dtype=float).reshape(3, 2)
        out = _validate_frame(arr)
        assert isinstance(out, pl.DataFrame)
        assert out.shape == (3, 2)

    def test_unsupported_type_raises_type_error(self):
        with pytest.raises(TypeError, match="must be"):
            _validate_frame(42)

    def test_shape_mismatch_raises(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="rows"):
            _validate_frame(df, data_shape=(5, 100), frame_type="Y")
