"""Tests for nltools.data.braindata.validation helpers."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nltools.data.braindata.validation import validate_frame


class TestValidateFrame:
    def test_none_returns_empty_polars(self):
        out = validate_frame(None)
        assert isinstance(out, pl.DataFrame)
        assert out.is_empty()

    def test_polars_passthrough(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        out = validate_frame(df)
        assert isinstance(out, pl.DataFrame)
        assert out.equals(df)

    def test_accepts_pandas_converts_to_polars(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        out = validate_frame(df)
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
        out = validate_frame(dm)
        assert isinstance(out, pl.DataFrame)
        assert out.shape == (3, 3)
        assert set(out.columns) == {"stim", "drift", "poly_0"}

    def test_accepts_numpy_2d(self):
        arr = np.arange(6, dtype=float).reshape(3, 2)
        out = validate_frame(arr)
        assert isinstance(out, pl.DataFrame)
        assert out.shape == (3, 2)

    def test_reads_csv_no_header(self, tmp_path: Path):
        # BrainData's X/Y CSVs are written header-less (see Simulator); match
        # that legacy convention rather than pandas' default header=0.
        p = tmp_path / "y.csv"
        p.write_text("1\n2\n3\n")
        out = validate_frame(p)
        assert isinstance(out, pl.DataFrame)
        assert out.shape == (3, 1)

    def test_reads_csv_multi_column_no_header(self, tmp_path: Path):
        p = tmp_path / "x.csv"
        p.write_text("1,2\n3,4\n5,6\n")
        out = validate_frame(p)
        assert out.shape == (3, 2)

    def test_reads_csv_via_str(self, tmp_path: Path):
        p = tmp_path / "y.csv"
        p.write_text("1,2\n3,4\n")
        out = validate_frame(str(p))
        assert isinstance(out, pl.DataFrame)
        assert out.shape == (2, 2)

    def test_bad_csv_path_raises_value_error(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Could not read"):
            validate_frame(tmp_path / "does_not_exist.csv")

    def test_unsupported_type_raises_type_error(self):
        with pytest.raises(TypeError, match="must be"):
            validate_frame(42)

    def test_shape_mismatch_raises(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="rows"):
            validate_frame(df, data_shape=(5, 100), frame_type="Y")

    def test_shape_match_ok(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        out = validate_frame(df, data_shape=(3, 100), frame_type="Y")
        assert out.shape == (3, 1)

    def test_empty_frame_skips_shape_check(self):
        out = validate_frame(None, data_shape=(5, 100))
        assert out.is_empty()
