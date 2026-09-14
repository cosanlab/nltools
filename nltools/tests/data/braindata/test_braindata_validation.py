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
            _validate_frame(df, n_rows=5, frame_type="Y")


class TestMetadataRowCount:
    """Metadata rows are validated at ingress, not at the next unrelated operation.

    Each row of `X`/`Y` describes one image, so a count that does not match the
    data is wrong the moment it arrives.
    """

    @staticmethod
    def _brain(n_images):
        import nibabel as nib
        from nltools.data import BrainData

        mask = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))
        return BrainData(np.zeros((n_images, 8)), mask=mask)

    def test_constructor_rejects_a_short_frame(self):
        import nibabel as nib
        from nltools.data import BrainData

        mask = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))
        with pytest.raises(ValueError, match="rows"):
            BrainData(
                np.zeros((3, 8)), mask=mask, Y=pl.DataFrame({"label": [1.0, 2.0]})
            )

    def test_assignment_rejects_a_short_frame(self):
        brain = self._brain(3)
        with pytest.raises(ValueError, match="rows"):
            brain.X = pl.DataFrame({"cond": [1.0, 2.0]})

    def test_a_single_image_takes_exactly_one_row(self):
        single = self._brain(3)[0]
        single.Y = pl.DataFrame({"label": [1.0]})
        assert single.Y.height == 1
        with pytest.raises(ValueError, match="rows"):
            single.Y = pl.DataFrame({"label": [1.0, 2.0]})

    def test_an_empty_object_accepts_any_height(self):
        from nltools.data import BrainData

        brain = BrainData()
        brain.Y = pl.DataFrame({"label": [1.0, 2.0, 3.0]})
        assert brain.Y.height == 3
