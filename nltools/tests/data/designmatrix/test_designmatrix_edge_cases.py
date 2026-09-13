import pytest
from nltools.data.designmatrix import DesignMatrix


class TestDesignMatrixEdgeCases:
    """
    Test edge cases and error conditions.

    Ensures robustness of implementation.
    """

    def test_vif_requires_multiple_columns(self):
        """VIF should error with only 1 column"""
        dm = DesignMatrix({"a": [1, 2, 3]}, sampling_freq=1)

        with pytest.raises(ValueError):
            dm.vif()

    def test_append_requires_matching_sampling_freq(self):
        """Appending DesignMatrix with different sampling_freq should error"""
        dm1 = DesignMatrix({"a": [1]}, sampling_freq=1)
        dm2 = DesignMatrix({"b": [2]}, sampling_freq=2)

        with pytest.raises(ValueError):
            dm1.append(dm2, axis=0)

    def test_convolve_requires_sampling_freq(self):
        """Convolution needs sampling_freq to be set"""
        dm = DesignMatrix({"a": [1, 0, 0, 0]})  # No sampling_freq

        with pytest.raises(ValueError):
            dm.convolve()

    def test_upsample_target_must_be_higher(self):
        """Upsample target must be > current sampling_freq"""
        dm = DesignMatrix({"a": list(range(10))}, sampling_freq=1.0)

        with pytest.raises(ValueError):
            dm.upsample(target=0.5)  # Target lower than current


class TestNRowsContract:
    """`n_rows` makes a column-less DesignMatrix self-describing — and must
    stay self-describing through copies, and refuse values it cannot honor.
    """

    @staticmethod
    def _empty_dm(n=60):
        import polars as pl

        return DesignMatrix(pl.DataFrame(), sampling_freq=0.5, n_rows=n)

    def test_n_rows_survives_copy(self):
        dm = self._empty_dm()
        assert dm.copy().shape == (60, 0)
        assert len(dm.copy()) == 60

    def test_conflicting_n_rows_raises(self):
        """A row count that contradicts the data is an error, not ignored."""
        with pytest.raises(ValueError, match="n_rows"):
            DesignMatrix({"a": [1, 2, 3]}, sampling_freq=1, n_rows=99)

    def test_negative_n_rows_raises(self):
        with pytest.raises(ValueError, match="n_rows"):
            self._empty_dm(n=-1)

    def test_to_numpy_honors_n_rows(self):
        import numpy as np

        dm = self._empty_dm()
        assert dm.to_numpy().shape == (60, 0)
        assert np.asarray(dm).shape == (60, 0)
