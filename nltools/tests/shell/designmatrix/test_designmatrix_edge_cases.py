import pytest
from nltools.data.designmatrix import DesignMatrix


class TestDesignMatrixEdgeCases:
    """
    Test edge cases and error conditions.

    Ensures robustness of implementation.
    """

    def test_single_column_design_matrix(self):
        """Single column should work (edge case for VIF, etc.)"""
        dm = DesignMatrix({"a": [1, 2, 3]}, sampling_freq=1)

        assert dm.shape == (3, 1)
        assert dm["a"].to_list() == [1, 2, 3]

    def test_single_row_design_matrix(self):
        """Single row should work (unusual but valid)"""
        dm = DesignMatrix({"a": [1], "b": [2]}, sampling_freq=1)

        assert dm.shape == (1, 2)

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

    def test_downsample_target_must_be_lower(self):
        """Downsample target must be < current sampling_freq"""
        dm = DesignMatrix({"a": list(range(10))}, sampling_freq=1.0)

        with pytest.raises(ValueError):
            dm.downsample(target=2.0)  # Target higher than current

    def test_upsample_target_must_be_higher(self):
        """Upsample target must be > current sampling_freq"""
        dm = DesignMatrix({"a": list(range(10))}, sampling_freq=1.0)

        with pytest.raises(ValueError):
            dm.upsample(target=0.5)  # Target lower than current
