"""Tests for nltools.algorithms.signal — temporal signal processing."""

import numpy as np
import polars as pl

from nltools.algorithms.signal import downsample, upsample, calc_bpm, make_cosine_basis


class TestDownsample:
    """Test downsampling algorithm."""

    def test_downsample_mean(self):
        """Downsample with mean aggregation."""
        x = pl.Series("x", list(range(100)))
        y = np.repeat(range(1, 11), 10)

        result = downsample(
            data=x, sampling_freq=10, target=1, target_type="hz", method="mean"
        )
        result_values = result.to_numpy()
        expected = np.array(
            [
                np.mean([v for v, g in zip(x.to_numpy(), y) if g == k])
                for k in range(1, 11)
            ]
        )
        assert (result_values == expected).all()

    def test_fractional_ratio_bins_by_floor(self):
        """A 5 Hz to 2 Hz ratio of 2.5 gives four bins, not five (G-09).

        Floor grouping spreads the leftover rows across bins instead of
        truncating the bin width and opening an extra group at the end.
        """
        result = downsample(
            pl.Series("x", range(10)), sampling_freq=5, target=2, target_type="hz"
        )

        assert result.to_list() == [1.0, 3.5, 6.0, 8.5]

    def test_a_group_idx_column_is_not_clobbered(self):
        """The transient grouping key must not collide with a real column."""
        data = pl.DataFrame({"_group_idx": [0.0, 2.0, 4.0, 6.0]})

        result = downsample(data, sampling_freq=2, target=1, target_type="hz")

        assert result.columns == ["_group_idx"]
        assert result["_group_idx"].to_list() == [1.0, 5.0]


class TestUpsample:
    """Test upsampling algorithm."""

    def test_upsample_2x(self):
        """Upsample by factor of 2."""
        dat = pl.DataFrame({"x": list(range(100)), "y": np.repeat(range(1, 11), 10)})
        fs = 2
        us = upsample(dat, sampling_freq=1, target=fs, target_type="hz")
        assert dat.shape[0] * fs - fs == us.shape[0]


class TestMakeCosineBasis:
    """Test discrete cosine basis function generation."""

    def test_basic_output_shape(self):
        """Cosine basis should return correct shape."""
        n_timepoints = 100
        basis = make_cosine_basis(
            n_timepoints, sampling_freq=1, filter_length=128, drop=0
        )
        assert basis.shape[0] == n_timepoints
        assert basis.shape[1] >= 1

    def test_drop_removes_leading_columns(self):
        """drop=k removes the k lowest-frequency bases, keeping the remainder."""
        full = make_cosine_basis(128, 1, 32, drop=0)
        dropped = make_cosine_basis(128, 1, 32, drop=2)
        assert dropped.shape[1] == full.shape[1] - 2
        np.testing.assert_allclose(dropped, full[:, 2:])


class TestCalcBpm:
    """Test beats-per-minute calculation."""

    def test_basic_bpm(self):
        """Calculate BPM from beat intervals."""
        # Beat intervals: 0.833 seconds between beats = 72 BPM
        beat_interval = pl.Series("ibi", [0.833, 0.833, 0.833, 0.833, 0.833])
        result = calc_bpm(beat_interval, sampling_freq=1)
        # Each interval should map to ~72 BPM
        bpm_values = result.to_numpy() if hasattr(result, "to_numpy") else result.values
        assert np.all(bpm_values > 60)
        assert np.all(bpm_values < 80)
