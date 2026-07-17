"""Tests for nltools.stats.timeseries — temporal signal processing."""

import numpy as np
import polars as pl
import pytest

from nltools.stats.timeseries import downsample, upsample, calc_bpm, make_cosine_basis


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

    def test_downsample_median(self):
        """Downsample with median aggregation."""
        x = pl.Series("x", list(range(100)))
        y = np.repeat(range(1, 11), 10)

        result = downsample(
            data=x, sampling_freq=10, target=1, target_type="hz", method="median"
        )
        result_values = result.to_numpy()
        expected = np.array(
            [
                np.median([v for v, g in zip(x.to_numpy(), y) if g == k])
                for k in range(1, 11)
            ]
        )
        assert (result_values == expected).all()


class TestUpsample:
    """Test upsampling algorithm."""

    def test_upsample_2x(self):
        """Upsample by factor of 2."""
        dat = pl.DataFrame({"x": list(range(100)), "y": np.repeat(range(1, 11), 10)})
        fs = 2
        us = upsample(dat, sampling_freq=1, target=fs, target_type="hz")
        assert dat.shape[0] * fs - fs == us.shape[0]

    def test_upsample_3x(self):
        """Upsample by factor of 3."""
        dat = pl.DataFrame({"x": list(range(100)), "y": np.repeat(range(1, 11), 10)})
        fs = 3
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

    def test_column_count_matches_filter_length(self):
        """Basis column count follows order = trunc(2·n·f/L + 1) minus intercept.

        For n=128, sampling_freq=1: filter_length=128 → order 3 → 2 bases;
        filter_length=32 → order 9 → 8 bases. A shorter filter (higher cutoff)
        admits more drift bases.
        """
        assert make_cosine_basis(128, 1, 128, drop=0).shape == (128, 2)
        assert make_cosine_basis(128, 1, 32, drop=0).shape == (128, 8)

    def test_columns_near_orthogonal(self):
        """DCT basis columns are mutually orthogonal (off-diagonal Gram ≈ 0)."""
        basis = make_cosine_basis(128, 1, 32, drop=0)
        gram = basis.T @ basis
        off_diagonal = gram - np.diag(np.diag(gram))
        assert np.abs(off_diagonal).max() < 1e-8

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

    def test_bpm_mapping_not_hardcoded(self):
        """BPM = 60·sampling_freq/beat_interval across several inputs."""
        # 1000 samples between beats at 1000 Hz = 1 s interval = 60 BPM.
        assert calc_bpm(1000, sampling_freq=1000) == pytest.approx(60.0)
        # Halving the interval doubles the rate.
        assert calc_bpm(500, sampling_freq=1000) == pytest.approx(120.0)
        # A different sampling frequency scales linearly.
        assert calc_bpm(2, sampling_freq=1) == pytest.approx(30.0)
