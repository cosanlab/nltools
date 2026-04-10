"""Tests for nltools.stats.timeseries — temporal signal processing."""

import numpy as np
import pandas as pd
import polars as pl

from nltools.stats.timeseries import downsample, upsample, calc_bpm, make_cosine_basis


class TestDownsample:
    """Test downsampling algorithm."""

    def test_downsample_mean(self):
        """Downsample with mean aggregation."""
        dat = pd.DataFrame()
        dat["x"] = range(0, 100)
        dat["y"] = np.repeat(range(1, 11), 10)

        result = downsample(
            data=dat["x"], sampling_freq=10, target=1, target_type="hz", method="mean"
        )
        if isinstance(result, pl.Series):
            result_values = result.to_numpy()
        else:
            result_values = result.values
        expected = dat.groupby("y").mean().values.ravel()
        assert (result_values == expected).all()

    def test_downsample_median(self):
        """Downsample with median aggregation."""
        dat = pd.DataFrame()
        dat["x"] = range(0, 100)
        dat["y"] = np.repeat(range(1, 11), 10)

        result = downsample(
            data=dat["x"], sampling_freq=10, target=1, target_type="hz", method="median"
        )
        if isinstance(result, pl.Series):
            result_values = result.to_numpy()
        else:
            result_values = result.values
        expected = dat.groupby("y").median().values.ravel()
        assert (result_values == expected).all()


class TestUpsample:
    """Test upsampling algorithm."""

    def test_upsample_2x(self):
        """Upsample by factor of 2."""
        dat = pd.DataFrame()
        dat["x"] = range(0, 100)
        dat["y"] = np.repeat(range(1, 11), 10)
        fs = 2
        us = upsample(dat, sampling_freq=1, target=fs, target_type="hz")
        assert dat.shape[0] * fs - fs == us.shape[0]

    def test_upsample_3x(self):
        """Upsample by factor of 3."""
        dat = pd.DataFrame()
        dat["x"] = range(0, 100)
        dat["y"] = np.repeat(range(1, 11), 10)
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


class TestCalcBpm:
    """Test beats-per-minute calculation."""

    def test_basic_bpm(self):
        """Calculate BPM from beat intervals."""
        # Beat intervals: 0.833 seconds between beats = 72 BPM
        beat_interval = pd.Series([0.833, 0.833, 0.833, 0.833, 0.833])
        result = calc_bpm(beat_interval, sampling_freq=1)
        # Each interval should map to ~72 BPM
        bpm_values = result.to_numpy() if hasattr(result, "to_numpy") else result.values
        assert np.all(bpm_values > 60)
        assert np.all(bpm_values < 80)
