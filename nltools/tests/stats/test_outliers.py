"""Tests for nltools.stats.outliers — outlier detection and robust statistics."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nltools.stats.outliers import winsorize, find_spikes, zscore


class TestWinsorize:
    """Test winsorizing outlier handling."""

    def test_quantile_replace_with_nearest(self, outlier_data):
        """Winsorize by quantile, replacing outliers with nearest non-outlier."""
        out = winsorize(
            outlier_data, cutoff={"quantile": [0.05, 0.95]}, replace_with_cutoff=False
        )
        if isinstance(out, pl.DataFrame):
            out = out.to_numpy().squeeze()
        else:
            out = out.values.squeeze()
        expected = np.array(
            [
                92,
                19,
                101,
                58,
                101,
                91,
                26,
                78,
                10,
                13,
                -5,
                101,
                86,
                85,
                15,
                89,
                89,
                28,
                -5,
                41,
            ]
        )
        assert np.sum(out == expected) == 20

    def test_std_replace_with_nearest(self, outlier_data):
        """Winsorize by std, replacing outliers with nearest non-outlier."""
        out = winsorize(outlier_data, cutoff={"std": [2, 2]}, replace_with_cutoff=False)
        if isinstance(out, pl.DataFrame):
            out = out.to_numpy().squeeze()
        else:
            out = out.values.squeeze()
        expected = np.array(
            [
                92,
                19,
                101,
                58,
                101,
                91,
                26,
                78,
                10,
                13,
                -40,
                101,
                86,
                85,
                15,
                89,
                89,
                28,
                -5,
                41,
            ]
        )
        assert np.sum(out == expected) == 20

    def test_std_replace_with_cutoff(self, outlier_data):
        """Winsorize by std, replacing outliers with cutoff values."""
        out = winsorize(outlier_data, cutoff={"std": [2, 2]}, replace_with_cutoff=True)
        if isinstance(out, pl.DataFrame):
            out = out.to_numpy().squeeze()
        else:
            out = out.values.squeeze()
        expected = np.array(
            [
                92.0,
                19.0,
                101.0,
                58.0,
                556.97961997,
                91.0,
                26.0,
                78.0,
                10.0,
                13.0,
                -40.0,
                101.0,
                86.0,
                85.0,
                15.0,
                89.0,
                89.0,
                28.0,
                -5.0,
                41.0,
            ]
        )
        assert np.round(np.mean(out)) == np.round(np.mean(expected))


class TestZscore:
    """Test z-score normalization."""

    def test_zscore_series(self):
        """Z-scoring a pandas Series should produce mean~0, std~1."""
        data = pd.Series(np.random.randn(100) * 5 + 10)
        result = zscore(data)
        np.testing.assert_almost_equal(np.mean(result), 0, decimal=10)
        np.testing.assert_almost_equal(np.std(result), 1, decimal=10)

    def test_zscore_dataframe(self):
        """Z-scoring a DataFrame should normalize each column."""
        data = pd.DataFrame(np.random.randn(100, 3) * 5 + 10)
        result = zscore(data)
        for col in result.columns:
            np.testing.assert_almost_equal(np.mean(result[col]), 0, decimal=10)


class TestFindSpikes:
    """Test spike detection in neuroimaging data."""

    @pytest.mark.slow
    def test_find_spikes_brain_data(self):
        """Find spikes in simulated BrainData."""
        from nltools.data.simulator import Simulator
        from nltools.mask import create_sphere

        sim = Simulator()
        s1 = create_sphere([0, 0, 0], radius=3)
        d1 = sim.create_data([0, 1], 1, reps=50, output_dir=None).apply_mask(s1)

        spikes = find_spikes(d1)
        assert isinstance(spikes, pl.DataFrame)
        assert spikes.shape[0] == len(d1)

    @pytest.mark.slow
    def test_find_spikes_nifti(self):
        """Find spikes from a NIfTI image."""
        from nltools.data.simulator import Simulator
        from nltools.mask import create_sphere

        sim = Simulator()
        s1 = create_sphere([0, 0, 0], radius=3)
        d1 = sim.create_data([0, 1], 1, reps=50, output_dir=None).apply_mask(s1)

        spikes = find_spikes(d1.to_nifti())
        assert isinstance(spikes, pl.DataFrame)
        assert spikes.shape[0] == len(d1)
