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
        out = out.to_numpy().squeeze()
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
        out = out.to_numpy().squeeze()
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
        out = out.to_numpy().squeeze()
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

    def test_zscore_pandas_series_returns_polars(self):
        """Z-scoring a pandas Series returns a polars Series with mean~0, std~1."""
        data = pd.Series(np.random.randn(100) * 5 + 10, name="x")
        result = zscore(data)
        assert isinstance(result, pl.Series)
        np.testing.assert_almost_equal(result.mean(), 0, decimal=10)
        np.testing.assert_almost_equal(result.std(), 1, decimal=10)

    def test_zscore_pandas_dataframe_returns_polars(self):
        """Z-scoring a pandas DataFrame returns a polars DataFrame with each column normalized."""
        data = pd.DataFrame(np.random.randn(100, 3) * 5 + 10, columns=["a", "b", "c"])
        result = zscore(data)
        assert isinstance(result, pl.DataFrame)
        for col in result.columns:
            np.testing.assert_almost_equal(result[col].mean(), 0, decimal=10)
            np.testing.assert_almost_equal(result[col].std(), 1, decimal=10)

    def test_zscore_polars_series(self):
        """Z-scoring a polars Series returns a polars Series."""
        data = pl.Series("x", np.random.randn(100) * 5 + 10)
        result = zscore(data)
        assert isinstance(result, pl.Series)
        np.testing.assert_almost_equal(result.mean(), 0, decimal=10)
        np.testing.assert_almost_equal(result.std(), 1, decimal=10)

    def test_zscore_polars_dataframe(self):
        """Z-scoring a polars DataFrame returns a polars DataFrame."""
        data = pl.DataFrame(
            {"a": np.random.randn(100), "b": np.random.randn(100) * 2 + 5}
        )
        result = zscore(data)
        assert isinstance(result, pl.DataFrame)
        for col in result.columns:
            np.testing.assert_almost_equal(result[col].mean(), 0, decimal=10)
            np.testing.assert_almost_equal(result[col].std(), 1, decimal=10)

    def test_zscore_rejects_invalid_input(self):
        """Z-scoring a non-DataFrame/Series input raises."""
        with pytest.raises(ValueError, match="Polars or pandas"):
            zscore([1, 2, 3])


class TestFindSpikes:
    """Test spike detection in neuroimaging data.

    `find_spikes` returns a `DesignMatrix` with spike indicator columns
    pre-marked as confounds. The legacy `TR` index column is dropped (row
    position is the time axis in the Polars-backed DM). Pass `TR=` or
    `sampling_freq=` to make `.convolve()` / `.append()` happy downstream.
    """

    @pytest.fixture
    def spike_nifti(self):
        """Tiny synthetic 4D nifti with two known global spikes."""
        import nibabel as nib

        rng = np.random.default_rng(0)
        n_tr = 30
        data = rng.standard_normal((4, 4, 4, n_tr))
        # Inject two global spikes well above 3σ.
        data[..., 5] += 50
        data[..., 20] += 50
        return nib.Nifti1Image(data, affine=np.eye(4))

    def test_returns_designmatrix(self, spike_nifti):
        from nltools.data import DesignMatrix

        dm = find_spikes(spike_nifti)
        assert isinstance(dm, DesignMatrix)

    def test_drops_tr_column(self, spike_nifti):
        """The legacy 'TR' (1-indexed timestamp) column is no longer included."""
        dm = find_spikes(spike_nifti)
        assert "TR" not in dm.columns

    def test_spike_columns_marked_as_confounds(self, spike_nifti):
        dm = find_spikes(spike_nifti)
        spike_cols = [c for c in dm.columns if "spike" in c]
        assert spike_cols, "expected at least one spike column"
        for c in spike_cols:
            assert c in dm.confounds

    def test_row_count_matches_input(self, spike_nifti):
        dm = find_spikes(spike_nifti)
        assert dm.shape[0] == 30  # n_tr

    def test_sampling_freq_kwarg_propagates(self, spike_nifti):
        dm = find_spikes(spike_nifti, sampling_freq=0.5)
        assert dm.sampling_freq == 0.5

    def test_tr_kwarg_propagates(self, spike_nifti):
        dm = find_spikes(spike_nifti, TR=2.0)
        assert dm.sampling_freq == pytest.approx(0.5)

    def test_no_freq_kwarg_leaves_sampling_freq_unset(self, spike_nifti):
        """Without TR/sampling_freq, the DM has sampling_freq=None.

        Downstream `.convolve()` will error helpfully; users append into a DM
        that already has sampling_freq set.
        """
        dm = find_spikes(spike_nifti)
        assert dm.sampling_freq is None

    @pytest.mark.slow
    def test_find_spikes_brain_data(self):
        """Find spikes in simulated BrainData (slow — uses Simulator)."""
        from nltools.data import DesignMatrix
        from nltools.data.simulator import Simulator
        from nltools.mask import create_sphere

        sim = Simulator()
        s1 = create_sphere([0, 0, 0], radius=3)
        d1 = sim.create_data([0, 1], 1, reps=50, output_dir=None).apply_mask(s1)

        dm = find_spikes(d1)
        assert isinstance(dm, DesignMatrix)
        assert dm.shape[0] == len(d1)
