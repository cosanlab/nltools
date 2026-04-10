import os

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nltools.data import DesignMatrix
from nltools.io import onsets_to_dm
from nltools.utils import get_resource_path


class TestOnsetsToDm:
    """Tests for nltools.io.file_reader.onsets_to_dm."""

    @pytest.fixture
    def onsets_path(self):
        return os.path.join(get_resource_path(), "onsets_example.csv")

    @pytest.fixture
    def onsets_data(self, onsets_path):
        return pd.read_csv(onsets_path)

    def test_from_filepath(self, onsets_path):
        """Loading from a CSV path returns a DesignMatrix."""
        dm = onsets_to_dm(onsets_path, run_length=1364, TR=2.0, hrf_model=None)
        assert isinstance(dm, DesignMatrix)

    def test_shape(self, onsets_path, onsets_data):
        """Output has correct shape: (run_length, n_trial_types + intercept)."""
        run_length = 1364
        dm = onsets_to_dm(onsets_path, run_length, TR=2.0, hrf_model=None)
        assert dm.shape == (run_length, onsets_data.trial_type.nunique() + 1)

    def test_onset_sums(self, onsets_path, onsets_data):
        """Column sums reflect stimulus counts x (duration / TR)."""
        TR = 2.0
        duration = 10.0  # from onsets_example.csv
        dm = onsets_to_dm(onsets_path, 1364, TR, hrf_model=None)
        dm = dm.drop(columns=["constant"])
        stim_counts = onsets_data.trial_type.value_counts(sort=False)[dm.columns]
        expected = stim_counts.values * (duration / TR)
        assert np.allclose(expected, dm.sum().to_numpy())

    def test_multiple_runs(self, onsets_data):
        """Passing a list of timings returns a list of DesignMatrices."""
        run_length = 1364
        result = onsets_to_dm(
            [onsets_data, onsets_data], [run_length, run_length], TR=2.0
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(dm, DesignMatrix) for dm in result)

    def test_mismatched_lengths_raises(self, onsets_path):
        """Mismatched timings/run_length lists raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            onsets_to_dm([onsets_path, onsets_path], [100], TR=2.0)

    def test_accepts_polars_dataframe(self, onsets_data):
        """onsets_to_dm should accept a polars DataFrame (nilearn requires pandas internally)."""
        onsets_pl = pl.DataFrame(
            {col: onsets_data[col].tolist() for col in onsets_data.columns}
        )
        dm = onsets_to_dm(onsets_pl, run_length=1364, TR=2.0, hrf_model=None)
        assert isinstance(dm, DesignMatrix)
