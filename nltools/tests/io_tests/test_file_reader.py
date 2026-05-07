"""Tests for the file-path / events-DataFrame paths in DesignMatrix.

The old standalone `nltools.io.onsets_to_dm` was folded into
`DesignMatrix.__init__` (file path) and `nltools.data.designmatrix.io.events_to_dm`
(in-memory events DataFrame).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nltools.data import DesignMatrix
from nltools.data.designmatrix.io import events_to_dm
from nltools.utils import get_resource_path


@pytest.fixture
def onsets_path():
    return os.path.join(get_resource_path(), "onsets_example.csv")


@pytest.fixture
def onsets_data(onsets_path):
    return pd.read_csv(onsets_path)


class TestDesignMatrixFromEventsFile:
    """`DesignMatrix(<events file>, ...)` HRF-convolves by default.

    The constructor matches nilearn's default (`hrf_model='glover'`).
    Pass `hrf_model=None` to get raw boxcar regressors instead — useful for
    pedagogy, FIR designs, or PPI flows that build interaction terms before
    convolution.
    """

    def test_from_filepath_returns_designmatrix(self, onsets_path):
        dm = DesignMatrix(onsets_path, run_length=1364, TR=2.0)
        assert isinstance(dm, DesignMatrix)
        assert dm.sampling_freq == pytest.approx(0.5)

    def test_default_is_convolved(self, onsets_path, onsets_data):
        """Default is HRF-convolved; columns get the canonical `_c0` suffix."""
        dm = DesignMatrix(onsets_path, run_length=1364, TR=2.0)
        for trial_type in onsets_data.trial_type.unique():
            assert f"{trial_type}_c0" in dm.columns
            assert trial_type not in dm.columns
        assert set(dm.convolved) == set(dm.columns)

    def test_shape(self, onsets_path, onsets_data):
        """Output is (run_length, n_unique_trial_types) — no auto-intercept."""
        run_length = 1364
        dm = DesignMatrix(onsets_path, run_length=run_length, TR=2.0)
        assert dm.shape == (run_length, onsets_data.trial_type.nunique())

    def test_no_constant_column(self, onsets_path):
        """nilearn's auto 'constant' is dropped — caller adds intercept via add_poly(0)."""
        dm = DesignMatrix(onsets_path, run_length=1364, TR=2.0)
        assert "constant" not in dm.columns

    def test_hrf_model_none_gives_boxcar(self, onsets_path, onsets_data):
        """Opt-out: boxcar regressors with raw trial-type column names."""
        dm = DesignMatrix(onsets_path, run_length=1364, TR=2.0, hrf_model=None)
        for trial_type in onsets_data.trial_type.unique():
            assert trial_type in dm.columns
            assert f"{trial_type}_c0" not in dm.columns
        assert dm.convolved == []

    def test_boxcar_onset_sums(self, onsets_path, onsets_data):
        """Boxcar (opt-out) sums reflect stimulus counts × (duration / TR)."""
        TR = 2.0
        duration = 10.0  # from onsets_example.csv
        dm = DesignMatrix(onsets_path, run_length=1364, TR=TR, hrf_model=None)
        stim_counts = onsets_data.trial_type.value_counts(sort=False)[dm.columns]
        expected = stim_counts.values * (duration / TR)
        assert np.allclose(expected, dm.sum().to_numpy())

    def test_unknown_hrf_model_errors(self, onsets_path):
        with pytest.raises(ValueError, match="hrf_model"):
            DesignMatrix(onsets_path, run_length=1364, TR=2.0, hrf_model="not_a_model")

    def test_path_object_accepted(self, onsets_path):
        dm = DesignMatrix(Path(onsets_path), run_length=1364, TR=2.0)
        assert isinstance(dm, DesignMatrix)

    def test_sampling_freq_alternative_to_tr(self, onsets_path):
        dm = DesignMatrix(onsets_path, run_length=1364, sampling_freq=0.5)
        assert dm.sampling_freq == 0.5

    def test_infer_run_length_rejected_for_events(self, onsets_path):
        with pytest.raises(ValueError, match="run_length='infer' is not valid"):
            DesignMatrix(onsets_path, run_length="infer", TR=2.0)


class TestDesignMatrixFromTabularFile:
    """`DesignMatrix(<confounds-style file>, ...)` reads as-is."""

    @pytest.fixture
    def confounds_path(self, tmp_path):
        # fmriprep-style: 3 motion regressors, n/a in the first row
        # (mimicking derivative columns whose lag-1 value is undefined).
        path = tmp_path / "sub-01_desc-confounds_timeseries.tsv"
        n = 50
        df = pd.DataFrame(
            {
                "trans_x": np.linspace(-0.1, 0.1, n),
                "trans_y": np.linspace(0.0, 0.2, n),
                "rot_z": np.linspace(0.0, 0.05, n),
            }
        )
        df.iloc[0, :] = np.nan
        df.to_csv(path, sep="\t", index=False, na_rep="n/a")
        return path

    def test_infer_run_length(self, confounds_path):
        dm = DesignMatrix(confounds_path, run_length="infer", TR=2.0)
        assert dm.shape == (50, 3)
        # No confounds/convolved set automatically — explicit on append() instead.
        assert dm.confounds == []

    def test_explicit_run_length_matches(self, confounds_path):
        dm = DesignMatrix(confounds_path, run_length=50, TR=2.0)
        assert dm.shape == (50, 3)

    def test_explicit_run_length_mismatch_errors(self, confounds_path):
        with pytest.raises(ValueError, match="row count|run_length"):
            DesignMatrix(confounds_path, run_length=100, TR=2.0)

    def test_na_strings_become_null(self, confounds_path):
        dm = DesignMatrix(confounds_path, run_length="infer", TR=2.0)
        # First row was "n/a" across all columns
        assert dm.data["trans_x"].is_null().sum() == 1

    def test_hrf_model_ignored_for_tabular(self, confounds_path):
        """`hrf_model` only fires on events files — tabular columns pass through unchanged."""
        dm_default = DesignMatrix(confounds_path, run_length="infer", TR=2.0)
        dm_explicit = DesignMatrix(
            confounds_path, run_length="infer", TR=2.0, hrf_model="glover"
        )
        assert list(dm_default.columns) == list(dm_explicit.columns)
        assert dm_default.convolved == [] == dm_explicit.convolved


class TestInitValidation:
    """File-path init requires run_length and exactly one of TR/sampling_freq."""

    def test_both_tr_and_sampling_freq_errors(self, onsets_path):
        with pytest.raises(ValueError, match="exactly one of"):
            DesignMatrix(onsets_path, run_length=1364, TR=2.0, sampling_freq=0.5)

    def test_missing_run_length_errors(self, onsets_path):
        with pytest.raises(ValueError, match="run_length"):
            DesignMatrix(onsets_path, TR=2.0)

    def test_missing_tr_or_sampling_freq_errors(self, onsets_path):
        with pytest.raises(ValueError, match="TR.*sampling_freq|sampling_freq.*TR"):
            DesignMatrix(onsets_path, run_length=1364)

    def test_tr_works_without_file(self):
        """TR is a generic convenience — works even without file input."""
        dm = DesignMatrix(np.zeros((10, 2)), TR=2.0, columns=["a", "b"])
        assert dm.sampling_freq == 0.5


class TestEventsToDmHelper:
    """The shared helper used by both file-path and datasets.py."""

    def test_accepts_pandas(self, onsets_data):
        df = events_to_dm(onsets_data, run_length=1364, sampling_freq=0.5)
        assert isinstance(df, pl.DataFrame)
        assert df.height == 1364

    def test_accepts_polars(self, onsets_data):
        # Construct via dict to avoid pyarrow on the pandas → polars hop.
        onsets_pl = pl.DataFrame(
            {col: onsets_data[col].tolist() for col in onsets_data.columns}
        )
        df = events_to_dm(onsets_pl, run_length=1364, sampling_freq=0.5)
        assert isinstance(df, pl.DataFrame)
        assert df.height == 1364

    def test_drops_constant(self, onsets_data):
        df = events_to_dm(onsets_data, run_length=1364, sampling_freq=0.5)
        assert "constant" not in df.columns


class TestAppendAsConfounds:
    """`append(axis=1, as_confounds=True)` promotes appended cols to confounds."""

    def test_promotes_dm_columns_to_confounds(self, onsets_path):
        events_dm = DesignMatrix(onsets_path, run_length=1364, TR=2.0)
        # A second DM whose cols should land in .confounds via as_confounds
        confounds_dm = DesignMatrix(
            np.random.default_rng(0).standard_normal((1364, 3)),
            sampling_freq=0.5,
            columns=["mx", "my", "mz"],
        )
        merged = events_dm.append(confounds_dm, axis=1, as_confounds=True)
        for c in confounds_dm.columns:
            assert c in merged.confounds
        # Original event cols stay non-nuisance
        for c in events_dm.columns:
            assert c not in merged.confounds

    def test_default_does_not_promote(self, onsets_path):
        events_dm = DesignMatrix(onsets_path, run_length=1364, TR=2.0)
        confounds_dm = DesignMatrix(
            np.zeros((1364, 2)), sampling_freq=0.5, columns=["mx", "my"]
        )
        merged = events_dm.append(confounds_dm, axis=1)
        assert merged.confounds == []
