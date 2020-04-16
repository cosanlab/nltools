from nltools.data import Design_Matrix
from nltools.file_reader import onsets_to_dm
from nltools.utils import get_resource_path
import numpy as np
import pandas as pd
import os


def test_onsets_to_dm():
    fpath = os.path.join(get_resource_path(), "onsets_example.txt")
    data = pd.read_csv(os.path.join(get_resource_path(), "onsets_example.txt"))
    sampling_freq = 0.5
    run_length = 1364
    Duration = 10
    TR = 1 / sampling_freq

    # Two-column
    # Test loading from a file
    dm = onsets_to_dm(fpath, sampling_freq, run_length)
    assert isinstance(dm, Design_Matrix)

    # Check it has run_length rows and nStim columns
    assert dm.shape == (run_length, data.Stim.nunique())

    # Get the unique number of presentations of each Stim from the original file
    stim_counts = data.Stim.value_counts(sort=False)[dm.columns]

    # Check there are only as many onsets as occurences of each Stim
    np.allclose(stim_counts.values, dm.sum().values)

    # Three-column with loading from dataframe
    data["Duration"] = Duration
    dm = onsets_to_dm(data, sampling_freq, run_length)

    # Check it has run_length rows and nStim columns
    assert dm.shape == (run_length, data.Stim.nunique())

    # Because timing varies in seconds and isn't TR-locked each stimulus should last at Duration/TR number of TRs and at most Duration/TR + 1 TRs
    # Check that the total number of TRs for each stimulus >= 1 + (Duration/TR) and <= 1 + (Duration/TR + 1)
    onsets = dm.sum().values
    durations = data.groupby("Stim").Duration.mean().values
    for o, c, d in zip(onsets, stim_counts, durations):
        assert c * (d / TR) <= o <= c * ((d / TR) + 1)

    # Multiple onsets
    dm = onsets_to_dm([data, data], sampling_freq, run_length)

    # Check it has run_length rows and nStim columns
    assert dm.shape == (run_length * 2, data.Stim.nunique())

    # Multiple onsets with polynomials auto-added
    dm = onsets_to_dm([data, data], sampling_freq, run_length, add_poly=2)
    assert dm.shape == (run_length * 2, data.Stim.nunique() + (3 * 2))
    
    dm = onsets_to_dm(
        [data, data], sampling_freq, run_length, add_poly=2, keep_separate=False
    )
    assert dm.shape == (run_length * 2, data.Stim.nunique() + 3)

    # Three-column from file with variable durations
    data = pd.read_csv(os.path.join(get_resource_path(), "onsets_example_with_dur.txt"))
    run_length = 472
    dm = onsets_to_dm(data, sampling_freq, run_length)

    assert dm.shape == (run_length, data.Stim.nunique())

    onsets = dm.sum().values
    stim_counts = data.Stim.value_counts().values
    durations = data.groupby("Stim").Duration.mean().values
    for o, c, d in zip(onsets, stim_counts, durations):
        assert c * (d / TR) <= o <= c * ((d / TR) + 1)
