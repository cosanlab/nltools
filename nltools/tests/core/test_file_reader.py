from nltools.data import Design_Matrix
from nltools.file_reader import onsets_to_dm
from nltools.utils import get_resource_path
import numpy as np
import pandas as pd
import os


def test_onsets_to_dm():
    fpath = os.path.join(get_resource_path(), "onsets_example.csv")
    data = pd.read_csv(os.path.join(get_resource_path(), "onsets_example.csv"))
    run_length = 1364
    TR = 2.0

    # Auto-convolves with glover, disable for first checks
    dm = onsets_to_dm(fpath, run_length, TR, hrf_model=None)
    assert isinstance(dm, Design_Matrix)

    # Check it has run_length rows and nStim columns + intercept
    assert dm.shape == (run_length, data.trial_type.nunique() + 1)

    # Drop intercept to simplify next checks
    dm = dm.drop(columns=["constant"])

    # Get the unique number of presentations of each trial_type from the original file
    stim_counts = data.trial_type.value_counts(sort=False)[dm.columns]

    # Check there are only as many onsets as occurences of each trial_type
    # Each event has duration=10s, with TR=2s that's 5 TRs per event
    # So sum should equal count × (duration/TR)
    # Use Polars Series .to_numpy() to convert to numpy array for comparison
    duration = 10.0  # From onsets_example.csv
    expected_sum = stim_counts.values * (duration / TR)
    assert np.allclose(expected_sum, dm.sum().to_numpy())

    # Three-column with loading from dataframe
    dm = onsets_to_dm([data, data], [run_length, run_length], TR)

    # Check it has run_length rows and nStim columns
    assert isinstance(dm, list)
    assert len(dm) == 2
    # Use Pythonic equality operator (calls __eq__)
    assert dm[0] == dm[0]
