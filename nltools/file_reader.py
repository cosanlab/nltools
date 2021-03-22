"""
NeuroLearn File Reading Tools
=============================

"""

__all__ = ["onsets_to_dm"]
__author__ = ["Eshin Jolly"]
__license__ = "MIT"

import pandas as pd
import numpy as np
from nltools.data import Design_Matrix
import warnings
from pathlib import Path


def onsets_to_dm(
    F,
    sampling_freq,
    run_length,
    header="infer",
    sort=False,
    keep_separate=True,
    add_poly=None,
    unique_cols=None,
    fill_na=None,
    **kwargs,
):
    """
    This function can assist in reading in one or several in a 2-3 column onsets files, specified in seconds and converting it to a Design Matrix organized as samples X Stimulus Classes. sampling_freq should be specified in hertz; for TRs use hertz = 1/TR. Onsets files **must** be organized with columns in one of the following 4 formats:

    1) 'Stim, Onset'
    2) 'Onset, Stim'
    3) 'Stim, Onset, Duration'
    4) 'Onset, Duration, Stim'

    No other file organizations are currently supported. *Note:* Stimulus offsets (onset + duration) that fall into an adjacent TR include that full TR. E.g. offset of 10.16s with TR = 2 has an offset of TR 5, which spans 10-12s, rather than an offset of TR 4, which spans 8-10s.

    Args:
        F (filepath/DataFrame/list): path to file, pandas dataframe, or list of files or pandas dataframes
        sampling_freq (float): sampling frequency in hertz; for TRs use (1 / TR)         run_length (int): number of TRs in the run these onsets came from
        sort (bool, optional): whether to sort the columns of the resulting
                                design matrix alphabetically; defaults to
                                False
        addpoly (int, optional: what order polynomial terms to add as new columns (e.g. 0 for intercept, 1 for linear trend and intercept, etc); defaults to None
        header (str,optional): None if missing header, otherwise pandas
                                header keyword; defaults to 'infer'
        keep_separate (bool): whether to seperate polynomial columns if reading a list of files and using the addpoly option; defaults to True
        unique_cols (list, optional): additional columns to keep seperate across files (e.g. spikes); defaults to []
        fill_na (str/int/float, optional): what value fill NaNs in with if reading in a list of files; defaults to None
        kwargs: additional inputs to pandas.read_csv

        Returns:
            Design_Matrix class

    """

    if not isinstance(F, list):
        F = [F]

    if not isinstance(sampling_freq, (float, np.floating)):
        raise TypeError("sampling_freq must be a float")

    out = []
    TR = 1.0 / sampling_freq
    for f in F:
        if isinstance(f, str) or isinstance(f, Path):
            df = pd.read_csv(f, header=header, **kwargs)
        elif isinstance(f, pd.core.frame.DataFrame):
            df = f.copy()
        else:
            raise TypeError("Input needs to be file path or pandas dataframe!")
        # Keep an unaltered copy of the original dataframe for checking purposes below
        data = df.copy()

        if df.shape[1] == 2:
            warnings.warn(
                "Only 2 columns in file, assuming all stimuli are the same duration"
            )
        elif df.shape[1] == 1 or df.shape[1] > 3:
            raise ValueError("Can only handle files with 2 or 3 columns!")

        # Try to infer the header
        if header is None:
            possibleHeaders = ["Stim", "Onset", "Duration"]
            if isinstance(df.iloc[0, 0], str):
                df.columns = possibleHeaders[: df.shape[1]]
            elif isinstance(df.iloc[0, df.shape[1] - 1], str):
                df.columns = possibleHeaders[1:] + [possibleHeaders[0]]
            else:
                raise ValueError(
                    "Can't figure out onset file organization. Make sure file has no more than 3 columns specified as 'Stim,Onset,Duration' or 'Onset,Duration,Stim'"
                )

        # Compute an offset in seconds if a Duration is provided
        if df.shape[1] == 3:
            df["Offset"] = df["Onset"] + df["Duration"]
        # Onset always starts at the closest TR rounded down, e.g.
        # with TR = 2, and onset = 10.1 or 11.7 will both have onset of TR 5 as it spans the window 10-12s
        df["Onset"] = df["Onset"].apply(lambda x: int(np.floor(x / TR)))

        # Offset includes the subsequent if Offset falls within window covered by that TR
        # but not if it falls exactly on the subsequent TR, e.g. if TR = 2, and offset = 10.16, then TR 5 will be included but if offset = 10.00, TR 5 will not be included, as it covers the window 10-12s
        if "Offset" in df.columns:

            def conditional_round(x, TR):
                """Conditional rounding to the next TR if offset falls within window, otherwise not"""
                dur_in_TRs = x / TR
                dur_in_TRs_rounded_down = np.floor(dur_in_TRs)
                # If in the future we wanted to enable the ability to include a TR based on a % of that TR we can change the next line to compare to some value, e.g. at least 0.5s into that TR: dur_in_TRs - dur_in_TRs_rounded_down > 0.5
                if dur_in_TRs > dur_in_TRs_rounded_down:
                    return dur_in_TRs_rounded_down
                else:
                    return dur_in_TRs_rounded_down - 1

            # Apply function
            df["Offset"] = df["Offset"].apply(conditional_round, args=(TR,))

        # Build dummy codes
        X = Design_Matrix(
            np.zeros([run_length, df["Stim"].nunique()]),
            columns=df["Stim"].unique(),
            sampling_freq=sampling_freq,
        )
        for i, row in df.iterrows():
            if "Offset" in df.columns:
                X.loc[row["Onset"] : row["Offset"], row["Stim"]] = 1
            else:
                X.loc[row["Onset"], row["Stim"]] = 1
        # Run a check
        if "Offset" in df.columns:
            onsets = X.sum().values
            stim_counts = data.Stim.value_counts(sort=False)[X.columns]
            durations = data.groupby("Stim").Duration.mean().values
            for i, (o, c, d) in enumerate(zip(onsets, stim_counts, durations)):
                if c * (d / TR) <= o <= c * ((d / TR) + 1):
                    pass
                else:
                    warnings.warn(
                        f"Computed onsets for {data.Stim.unique()[i]} are inconsistent with expected values. Please manually verify the outputted Design_Matrix!"
                    )

        if sort:
            X = X.reindex(sorted(X.columns), axis=1)

        out.append(X)
    if len(out) > 1:
        if add_poly is not None:
            out = [e.add_poly(add_poly) for e in out]

        out_dm = out[0].append(
            out[1:],
            keep_separate=keep_separate,
            unique_cols=unique_cols,
            fill_na=fill_na,
        )
    else:
        out_dm = out[0]
        if add_poly is not None:
            out_dm = out_dm.add_poly(add_poly)
        if fill_na is not None:
            out_dm = out_dm.fill_na(fill_na)

    return out_dm
