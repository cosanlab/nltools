'''
NeuroLearn File Reading Tools
=============================

'''

__all__ = ['onsets_to_dm']
__author__ = ["Eshin Jolly"]
__license__ = "MIT"

import pandas as pd
import numpy as np
import six
from nltools.data import Design_Matrix
import warnings


def onsets_to_dm(F, sampling_freq, run_length, header='infer', sort=False, keep_separate=True, add_poly=None, unique_cols=[], fill_na=None, **kwargs):
    """
    This function can assist in reading in one or several in a 2-3 column onsets files, specified in seconds and converting it to a Design Matrix organized as samples X Stimulus Classes. Onsets files **must** be organized with columns in one of the following 4 formats:

    1) 'Stim, Onset'
    2) 'Onset, Stim'
    3) 'Stim, Onset, Duration'
    4) 'Onset, Duration, Stim'

    No other file organizations are currently supported

    Args:
        F (filepath/DataFrame/list): path to file, pandas dataframe, or list of files or pandas dataframes
        sampling_freq (float): sampling frequency in hertz; for TRs use (1 / TR)         run_length (int): number of TRs in the run these onsets came from
        sort (bool, optional): whether to sort the columns of the resulting
                                design matrix alphabetically; defaults to
                                False
        addpoly (int, optional: what order polynomial terms to add as new columns (e.g. 0 for intercept, 1 for linear trend and intercept, etc); defaults to None
        header (str,optional): None if missing header, otherwise pandas
                                header keyword; defaults to 'infer'
        keep_separate (bool): whether to seperate polynomial columns if reading a list of files and using the addpoly option
        unique_cols (list): additional columns to keep seperate across files (e.g. spikes)
        fill_nam (str/int/float): what value fill NaNs in with if reading in a list of files
        kwargs: additional inputs to pandas.read_csv

        Returns:
            Design_Matrix class

    """
    if not isinstance(F, list):
        F = [F]

    out = []
    TR = 1. / sampling_freq
    for f in F:
        if isinstance(f, six.string_types):
            df = pd.read_csv(f, header=header, **kwargs)
        elif isinstance(f, pd.core.frame.DataFrame):
            df = f.copy()
        else:
            raise TypeError("Input needs to be file path or pandas dataframe!")
        if df.shape[1] == 2:
            warnings.warn("Only 2 columns in file, assuming all stimuli are the same duration")
        elif df.shape[1] == 1 or df.shape[1] > 3:
            raise ValueError("Can only handle files with 2 or 3 columns!")

        # Try to infer the header
        if header is None:
            possibleHeaders = ['Stim', 'Onset', 'Duration']
            if isinstance(df.iloc[0, 0], six.string_types):
                df.columns = possibleHeaders[:df.shape[1]]
            elif isinstance(df.iloc[0, df.shape[1]-1], six.string_types):
                df.columns = possibleHeaders[1:] + [possibleHeaders[0]]
            else:
                raise ValueError("Can't figure out onset file organization. Make sure file has no more than 3 columns specified as 'Stim,Onset,Duration' or 'Onset,Duration,Stim'")
        df['Onset'] = df['Onset'].apply(lambda x: int(np.floor(x/TR)))

        # Build dummy codes
        X = Design_Matrix(np.zeros([run_length, len(df['Stim'].unique())]), columns=df['Stim'].unique(), sampling_freq=sampling_freq)
        for i, row in df.iterrows():
            if df.shape[1] == 3:
                dur = np.ceil(row['Duration']/TR)
                X.ix[row['Onset']-1:row['Onset']+dur-1, row['Stim']] = 1
            elif df.shape[1] == 2:
                X.ix[row['Onset'], row['Stim']] = 1
        if sort:
            X = X.reindex(sorted(X.columns), axis=1)

        out.append(X)
    if len(out) > 1:
        out_dm = out[0].append(out[1:], keep_separate=keep_separate, add_poly=add_poly, unique_cols=unique_cols, fill_na=fill_na)
    else:
        if add_poly is not None:
            out_dm = out[0].add_poly(add_poly)
        else:
            out_dm = out[0]

    return out_dm
