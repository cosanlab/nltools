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


def onsets_to_dm(F, TR, runLength, header='infer', sort=False,
                addIntercept=False, **kwargs):
    """Function to read in a 2 or 3 column onsets file, specified in seconds,
        organized as: 'Stimulus,Onset','Onset,Stimulus','Stimulus,Onset,
        Duration', or 'Onset,Duration,Stimulus'.

        Args:
            df (str or dataframe): path to file or pandas dataframe
            TR (float): length of TR in seconds the run was collected at
            runLength (int): number of TRs in the run these onsets came from
            sort (bool, optional): whether to sort the columns of the resulting
                                    design matrix alphabetically; defaults to
                                    False
            addIntercept (bool, optional: whether to add an intercept to the
                                    resulting dataframe; defaults to False
            header (str,optional): None if missing header, otherwise pandas
                                    header keyword; defaults to 'infer'
            kwargs: additional inputs to pandas.read_csv

        Returns:
            Design_Matrix class

    """
    if isinstance(F,six.string_types):
        df = pd.read_csv(F,header=header,**kwargs)
    elif isinstance(F,pd.core.frame.DataFrame):
        df = F.copy()
    else:
        raise TypeError("Input needs to be file path or pandas dataframe!")
    if df.shape[1] == 2:
        warnings.warn("Only 2 columns in file, assuming all stimuli are the same duration")
    elif df.shape[1] == 1 or df.shape[1] > 3:
        raise ValueError("Can only handle files with 2 or 3 columns!")

    #Try to infer the header
    if header is None:
        possibleHeaders = ['Stim','Duration','Onset']
        if isinstance(df.iloc[0,0],six.string_types):
            df.columns = possibleHeaders[:df.shape[1]]
        elif isinstance(df.iloc[0,df.shape[1]-1],six.string_types):
            df.columns = possibleHeaders[1:] + [possibleHeaders[0]]
        else:
            raise ValueError("Can't figure out data organization. Make sure file has no more than 3 columns specified as 'Stim,Onset,Duration' or 'Onset,Duration,Stim'")
    df['Onset'] = df['Onset'].apply(lambda x: int(np.floor(x/TR)))

    #Build dummy codes
    X = Design_Matrix(columns=df['Stim'].unique(),
                    data=np.zeros([runLength,
                    len(df['Stim'].unique())]))
    for i, row in df.iterrows():
        if df.shape[1] == 3:
            dur = np.ceil(row['Duration']/TR)
            X.ix[row['Onset']-1:row['Onset']+dur-1, row['Stim']] = 1
        elif df.shape[1] == 2:
            X.ix[row['Onset'], row['Stim']] = 1
    X.TR = TR
    if sort:
        X = X.reindex_axis(sorted(X.columns), axis=1)

    if addIntercept:
        X['intercept'] = 1
        X.hasIntercept = True

    return X
