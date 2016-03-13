"""Various statistical helper functions"""

__all__ = ['pearson', 'zscore', 'fdr']

import numpy as np
import pandas as pdg
from scipy.stats import ss
from copy import deepcopy

def pearson(x, y):
    """ Correlates row vector x with each row vector in 2D array y. 
    From neurosynth.stats.py - author: Tal Yarkoni
    """
    data = np.vstack((x, y))
    ms = data.mean(axis=1)[(slice(None, None, None), None)]
    datam = data - ms
    datass = np.sqrt(ss(datam, axis=1))
    temp = np.dot(datam[1:], datam[0].T)
    rs = temp / (datass[1:] * datass[0])
    return rs

def zscore(df):
    """ zscore every column in a pandas dataframe.
        
        Args:
            df: Pandas DataFrame instance
        
        Returns:
            z_data: z-scored pandas DataFrame instance
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Data is not a Pandas DataFrame instance")
    
    return df.apply(lambda x: (x - x.mean())/x.std())

def fdr(p, q=.05):
    """ Determine FDR threshold given a p value array and desired false
    discovery rate q. Written by Tal Yarkoni 

    Args:
        p: vector of p-values (numpy array) (only considers non-zero p-values)
        q: false discovery rate level
 
    Returns:
        fdr_p: p-value threshold based on independence or positive dependence

    """
    
    if not isinstance(p, np.array):
        raise ValueError('Make sure vector of p-values is a numpy array')

    s = np.sort(p)
    nvox = p.shape[0]
    null = np.array(range(1, nvox + 1), dtype='float') * q / nvox
    below = np.where(s <= null)[0]
    fdr_p = s[max(below)] if any(below) else -1
    return fdr_p



    