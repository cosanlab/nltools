"""Various statistical helper functions"""

__all__ = ['pearson', 'zscore', 'threshold', 'fdr']

import numpy as np
from scipy.stats import ss
from nltools.data import Brain_Data
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
    
    z_df = df.apply(lambda x: (x - x.mean())/x.std())

    return z_df

def threshold(stat, p, threshold_dict={'unc':.001}):
    """ Calculate one sample t-test across each voxel (two-sided)

    Args:
        stat: Brain_Data instance of arbitrary statistic metric (e.g., beta, t, etc)
        p: Brain_data instance of p-values
        threshold_dict: a dictionary of threshold parameters {'unc':.001} or {'fdr':.05}
 
    Returns:
        out: Thresholded Brain_Data instance
    
    """
 
    if not isinstance(stat, Brain_Data):
        raise ValueError('Make sure stat is a Brain_Data instance')
        
    if not isinstance(p, Brain_Data):
        raise ValueError('Make sure p is a Brain_Data instance')

    out = deepcopy(stat)
    if 'unc' in threshold_dict:
        out.data[p.data > threshold_dict['unc']] = np.nan
    elif 'fdr' in threshold_dict:
        out.data[p.data > threshold_dict['fdr']] = np.nan
    return out

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





    