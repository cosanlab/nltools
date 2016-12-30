from __future__ import division

'''
NeuroLearn Statistics Tools
===========================

Tools to help with statistical analyses.

'''

__all__ = ['pearson', 
            'zscore', 
            'fdr', 
            'threshold', 
            'multi_threshold',
            'winsorize',
            'calc_bpm',
            'downsample',
            'fisher_r_to_z',
            'one_sample_permutation',
            'two_sample_permutation',
            'correlation_permutation']

import numpy as np
import pandas as pd
from scipy.stats import ss, pearsonr, spearmanr
from copy import deepcopy
import nibabel as nib

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
    """ zscore every column in a pandas dataframe or series.
        
        Args:
            df: Pandas DataFrame instance
        
        Returns:
            z_data: z-scored pandas DataFrame or series instance
    """

    if isinstance(df, pd.DataFrame):
        return df.apply(lambda x: (x - x.mean())/x.std())
    elif isinstance(df, pd.Series):
        return (df-np.mean(df))/np.std(df)
    else:
        raise ValueError("Data is not a Pandas DataFrame or Series instance")

def fdr(p, q=.05):
    """ Determine FDR threshold given a p value array and desired false
    discovery rate q. Written by Tal Yarkoni 

    Args:
        p: vector of p-values (numpy array) (only considers non-zero p-values)
        q: false discovery rate level
 
    Returns:
        fdr_p: p-value threshold based on independence or positive dependence

    """
    
    if not isinstance(p, np.ndarray):
        raise ValueError('Make sure vector of p-values is a numpy array')

    s = np.sort(p)
    nvox = p.shape[0]
    null = np.array(range(1, nvox + 1), dtype='float') * q / nvox
    below = np.where(s <= null)[0]
    fdr_p = s[max(below)] if any(below) else -1
    return fdr_p

def threshold(stat, p, thr=.05):
    """ Threshold test image by p-value from p image

    Args:
        stat: Brain_Data instance of arbitrary statistic metric (e.g., beta, t, etc)
        p: Brain_data instance of p-values
        threshold: p-value to threshold stat image
 
    Returns:
        out: Thresholded Brain_Data instance
    
    """
    from nltools.data import Brain_Data

    if not isinstance(stat, Brain_Data):
        raise ValueError('Make sure stat is a Brain_Data instance')
        
    if not isinstance(p, Brain_Data):
        raise ValueError('Make sure p is a Brain_Data instance')

    # Create Mask
    mask = deepcopy(p)
    if thr >0:
        mask.data = (mask.data<thr).astype(int)
    else:
        mask.data = np.zeros(len(mask.data),dtype=int)
   
    # Apply Threshold Mask
    out = deepcopy(stat)
    if np.sum(mask.data) > 0:
        out = out.apply_mask(mask)
        out.data = out.data.squeeze()
    else:
        out.data = np.zeros(len(mask.data),dtype=int)

    return out

def multi_threshold(t_map,p_map,thresh):
    """ Threshold test image by multiple p-value from p image

    Args:
        stat: Brain_Data instance of arbitrary statistic metric (e.g., beta, t, etc)
        p: Brain_data instance of p-values
        threshold: list of p-values to threshold stat image
 
    Returns:
        out: Thresholded Brain_Data instance
    
    """
    from nltools.data import Brain_Data

    if not isinstance(t_map, Brain_Data):
        raise ValueError('Make sure stat is a Brain_Data instance')
        
    if not isinstance(p_map, Brain_Data):
        raise ValueError('Make sure p is a Brain_Data instance')
        
    if not isinstance(thresh,list):
        raise ValueError('Make sure thresh is a list of p-values')

    affine = t_map.to_nifti().get_affine()
    pos_out = np.zeros(t_map.to_nifti().shape)
    neg_out = deepcopy(pos_out)
    for thr in thresh:
        t = threshold(t_map,p_map,thr=thr)
        t_pos = deepcopy(t)
        t_pos.data = np.zeros(len(t_pos.data))
        t_neg = deepcopy(t_pos)
        t_pos.data[t.data>0] = 1
        t_neg.data[t.data<0] = 1
        pos_out = pos_out+t_pos.to_nifti().get_data()
        neg_out = neg_out+t_neg.to_nifti().get_data()
    pos_out = pos_out + neg_out*-1
    return Brain_Data(nib.Nifti1Image(pos_out,affine))

def winsorize(data, cutoff=None):
    ''' Winsorize a Pandas Series 
    
        Args:
            data: a pandas.Series
            cutoff: a dictionary with keys {'std':[low,high]} or {'quantile':[low,high]}
            
        Returns:
            pandas.Series
    '''
    
    if not isinstance(data,pd.Series):
        raise ValueError('Make sure that you are applying winsorize to a pandas series.')
    
    if isinstance(cutoff,dict):
        if 'quantile' in cutoff:
            q = data.quantile(cutoff['quantile'])
        elif 'std' in cutoff:
            std = [data.mean()-data.std()*cutoff['std'][0],data.mean()+data.std()*cutoff['std'][1]]
            q = pd.Series(index=cutoff['std'],data=std)
    else:
        raise ValueError('cutoff must be a dictionary with quantile or std keys.')
    if isinstance(q, pd.Series) and len(q) == 2:
        data[data < q.iloc[0]] = q.iloc[0]
        data[data > q.iloc[1]] = q.iloc[1]
    return data

def calc_bpm(beat_interval, sampling_freq):
    ''' Calculate instantaneous BPM from beat to beat interval

        Args: 
            beat_interval: number of samples in between each beat (typically R-R Interval)
            sampling_freq: sampling frequency in Hz

        Returns:
            bpm:  beats per minute for time interval
    '''
    return 60*sampling_freq*(1/(beat_interval))

def downsample(data,sampling_freq=None, target=None, target_type='samples'):
    ''' Downsample pandas to a new target frequency or number of samples using averaging.
    
        Args:
            data: Pandas DataFrame or Series
            sampling_freq:  Sampling frequency of data 
            target: downsampling target
            target_type: type of target can be [samples,seconds,hz]
        Returns:
            downsampled pandas object
            
    '''
      
    if not isinstance(data,(pd.DataFrame,pd.Series)):
        raise ValueError('Data must by a pandas DataFrame or Series instance.')
               
    if target_type is 'samples':
        n_samples = target
    elif target_type is 'seconds':
        n_samples = target*sampling_freq
    elif target_type is 'hz':
        n_samples = sampling_freq/target
    else:
        raise ValueError('Make sure target_type is "samples", "seconds", or "hz".')

    idx = np.sort(np.repeat(np.arange(1,data.shape[0]/n_samples,1),n_samples))
    # if data.shape[0] % n_samples:
    if data.shape[0] > len(idx):
        idx = np.concatenate([idx, np.repeat(idx[-1]+1,data.shape[0]-len(idx))])
    return data.groupby(idx).mean()

def fisher_r_to_z(r):
    ''' Use Fisher transformation to convert correlation to z score '''

    return .5*np.log((1+r)/(1-r))

def one_sample_permutation(data, n_permute = 5000):
    ''' One sample permutation test using randomization.

        Args:
            data: Pandas DataFrame or Series or numpy array
            n_permute: (int) number of permutations
        Returns:
            stats: (dict) dictionary of permutation results ['mean','p']
    
    '''

    data = np.array(data)
    stats = dict()
    stats['mean'] = np.mean(data)
    all_p = []
    for p in range(n_permute):
        all_p.append(np.mean(data*np.random.choice([1,-1],len(data))))
    if stats['mean']>=0:
        stats['p'] = np.mean(all_p>=stats['mean'])
    else:
        stats['p'] = np.mean(all_p<=stats['mean'])
    return stats

def two_sample_permutation(data1, data2, n_permute=5000):
    ''' Independent sample permutation test.

        Args:
            data1: Pandas DataFrame or Series or numpy array
            data2: Pandas DataFrame or Series or numpy array
            n_permute: (int) number of permutations
        Returns:
            stats: (dict) dictionary of permutation results ['mean','p']
    
    '''

    stats = dict()
    stats['mean'] = np.mean(data1)-np.mean(data2)
    data = pd.DataFrame(data={'Values':data1,'Group':np.ones(len(data1))})
    data = data.append(pd.DataFrame(data={'Values':data2,'Group':np.zeros(len(data2))}))
    all_p = []
    for p in range(n_permute):
        perm_label = np.random.permutation(data['Group'])
        all_p.append(np.mean(data.loc[perm_label==1,'Values'])-np.mean(data.loc[perm_label==0,'Values']))
    if stats['mean']>=0:
        stats['p'] = np.mean(all_p>=stats['mean'])
    else:
        stats['p'] = np.mean(all_p<=stats['mean'])
    return stats

def correlation_permutation(data1, data2, n_permute=5000, metric='spearman'):
    ''' Permute correlation.

        Args:
            data1: Pandas DataFrame or Series or numpy array
            data2: Pandas DataFrame or Series or numpy array
            n_permute: (int) number of permutations
            metric: (str) type of association metric ['spearman','pearson']
            
        Returns:
            stats: (dict) dictionary of permutation results ['correlation','p']
    
    '''

    stats = dict()
    data1 = np.array(data1)
    data2 = np.array(data2)

    if metric is 'spearman':
        stats['correlation'] = spearmanr(data1,data2)[0]
    elif metric is 'pearson':
        stats['correlation'] = pearsonr(data1,data2)[0]
    else:
        raise ValueError('metric must be "spearman" or "pearson"')

    all_p = []
    for p in range(n_permute):
        if metric is 'spearman':
            stats['correlation'] = spearmanr(data1,data2)[0]
            all_p.append(spearmanr(np.random.permutation(data1),data2)[0])
        elif metric is 'pearson':
            all_p.append(pearsonr(np.random.permutation(data1),data2)[0])
    if stats['correlation']>=0:
        stats['p'] = np.mean(all_p>=stats['correlation'])
    else:
        stats['p'] = np.mean(all_p<=stats['correlation'])
    return stats


