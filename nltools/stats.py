from __future__ import division

'''
NeuroLearn Statistics Tools
===========================

Tools to help with statistical analyses.

'''

__all__ = ['pearson',
            'zscore',
            'fdr',
            'holm_bonf',
            'threshold',
            'multi_threshold',
            'winsorize',
            'trim',
            'calc_bpm',
            'downsample',
            'upsample',
            'fisher_r_to_z',
            'one_sample_permutation',
            'two_sample_permutation',
            'correlation_permutation',
            'make_cosine_basis',
            '_robust_estimator',
            'summarize_bootstrap',
            'regress',
            'procrustes',
            'align']

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau, norm
from scipy.stats import t as t_dist
from copy import deepcopy
import nibabel as nib
from scipy.interpolate import interp1d
import warnings
import itertools
from joblib import Parallel, delayed
import six
from .utils import attempt_to_import
from .external.srm import SRM, DetSRM
from scipy.linalg import orthogonal_procrustes
from sklearn.utils import check_random_state

MAX_INT = np.iinfo(np.int32).max

# Optional dependencies
sm = attempt_to_import('statsmodels.tsa.arima_model',name='sm')

def pearson(x, y):
    """ Correlates row vector x with each row vector in 2D array y.
    From neurosynth.stats.py - author: Tal Yarkoni
    """
    data = np.vstack((x, y))
    ms = data.mean(axis=1)[(slice(None, None, None), None)]
    datam = data - ms
    datass = np.sqrt(np.sum(datam*datam, axis=1))
    # datass = np.sqrt(ss(datam, axis=1))
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

def holm_bonf(p, alpha=.05):
    """ Compute corrected p-values based on the Holm-Bonferroni method, i.e. step-down procedure applying iteratively less correction to highest p-values. Modified from stats.models approach.

    Args:
        p: vector of p-values (numpy array)
        alpha: alpha level

    Returns:
        p_corrected: numpy array of corrected p-values

    """

    if not isinstance(p, np.ndarray):
        raise ValueError('Make sure vector of p-values is a numpy array')

    idx = p.argsort()
    s = p[idx]
    s *= np.arange(len(s),0,-1)
    s = np.maximum.accumulate(s)
    p_corrected = np.empty_like(p)
    p_corrected[idx] = s

    return p_corrected

def threshold(stat, p, thr=.05):
    """ Threshold test image by p-value from p image

    Args:
        stat: Brain_Data instance of arbitrary statistic metric (e.g., beta,
            t, etc)
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
    if thr > 0:
        mask.data = (mask.data < thr).astype(int)
    else:
        mask.data = np.zeros(len(mask.data), dtype=int)

    # Apply Threshold Mask
    out = deepcopy(stat)
    if np.sum(mask.data) > 0:
        out = out.apply_mask(mask)
        out.data = out.data.squeeze()
    else:
        out.data = np.zeros(len(mask.data), dtype=int)

    return out

def multi_threshold(t_map, p_map,thresh):
    """ Threshold test image by multiple p-value from p image

    Args:
        stat: Brain_Data instance of arbitrary statistic metric
            (e.g., beta, t, etc)
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

    if not isinstance(thresh, list):
        raise ValueError('Make sure thresh is a list of p-values')

    affine = t_map.to_nifti().get_affine()
    pos_out = np.zeros(t_map.to_nifti().shape)
    neg_out = deepcopy(pos_out)
    for thr in thresh:
        t = threshold(t_map, p_map, thr=thr)
        t_pos = deepcopy(t)
        t_pos.data = np.zeros(len(t_pos.data))
        t_neg = deepcopy(t_pos)
        t_pos.data[t.data > 0] = 1
        t_neg.data[t.data < 0] = 1
        pos_out = pos_out+t_pos.to_nifti().get_data()
        neg_out = neg_out+t_neg.to_nifti().get_data()
    pos_out = pos_out + neg_out*-1
    return Brain_Data(nib.Nifti1Image(pos_out, affine))

def winsorize(data, cutoff=None,replace_with_cutoff=True):
    ''' Winsorize a Pandas DataFrame or Series with the largest/lowest value not considered outlier

        Args:
            data: a pandas.DataFrame or pandas.Series
            cutoff: a dictionary with keys {'std':[low,high]} or
                    {'quantile':[low,high]}
            replace_with_cutoff (default: False): If True, replace outliers with cutoff.
                    If False, replaces outliers with closest existing values.
        Returns:
            winsorized pandas.DataFrame or pandas.Series
    '''
    return _transform_outliers(data, cutoff,replace_with_cutoff=replace_with_cutoff,method='winsorize')

def trim(data, cutoff=None):
    ''' Trim a Pandas DataFrame or Series by replacing outlier values with NaNs

        Args:
            data: a pandas.DataFrame or pandas.Series
            cutoff: a dictionary with keys {'std':[low,high]} or
                    {'quantile':[low,high]}
        Returns:
            trimmed pandas.DataFrame or pandas.Series
    '''
    return _transform_outliers(data, cutoff, replace_with_cutoff=None, method='trim')

def _transform_outliers(data, cutoff, replace_with_cutoff, method):
    ''' This function is not exposed to user but is called by either trim or winsorize.

        Args:
            data: a pandas.DataFrame or pandas.Series
            cutoff: a dictionary with keys {'std':[low,high]} or
                    {'quantile':[low,high]}
            replace_with_cutoff (default: False): If True, replace outliers with cutoff.
                    If False, replaces outliers with closest existing values.
            method: 'winsorize' or 'trim'
        Returns:
            transformed pandas.DataFrame or pandas.Series
    '''
    df = data.copy() # To not overwrite data make a copy
    def _transform_outliers_sub(data, cutoff, replace_with_cutoff, method='trim'):
        if not isinstance(data,pd.Series):
            raise ValueError('Make sure that you are applying winsorize to a pandas dataframe or series.')
        if isinstance(cutoff,dict):
            # calculate cutoff values
            if 'quantile' in cutoff:
                q = data.quantile(cutoff['quantile'])
            elif 'std' in cutoff:
                std = [data.mean()-data.std()*cutoff['std'][0], data.mean()+data.std()*cutoff['std'][1]]
                q = pd.Series(index=cutoff['std'], data=std)
            # if replace_with_cutoff is false, replace with true existing values closest to cutoff
            if method=='winsorize':
                if not replace_with_cutoff:
                    q.iloc[0] = data[data > q.iloc[0]].min()
                    q.iloc[1] = data[data < q.iloc[1]].max()
        else:
            raise ValueError('cutoff must be a dictionary with quantile or std keys.')
        if method == 'winsorize':
            if isinstance(q, pd.Series) and len(q) == 2:
                data[data < q.iloc[0]] = q.iloc[0]
                data[data > q.iloc[1]] = q.iloc[1]
        elif method== 'trim':
                data[data < q.iloc[0]] = np.nan
                data[data > q.iloc[1]] = np.nan
        return data

    # transform each column if a dataframe, if series just transform data
    if isinstance(df,pd.DataFrame):
        for col in df.columns:
            df.loc[:,col] = _transform_outliers_sub(df.loc[:,col],cutoff=cutoff,replace_with_cutoff=replace_with_cutoff, method=method)
        return df
    elif isinstance(df,pd.Series):
        return _transform_outliers_sub(df,cutoff=cutoff,replace_with_cutoff=replace_with_cutoff, method=method)
    else:
        raise ValueError('Data must be a pandas DataFrame or Series')

def calc_bpm(beat_interval, sampling_freq):
    ''' Calculate instantaneous BPM from beat to beat interval

        Args:
            beat_interval: number of samples in between each beat
                            (typically R-R Interval)
            sampling_freq: sampling frequency in Hz

        Returns:
            bpm:  beats per minute for time interval
    '''
    return 60*sampling_freq*(1/(beat_interval))

def downsample(data,sampling_freq=None, target=None, target_type='samples',
            method='mean'):
    ''' Downsample pandas to a new target frequency or number of samples
        using averaging.

        Args:
            data: Pandas DataFrame or Series
            sampling_freq:  Sampling frequency of data in hertz
            target: downsampling target
                    target_type: type of target can be [samples,seconds,hz]
            method: (str) type of downsample method ['mean','median'],
                    default: mean
        Returns:
            downsampled pandas object

    '''

    if not isinstance(data,(pd.DataFrame,pd.Series)):
        raise ValueError('Data must by a pandas DataFrame or Series instance.')
    if not (method=='median') | (method=='mean'):
        raise ValueError("Metric must be either 'mean' or 'median' ")

    if target_type is 'samples':
        n_samples = target
    elif target_type is 'seconds':
        n_samples = target*sampling_freq
    elif target_type is 'hz':
        n_samples = sampling_freq/target
    else:
        raise ValueError('Make sure target_type is "samples", "seconds", '
                        ' or "hz".')

    idx = np.sort(np.repeat(np.arange(1,data.shape[0]/n_samples,1),n_samples))
    # if data.shape[0] % n_samples:
    if data.shape[0] > len(idx):
        idx = np.concatenate([idx, np.repeat(idx[-1]+1,data.shape[0]-len(idx))])
    if method=='mean':
        return data.groupby(idx).mean().reset_index(drop=True)
    elif method=='median':
        return data.groupby(idx).median().reset_index(drop=True)

def upsample(data,sampling_freq=None, target=None, target_type='samples',method='linear'):
    ''' Upsample pandas to a new target frequency or number of samples using interpolation.

        Args:
            data: Pandas Series or DataFrame (Note: will drop non-numeric columns from DataFrame)
            sampling_freq:  Sampling frequency of data in hertz
            target: downsampling target
            target_type: type of target can be [samples,seconds,hz]
            method (str):'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic' where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of zeroth, first, second or third order  default: linear
        Returns:
            upsampled pandas object

    '''

    methods = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
    if not method in methods:
        raise ValueError("Method must be 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'")

    if target_type is 'samples':
        n_samples = target
    elif target_type is 'seconds':
        n_samples = target*sampling_freq
    elif target_type is 'hz':
        n_samples = float(sampling_freq)/float(target)
    else:
        raise ValueError('Make sure target_type is "samples", "seconds", or "hz".')

    orig_spacing = np.arange(0,data.shape[0],1)
    new_spacing = np.arange(0,data.shape[0]-1,n_samples)

    if isinstance(data,pd.Series):
        interpolate = interp1d(orig_spacing,data,kind=method)
        return interpolate(new_spacing)
    elif isinstance(data,pd.DataFrame):
        numeric_data = data._get_numeric_data()
        if data.shape[1] != numeric_data.shape[1]:
            warnings.warn('Dropping %s non-numeric columns' % (data.shape[1] - numeric_data.shape[1]), UserWarning)
        out = pd.DataFrame(columns=numeric_data.columns, index=None)
        for i,x in numeric_data.iteritems():
            interpolate = interp1d(orig_spacing, x, kind=method)
            out.loc[:, i] = interpolate(new_spacing)
        return out
    else:
        raise ValueError('Data must by a pandas DataFrame or Series instance.')

def fisher_r_to_z(r):
    ''' Use Fisher transformation to convert correlation to z score '''

    return .5*np.log((1+r)/(1-r))

def _permute_sign(data, random_state=None):
    random_state = check_random_state(random_state)
    return np.mean(data*random_state.choice([1, -1], len(data)))

def _permute_group(data, random_state=None):
    random_state = check_random_state(random_state)
    perm_label = random_state.permutation(data['Group'])
    return (np.mean(data.loc[perm_label==1, 'Values']) -
            np.mean(data.loc[perm_label==0, 'Values']))


def _calc_pvalue(all_p, stat, tail):
    """Calculates p value based on distribution of correlations
    This function is called by the permutation functions
        all_p: list of correlation values from permutation
        stat: actual value being tested, i.e., stats['correlation'] or stats['mean']
        tail: (int) either 2 or 1 for two-tailed p-value or one-tailed
    """
    if tail==2:
        p= np.mean( np.abs(all_p) >= np.abs(stat))
    elif tail==1:
        if stat >= 0:
            p = np.mean(all_p >= stat)
        else:
            p = np.mean(all_p <= stat)
    else:
        raise ValueError('tail must be either 1 or 2')
    return p

def one_sample_permutation(data, n_permute=5000, tail=2, n_jobs=-1, random_state=None):
    ''' One sample permutation test using randomization.

        Args:
            data: Pandas DataFrame or Series or numpy array
            n_permute: (int) number of permutations
            tail: (int) either 1 for one-tail or 2 for two-tailed test (default: 2)
            n_jobs: (int) The number of CPUs to use to do the computation.
                    -1 means all CPUs.

        Returns:
            stats: (dict) dictionary of permutation results ['mean','p']

    '''

    random_state = check_random_state(random_state)
    seeds = random_state.randint(MAX_INT, size=n_permute)

    data = np.array(data)
    stats = dict()
    stats['mean'] = np.mean(data)

    all_p = Parallel(n_jobs=n_jobs)(delayed(_permute_sign)(data,
                     random_state=seeds[i]) for i in range(n_permute))
    stats['p'] = _calc_pvalue(all_p,stats['mean'],tail)
    return stats

def two_sample_permutation(data1, data2, n_permute=5000,
                           tail=2, n_jobs=-1, random_state=None):
    ''' Independent sample permutation test.

        Args:
            data1: Pandas DataFrame or Series or numpy array
            data2: Pandas DataFrame or Series or numpy array
            n_permute: (int) number of permutations
            tail: (int) either 1 for one-tail or 2 for two-tailed test (default: 2)
            n_jobs: (int) The number of CPUs to use to do the computation.
                    -1 means all CPUs.
        Returns:
            stats: (dict) dictionary of permutation results ['mean','p']

    '''

    random_state = check_random_state(random_state)
    seeds = random_state.randint(MAX_INT, size=n_permute)

    stats = dict()
    stats['mean'] = np.mean(data1)-np.mean(data2)
    data = pd.DataFrame(data={'Values':data1, 'Group':np.ones(len(data1))})
    data = data.append(pd.DataFrame(data={
                                        'Values':data2,
                                        'Group':np.zeros(len(data2))}))
    all_p = Parallel(n_jobs=n_jobs)(delayed(_permute_group)(data,
                     random_state=seeds[i]) for i in range(n_permute))

    stats['p'] = _calc_pvalue(all_p,stats['mean'],tail)
    return stats

def correlation_permutation(data1, data2, n_permute=5000, metric='spearman',
                            tail=2, n_jobs=-1, random_state=None):
    ''' Permute correlation.

        Args:
            data1: Pandas DataFrame or Series or numpy array
            data2: Pandas DataFrame or Series or numpy array
            n_permute: (int) number of permutations
            metric: (str) type of association metric ['spearman','pearson',
                    'kendall']
            tail: (int) either 1 for one-tail or 2 for two-tailed test (default: 2)
            n_jobs: (int) The number of CPUs to use to do the computation.
                    -1 means all CPUs.

        Returns:
            stats: (dict) dictionary of permutation results ['correlation','p']

    '''

    random_state = check_random_state(random_state)

    stats = dict()
    data1 = np.array(data1)
    data2 = np.array(data2)

    if metric == 'spearman':
        stats['correlation'] = spearmanr(data1, data2)[0]
    elif metric == 'pearson':
        stats['correlation'] = pearsonr(data1, data2)[0]
    elif metric == 'kendall':
        stats['correlation'] = kendalltau(data1, data2)[0]
    else:
        raise ValueError('metric must be "spearman" or "pearson" or "kendall"')

    if metric is 'spearman':
        all_p = Parallel(n_jobs=n_jobs)(delayed(spearmanr)(
                        random_state.permutation(data1), data2)
                        for i in range(n_permute))
    elif metric is 'pearson':
        all_p = Parallel(n_jobs=n_jobs)(delayed(pearsonr)(
                        random_state.permutation(data1), data2)
                        for i in range(n_permute))
    elif metric is 'kendall':
        all_p = Parallel(n_jobs=n_jobs)(delayed(kendalltau)(
                        random_state.permutation(data1), data2)
                        for i in range(n_permute))
    all_p = [x[0] for x in all_p]

    stats['p'] = _calc_pvalue(all_p,stats['correlation'],tail)
    return stats

def make_cosine_basis(nsamples, sampling_freq, filter_length, unit_scale=True, drop=0):
    """ Create a series of cosine basis functions for a discrete cosine
        transform. Based off of implementation in spm_filter and spm_dctmtx
        because scipy dct can only apply transforms but not return the basis
        functions. Like SPM, does not add constant (i.e. intercept), but does
        retain first basis (i.e. sigmoidal/linear drift)

    Args:
        nsamples (int): number of observations (e.g. TRs)
        sampling_freq (float): sampling frequency in hertz (i.e. 1 / TR)
        filter_length (int): length of filter in seconds
        unit_scale (true): assure that the basis functions are on the normalized range [-1, 1]; default True
        drop (int): index of which early/slow bases to drop if any; default is
            to drop constant (i.e. intercept) like SPM. Unlike SPM, retains
            first basis (i.e. linear/sigmoidal). Will cumulatively drop bases
            up to and inclusive of index provided (e.g. 2, drops bases 1 and 2)

    Returns:
        out (ndarray): nsamples x number of basis sets numpy array

    """

    #Figure out number of basis functions to create
    order = int(np.fix(2 * (nsamples * sampling_freq)/filter_length + 1))

    n = np.arange(nsamples)

    #Initialize basis function matrix
    C = np.zeros((len(n),order))

    #Add constant
    C[:,0] = np.ones((1,len(n)))/np.sqrt(nsamples)

    #Insert higher order cosine basis functions
    for i in range(1,order):
        C[:,i] = np.sqrt(2./nsamples) * np.cos(np.pi*(2*n+1) * i/(2*nsamples))

    # Drop intercept ala SPM
    C = C[:,1:]

    if C.size == 0:
        raise ValueError('Basis function creation failed! nsamples is too small for requested filter_length.')

    if unit_scale:
        C *= 1. / C[0,0]

    C = C[:, drop:]

    return C

def transform_pairwise(X, y):
    '''Transforms data into pairs with balanced labels for ranking
    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.
    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1

    Reference: "Large Margin Rank Boundaries for Ordinal Regression",
    R. Herbrich, T. Graepel, K. Obermayer.
    Authors: Fabian Pedregosa <fabian@fseoane.net>
             Alexandre Gramfort <alexandre.gramfort@inria.fr>
    Args:
        X : array, shape (n_samples, n_features)
            The data
        y : array, shape (n_samples,) or (n_samples, 2)
            Target labels. If it's a 2D array, the second column represents
            the grouping of samples, i.e., samples with different groups will
            not be considered.

    Returns:
        X_trans : array, shape (k, n_feaures)
            Data as pairs, where k = n_samples * (n_samples-1)) / 2 if grouping
            values were not passed. If grouping variables exist, then returns
            values computed for each group.
        y_trans : array, shape (k,)
            Output class labels, where classes have values {-1, +1}
            If y was shape (n_samples, 2), then returns (k, 2) with groups on
            the second dimension.
    '''

    X_new, y_new, y_group = [], [], []
    y_ndim = y.ndim
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        y_group.append(y[i,1])
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    if y_ndim == 1:
        return np.asarray(X_new), np.asarray(y_new).ravel()
    elif y_ndim == 2:
        return np.asarray(X_new), np.vstack((np.asarray(y_new),np.asarray(y_group))).T

def _robust_estimator(vals,X,robust_estimator='hc0',nlags=1):
    """
    Computes robust sandwich estimators for standard errors used in OLS computation. Types include:
    'hc0': Huber (1980) sandwich estimator to return robust standard error estimates.
    'hc3': MacKinnon and White (1985) HC3 sandwich estimator. Provides more robustness in smaller samples than HC0 Long & Ervin (2000)
    'hac': Newey-West (1987) estimator for robustness to heteroscedasticity as well as serial auto-correlation at given lags.

    Refs: https://www.wikiwand.com/en/Heteroscedasticity-consistent_standard_errors
    https://github.com/statsmodels/statsmodels/blob/master/statsmodels/regression/linear_model.py
    https://cran.r-project.org/web/packages/sandwich/vignettes/sandwich.pdf
    https://www.stata.com/manuals13/tsnewey.pdf

    Args:
        vals (np.ndarray): 1d array of residuals
        X (np.ndarray): design matrix used in OLS, e.g. Brain_Data().X
        robust_estimator (str): estimator type, 'hc0' (default), 'hc3', or 'hac'
        nlags (int): number of lags, only used with 'hac' estimator, default is 1

    Returns:
        stderr (np.ndarray): 1d array of standard errors with length == X.shape[1]
    """

    assert robust_estimator in ['hc0','hc3','hac'], "robust_estimator must be one of hc0, hc3 or hac"

    # Make a sandwich!
    # First we need bread
    bread = np.linalg.pinv(np.dot(X.T,X))

    # Then we need meat
    if robust_estimator == 'hc0':
        V = np.diag(vals**2)
        meat = np.dot(np.dot(X.T,V),X)

    elif robust_estimator == 'hc3':
        V = np.diag(vals**2)/(1-np.diag(np.dot(X,np.dot(bread,X.T))))**2
        meat = np.dot(np.dot(X.T,V),X)

    elif robust_estimator == 'hac':
        weights = 1 - np.arange(nlags+1.)/(nlags+1.)

        #First compute lag 0
        V = np.diag(vals**2)
        meat = weights[0] * np.dot(np.dot(X.T,V),X)

        #Now loop over additional lags
        for l in range(1, nlags+1):

            V = np.diag(vals[l:] * vals[:-l])
            meat_1 = np.dot(np.dot(X[l:].T,V),X[:-l])
            meat_2 = np.dot(np.dot(X[:-l].T,V),X[l:])

            meat += weights[l] * (meat_1 + meat_2)

    # Then we make a sandwich
    vcv = np.dot(np.dot(bread,meat),bread)

    return np.sqrt(np.diag(vcv))


def summarize_bootstrap(data, save_weights=False):
    """ Calculate summary of bootstrap samples

    Args:
        sample: Brain_Data instance of samples
        save_weights: (bool) save bootstrap weights

    Returns:
        output: dictionary of Brain_Data summary images

    """

    # Calculate SE of bootstraps
    wstd = data.std()
    wmean = data.mean()
    wz = deepcopy(wmean)
    wz.data = wmean.data / wstd.data
    wp = deepcopy(wmean)
    wp.data = 2*(1-norm.cdf(np.abs(wz.data)))
    # Create outputs
    output = {'Z':wz, 'p': wp, 'mean':wmean}
    if save_weights:
        output['samples'] = data
    return output

def _arma_func(X,Y,idx=None,**kwargs):

    """
    Fit an ARMA(p,q) model. If Y is a matrix and not a vector, expects an idx argument that refers to columns of Y. Used by regress().
    """
    method = kwargs.pop('method','css-mle')
    order = kwargs.pop('order',(1,1))

    maxiter = kwargs.pop('maxiter',50)
    disp = kwargs.pop('disp',-1)
    start_ar_lags = kwargs.pop('start_ar_lags',order[0]+1)
    transparams = kwargs.pop('transparams',False)
    trend = kwargs.pop('trend','nc')

    if len(Y.shape) == 2:
        model = sm.tsa.arima_model.ARMA(endog=Y[:,idx],exog=X.values,order=order)
    else:
        model = sm.tsa.arima_model.ARMA(endog=Y,exog=X.values,order=order)
    try:
        res = model.fit(trend=trend,method=method,transparams=transparams,
                maxiter=maxiter,disp=disp,start_ar_lags=start_ar_lags,**kwargs)
    except:
        res = model.fit(trend=trend,method=method,transparams=transparams,
                maxiter=maxiter,disp=disp,start_ar_lags=start_ar_lags,start_params=np.repeat(1.,X.shape[1]+2))

    return (res.params[:-2], res.tvalues[:-2],res.pvalues[:-2],res.df_resid, res.resid)

def regress(X,Y,mode='ols',**kwargs):
    """ This is a flexible function to run several types of regression models provided X and Y numpy arrays. Y can be a 1d numpy array or 2d numpy array. In the latter case, results will be output with shape 1 x Y.shape[1], in other words fitting a separate regression model to each column of Y.

    Does NOT add an intercept automatically to the X matrix before fitting like some other software packages. This is left up to the user.

    This function can compute regression in 3 ways:
    1) Standard OLS
    2) OLS with robust sandwich estimators for standard errors. 3 robust types of estimators exist:
        1) 'hc0' - classic huber-white estimator robust to heteroscedasticity (default)
        2) 'hc3' - a variant on huber-white estimator slightly more conservative when sample sizes are small
        3) 'hac' - an estimator robust to both heteroscedasticity and auto-correlation; auto-correlation lag can be controlled with the 'nlags' keyword argument; default is 1
    3) ARMA (auto-regressive moving-average) model (experimental). This model is fit through statsmodels.tsa.arima_model.ARMA, so more information about options can be found there. Any settings can be passed in as kwargs. By default fits a (1,1) model with starting lags of 2. This mode is **computationally intensive** and can take quite a while if Y has many columns.  If Y is a 2d array joblib.Parallel is used for faster fitting by parallelizing fits across columns of Y. Parallelization can be controlled by passing in kwargs. Defaults to multi-threading using 10 separate threads, as threads don't require large arrays to be duplicated in memory. Defaults are also set to enable memory-mapping for very large arrays if backend='multiprocessing' to prevent crashes and hangs. Various levels of progress can be monitored using the 'disp' (statsmodels) and 'verbose' (joblib) keyword arguments with integer values > 0.

    Args:
        X (ndarray): design matrix; assumes intercept is included
        Y (ndarray): dependent variable array; if 2d, a model is fit to each column of Y separately
        mode (str): kind of model to fit; must be one of 'ols' (default), 'robust', or 'arma'
        robust_estimator (str,optional): kind of robust estimator to use if mode = 'robust'; default 'hc0'
        nlags (int,optional): auto-correlation lag correction if mode = 'robust' and robust_estimator = 'hac'; default 1
        order (tuple,optional): auto-regressive and moving-average orders for mode = 'arma'; default (1,1)
        kwargs (dict): additional keyword arguments to statsmodels.tsa.arima_model.ARMA and joblib.Parallel

    Returns:
        b: coefficients
        t: t-statistics (coef/sterr)
        p : p-values
        df: degrees of freedom
        res: residuals

    Examples:
        Standard OLS

        >>> results = regress(X,Y,mode='ols')

        Robust OLS with heteroscedasticity (hc0) robust standard errors

        >>> results = regress(X,Y,mode='robust')

        Robust OLS with heteroscedasticty and auto-correlation (with lag 2) robust standard errors

        >>> results = regress(X,Y,mode='robust',robust_estimator='hac',nlags=2)

        Auto-regressive mode with auto-regressive and moving-average lags = 1

        >>> results = regress(X,Y,mode='arma',order=(1,1))

        Auto-regressive model with auto-regressive lag = 2, moving-average lag = 3, and multi-processing instead of multi-threading using 8 cores (this can use a lot of memory if input arrays are very large!).

        >>> results = regress(X,Y,mode='arma',order=(2,3),backend='multiprocessing',n_jobs=8)

    """

    if not isinstance(mode,six.string_types):
        raise ValueError('mode must be a string')

    assert mode in ['ols','robust','arma'], "Mode must be one of 'ols','robust' or 'arma'"

    # Compute standard errors based on regression mode
    if mode == 'ols' or mode == 'robust':

        b = np.dot(np.linalg.pinv(X),Y)
        res = Y - np.dot(X,b)

        # Vanilla OLS
        if mode == 'ols':
            sigma = np.std(res,axis=0,ddof=X.shape[1])
            stderr = np.sqrt(np.diag(np.linalg.pinv(np.dot(X.T,X))))[:,np.newaxis] * sigma[np.newaxis,:]

        # OLS with robust sandwich estimator based standard-errors
        elif mode == 'robust':
            robust_estimator = kwargs.pop('robust_estimator','hc0')
            nlags = kwargs.pop('nlags',1)
            axis_func = [_robust_estimator,0,res,X,robust_estimator,nlags]
            stderr = np.apply_along_axis(*axis_func)

        t = b / stderr
        df = np.array([X.shape[0]-X.shape[1]] * t.shape[1])
        p = 2*(1-t_dist.cdf(np.abs(t),df))

    # ARMA regression
    elif mode == 'arma':
        n_jobs = kwargs.pop('n_jobs',-1)
        backend = kwargs.pop('backend','threading')
        max_nbytes = kwargs.pop('max_nbytes',1e8)
        verbose = kwargs.pop('verbose',0)

        # Parallelize if Y vector contains more than 1 column
        if len(Y.shape) == 2:
            if backend == 'threading' and n_jobs == -1:
                n_jobs = 10
            par_for = Parallel(n_jobs=n_jobs,verbose=verbose,backend=backend,max_nbytes=max_nbytes)
            out_arma = par_for(delayed(_arma_func)(X,Y,idx=i,**kwargs) for i in range(Y.shape[-1]))

            b = np.column_stack([elem[0] for elem in out_arma])
            t = np.column_stack([elem[1] for elem in out_arma])
            p = np.column_stack([elem[2] for elem in out_arma])
            df = np.array([elem[3] for elem in out_arma])
            res = np.column_stack([elem[4] for elem in out_arma])

        else:
            b,t,p,df,res = _arma_func(X,Y,**kwargs)

    return b, t, p, df, res


def align(data, method='deterministic_srm', n_features=None, axis=0,
            *args, **kwargs):
    ''' Align subject data into a common response model.

        Can be used to hyperalign source data to target data using
        Hyperalignemnt from Dartmouth (i.e., procrustes transformation; see
        nltools.stats.procrustes) or Shared Response Model from Princeton (see
        nltools.external.srm). (see nltools.data.Brain_Data.align for aligning
        a single Brain object to another). Common Model is shared response
        model or centered target data.Transformed data can be back projected to
        original data using Tranformation matrix.

        Examples:
            Hyperalign using procrustes transform:
                out = align(data, method='procrustes')

            Align using shared response model:
                out = align(data, method='probabilistic_srm', n_features=None)

            Project aligned data into original data:
                original_data = [np.dot(t.data,tm.T) for t,tm in zip(out['transformed'], out['transformation_matrix'])]

        Args:
            data: (list) A list of Brain_Data objects
            method: (str) alignment method to use
                ['probabilistic_srm','deterministic_srm','procrustes']
            n_features: (int) number of features to align to common space.
                If None then will select number of voxels
            axis: (int) axis to align on

        Returns:
            out: (dict) a dictionary containing a list of transformed subject
                matrices, a list of transformation matrices, and the shared
                response matrix

    '''

    from nltools.data import Brain_Data

    assert isinstance(data, list), 'Make sure you are inputting data is a list.'
    assert all([type(x) for x in data]), 'Make sure all objects in the list are the same type.'
    assert method in ['probabilistic_srm','deterministic_srm','procrustes'], "Method must be ['probabilistic_srm','deterministic_srm','procrustes']"

    data = deepcopy(data)

    if isinstance(data[0], Brain_Data):
        data_type = 'Brain_Data'
        data_out = [x.copy() for x in data]
        data = [x.data.T for x in data]
    elif isinstance(data[0], np.ndarray):
        data_type = 'numpy'
    else:
        raise ValueError('Type %s is not implemented yet.' % type(data[0]))

    # Align over time or voxels
    if axis == 1:
        data = [x.T for x in data]
    elif axis != 0:
        raise ValueError('axis must be 0 or 1.')

    out = dict()
    if method in ['deterministic_srm', 'probabilistic_srm']:
        if n_features is None:
            n_features = int(data[0].shape[0])
        if method == 'deterministic_srm':
            srm = DetSRM(features=n_features, *args, **kwargs)
        elif method == 'probabilistic_srm':
            srm = SRM(features=n_features, *args, **kwargs)
        srm.fit(data)
        out['transformed'] = [x for x in srm.transform(data)]
        out['common_model'] = srm.s_
        out['transformation_matrix'] = srm.w_

    elif method == 'procrustes':
        if n_features != None:
            raise NotImplementedError('Currently must use all voxels.'
                                      'Eventually will add a PCA reduction,'
                                      'must do this manually for now.')

        ##STEP 0: STANDARDIZE SIZE AND SHAPE##
        sizes_0 = [x.shape[0] for x in data]
        sizes_1 = [x.shape[1] for x in data]

        #find the smallest number of rows
        R = min(sizes_0)
        C = max(sizes_1)

        m = [np.empty((R, C), dtype=np.ndarray)] * len(data)

        #Pad rows with different sizes with zeros
        for i, x in enumerate(data):
            y = x[0:R, :]
            missing = C - y.shape[1]
            add = np.zeros((y.shape[0], missing))
            y = np.append(y, add, axis=1)
            m[i] = y

        ##STEP 1: CREATE INITIAL AVERAGE TEMPLATE##
        for i, x in enumerate(m):
            if i == 0:
                # use first data as template
                template = np.copy(x.T)
            else:
                _, trans, _, _, _ = procrustes(template/i, x.T)
                template += trans
        template /= len(m)

        ##STEP 2: CREATE NEW COMMON TEMPLATE##
        #align each subj to the template from STEP 1
        #and create a new common template based on avg
        common = np.zeros(template.shape)
        for i, x in enumerate(m):
            _, trans, _, _, _ = procrustes(template, x.T)
            common += trans
        common /= len(m)

        #STEP 3 (below): ALIGN TO NEW TEMPLATE
        aligned = []; transformation_matrix = []; disparity = []; scale = []
        for i, x in enumerate(m):
            _, transformed, d, t, s = procrustes(common, x.T)
            aligned.append(transformed.T)
            transformation_matrix.append(t)
            disparity.append(d)
            scale.append(s)
        out['transformed'] = aligned
        out['common_model'] = common.T
        out['transformation_matrix'] = transformation_matrix
        out['disparity'] = disparity
        out['scale'] = scale

    if axis == 1:
        out['transformed'] = [x.T for x in out['transformed']]
        out['common_model'] = out['common_model'].T

    if data_type == 'Brain_Data':
        for i, x in enumerate(out['transformed']):
            data_out[i].data = x.T
        out['transformed'] = data_out
        common = data_out[0].copy()
        common.data = out['common_model'].T
        out['common_model'] = common
    return out

def procrustes(data1, data2):
    '''Procrustes analysis, a similarity test for two data sets.

    Each input matrix is a set of points or vectors (the rows of the matrix).
    The dimension of the space is the number of columns of each matrix. Given
    two identically sized matrices, procrustes standardizes both such that:
    - :math:`tr(AA^{T}) = 1`.
    - Both sets of points are centered around the origin.
    Procrustes ([1]_, [2]_) then applies the optimal transform to the second
    matrix (including scaling/dilation, rotations, and reflections) to minimize
    :math:`M^{2}=\sum(data1-data2)^{2}`, or the sum of the squares of the
    pointwise differences between the two input datasets.
    This function was not designed to handle datasets with different numbers of
    datapoints (rows).  If two data sets have different dimensionality
    (different number of columns), this function will add columns of zeros to
    the smaller of the two.

    Args:
        data1 : array_like
            Matrix, n rows represent points in k (columns) space `data1` is the
            reference data, after it is standardised, the data from `data2`
            will be transformed to fit the pattern in `data1` (must have >1
            unique points).
        data2 : array_like
            n rows of data in k space to be fit to `data1`.  Must be the  same
            shape ``(numrows, numcols)`` as data1 (must have >1 unique points).

    Returns:
        mtx1 : array_like
            A standardized version of `data1`.
        mtx2 : array_like
            The orientation of `data2` that best fits `data1`. Centered, but not
            necessarily :math:`tr(AA^{T}) = 1`.
        disparity : float
            :math:`M^{2}` as defined above.
        R : (N, N) ndarray
            The matrix solution of the orthogonal Procrustes problem.
            Minimizes the Frobenius norm of dot(data1, R) - data2, subject to
            dot(R.T, R) == I.
        scale : float
            Sum of the singular values of ``dot(data1.T, data2)``.


    '''

    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape[0] != mtx2.shape[0]:
        raise ValueError("Input matrices must have same number of rows.")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")
    if mtx1.shape[1] != mtx2.shape[1]:
        # Pad with zeros
        if mtx1.shape[1] > mtx2.shape[1]:
            mtx2 = np.append(mtx2, np.zeros((mtx1.shape[0], mtx1.shape[1] - mtx2.shape[1])), axis=1)
        else:
            mtx1 = np.append(mtx1, np.zeros((mtx1.shape[0], mtx2.shape[1] - mtx1.shape[1])), axis=1)

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return mtx1, mtx2, disparity, R, s
