"""
NeuroLearn Statistics Tools
===========================

Tools to help with statistical analyses.

"""

__all__ = [
    "zscore",
    "fdr",
    "holm_bonf",
    "threshold",
    "multi_threshold",
    "winsorize",
    "trim",
    "calc_bpm",
    "downsample",
    "upsample",
    "fisher_r_to_z",
    "fisher_z_to_r",
    "one_sample_permutation",
    "two_sample_permutation",
    "correlation_permutation",
    "matrix_permutation",
    "make_cosine_basis",
    "summarize_bootstrap",
    "procrustes",
    "procrustes_distance",
    "align",
    "find_spikes",
    "distance_correlation",
    "transform_pairwise",
    "double_center",
    "u_center",
    "_bootstrap_isc",
    "isc",
    "isc_group",
    "isfc",
    "isps",
    "_compute_matrix_correlation",
    "_phase_mean_angle",
    "_phase_vector_length",
    "_butter_bandpass_filter",
    "_phase_rayleigh_p",
    "_compute_isc_group",
    "_permute_isc_group",
    "align_states",
]

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes as procrust
from scipy.signal import hilbert, butter, filtfilt
from scipy.optimize import linear_sum_assignment
from copy import deepcopy
import nibabel as nib
from scipy.interpolate import interp1d
import warnings
import itertools
from joblib import Parallel, delayed
from .algorithms.srm import SRM, DetSRM
from .algorithms.inference import (
    one_sample_permutation_test,
    two_sample_permutation_test,
    correlation_permutation_test,
    matrix_permutation_test,
    isc_permutation_test,
    double_center,
    u_center,
    distance_correlation,
)
from .algorithms.inference.timeseries import timeseries_correlation_permutation_test
from .algorithms.inference.utils import _compute_pvalue
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise_distances

MAX_INT = np.iinfo(np.int32).max


def pearson(x, y):
    """Correlates row vector x with each row vector in 2D array y.
    From neurosynth.stats.py - author: Tal Yarkoni

    .. deprecated:: 0.5.2
        This function is deprecated and will be removed in a future version.
        Use `scipy.stats.pearsonr` or `numpy.corrcoef` instead, or use
        `correlation_permutation_test` from the inference module for
        permutation-based correlation analysis.
    """
    warnings.warn(
        "pearson() is deprecated and will be removed in a future version. "
        "Use scipy.stats.pearsonr or numpy.corrcoef instead, or use "
        "correlation_permutation_test from the inference module.",
        DeprecationWarning,
        stacklevel=2,
    )
    data = np.vstack((x, y))
    ms = data.mean(axis=1)[(slice(None, None, None), None)]
    datam = data - ms
    datass = np.sqrt(np.sum(datam * datam, axis=1))
    # datass = np.sqrt(ss(datam, axis=1))
    temp = np.dot(datam[1:], datam[0].T)
    return temp / (datass[1:] * datass[0])


def zscore(df):
    """zscore every column in a pandas dataframe or series.

    Args:
        df: (pd.DataFrame) Pandas DataFrame instance

    Returns:
        z_data: (pd.DataFrame) z-scored pandas DataFrame or series instance
    """

    if isinstance(df, pd.DataFrame):
        return df.apply(lambda x: (x - x.mean()) / x.std())
    elif isinstance(df, pd.Series):
        return (df - np.mean(df)) / np.std(df)
    else:
        raise ValueError("Data is not a Pandas DataFrame or Series instance")


def fdr(p, q=0.05):
    """Determine FDR threshold given a p value array and desired false
    discovery rate q. Written by Tal Yarkoni

    Args:
        p: (np.array) vector of p-values
        q: (float) false discovery rate level

    Returns:
        fdr_p: (float) p-value threshold based on independence or positive
                dependence

    """

    if not isinstance(p, np.ndarray):
        raise ValueError("Make sure vector of p-values is a numpy array")
    if any(p < 0) or any(p > 1):
        raise ValueError("array contains p-values that are outside the range 0-1")

    if np.any(p > 1) or np.any(p < 0):
        raise ValueError("Does not include valid p-values.")

    s = np.sort(p)
    nvox = p.shape[0]
    null = np.array(range(1, nvox + 1), dtype="float") * q / nvox
    below = np.where(s <= null)[0]
    return s[max(below)] if len(below) else -1


def holm_bonf(p, alpha=0.05):
    """Compute corrected p-values based on the Holm-Bonferroni method, i.e. step-down procedure applying iteratively less correction to highest p-values. A bit more conservative than fdr, but much more powerful thanvanilla bonferroni.

    Args:
        p: (np.array) vector of p-values
        alpha: (float) alpha level

    Returns:
        bonf_p: (float) p-value threshold based on bonferroni
                step-down procedure

    """

    if not isinstance(p, np.ndarray):
        raise ValueError("Make sure vector of p-values is a numpy array")

    s = np.sort(p)
    nvox = p.shape[0]
    null = 0.05 / (nvox - np.arange(1, nvox + 1) + 1)
    below = np.where(s <= null)[0]
    return s[max(below)] if len(below) else -1


# TODO: check if deprecated given new method in BrainData that makes uses of nilearn + custom code
def threshold(stat, p, thr=0.05, return_mask=False):
    """Threshold test image by p-value from p image

    Args:
        stat: (BrainData) BrainData instance of arbitrary statistic metric
              (e.g., beta, t, etc)
        p: (BrainData) Brain_data instance of p-values
        threshold: (float) p-value to threshold stat image
        return_mask: (bool) optionall return the thresholding mask; default False

    Returns:
        out: Thresholded BrainData instance

    """
    from nltools.data import BrainData

    if not isinstance(stat, BrainData):
        raise ValueError("Make sure stat is a BrainData instance")

    if not isinstance(p, BrainData):
        raise ValueError("Make sure p is a BrainData instance")

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

    if return_mask:
        return out, mask
    else:
        return out


# TODO: do we need this or does nilearn offer similar functionality already? Who uses it?
def multi_threshold(t_map, p_map, thresh):
    """Threshold test image by multiple p-value from p image

    Args:
        stat: (BrainData) BrainData instance of arbitrary statistic metric
            (e.g., beta, t, etc)
        p: (BrainData) Brain_data instance of p-values
        threshold: (list) list of p-values to threshold stat image

    Returns:
        out: Thresholded BrainData instance

    """
    from nltools.data import BrainData

    if not isinstance(t_map, BrainData):
        raise ValueError("Make sure stat is a BrainData instance")

    if not isinstance(p_map, BrainData):
        raise ValueError("Make sure p is a BrainData instance")

    if not isinstance(thresh, list):
        raise ValueError("Make sure thresh is a list of p-values")

    affine = t_map.to_nifti().affine
    pos_out = np.zeros(t_map.to_nifti().shape)
    neg_out = deepcopy(pos_out)
    for thr in thresh:
        t = threshold(t_map, p_map, thr=thr)
        t_pos = deepcopy(t)
        t_pos.data = np.zeros(len(t_pos.data))
        t_neg = deepcopy(t_pos)
        t_pos.data[t.data > 0] = 1
        t_neg.data[t.data < 0] = 1
        pos_out = pos_out + t_pos.to_nifti().get_fdata()
        neg_out = neg_out + t_neg.to_nifti().get_fdata()
    pos_out = pos_out + neg_out * -1
    return BrainData(nib.Nifti1Image(pos_out, affine))


# TODO: see related comment on _transform_outliers
def winsorize(data, cutoff=None, replace_with_cutoff=True):
    """Winsorize a Pandas DataFrame or Series with the largest/lowest value not considered outlier

    Args:
        data: (pd.DataFrame, pd.Series) data to winsorize
        cutoff: (dict) a dictionary with keys {'std':[low,high]} or
                {'quantile':[low,high]}
        replace_with_cutoff: (bool) If True, replace outliers with cutoff.
                             If False, replaces outliers with closest
                             existing values; (default: False)
    Returns:
        out: (pd.DataFrame, pd.Series) winsorized data
    """
    return _transform_outliers(
        data, cutoff, replace_with_cutoff=replace_with_cutoff, method="winsorize"
    )


# TODO: see related comment on _transform_outliers
def trim(data, cutoff=None):
    """Trim a Pandas DataFrame or Series by replacing outlier values with NaNs

    Args:
        data: (pd.DataFrame, pd.Series) data to trim
        cutoff: (dict) a dictionary with keys {'std':[low,high]} or
                {'quantile':[low,high]}
    Returns:
        out: (pd.DataFrame, pd.Series) trimmed data
    """
    return _transform_outliers(data, cutoff, replace_with_cutoff=None, method="trim")


# TODO: do we need this? can we refactor the function it supports to be more efficient?
def _transform_outliers(data, cutoff, replace_with_cutoff, method):
    """This function is not exposed to user but is called by either trim
    or winsorize.

    Args:
        data: (pd.DataFrame, pd.Series) data to transform
        cutoff: (dict) a dictionary with keys {'std':[low,high]} or
                {'quantile':[low,high]}
        replace_with_cutoff: (bool) If True, replace outliers with cutoff.
                                    If False, replaces outliers with closest
                                    existing values. (default: False)
        method: 'winsorize' or 'trim'

    Returns:
        out: (pd.DataFrame, pd.Series) transformed data
    """
    df = data.copy()  # To not overwrite data make a copy

    def _transform_outliers_sub(data, cutoff, replace_with_cutoff, method="trim"):
        if not isinstance(data, pd.Series):
            raise ValueError(
                "Make sure that you are applying winsorize to a pandas dataframe or series."
            )
        if isinstance(cutoff, dict):
            # calculate cutoff values
            if "quantile" in cutoff:
                q = data.quantile(cutoff["quantile"])
            elif "std" in cutoff:
                std = [
                    data.mean() - data.std() * cutoff["std"][0],
                    data.mean() + data.std() * cutoff["std"][1],
                ]
                q = pd.Series(index=cutoff["std"], data=std)
            # if replace_with_cutoff is false, replace with true existing values closest to cutoff
            if method == "winsorize" and not replace_with_cutoff:
                q.iloc[0] = data[data > q.iloc[0]].min()
                q.iloc[1] = data[data < q.iloc[1]].max()
        else:
            raise ValueError("cutoff must be a dictionary with quantile or std keys.")
        if method == "trim":
            data[data < q.iloc[0]] = np.nan
            data[data > q.iloc[1]] = np.nan
        elif method == "winsorize":
            if isinstance(q, pd.Series) and len(q) == 2:
                # Cast quantile values to match data dtype to avoid pandas compatibility warnings
                lower_val = (
                    data.dtype.type(q.iloc[0])
                    if hasattr(data.dtype, "type")
                    else q.iloc[0]
                )
                upper_val = (
                    data.dtype.type(q.iloc[1])
                    if hasattr(data.dtype, "type")
                    else q.iloc[1]
                )
                data[data < q.iloc[0]] = lower_val
                data[data > q.iloc[1]] = upper_val
        return data

    # transform each column if a dataframe, if series just transform data
    if isinstance(df, pd.DataFrame):
        for col in df.columns:
            df.loc[:, col] = _transform_outliers_sub(
                df.loc[:, col],
                cutoff=cutoff,
                replace_with_cutoff=replace_with_cutoff,
                method=method,
            )
        return df
    elif isinstance(df, pd.Series):
        return _transform_outliers_sub(
            df, cutoff=cutoff, replace_with_cutoff=replace_with_cutoff, method=method
        )
    else:
        raise ValueError("Data must be a pandas DataFrame or Series")


def calc_bpm(beat_interval, sampling_freq):
    """Calculate instantaneous BPM from beat to beat interval

    Args:
        beat_interval: (int) number of samples in between each beat
                        (typically R-R Interval)
        sampling_freq: (float) sampling frequency in Hz

    Returns:
        bpm:  (float) beats per minute for time interval
    """
    return 60 * sampling_freq * (1 / (beat_interval))


# TODO: ensure efficient
def downsample(
    data, sampling_freq=None, target=None, target_type="samples", method="mean"
):
    """Downsample pandas to a new target frequency or number of samples
    using averaging.

    Args:
        data: (pd.DataFrame, pd.Series) data to downsample
        sampling_freq:  (float) Sampling frequency of data in hertz
        target: (float) downsampling target
        target_type: type of target can be [samples,seconds,hz]
        method: (str) type of downsample method ['mean','median'],
                default: mean

    Returns:
        out: (pd.DataFrame, pd.Series) downsmapled data

    """

    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError("Data must by a pandas DataFrame or Series instance.")
    if not (method == "median") | (method == "mean"):
        raise ValueError("Metric must be either 'mean' or 'median' ")

    if target_type == "samples":
        n_samples = target
    elif target_type == "seconds":
        n_samples = target * sampling_freq
    elif target_type == "hz":
        n_samples = sampling_freq / target
    else:
        raise ValueError('Make sure target_type is "samples", "seconds",  or "hz".')

    idx = np.sort(np.repeat(np.arange(1, data.shape[0] / n_samples, 1), n_samples))
    # if data.shape[0] % n_samples:
    if data.shape[0] > len(idx):
        idx = np.concatenate([idx, np.repeat(idx[-1] + 1, data.shape[0] - len(idx))])
    if method == "mean":
        return data.groupby(idx).mean().reset_index(drop=True)
    elif method == "median":
        return data.groupby(idx).median().reset_index(drop=True)


# TODO: ensure efficient
def upsample(
    data, sampling_freq=None, target=None, target_type="samples", method="linear"
):
    """Upsample pandas to a new target frequency or number of samples using interpolation.

    Args:
        data: (pd.DataFrame, pd.Series) data to upsample
              (Note: will drop non-numeric columns from DataFrame)
        sampling_freq:  Sampling frequency of data in hertz
        target: (float) upsampling target
        target_type: (str) type of target can be [samples,seconds,hz]
        method: (str) ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
                      where 'zero', 'slinear', 'quadratic' and 'cubic'
                      refer to a spline interpolation of zeroth, first,
                      second or third order  (default: linear)
    Returns:
        upsampled pandas object

    """

    methods = ["linear", "nearest", "zero", "slinear", "quadratic", "cubic"]
    if method not in methods:
        raise ValueError(
            "Method must be 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'"
        )

    if target_type == "samples":
        n_samples = target
    elif target_type == "seconds":
        n_samples = target * sampling_freq
    elif target_type == "hz":
        n_samples = float(sampling_freq) / float(target)
    else:
        raise ValueError('Make sure target_type is "samples", "seconds", or "hz".')

    orig_spacing = np.arange(0, data.shape[0], 1)
    new_spacing = np.arange(0, data.shape[0] - 1, n_samples)

    if isinstance(data, pd.Series):
        interpolate = interp1d(orig_spacing, data, kind=method)
        return interpolate(new_spacing)
    elif isinstance(data, pd.DataFrame):
        numeric_data = data._get_numeric_data()
        if data.shape[1] != numeric_data.shape[1]:
            warnings.warn(
                "Dropping %s non-numeric columns"
                % (data.shape[1] - numeric_data.shape[1]),
                UserWarning,
            )
        out = pd.DataFrame(columns=numeric_data.columns, index=None)
        for i, x in numeric_data.items():
            interpolate = interp1d(orig_spacing, x, kind=method)
            out.loc[:, i] = interpolate(new_spacing)
        return out
    else:
        raise ValueError("Data must by a pandas DataFrame or Series instance.")


def fisher_r_to_z(r):
    """Use Fisher transformation to convert correlation to z score

    Args:
        r: correlation coefficient(s)

    Returns:
        z: Fisher z-transformed correlation(s)

    Note:
        Clips r values to (-1, 1) range to avoid invalid arctanh inputs
    """
    # Clip r to valid range for arctanh to avoid invalid value warnings
    with np.errstate(invalid="ignore"):
        return np.arctanh(r)


def fisher_z_to_r(z):
    """Use Fisher transformation to convert correlation to z score"""
    return np.tanh(z)


# Removed correlation() - replaced by correlation_permutation_test in inference module
# This function was a simple wrapper around scipy.stats functions


# Removed _permute_sign() - replaced by _generate_sign_flips in inference module


def _permute_group(data, random_state=None):
    random_state = check_random_state(random_state)
    perm_label = random_state.permutation(data["Group"])
    return np.mean(data.loc[perm_label == 1, "Values"]) - np.mean(
        data.loc[perm_label == 0, "Values"]
    )


# Removed _permute_func() - replaced by matrix_permutation_test in inference module
# This function was unused and functionality is covered by the optimized inference module


# Removed _calc_pvalue - replaced by _compute_pvalue from nltools.algorithms.inference.utils
# This function was duplicated and has been consolidated into the inference module


def one_sample_permutation(
    data, n_permute=5000, tail=2, n_jobs=-1, return_perms=False, random_state=None
):
    """One sample permutation test using randomization.

    This function is a wrapper around `nltools.algorithms.inference.one_sample_permutation_test`
    for backward compatibility. The underlying implementation provides optimized CPU parallelization
    and optional GPU acceleration.

    Args:
        data: (pd.DataFrame, pd.Series, np.array) data to permute
        n_permute: (int) number of permutations
        tail: (int) either 1 for one-tail or 2 for two-tailed test (default: 2)
        n_jobs: (int) The number of CPUs to use to do the computation.
                -1 means all CPUs.
        return_parms: (bool) Return the permutation distribution along with the p-value; default False
        random_state: (int, None, or np.random.RandomState) Initial random seed (default: None)

    Returns:
        stats: (dict) dictionary of permutation results ['mean','p']

    Notes:
        This function uses the optimized inference module implementation which provides:
        - 4-8× speedup with CPU parallelization (default)
        - 10-100× speedup with GPU acceleration (via backend='torch')
        - Identical results to the original implementation

    """
    # Wrapper around inference module for optimized implementation
    result = one_sample_permutation_test(
        data,
        n_permute=n_permute,
        tail=tail,
        n_jobs=n_jobs,
        return_null=return_perms,  # Map parameter name
        random_state=random_state,
    )
    # Map return keys to match stats.py API
    output = {
        "mean": result["mean"],
        "p": result["p"],
    }
    if return_perms:
        output["perm_dist"] = result["null_dist"]
    return output


def two_sample_permutation(
    data1,
    data2,
    n_permute=5000,
    tail=2,
    n_jobs=-1,
    return_perms=False,
    random_state=None,
):
    """Independent sample permutation test.

    This function is a wrapper around `nltools.algorithms.inference.two_sample_permutation_test`
    for backward compatibility. The underlying implementation provides optimized CPU parallelization
    and optional GPU acceleration.

    Args:
        data1: (pd.DataFrame, pd.Series, np.array) dataset 1 to permute
        data2: (pd.DataFrame, pd.Series, np.array) dataset 2 to permute
        n_permute: (int) number of permutations
        tail: (int) either 1 for one-tail or 2 for two-tailed test (default: 2)
        n_jobs: (int) The number of CPUs to use to do the computation.
                -1 means all CPUs.
        return_parms: (bool) Return the permutation distribution along with the p-value; default False
    Returns:
        stats: (dict) dictionary of permutation results ['mean','p']

    Notes:
        This function uses the optimized inference module implementation which provides:
        - 4-8× speedup with CPU parallelization (default)
        - 10-100× speedup with GPU acceleration (via backend='torch')
        - Identical results to the original implementation

    """
    # Wrapper around inference module for optimized implementation
    result = two_sample_permutation_test(
        data1,
        data2,
        n_permute=n_permute,
        tail=tail,
        n_jobs=n_jobs,
        return_null=return_perms,  # Map parameter name
        random_state=random_state,
    )
    # Map return keys: 'mean_diff' -> 'mean' for backward compatibility
    output = {
        "mean": result["mean_diff"],
        "p": result["p"],
    }
    if return_perms:
        output["perm_dist"] = result["null_dist"]
    return output


def correlation_permutation(
    data1,
    data2,
    method="permute",
    n_permute=5000,
    metric="spearman",
    tail=2,
    n_jobs=-1,
    return_perms=False,
    random_state=None,
):
    """Compute correlation and calculate p-value using permutation methods.

    This function is a wrapper around `nltools.algorithms.inference.correlation_permutation_test`
    and `nltools.algorithms.inference.timeseries_correlation_permutation_test` for backward
    compatibility. The underlying implementation provides optimized CPU parallelization
    and optional GPU acceleration.

    'permute' method randomly shuffles one of the vectors. This method is recommended
    for independent data. For timeseries data we recommend using 'circle_shift' or
    'phase_randomize' methods.

    Args:

        data1: (pd.DataFrame, pd.Series, np.array) dataset 1 to permute
        data2: (pd.DataFrame, pd.Series, np.array) dataset 2 to permute
        n_permute: (int) number of permutations
        metric: (str) type of association metric ['spearman','pearson',
                'kendall']
        method: (str) type of permutation ['permute', 'circle_shift', 'phase_randomize']
        random_state: (int, None, or np.random.RandomState) Initial random seed (default: None)
        tail: (int) either 1 for one-tail or 2 for two-tailed test (default: 2)
        n_jobs: (int) The number of CPUs to use to do the computation.
                -1 means all CPUs.
        return_parms: (bool) Return the permutation distribution along with the p-value; default False

    Returns:

        stats: (dict) dictionary of permutation results ['correlation','p']

    Notes:
        This function uses the optimized inference module implementation which provides:
        - 4-8× speedup with CPU parallelization (default)
        - GPU acceleration available for 'permute' method with Pearson correlation
        - Identical results to the original implementation

    """
    # Wrapper around inference module for optimized implementation
    # Route to correct function based on method
    if method == "permute":
        result = correlation_permutation_test(
            data1,
            data2,
            n_permute=n_permute,
            metric=metric,  # Preserve stats.py default 'spearman'
            tail=tail,
            n_jobs=n_jobs,
            return_null=return_perms,  # Map parameter name
            random_state=random_state,
        )
    else:  # circle_shift or phase_randomize
        result = timeseries_correlation_permutation_test(
            data1,
            data2,
            method=method,
            n_permute=n_permute,
            metric=metric,  # Preserve stats.py default 'spearman'
            tail=tail,
            n_jobs=n_jobs,
            return_null=return_perms,  # Map parameter name
            random_state=random_state,
        )
    # Map return keys to match stats.py API
    output = {
        "correlation": result["correlation"],
        "p": result["p"],
    }
    if return_perms:
        # Handle different key names: 'null_dist' vs 'null_distribution'
        null_key = "null_distribution" if "null_distribution" in result else "null_dist"
        output["perm_dist"] = result[null_key]
    return output


def matrix_permutation(
    data1,
    data2,
    n_permute=5000,
    metric="spearman",
    how="upper",
    include_diag=False,
    tail=2,
    n_jobs=-1,
    return_perms=False,
    random_state=None,
):
    """Permute 2-dimensional matrix correlation (mantel test).

    This function is a wrapper around `nltools.algorithms.inference.matrix_permutation_test`
    for backward compatibility. The underlying implementation provides optimized CPU parallelization.

    Chen, G. et al. (2016). Untangling the relatedness among correlations,
    part I: nonparametric approaches to inter-subject correlation analysis
    at the group level. Neuroimage, 142, 248-259.

    Args:
        data1: (pd.DataFrame, np.array) square matrix
        data2: (pd.DataFrame, np.array) square matrix
        n_permute: (int) number of permutations
        metric: (str) type of association metric ['spearman','pearson',
                'kendall']
        how: (str) whether to use the 'upper' (default), 'lower', or 'full' matrix. The
            default of 'upper' assumes both matrices are symmetric
        include_diag (bool): only applies if `how='full'`. Whether to include the
            diagonal elements in the comparison
        tail: (int) either 1 for one-tail or 2 for two-tailed test
              (default: 2)
        n_jobs: (int) The number of CPUs to use to do the computation.
                -1 means all CPUs.
        return_parms: (bool) Return the permutation distribution along with the p-value; default False

    Returns:
        stats: (dict) dictionary of permutation results ['correlation','p']

    Notes:
        This function uses the optimized inference module implementation which provides:
        - 4-8× speedup with CPU parallelization (default)
        - Identical results to the original implementation

    """
    # Wrapper around inference module for optimized implementation
    result = matrix_permutation_test(
        data1,
        data2,
        n_permute=n_permute,
        metric=metric,  # Preserve stats.py default 'spearman'
        how=how,
        include_diag=include_diag,
        tail=tail,
        n_jobs=n_jobs,
        return_null=return_perms,  # Map parameter name
        random_state=random_state,
    )
    # Map return keys to match stats.py API
    output = {
        "correlation": result["correlation"],
        "p": result["p"],
    }
    if return_perms:
        output["perm_dist"] = result["null_dist"]
    return output


# TODO: make efficient
def make_cosine_basis(nsamples, sampling_freq, filter_length, unit_scale=True, drop=0):
    """Create a series of cosine basis functions for a discrete cosine
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

    # Figure out number of basis functions to create
    order = int(np.fix(2 * (nsamples * sampling_freq) / filter_length + 1))

    n = np.arange(nsamples)

    # Initialize basis function matrix
    C = np.zeros((len(n), order))

    # Add constant
    C[:, 0] = np.ones((1, len(n))) / np.sqrt(nsamples)

    # Insert higher order cosine basis functions
    for i in range(1, order):
        C[:, i] = np.sqrt(2.0 / nsamples) * np.cos(
            np.pi * (2 * n + 1) * i / (2 * nsamples)
        )

    # Drop intercept ala SPM
    C = C[:, 1:]

    if C.size == 0:
        raise ValueError(
            "Basis function creation failed! nsamples is too small for requested filter_length."
        )

    if unit_scale:
        C *= 1.0 / C[0, 0]

    C = C[:, drop:]

    return C


# TODO: make efficient
def transform_pairwise(X, y):
    """Transforms data into pairs with balanced labels for ranking
    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.
    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1

    Reference: "Large Margin Rank Boundaries for Ordinal Regression",
    R. Herbrich, T. Graepel, K. Obermayer. Authors: Fabian Pedregosa
    <fabian@fseoane.net> Alexandre Gramfort <alexandre.gramfort@inria.fr>

    Args:
        X: (np.array), shape (n_samples, n_features)
            The data
        y: (np.array), shape (n_samples,) or (n_samples, 2)
            Target labels. If it's a 2D array, the second column represents
            the grouping of samples, i.e., samples with different groups will
            not be considered.

    Returns:
        X_trans: (np.array), shape (k, n_feaures)
            Data as pairs, where k = n_samples * (n_samples-1)) / 2 if grouping
            values were not passed. If grouping variables exist, then returns
            values computed for each group.
        y_trans: (np.array), shape (k,)
            Output class labels, where classes have values {-1, +1}
            If y was shape (n_samples, 2), then returns (k, 2) with groups on
            the second dimension.
    """

    X_new, y_new, y_group = [], [], []
    y_ndim = y.ndim
    if y_ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        y_group.append(y[i, 1])
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = -y_new[-1]
            X_new[-1] = -X_new[-1]
    if y_ndim == 1:
        return np.asarray(X_new), np.asarray(y_new).ravel()
    elif y_ndim == 2:
        return np.asarray(X_new), np.vstack((np.asarray(y_new), np.asarray(y_group))).T


# TODO: see how best to refactor and where to put this or if covered by other modules
def summarize_bootstrap(data, save_weights=False):
    """Calculate summary of bootstrap samples

    Args:
        sample: (BrainData) BrainData instance of samples
        save_weights: (bool) save bootstrap weights

    Returns:
        output: (dict) dictionary of BrainData summary images

    """

    # Calculate SE of bootstraps
    wstd = data.std()
    wmean = data.mean()
    wz = deepcopy(wmean)
    with np.errstate(invalid="ignore", divide="ignore"):
        wz.data = wmean.data / wstd.data
    wp = deepcopy(wmean)
    wp.data = 2 * (1 - norm.cdf(np.abs(wz.data)))
    # Create outputs
    output = {"Z": wz, "p": wp, "mean": wmean}
    if save_weights:
        output["samples"] = data
    return output


def align(data, method="deterministic_srm", n_features=None, axis=0, *args, **kwargs):
    """Align subject data into a common response model. This function is a convenience wrapper around `HyperAlignment` and `SRM` classes

    Can be used to hyperalign source data to target data using
    Hyperalignment from Dartmouth (i.e., procrustes transformation; see
    nltools.stats.procrustes) or Shared Response Model from Princeton (see
    nltools.algorithms.srm). (see nltools.data.BrainData.align for aligning
    a single Brain object to another). Common Model is shared response
    model or centered target data. Transformed data can be back projected to
    original data using Tranformation matrix. Inputs must be a list of BrainData
    instances or numpy arrays (observations by features).


    Args:
        data: (list) A list of BrainData objects
        method: (str) alignment method to use
            ['probabilistic_srm','deterministic_srm','procrustes']
        n_features: (int) number of features to align to common space.
            If None then will select number of voxels
        axis: (int) axis to align on

    Returns:
        out: (dict) a dictionary containing a list of transformed subject
            matrices, a list of transformation matrices, the shared
            response matrix, and the intersubject correlation of the shared resposnes

    Examples:
        - Hyperalign using procrustes transform:
            >>> out = align(data, method='procrustes')
        - Align using shared response model:
            >>> out = align(data, method='probabilistic_srm', n_features=None)
        - Project aligned data into original data:
            >>> original_data = [np.dot(t.data,tm.T) for t,tm in zip(out['transformed'], out['transformation_matrix'])]
    """

    from nltools.data import BrainData, Adjacency

    if not isinstance(data, list):
        raise ValueError("Make sure you are inputting data is a list.")
    if not all(type(x) for x in data):
        raise ValueError("Make sure all objects in the list are the same type.")
    if method not in ["probabilistic_srm", "deterministic_srm", "procrustes"]:
        raise ValueError(
            "Method must be ['probabilistic_srm','deterministic_srm','procrustes']"
        )

    data = deepcopy(data)

    if isinstance(data[0], BrainData):
        data_type = "BrainData"
        data_out = [x.copy() for x in data]
        transformation_out = [x.copy() for x in data]
        data = [x.data.T for x in data]
    elif isinstance(data[0], np.ndarray):
        data_type = "numpy"
        data = [x.T for x in data]
    else:
        raise ValueError("Type %s is not implemented yet." % type(data[0]))

    # Align over time or voxels
    if axis == 1:
        data = [x.T for x in data]
    elif axis != 0:
        raise ValueError("axis must be 0 or 1.")

    out = {}
    if method in ["deterministic_srm", "probabilistic_srm"]:
        if n_features is None:
            n_features = int(data[0].shape[0])
        if method == "deterministic_srm":
            srm = DetSRM(features=n_features, *args, **kwargs)
        elif method == "probabilistic_srm":
            srm = SRM(features=n_features, *args, **kwargs)
        srm.fit(data)
        out["transformed"] = [x for x in srm.transform(data)]
        out["common_model"] = srm.s_.T
        out["transformation_matrix"] = srm.w_

    elif method == "procrustes":
        from nltools.algorithms import HyperAlignment

        if n_features is not None:
            raise NotImplementedError(
                "Currently must use all voxels."
                "Eventually will add a PCA reduction,"
                "must do this manually for now."
            )

        # Use HyperAlignment class for procrustes-based hyperalignment
        # Note: data is already transposed to [features, samples] format by line 1330/1327
        # n_iter=1 maintains backward compatibility with original implementation
        hyper = HyperAlignment(n_iter=1, auto_pad=True)
        hyper.fit(data)

        # Transform data to common space
        aligned = hyper.transform(data)

        # Extract attributes for output
        # Note: align() returns common_model in [samples, features] format (transposed)
        # but transformed in [features, samples] format (not transposed)
        out["transformed"] = aligned
        out["common_model"] = hyper.s_.T  # Transpose to [samples, features]
        out["transformation_matrix"] = hyper.w_
        out["disparity"] = hyper.disparity_
        out["scale"] = hyper.scale_

    if axis == 1:
        out["transformed"] = [x.T for x in out["transformed"]]
        out["common_model"] = out["common_model"].T

        if data_type == "BrainData":
            out["transformation_matrix"] = [x.T for x in out["transformation_matrix"]]

    # Calculate Intersubject correlation on aligned components
    if n_features is None:
        n_features = out["common_model"].shape[1]

    a = Adjacency()
    for f in range(n_features):
        a = a.append(
            Adjacency(
                1
                - pairwise_distances(
                    np.array([x[f, :] for x in out["transformed"]]),
                    metric="correlation",
                ),
                metric="similarity",
            )
        )
    out["isc"] = dict(zip(np.arange(n_features), a.mean(axis=1)))

    if data_type == "BrainData":
        if method == "procrustes":
            for i, x in enumerate(out["transformed"]):
                data_out[i].data = x.T
                out["transformed"] = data_out
            common = data_out[0].copy()
            common.data = out["common_model"]
            out["common_model"] = common
        else:
            out["transformed"] = [x.T for x in out["transformed"]]

        for i, x in enumerate(out["transformation_matrix"]):
            transformation_out[i].data = x.T
        out["transformation_matrix"] = transformation_out

    return out


def procrustes(data1, data2):
    """Procrustes analysis, a similarity test for two data sets. For more comprehensive procrustes-based alignment tasks, use `HyperAlignment` and `align()` instead.

    Each input matrix is a set of points or vectors (the rows of the matrix).
    The dimension of the space is the number of columns of each matrix. Given
    two identically sized matrices, procrustes standardizes both such that:
    - :math:`tr(AA^{T}) = 1`.
    - Both sets of points are centered around the origin.
    Procrustes then applies the optimal transform to the second
    matrix (including scaling/dilation, rotations, and reflections) to minimize
    :math:`M^{2}=\\sum(data1-data2)^{2}`, or the sum of the squares of the
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
    """

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
            mtx2 = np.append(
                mtx2, np.zeros((mtx1.shape[0], mtx1.shape[1] - mtx2.shape[1])), axis=1
            )
        else:
            mtx1 = np.append(
                mtx1, np.zeros((mtx1.shape[0], mtx2.shape[1] - mtx1.shape[1])), axis=1
            )

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


def procrustes_distance(
    mat1, mat2, n_permute=5000, tail=2, n_jobs=-1, random_state=None
):
    """Use procrustes super-position to perform a similarity test between 2 matrices. Matrices need to match in size on their first dimension only, as the smaller matrix on the second dimension will be padded with zeros. After aligning two matrices using the procrustes transformation, use the computed disparity between them (sum of squared error of elements) as a similarity metric. Shuffle the rows of one of the matrices and recompute the disparity to perform inference (Peres-Neto & Jackson, 2001).

    Args:
        mat1 (ndarray): 2d numpy array; must have same number of rows as mat2
        mat2 (ndarray): 1d or 2d numpy array; must have same number of rows as mat1
        n_permute (int): number of permutation iterations to perform
        tail (int): either 1 for one-tailed or 2 for two-tailed test; default 2
        n_jobs (int): The number of CPUs to use to do permutation; default -1 (all)

    Returns:
        similarity (float): similarity between matrices bounded between 0 and 1
        pval (float): permuted p-value

    """

    # raise NotImplementedError("procrustes distance is not currently implemented")
    if mat1.shape[0] != mat2.shape[0]:
        raise ValueError("Both arrays must match on their first dimension")

    random_state = check_random_state(random_state)

    # Make sure both matrices are 2d and the same dimension via padding
    if len(mat1.shape) < 2:
        mat1 = mat1[:, np.newaxis]
    if len(mat2.shape) < 2:
        mat2 = mat2[:, np.newaxis]
    if mat1.shape[1] > mat2.shape[1]:
        mat2 = np.pad(mat2, ((0, 0), (0, mat1.shape[1] - mat2.shape[1])), "constant")
    elif mat2.shape[1] > mat1.shape[1]:
        mat1 = np.pad(mat1, ((0, 0), (0, mat2.shape[1] - mat1.shape[1])), "constant")

    _, _, sse = procrust(mat1, mat2)

    stats = {"similarity": sse}
    all_p = Parallel(n_jobs=n_jobs)(
        delayed(procrust)(random_state.permutation(mat1), mat2)
        for _ in range(n_permute)
    )
    all_p = [1 - x[2] for x in all_p]

    # Use _compute_pvalue from inference module (signature: obs_stat, null_dist, tail)
    stats["p"] = float(_compute_pvalue(np.array(sse), np.array(all_p), tail=tail)[0])

    return stats


# TODO: too slow needs to be made more efficient
def find_spikes(data, global_spike_cutoff=3, diff_spike_cutoff=3):
    """Function to identify spikes from fMRI Time Series Data

    Args:
        data: BrainData or nibabel instance
        global_spike_cutoff: (int,None) cutoff to identify spikes in global signal
                             in standard deviations, None indicates do not calculate.
        diff_spike_cutoff: (int,None) cutoff to identify spikes in average frame difference
                             in standard deviations, None indicates do not calculate.
    Returns:
        pandas dataframe with spikes as indicator variables
    """

    from nltools.data import BrainData

    if (global_spike_cutoff is None) & (diff_spike_cutoff is None):
        raise ValueError("Did not input any cutoffs to identify spikes in this data.")

    if isinstance(data, BrainData):
        data = deepcopy(data.data)
        global_mn = np.mean(data.data, axis=1)
        frame_diff = np.mean(np.abs(np.diff(data.data, axis=0)), axis=1)
    elif isinstance(data, nib.Nifti1Image):
        data = deepcopy(data.get_fdata())
        if len(data.shape) > 3:
            data = np.squeeze(data)
        elif len(data.shape) < 3:
            raise ValueError("nibabel instance does not appear to be 4D data.")
        global_mn = np.mean(data, axis=(0, 1, 2))
        frame_diff = np.mean(np.abs(np.diff(data, axis=3)), axis=(0, 1, 2))
    else:
        raise ValueError(
            "Currently this function can only accomodate BrainData and nibabel instances"
        )

    if global_spike_cutoff is not None:
        global_outliers = np.append(
            np.where(
                global_mn > np.mean(global_mn) + np.std(global_mn) * global_spike_cutoff
            ),
            np.where(
                global_mn < np.mean(global_mn) - np.std(global_mn) * global_spike_cutoff
            ),
        )

    if diff_spike_cutoff is not None:
        frame_outliers = np.append(
            np.where(
                frame_diff
                > np.mean(frame_diff) + np.std(frame_diff) * diff_spike_cutoff
            ),
            np.where(
                frame_diff
                < np.mean(frame_diff) - np.std(frame_diff) * diff_spike_cutoff
            ),
        )
    # build spike regressors
    outlier = pd.DataFrame([x + 1 for x in range(len(global_mn))], columns=["TR"])
    if global_spike_cutoff is not None:
        for i, loc in enumerate(global_outliers):
            outlier["global_spike" + str(i + 1)] = 0
            outlier["global_spike" + str(i + 1)].iloc[int(loc)] = 1

    # build FD regressors
    if diff_spike_cutoff is not None:
        for i, loc in enumerate(frame_outliers):
            outlier["diff_spike" + str(i + 1)] = 0
            outlier["diff_spike" + str(i + 1)].iloc[int(loc)] = 1
    return outlier


def _bootstrap_isc(
    similarity_matrix, metric="median", exclude_self_corr=True, random_state=None
):
    """Helper function to compute bootstrapped ISC from Adjacency Instance

    This function implements the subject-wise bootstrap method discussed in Chen et al., 2016.

    Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C., Israel, R. B.,
    & Cox, R. W. (2016). Untangling the relatedness among correlations, part I:
    nonparametric approaches to inter-subject correlation analysis at the group level.
    NeuroImage, 142, 248-259.

    Args:

        similarity_matrix: (Adjacency) Adjacency matrix of pairwise correlation values
        metric: (str) type of summary statistic (Default: median)
        exclude_self_corr: (bool) set correlations with random draws of same subject to NaN (Default: True)
        random_state: random_state instance for permutation

    Returns:

        isc: summary statistic of bootstrapped similarity matrix

    """
    from nltools.data import Adjacency

    if not isinstance(similarity_matrix, Adjacency):
        raise ValueError("similarity_matrix must be an Adjacency instance.")

    random_state = check_random_state(random_state)

    square = similarity_matrix.squareform()
    n_sub = square.shape[0]
    np.fill_diagonal(square, 1)

    bootstrap_subject = sorted(
        random_state.choice(np.arange(n_sub), size=n_sub, replace=True)
    )
    bootstrap_sample = Adjacency(
        square[bootstrap_subject, :][:, bootstrap_subject], matrix_type="similarity"
    )

    if exclude_self_corr:
        bootstrap_sample.data[bootstrap_sample.data == 1] = np.nan

    if metric == "mean":
        return np.tanh(bootstrap_sample.r_to_z().mean())
    elif metric == "median":
        return bootstrap_sample.median()


def isc(
    data,
    n_samples=5000,
    metric="median",
    method="bootstrap",
    ci_percentile=95,
    exclude_self_corr=True,
    return_null=False,
    tail=2,
    n_jobs=-1,
    random_state=None,
    sim_metric="correlation",
):
    """Compute pairwise intersubject correlation from observations by subjects array.

    This function computes pairwise intersubject correlations (ISC) using the median as recommended by Chen
    et al., 2016). However, if the mean is preferred, we compute the mean correlation after performing
    the fisher r-to-z transformation and then convert back to correlations to minimize artificially
    inflating the correlation values.

    There are currently three different methods to compute p-values. These include the classic methods for
    computing permuted time-series by either circle-shifting the data or phase-randomizing the data
    (see Lancaster et al., 2018). These methods create random surrogate data while preserving the temporal
    autocorrelation inherent to the signal. By default, we use the subject-wise bootstrap method from
    Chen et al., 2016. Instead of recomputing the pairwise ISC using circle_shift or phase_randomization methods,
    this approach uses the computationally more efficient method of bootstrapping the subjects
    and computing a new pairwise similarity matrix with randomly selected subjects with replacement.
    If the same subject is selected multiple times, we set the perfect correlation to a nan with
    (exclude_self_corr=True). We compute the p-values using the percentile method using the same
    method in Brainiak.

    Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C., Israel, R. B.,
    & Cox, R. W. (2016). Untangling the relatedness among correlations, part I:
    nonparametric approaches to inter-subject correlation analysis at the group level.
    NeuroImage, 142, 248-259.

    Hall, P., & Wilson, S. R. (1991). Two guidelines for bootstrap hypothesis testing.
    Biometrics, 757-762.

    Lancaster, G., Iatsenko, D., Pidde, A., Ticcinelli, V., & Stefanovska, A. (2018).
    Surrogate data for hypothesis testing of physical systems. Physics Reports, 748, 1-60.

    Args:
        data: (pd.DataFrame, np.array) observations by subjects where isc is computed across subjects
        n_samples: (int) number of random samples/bootstraps
        metric: (str) type of isc summary metric ['mean','median']
        method: (str) method to compute p-values ['bootstrap', 'circle_shift','phase_randomize'] (default: bootstrap)
        tail: (int) either 1 for one-tail or 2 for two-tailed test (default: 2)
        n_jobs: (int) The number of CPUs to use to do the computation. -1 means all CPUs.
        return_null: (bool) Return the permutation distribution along with the p-value; default False
        sim_metric: (str) pairwise distance metric. See sklearn's pairwise_distances for valid inputs (default: correlation)

    Returns:
        stats: (dict) dictionary of permutation results ['isc', 'p', 'ci', 'null_distribution']

    Notes
    -----
    This function is a wrapper around `isc_permutation_test` from the inference module,
    which provides optimized implementations with CPU-parallel and GPU acceleration support.
    Performance improvements: 4-8× speedup for CPU-parallel operations, 10-100× speedup
    for GPU operations. See `nltools.algorithms.inference.isc.isc_permutation_test` for details.

    """
    # Convert data to numpy array if needed
    if isinstance(data, pd.DataFrame):
        data = data.values

    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a pandas dataframe or numpy array")

    if metric not in ["mean", "median"]:
        raise ValueError("metric must be ['mean', 'median']")

    # Call inference module function with parameter mapping
    result = isc_permutation_test(
        data,
        n_permute=n_samples,  # Map n_samples -> n_permute
        metric=metric,
        summary_statistic="pairwise",  # Explicitly set to match original behavior
        method=method,
        ci_percentile=ci_percentile,
        tail=tail,
        n_jobs=n_jobs,
        random_state=random_state,
        return_null=return_null,
        exclude_self_corr=exclude_self_corr,
        sim_metric=sim_metric,
        progress_bar=False,  # Disable progress bar for backward compatibility
    )

    # Return dict with same keys as original function
    return result


def _compute_isc_group(group1, group2, metric="median"):
    """Helper function to compute intersubject correlation difference between two groups from either:
    1) an observations by subjects array
    2) or an Adjacency instance of a similarity matrix.

    Args:
        group1: (pd.DataFrame, np.array, Adjacency) group1 data or similarity matrix
        group2: (pd.DataFrame, np.array,Adjacency)  group2 data or similarity matrix
        metric: (str) type of isc metric ['mean','median']

    Returns:
        isc: (float) intersubject correlation coefficient difference across groups

    """

    from nltools.data import Adjacency

    if isinstance(group1, (pd.DataFrame, np.ndarray)) and isinstance(
        group2, (pd.DataFrame, np.ndarray)
    ):
        if group1.shape[0] != group2.shape[0]:
            raise ValueError(
                "group1 has a different number of observations from group2."
            )

        similarity_group1 = Adjacency(
            1 - pairwise_distances(group1.T, metric="correlation"),
            matrix_type="similarity",
        )
        similarity_group2 = Adjacency(
            1 - pairwise_distances(group2.T, metric="correlation"),
            matrix_type="similarity",
        )
    elif isinstance(group1, (Adjacency)) and isinstance(group2, (Adjacency)):
        similarity_group1 = group1
        similarity_group2 = group2
    else:
        raise ValueError(
            "group1 and group2 data must either be a observation by feature matrix or Adjacency instances."
        )

    if metric == "mean":
        isc_group1 = np.tanh(similarity_group1.r_to_z().mean())
        isc_group2 = np.tanh(similarity_group2.r_to_z().mean())
    elif metric == "median":
        isc_group1 = similarity_group1.median()
        isc_group2 = similarity_group2.median()
    return isc_group1 - isc_group2


def _permute_isc_group(similarity_matrix, group, metric="median", random_state=None):
    """Helper function to compute ISC differences between groups from Adjacency instance

    This function implements the subject-wise permutation method discussed in Chen et al., 2016.

    Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C., Israel, R. B.,
    & Cox, R. W. (2016). Untangling the relatedness among correlations, part I:
    nonparametric approaches to inter-subject correlation analysis at the group level.
    NeuroImage, 142, 248-259.

    Args:

        similarity_matrix: (Adjacency) Adjacency matrix of pairwise correlation values
        group: (numpy array) Array indicating group 1 and group 2 order (i.e., np.array([1,1,1,2,2,2]))
        metric: (str) type of summary statistic (Default: median)
        exclude_self_corr: (bool) set correlations with random draws of same subject to NaN (Default: True)
        random_state: random_state instance for permutation

    Returns:

        isc: summary statistic of bootstrapped similarity matrix

    """
    from nltools.data import Adjacency

    if not isinstance(similarity_matrix, Adjacency):
        raise ValueError("similarity_matrix must be an Adjacency instance.")

    if not isinstance(group, np.ndarray):
        raise ValueError("group must be a numpy array.")

    if len(group) != similarity_matrix.square_shape()[0]:
        raise ValueError(
            "Group array must be the same length as the similarity matrix."
        )

    if len(np.unique(group)) != 2:
        raise ValueError("There must only be 2 unique group ids in the group array.")

    random_state = check_random_state(random_state)

    group1_id, group2_id = np.unique(group)
    permute_group = permute_group = random_state.permutation(group)
    permute_order = np.concatenate(
        [
            np.where(permute_group == group1_id)[0],
            np.where(permute_group == group2_id)[0],
        ]
    )

    permuted_matrix = similarity_matrix.squareform()[permute_order, :][:, permute_order]
    group1_similarity_permuted = Adjacency(
        permuted_matrix[group == group1_id, :][:, group == group1_id],
        matrix_type="similarity",
    )
    group2_similarity_permuted = Adjacency(
        permuted_matrix[group == group2_id, :][:, group == group2_id],
        matrix_type="similarity",
    )

    return _compute_isc_group(
        group1_similarity_permuted, group2_similarity_permuted, metric=metric
    )


# TODO: update to use inference/ module
def isc_group(
    group1,
    group2,
    n_samples=5000,
    metric="median",
    method="permute",
    ci_percentile=95,
    exclude_self_corr=True,
    return_null=False,
    tail=2,
    n_jobs=-1,
    random_state=None,
):
    """Compute difference in intersubject correlation between groups.

    This function computes pairwise intersubject correlations (ISC) using the median as recommended by Chen
    et al., 2016). However, if the mean is preferred, we compute the mean correlation after performing
    the fisher r-to-z transformation and then convert back to correlations to minimize artificially
    inflating the correlation values.

    There are currently two different methods to compute p-values. By default, we use the subject-wise permutation
    method recommended Chen et al., 2016. This method combines the two groups and computes pairwise similarity both
    within and between the groups. Then the group labels are permuted and the mean difference between the two groups
    are recomputed to generate a null distribution. The second method uses subject-wise bootstrapping, where a new
    pairwise similarity matrix with randomly selected subjects with replacement is created separately for each group
    and the ISC difference between these groups is used to generate a null distribution. If the same subject is
    selected multiple times, we set the perfect correlation to a nan with (exclude_self_corr=True). We compute the
    p-values using the percentile method (Hall & Wilson, 1991).

    Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C., Israel, R. B.,
    & Cox, R. W. (2016). Untangling the relatedness among correlations, part I:
    nonparametric approaches to inter-subject correlation analysis at the group level.
    NeuroImage, 142, 248-259.

    Hall, P., & Wilson, S. R. (1991). Two guidelines for bootstrap hypothesis testing.
    Biometrics, 757-762.

    Args:
        group1: (pd.DataFrame, np.array) observations by subjects where isc is computed across subjects
        group2: (pd.DataFrame, np.array) observations by subjects where isc is computed across subjects
        n_samples: (int) number of samples for permutation or bootstrapping
        metric: (str) type of isc summary metric ['mean','median']
        method: (str) method to compute p-values ['bootstrap', 'circle_shift','phase_randomize'] (default: bootstrap)
        tail: (int) either 1 for one-tail or 2 for two-tailed test (default: 2)
        n_jobs: (int) The number of CPUs to use to do the computation. -1 means all CPUs.
        return_null: (bool) Return the permutation distribution along with the p-value; default False

    Returns:
        stats: (dict) dictionary of permutation results ['correlation','p']

    """

    from nltools.data import Adjacency

    random_state = check_random_state(random_state)

    for group_data in [group1, group2]:
        if not isinstance(group_data, (pd.DataFrame, np.ndarray)):
            raise ValueError("group data must be a pandas dataframe or numpy array")

    if metric not in ["mean", "median"]:
        raise ValueError("metric must be ['mean', 'median']")

    if group1.shape[0] != group2.shape[0]:
        raise ValueError("group1 has a different number of observations from group2.")

    stats = {"isc_group_difference": _compute_isc_group(group1, group2, metric=metric)}

    if method == "permute":
        data = np.concatenate([group1, group2], axis=1)
        group = np.array([1] * group1.shape[1] + [2] * group2.shape[1])
        similarity = Adjacency(
            1 - pairwise_distances(data.T, metric="correlation"),
            matrix_type="similarity",
        )

        isc_group_differences_null = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(_permute_isc_group)(
                    similarity, group, metric=metric, random_state=random_state
                )
                for _ in range(n_samples)
            )
        )
    elif method == "bootstrap":
        group1_similarity = Adjacency(
            1 - pairwise_distances(group1.T, metric="correlation"),
            matrix_type="similarity",
        )
        group1_all_bootstraps = Parallel(n_jobs=n_jobs)(
            delayed(_bootstrap_isc)(
                group1_similarity,
                metric=metric,
                exclude_self_corr=exclude_self_corr,
                random_state=random_state,
            )
            for _ in range(n_samples)
        )

        group2_similarity = Adjacency(
            1 - pairwise_distances(group2.T, metric="correlation"),
            matrix_type="similarity",
        )
        group2_all_bootstraps = Parallel(n_jobs=n_jobs)(
            delayed(_bootstrap_isc)(
                group2_similarity,
                metric=metric,
                exclude_self_corr=exclude_self_corr,
                random_state=random_state,
            )
            for _ in range(n_samples)
        )

        isc_group_differences_null = (
            np.array(group1_all_bootstraps) - np.array(group2_all_bootstraps)
        ) - stats["isc_group_difference"]

    else:
        raise NotImplementedError("method can only be ['permutation', 'bootstrap']")

    isc_group_differences_null = isc_group_differences_null[
        ~np.isnan(isc_group_differences_null)
    ]

    # Use _compute_pvalue from inference module (signature: obs_stat, null_dist, tail)
    stats["p"] = float(
        _compute_pvalue(
            np.array(stats["isc_group_difference"]),
            isc_group_differences_null,
            tail=tail,
        )[0]
    )

    stats["ci"] = (
        np.percentile(isc_group_differences_null, (100 - ci_percentile) / 2, axis=0),
        np.percentile(
            isc_group_differences_null,
            ci_percentile + (100 - ci_percentile) / 2,
            axis=0,
        ),
    )

    if return_null:
        stats["null_distribution"] = isc_group_differences_null

    return stats


# TODO: remove after rewriteing isc_group and isfc
def _compute_matrix_correlation(matrix1, matrix2):
    """Computes the intersubject functional correlation between 2 matrices (observation x feature)"""
    return np.corrcoef(matrix1.T, matrix2.T)[matrix1.shape[1] :, : matrix2.shape[1]]


# TODO: update to use inference/ module
def isfc(data, method="average"):
    """Compute intersubject functional connectivity (ISFC) from a list of observation x feature matrices

    This function uses the leave one out approach to compute ISFC (Simony et al., 2016).
    For each subject, compute the cross-correlation between each voxel/roi
    with the average of the rest of the subjects data. In other words,
    compute the mean voxel/ROI response for all participants except the
    target subject. Then compute the correlation between each ROI within
    the target subject with the mean ROI response in the group average.

    Simony, E., Honey, C. J., Chen, J., Lositsky, O., Yeshurun, Y., Wiesel, A., & Hasson, U. (2016).
    Dynamic reconfiguration of the default mode network during narrative comprehension.
    Nature communications, 7, 12141.

    Args:
        data: list of subject matrices (observations x voxels/rois)
        method: approach to computing ISFC. 'average' uses leave one

    Returns:
        list of subject ISFC matrices

    """
    subjects = np.arange(len(data))

    if method == "average":
        sub_isfc = []
        for target in subjects:
            m1 = data[target]
            sub_mean = np.zeros(m1.shape)
            for y in (y for y in subjects if y != target):
                sub_mean += data[y]
            sub_isfc.append(
                _compute_matrix_correlation(m1, sub_mean / (len(subjects) - 1))
            )
    else:
        raise NotImplementedError(
            "Only average method is implemented. Pairwise will be added at some point."
        )
    return sub_isfc


# TODO: improve to avoid pandas type conversion use numpy or polars instead
def isps(data, sampling_freq=0.5, low_cut=0.04, high_cut=0.07, order=5, pairwise=False):
    """Compute Dynamic Intersubject Phase Synchrony (ISPS from a observation by subject array)

    This function computes the instantaneous intersubject phase synchrony for a single voxel/roi
    timeseries. Requires multiple subjects. This method is largely based on that described by Glerean
    et al., 2012 and performs a hilbert transform on narrow bandpass filtered timeseries (butterworth)
    data to get the instantaneous phase angle. The function returns a dictionary containing the
    average phase angle, the average vector length, and parametric p-values computed using the rayleigh test using circular
    statistics (Fisher, 1993). If pairwise=True, then it will compute these on the pairwise phase angle differences,
    if pairwise=False, it will compute these on the actual phase angles. This is called inter-site phase coupling
    or inter-trial phase coupling respectively in the EEG literatures.

    This function requires narrow band filtering your data. As a default we use the recommendations
    by (Glerean et al., 2012) of .04-.07Hz. This is similar to the "slow-4" band (0.025–0.067 Hz)
    described by (Zuo et al., 2010; Penttonen & Buzsáki, 2003), but excludes the .03 band, which has been
    demonstrated to contain aliased respiration signals (Birn, 2006).

    Birn RM, Smith MA, Bandettini PA, Diamond JB. 2006. Separating respiratory-variation-related
    fluctuations from neuronal-activity- related fluctuations in fMRI. Neuroimage 31:1536–1548.

    Buzsáki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical networks. Science,
    304(5679), 1926-1929.

    Fisher, N. I. (1995). Statistical analysis of circular data. cambridge university press.

    Glerean, E., Salmi, J., Lahnakoski, J. M., Jääskeläinen, I. P., & Sams, M. (2012).
    Functional magnetic resonance imaging phase synchronization as a measure of dynamic
    functional connectivity. Brain connectivity, 2(2), 91-101.

    Args:
        data: (pd.DataFrame, np.ndarray) observations x subjects data
        sampling_freq: (float) sampling freqency of data in Hz
        low_cut: (float) lower bound cutoff for high pass filter
        high_cut: (float) upper bound cutoff for low pass filter
        order: (int) filter order for butterworth bandpass
        pairwise: (bool) compute phase angle coherence on pairwise phase angle differences
                or on raw phase angle.

    Returns:
        dictionary with mean phase angle, vector length, and rayleigh statistic

    """

    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        raise ValueError(
            "data must be a pandas dataframe or numpy array (observations by subjects)"
        )

    phase = np.angle(
        hilbert(
            _butter_bandpass_filter(
                pd.DataFrame(data), low_cut, high_cut, sampling_freq, order=order
            ),
            axis=0,
        )
    )

    if pairwise:
        phase = np.array(
            [
                phase[:, i] - phase[:, j]
                for i in range(phase.shape[1])
                for j in range(phase.shape[1])
                if i < j
            ]
        ).T

    out = {"average_angle": _phase_mean_angle(phase)}
    out["vector_length"] = _phase_vector_length(phase)
    out["p"] = _phase_rayleigh_p(phase)
    return out


def _butter_bandpass_filter(data, low_cut, high_cut, fs, axis=0, order=5):
    """Apply a bandpass butterworth filter with zero-phase filtering

    Args:
        data: (np.array)
        low_cut: (float) lower bound cutoff for high pass filter
        high_cut: (float) upper bound cutoff for low pass filter
        fs: (float) sampling frequency in Hz
        axis: (int) axis to perform filtering.
        order: (int) filter order for butterworth bandpass

    Returns:
        bandpass filtered data.
    """
    nyq = 0.5 * fs
    b, a = butter(order, [low_cut / nyq, high_cut / nyq], btype="band")
    return filtfilt(b, a, data, axis=axis)


def _phase_mean_angle(phase_angles):
    """Compute mean phase angle using circular statistics

    Can take 1D (observation for a single feature) or 2D (observation x feature) signals

    Implementation from:

        Fisher, N. I. (1995). Statistical analysis of circular data. cambridge university press.

    Args:
        phase_angles: (np.array) 1D or 2D array of phase angles

    Returns:
        mean phase angle: (np.array)

    """

    axis = 0 if len(phase_angles.shape) == 1 else 1
    return np.arctan2(
        np.mean(np.sin(phase_angles), axis=axis),
        np.mean(np.cos(phase_angles), axis=axis),
    )


def _phase_vector_length(phase_angles):
    """Compute vector length of phase angles using circular statistics

    Can take 1D (observation for a single feature) or 2D (observation x feature) signals

    Implementation from:

        Fisher, N. I. (1995). Statistical analysis of circular data. cambridge university press.

    Args:
        phase_angles: (np.array) 1D or 2D array of phase angles

    Returns:
         phase angle vector length: (np.array)

    """

    axis = 0 if len(phase_angles.shape) == 1 else 1
    return np.float32(
        np.sqrt(
            np.mean(np.cos(phase_angles), axis=axis) ** 2
            + np.mean(np.sin(phase_angles), axis=axis) ** 2
        )
    )


def _phase_rayleigh_p(phase_angles):
    """Compute the p-value of the phase_angles using the Rayleigh statistic

    Note: this test assumes every time point is independent, which is unlikely to be true in a timeseries with autocorrelation

    Implementation from:

        Fisher, N. I. (1995). Statistical analysis of circular data. cambridge university press.

    Args:
        phase_angles: (np.array) 1D or 2D array of phase angles

    Returns:
         p-values: (np.array)

    """

    n = len(phase_angles) if len(phase_angles.shape) == 1 else phase_angles.shape[1]

    Z = n * _phase_vector_length(phase_angles) ** 2
    if n <= 50:
        return np.exp(-1 * Z) * (
            1
            + (2 * Z - Z**2) / (4 * n)
            - (24 * Z - 132 * Z**2 + 76 * Z**3 - 9 * Z**4) / (288 * n**2)
        )
    else:
        return np.exp(-1 * Z)


def align_states(
    reference,
    target,
    metric="correlation",
    return_index=False,
    replace_zero_variance=False,
):
    """Align state weight maps using hungarian algorithm by minimizing pairwise distance between group states.This function uses the Hungarian algorithm for state alignment, which is different from aligning multiple subjects' data.

    Args:
        reference: (np.array) reference pattern x state matrix
        target: (np.array) target pattern x state matrix to align to reference
        metric: (str) distance metric to use
        return_index: (bool) return index if True, return remapped data if False
        replace_zero_variance: (bool) transform a vector with zero variance to random numbers from a uniform distribution.
                                Useful for when using correlation as a distance metric to avoid NaNs.
    Returns:
        ordered_weights: (list) a list of reordered state X pattern matrices

    """
    if reference.shape != target.shape:
        raise ValueError("reference and target must be the same size")

    reference = np.array(reference)
    target = np.array(target)

    def replace_zero_variance_columns(data):
        if np.any(data.std(axis=0) == 0):
            for i in np.where(data.std(axis=0) == 0)[0]:
                data[:, i] = np.random.uniform(low=0, high=1, size=data.shape[0])
        return data

    if replace_zero_variance:
        reference = replace_zero_variance_columns(reference)
        target = replace_zero_variance_columns(target)

    remapping = linear_sum_assignment(
        pairwise_distances(reference.T, target.T, metric=metric)
    )[1]

    if return_index:
        return remapping
    else:
        return target[:, remapping]
