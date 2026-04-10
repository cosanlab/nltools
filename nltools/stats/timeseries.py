"""Temporal signal processing — resampling, filtering, and basis functions."""

__all__ = [
    "downsample",
    "upsample",
    "calc_bpm",
    "make_cosine_basis",
]

import numpy as np
import polars as pl
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt


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


def downsample(
    data, sampling_freq=None, target=None, target_type="samples", method="mean"
):
    """Downsample Polars or pandas DataFrame/Series to a new target frequency or number of samples using averaging.

    Args:
        data: (pl.DataFrame, pl.Series, pd.DataFrame, pd.Series) data to downsample
        sampling_freq:  (float) Sampling frequency of data in hertz
        target: (float) downsampling target
        target_type: type of target can be [samples,seconds,hz]
        method: (str) type of downsample method ['mean','median'],
                default: mean

    Returns:
        out: (pl.DataFrame, pl.Series) downsampled data (same type as input)
    """
    import pandas as pd

    # Convert pandas to Polars if needed (for backward compatibility)
    if isinstance(data, pd.DataFrame):
        df = pl.from_pandas(data)
        return_series = False
    elif isinstance(data, pd.Series):
        df = pl.DataFrame({data.name or "0": data})
        return_series = True
    elif isinstance(data, pl.DataFrame):
        df = data.clone()
        return_series = False
    elif isinstance(data, pl.Series):
        df = pl.DataFrame({data.name or "0": data})
        return_series = True
    else:
        raise ValueError(
            "Data must be a Polars or pandas DataFrame or Series instance."
        )

    if method not in ("mean", "median"):
        raise ValueError("Metric must be either 'mean' or 'median'")

    if target_type == "samples":
        n_samples = target
    elif target_type == "seconds":
        n_samples = target * sampling_freq
    elif target_type == "hz":
        n_samples = sampling_freq / target
    else:
        raise ValueError('Make sure target_type is "samples", "seconds",  or "hz".')

    # Calculate grouping indices more efficiently (matches design_matrix.py pattern)
    n_groups = int(np.ceil(df.shape[0] / n_samples))
    idx = pl.Series(np.repeat(np.arange(n_groups), int(n_samples))[: df.shape[0]])

    # Handle remainder samples (last incomplete group)
    if df.shape[0] > len(idx):
        remainder = pl.Series(np.repeat(idx[-1] + 1, df.shape[0] - len(idx)))
        idx = pl.concat([idx, remainder])

    # Add grouping index to dataframe
    df_with_idx = df.with_columns(idx.alias("_group_idx"))

    # Group by index and aggregate using Polars group_by
    if method == "mean":
        downsampled_df = (
            df_with_idx.group_by("_group_idx", maintain_order=True)
            .agg([pl.col(col).mean() for col in df.columns])
            .drop("_group_idx")
        )
    else:  # median
        downsampled_df = (
            df_with_idx.group_by("_group_idx", maintain_order=True)
            .agg([pl.col(col).median() for col in df.columns])
            .drop("_group_idx")
        )

    # Return Series if input was Series, otherwise DataFrame
    if return_series:
        return downsampled_df.to_series(0)
    return downsampled_df


def upsample(
    data, sampling_freq=None, target=None, target_type="samples", method="linear"
):
    """Upsample Polars or pandas DataFrame/Series to a new target frequency or number of samples using interpolation.

    Args:
        data: (pl.DataFrame, pl.Series, pd.DataFrame, pd.Series) data to upsample
              (Note: will drop non-numeric columns from DataFrame)
        sampling_freq:  Sampling frequency of data in hertz
        target: (float) upsampling target
        target_type: (str) type of target can be [samples,seconds,hz]
        method: (str) ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
                      where 'zero', 'slinear', 'quadratic' and 'cubic'
                      refer to a spline interpolation of zeroth, first,
                      second or third order  (default: linear)
    Returns:
        upsampled Polars DataFrame or Series (same type as input)
    """
    import pandas as pd

    # Convert pandas to Polars if needed (for backward compatibility)
    if isinstance(data, pd.DataFrame):
        df = pl.from_pandas(data)
        return_series = False
    elif isinstance(data, pd.Series):
        df = pl.DataFrame({data.name or "0": data})
        return_series = True
    elif isinstance(data, pl.DataFrame):
        df = data.clone()
        return_series = False
    elif isinstance(data, pl.Series):
        df = pl.DataFrame({data.name or "0": data})
        return_series = True
    else:
        raise ValueError(
            "Data must be a Polars or pandas DataFrame or Series instance."
        )

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

    orig_spacing = np.arange(0, df.shape[0], 1)
    new_spacing = np.arange(0, df.shape[0] - 1, n_samples)

    # Interpolate each column using scipy (matches stats.upsample logic)
    upsampled_data = {}
    for col in df.columns:
        col_data = df[col].to_numpy()

        # Create interpolation function
        interpolate = interp1d(orig_spacing, col_data, kind=method)

        # Interpolate to new indices
        upsampled_data[col] = interpolate(new_spacing)

    # Create new Polars DataFrame
    upsampled_df = pl.DataFrame(upsampled_data)

    # Return Series if input was Series, otherwise DataFrame
    if return_series:
        return upsampled_df.to_series(0)
    return upsampled_df


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
    C[:, 0] = np.ones(len(n)) / np.sqrt(nsamples)

    # Insert higher order cosine basis functions (vectorized)
    if order > 1:
        # Vectorize: create index matrix for broadcasting
        i_indices = np.arange(1, order)[:, np.newaxis]  # (order-1, 1)
        n_indices = n[np.newaxis, :]  # (1, nsamples)
        # Compute all cosine basis functions at once
        C[:, 1:] = (
            np.sqrt(2.0 / nsamples)
            * np.cos(np.pi * (2 * n_indices + 1) * i_indices / (2 * nsamples))
        ).T

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
