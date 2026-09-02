"""Temporal signal processing — resampling, filtering, and basis functions."""

__all__ = [
    "calc_bpm",
    "downsample",
    "make_cosine_basis",
    "upsample",
]

import numpy as np
import polars as pl
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt


def calc_bpm(beat_interval, sampling_freq):
    """Calculate instantaneous BPM from beat to beat interval.

    Args:
        beat_interval (int): Number of samples between beats (typically the R-R interval).
        sampling_freq (float): Sampling frequency in Hz.

    Returns:
        float: Beats per minute for the time interval.
    """
    return 60 * sampling_freq * (1 / (beat_interval))


def downsample(
    data, *, sampling_freq=None, target=None, target_type="samples", method="mean"
):
    """Downsample a Polars DataFrame/Series to a new target frequency or number of samples using averaging.

    Args:
        data (pl.DataFrame | pl.Series): Data to downsample.
        sampling_freq (float): Sampling frequency of the data in Hz.
        target (float): Downsampling target.
        target_type (str): Unit of `target`, one of 'samples', 'seconds', or 'hz'.
            Defaults to 'samples'.
        method (str): Aggregation within each bin, 'mean' or 'median'. Defaults to 'mean'.

    Returns:
        pl.DataFrame | pl.Series: Downsampled data (same type as input).
    """
    if isinstance(data, pl.DataFrame):
        df = data.clone()
        return_series = False
    elif isinstance(data, pl.Series):
        df = pl.DataFrame({data.name or "0": data})
        return_series = True
    else:
        raise ValueError("Data must be a Polars DataFrame or Series instance.")

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
    data, *, sampling_freq=None, target=None, target_type="samples", method="linear"
):
    """Upsample a Polars DataFrame/Series to a new target frequency or number of samples using interpolation.

    Args:
        data (pl.DataFrame | pl.Series): Data to upsample. Non-numeric columns
            are dropped from a DataFrame.
        sampling_freq (float): Sampling frequency of the data in Hz.
        target (float): Upsampling target.
        target_type (str): Unit of `target`, one of 'samples', 'seconds', or 'hz'.
        method (str): Interpolation method, one of 'linear', 'nearest', 'zero',
            'slinear', 'quadratic', or 'cubic'; 'zero', 'slinear', 'quadratic'
            and 'cubic' refer to spline interpolation of zeroth, first, second
            or third order (default: 'linear').

    Returns:
        pl.DataFrame | pl.Series: Upsampled data, the same type as the input.
    """
    if isinstance(data, pl.DataFrame):
        df = data.clone()
        return_series = False
    elif isinstance(data, pl.Series):
        df = pl.DataFrame({data.name or "0": data})
        return_series = True
    else:
        raise ValueError("Data must be a Polars DataFrame or Series instance.")

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
    """Create basis functions for a discrete cosine transform.

    Based on the implementation in ``spm_filter`` and ``spm_dctmtx`` because
    scipy DCT can only apply transforms but not return the basis functions. Like
    SPM, this does not add a constant (i.e. intercept), but does retain the first
    basis (i.e. sigmoidal/linear drift).

    Args:
        nsamples (int): Number of observations (e.g. TRs).
        sampling_freq (float): Sampling frequency in Hz (i.e. 1 / TR).
        filter_length (int): Filter length in seconds.
        unit_scale (bool): Scale the basis functions to the range [-1, 1]. Defaults to True.
        drop (int): Number of leading (slowest) bases to drop after the constant is
            removed; `drop=2` removes the first two. Defaults to 0, which keeps
            the linear/sigmoidal drift basis that SPM discards.

    Returns:
        np.ndarray: Basis matrix of shape (nsamples, n_bases).
    """

    # Figure out number of basis functions to create
    order = int(np.trunc(2 * (nsamples * sampling_freq) / filter_length + 1))

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
    """Apply a bandpass butterworth filter with zero-phase filtering.

    Args:
        data (np.ndarray): Signal(s) to filter.
        low_cut (float): Lower cutoff frequency (high-pass edge) in Hz.
        high_cut (float): Upper cutoff frequency (low-pass edge) in Hz.
        fs (float): Sampling frequency in Hz.
        axis (int): Axis along which to filter. Defaults to 0.
        order (int): Butterworth filter order. Defaults to 5.

    Returns:
        np.ndarray: Bandpass-filtered data with the same shape as `data`.
    """
    nyq = 0.5 * fs
    b, a = butter(order, [low_cut / nyq, high_cut / nyq], btype="band")
    return filtfilt(b, a, data, axis=axis)


def _phase_mean_angle(phase_angles):
    """Compute the circular mean of phase angles.

    Follows Fisher, N. I. (1995), *Statistical Analysis of Circular Data*.

    Args:
        phase_angles (np.ndarray): A 1D array of angles, or a 2D array whose rows
            are sets of angles (e.g. time points x subjects).

    Returns:
        np.ndarray: The mean angle; one value per row for 2D input.
    """

    axis = 0 if len(phase_angles.shape) == 1 else 1
    return np.arctan2(
        np.mean(np.sin(phase_angles), axis=axis),
        np.mean(np.cos(phase_angles), axis=axis),
    )


def _phase_vector_length(phase_angles):
    """Compute the mean resultant vector length of phase angles.

    Follows Fisher, N. I. (1995), *Statistical Analysis of Circular Data*.

    Args:
        phase_angles (np.ndarray): A 1D array of angles, or a 2D array whose rows
            are sets of angles (e.g. time points x subjects).

    Returns:
        np.ndarray: Vector length in [0, 1]; one value per row for 2D input.
    """

    axis = 0 if len(phase_angles.shape) == 1 else 1
    return np.float32(
        np.sqrt(
            np.mean(np.cos(phase_angles), axis=axis) ** 2
            + np.mean(np.sin(phase_angles), axis=axis) ** 2
        )
    )


def _phase_rayleigh_p(phase_angles):
    """Compute Rayleigh-test p-values for non-uniformity of phase angles.

    Follows Fisher, N. I. (1995), *Statistical Analysis of Circular Data*.

    Args:
        phase_angles (np.ndarray): A 1D array of angles, or a 2D array whose rows
            are sets of angles (e.g. time points x subjects).

    Returns:
        np.ndarray: Rayleigh p-values; one value per row for 2D input.

    Note:
        The test treats the angles in each set as independent, which
        autocorrelated timeseries violate.
    """

    n = len(phase_angles) if len(phase_angles.shape) == 1 else phase_angles.shape[1]

    Z = n * _phase_vector_length(phase_angles) ** 2
    if n <= 50:
        return np.exp(-1 * Z) * (
            1
            + (2 * Z - Z**2) / (4 * n)
            - (24 * Z - 132 * Z**2 + 76 * Z**3 - 9 * Z**4) / (288 * n**2)
        )
    return np.exp(-1 * Z)
