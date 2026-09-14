"""Temporal signal processing — resampling, filtering, and basis functions."""

import warnings

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

    Note:
        Rows are binned by `floor(row / n_samples)`, the same rule
        `DesignMatrix.downsample` uses, so a non-integer ratio spreads its
        leftover rows across the bins rather than into an extra final one.
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

    # Assign each row to a group via floor(row / n_samples), the same rule
    # DesignMatrix.downsample uses. For integer ratios this reproduces the old
    # [0,0,1,1,...] grouping exactly; for non-integer ratios it spreads the
    # leftover rows evenly across bins instead of truncating the bin width and
    # opening an extra group at the end.
    idx = pl.Series(np.floor(np.arange(df.shape[0]) / n_samples).astype(int))

    # The grouping key is a transient that never reaches the result, so it takes
    # a name the frame does not already use.
    group_key = "_group_idx"
    while group_key in df.columns:
        group_key += "_"

    # Add grouping index to dataframe
    df_with_idx = df.with_columns(idx.alias(group_key))

    # Group by index and aggregate using Polars group_by
    if method == "mean":
        downsampled_df = (
            df_with_idx.group_by(group_key, maintain_order=True)
            .agg([pl.col(col).mean() for col in df.columns])
            .drop(group_key)
        )
    else:  # median
        downsampled_df = (
            df_with_idx.group_by(group_key, maintain_order=True)
            .agg([pl.col(col).median() for col in df.columns])
            .drop(group_key)
        )

    # Return Series if input was Series, otherwise DataFrame
    if return_series:
        return downsampled_df.to_series(0)
    return downsampled_df


def _upsample_indices(n_rows, n_samples):
    """Positions of the upsampled samples, in units of original samples.

    Args:
        n_rows (int): Number of original samples.
        n_samples (float): Spacing between new samples, in original samples.

    Returns:
        np.ndarray: The new sample positions.
    """
    return np.arange(0, n_rows - 1, n_samples)


def _upsample_frame(frame, n_rows, n_samples, method):
    """Interpolate every column of `frame` onto a finer, evenly spaced grid.

    Args:
        frame (pl.DataFrame): Columns to interpolate.
        n_rows (int): Number of original samples. A column-less frame carries its
            row count outside the frame, so callers pass it explicitly.
        n_samples (float): Spacing between new samples, in original samples.
        method (str): Interpolation kind passed to `scipy.interpolate.interp1d`.

    Returns:
        pl.DataFrame: One interpolated column per input column.
    """
    orig_spacing = np.arange(0, n_rows, 1)
    new_spacing = _upsample_indices(n_rows, n_samples)
    return pl.DataFrame(
        {
            col: interp1d(orig_spacing, frame[col].to_numpy(), kind=method)(new_spacing)
            for col in frame.columns
        }
    )


def upsample(
    data, *, sampling_freq=None, target=None, target_type="samples", method="linear"
):
    """Upsample a Polars DataFrame/Series to a new target frequency or number of samples using interpolation.

    Args:
        data (pl.DataFrame | pl.Series): Data to upsample. Non-numeric columns
            are dropped from a DataFrame; Boolean columns count as numeric and
            are interpolated as 0/1.
        sampling_freq (float): Sampling frequency of the data in Hz.
        target (float): Upsampling target.
        target_type (str): Unit of `target`, one of 'samples', 'seconds', or 'hz'.
        method (str): Interpolation method, one of 'linear', 'nearest', 'zero',
            'slinear', 'quadratic', or 'cubic'; 'zero', 'slinear', 'quadratic'
            and 'cubic' refer to spline interpolation of zeroth, first, second
            or third order (default: 'linear').

    Returns:
        pl.DataFrame | pl.Series: Upsampled data, the same type as the input.

    Raises:
        ValueError: If `data` is not Polars, if `method` or `target_type` is
            unknown, or if the frame has no numeric columns to interpolate.

    Note:
        Dropping non-numeric columns emits a `UserWarning` naming them.
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

    # The docstring promises non-numeric columns are dropped, as v0.5.1 did;
    # handing one to interp1d raises instead. Booleans count as numeric here —
    # pandas' `_get_numeric_data()` kept them, so v0.5.1 interpolated them —
    # and are cast to float for interp1d.
    kept = [
        c for c in df.columns if df[c].dtype.is_numeric() or df[c].dtype == pl.Boolean
    ]
    dropped = [c for c in df.columns if c not in kept]
    if dropped:
        warnings.warn(
            f"Dropping {len(dropped)} non-numeric column(s) before "
            f"interpolating: {', '.join(dropped)}",
            UserWarning,
            stacklevel=2,
        )
    if not kept:
        raise ValueError("Data has no numeric columns to upsample.")

    upsampled_df = _upsample_frame(
        df.select(pl.col(kept).cast(pl.Float64)), df.shape[0], n_samples, method
    )

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
        sampling_freq (float): Sampling *interval* in seconds (the TR), matching
            SPM's `RT` — not a frequency, despite the name. The number of bases
            is `trunc(2 * nsamples * sampling_freq / filter_length + 1)`, minus
            the constant. `DesignMatrix.add_dct_basis` passes
            `1 / DesignMatrix.sampling_freq` for this reason.
        filter_length (int): Filter length in seconds.
        unit_scale (bool): Scale the basis functions to the range [-1, 1]. Defaults to True.
        drop (int): Number of leading (slowest) bases to drop after the constant is
            removed; `drop=2` removes the first two. Defaults to 0, which keeps
            the linear/sigmoidal drift basis that SPM discards.

    Returns:
        np.ndarray: Basis matrix of shape (nsamples, n_bases).

    Note:
        The basis count follows `spm_dctmtx`'s `k = fix(2*(n*RT)/HParam + 1)`, so
        100 TRs of 2 s with a 128 s filter give the same four bases SPM gives.
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
