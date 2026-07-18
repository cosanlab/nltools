"""Outlier detection, robust statistics, and data normalization."""

import numpy as np
import polars as pl
import nibabel as nib

__all__ = ["find_spikes", "trim", "winsorize", "zscore"]


def zscore(data):
    """Z-score every column of a Polars or pandas DataFrame/Series.

    Accepts pandas inputs at the boundary for convenience and converts to
    Polars internally. Always returns Polars output (DataFrame or Series,
    matching the input shape).

    Args:
        data: pl.DataFrame, pl.Series, pd.DataFrame, or pd.Series.

    Returns:
        pl.DataFrame or pl.Series with each column z-scored using sample
        standard deviation (ddof=1), matching the input shape.
    """
    import pandas as pd

    return_series = False
    if isinstance(data, pd.DataFrame):
        df = pl.from_pandas(data)
    elif isinstance(data, pd.Series):
        df = pl.DataFrame({data.name or "0": data})
        return_series = True
    elif isinstance(data, pl.DataFrame):
        df = data
    elif isinstance(data, pl.Series):
        df = pl.DataFrame({data.name or "0": data})
        return_series = True
    else:
        raise ValueError("Data must be a Polars or pandas DataFrame or Series")

    result = df.select(
        [
            ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c)
            for c in df.columns
        ]
    )

    if return_series:
        return result.to_series(0)
    return result


def winsorize(data, cutoff=None, replace_with_cutoff=True):
    """Winsorize a Polars DataFrame/Series with the largest/lowest value not considered outlier.

    Args:
        data: (pl.DataFrame, pl.Series) data to winsorize
        cutoff: (dict) a dictionary with keys {'std':[low,high]} or
                {'quantile':[low,high]}
        replace_with_cutoff: (bool) If True, replace outliers with cutoff.
                             If False, replaces outliers with closest
                             existing values; (default: True)
    Returns:
        out: (pl.DataFrame, pl.Series) winsorized data (same type as input)
    """
    return _transform_outliers(
        data, cutoff, replace_with_cutoff=replace_with_cutoff, method="winsorize"
    )


def trim(data, cutoff=None):
    """Trim a Polars DataFrame/Series by replacing outlier values with NaNs.

    Args:
        data: (pl.DataFrame, pl.Series) data to trim
        cutoff: (dict) a dictionary with keys {'std':[low,high]} or
                {'quantile':[low,high]}
    Returns:
        out: (pl.DataFrame, pl.Series) trimmed data (same type as input)
    """
    return _transform_outliers(data, cutoff, replace_with_cutoff=None, method="trim")


def _transform_outliers(data, cutoff, replace_with_cutoff, method):
    """Transform outliers in a Polars DataFrame/Series using winsorize or trim.

    This function is not exposed to user but is called by either trim
    or winsorize.

    Args:
        data: (pl.DataFrame, pl.Series) data to transform
        cutoff: (dict) a dictionary with keys {'std':[low,high]} or
                {'quantile':[low,high]}
        replace_with_cutoff: (bool) If True, replace outliers with cutoff.
                                    If False, replaces outliers with closest
                                    existing values. (default: True)
        method: 'winsorize' or 'trim'

    Returns:
        out: (pl.DataFrame, pl.Series) transformed data
    """
    return_series = False
    if isinstance(data, pl.DataFrame):
        df = data.clone()
    elif isinstance(data, pl.Series):
        df = pl.DataFrame({data.name or "0": data})
        return_series = True
    else:
        raise ValueError("Data must be a Polars DataFrame or Series")

    # Transform each column if a DataFrame, if Series just transform data
    if isinstance(df, pl.DataFrame):
        transformed_cols = []
        for col in df.columns:
            # Get the series for calculations
            series = df[col]

            # Calculate cutoff values
            if isinstance(cutoff, dict):
                if "quantile" in cutoff:
                    quantiles = cutoff["quantile"]
                    # Use numpy quantile to match pandas interpolation behavior
                    series_array = series.to_numpy()
                    lower_q = float(np.quantile(series_array, quantiles[0]))
                    upper_q = (
                        float(np.quantile(series_array, quantiles[1]))
                        if len(quantiles) > 1
                        else lower_q
                    )
                elif "std" in cutoff:
                    mean_val = series.mean()
                    std_val = series.std()
                    lower_q = mean_val - std_val * cutoff["std"][0]
                    upper_q = mean_val + std_val * cutoff["std"][1]
                else:
                    raise ValueError(
                        "cutoff must be a dictionary with quantile or std keys."
                    )

                # If replace_with_cutoff is false, replace with true existing values closest to cutoff
                if method == "winsorize" and not replace_with_cutoff:
                    filtered_lower = series.filter(series > lower_q)
                    filtered_upper = series.filter(series < upper_q)
                    if len(filtered_lower) > 0:
                        lower_q = filtered_lower.min()
                    if len(filtered_upper) > 0:
                        upper_q = filtered_upper.max()

                # Apply transformation using Polars expressions with column reference
                if method == "trim":
                    # Replace outliers with null (NaN)
                    transformed_expr = (
                        pl.when(pl.col(col) < lower_q)
                        .then(None)
                        .when(pl.col(col) > upper_q)
                        .then(None)
                        .otherwise(pl.col(col))
                    )
                elif method == "winsorize":
                    # Replace outliers with cutoff values
                    transformed_expr = (
                        pl.when(pl.col(col) < lower_q)
                        .then(lower_q)
                        .when(pl.col(col) > upper_q)
                        .then(upper_q)
                        .otherwise(pl.col(col))
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")

                transformed_cols.append(transformed_expr.alias(col))
            else:
                raise ValueError(
                    "cutoff must be a dictionary with quantile or std keys."
                )

        # Use with_columns to update all columns at once
        result_df = df.with_columns(transformed_cols)

        # Return Series if input was Series, otherwise DataFrame
        if return_series:
            return result_df.to_series(0)
        return result_df
    raise ValueError("Data must be a Polars or pandas DataFrame or Series")


def find_spikes(
    data,
    global_spike_cutoff=3,
    diff_spike_cutoff=3,
    *,
    TR: float | None = None,
    sampling_freq: float | None = None,
):
    """Identify spikes (motion artifacts, intensity outliers) in 4D fMRI data.

    Args:
        data: BrainData or nibabel instance
        global_spike_cutoff: (int, None) cutoff in std-deviations for spikes in
            the per-TR global signal. None to skip.
        diff_spike_cutoff: (int, None) cutoff in std-deviations for spikes in
            the per-TR mean absolute frame-to-frame difference. None to skip.
        TR: Repetition time in seconds. Sets the returned DesignMatrix's
            sampling_freq for downstream `.append(...)` / `.convolve()`.
            Pass exactly one of `TR` or `sampling_freq`.
        sampling_freq: Sampling frequency in Hz (= 1/TR). See `TR`.

    Returns:
        DesignMatrix: one indicator column per detected spike, with all
        spike columns pre-marked as confounds. Row position is the time
        axis (no separate `TR` index column — that was a pandas-era
        artifact). When `TR` / `sampling_freq` aren't provided the DM has
        `sampling_freq=None`; you can still `.append()` it onto a DM that
        does have one.
    """

    from nltools.data import BrainData

    if (global_spike_cutoff is None) & (diff_spike_cutoff is None):
        raise ValueError("Did not input any cutoffs to identify spikes in this data.")

    if isinstance(data, BrainData):
        # Avoid deepcopy overhead - just copy the data array
        data_array = data.data.copy()
        global_mn = np.mean(data_array, axis=1)
        frame_diff = np.mean(np.abs(np.diff(data_array, axis=0)), axis=1)
    elif isinstance(data, nib.Nifti1Image):
        # Avoid deepcopy overhead - just copy the data array
        data_array = data.get_fdata().copy()
        if len(data_array.shape) > 3:
            data_array = np.squeeze(data_array)
        elif len(data_array.shape) < 3:
            raise ValueError("nibabel instance does not appear to be 4D data.")
        global_mn = np.mean(data_array, axis=(0, 1, 2))
        frame_diff = np.mean(np.abs(np.diff(data_array, axis=3)), axis=(0, 1, 2))
    else:
        raise ValueError(
            "Currently this function can only accomodate BrainData and nibabel instances"
        )

    if global_spike_cutoff is not None:
        # Vectorize outlier detection - avoid np.append in loops
        global_mean = np.mean(global_mn)
        global_std = np.std(global_mn)
        upper_threshold = global_mean + global_std * global_spike_cutoff
        lower_threshold = global_mean - global_std * global_spike_cutoff
        global_outliers = np.where(
            (global_mn > upper_threshold) | (global_mn < lower_threshold)
        )[0]

    if diff_spike_cutoff is not None:
        # Vectorize outlier detection - avoid np.append in loops
        diff_mean = np.mean(frame_diff)
        diff_std = np.std(frame_diff)
        upper_threshold = diff_mean + diff_std * diff_spike_cutoff
        lower_threshold = diff_mean - diff_std * diff_spike_cutoff
        frame_outliers = np.where(
            (frame_diff > upper_threshold) | (frame_diff < lower_threshold)
        )[0]
    # Build spike regressors using Polars. Row position is the time axis;
    # no separate "TR" index column (pandas-era artifact, dropped in v0.6.0).
    outlier_data: dict[str, list[int]] = {}

    if global_spike_cutoff is not None:
        for i, loc in enumerate(global_outliers):
            col_name = f"global_spike{i + 1}"
            col_values = [0] * len(global_mn)
            col_values[int(loc)] = 1
            outlier_data[col_name] = col_values

    if diff_spike_cutoff is not None:
        for i, loc in enumerate(frame_outliers):
            col_name = f"diff_spike{i + 1}"
            col_values = [0] * len(global_mn)
            col_values[int(loc)] = 1
            outlier_data[col_name] = col_values

    if TR is not None and sampling_freq is not None:
        raise ValueError(
            "find_spikes: pass exactly one of `TR` or `sampling_freq`, not both."
        )
    if TR is not None:
        sampling_freq = 1.0 / TR

    # Synthesize an empty regressor frame when no spikes were found so the
    # DesignMatrix has the right row count for downstream .append().
    if not outlier_data:
        df = pl.DataFrame({"_no_spikes": [0] * len(global_mn)}).drop("_no_spikes")
    else:
        df = pl.DataFrame(outlier_data)

    from nltools.data import DesignMatrix

    spike_cols = list(df.columns)
    return DesignMatrix(df, sampling_freq=sampling_freq, confounds=spike_cols)
