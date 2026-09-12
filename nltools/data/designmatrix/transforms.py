"""Standardize and resample a DesignMatrix.

`standardize` normalizes columns; `downsample` and `upsample` change the temporal
resolution. Each returns a new `DesignMatrix` with metadata preserved (and
`sampling_freq` updated when resampling).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from .utils import copy_with, get_data_columns

if TYPE_CHECKING:
    from nltools.data.designmatrix import DesignMatrix


def standardize(
    dm: DesignMatrix,
    *,
    method: str = "center",
    columns: list[str] | None = None,
) -> DesignMatrix:
    """Standardize columns by centering them, optionally scaling to unit variance.

    Args:
        dm (DesignMatrix): DesignMatrix instance to transform.
        method (str): ``'center'`` subtracts the mean (default); ``'zscore'``
            subtracts the mean and divides by the standard deviation.
        columns (list[str] | None): Columns to standardize. If None,
            standardize all non-confound columns.

    Returns:
        DesignMatrix: New DesignMatrix with standardized columns.

    Raises:
        ValueError: If `method` is neither ``'center'`` nor ``'zscore'``.

    Examples:
        ```python
        dm = DesignMatrix(np.random.randn(100, 3))
        dm_c = standardize(dm)  # center every non-confound column
        dm_z = standardize(dm, method="zscore")  # center and scale
        ```
    """
    if method not in ("center", "zscore"):
        raise ValueError(f"method must be 'center' or 'zscore', got {method!r}")

    if columns is None:
        columns = get_data_columns(dm, exclude_confounds=True)

    def standardized(col: str) -> pl.Expr:
        expr = pl.col(col) - pl.col(col).mean()
        if method == "zscore":
            expr = expr / pl.col(col).std()
        return expr.alias(col)

    return copy_with(dm, dm.data.with_columns(standardized(col) for col in columns))


def downsample(dm: DesignMatrix, target: float, method: str = "mean") -> DesignMatrix:
    """Reduce temporal resolution by aggregating consecutive samples.

    Args:
        dm (DesignMatrix): DesignMatrix instance to transform.
        target (float): Target sampling frequency in Hz (must be < current
            `sampling_freq`).
        method (str): Aggregation method, ``'mean'`` or ``'median'``.
            Default: ``'mean'``.

    Returns:
        DesignMatrix: Downsampled DesignMatrix with updated `sampling_freq`.

    Raises:
        ValueError: If `sampling_freq` is not set, `target` >= current
            `sampling_freq`, or `method` is invalid.

    Examples:
        ```python
        dm = DesignMatrix({"a": list(range(100))}, sampling_freq=1.0)
        dm_down = downsample(dm, target=0.5)  # 1 Hz → 0.5 Hz (100 → 50 samples)
        ```
    """
    if dm.sampling_freq is None:
        raise ValueError(
            "DesignMatrix must have sampling_freq set for downsampling. "
            "Specify sampling_freq when creating: DesignMatrix(..., sampling_freq=0.5)"
        )

    if target >= dm.sampling_freq:
        raise ValueError(
            f"Downsampling target ({target} Hz) must be less than current sampling_freq "
            f"({dm.sampling_freq} Hz). For upsampling, use .upsample() instead."
        )

    if method not in ("mean", "median"):
        raise ValueError("method must be 'mean' or 'median'")

    # Calculate n_samples (number of original samples per downsampled sample)
    # This replicates stats.downsample() logic: n_samples = sampling_freq / target
    n_samples = dm.sampling_freq / target

    # Assign each row to a group via floor(row / n_samples). For integer ratios
    # this reproduces the old [0,0,1,1,...] grouping exactly; for non-integer
    # ratios it spreads the leftover rows evenly across bins instead of lumping
    # them all into one oversized final group (F083).
    idx = pl.Series(np.floor(np.arange(dm.shape[0]) / n_samples).astype(int))

    # Add grouping index to dataframe
    df_with_idx = dm.data.with_columns(idx.alias("_group_idx"))

    # Get all data columns
    data_cols = get_data_columns(dm, exclude_confounds=False)

    # Group by index and aggregate
    if method == "mean":
        downsampled_df = (
            df_with_idx.group_by("_group_idx", maintain_order=True)
            .agg([pl.col(col).mean() for col in data_cols])
            .drop("_group_idx")
        )
    else:  # median
        downsampled_df = (
            df_with_idx.group_by("_group_idx", maintain_order=True)
            .agg([pl.col(col).median() for col in data_cols])
            .drop("_group_idx")
        )

    return copy_with(dm, downsampled_df, sampling_freq=target)


def upsample(dm: DesignMatrix, target: float, method: str = "linear") -> DesignMatrix:
    """Increase temporal resolution by interpolating between samples.

    Args:
        dm (DesignMatrix): DesignMatrix instance to transform.
        target (float): Target sampling frequency in Hz (must be > current
            `sampling_freq`).
        method (str): Interpolation method, ``'linear'`` or ``'nearest'``.
            Default: ``'linear'``.

    Returns:
        DesignMatrix: Upsampled DesignMatrix with updated `sampling_freq`.

    Raises:
        ValueError: If `sampling_freq` is not set, `target` <= current
            `sampling_freq`, or `method` is invalid.

    Examples:
        ```python
        dm = DesignMatrix({"a": list(range(10))}, sampling_freq=1.0)
        dm_up = upsample(dm, target=2.0)  # 1 Hz → 2 Hz (10 → 18 samples)
        ```
    """
    from scipy.interpolate import interp1d

    if dm.sampling_freq is None:
        raise ValueError(
            "DesignMatrix must have sampling_freq set for upsampling. "
            "Specify sampling_freq when creating: DesignMatrix(..., sampling_freq=0.5)"
        )

    if target <= dm.sampling_freq:
        raise ValueError(
            f"Upsampling target ({target} Hz) must be greater than current sampling_freq "
            f"({dm.sampling_freq} Hz). For downsampling, use .downsample() instead."
        )

    if method not in ("linear", "nearest"):
        raise ValueError("method must be 'linear' or 'nearest'")

    # Calculate step size (this matches stats.upsample logic)
    # For hz target_type: n_samples = sampling_freq / target
    step_size = dm.sampling_freq / target

    # Create original and new index arrays (matches stats.upsample)
    orig_indices = np.arange(0, dm.shape[0], 1)
    new_indices = np.arange(0, dm.shape[0] - 1, step_size)

    # Get all data columns (including confounds - upsample everything)
    data_cols = get_data_columns(dm, exclude_confounds=False)

    # Interpolate each column using scipy (matches stats.upsample)
    upsampled_data = {}
    for col in data_cols:
        col_data = dm.data[col].to_numpy()

        # Create interpolation function
        interpolate = interp1d(orig_indices, col_data, kind=method)

        # Interpolate to new indices
        upsampled_data[col] = interpolate(new_indices)

    # Create new Polars DataFrame
    upsampled_df = pl.DataFrame(upsampled_data)

    return copy_with(dm, upsampled_df, sampling_freq=target)
