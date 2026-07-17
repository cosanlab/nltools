"""Standalone transform functions for DesignMatrix.

Each function takes a DesignMatrix instance as the first argument (`dm`)
and returns a new DesignMatrix via `copy_with(dm,...)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from .utils import copy_with, get_data_columns

if TYPE_CHECKING:
    from nltools.data.designmatrix import DesignMatrix


def zscore(dm: DesignMatrix, columns: list[str] | None = None) -> DesignMatrix:
    """Z-score standardize columns to mean zero and unit variance.

    Args:
        dm: DesignMatrix instance to transform.
        columns (list of str, optional): Columns to standardize. If None,
            standardize all non-polynomial columns.

    Returns:
        DesignMatrix: New DesignMatrix with standardized columns
    """
    # Determine which columns to z-score
    if columns is None:
        # Default: all columns except polynomials
        columns_to_zscore = get_data_columns(dm, exclude_confounds=True)
    else:
        columns_to_zscore = columns

    # Build Polars expressions for z-scoring
    # For each column: (col - mean) / std
    zscore_exprs = [
        ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
        for col in columns_to_zscore
    ]

    # Use .with_columns() to replace only the zscored columns
    # (automatically preserves untouched columns - idiomatic Polars pattern)
    zscored_df = dm.data.with_columns(zscore_exprs)

    return copy_with(dm, zscored_df)


def standardize(
    dm: DesignMatrix,
    columns: list[str] | None = None,
    method: str = "zscore",
) -> DesignMatrix:
    """Standardize columns using the specified method.

    This method provides a consistent API with BrainData and Collection
    for data normalization.

    Args:
        dm: DesignMatrix instance to transform.
        columns: Columns to standardize. If None, standardize all
            non-polynomial columns.
        method: Standardization method. Options are:
            - 'zscore': Z-score standardization (mean=0, std=1) [default]
            - 'center': Mean centering only (mean=0)

    Returns:
        DesignMatrix: New DesignMatrix with standardized columns.

    Raises:
        ValueError: If an invalid method is specified.

    Examples:
        >>> dm = DesignMatrix(np.random.randn(100, 3))
        >>> dm_z = standardize(dm, method='zscore')  # z-score all columns
        >>> dm_c = standardize(dm, method='center')  # center only
    """
    if method == "zscore":
        return zscore(dm, columns=columns)
    if method == "center":
        # Determine which columns to center
        if columns is None:
            columns_to_center = get_data_columns(dm, exclude_confounds=True)
        else:
            columns_to_center = columns

        # Build Polars expressions for centering: (col - mean)
        center_exprs = [
            (pl.col(col) - pl.col(col).mean()).alias(col) for col in columns_to_center
        ]

        centered_df = dm.data.with_columns(center_exprs)
        return copy_with(dm, centered_df)
    raise ValueError(f"Invalid method '{method}'. Must be 'zscore' or 'center'.")


def downsample(dm: DesignMatrix, target: float, **kwargs) -> DesignMatrix:
    """Reduce temporal resolution using Polars-native operations.

    Args:
        dm: DesignMatrix instance to transform.
        target (float): Target sampling frequency in Hz (must be < current
            sampling_freq).
        **kwargs: Additional keyword arguments:

            - **method** (str): Aggregation method - 'mean' or 'median'.
              Default: 'mean'.

    Returns:
        DesignMatrix: Downsampled DesignMatrix with updated sampling_freq.

    Raises:
        ValueError: If sampling_freq is not set, target >= current sampling_freq,
            or method is invalid.

    Examples:
        >>> dm = DesignMatrix({"a": list(range(100))}, sampling_freq=1.0)
        >>> dm_down = downsample(dm, target=0.5)  # 1 Hz -> 0.5 Hz (100 -> 50 samples)
    """
    method = kwargs.pop("method", "mean")

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


def upsample(
    dm: DesignMatrix, target: float, method: str = "linear", **kwargs
) -> DesignMatrix:
    """Increase temporal resolution using Polars-native interpolation.

    Args:
        dm: DesignMatrix instance to transform.
        target (float): Target sampling frequency in Hz (must be > current
            sampling_freq)
        method (str): Interpolation method - 'linear' or 'nearest'
            (default: 'linear')
        **kwargs: Reserved for future extensions

    Returns:
        DesignMatrix: Upsampled DesignMatrix with updated sampling_freq.

    Raises:
        ValueError: If sampling_freq is not set, target <= current sampling_freq,
            or method is invalid.

    Examples:
        >>> dm = DesignMatrix({"a": list(range(10))}, sampling_freq=1.0)
        >>> dm_up = upsample(dm, target=2.0)  # 1 Hz -> 2 Hz (10 -> 19 samples)
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
