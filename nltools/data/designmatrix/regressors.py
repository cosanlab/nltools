"""
Standalone regressor functions for DesignMatrix.

Each function takes a DesignMatrix as its first argument (`dm`) and returns
a new DesignMatrix with the requested transformation applied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import polars as pl

from .utils import copy_with, get_data_columns

if TYPE_CHECKING:
    from . import DesignMatrix


def convolve(
    dm: DesignMatrix,
    conv_func: Union[str, np.ndarray] = "hrf",
    columns: Optional[List[str]] = None,
) -> DesignMatrix:
    """
    Convolve columns with HRF or custom kernel.

    Args:
        dm: DesignMatrix to convolve.
        conv_func (str or ndarray): 'hrf' for canonical Glover HRF, or custom kernel(s).
            Can be 1D array (single kernel) or 2D (samples x kernels)
        columns (list of str, optional): Columns to convolve (default: all non-polynomial columns)

    Returns:
        DesignMatrix: New DesignMatrix with convolved columns

    Examples:
        >>> # Default HRF convolution
        >>> dm_conv = convolve(dm)

        >>> # Custom kernel
        >>> kernel = np.array([0.5, 1.0, 0.5])
        >>> dm_conv = convolve(dm, conv_func=kernel)

        >>> # Multiple kernels (FIR model)
        >>> kernels = np.array([[1.0, 0.5], [0.5, 1.0]]).T  # 2 kernels
        >>> dm_conv = convolve(dm, conv_func=kernels)  # Creates col_c0, col_c1
    """
    from nltools.algorithms.hrf import glover_hrf

    if dm.sampling_freq is None:
        raise ValueError(
            "DesignMatrix must have sampling_freq set for convolution. "
            "Specify sampling_freq when creating: DesignMatrix(..., sampling_freq=0.5)"
        )

    # Determine which columns to convolve
    if columns is None:
        # Default: all columns except polynomials
        columns_to_convolve = get_data_columns(dm, exclude_polys=True)
    else:
        columns_to_convolve = columns

    # Get the convolution kernel
    if isinstance(conv_func, str):
        if conv_func != "hrf":
            raise ValueError(
                f"String conv_func must be 'hrf', got '{conv_func}'. "
                "Use conv_func='hrf' or provide a numpy array. "
                "Tip: Use nltools.utils.glover_hrf() to generate custom HRFs."
            )
        # Generate Glover HRF at this sampling frequency
        # TR = 1 / sampling_freq
        conv_func = glover_hrf(1.0 / dm.sampling_freq, oversampling=1.0)
    elif isinstance(conv_func, np.ndarray):
        if len(conv_func.shape) > 2:
            raise ValueError(
                f"HRF function must be 1D (shape: (samples,)) or 2D (shape: (samples, n_kernels)). "
                f"Got shape: {conv_func.shape}. "
                "Tip: Use nltools.utils.glover_hrf() to generate HRFs."
            )
    else:
        raise TypeError(
            f"conv_func must be 'hrf' (str) or numpy array, got {type(conv_func).__name__}. "
            "Tip: Use conv_func='hrf' for canonical HRF."
        )

    # Perform convolution
    n_rows = dm.shape[0]

    if len(conv_func.shape) == 1:
        # Single kernel: keep original column names (replace in-place)
        convolved_series = []
        for col in columns_to_convolve:
            # Convert to numpy for convolution operation
            # NECESSARY: np.convolve requires numpy arrays (no Polars equivalent)
            col_data = dm._df[col].to_numpy()
            convolved = np.convolve(col_data, conv_func)[:n_rows]
            convolved_series.append(pl.Series(col, convolved))

        # Use .with_columns() to replace only convolved columns
        # (automatically preserves non-convolved columns and column order)
        new_df = dm._df.with_columns(convolved_series)

    else:
        # Multiple kernels: shape is (samples, n_kernels)
        n_kernels = conv_func.shape[1]
        convolved_series = []

        for col in columns_to_convolve:
            col_data = dm._df[col].to_numpy()
            for k_idx in range(n_kernels):
                kernel = conv_func[:, k_idx]
                convolved = np.convolve(col_data, kernel)[:n_rows]
                convolved_series.append(pl.Series(f"{col}_c{k_idx}", convolved))

        # Drop original columns, add convolved variants (idiomatic Polars pattern)
        new_df = dm._df.drop(columns_to_convolve).with_columns(convolved_series)

    # Update metadata
    return copy_with(dm, new_df, convolved=columns_to_convolve)


def add_poly(
    dm: DesignMatrix,
    order: int = 0,
    include_lower: bool = True,
) -> DesignMatrix:
    """
    Add Legendre polynomial drift terms.

    Args:
        dm: DesignMatrix to add polynomials to.
        order (int): Polynomial order (0=intercept, 1=linear, 2=quadratic, ...).
            Default: 0.
        include_lower (bool): If True, include all orders from 0 to order.
            Default: True.

    Returns:
        DesignMatrix: New DesignMatrix with polynomial columns appended.

    Raises:
        ValueError: If order < 0 or if ambiguous polynomials exist from a
            previous append operation.
    """
    from scipy.special import legendre

    if order < 0:
        raise ValueError(
            f"Polynomial order must be >= 0, got {order}. "
            "Common orders: 0 (intercept only), 1 (linear trend), 2 (quadratic), 3 (cubic)."
        )

    # Check for ambiguous polynomials from previous append operations
    if dm.polys and any(elem.count("_") == 2 for elem in dm.polys):
        raise ValueError(
            "This Design Matrix contains polynomial terms that were kept "
            "separate from a previous append operation. This makes it ambiguous "
            "for adding polynomial terms. Try calling .add_poly() on each "
            "separate Design Matrix before appending them instead."
        )

    # Determine which polynomials to add
    if include_lower:
        orders_to_add = range(order + 1)
    else:
        orders_to_add = [order]

    # Check if we already have these polynomials (idempotent)
    new_poly_cols = {}
    for i in orders_to_add:
        poly_name = f"poly_{i}"
        if poly_name in dm.polys:
            print(f"Design Matrix already has {i}th order polynomial...skipping")
        else:
            # Create normalized Legendre polynomial over [-1, 1]
            norm_order = np.linspace(-1, 1, dm.shape[0])
            poly_values = legendre(i)(norm_order)
            new_poly_cols[poly_name] = poly_values

    # If no new polynomials to add, return dm unchanged
    if not new_poly_cols:
        return dm

    # Add new polynomial columns using Polars .with_columns()
    new_df = dm._df.with_columns(
        [pl.Series(name, values) for name, values in new_poly_cols.items()]
    )

    # Update polys metadata
    new_polys = dm.polys.copy() if dm.polys else []
    new_polys.extend(new_poly_cols.keys())

    # Return new DesignMatrix with updated data and metadata
    return copy_with(dm, new_df, polys=new_polys)


def add_dct_basis(
    dm: DesignMatrix,
    duration: float = 180,
    drop: int = 0,
) -> DesignMatrix:
    """
    Add discrete cosine transform basis functions (high-pass filter).

    Args:
        dm: DesignMatrix to add DCT basis to.
        duration (float): Filter duration in seconds. Default: 180.
        drop (int): Number of low-frequency bases to drop. Default: 0.

    Returns:
        DesignMatrix: New DesignMatrix with DCT basis columns appended.

    Raises:
        ValueError: If sampling_freq is not set or if ambiguous cosine bases
            exist from a previous append operation.
    """
    from nltools.stats import make_cosine_basis

    if dm.sampling_freq is None:
        raise ValueError(
            "DesignMatrix must have sampling_freq set for DCT basis functions. "
            "Specify sampling_freq when creating: DesignMatrix(..., sampling_freq=0.5)"
        )

    # Check for ambiguous cosine bases from previous append operations
    if dm.polys and any(elem.count("_") == 2 and "cosine" in elem for elem in dm.polys):
        raise ValueError(
            "This Design Matrix contains cosine bases that were kept "
            "separate from a previous append operation. This makes it ambiguous "
            "for adding polynomial terms. Try calling .add_dct_basis() on each "
            "separate Design Matrix before appending them instead."
        )

    # Create DCT basis matrix using stats function
    basis_mat = make_cosine_basis(
        dm.shape[0], 1.0 / dm.sampling_freq, duration, drop=drop
    )

    # Generate column names (cosine_1, cosine_2, ...)
    # Note: If drop > 0, numbering starts from drop+1 to reflect original indices
    # e.g., drop=2 -> cosine_3, cosine_4, ... (skipped cosine_1, cosine_2)
    basis_col_names = [f"cosine_{drop + i + 1}" for i in range(basis_mat.shape[1])]

    # Check which bases we don't already have (idempotent)
    if dm.polys:
        basis_to_add = [name for name in basis_col_names if name not in dm.polys]
    else:
        basis_to_add = basis_col_names

    # If no new bases to add, return dm unchanged
    if not basis_to_add:
        print("All basis functions already exist...skipping")
        return dm

    # Print message if only adding some bases
    if len(basis_to_add) < len(basis_col_names):
        print("Some basis functions already exist...skipping")

    # Add new cosine basis columns
    # Only add the columns we don't already have
    new_basis_cols = {}
    for i, name in enumerate(basis_col_names):
        if name in basis_to_add:
            new_basis_cols[name] = basis_mat[:, i]

    new_df = dm._df.with_columns(
        [pl.Series(name, values) for name, values in new_basis_cols.items()]
    )

    # Update polys metadata
    new_polys = dm.polys.copy() if dm.polys else []
    new_polys.extend(new_basis_cols.keys())

    # Return new DesignMatrix with updated data and metadata
    return copy_with(dm, new_df, polys=new_polys)
