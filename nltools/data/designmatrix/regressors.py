"""Build regressors for a DesignMatrix: HRF convolution and drift terms.

`convolve` applies the canonical Glover HRF or a custom kernel; `add_poly` and
`add_dct_basis` add Legendre polynomial and discrete-cosine drift regressors in
the reserved ``.nl_`` namespace. Each function returns a new `DesignMatrix`
with metadata updated.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from nltools.utils import DesignMatrixWarning, find_stack_level, reserved_name

from .utils import copy_with, get_data_columns, has_run_separated_drift

if TYPE_CHECKING:
    from . import DesignMatrix


def convolve(
    dm: DesignMatrix,
    conv_func: str | np.ndarray = "hrf",
    columns: list[str] | None = None,
) -> DesignMatrix:
    """Convolve columns with an HRF or custom kernel.

    Args:
        dm (DesignMatrix): DesignMatrix to convolve.
        conv_func (str | np.ndarray): ``'hrf'`` for the canonical Glover HRF, or
            custom kernel(s) as a 1D array (single kernel) or 2D array
            (samples x kernels).
        columns (list[str] | None): Columns to convolve. Default: all
            non-confound columns that are not already convolved.

    Returns:
        DesignMatrix: New DesignMatrix with convolved columns.

    Examples:
        ```python
        # Default HRF convolution → produces 'stim_c0'
        dm_conv = convolve(dm)

        # Custom 1-D kernel → produces 'stim_c0'
        kernel = np.array([0.5, 1.0, 0.5])
        dm_conv = convolve(dm, conv_func=kernel)

        # Multiple kernels (FIR model) → produces 'stim_c0', 'stim_c1'
        kernels = np.array([[1.0, 0.5], [0.5, 1.0]]).T  # 2 kernels
        dm_conv = convolve(dm, conv_func=kernels)
        ```

    Note:
        Convolved columns are always renamed to ``<col>_c{i}``; the source
        column is dropped. ``dm.convolved`` records the post-suffix names
        (the columns that actually exist in the returned dataframe), so
        downstream metadata propagation through ``.append()`` stays in
        sync with the dataframe.
    """
    from nltools.algorithms.hrf import glover_hrf

    if dm.sampling_freq is None:
        raise ValueError(
            "DesignMatrix must have sampling_freq set for convolution. "
            "Specify sampling_freq when creating: DesignMatrix(..., sampling_freq=0.5)"
        )

    # Determine which columns to convolve
    already_convolved = set(dm.convolved)
    if columns is None:
        # Default: experimental regressors only (drop confounds & polys),
        # idempotent over already-convolved columns — re-convolving would
        # produce ``<col>_c0_c0``, which has no biological meaning.
        columns_to_convolve = [
            c
            for c in get_data_columns(dm, exclude_confounds=True)
            if c not in already_convolved
        ]
        if not columns_to_convolve:
            warnings.warn(
                "All experimental regressors are already convolved; "
                ".convolve() is a no-op.",
                DesignMatrixWarning,
                stacklevel=find_stack_level(),
            )
            return dm
    else:
        # Explicit columns=. Refuse names already in dm.convolved — there is
        # no mathematically sensible "re-convolve" operation. Convolving an
        # HRF-shaped signal with another kernel produces a doubly-blurred
        # thing that doesn't correspond to any neural or hemodynamic process.
        # If the user wants a different kernel, they should rebuild the DM
        # from boxcar regressors and convolve fresh.
        invalid = [c for c in columns if c in already_convolved]
        if invalid:
            raise ValueError(
                f"Cannot re-convolve already-convolved columns: {invalid}. "
                "Convolving an HRF-shaped signal with another kernel has no "
                "biological meaning. To use a different kernel, drop the "
                "convolved column, re-add the boxcar source, and call "
                ".convolve() with the new kernel."
            )
        columns_to_convolve = list(columns)

    # Get the convolution kernel
    if isinstance(conv_func, str):
        if conv_func != "hrf":
            raise ValueError(
                f"String conv_func must be 'hrf', got '{conv_func}'. "
                "Use conv_func='hrf' or provide a numpy array. "
                "Tip: Use nltools.algorithms.hrf.glover_hrf() to generate custom HRFs."
            )
        # Generate Glover HRF at this sampling frequency
        # TR = 1 / sampling_freq
        conv_func = glover_hrf(1.0 / dm.sampling_freq, oversampling=1.0)
    elif isinstance(conv_func, np.ndarray):
        if len(conv_func.shape) > 2:
            raise ValueError(
                f"HRF function must be 1D (shape: (samples,)) or 2D (shape: (samples, n_kernels)). "
                f"Got shape: {conv_func.shape}. "
                "Tip: Use nltools.algorithms.hrf.glover_hrf() to generate HRFs."
            )
    else:
        raise TypeError(
            f"conv_func must be 'hrf' (str) or numpy array, got {type(conv_func).__name__}. "
            "Tip: Use conv_func='hrf' for canonical HRF."
        )

    # Normalize to 2-D (samples, n_kernels) so 1-D and 2-D paths share code.
    kernels_2d = conv_func.reshape(-1, 1) if conv_func.ndim == 1 else conv_func
    n_kernels = kernels_2d.shape[1]
    n_rows = dm.shape[0]

    convolved_series: list[pl.Series] = []
    new_convolved: list[str] = []
    for col in columns_to_convolve:
        # NECESSARY: np.convolve requires numpy arrays (no Polars equivalent)
        col_data = dm.data[col].to_numpy()
        for k_idx in range(n_kernels):
            kernel = kernels_2d[:, k_idx]
            result = np.convolve(col_data, kernel)[:n_rows]
            new_name = f"{col}_c{k_idx}"
            convolved_series.append(pl.Series(new_name, result))
            new_convolved.append(new_name)

    # Drop source columns and add suffixed variants. Single-kernel and
    # multi-kernel are now uniform: source name never survives, output is
    # always ``<col>_c{i}``.
    new_df = dm.data.drop(columns_to_convolve).with_columns(convolved_series)

    # Re-convolution of already-convolved columns is refused above, so any
    # entries in ``dm.convolved`` survived in ``new_df`` untouched; just
    # append the freshly convolved names.
    return copy_with(dm, new_df, convolved=list(dm.convolved) + new_convolved)


def add_poly(
    dm: DesignMatrix,
    order: int = 0,
    include_lower: bool = True,
) -> DesignMatrix:
    """Add Legendre polynomial drift terms.

    Args:
        dm (DesignMatrix): DesignMatrix to add polynomials to.
        order (int): Polynomial order (0=intercept, 1=linear, 2=quadratic, ...).
            Default: 0.
        include_lower (bool): If True, include all orders from 0 to order.
            Default: True.

    Returns:
        DesignMatrix: New DesignMatrix with polynomial columns appended, named
            ``.nl_poly_{order}`` in the reserved namespace (see `RESERVED_PREFIX`).

    Raises:
        ValueError: If order < 0, or if the design already carries run-separated
            drift terms from a previous multi-run append.
    """
    from scipy.special import legendre

    if order < 0:
        raise ValueError(
            f"Polynomial order must be >= 0, got {order}. "
            "Common orders: 0 (intercept only), 1 (linear trend), 2 (quadratic), 3 (cubic)."
        )

    # Adding a global drift term on top of per-run ones is ambiguous.
    if has_run_separated_drift(dm):
        raise ValueError(
            "This Design Matrix contains run-separated drift terms (polynomial "
            "or cosine) from a previous append operation, which makes adding "
            "global polynomial terms ambiguous. Call .add_poly() on each "
            "single-run Design Matrix before appending them instead."
        )

    # Determine which polynomials to add
    if include_lower:
        orders_to_add = range(order + 1)
    else:
        orders_to_add = [order]

    # Detect existing intercept columns (any all-ones confound)
    _has_intercept = False
    if dm.confounds:
        for p in dm.confounds:
            col_vals = dm[p].to_numpy().flatten()
            if np.allclose(col_vals, 1.0):
                _has_intercept = True
                break

    # Check if we already have these polynomials (idempotent)
    new_poly_cols = {}
    for i in orders_to_add:
        poly_name = reserved_name(f"poly_{i}")
        if poly_name in dm.confounds:
            warnings.warn(
                f"Design Matrix already has {i}th order polynomial...skipping",
                DesignMatrixWarning,
                stacklevel=find_stack_level(),
            )
        elif i == 0 and _has_intercept:
            warnings.warn(
                f"Design Matrix already has an intercept column...skipping {poly_name}",
                DesignMatrixWarning,
                stacklevel=find_stack_level(),
            )
        else:
            # Create normalized Legendre polynomial over [-1, 1]
            norm_order = np.linspace(-1, 1, dm.shape[0])
            poly_values = legendre(i)(norm_order)
            new_poly_cols[poly_name] = poly_values

    # If no new polynomials to add, return dm unchanged
    if not new_poly_cols:
        return dm

    # Add new polynomial columns using Polars .with_columns()
    new_df = dm.data.with_columns(
        [pl.Series(name, values) for name, values in new_poly_cols.items()]
    )

    # Update confounds metadata
    new_confounds = dm.confounds.copy() if dm.confounds else []
    new_confounds.extend(new_poly_cols.keys())

    # Return new DesignMatrix with updated data and metadata
    return copy_with(dm, new_df, confounds=new_confounds)


def add_dct_basis(
    dm: DesignMatrix,
    *,
    duration: float = 180,
    drop: int = 0,
    include_constant: bool = True,
) -> DesignMatrix:
    """Add discrete cosine transform basis functions for high-pass filtering.

    Args:
        dm (DesignMatrix): DesignMatrix to add the DCT basis to.
        duration (float): Filter duration in seconds. Default: 180.
        drop (int): Number of low-frequency bases to drop. Default: 0.
        include_constant (bool): If True, also add a constant/intercept column
            named ``.nl_cosine_0`` (analogous to ``.nl_poly_0`` in `add_poly`).
            The underlying DCT basis drops the constant per SPM convention;
            set False to match SPM behavior. Default: True.

    Returns:
        DesignMatrix: New DesignMatrix with DCT basis columns appended, named
            ``.nl_cosine_{i}`` in the reserved namespace (see `RESERVED_PREFIX`).

    Raises:
        ValueError: If sampling_freq is not set, or if the design already
            carries run-separated drift terms from a previous multi-run append.
    """
    from nltools.algorithms.signal import make_cosine_basis

    if dm.sampling_freq is None:
        raise ValueError(
            "DesignMatrix must have sampling_freq set for DCT basis functions. "
            "Specify sampling_freq when creating: DesignMatrix(..., sampling_freq=0.5)"
        )

    # Adding a global drift term on top of per-run ones is ambiguous.
    if has_run_separated_drift(dm):
        raise ValueError(
            "This Design Matrix contains run-separated drift terms (polynomial "
            "or cosine) from a previous append operation, which makes adding "
            "global cosine bases ambiguous. Call .add_dct_basis() on each "
            "single-run Design Matrix before appending them instead."
        )

    # Create DCT basis matrix using stats function
    basis_mat = make_cosine_basis(
        dm.shape[0], 1.0 / dm.sampling_freq, duration, drop=drop
    )

    # Generate column names (.nl_cosine_1, .nl_cosine_2, ...)
    # Note: If drop > 0, numbering starts from drop+1 to reflect original indices
    # e.g., drop=2 -> .nl_cosine_3, .nl_cosine_4, ... (skipped 1 and 2)
    basis_col_names = [
        reserved_name(f"cosine_{drop + i + 1}") for i in range(basis_mat.shape[1])
    ]

    # Optionally prepend the constant/intercept — mirrors .nl_poly_0 in add_poly.
    # make_cosine_basis drops the constant per SPM; we re-add it here when asked,
    # and skip if an intercept-like confounds column already exists.
    if include_constant:
        constant_name = reserved_name("cosine_0")
        _has_intercept = False
        if dm.confounds:
            for p in dm.confounds:
                col_vals = dm[p].to_numpy().flatten()
                if np.allclose(col_vals, 1.0):
                    _has_intercept = True
                    break
        if constant_name in (dm.confounds or []) or _has_intercept:
            warnings.warn(
                f"Design Matrix already has an intercept column...skipping {constant_name}",
                DesignMatrixWarning,
                stacklevel=find_stack_level(),
            )
        else:
            basis_col_names.insert(0, constant_name)
            basis_mat = np.column_stack([np.ones(dm.shape[0]), basis_mat])

    # Check which bases we don't already have (idempotent)
    if dm.confounds:
        basis_to_add = [name for name in basis_col_names if name not in dm.confounds]
    else:
        basis_to_add = basis_col_names

    # If no new bases to add, return dm unchanged
    if not basis_to_add:
        warnings.warn(
            "All basis functions already exist...skipping",
            DesignMatrixWarning,
            stacklevel=find_stack_level(),
        )
        return dm

    if len(basis_to_add) < len(basis_col_names):
        warnings.warn(
            "Some basis functions already exist...skipping",
            DesignMatrixWarning,
            stacklevel=find_stack_level(),
        )

    # Add new cosine basis columns
    # Only add the columns we don't already have
    new_basis_cols = {}
    for i, name in enumerate(basis_col_names):
        if name in basis_to_add:
            new_basis_cols[name] = basis_mat[:, i]

    new_df = dm.data.with_columns(
        [pl.Series(name, values) for name, values in new_basis_cols.items()]
    )

    # Update confounds metadata
    new_confounds = dm.confounds.copy() if dm.confounds else []
    new_confounds.extend(new_basis_cols.keys())

    # Return new DesignMatrix with updated data and metadata
    return copy_with(dm, new_df, confounds=new_confounds)
