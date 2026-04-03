"""Diagnostic and utility functions for DesignMatrix."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from nltools.data.designmatrix import DesignMatrix


def vif(dm: DesignMatrix, exclude_polys: bool = True) -> np.ndarray | None:
    """
    Compute variance inflation factor for each column.

    Uses diagonal elements of inverted correlation matrix
    (same method as Matlab and R).

    Args:
        dm: DesignMatrix instance.
        exclude_polys (bool): Skip polynomial columns. Default: True.

    Returns:
        np.ndarray: VIF values for each included column. Returns None if the
            correlation matrix is singular (perfect collinearity detected).

    Raises:
        ValueError: If the DesignMatrix has only 1 column.
    """
    import polars.selectors as cs

    if dm.shape[1] <= 1:
        raise ValueError(
            "Can't compute VIF with only 1 column! "
            "VIF measures multicollinearity and requires at least 2 columns. "
            f"Your DesignMatrix has shape {dm.shape}."
        )

    # Determine which columns to include (using polars selectors for declarative filtering)
    if exclude_polys and dm.polys:
        # Use polars selector: "select all columns except polynomial terms"
        subset_df = dm._df.select(cs.exclude(dm.polys))
    elif exclude_polys:
        # No polys to exclude, use all columns
        subset_df = dm._df
    else:
        # Always exclude intercept (poly_0) columns even when exclude_polys=False
        cols_to_use = [c for c in dm.columns if "poly_0" not in c]
        subset_df = dm._df.select(cols_to_use)

    # Edge case: single column has VIF = 1 (no multicollinearity)
    if subset_df.shape[1] == 1:
        return np.array([1.0])

    # Convert to numpy for correlation matrix and linear algebra
    # NECESSARY: Polars doesn't have correlation matrix or matrix inversion
    data_array = subset_df.to_numpy()

    # Compute correlation matrix
    corr_matrix = np.corrcoef(data_array, rowvar=False)

    # Compute VIF = diagonal of inverse correlation matrix
    try:
        inv_corr = np.linalg.inv(corr_matrix)
        return np.diag(inv_corr)
    except np.linalg.LinAlgError:
        # Matrix is singular - perfect collinearity detected
        # Return None and warn user (matches old behavior)
        print(
            "ERROR: Cannot compute VIF! Design Matrix is singular because it has "
            "some perfectly correlated or duplicated columns. Using .clean() may help."
        )
        return None


def clean(
    dm: DesignMatrix,
    fill_na: Union[int, float, None] = 0,
    exclude_polys: bool = False,
    thresh: float = 0.95,
    verbose: bool = True,
) -> DesignMatrix:
    """
    Remove highly correlated columns.

    Removes columns with correlation >= threshold. Keeps first instance
    of correlated pair, drops duplicates.

    Args:
        dm: DesignMatrix instance.
        fill_na (int, float, or None): Fill NaN values before checking correlations.
            Default: 0.
        exclude_polys (bool): Skip polynomial columns from correlation check.
            Default: False.
        thresh (float): Correlation threshold (drop if abs(r) >= thresh).
            Default: 0.95.
        verbose (bool): Print dropped column names. Default: True.

    Returns:
        DesignMatrix: Cleaned matrix with highly correlated columns removed
    """
    # Check for duplicate column names
    if len(dm.columns) != len(set(dm.columns)):
        raise ValueError(
            "Duplicate column names detected. Using .clean() with duplicate "
            "columns is not supported as it can produce unexpected results."
        )

    # Start with a copy
    result = dm

    # Fill NaN if requested
    if fill_na is not None:
        result = result.fillna(fill_na)

    # Determine which columns to check for correlation
    if exclude_polys:
        cols_to_check = result._get_data_columns(exclude_polys=True)
    else:
        cols_to_check = list(result.columns)

    if len(cols_to_check) <= 1:
        if verbose:
            print("Only 1 column to check...skipping")
        return result

    # Compute pairwise correlations and identify columns to drop
    keep = []
    remove = []

    # Convert to numpy for pairwise correlation computation
    # NECESSARY: More efficient than Polars for this operation
    subset_df = result._df.select(cols_to_check)
    data_array = subset_df.to_numpy()

    # Check each pair of columns
    for i in range(len(cols_to_check)):
        col_i = cols_to_check[i]
        col_i_data = data_array[:, i]

        for j in range(i + 1, len(cols_to_check)):
            col_j = cols_to_check[j]
            col_j_data = data_array[:, j]

            # Skip if already marked for removal or keeping
            if col_j in keep or col_j in remove:
                continue

            # Check for constant arrays (avoid correlation warnings)
            if np.var(col_i_data) == 0 or np.var(col_j_data) == 0:
                r = 0.0
            else:
                # Compute correlation
                r = np.abs(np.corrcoef(col_i_data, col_j_data)[0, 1])

            # Mark for removal if correlation exceeds threshold
            if r >= thresh and col_i not in keep and col_i not in remove:
                if verbose:
                    print(
                        f"{col_i} and {col_j} correlated at {r:.2f} which is >= "
                        f"threshold of {thresh}. Dropping {col_j}"
                    )
                keep.append(col_i)
                remove.append(col_j)

    # Drop correlated columns
    if remove:
        # Drop from DataFrame
        new_df = result._df.drop(remove)

        # Update polys metadata
        new_polys = [p for p in result.polys if p not in remove]

        # Return cleaned matrix
        return result._copy_with(new_df, polys=new_polys)
    else:
        if verbose:
            print("Dropping columns not needed...skipping")
        return result


def details(dm: DesignMatrix) -> str:
    """
    Return human-readable metadata summary.

    Args:
        dm: DesignMatrix instance.

    Returns:
        str: Formatted string showing sampling_freq, shape, convolved columns,
            and polynomial columns.
    """
    lines = [
        f"DesignMatrix(sampling_freq={dm.sampling_freq}, shape={dm.shape})",
    ]

    if dm.convolved:
        lines.append(f"  convolved: {dm.convolved}")

    if dm.polys:
        lines.append(f"  polys: {dm.polys}")

    return "\n".join(lines)
