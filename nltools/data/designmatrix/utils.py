"""Shared helpers for DesignMatrix submodules.

These are internal utilities used by the facade and submodules — not part of the
public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from nltools.data.designmatrix import DesignMatrix


def copy_with(
    dm: DesignMatrix,
    new_df: pl.DataFrame,
    **metadata_updates,
) -> DesignMatrix:
    """
    Create new DesignMatrix with updated data/metadata.

    This is the core pattern for immutable transformations.
    All methods that transform data should use this helper.

    Args:
        dm: Source DesignMatrix whose metadata to copy.
        new_df (pl.DataFrame): New underlying data.
        **metadata_updates: Metadata attributes to override
            (e.g., convolved=['stim']).

    Returns:
        DesignMatrix: New DesignMatrix with updated data and metadata.
    """
    from nltools.data.designmatrix import DesignMatrix

    # Start with current metadata
    metadata = get_metadata(dm)

    # Apply updates
    metadata.update(metadata_updates)

    # Create new DesignMatrix
    new_dm = DesignMatrix(
        new_df,
        sampling_freq=metadata["sampling_freq"],
        convolved=metadata["convolved"],
        polys=metadata["polys"],
    )
    new_dm.multi = metadata["multi"]

    return new_dm


def get_metadata(dm: DesignMatrix) -> dict:
    """Extract metadata as dict (for copying).

    Args:
        dm: DesignMatrix instance.

    Returns:
        dict: Dictionary with keys 'sampling_freq', 'convolved', 'polys', 'multi'.
    """
    return {
        "sampling_freq": dm.sampling_freq,
        "convolved": dm.convolved.copy(),
        "polys": dm.polys.copy(),
        "multi": dm.multi,
    }


def get_data_columns(dm: DesignMatrix, exclude_polys: bool = True) -> list[str]:
    """
    Get column names, optionally excluding polynomials.

    This helper reduces code duplication across methods that need to
    distinguish between data columns and polynomial/nuisance columns.

    Args:
        dm: DesignMatrix instance.
        exclude_polys (bool, default=True): If True, exclude polynomial/nuisance
            columns from the result.

    Returns:
        list of str: Column names (excluding polys if requested).
    """
    if exclude_polys and dm.polys:
        return [col for col in dm.columns if col not in dm.polys]
    return list(dm.columns)
