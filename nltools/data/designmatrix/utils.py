"""Shared helpers for DesignMatrix submodules.

These are internal utilities used by the facade and submodules — not part of the
public API.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from nltools.data.designmatrix import DesignMatrix


WRAP_AS_DESIGNMATRIX = frozenset({"slice", "filter", "select"})


def df_passthrough(dm: DesignMatrix, name: str):
    """Resolve ``name`` on ``dm.data``; re-wrap DataFrame results for allowlisted methods.

    Used by ``DesignMatrix.__getattr__`` to expose polars' DataFrame API without
    duplicating every method. Row-preserving ops in ``WRAP_AS_DESIGNMATRIX`` return
    a new ``DesignMatrix`` with metadata copied via ``copy_with``; everything else
    returns the raw polars object.
    """
    attr = getattr(dm.data, name)
    if not callable(attr) or name not in WRAP_AS_DESIGNMATRIX:
        return attr

    @functools.wraps(attr)
    def wrapped(*args, **kwargs):
        result = attr(*args, **kwargs)
        if isinstance(result, pl.DataFrame):
            return copy_with(dm, result)
        return result

    return wrapped


def copy_with(
    dm: DesignMatrix,
    new_df: pl.DataFrame,
    **metadata_updates,
) -> DesignMatrix:
    """Create a new DesignMatrix with updated data and metadata.

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

    # Start with current metadata, apply caller overrides
    metadata = get_metadata(dm)
    metadata.update(metadata_updates)

    # Prune column-name metadata against the new frame so that ops which drop
    # columns (select, __getitem__ with a list) don't leave dangling references.
    new_cols = set(new_df.columns)
    metadata["convolved"] = [c for c in metadata["convolved"] if c in new_cols]
    metadata["confounds"] = [c for c in metadata["confounds"] if c in new_cols]

    new_dm = DesignMatrix(
        new_df,
        sampling_freq=metadata["sampling_freq"],
        convolved=metadata["convolved"],
        confounds=metadata["confounds"],
    )
    new_dm.multi = metadata["multi"]

    return new_dm


def get_metadata(dm: DesignMatrix) -> dict:
    """Extract metadata as dict (for copying).

    Args:
        dm: DesignMatrix instance.

    Returns:
        dict: Dictionary with keys 'sampling_freq', 'convolved', 'confounds', 'multi'.
    """
    return {
        "sampling_freq": dm.sampling_freq,
        "convolved": dm.convolved.copy(),
        "confounds": dm.confounds.copy(),
        "multi": dm.multi,
    }


def get_data_columns(dm: DesignMatrix, exclude_confounds: bool = True) -> list[str]:
    """Get column names, optionally excluding confound regressors.

    This helper reduces code duplication across methods that need to
    distinguish between experimental regressors and nuisance/confound columns
    (polynomial drift, DCT cosines, motion, etc.).

    Args:
        dm: DesignMatrix instance.
        exclude_confounds (bool, default=True): If True, exclude nuisance
            columns tracked in ``dm.confounds`` from the result.

    Returns:
        list of str: Column names (excluding confounds if requested).
    """
    if exclude_confounds and dm.confounds:
        return [col for col in dm.columns if col not in dm.confounds]
    return list(dm.columns)
