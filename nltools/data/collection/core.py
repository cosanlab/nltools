"""Stateless helpers for BrainCollection.

Constants and pure-function helpers shared across the collection submodules
(metadata coercion, axis-name normalization). Instance-touching helpers like
``_resolve_mask`` and ``_load_item`` remain on the BrainCollection class
itself, since they read/mutate the collection's internal state across many
call sites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    import pandas as pd


# Axis name mapping for intuitive access on BrainCollection
AXIS_NAMES = {
    "images": 0,
    "subjects": 0,
    "image": 0,
    "subject": 0,
    "observations": 1,
    "time": 1,
    "timepoints": 1,
    "obs": 1,
    "voxels": 2,
    "space": 2,
    "spatial": 2,
}


def coerce_metadata(
    metadata: pl.DataFrame | pd.DataFrame | dict | None,
    n_items: int,
) -> pl.DataFrame:
    """Coerce metadata input to a polars DataFrame.

    Accepts polars/pandas DataFrame, dict-of-columns, or None (→ empty frame
    of length ``n_items``). Pandas is taken at the boundary as a convenience
    affordance; internal state is always polars.
    """
    if metadata is None:
        return pl.DataFrame()

    if isinstance(metadata, pl.DataFrame):
        out = metadata
    elif isinstance(metadata, dict):
        out = pl.DataFrame(metadata)
    else:
        try:
            import pandas as pd
        except ImportError:
            pd = None
        if pd is not None and isinstance(metadata, pd.DataFrame):
            out = pl.DataFrame(
                {str(c): metadata[c].to_numpy() for c in metadata.columns}
            )
        else:
            raise TypeError(
                "metadata must be a polars/pandas DataFrame, dict, or None. "
                f"Received {type(metadata).__name__}"
            )

    if not out.is_empty() and out.height != n_items:
        raise ValueError(
            f"metadata length ({out.height}) must match items length ({n_items})"
        )
    return out


def normalize_axis(
    axis: int | str | tuple[int | str, ...],
) -> int | tuple[int, ...]:
    """Convert axis name (or tuple of names) to integer axis index."""
    if isinstance(axis, str):
        if axis.lower() not in AXIS_NAMES:
            raise ValueError(
                f"Unknown axis name: {axis}. Valid names: {list(AXIS_NAMES.keys())}"
            )
        return AXIS_NAMES[axis.lower()]
    if isinstance(axis, tuple):
        normalized: list[int] = []
        for a in axis:
            result = normalize_axis(a)
            if isinstance(result, tuple):
                raise ValueError("Nested tuple axes are not supported")
            normalized.append(result)
        return tuple(normalized)
    return axis
