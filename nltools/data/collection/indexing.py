"""Indexing helpers for BrainCollection.

Numpy-style multi-dimensional indexing extracted from BrainCollection.
All BrainCollection indexing methods converted to functions taking ``bc`` as
first argument.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nltools.data.braindata import BrainData
    from nltools.data.collection import BrainCollection


def getitem(bc: BrainCollection, key) -> BrainData | BrainCollection:
    """Numpy-style 3D indexing dispatcher.

    Supports:
        bc[i]           -> BrainData (obs, voxels) - single image
        bc[i, j]        -> BrainData (voxels,) - single image, single obs
        bc[i, j, k]     -> scalar or array - single image, single obs, voxel slice
        bc[slice]       -> BrainCollection - subset of images
        bc[:, slice]    -> BrainCollection - all images, sliced observations
        bc['sub-01']    -> BrainData - by metadata lookup (if 'subject' column)

    Args:
        bc: BrainCollection instance.
        key: Index, slice, list, tuple, or string.

    Returns:
        BrainData for single-image access, BrainCollection for multi-image.
    """
    # String key: lookup by metadata
    if isinstance(key, str):
        return getitem_by_metadata(bc, key)

    # Tuple: multi-dimensional indexing
    if isinstance(key, tuple):
        return getitem_multidim(bc, key)

    # Single index
    if isinstance(key, int):
        return bc._load_item(key)

    # Slice or list
    if isinstance(key, slice):
        indices = range(*key.indices(len(bc)))
        return subset(bc, list(indices))

    if isinstance(key, (list, np.ndarray)):
        return subset(bc, list(key))

    raise TypeError(f"Invalid index type: {type(key).__name__}")


def getitem_by_metadata(bc: BrainCollection, key: str) -> BrainData:
    """Get item by metadata value (e.g., subject ID)."""
    for col in ["subject", "subject_id", "sub", "id"]:
        if col in bc._metadata.columns:
            mask = (bc._metadata[col] == key).to_numpy()
            match_idx = np.flatnonzero(mask)
            if len(match_idx) == 1:
                return bc._load_item(int(match_idx[0]))
            if len(match_idx) > 1:
                raise KeyError(
                    f"Multiple images match '{key}' in column '{col}'. "
                    "Use integer indexing or more specific key."
                )
    raise KeyError(
        f"No image found for key '{key}'. "
        "Ensure metadata has 'subject' column or use integer indexing."
    )


def getitem_multidim(bc: BrainCollection, key: tuple) -> BrainData | BrainCollection:
    """Handle multi-dimensional indexing: bc[i, j] or bc[i, j, k]."""
    from nltools.data.braindata import BrainData
    from nltools.data.collection import BrainCollection

    if len(key) == 0:
        raise IndexError("Empty index")

    # First dimension: images
    img_key = key[0]

    # Single image case
    if isinstance(img_key, int):
        bd = bc._load_item(img_key)

        if len(key) == 1:
            return bd

        # Observation indexing
        obs_key = key[1]
        if isinstance(obs_key, int):
            # Single observation
            if bd.data.ndim == 1:
                if obs_key != 0:
                    raise IndexError(
                        f"Observation index {obs_key} out of range for single image"
                    )
                sliced_data = bd.data
            else:
                sliced_data = bd.data[obs_key]

            # Return as BrainData with single observation
            result = BrainData(mask=bd.mask)
            result.data = sliced_data
            return result

        if isinstance(obs_key, slice):
            # Slice observations
            if bd.data.ndim == 1:
                sliced_data = bd.data[np.newaxis, :][obs_key]
            else:
                sliced_data = bd.data[obs_key]

            result = BrainData(mask=bd.mask)
            result.data = sliced_data
            return result

        raise TypeError(f"Invalid observation index type: {type(obs_key)}")

    # Multiple images case
    if isinstance(img_key, slice):
        indices = list(range(*img_key.indices(len(bc))))
    elif isinstance(img_key, (list, np.ndarray)):
        indices = list(img_key)
    else:
        raise TypeError(f"Invalid image index type: {type(img_key)}")

    # Create subset collection
    sub = subset(bc, indices)

    if len(key) == 1:
        return sub

    # Apply observation slicing to each image
    obs_key = key[1]
    if isinstance(obs_key, (int, slice)):

        def slice_obs(bd: BrainData) -> BrainData:
            """Slice observations from a BrainData object using the captured obs_key.

            Handles 1-D data by temporarily expanding to 2-D before indexing.
            """
            if bd.data.ndim == 1:
                data = bd.data[np.newaxis, :]
            else:
                data = bd.data

            if isinstance(obs_key, int):
                sliced = data[obs_key]
            else:
                sliced = data[obs_key]

            result = BrainData(mask=bd.mask)
            result.data = sliced
            return result

        # Apply without creating new collection
        new_items = [slice_obs(sub._load_item(i)) for i in range(len(sub))]
        return BrainCollection(new_items, mask=bc._mask, metadata=sub._metadata)

    raise TypeError(f"Invalid observation index type: {type(obs_key)}")


def subset(bc: BrainCollection, indices: list[int]) -> BrainCollection:
    """Create a new BrainCollection with subset of items."""
    from nltools.data.collection import BrainCollection

    new_items = [bc._items[i] for i in indices]
    if bc._metadata.is_empty():
        new_metadata = bc._metadata
    else:
        new_metadata = bc._metadata[list(indices)]

    # Create new collection without re-validating
    new_bc = object.__new__(BrainCollection)
    new_bc._mask = bc._mask
    new_bc._n_voxels = bc._n_voxels
    new_bc._items = new_items
    new_bc._is_loaded = [bc._is_loaded[i] for i in indices]
    new_bc._sample_counts = [bc._sample_counts[i] for i in indices]
    new_bc._metadata = new_metadata

    return new_bc
