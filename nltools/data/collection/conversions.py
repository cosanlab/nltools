"""In-memory shape conversions for BrainCollection.

Reshape and batch helpers (to_tensor / to_list / to_stacked / iter_batches)
extracted from BrainCollection. All BrainCollection conversion methods
converted to functions taking ``bc`` as first argument.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np

from nltools.utils import attempt_to_import

if TYPE_CHECKING:
    from nltools.data.braindata import BrainData
    from nltools.data.collection import BrainCollection

tqdm = attempt_to_import("tqdm", "tqdm")


def to_tensor(
    bc: BrainCollection,
    batch_size: int | None = None,
) -> np.ndarray | Generator[np.ndarray, None, None]:
    """Convert collection to numpy array (n_images, n_obs, n_voxels)."""
    # First, ensure all sample counts are known
    for i in range(len(bc)):
        if bc._sample_counts[i] is None:
            bc._load_item(i)

    # Check for uniform observation counts
    unique_counts = set(bc._sample_counts)
    if len(unique_counts) > 1:
        raise ValueError(
            f"Cannot convert to tensor: images have variable observation counts "
            f"{sorted(unique_counts)}. Use to_list() instead."
        )

    n_obs = bc._sample_counts[0]

    if batch_size is not None:
        return to_tensor_batched(bc, batch_size, n_obs)

    # Full tensor
    tensor = np.zeros((bc.n_images, n_obs, bc.n_voxels))
    for i in range(bc.n_images):
        bd = bc._load_item(i)
        data = bd.data
        if data.ndim == 1:
            data = data[np.newaxis, :]
        tensor[i] = data

    return tensor


def to_tensor_batched(
    bc: BrainCollection, batch_size: int, n_obs: int
) -> Generator[np.ndarray, None, None]:
    """Generator yielding batches of the tensor."""
    for start in range(0, bc.n_images, batch_size):
        end = min(start + batch_size, bc.n_images)
        batch_tensor = np.zeros((end - start, n_obs, bc.n_voxels))
        for i, idx in enumerate(range(start, end)):
            bd = bc._load_item(idx)
            data = bd.data
            if data.ndim == 1:
                data = data[np.newaxis, :]
            batch_tensor[i] = data
        yield batch_tensor


def to_list(bc: BrainCollection) -> list[BrainData]:
    """Return list of BrainData objects, loading any lazy items."""
    return [bc._load_item(i) for i in range(len(bc))]


def to_stacked(bc: BrainCollection) -> BrainData:
    """Stack all images into a single BrainData (n_total_obs, n_voxels)."""
    from nltools.data.braindata import BrainData

    all_data = []
    for i in range(len(bc)):
        bd = bc._load_item(i)
        data = bd.data
        if data.ndim == 1:
            data = data[np.newaxis, :]
        all_data.append(data)

    stacked_data = np.vstack(all_data)

    result = BrainData(mask=bc._mask)
    result.data = stacked_data
    return result


def iter_batches(
    bc: BrainCollection,
    batch_size: int,
    axis: int = 0,
    progress_bar: bool = False,
) -> Generator[BrainCollection, None, None]:
    """Iterate the collection in batches along the specified axis."""
    from nltools.data.collection import BrainCollection
    from nltools.data.collection.indexing import subset

    axis = bc._normalize_axis(axis)

    if axis == 0:
        # Batch over images
        n_batches = int(np.ceil(bc.n_images / batch_size))
        iterator = range(n_batches)

        if progress_bar and tqdm is not None:
            iterator = tqdm.tqdm(iterator, desc="Batching images", total=n_batches)

        for batch_idx in iterator:
            start = batch_idx * batch_size
            end = min(start + batch_size, bc.n_images)
            yield subset(bc, list(range(start, end)))

    elif axis == 1:
        # Batch over observations - requires uniform obs counts
        for i in range(len(bc)):
            if bc._sample_counts[i] is None:
                bc._load_item(i)

        unique_counts = set(bc._sample_counts)
        if len(unique_counts) > 1:
            raise ValueError(
                "Cannot batch over observations with variable counts. "
                f"Found counts: {sorted(unique_counts)}"
            )

        n_obs = bc._sample_counts[0]
        n_batches = int(np.ceil(n_obs / batch_size))
        iterator = range(n_batches)

        if progress_bar and tqdm is not None:
            iterator = tqdm.tqdm(
                iterator, desc="Batching observations", total=n_batches
            )

        for batch_idx in iterator:
            start = batch_idx * batch_size
            end = min(start + batch_size, n_obs)
            # Slice observations for each image
            batch = bc[:, start:end]
            assert isinstance(batch, BrainCollection)
            yield batch

    else:
        raise ValueError(f"Cannot batch over axis {axis}. Use axis=0 or axis=1.")
