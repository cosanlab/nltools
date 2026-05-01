"""Aggregation helpers for BrainCollection.

Streaming axis-wise reductions (mean/std/var/sum/min/max/median) extracted
from BrainCollection. All BrainCollection aggregation methods converted to
functions taking ``bc`` as first argument.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nltools.utils import attempt_to_import

if TYPE_CHECKING:
    from nltools.data.braindata import BrainData
    from nltools.data.collection import BrainCollection

tqdm = attempt_to_import("tqdm", "tqdm")


def aggregate_axis0(
    bc: BrainCollection,
    func: str,
    batch_size: int | None = None,
) -> BrainData:
    """Aggregate across images (axis=0) using streaming algorithm."""
    from nltools.data.braindata import BrainData

    # Ensure all sample counts are known
    for i in range(len(bc)):
        if bc._sample_counts[i] is None:
            bc._load_item(i)

    # Check uniform observation counts
    unique_counts = set(bc._sample_counts)
    if len(unique_counts) > 1:
        raise ValueError(
            f"Cannot aggregate axis=0: images have variable observation counts "
            f"{sorted(unique_counts)}. Use apply() for per-image operations."
        )

    n_obs = bc._sample_counts[0]

    if func == "mean":
        # Welford's online mean algorithm
        running_mean = np.zeros((n_obs, bc.n_voxels))

        iterator = range(bc.n_images)
        if tqdm is not None:
            iterator = tqdm.tqdm(iterator, desc="Computing mean (axis=0)")

        for count, i in enumerate(iterator, start=1):
            bd = bc._load_item(i)
            data = bd.data if bd.data.ndim == 2 else bd.data[np.newaxis, :]
            delta = data - running_mean
            running_mean += delta / count

        result = BrainData(mask=bc._mask)
        result.data = running_mean
        return result

    if func == "sum":
        running_sum = np.zeros((n_obs, bc.n_voxels))
        for i in range(bc.n_images):
            bd = bc._load_item(i)
            data = bd.data if bd.data.ndim == 2 else bd.data[np.newaxis, :]
            running_sum += data
        result = BrainData(mask=bc._mask)
        result.data = running_sum
        return result

    if func in ("std", "var"):
        # Welford's online variance algorithm
        running_mean = np.zeros((n_obs, bc.n_voxels))
        running_m2 = np.zeros((n_obs, bc.n_voxels))
        count = 0

        for i in range(bc.n_images):
            bd = bc._load_item(i)
            data = bd.data if bd.data.ndim == 2 else bd.data[np.newaxis, :]
            count += 1
            delta = data - running_mean
            running_mean += delta / count
            delta2 = data - running_mean
            running_m2 += delta * delta2

        variance = running_m2 / max(count - 1, 1)  # Sample variance
        result = BrainData(mask=bc._mask)
        result.data = np.sqrt(variance) if func == "std" else variance
        return result

    if func in ("min", "max"):
        agg_func = np.minimum if func == "min" else np.maximum
        running = None
        for i in range(bc.n_images):
            bd = bc._load_item(i)
            data = bd.data if bd.data.ndim == 2 else bd.data[np.newaxis, :]
            if running is None:
                running = data.copy()
            else:
                running = agg_func(running, data)
        result = BrainData(mask=bc._mask)
        result.data = running
        return result

    if func == "median":
        # Median requires all data in memory
        tensor = bc.to_tensor()
        median_data = np.median(tensor, axis=0)
        result = BrainData(mask=bc._mask)
        result.data = median_data
        return result

    raise ValueError(f"Unknown aggregation function: {func}")


def aggregate_axis1(bc: BrainCollection, func: str) -> BrainCollection:
    """Aggregate across observations (axis=1) per image."""
    from nltools.data.braindata import BrainData
    from nltools.data.collection import BrainCollection

    agg_func = getattr(np, func)
    new_items = []

    iterator = range(bc.n_images)
    if tqdm is not None:
        iterator = tqdm.tqdm(iterator, desc=f"Computing {func} (axis=1)")

    for i in iterator:
        bd = bc._load_item(i)
        data = bd.data if bd.data.ndim == 2 else bd.data[np.newaxis, :]
        agg_data = agg_func(data, axis=0)
        new_bd = BrainData(mask=bc._mask)
        new_bd.data = agg_data
        new_items.append(new_bd)

    return BrainCollection(new_items, mask=bc._mask, metadata=bc._metadata)


def aggregate_axis2(bc: BrainCollection, func: str) -> np.ndarray:
    """Aggregate across voxels (axis=2) -> numpy array."""
    agg_func = getattr(np, func)
    results = []

    for i in range(bc.n_images):
        bd = bc._load_item(i)
        data = bd.data if bd.data.ndim == 2 else bd.data[np.newaxis, :]
        agg_data = agg_func(data, axis=1)  # Aggregate over voxels
        results.append(agg_data)

    return np.array(results)


def aggregate(
    bc: BrainCollection,
    func: str,
    axis: int | str | tuple[int, ...],
    batch_size: int | None = None,
) -> BrainData | BrainCollection | np.ndarray:
    """Dispatch aggregation to appropriate axis handler."""
    axis = bc._normalize_axis(axis)

    # Handle tuple of axes
    if isinstance(axis, tuple):
        # Sort axes to process in order
        axes = sorted(axis)
        result = bc
        for ax in reversed(axes):
            # After each reduction, axis indices shift
            result = aggregate(result, func, ax, batch_size)
        return result

    if axis == 0:
        return aggregate_axis0(bc, func, batch_size)
    if axis == 1:
        return aggregate_axis1(bc, func)
    if axis == 2:
        return aggregate_axis2(bc, func)
    raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, 2, or tuple.")
