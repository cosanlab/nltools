"""
BrainCollection transform functions.

Extracted from BrainCollection methods — each function takes a BrainCollection
as its first argument (``bc``) instead of ``self``.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import nibabel as nib
    from . import BrainCollection


def map_collection(
    bc: "BrainCollection",
    fn: Callable,
    axis: int | str = 0,
    n_jobs: int = 1,
    show_progress: bool = True,
) -> "BrainCollection":
    """
    Apply function across specified axis.

    This is the general-purpose transformation method. For common operations,
    use convenience methods like standardize(), smooth(), etc.

    Args:
        bc: BrainCollection to transform.
        fn: Function to apply. Signature depends on axis:
            - axis=0: fn(BrainData) -> BrainData (per image)
            - axis=1: fn(BrainData) -> BrainData (per timepoint slice)
            - axis=2: fn(ndarray[n_obs]) -> ndarray (per voxel timeseries)
        axis: Axis to iterate over:
            - 0 or 'images': Apply fn to each image independently
            - 1 or 'time': Apply fn to each timepoint across images
            - 2 or 'voxels': Apply fn to each voxel timeseries per image
        n_jobs: Number of parallel jobs. -1 for all cores. Default 1.
        show_progress: Show tqdm progress bar. Default True.

    Returns:
        BrainCollection with transformed data.

    Examples:
        >>> # Per-image operation
        >>> bc.map(lambda bd: bd.standardize())

        >>> # Per-voxel timeseries (e.g., detrend each voxel)
        >>> from scipy.signal import detrend
        >>> bc.map(detrend, axis=2)

        >>> # Parallel processing
        >>> bc.map(expensive_fn, n_jobs=-1)
    """
    axis = bc._normalize_axis(axis)

    if axis == 0:
        return map_axis0(bc, fn, n_jobs, show_progress)
    elif axis == 1:
        return map_axis1(bc, fn, n_jobs, show_progress)
    elif axis == 2:
        return map_axis2(bc, fn, n_jobs, show_progress)
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, or 2.")


def map_axis0(
    bc: "BrainCollection",
    fn: Callable,
    n_jobs: int,
    show_progress: bool,
) -> "BrainCollection":
    """Map function over images (axis=0)."""
    from joblib import Parallel, delayed
    from nltools.utils import attempt_to_import
    from . import BrainCollection

    tqdm = attempt_to_import("tqdm", "tqdm")

    indices = range(bc.n_images)

    if n_jobs == 1:
        # Sequential processing
        if show_progress and tqdm is not None:
            indices = tqdm.tqdm(indices, desc="Mapping over images")

        results = []
        for i in indices:
            bd = bc._load_item(i)
            results.append(fn(bd))
    else:
        # Parallel processing
        def _process(i):
            bd = bc._load_item(i)
            return fn(bd)

        if show_progress and tqdm is not None:
            indices = tqdm.tqdm(indices, desc="Mapping over images")

        results = Parallel(n_jobs=n_jobs)(delayed(_process)(i) for i in indices)

    return BrainCollection(results, mask=bc._mask, metadata=bc._metadata)


def map_axis1(
    bc: "BrainCollection",
    fn: Callable,
    n_jobs: int,
    show_progress: bool,
) -> "BrainCollection":
    """Map function over timepoints (axis=1)."""
    from ..braindata import BrainData
    from nltools.utils import attempt_to_import
    from . import BrainCollection

    tqdm = attempt_to_import("tqdm", "tqdm")

    # Ensure all sample counts known and uniform
    for i in range(len(bc)):
        if bc._sample_counts[i] is None:
            bc._load_item(i)

    unique_counts = set(bc._sample_counts)
    if len(unique_counts) > 1:
        raise ValueError(
            f"map(axis=1) requires uniform observation counts. "
            f"Found: {sorted(unique_counts)}"
        )

    n_obs = bc._sample_counts[0]
    indices = range(n_obs)

    if show_progress and tqdm is not None:
        indices = tqdm.tqdm(indices, desc="Mapping over timepoints")

    # For each timepoint, create a BrainCollection slice and apply fn
    results_per_t = []
    for t in indices:
        # Get timepoint slice: bc[:, t] returns BrainCollection with 1 obs each
        t_slice = bc[:, t]
        result = fn(t_slice)
        results_per_t.append(result)

    # results_per_t is list of BrainData (one per timepoint)
    # Need to reassemble into images
    # Each result should be a BrainData with shape (n_voxels,) or similar
    # Stack them back into (n_obs, n_voxels) per image

    # If fn returns BrainData, stack across timepoints
    if isinstance(results_per_t[0], BrainData):
        # Reassemble: each image gets data from all timepoints
        new_items = []
        for img_idx in range(bc.n_images):
            img_data = []
            for t in range(n_obs):
                # results_per_t[t] is result for timepoint t
                # If it's a single BrainData (reduced), extract scalar per image
                if hasattr(results_per_t[t], "data"):
                    img_data.append(results_per_t[t].data)
            stacked = np.vstack(img_data) if len(img_data) > 1 else img_data[0]
            new_bd = BrainData(mask=bc._mask)
            new_bd.data = stacked
            new_items.append(new_bd)
        return BrainCollection(new_items, mask=bc._mask, metadata=bc._metadata)
    else:
        raise TypeError(
            f"map(axis=1) function must return BrainData, got {type(results_per_t[0])}"
        )


def map_axis2(
    bc: "BrainCollection",
    fn: Callable,
    n_jobs: int,
    show_progress: bool,
) -> "BrainCollection":
    """Map function over voxels (axis=2) per image."""
    from ..braindata import BrainData
    from nltools.utils import attempt_to_import
    from . import BrainCollection

    tqdm = attempt_to_import("tqdm", "tqdm")

    indices = range(bc.n_images)
    if show_progress and tqdm is not None:
        indices = tqdm.tqdm(indices, desc="Mapping over voxels")

    results = []
    for i in indices:
        bd = bc._load_item(i)
        data = bd.data
        if data.ndim == 1:
            data = data[np.newaxis, :]

        # Apply fn to each voxel's timeseries (each column)
        # fn receives (n_obs,) array, returns (n_obs,) or scalar
        transformed_cols = []
        for v in range(data.shape[1]):
            transformed_cols.append(fn(data[:, v]))

        transformed = np.column_stack(transformed_cols)

        new_bd = BrainData(mask=bc._mask)
        new_bd.data = (
            transformed.squeeze() if transformed.shape[0] == 1 else transformed
        )
        results.append(new_bd)

    return BrainCollection(results, mask=bc._mask, metadata=bc._metadata)


def filter_collection(
    bc: "BrainCollection",
    predicate: "Callable | list | np.ndarray",
) -> "BrainCollection":
    """
    Filter collection by predicate.

    Args:
        bc: BrainCollection to filter.
        predicate: Filter condition. Can be:
            - callable: fn(BrainData) -> bool
            - list/ndarray: Boolean mask of length n_images
            - pd.Series: Boolean series (index ignored)

    Returns:
        BrainCollection with subset of images matching predicate.

    Examples:
        >>> # Filter by callable
        >>> bc.filter(lambda bd: bd.data.mean() > 0)

        >>> # Filter by boolean mask
        >>> mask = [True, False, True]
        >>> bc.filter(mask)

        >>> # Filter by metadata condition
        >>> bc.filter(bc.metadata['group'] == 'control')
    """
    import pandas as pd

    if isinstance(predicate, pd.Series):
        mask = predicate.values.astype(bool)
    elif isinstance(predicate, (list, np.ndarray)):
        mask = np.asarray(predicate, dtype=bool)
    elif callable(predicate):
        # Apply predicate to each image
        mask_list: list[bool] = []
        for i in range(bc.n_images):
            bd = bc._load_item(i)
            mask_list.append(bool(predicate(bd)))
        mask = np.array(mask_list)
    else:
        mask = np.asarray(predicate, dtype=bool)

    if len(mask) != bc.n_images:
        raise ValueError(
            f"Predicate length ({len(mask)}) must match n_images ({bc.n_images})"
        )

    indices = np.where(mask)[0].tolist()
    return bc._subset(indices)


def align(
    bc: "BrainCollection",
    method: str = "procrustes",
    scheme: str = "searchlight",
    radius_mm: float = 10.0,
    parcellation: "nib.Nifti1Image | None" = None,
    n_features: int | None = None,
    n_iter: int = 3,
    parallel: str | None = "cpu",
    n_jobs: int = -1,
    return_model: bool = False,
    show_progress: bool = True,
) -> "BrainCollection | tuple[BrainCollection, object]":
    """
    Align subjects using local functional alignment.

    Performs neighborhood-based functional alignment across subjects using
    LocalAlignment. Each subject's data is aligned to a common template space
    using local transforms learned within searchlight spheres or parcels.

    Args:
        bc: BrainCollection to align.
        method: Alignment method. Options:
            - 'procrustes': Orthogonal Procrustes (default, preserves dimensions)
            - 'srm': Shared Response Model (dimensionality reduction)
            - 'hyperalignment': Hyperalignment (iterative Procrustes)
        scheme: Spatial scheme. Options:
            - 'searchlight': Overlapping spheres with center-only aggregation
            - 'piecewise': Non-overlapping parcels (requires parcellation)
        radius_mm: Sphere radius in millimeters for searchlight scheme.
        parcellation: Parcellation image for piecewise scheme (required if
            scheme='piecewise').
        n_features: Number of features for SRM. None uses full dimensions.
        n_iter: Number of iterations for alignment refinement.
        parallel: Parallelization mode. Options:
            - None: Single-threaded
            - 'cpu': CPU parallelization with joblib
            - 'gpu': GPU acceleration via PyTorch
        n_jobs: Number of parallel jobs for CPU mode (-1 = auto).
        return_model: If True, return (aligned_collection, model) tuple for
            fit/transform workflow with new data.
        show_progress: Show progress bar during fitting.

    Returns:
        BrainCollection with aligned data. If return_model=True, returns
        tuple of (aligned_collection, LocalAlignment_model).

    Examples:
        >>> # Basic searchlight alignment
        >>> aligned_bc = bc.align(method='procrustes', radius_mm=10.0)

        >>> # Piecewise alignment with parcellation
        >>> aligned_bc = bc.align(
        ...     scheme='piecewise',
        ...     parcellation=parcellation_img,
        ...     method='srm',
        ...     n_features=50
        ... )

        >>> # Fit/transform workflow for train/test split
        >>> aligned_train, model = train_bc.align(return_model=True)
        >>> aligned_test = model.transform(test_data_list)

        >>> # GPU-accelerated alignment
        >>> aligned_bc = bc.align(parallel='gpu')

    Notes:
        Based on Bazeille et al. 2021 "An empirical evaluation of functional
        alignment using inter-subject decoding". Center-only aggregation is
        used for searchlight to preserve local orthogonality of transforms.

    See Also:
        nltools.algorithms.alignment.LocalAlignment: Underlying alignment class.
    """
    from nltools.algorithms.alignment import LocalAlignment
    from ..braindata import BrainData
    from . import BrainCollection

    # Validate inputs
    if scheme == "piecewise" and parcellation is None:
        raise ValueError("parcellation is required for piecewise scheme")

    # Extract data from collection as list of (n_voxels, n_samples) arrays
    # BrainData.data is (n_samples, n_voxels), LocalAlignment expects (n_voxels, n_samples)
    data_list = []
    for i in range(bc.n_images):
        bd = bc._load_item(i)
        data = bd.data
        if data.ndim == 1:
            data = data[np.newaxis, :]  # (1, n_voxels)
        # Transpose: (n_samples, n_voxels) -> (n_voxels, n_samples)
        data_list.append(data.T)

    # Create and fit LocalAlignment
    la = LocalAlignment(
        scheme=scheme,
        method=method,
        radius_mm=radius_mm,
        parcellation=parcellation,
        n_features=n_features,
        n_iter=n_iter,
        parallel=parallel,
        n_jobs=n_jobs,
    )

    # Fit and transform
    aligned_data = la.fit_transform(data_list, bc._mask)

    # Convert aligned data back to BrainCollection
    # aligned_data is list of (n_voxels, n_samples), need (n_samples, n_voxels)
    aligned_items = []
    for i, aligned in enumerate(aligned_data):
        # Transpose back: (n_voxels, n_samples) -> (n_samples, n_voxels)
        aligned_transposed = aligned.T
        new_bd = BrainData(mask=bc._mask)
        new_bd.data = (
            aligned_transposed.squeeze()
            if aligned_transposed.shape[0] == 1
            else aligned_transposed
        )
        aligned_items.append(new_bd)

    aligned_collection = BrainCollection(
        aligned_items, mask=bc._mask, metadata=bc._metadata.copy()
    )

    if return_model:
        return aligned_collection, la
    return aligned_collection


def standardize(
    bc: "BrainCollection",
    axis: int = 0,
    method: str = "center",
    n_jobs: int = 1,
    show_progress: bool = True,
    verbose: bool = True,
) -> "BrainCollection":
    """
    Standardize each image.

    Delegates to BrainData.standardize() for each image.

    Args:
        bc: BrainCollection to standardize.
        axis: Axis for standardization within each image:
            - 0: Standardize across observations (time) per voxel
            - 1: Standardize across voxels per observation
        method: 'center' (subtract mean) or 'zscore' (subtract mean, divide std)
        n_jobs: Number of parallel jobs.
        show_progress: Show progress bar.
        verbose: If False, suppress sklearn numerical warnings. Default: True.

    Returns:
        BrainCollection with standardized images.

    Examples:
        >>> bc.standardize()  # Center each image across time
        >>> bc.standardize(method='zscore')  # Z-score each image
        >>> bc.standardize(axis=1)  # Standardize across voxels
    """
    return bc.map(
        lambda bd: bd.standardize(axis=axis, method=method, verbose=verbose),
        axis=0,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )


def smooth(
    bc: "BrainCollection",
    fwhm: float,
    n_jobs: int = 1,
    show_progress: bool = True,
) -> "BrainCollection":
    """
    Spatially smooth each image.

    Delegates to BrainData.smooth() for each image.

    Args:
        bc: BrainCollection to smooth.
        fwhm: Full width at half maximum of Gaussian kernel in mm.
        n_jobs: Number of parallel jobs.
        show_progress: Show progress bar.

    Returns:
        BrainCollection with smoothed images.

    Examples:
        >>> bc.smooth(fwhm=6)  # 6mm FWHM smoothing
    """
    return bc.map(
        lambda bd: bd.smooth(fwhm),
        axis=0,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )


def threshold(
    bc: "BrainCollection",
    upper: float | str | None = None,
    lower: float | str | None = None,
    binarize: bool = False,
    coerce_nan: bool = True,
    n_jobs: int = 1,
    show_progress: bool = True,
) -> "BrainCollection":
    """
    Threshold each image.

    Delegates to BrainData.threshold() for each image.

    Args:
        bc: BrainCollection to threshold.
        upper: Upper cutoff. String interpreted as percentile.
        lower: Lower cutoff. String interpreted as percentile.
        binarize: Return binary mask.
        coerce_nan: Replace NaN with 0.
        n_jobs: Number of parallel jobs.
        show_progress: Show progress bar.

    Returns:
        BrainCollection with thresholded images.

    Examples:
        >>> bc.threshold(lower=0)  # Zero out negative values
        >>> bc.threshold(upper='95%')  # Keep top 5%
        >>> bc.threshold(lower=2, binarize=True)  # Binary mask
    """
    return bc.map(
        lambda bd: bd.threshold(
            upper=upper, lower=lower, binarize=binarize, coerce_nan=coerce_nan
        ),
        axis=0,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )


def detrend(
    bc: "BrainCollection",
    method: str = "linear",
    n_jobs: int = 1,
    show_progress: bool = True,
) -> "BrainCollection":
    """
    Remove trend from each image.

    Delegates to BrainData.detrend() for each image.

    Args:
        bc: BrainCollection to detrend.
        method: 'linear' or 'constant'.
        n_jobs: Number of parallel jobs.
        show_progress: Show progress bar.

    Returns:
        BrainCollection with detrended images.

    Examples:
        >>> bc.detrend()  # Remove linear trend
        >>> bc.detrend(method='constant')  # Remove mean only
    """
    return bc.map(
        lambda bd: bd.detrend(method=method),
        axis=0,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )
