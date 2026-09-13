"""Analysis operations on `BrainData`.

Functions for similarity, distance, masking, ROI extraction, filtering,
thresholding, decomposition, alignment, smoothing, and related operations.
Each takes a `BrainData` as its first argument; the corresponding
`BrainData` methods delegate here.
"""

from pathlib import Path

import numpy as np
import polars as pl

from .utils import _result_from_array, _result_from_rows, _result_with_mask


def _subset_mask(bd, columns):
    """Build a spatial mask for selected columns of a masked array."""
    from nilearn.masking import unmask

    return unmask(np.asarray(columns, dtype=np.uint8), bd.mask)


def _mask_support(bd):
    """Count the voxels a BrainData object's mask keeps."""
    return int(np.count_nonzero(bd.mask.get_fdata() > 0))


def _brain_result(source, values, name, *, rows):
    """Wrap an alignment value as a `BrainData` once its voxel axis checks out.

    An alignment value may only become a `BrainData` when its columns are a
    voxel axis that matches the mask it will carry. Checking here keeps a
    mismatch from becoming an object whose `to_nifti` fails much later.

    Args:
        source (BrainData): Object supplying the mask and metadata.
        values (np.ndarray): Result data whose last axis must be voxels.
        name (str): Result key, used in the error message.
        rows (str): Row-metadata policy, ``'preserve'`` or ``'clear'``.

    Returns:
        BrainData: Independently owned result carrying ``source``'s mask.

    Raises:
        ValueError: If the column count differs from the mask support.
    """
    values = np.asarray(values)
    support = _mask_support(source)
    if values.shape[-1] != support:
        raise ValueError(
            f"align() cannot return {name!r} as a BrainData: its shape is "
            f"{values.shape}, so it has {values.shape[-1]} columns, but the "
            f"mask supports {support} voxels. Align data that shares the "
            f"source voxel axis."
        )
    return _result_from_array(source, values, rows=rows)


def _aligned_array(value):
    """Return the array behind an alignment value that may be a `BrainData`."""
    return value if isinstance(value, np.ndarray) else value.data


def _check_masks(bd, image):
    """Ensure two datasets use compatible masks, creating a union mask if needed.

    Args:
        bd (BrainData): Reference dataset.
        image (BrainData): Dataset whose mask is compared with ``bd``'s.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(data, image_data)`` arrays sampled on
            a shared mask.
    """
    from nilearn.masking import apply_mask, intersect_masks

    if np.sum(bd.mask.get_fdata() == 1) != np.sum(image.mask.get_fdata() == 1):
        new_mask = intersect_masks(
            [bd.mask, image.mask],
            threshold=1,
            connected=False,
        )
        data2 = apply_mask(bd.to_nifti(), new_mask)
        image2 = apply_mask(image.to_nifti(), new_mask)
    else:
        data2 = bd.data
        image2 = image.data
    return data2, image2


def _similarity(bd, image, metric="correlation"):
    """Calculate similarity to a single BrainData or nibabel image.

    Args:
        bd (BrainData): Dataset to compare.
        image (BrainData | Nifti1Image): Image to evaluate similarity against.
        metric (str): Similarity metric, one of ``'correlation'``, ``'pearson'``,
            ``'rank_correlation'``, ``'spearman'``, ``'dot_product'``, or
            ``'cosine'``.

    Returns:
        np.ndarray: Similarity values.
    """
    from nltools.algorithms.similarity import compute_similarity
    from .utils import _check_brain_data

    supported_metrics = [
        "correlation",
        "pearson",
        "rank_correlation",
        "spearman",
        "dot_product",
        "cosine",
    ]
    if metric not in supported_metrics:
        raise ValueError(f"metric must be one of {supported_metrics}")

    image = _check_brain_data(image)
    data2, image2 = _check_masks(bd, image)

    # Delegate to functional core (stats.py)
    return compute_similarity(data2, image2, metric=metric)


def _distance(  # nosemgrep: kwargs-internal-forwarding  # forwards to scipy.spatial.distance.cdist
    bd,
    metric="euclidean",
    *,
    spatial_scale: str = "whole_brain",
    roi_mask=None,
    radius: float = 10.0,
    **kwargs,
):
    """Calculate distance between images within a BrainData() instance.

    Args:
        bd (BrainData): Dataset whose images are compared.
        metric (str): Any distance metric supported by
            ``scipy.spatial.distance.cdist`` (e.g. ``'euclidean'``,
            ``'cityblock'``, ``'cosine'``, ``'correlation'``, ``'hamming'``,
            ``'jaccard'``).
        spatial_scale (str): ``'whole_brain'`` (default), ``'roi'``, or
            ``'searchlight'``. See `BrainData.distance`.
        roi_mask (BrainData | Nifti1Image | str | None): Atlas for
            ``spatial_scale='roi'``.
        radius (float): Searchlight radius for ``spatial_scale='searchlight'``.
        **kwargs (dict): Forwarded to ``scipy.spatial.distance.cdist``.

    Returns:
        Adjacency: Whole-brain pairwise distance matrix, or an ordinary stack.
            ROI matrices follow sorted nonzero atlas labels present in the source
            mask after resampling; searchlights follow source-mask voxel order.
    """
    valid = {"whole_brain", "roi", "searchlight"}
    if spatial_scale not in valid:
        raise ValueError(
            f"spatial_scale must be one of {sorted(valid)}, got {spatial_scale!r}"
        )

    if spatial_scale == "whole_brain":
        from scipy.spatial.distance import cdist

        from nltools.data import Adjacency

        dist_matrix = cdist(bd.data, bd.data, metric=metric, **kwargs)
        return Adjacency(dist_matrix, matrix_type="Distance")

    if spatial_scale == "searchlight":
        return _distance_searchlight(bd, metric=metric, radius=radius, **kwargs)

    # spatial_scale == "roi"
    return _distance_roi(bd, metric=metric, roi_mask=roi_mask, **kwargs)


def _resolve_atlas_label_vec(bd, roi_mask):
    """Resolve an atlas image + label vector aligned with bd.mask.

    Coerces ``roi_mask`` (BrainData / Nifti / path) to a Nifti, resamples
    to ``bd.mask`` (nearest-neighbor) if needed, and returns
    ``(atlas_img, label_vec, unique_labels)`` for use by per-parcel
    operations.
    """
    from pathlib import Path

    import nibabel as nib
    from nilearn.image import resample_to_img
    from nilearn.masking import apply_mask

    from nltools.data import BrainData

    if roi_mask is None:
        raise ValueError("roi_mask is required when spatial_scale='roi'.")
    if isinstance(roi_mask, BrainData):
        roi_img = roi_mask.to_nifti()
    elif isinstance(roi_mask, (str, Path)):
        roi_img = nib.load(str(roi_mask))
    else:
        roi_img = roi_mask

    if roi_img.shape != bd.mask.shape or not np.allclose(
        roi_img.affine, bd.mask.affine
    ):
        roi_img = resample_to_img(
            roi_img,
            bd.mask,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )

    label_vec = apply_mask(roi_img, bd.mask).astype(np.int64)
    unique_labels = np.unique(label_vec)
    unique_labels = unique_labels[unique_labels != 0]
    if unique_labels.size == 0:
        raise ValueError("roi_mask has no nonzero labels in the BrainData mask space.")
    return roi_img, label_vec, unique_labels


def _align_per_roi(bd, target, *, method, axis, roi_mask):
    """Per-parcel functional alignment + voxel-space reassembly.

    For each atlas parcel, runs ``align()`` on the slice of ``bd`` and
    ``target`` restricted to that parcel's voxels and collects results.
    The ``transformed`` field is reassembled into a single
    `BrainData` of the same shape as the input (each voxel filled
    with its parcel's transformed value per image; voxels outside any
    parcel = NaN). Per-parcel transform matrices and common models stay
    keyed by atlas label, since matrices over different voxel subsets
    cannot be painted into one image.

    Args:
        bd (BrainData): Source data to align.
        target (BrainData | np.ndarray): Alignment target (a `BrainData` for
            ``'procrustes'``; a common-model array for the SRM methods).
        method (str): ``'procrustes'``, ``'probabilistic_srm'``, or
            ``'deterministic_srm'``.
        axis (int): Axis to align over; see `align`.
        roi_mask (BrainData | Nifti1Image | str): Integer-labeled atlas defining
            the parcels.

    Returns:
        dict: ``'transformed'`` (one stitched `BrainData` on the source voxel
            axis), ``'transformation_matrix'`` and ``'common_model'`` (dicts
            keyed by atlas label), and ``'roi_labels'``, plus the per-parcel
            arrays ``'disparity'`` and ``'scale'`` for ``'procrustes'``. Each
            parcel value follows `align`'s rule: a `BrainData` carrying that
            parcel's mask where its columns are that parcel's voxels, and a raw
            `np.ndarray` where they are the model's features or images.

    Raises:
        ValueError: If a parcel's aligned data cannot be painted back onto that
            parcel's voxels, which happens when an SRM common model has a
            different feature count from the parcel's voxel count.
    """
    roi_img, label_vec, unique_labels = _resolve_atlas_label_vec(bd, roi_mask)

    if method == "procrustes":
        # Need a target BrainData to slice.
        from .utils import _check_brain_data

        target_bd = _check_brain_data(target)
        if target_bd.shape[-1] != bd.shape[-1]:
            raise ValueError(
                "For spatial_scale='roi' procrustes, target must share "
                "the BrainData voxel axis."
            )
    elif method in ("probabilistic_srm", "deterministic_srm"):
        target_bd = None  # target stays a numpy array; we don't slice it
    else:
        raise ValueError(
            "method must be ['procrustes','probabilistic_srm','deterministic_srm']"
        )

    transforms = {}
    common_models = {}
    disparities = []
    scales = []
    transformed_per_parcel: list[np.ndarray] = []

    for label in unique_labels:
        cols = label_vec == label
        sub = _result_with_mask(
            bd, bd.data[:, cols], _subset_mask(bd, cols), rows="preserve"
        )
        if method == "procrustes":
            t_sub = _result_with_mask(
                target_bd,
                target_bd.data[:, cols],
                _subset_mask(target_bd, cols),
                rows="preserve",
            )
            sub_target = t_sub
        else:
            sub_target = target  # SRM common model is voxel-agnostic

        sub_out = _align(sub, sub_target, method=method, axis=axis)

        # Accumulate: every spatial value is already an owned BrainData.
        transforms[int(label)] = sub_out["transformation_matrix"]
        common_models[int(label)] = sub_out["common_model"]
        if method == "procrustes":
            disparities.append(float(sub_out["disparity"]))
            scales.append(float(sub_out["scale"]))

        transformed_per_parcel.append(
            np.asarray(_aligned_array(sub_out["transformed"]))
        )

    # Stitch transformed → (n_images, n_voxels) BrainData.
    n_images = transformed_per_parcel[0].shape[0]
    out_arr = np.full((n_images, label_vec.shape[0]), np.nan, dtype=float)
    for label, parcel_arr in zip(unique_labels, transformed_per_parcel):
        cols = label_vec == label
        n_parcel_voxels = int(cols.sum())
        if parcel_arr.shape[-1] != n_parcel_voxels:
            raise ValueError(
                f"Aligned data for parcel {int(label)} has "
                f"{parcel_arr.shape[-1]} columns but the parcel covers "
                f"{n_parcel_voxels} voxels, so it cannot be painted back onto "
                f"the voxel axis. Use a common model with one feature per "
                f"parcel voxel."
            )
        out_arr[:, cols] = parcel_arr

    transformed_bd = _result_from_array(bd, out_arr, rows="preserve")
    out = {
        "transformed": transformed_bd,
        "transformation_matrix": transforms,
        "common_model": common_models,
        "roi_labels": unique_labels,
    }
    if method == "procrustes":
        out["disparity"] = np.asarray(disparities, dtype=float)
        out["scale"] = np.asarray(scales, dtype=float)
    return out


def _distance_roi(bd, *, metric, roi_mask, **kwargs):
    """Compute a pairwise distance matrix for each atlas parcel.

    Return an ordinary stack in sorted nonzero atlas-label order within the source mask.
    """
    from pathlib import Path

    import nibabel as nib
    from nilearn.image import resample_to_img
    from nilearn.masking import apply_mask
    from scipy.spatial.distance import cdist

    from nltools.data import Adjacency, BrainData

    if roi_mask is None:
        raise ValueError("roi_mask is required when spatial_scale='roi'.")

    # Coerce roi_mask to a Nifti1Image aligned with bd.mask.
    if isinstance(roi_mask, BrainData):
        roi_img = roi_mask.to_nifti()
    elif isinstance(roi_mask, (str, Path)):
        roi_img = nib.load(str(roi_mask))
    else:
        roi_img = roi_mask

    if roi_img.shape != bd.mask.shape or not np.allclose(
        roi_img.affine, bd.mask.affine
    ):
        roi_img = resample_to_img(
            roi_img,
            bd.mask,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )

    # Per-mask-voxel atlas labels.
    label_vec = apply_mask(roi_img, bd.mask).astype(np.int64)
    unique_labels = np.unique(label_vec)
    unique_labels = unique_labels[unique_labels != 0]

    if unique_labels.size == 0:
        raise ValueError("roi_mask has no nonzero labels in the BrainData mask space.")

    matrices = []
    for label in unique_labels:
        cols = label_vec == label
        matrices.append(
            cdist(bd.data[:, cols], bd.data[:, cols], metric=metric, **kwargs)
        )

    return Adjacency(matrices, matrix_type="distance")


def _distance_searchlight(bd, *, metric, radius, **kwargs):
    """Compute a pairwise distance matrix for each searchlight center.

    Return an ordinary stack in source-mask voxel order. Map per-center values
    externally with `nilearn.masking.unmask(values, bd.mask)`.
    """
    from scipy.spatial.distance import cdist

    from nltools.data import Adjacency

    from nltools.algorithms.neighborhoods import compute_searchlight_neighborhoods

    nbrs = compute_searchlight_neighborhoods(bd.mask, radius=radius)
    n_voxels = nbrs.n_voxels

    matrices = []
    for i in range(n_voxels):
        cols = nbrs.get_neighbors(i)
        matrices.append(
            cdist(bd.data[:, cols], bd.data[:, cols], metric=metric, **kwargs)
        )

    return Adjacency(matrices, matrix_type="distance")


def _multivariate_similarity(bd, images, tail=2):
    """Predict a BrainData spatial distribution from a linear combination.

    The predictors may be other BrainData instances or nibabel images.

    Args:
        bd (BrainData): Single image to be explained.
        images (BrainData | Nifti1Image): Predictor images (weight maps).
        tail (int): ``1`` or ``2`` for one- or two-tailed p-values.

    Returns:
        dict: Raw regression statistics (numpy arrays/scalars, not BrainData)
            with keys ``'beta'``, ``'t'``, ``'p'``, ``'df'``, ``'sigma'``,
            ``'residual'``.
    """
    # Notes:  Should add ridge, and lasso, elastic net options options
    from nltools.algorithms.similarity import _compute_multivariate_similarity
    from .utils import _check_brain_data

    if len(bd.shape) > 1:
        raise ValueError("This method can only decompose a single brain image.")

    images = _check_brain_data(images)
    data2, image2 = _check_masks(bd, images)

    # Prepare data for functional core: y is single image, X is predictors
    # image2 shape: (n_images, n_voxels) -> transpose to (n_voxels, n_images)
    y = data2.squeeze()  # Single image: (n_voxels,)
    X = image2.T  # Predictors: (n_voxels, n_images)

    # Delegate to functional core (stats.py)
    return _compute_multivariate_similarity(y, X, tail=tail)


def _mask_image_on_source_grid(bd, mask):
    """Return ``mask`` as a single 3-D image verified to share ``bd``'s grid.

    Sameness is `_check_space_match`, the one predicate the loader also uses, so
    a mask the constructor would have accepted without resampling is accepted
    here too. Nothing is resampled: a foreign grid is an error, not something to
    fix silently, because resampling either operand would change the voxel axis
    the caller asked to keep. A grid that matches only within that tolerance
    adopts the source's affine verbatim, which moves no data but keeps the
    stricter checks downstream in nilearn from rejecting sub-tolerance drift.
    """
    import nibabel as nib

    from . import BrainData
    from .io import _check_space_match

    if isinstance(mask, BrainData):
        mask_img = mask.to_nifti()
    elif isinstance(mask, (str, Path)):
        mask_img = nib.load(str(mask))
    elif isinstance(mask, nib.Nifti1Image):
        mask_img = mask
    else:
        raise TypeError(
            "mask must be a BrainData, nibabel image, or file path. "
            f"Received {type(mask).__name__}"
        )

    if len(mask_img.shape) != 3:
        raise ValueError("Mask must be a single image")

    if not _check_space_match(mask_img, bd.mask):
        raise ValueError(
            "apply_mask requires a mask on the same grid as the data: the data "
            f"is {bd.mask.shape} with affine\n{bd.mask.affine}\nand the mask is "
            f"{mask_img.shape} with affine\n{mask_img.affine}\n"
            "Bring them onto a common grid with resample() first."
        )

    if not np.array_equal(mask_img.affine, bd.mask.affine):
        mask_img = nib.Nifti1Image(mask_img.dataobj, bd.mask.affine)
    return mask_img


def _apply_mask(bd, mask):
    """Restrict BrainData to a mask's support without changing the grid.

    Support is every voxel of ``mask`` greater than zero. The mask defines the
    result's voxel axis on its own: where it reaches past ``bd``'s current
    support the result gains those voxels with zero values, so a mask larger
    than the data's own mask widens the array rather than intersecting with it.

    Args:
        bd (BrainData): Data to mask.
        mask (BrainData | Nifti1Image | str | Path): A single 3-D mask on the
            same grid and with the same affine as ``bd``.

    Returns:
        BrainData: Masked copy of ``bd`` with row metadata preserved.

    Raises:
        ValueError: If the mask is not a single 3-D image, or its shape or
            affine differs from ``bd``'s. Use ``resample()`` first in that case.
        TypeError: If ``mask`` is not a BrainData, nibabel image, or file path.

    Note:
        Masking is delegated to ``nilearn.masking.apply_mask``.
    """
    from nilearn.masking import apply_mask as nilearn_apply_mask

    mask_img = _mask_image_on_source_grid(bd, mask)

    # Use nilearn's apply_mask for efficient masking (C-optimized, single path, memory efficient)
    masked_data = nilearn_apply_mask(bd.to_nifti(), mask_img)
    masked = _result_with_mask(bd, masked_data, mask_img, rows="preserve")

    # Preserve 1D output for single images (backward compatibility)
    if (len(masked.shape) > 1) & (masked.shape[0] == 1):
        masked.data = masked.data.flatten()

    return masked


def _extract_roi(bd, mask, method="mean", n_components=None):
    """Extract activity from a binary mask or a labeled ROI atlas.

    `extract_roi` is an extraction convenience, not a masking primitive: unlike
    the strict same-grid `apply_mask`, it resamples `mask` onto `bd`'s own grid
    with nearest-neighbor interpolation before extracting, the same way
    nilearn's `NiftiLabelsMasker` resamples labels onto data. A mask already on
    `bd`'s grid is used as given. Labeled atlases (multiple ROIs) are handled
    with nilearn's ``NiftiLabelsMasker``.

    Args:
        bd (BrainData): Data to extract from.
        mask (BrainData | Nifti1Image | str): A binary mask (extracts from a
            single ROI) or a labeled atlas (extracts from every ROI), on any
            grid.
        method (str): Extraction method: ``'mean'`` (default), ``'median'``, or
            ``'pca'``.
        n_components (int | None): Number of components to return when
            ``method='pca'``.

    Returns:
        float | np.ndarray: For a binary mask, a scalar (single image) or 1D array
            of values (multiple images). For a labeled atlas, a 1D array with one
            value per ROI (single image), a 2D array of images x ROIs (multiple
            images), or the components array when `method='pca'`.

    Raises:
        ValueError: If, after resampling onto `bd`'s grid, `mask` has no
            overlap with `bd`.

    Examples:
        ```python
        # Extract mean from binary mask
        roi_values = brain.extract_roi(binary_mask)

        # Extract from atlas
        atlas_values = brain.extract_roi(atlas_mask)

        # PCA extraction
        components = brain.extract_roi(mask, method='pca', n_components=5)
        ```
    """
    from nilearn.maskers import NiftiLabelsMasker

    from . import BrainData
    from .io import _check_space_match
    from .utils import _check_brain_data_is_single

    methods = ["mean", "median", "pca"]
    if method not in methods:
        raise NotImplementedError(f"method must be one of {methods}, got {method}")

    # Coerce mask onto bd's own grid before extracting. A BrainData mask on a
    # foreign grid is resampled explicitly (nearest, so labels survive); any
    # other input (Nifti1Image, path) loads directly against bd's mask, which
    # resamples it implicitly the same way BrainData loading always has.
    if isinstance(mask, BrainData):
        mask_brain = mask
        if not _check_space_match(mask_brain.mask, bd.mask):
            mask_brain = mask_brain.resample(img=bd.mask, interpolation="nearest")
    else:
        mask_brain = BrainData(mask, mask=bd.mask, interpolation="nearest")

    # Check if binary or labeled mask
    unique_values = np.unique(mask_brain.data)
    n_unique = len(unique_values)

    if n_unique < 2:
        raise ValueError(
            "No voxels remain after masking - mask may not overlap with data"
        )

    mask_img = mask_brain.to_nifti()

    if n_unique == 2:
        # Binary mask - use simple extraction
        masked = _apply_mask(bd, mask_brain)
        is_single = _check_brain_data_is_single(masked)

        if method == "mean":
            out = masked.mean() if is_single else masked.mean(axis=1)
        elif method == "median":
            out = masked.median() if is_single else masked.median(axis=1)
        elif method == "pca":
            if is_single:
                raise ValueError("Cannot run PCA on a single image")
            output = _decompose(
                masked, method="pca", n_components=n_components, axis="images"
            )
            out = output["weights"].T

    elif n_unique > 2:
        # Labeled atlas - use NiftiLabelsMasker for efficiency
        # Round values to ensure integer labels (use int32 for nilearn/FSL/SPM
        # compatibility) on a copy, so a caller's mask is never mutated.
        mask_brain = mask_brain.copy()
        mask_brain.data = np.round(mask_brain.data).astype(np.int32)
        mask_img = mask_brain.to_nifti()

        # Create masker based on method
        if method in ["mean", "median"]:
            # For mean/median, use NiftiLabelsMasker
            strategy = "mean" if method == "mean" else "median"
            labels_masker = NiftiLabelsMasker(
                labels_img=mask_img,
                strategy=strategy,
                mask_img=bd.mask,
                standardize=None,  # nilearn >= 0.15 rejects the bool spelling
                resampling_target="data" if hasattr(bd, "mask") else None,
            )

            # Transform data
            data_4d = bd.to_nifti()
            out = labels_masker.fit_transform(data_4d)

            # If single image, return 1D array
            if out.shape[0] == 1:
                out = out[0]
            else:
                # For multiple images, transpose to (n_labels, n_images)
                out = out.T

        elif method == "pca":
            # Extract voxels from the whole atlas once, then slice by label in
            # numpy. This avoids rebuilding the nifti and re-resampling per ROI.
            if _check_brain_data_is_single(bd):
                raise ValueError("Cannot run PCA on a single image")

            atlas_mask = _result_from_array(
                mask_brain, (mask_brain.data > 0).astype(float), rows="preserve"
            )
            all_masked = _apply_mask(bd, atlas_mask)

            # apply_mask preserves voxel ordering relative to the mask, so the
            # label vector lines up with the columns of all_masked.data.
            labels_flat = mask_brain.data[mask_brain.data > 0]
            unique_labels = np.unique(labels_flat)

            out = []
            for label in unique_labels:
                roi = _result_with_mask(
                    all_masked,
                    all_masked.data[:, labels_flat == label],
                    _subset_mask(all_masked, labels_flat == label),
                    rows="preserve",
                )
                output = _decompose(
                    roi, method="pca", n_components=n_components, axis="images"
                )
                out.append(output["weights"].T)

            if len(out) > 0:
                out = np.array(out) if n_components == 1 else out

    else:
        raise ValueError(
            "Mask must be binary (2 unique values) or labeled atlas (>2 unique values)"
        )

    return out


def _detrend_data(bd, method="linear"):
    """Remove the linear trend from each voxel.

    Args:
        bd (BrainData): Data to detrend (must hold more than one image).
        method (str): ``'linear'`` (default) or ``'constant'``.

    Returns:
        BrainData: Detrended copy of ``bd``.
    """
    from scipy.signal import detrend

    if len(bd.shape) == 1:
        raise ValueError("Make sure there is more than one image in order to detrend.")

    out = _result_from_array(bd, detrend(bd.data, type=method, axis=0), rows="preserve")
    return out


def _r_to_z(bd):
    """Apply Fisher's r-to-z transformation to each data element.

    Args:
        bd (BrainData): Correlation values to transform.

    Returns:
        BrainData: Transformed copy of ``bd``.
    """
    from nltools.algorithms.similarity import fisher_r_to_z

    out = _result_from_array(bd, fisher_r_to_z(bd.data), rows="preserve")
    return out


def _z_to_r(bd):
    """Convert Fisher z scores back into r values for each data element.

    Args:
        bd (BrainData): z-scored values to transform.

    Returns:
        BrainData: Transformed copy of ``bd``.
    """
    from nltools.algorithms.similarity import fisher_z_to_r

    out = _result_from_array(bd, fisher_z_to_r(bd.data), rows="preserve")
    return out


def _filter_data(  # nosemgrep: kwargs-internal-forwarding  # forwards to nilearn.signal.clean
    bd, *, sampling_freq=None, high_pass=None, low_pass=None, **kwargs
):
    """Apply a Butterworth filter to data (wraps `nilearn.signal.clean`).

    Does not default to detrending and standardizing like nilearn
    implementation, but this can be overridden using kwargs.

    Args:
        bd (BrainData): Time series to filter.
        sampling_freq (float | None): Sampling frequency in Hz (i.e. 1 / TR).
        high_pass (float | None): High-pass cutoff frequency in Hz.
        low_pass (float | None): Low-pass cutoff frequency in Hz.
        **kwargs (dict): Forwarded to ``nilearn.signal.clean``. Common options:
            ``confounds`` (confound time series to remove), ``sample_mask``
            (volumes to exclude), ``detrend`` (default ``False``),
            ``standardize`` (``'zscore_sample'``, ``'psc'``, or ``None`` — the
            default; ``True``/``False`` are accepted as aliases for
            ``'zscore_sample'``/``None``), and ``ensure_finite`` (replace
            NaN/inf; default ``False``).

    Returns:
        BrainData: Filtered copy of ``bd``.

    See Also:
        ``nilearn.signal.clean`` for all available options.
    """
    from nilearn.signal import clean

    if sampling_freq is None:
        raise ValueError("Need to provide sampling rate (TR)!")
    if high_pass is None and low_pass is None:
        raise ValueError("high_pass and/or low_pass cutoff must be provided!")

    # Pop (not get) so these are not also forwarded via **kwargs below;
    # otherwise clean() receives detrend/standardize twice -> TypeError.
    # nilearn >= 0.15 drops boolean `standardize`; translate the aliases here
    # so callers keep the bool spelling without tripping its FutureWarning.
    standardize = kwargs.pop("standardize", None)
    if standardize is True:
        standardize = "zscore_sample"
    elif standardize is False:
        standardize = None
    detrend = kwargs.pop("detrend", False)

    data = clean(
        bd.data,
        t_r=1.0 / sampling_freq,
        detrend=detrend,
        standardize=standardize,
        high_pass=high_pass,
        low_pass=low_pass,
        **kwargs,
    )
    sample_mask = kwargs.get("sample_mask")
    if sample_mask is None:
        return _result_from_array(bd, data, rows="preserve")
    from .utils import _polars_row_select

    return _result_from_rows(
        bd,
        data,
        X=_polars_row_select(bd.X, sample_mask),
        Y=_polars_row_select(bd.Y, sample_mask),
    )


def _standardize(bd, *, method="center", axis=0):
    """Standardize data by centering it, optionally scaling to unit variance.

    Computed in float64 and cast back to the input dtype, so raw float32 BOLD
    (large offsets) stays exact. Constant voxels/observations z-score to 0.

    Args:
        bd (BrainData): Data to standardize.
        method (str): ``'center'`` subtracts the mean (default); ``'zscore'``
            subtracts the mean and divides by the standard deviation.
        axis (int): ``0`` to standardize each voxel across observations
            (default), ``1`` to standardize each observation across voxels.

    Returns:
        BrainData: Standardized copy of ``bd``.

    Raises:
        ValueError: If `method` is neither ``'center'`` nor ``'zscore'``.
    """
    if method not in ("center", "zscore"):
        raise ValueError(f"method must be 'center' or 'zscore', got {method!r}")
    if axis == 1 and len(bd.shape) == 1:
        raise IndexError(
            "BrainData is only 3d but standardization was requested over observations"
        )

    data = np.asarray(bd.data, dtype=np.float64)
    centered = data - data.mean(axis=axis, keepdims=True)
    if method == "zscore":
        std = centered.std(axis=axis, keepdims=True)
        std[std == 0] = 1.0  # constant along `axis` -> 0, not nan
        centered /= std

    # The output immediately replaces data, so avoid copying the source buffer.
    out = _result_from_array(
        bd, centered.astype(bd.data.dtype, copy=False), rows="preserve"
    )
    return out


def _scale_data(bd, scale_val=100.0, axis=None):
    """Scale data via mean scaling.

    Two scaling modes are available:

    - **Grand-mean scaling** (axis=None, default): Divides all values by the
      global mean across all voxels and timepoints. This is consistent with
      FSL and SPM behavior. Use scale_val=10000 for FSL-style scaling.

    - **Voxel-wise scaling** (axis=0): Divides each voxel's time-series by
      its own temporal mean. This is AFNI-style scaling and can be useful
      when voxels have very different baseline intensities. Voxels with
      zero or near-zero mean are set to zero to avoid NaN/Inf.

    When scale_val=100 (default), the result can be interpreted as something
    akin to (but not exactly) "percent signal change."

    Args:
        bd (BrainData): Data to scale.
        scale_val (float): Target value for the mean after scaling. Default
            ``100``.
        axis (int | None): ``None`` for grand-mean scaling (default, FSL/SPM
            style); ``0`` for voxel-wise scaling (AFNI style, each voxel scaled
            by its own temporal mean).

    Returns:
        BrainData: Scaled copy of ``bd``.

    Examples:
        ```python
        # Grand-mean scaling (default)
        scaled = brain.scale(100.0)

        # Voxel-wise scaling (AFNI style)
        scaled = brain.scale(100.0, axis=0)
        ```
    """
    data = bd.data

    if axis is None:
        # Grand-mean scaling: divide by global mean
        grand_mean = data.mean()
        if np.abs(grand_mean) < np.finfo(float).eps:
            data = np.zeros_like(data)
        else:
            data = data / grand_mean * scale_val
    elif axis == 0:
        # Voxel-wise scaling: divide each voxel by its temporal mean
        # Compute mean along time axis (axis=0), keeping dims for broadcasting
        voxel_means = data.mean(axis=0, keepdims=True)

        # Handle zero-mean voxels to avoid NaN/Inf
        # Set zero-mean voxels to 1 temporarily, then zero out result
        zero_mask = np.abs(voxel_means) < np.finfo(float).eps
        voxel_means_safe = np.where(zero_mask, 1.0, voxel_means)

        # Scale
        data = data / voxel_means_safe * scale_val

        # Zero out voxels that had zero mean
        if np.any(zero_mask):
            data[:, zero_mask.squeeze()] = 0.0
    else:
        raise ValueError(f"axis must be None or 0, got {axis}")

    return _result_from_array(bd, data, rows="preserve")


def _threshold_data(
    bd,
    *,
    upper=None,
    lower=None,
    binarize=False,
    coerce_nan=True,
    cluster_threshold=0,
):
    """Threshold BrainData instance with optional cluster filtering.

    Provide upper and lower values or percentages to perform two-sided
    thresholding. Binarize will return a mask image respecting thresholds
    if provided, otherwise respecting every non-zero value.

    Args:
        bd (BrainData): Data to threshold.
        upper (float | str | None): Upper cutoff. A string like ``'98%'``
            resolves as a percentile over the finite **nonzero** voxels (via
            `_resolve_threshold`; zeros on a masked map are absence of data and
            would skew the percentile). ``None`` for one-sided thresholding.
        lower (float | str | None): Lower cutoff, with the same percentile
            semantics as ``upper``. ``None`` for one-sided thresholding.
        binarize (bool): Return a binary image respecting the thresholds if
            provided, otherwise binarize every non-zero value. Default
            ``False``.
        coerce_nan (bool): Replace NaN values with 0 first. Default ``True``.
        cluster_threshold (int): Minimum cluster size in voxels. If ``> 0``,
            thresholds with ``nilearn.image.threshold_img`` and drops smaller
            clusters; band-pass thresholding (both ``upper`` and ``lower``) is
            not supported in that mode. Default ``0`` (disabled).

    Returns:
        BrainData: Thresholded copy of ``bd``.

    Note:
        With ``cluster_threshold=0`` (default) thresholding runs on the data
        array directly and supports band-pass thresholds; with
        ``cluster_threshold>0`` nilearn performs the cluster filtering.
    """

    if cluster_threshold > 0:
        # Use nilearn for cluster thresholding
        from nilearn.image import threshold_img
        from nilearn.masking import apply_mask as nilearn_apply_mask

        # Band-pass filtering not supported with cluster thresholding
        if upper is not None and lower is not None:
            raise ValueError(
                "Band-pass filtering (both upper and lower) not supported "
                "with cluster thresholding. Use one threshold only."
            )

        # Determine threshold value (from whichever is provided)
        threshold_val = upper if upper is not None else lower
        if threshold_val is None:
            raise ValueError("Must provide either upper or lower threshold")

        # Handle percentile strings
        b = _result_from_array(bd, bd.data, rows="preserve")
        if coerce_nan:
            b.data = np.nan_to_num(b.data)

        from .utils import _resolve_threshold

        threshold_val = _resolve_threshold(threshold_val, b.data)

        # Use nilearn's cluster thresholding
        out = _result_from_array(bd, bd.data, rows="preserve")
        thresholded_img = threshold_img(
            b.to_nifti(),
            threshold=threshold_val,
            cluster_threshold=cluster_threshold,
            two_sided=(upper is not None),
            copy_header=True,
        )

        # Convert back to data array
        out.data = nilearn_apply_mask(thresholded_img, bd.mask)

        if binarize:
            out.data = (out.data != 0).astype(float)

        return out

    # Use current efficient implementation (fast path)
    b = _result_from_array(bd, bd.data, rows="preserve")

    if coerce_nan:
        b.data = np.nan_to_num(b.data)

    from .utils import _resolve_threshold

    upper = _resolve_threshold(upper, b.data)
    lower = _resolve_threshold(lower, b.data)

    if upper is not None and lower is not None:
        b.data[(b.data < upper) & (b.data > lower)] = 0
    elif upper is not None:
        b.data[b.data < upper] = 0
    elif lower is not None:
        b.data[b.data > lower] = 0

    if binarize:
        b.data[b.data != 0] = 1
    return b


def _regions(
    bd,
    *,
    min_region_size=1350,
    method="local_regions",
    smoothing_fwhm=6,
    is_mask=False,
):
    """Extract brain connected regions into separate regions.

    Args:
        bd (BrainData): Image to segment.
        min_region_size (int): Minimum volume in mm³ for a region to be kept.
        method (str): ``'connected_components'`` labels each connected
            component directly; ``'local_regions'`` (default) seeds a marker at
            each component's peak and separates regions with a random-walker
            segmentation.
        smoothing_fwhm (float): Smooth the image first to extract sparser
            regions. Only used for ``method='local_regions'``.
        is_mask (bool): Treat ``bd`` as a boolean mask and use
            ``connected_label_regions`` instead. Default ``False``.

    Returns:
        BrainData: One image per extracted region.
    """
    from nilearn.regions import connected_label_regions, connected_regions

    from nltools.data import BrainData

    if is_mask:
        region_imgs, _ = connected_label_regions(bd.to_nifti())
    else:
        region_imgs, _ = connected_regions(
            bd.to_nifti(), min_region_size, method, smoothing_fwhm
        )

    return _result_from_array(
        bd, BrainData(region_imgs, mask=bd.mask).data, rows="clear"
    )


def _transform_pairwise_data(bd):
    """Transform BrainData into pairwise comparisons.

    Args:
        bd (BrainData): Data with a ``Y`` column to compare pairwise.

    Returns:
        BrainData: Pairwise-difference images with a recoded ``Y``.
    """
    from nltools.algorithms.similarity import transform_pairwise

    data, new_Y = transform_pairwise(bd.data, bd.Y.to_numpy())
    new_Y = np.where(np.asarray(new_Y) == -1, 0, new_Y)
    return _result_from_rows(bd, data, X=None, Y=pl.DataFrame(new_Y))


def _decompose(  # nosemgrep: kwargs-internal-forwarding  # forwards to the sklearn decomposition estimator
    bd, *, method="pca", axis="voxels", n_components=None, **kwargs
):
    """Decompose a BrainData object.

    Args:
        bd (BrainData): Data to decompose.
        method (str): Decomposition algorithm: ``'pca'`` (default), ``'ica'``,
            ``'nnmf'``, ``'fa'``, ``'dictionary'``, or ``'kernelpca'``.
        axis (str): Dimension to decompose: ``'voxels'`` (default) or
            ``'images'``.
        n_components (int | None): Number of components. ``None`` retains as
            many as possible.
        **kwargs (dict): Forwarded to the ``sklearn.decomposition`` estimator.

    Returns:
        dict: ``'decomposition_object'`` (the fitted sklearn estimator),
            ``'components'`` (`BrainData`), and ``'weights'`` (array).
    """
    import importlib

    _decomposition_algs = {
        "pca": "sklearn.decomposition.PCA",
        "ica": "sklearn.decomposition.FastICA",
        "nnmf": "sklearn.decomposition.NMF",
        "fa": "sklearn.decomposition.FactorAnalysis",
        "dictionary": "sklearn.decomposition.DictionaryLearning",
        "kernelpca": "sklearn.decomposition.KernelPCA",
    }
    if method not in _decomposition_algs:
        raise ValueError(
            f"Invalid decomposition method '{method}'. "
            f"Valid options: {list(_decomposition_algs)}"
        )
    module_path, class_name = _decomposition_algs[method].rsplit(".", 1)
    alg_class = getattr(importlib.import_module(module_path), class_name)

    out = {"decomposition_object": alg_class(n_components, **kwargs)}

    if axis == "images":
        out["decomposition_object"].fit(bd.data.T)
        out["components"] = _result_from_array(
            bd, out["decomposition_object"].transform(bd.data.T).T, rows="clear"
        )
        out["weights"] = out["decomposition_object"].components_.T
    elif axis == "voxels":
        out["decomposition_object"].fit(bd.data)
        out["weights"] = out["decomposition_object"].transform(bd.data)
        out["components"] = _result_from_array(
            bd, out["decomposition_object"].components_, rows="clear"
        )
    return out


def _align(bd, target, method="procrustes", axis=0):
    """Align a BrainData instance to a target using functional alignment.

    Alignment type can be hyperalignment or Shared Response Model. When
    using hyperalignment, `target` image can be another subject or an
    already estimated common model. When using SRM, `target` must be a previously
    estimated common model stored as a numpy array. Transformed data can be back
    projected to original data using Transformation matrix.

    See `nltools.algorithms.align` for aligning multiple BrainData instances.

    Args:
        bd (BrainData): Data to align.
        target (BrainData | np.ndarray): Alignment target — another subject or
            a fitted common model (array) for the SRM methods.
        method (str): ``'procrustes'`` (default), ``'probabilistic_srm'``, or
            ``'deterministic_srm'``.
        axis (int): Axis to align on. Default ``0``.

    Returns:
        dict: ``'transformed'``, ``'transformation_matrix'`` and
            ``'common_model'``, plus the floats ``'disparity'`` and
            ``'scale'`` for ``'procrustes'``. A value is a `BrainData` when its
            columns are a voxel axis matching the mask it carries, and a raw
            `np.ndarray` otherwise. ``'procrustes'`` therefore returns all
            three as independently owned `BrainData`: ``'transformed'`` on the
            source voxel axis, ``'common_model'`` on the target's, and
            ``'transformation_matrix'`` as ``(n_voxels, n_voxels)``. The SRM
            methods return ``'transformed'`` ``(n_images, n_features)`` and
            ``'common_model'`` ``(n_model_rows, n_features)`` as raw
            `np.ndarray`, because both span the common model's feature axis
            rather than voxels, and ``'transformation_matrix'`` as a
            `BrainData` of ``n_features`` voxel maps, shape
            ``(n_features, n_voxels)``. With ``axis=1`` the transformation
            matrix spans images on its column axis for either method, so it is
            a raw `np.ndarray` of shape ``(n_images, n_images)`` for
            ``'procrustes'`` and ``(n_model_rows, n_images)`` for the SRM
            methods.

    Raises:
        ValueError: If a value that must be returned as a `BrainData` has a
            column count other than the mask support. This is what a
            ``'procrustes'`` target with more voxels than the source produces,
            since the source data is zero-padded to the target's width.

    Examples:
        ```python
        # Hyperalign using procrustes transform
        out = data.align(target, method='procrustes')

        # Align using shared response model
        out = data.align(target, method='probabilistic_srm')

        # Project SRM-aligned data back into original voxel space
        original_data = np.dot(
            out['transformed'], out['transformation_matrix'].data
        )

        # Project procrustes-aligned data back into original voxel space
        original_voxels = np.dot(
            out['transformed'].data, out['transformation_matrix'].data.T
        )
        ```
    """
    from nltools.algorithms.alignment import procrustes
    from .utils import _check_brain_data

    if method not in ["probabilistic_srm", "deterministic_srm", "procrustes"]:
        raise ValueError(
            "Method must be ['probabilistic_srm','deterministic_srm','procrustes']"
        )

    data1 = bd.data.copy()

    if method == "procrustes":
        target = _check_brain_data(target)
        data2 = target.data.copy()

        # pad columns if different shapes
        sizes_1 = [x.shape[1] for x in [data1, data2]]
        C = max(sizes_1)
        y = data1[:, 0:C]
        missing = C - y.shape[1]
        add = np.zeros((y.shape[0], missing))
        data1 = np.append(y, add, axis=1)
    else:
        data2 = target.copy()

    if axis == 1:
        data1 = data1.T
        data2 = data2.T

    out = {}
    if method in ["deterministic_srm", "probabilistic_srm"]:
        if not isinstance(target, np.ndarray):
            raise ValueError(
                "Common Model must be a numpy array for  ['deterministic_srm', 'probabilistic_srm']"
            )

        if data2.shape[0] != data1.shape[0]:
            raise ValueError("The number of timepoints(TRs) does not match the model.")

        A = data1.T.dot(data2)

        # # Solve the Procrustes problem
        U, _, V = np.linalg.svd(A, full_matrices=False)

        transformation = U.dot(V).T
        transformed = data1.dot(transformation.T)
        if axis == 1:
            # Return the aligned data on the source (images, voxels) layout.
            transformed = transformed.T

        # On axis=1 the transformation spans images, not voxels, so it stays
        # an array. The transformed data and the common model always live on
        # the model's feature axis, so they stay arrays as in v0.5.1.
        out["transformation_matrix"] = (
            transformation
            if axis == 1
            else _brain_result(
                bd, transformation, "transformation_matrix", rows="clear"
            )
        )
        out["transformed"] = transformed
        out["common_model"] = np.array(target, copy=True)
    elif method == "procrustes":
        _, transformed, out["disparity"], tf_mtx, out["scale"] = procrustes(
            data2, data1
        )
        transformed_brain = _brain_result(
            bd,
            transformed.T if axis == 1 else transformed,
            "transformed",
            rows="preserve",
        )
        out["transformed"] = transformed_brain
        out["common_model"] = _brain_result(
            target, target.data, "common_model", rows="clear"
        )
        # `procrustes` solves for R with `transformed = original @ R.T`; store
        # the transpose so back-projection is `transformed @ T.T`, the same
        # convention as `nltools.algorithms.align`.
        out["transformation_matrix"] = (
            tf_mtx.T
            if axis == 1
            else _brain_result(
                transformed_brain, tf_mtx.T, "transformation_matrix", rows="clear"
            )
        )
    return out


def _smooth(bd, fwhm):
    """Apply spatial smoothing using nilearn's ``smooth_img``.

    Args:
        bd (BrainData): Data to smooth.
        fwhm (float): Full width at half maximum of the Gaussian kernel, in mm.

    Returns:
        BrainData: Smoothed copy of ``bd``.
    """
    from nilearn.image import smooth_img
    from nilearn.masking import apply_mask as nilearn_apply_mask

    from .utils import _check_brain_data_is_single

    # Single conversion: data -> nifti -> smooth -> data
    nifti = bd.to_nifti()
    smoothed_nifti = smooth_img(nifti, fwhm)
    smoothed_data = nilearn_apply_mask(smoothed_nifti, bd.mask)

    # Ensure single images remain 1D
    if _check_brain_data_is_single(bd):
        smoothed_data = smoothed_data.flatten()

    out = _result_from_array(bd, smoothed_data, rows="preserve")

    return out


def _find_spikes_data(
    bd,
    global_spike_cutoff=3,
    diff_spike_cutoff=3,
    *,
    TR=None,
    sampling_freq=None,
):
    """Identify spikes from time-series data; see `find_spikes`."""
    from nltools.algorithms.outliers import find_spikes

    return find_spikes(
        bd,
        global_spike_cutoff=global_spike_cutoff,
        diff_spike_cutoff=diff_spike_cutoff,
        TR=TR,
        sampling_freq=sampling_freq,
    )


def _temporal_resample(bd, *, sampling_freq=None, target=None, target_type="hz"):
    """Resample a BrainData time series to a target frequency or sample count.

    Resample BrainData timeseries to a new target frequency or number of samples
    using Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) interpolation.
    This function can up- or down-sample data.

    Args:
        bd (BrainData): Time series to resample.
        sampling_freq (float | None): Sampling frequency of the data in Hz.
        target (float | None): Resampling target, interpreted per
            ``target_type``.
        target_type (str): Units of ``target``: ``'hz'`` (default),
            ``'samples'``, or ``'seconds'``.

    Returns:
        BrainData: Resampled copy of ``bd``.

    Note:
        This function can use quite a bit of RAM.
    """
    from scipy.interpolate import pchip

    if target_type == "samples":
        n_samples = target
    elif target_type == "seconds":
        n_samples = target * sampling_freq
    elif target_type == "hz":
        n_samples = float(sampling_freq) / float(target)
    else:
        raise ValueError('Make sure target_type is "samples", "seconds", or "hz".')

    orig_spacing = np.arange(0, bd.shape[0], 1)
    new_spacing = np.arange(0, bd.shape[0], n_samples)

    resampled_data = np.zeros([len(new_spacing), bd.shape[1]])
    for i in range(bd.shape[1]):
        interpolate = pchip(orig_spacing, bd.data[:, i])
        resampled_data[:, i] = interpolate(new_spacing)
    out = _result_from_rows(bd, resampled_data, X=None, Y=None)
    return out
