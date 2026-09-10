"""Analysis operations on `BrainData`.

Functions for similarity, distance, masking, ROI extraction, filtering,
thresholding, decomposition, alignment, smoothing, and related operations.
Each takes a `BrainData` as its first argument; the corresponding
`BrainData` methods delegate here.
"""

import numpy as np
import polars as pl

from .utils import _copy_without_fit_state, resolve_roi_atlas


def check_masks(bd, image):
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


def similarity(bd, image, metric="correlation"):
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
    from .utils import check_brain_data

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

    image = check_brain_data(image)
    data2, image2 = check_masks(bd, image)

    # Delegate to functional core (stats.py)
    return compute_similarity(data2, image2, metric=metric)


def distance(  # nosemgrep: kwargs-internal-forwarding  # forwards to scipy.spatial.distance.cdist
    bd,
    metric="euclidean",
    *,
    spatial_scale: str = "whole_brain",
    roi_mask=None,
    radius_mm: float = 10.0,
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
            ``spatial_scale='roi'``: a 3-D label image or a path to one, a
            `BrainData` label vector, or a stacked binary mask from
            `expand_mask` (a `BrainData` or a 4-D image).
        radius_mm (float): Searchlight radius for ``spatial_scale='searchlight'``.
        **kwargs (dict): Forwarded to ``scipy.spatial.distance.cdist``.

    Returns:
        Adjacency: Whole-brain pairwise distance matrix, or a stacked Adjacency
            (one per parcel/searchlight) with ``spatial_scale`` provenance set.
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
        return _distance_searchlight(bd, metric=metric, radius_mm=radius_mm, **kwargs)

    # spatial_scale == "roi"
    return _distance_roi(bd, metric=metric, roi_mask=roi_mask, **kwargs)


def align_per_roi(bd, target, *, method, axis, roi_mask):
    """Per-parcel functional alignment + voxel-space reassembly.

    For each atlas parcel, runs ``align()`` on the slice of ``bd`` and
    ``target`` restricted to that parcel's voxels and collects results.
    The ``transformed`` field is reassembled into a single
    `BrainData` of the same shape as the input (each voxel filled
    with its parcel's transformed value per image; voxels outside any
    parcel = NaN). Per-parcel transform matrices and common-model
    objects are kept as dicts keyed by atlas label, since matrices over
    different voxel subsets can't be painted into one image.

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
        dict: ``'transformed'`` (`BrainData`), ``'transformation_matrix'`` and
            ``'common_model'`` (dicts keyed by atlas label), ``'disparity'`` and
            ``'scale'`` (arrays, one entry per parcel), and ``'roi_labels'``.
    """
    roi_img, label_vec, unique_labels = resolve_roi_atlas(bd, roi_mask)

    if method == "procrustes":
        # Need a target BrainData to slice.
        from .utils import check_brain_data

        target_bd = check_brain_data(target)
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

    transforms: dict[int, np.ndarray] = {}
    common_models: dict[int, np.ndarray | object] = {}
    disparities = []
    scales = []
    transformed_per_parcel: list[np.ndarray] = []

    for label in unique_labels:
        cols = label_vec == label
        sub = _copy_without_fit_state(bd, copy_data=False)
        sub.data = bd.data[:, cols]
        if method == "procrustes":
            t_sub = _copy_without_fit_state(target_bd, copy_data=False)
            t_sub.data = target_bd.data[:, cols]
            sub_target = t_sub
        else:
            sub_target = target  # SRM common model is voxel-agnostic

        sub_out = align(sub, sub_target, method=method, axis=axis)

        # Accumulate
        tf = sub_out["transformation_matrix"]
        transforms[int(label)] = tf.data if hasattr(tf, "data") else np.asarray(tf)
        common_models[int(label)] = sub_out.get("common_model")
        disparities.append(float(sub_out.get("disparity", np.nan)))
        scales.append(float(sub_out.get("scale", np.nan)))

        transformed = sub_out["transformed"]
        arr = transformed.data if hasattr(transformed, "data") else transformed
        transformed_per_parcel.append(np.asarray(arr))

    # Stitch transformed → (n_images, n_voxels) BrainData.
    n_images = transformed_per_parcel[0].shape[0]
    out_arr = np.full((n_images, label_vec.shape[0]), np.nan, dtype=float)
    for label, parcel_arr in zip(unique_labels, transformed_per_parcel):
        cols = label_vec == label
        out_arr[:, cols] = parcel_arr

    from nltools.data import BrainData

    transformed_bd = BrainData(out_arr, mask=bd.mask)
    return {
        "transformed": transformed_bd,
        "transformation_matrix": transforms,
        "common_model": common_models,
        "disparity": np.asarray(disparities, dtype=float),
        "scale": np.asarray(scales, dtype=float),
        "roi_labels": unique_labels,
    }


def reduce_per_roi(bd, reducer, *, roi_mask):
    """Apply a reducer within each parcel and paint results back to voxel space.

    This performs spatial smoothing via parcellation using a reducer such as
    ``np.mean``.

    For each image ``i`` and each parcel ``p``, computes
    ``reducer(bd.data[i, voxels-in-p])`` and assigns that scalar to every
    voxel in parcel ``p`` for image ``i``. Voxels outside any parcel get
    NaN. Output is a `BrainData` of the same shape as the input.

    Used by ``BrainData.{mean,std,median}(spatial_scale='roi')``.

    Args:
        bd (BrainData): Data to reduce.
        reducer (Callable): NumPy-style reducer accepting ``axis=``, e.g.
            ``np.mean``.
        roi_mask (BrainData | Nifti1Image | str): Integer-labeled atlas defining
            the parcels.

    Returns:
        BrainData: Parcel-wise reduced values painted back to voxel space.
    """
    from nltools.mask import roi_to_brain_from_atlas

    roi_img, label_vec, unique_labels = resolve_roi_atlas(bd, roi_mask)

    # bd.data is (n_images, n_voxels). For 1-D (single image) reshape.
    data = bd.data
    if data.ndim == 1:
        data = data.reshape(1, -1)

    per_parcel = np.column_stack(
        [reducer(data[:, label_vec == label], axis=1) for label in unique_labels]
    )  # (n_images, n_parcels)

    return roi_to_brain_from_atlas(
        per_parcel,
        atlas=roi_img,
        source_mask=bd.mask,
        roi_labels=unique_labels,
    )


def _distance_roi(bd, *, metric, roi_mask, **kwargs):
    """Compute a pairwise distance matrix for each atlas parcel.

    Returns a stacked Adjacency with ``SpatialScale`` provenance attached.
    """
    from scipy.spatial.distance import cdist

    from nltools.data import Adjacency, BrainData
    from nltools.data.adjacency.spatial import SpatialScale

    roi_img, label_vec, unique_labels = resolve_roi_atlas(bd, roi_mask)

    matrices = []
    for label in unique_labels:
        cols = label_vec == label
        matrices.append(
            cdist(bd.data[:, cols], bd.data[:, cols], metric=metric, **kwargs)
        )

    spatial_scale = SpatialScale(
        atlas=BrainData(roi_img, mask=bd.mask),
        roi_labels=unique_labels,
        source_mask=bd.mask,
        kind="roi",
    )
    return Adjacency(matrices, matrix_type="distance", spatial_scale=spatial_scale)


def _distance_searchlight(bd, *, metric, radius_mm, **kwargs):
    """Compute a pairwise distance matrix for each searchlight center.

    Returns a stacked Adjacency with ``SpatialScale(kind='searchlight')``.

    Each masked voxel is its own ``roi_label`` (1-indexed), and the
    synthetic atlas labels each masked voxel with its own ID — so
    ``Adjacency.to_brain(values)`` paints ``values[i]`` onto the i-th
    masked voxel (the searchlight center).
    """
    import nibabel as nib
    from scipy.spatial.distance import cdist

    from nltools.data import Adjacency, BrainData
    from nltools.data.adjacency.spatial import SpatialScale

    from .neighborhoods import compute_searchlight_neighborhoods

    nbrs = compute_searchlight_neighborhoods(
        bd.mask, radius_mm=radius_mm, use_cache=True
    )
    n_voxels = nbrs.n_voxels

    matrices = []
    for i in range(n_voxels):
        cols = nbrs.get_neighbors(i)
        matrices.append(
            cdist(bd.data[:, cols], bd.data[:, cols], metric=metric, **kwargs)
        )

    # Synthetic atlas: each masked voxel labeled with its own integer ID
    # (1-indexed so 0 stays "outside the atlas"). Built by writing
    # 1..n_voxels into the mask voxels.
    mask_arr = bd.mask.get_fdata().astype(bool)
    atlas_arr = np.zeros(mask_arr.shape, dtype=np.int32)
    voxel_ids = np.arange(1, n_voxels + 1, dtype=np.int32)
    atlas_arr[mask_arr] = voxel_ids
    atlas_img = nib.Nifti1Image(atlas_arr, bd.mask.affine, bd.mask.header)

    spatial_scale = SpatialScale(
        atlas=BrainData(atlas_img, mask=bd.mask),
        roi_labels=voxel_ids,
        source_mask=bd.mask,
        kind="searchlight",
    )
    return Adjacency(matrices, matrix_type="distance", spatial_scale=spatial_scale)


def multivariate_similarity(bd, images, method="ols", tail=2):
    """Predict a BrainData spatial distribution from a linear combination.

    The predictors may be other BrainData instances or nibabel images.

    Args:
        bd (BrainData): Single image to be explained.
        images (BrainData | Nifti1Image): Predictor images (weight maps).
        method (str): Regression method. Default: ``'ols'``.
        tail (int): ``1`` or ``2`` for one- or two-tailed p-values.

    Returns:
        dict: Raw regression statistics (numpy arrays/scalars, not BrainData)
            with keys ``'beta'``, ``'t'``, ``'p'``, ``'df'``, ``'sigma'``,
            ``'residual'``.
    """
    # Notes:  Should add ridge, and lasso, elastic net options options
    from nltools.algorithms.similarity import compute_multivariate_similarity
    from .utils import check_brain_data

    if len(bd.shape) > 1:
        raise ValueError("This method can only decompose a single brain image.")

    images = check_brain_data(images)
    data2, image2 = check_masks(bd, images)

    # Prepare data for functional core: y is single image, X is predictors
    # image2 shape: (n_images, n_voxels) -> transpose to (n_voxels, n_images)
    y = data2.squeeze()  # Single image: (n_voxels,)
    X = image2.T  # Predictors: (n_voxels, n_images)

    # Delegate to functional core (stats.py)
    return compute_multivariate_similarity(y, X, method=method, tail=tail)


def apply_mask(bd, mask, resample_mask_to_brain=False):
    """Mask BrainData instance using nilearn functionality.

    Note target data will be resampled into the same space as the mask. If you would like the mask
    resampled into the BrainData space, then set resample_mask_to_brain=True.

    Args:
        bd (BrainData): Data to mask.
        mask (BrainData | Nifti1Image): Mask to apply.
        resample_mask_to_brain (bool): Resample the mask into the brain's space
            before applying it. Default: ``False``.

    Returns:
        BrainData: Masked copy of ``bd``.

    Note:
        Masking is delegated to ``nilearn.masking.apply_mask``.
    """
    from nilearn.image import resample_to_img
    from nilearn.masking import apply_mask as nilearn_apply_mask

    from .utils import check_brain_data, check_brain_data_is_single

    # Coerce raw Niimg-like masks into the *target's* space, not the default
    # MNI152 template. Without bd.mask as context, check_brain_data re-homes a
    # raw nifti onto the package-default mask, which silently mismatches (and
    # then loudly fails) for any BrainData in a non-default space.
    mask = check_brain_data(mask, mask=bd.mask)
    if not check_brain_data_is_single(mask):
        raise ValueError("Mask must be a single image")

    # Handle resampling if requested (preserve existing feature)
    mask_img = mask.to_nifti()
    if resample_mask_to_brain:
        mask_img = resample_to_img(
            mask_img,
            bd.to_nifti(),
            interpolation="nearest",  # Masks are discrete, use nearest
            force_resample=True,
            copy_header=True,
        )

    # Use nilearn's apply_mask for efficient masking (C-optimized, single path, memory efficient)
    masked_data = nilearn_apply_mask(bd.to_nifti(), mask_img)
    masked = _copy_without_fit_state(bd, copy_data=False)
    masked.data = masked_data

    # Update mask, voxel resolution, and space
    masked.mask = mask_img
    affine = mask_img.affine
    masked._voxel_resolution = np.abs(np.diag(affine[:3, :3]))
    from .io import detect_space

    masked._space = detect_space(mask_img)

    # Preserve 1D output for single images (backward compatibility)
    if (len(masked.shape) > 1) & (masked.shape[0] == 1):
        masked.data = masked.data.flatten()

    return masked


def extract_roi(bd, mask, method="mean", n_components=None):
    """Extract activity from a binary mask or a labeled ROI atlas.

    Labeled atlases (multiple ROIs) are handled with nilearn's
    ``NiftiLabelsMasker``.

    Args:
        bd (BrainData): Data to extract from.
        mask (BrainData | Nifti1Image | str): A binary mask (extracts from a
            single ROI) or a labeled atlas (extracts from every ROI).
        method (str): Extraction method: ``'mean'`` (default), ``'median'``, or
            ``'pca'``.
        n_components (int | None): Number of components to return when
            ``method='pca'``.

    Returns:
        float | np.ndarray: For a binary mask, a scalar (single image) or 1D array
            of values (multiple images). For a labeled atlas, a 1D array with one
            value per ROI (single image), a 2D array of images x ROIs (multiple
            images), or the components array when `method='pca'`.

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

    from .utils import check_brain_data, check_brain_data_is_single

    methods = ["mean", "median", "pca"]
    if method not in methods:
        raise NotImplementedError(f"method must be one of {methods}, got {method}")

    # Convert mask to BrainData if needed
    mask_brain = check_brain_data(mask)
    mask_img = mask_brain.to_nifti()

    # Check if binary or labeled mask
    unique_values = np.unique(mask_brain.data)
    n_unique = len(unique_values)

    if n_unique == 2:
        # Binary mask - use simple extraction
        masked = apply_mask(bd, mask_brain)
        is_single = check_brain_data_is_single(masked)

        if method == "mean":
            out = masked.mean() if is_single else masked.mean(axis=1)
        elif method == "median":
            out = masked.median() if is_single else masked.median(axis=1)
        elif method == "pca":
            if is_single:
                raise ValueError("Cannot run PCA on a single image")
            # Check if masked has any data
            if masked.data.size == 0 or masked.data.shape[1] == 0:
                raise ValueError(
                    "No voxels remain after masking - mask may not overlap with data"
                )
            output = decompose(
                masked, method="pca", n_components=n_components, axis="images"
            )
            out = output["weights"].T

    elif n_unique > 2:
        # Labeled atlas - use NiftiLabelsMasker for efficiency
        # Round values to ensure integer labels (use int32 for nilearn/FSL/SPM compatibility)
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
            if check_brain_data_is_single(bd):
                raise ValueError("Cannot run PCA on a single image")

            atlas_mask = _copy_without_fit_state(mask_brain, copy_data=False)
            atlas_mask.data = (mask_brain.data > 0).astype(float)
            all_masked = apply_mask(bd, atlas_mask)

            # apply_mask preserves voxel ordering relative to the mask, so the
            # label vector lines up with the columns of all_masked.data.
            labels_flat = mask_brain.data[mask_brain.data > 0]
            unique_labels = np.unique(labels_flat)

            out = []
            for label in unique_labels:
                roi = _copy_without_fit_state(all_masked, copy_data=False)
                roi.data = all_masked.data[:, labels_flat == label]
                output = decompose(
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


def detrend_data(bd, method="linear"):
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

    out = _copy_without_fit_state(bd, copy_data=False)
    out.data = detrend(bd.data, type=method, axis=0)
    return out


def r_to_z(bd):
    """Apply Fisher's r-to-z transformation to each data element.

    Args:
        bd (BrainData): Correlation values to transform.

    Returns:
        BrainData: Transformed copy of ``bd``.
    """
    from nltools.algorithms.similarity import fisher_r_to_z

    out = _copy_without_fit_state(bd, copy_data=False)
    # fisher_r_to_z creates a new array
    out.data = fisher_r_to_z(bd.data)
    return out


def z_to_r(bd):
    """Convert Fisher z scores back into r values for each data element.

    Args:
        bd (BrainData): z-scored values to transform.

    Returns:
        BrainData: Transformed copy of ``bd``.
    """
    from nltools.algorithms.similarity import fisher_z_to_r

    out = _copy_without_fit_state(bd, copy_data=False)
    # fisher_z_to_r creates a new array
    out.data = fisher_z_to_r(bd.data)
    return out


def filter_data(  # nosemgrep: kwargs-internal-forwarding  # forwards to nilearn.signal.clean
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

    # The output immediately replaces data, so avoid copying the source buffer.
    out = _copy_without_fit_state(bd, copy_data=False)
    out.data = clean(
        bd.data,
        t_r=1.0 / sampling_freq,
        detrend=detrend,
        standardize=standardize,
        high_pass=high_pass,
        low_pass=low_pass,
        **kwargs,
    )
    return out


def standardize(bd, *, axis=0, method="center"):
    """Standardize BrainData() instance.

    Computed in float64 and cast back to the input dtype, so raw float32 BOLD
    (large offsets) stays exact. Constant voxels/observations z-score to 0.

    Args:
        bd (BrainData): Data to standardize.
        axis (int): ``0`` to standardize each voxel across observations
            (default), ``1`` to standardize each observation across voxels.
        method (str): ``'center'`` (default) or ``'zscore'``.

    Returns:
        BrainData: Standardized copy of ``bd``.
    """
    if axis == 1 and len(bd.shape) == 1:
        raise IndexError(
            "BrainData is only 3d but standardization was requested over observations"
        )
    if method not in ("center", "zscore"):
        raise ValueError('method must be ["center","zscore"')

    data = np.asarray(bd.data, dtype=np.float64)
    centered = data - data.mean(axis=axis, keepdims=True)
    if method == "zscore":
        std = centered.std(axis=axis, keepdims=True)
        std[std == 0] = 1.0  # constant along `axis` -> 0, not nan
        centered /= std

    # The output immediately replaces data, so avoid copying the source buffer.
    out = _copy_without_fit_state(bd, copy_data=False)
    out.data = centered.astype(bd.data.dtype, copy=False)
    return out


def scale_data(bd, scale_val=100.0, axis=None):
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
    out = _copy_without_fit_state(bd)

    if axis is None:
        # Grand-mean scaling: divide by global mean
        grand_mean = out.data.mean()
        if np.abs(grand_mean) < np.finfo(float).eps:
            out.data = np.zeros_like(out.data)
        else:
            out.data = out.data / grand_mean * scale_val
    elif axis == 0:
        # Voxel-wise scaling: divide each voxel by its temporal mean
        # Compute mean along time axis (axis=0), keeping dims for broadcasting
        voxel_means = out.data.mean(axis=0, keepdims=True)

        # Handle zero-mean voxels to avoid NaN/Inf
        # Set zero-mean voxels to 1 temporarily, then zero out result
        zero_mask = np.abs(voxel_means) < np.finfo(float).eps
        voxel_means_safe = np.where(zero_mask, 1.0, voxel_means)

        # Scale
        out.data = out.data / voxel_means_safe * scale_val

        # Zero out voxels that had zero mean
        if np.any(zero_mask):
            out.data[:, zero_mask.squeeze()] = 0.0
    else:
        raise ValueError(f"axis must be None or 0, got {axis}")

    return out


def threshold_data(
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
            `nltools.utils.resolve_threshold` — zeros on a masked map are
            absence of data and would skew the percentile). ``None`` for
            one-sided thresholding.
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
        b = _copy_without_fit_state(bd)
        if coerce_nan:
            b.data = np.nan_to_num(b.data)

        from nltools.utils import resolve_threshold

        threshold_val = resolve_threshold(threshold_val, b.data)

        # Use nilearn's cluster thresholding
        out = _copy_without_fit_state(bd, copy_data=False)
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
    b = _copy_without_fit_state(bd)

    if coerce_nan:
        b.data = np.nan_to_num(b.data)

    from nltools.utils import resolve_threshold

    upper = resolve_threshold(upper, b.data)
    lower = resolve_threshold(lower, b.data)

    if upper is not None and lower is not None:
        b.data[(b.data < upper) & (b.data > lower)] = 0
    elif upper is not None:
        b.data[b.data < upper] = 0
    elif lower is not None:
        b.data[b.data > lower] = 0

    if binarize:
        b.data[b.data != 0] = 1
    return b


def regions(
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

    return BrainData(region_imgs, mask=bd.mask)


def transform_pairwise_data(bd):
    """Transform BrainData into pairwise comparisons.

    Args:
        bd (BrainData): Data with a ``Y`` column to compare pairwise.

    Returns:
        BrainData: Pairwise-difference images with a recoded ``Y``.
    """
    from nltools.algorithms.similarity import transform_pairwise

    out = _copy_without_fit_state(bd, copy_data=False)
    out.data, new_Y = transform_pairwise(bd.data, bd.Y.to_numpy())
    new_Y = np.where(np.asarray(new_Y) == -1, 0, new_Y)
    out.Y = pl.DataFrame(new_Y)
    return out


def decompose(  # nosemgrep: kwargs-internal-forwarding  # forwards to the sklearn decomposition estimator
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
        out["components"] = bd.create_empty()
        out["components"].data = out["decomposition_object"].transform(bd.data.T).T
        out["weights"] = out["decomposition_object"].components_.T
    elif axis == "voxels":
        out["decomposition_object"].fit(bd.data)
        out["weights"] = out["decomposition_object"].transform(bd.data)
        out["components"] = bd.create_empty()
        out["components"].data = out["decomposition_object"].components_
    return out


def align(bd, target, method="procrustes", axis=0):
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
        dict: ``'transformed'``, ``'transformation_matrix'``, and
            ``'common_model'`` (plus ``'disparity'`` and ``'scale'`` for
            ``'procrustes'``).

    Examples:
        ```python
        # Hyperalign using procrustes transform
        out = data.align(target, method='procrustes')

        # Align using shared response model
        out = data.align(target, method='probabilistic_srm')

        # Project aligned data back into original data space
        original_data = np.dot(out['transformed'].data, out['transformation_matrix'].T)
        ```
    """
    from nltools.algorithms.alignment import procrustes
    from .utils import check_brain_data

    if method not in ["probabilistic_srm", "deterministic_srm", "procrustes"]:
        raise ValueError(
            "Method must be ['probabilistic_srm','deterministic_srm','procrustes']"
        )

    data1 = bd.data.copy()

    if method == "procrustes":
        target = check_brain_data(target)
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

        transformation = _copy_without_fit_state(bd, copy_data=False)
        transformation.data = U.dot(V).T
        out["transformation_matrix"] = transformation

        out["transformed"] = data1.dot(out["transformation_matrix"].data.T)
        out["common_model"] = target
    elif method == "procrustes":
        _, transformed, out["disparity"], tf_mtx, out["scale"] = procrustes(
            data2, data1
        )
        transformed_brain = _copy_without_fit_state(bd, copy_data=False)
        transformed_brain.data = transformed
        out["transformed"] = transformed_brain
        out["common_model"] = target
        out["transformation_matrix"] = _copy_without_fit_state(
            transformed_brain, copy_data=False
        )
        out["transformation_matrix"].data = tf_mtx
    if axis == 1:
        if method == "procrustes":
            out["transformed"].data = out["transformed"].data.T
        else:
            out["transformed"] = out["transformed"].T

    return out


def smooth(bd, fwhm):
    """Apply spatial smoothing using nilearn's ``smooth_img``.

    Args:
        bd (BrainData): Data to smooth.
        fwhm (float): Full width at half maximum of the Gaussian kernel, in mm.

    Returns:
        BrainData: Smoothed copy of ``bd``.
    """
    from nilearn.image import smooth_img
    from nilearn.masking import apply_mask as nilearn_apply_mask

    from .utils import check_brain_data_is_single

    # Single conversion: data -> nifti -> smooth -> data
    nifti = bd.to_nifti()
    smoothed_nifti = smooth_img(nifti, fwhm)
    smoothed_data = nilearn_apply_mask(smoothed_nifti, bd.mask)

    # Ensure single images remain 1D
    if check_brain_data_is_single(bd):
        smoothed_data = smoothed_data.flatten()

    out = _copy_without_fit_state(bd, copy_data=False)
    out.data = smoothed_data

    return out


def find_spikes_data(
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


def temporal_resample(bd, *, sampling_freq=None, target=None, target_type="hz"):
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
    out = _copy_without_fit_state(bd, copy_data=False)
    out.data = resampled_data
    return out
