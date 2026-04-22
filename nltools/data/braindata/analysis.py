"""
BrainData analysis functions.

Standalone functions extracted from BrainData class methods for similarity,
distance, masking, ROI extraction, ICC, filtering, thresholding, decomposition,
alignment, smoothing, and other analytical operations.
"""

import numpy as np
import polars as pl

from .utils import shallow_copy


def check_masks(bd, image):
    """Check to make sure masks are the same for each dataset and if not create a union mask

    Args:
        bd: BrainData instance
        image: BrainData instance to compare masks with

    Returns:
        tuple: (data2, image2) arrays with compatible masks
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


def similarity(bd, image, method="correlation"):
    """Calculate similarity of BrainData() instance with single
    BrainData or Nibabel image

    Args:
        bd: BrainData instance.
        image: (BrainData, nifti)  image to evaluate similarity
        method: (str) Type of similarity
                ['correlation', 'pearson', 'rank_correlation', 'spearman', 'dot_product', 'cosine']

    Returns:
        np.ndarray: Similarity values.

    """
    from nltools.stats import compute_similarity
    from .utils import check_brain_data

    supported_metrics = [
        "correlation",
        "pearson",
        "rank_correlation",
        "spearman",
        "dot_product",
        "cosine",
    ]
    if method not in supported_metrics:
        raise ValueError(f"method must be one of {supported_metrics}")

    image = check_brain_data(image)
    data2, image2 = check_masks(bd, image)

    # Delegate to functional core (stats.py)
    return compute_similarity(data2, image2, method=method)


def distance(bd, metric="euclidean", **kwargs):
    """Calculate distance between images within a BrainData() instance.

    Args:
        bd: BrainData instance.
        metric: (str) type of distance metric (can use any scipy.spatial.distance
                metric supported by cdist, e.g., 'euclidean', 'cityblock', 'cosine',
                'correlation', 'hamming', 'jaccard', etc.)
        **kwargs: Additional arguments passed to scipy.spatial.distance.cdist.

    Returns:
        dist: (Adjacency) Outputs a 2D distance matrix.

    """
    from scipy.spatial.distance import cdist

    from nltools.data import Adjacency

    # Use scipy.spatial.distance.cdist directly for efficiency
    # Computes pairwise distances between all images (rows)
    dist_matrix = cdist(bd.data, bd.data, metric=metric, **kwargs)
    return Adjacency(dist_matrix, matrix_type="Distance")


def multivariate_similarity(bd, images, method="ols"):
    """Predict spatial distribution of BrainData() instance from linear
    combination of other BrainData() instances or Nibabel images

    Args:
        bd: BrainData instance of data to be applied
        images: BrainData instance of weight map
        method (str): Regression method. Default: 'ols'.

    Returns:
        out: dictionary of regression statistics in BrainData
            instances {'beta','t','p','df','residual'}

    """
    # Notes:  Should add ridge, and lasso, elastic net options options
    from nltools.stats import compute_multivariate_similarity
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
    return compute_multivariate_similarity(y, X, method=method)


def apply_mask(bd, mask, resample_mask_to_brain=False):
    """Mask BrainData instance using nilearn functionality.

    Note target data will be resampled into the same space as the mask. If you would like the mask
    resampled into the BrainData space, then set resample_mask_to_brain=True.

    Args:
        bd: BrainData instance.
        mask: (BrainData or nifti object) mask to apply to BrainData object.
        resample_mask_to_brain: (bool) Will resample mask to brain space before applying mask (default=False).

    Returns:
        masked: (BrainData) masked BrainData object

    Note:
        Uses nilearn.masking.apply_mask for efficient, validated masking.
        Simplified from 47-line manual implementation to leverage nilearn's
        Cython-optimized code with better validation and memory management.

    """
    from nilearn.image import resample_to_img
    from nilearn.masking import apply_mask as nilearn_apply_mask

    from .utils import check_brain_data, check_brain_data_is_single

    masked = shallow_copy(bd)
    mask = check_brain_data(mask)
    if not check_brain_data_is_single(mask):
        raise ValueError("Mask must be a single image")

    # Handle resampling if requested (preserve existing feature)
    mask_img = mask.to_nifti()
    if resample_mask_to_brain:
        mask_img = resample_to_img(
            mask_img,
            masked.to_nifti(),
            interpolation="nearest",  # Masks are discrete, use nearest
            force_resample=True,
            copy_header=True,
        )

    # Use nilearn's apply_mask for efficient masking (C-optimized, single path, memory efficient)
    masked.data = nilearn_apply_mask(masked.to_nifti(), mask_img)

    # Update mask, voxel resolution, and space
    masked.mask = mask_img
    affine = mask_img.affine
    masked._voxel_resolution = np.abs(np.diag(affine[:3, :3]))
    from .io import detect_space

    masked._space = detect_space(masked, mask_img)

    # Preserve 1D output for single images (backward compatibility)
    if (len(masked.shape) > 1) & (masked.shape[0] == 1):
        masked.data = masked.data.flatten()

    return masked


def extract_roi(bd, mask, metric="mean", n_components=None):
    """Extract activity from mask or ROI atlas using NiftiLabelsMasker.

    This method now uses nilearn's NiftiLabelsMasker for efficient ROI extraction
    when dealing with labeled atlases (multiple ROIs).

    Args:
        bd: BrainData instance.
        mask: BrainData, nibabel image, or file path. Can be:

              - Binary mask (extracts from single ROI)
              - Labeled atlas (extracts from multiple ROIs)
        metric: Extraction method ('mean', 'median', 'pca'). Default: 'mean'
                Note: 'median' and 'pca' require additional computation after extraction
        n_components: If metric='pca', number of components to return

    Returns:
        For binary mask:

            - Single image: scalar value
            - Multiple images: 1D array of values

        For labeled atlas:

            - Single image: 1D array (one value per ROI)
            - Multiple images: 2D array (images x ROIs)
            - If metric='pca': returns components array

    Examples:
        >>> # Extract mean from binary mask
        >>> roi_values = brain.extract_roi(binary_mask)
        >>> # Extract from atlas
        >>> atlas_values = brain.extract_roi(atlas_mask)
        >>> # PCA extraction
        >>> components = brain.extract_roi(mask, metric='pca', n_components=5)
    """
    from nilearn.maskers import NiftiLabelsMasker

    from .utils import check_brain_data, check_brain_data_is_single

    metrics = ["mean", "median", "pca"]
    if metric not in metrics:
        raise NotImplementedError(f"metric must be one of {metrics}, got {metric}")

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

        if metric == "mean":
            out = masked.mean() if is_single else masked.mean(axis=1)
        elif metric == "median":
            out = masked.median() if is_single else masked.median(axis=1)
        elif metric == "pca":
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

        # Create masker based on metric
        if metric in ["mean", "median"]:
            # For mean/median, use NiftiLabelsMasker
            strategy = "mean" if metric == "mean" else "median"
            labels_masker = NiftiLabelsMasker(
                labels_img=mask_img,
                strategy=strategy,
                mask_img=bd.mask,
                standardize=False,
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

        elif metric == "pca":
            # Extract voxels from the whole atlas once, then slice by label in
            # numpy. This avoids rebuilding the nifti and re-resampling per ROI.
            if check_brain_data_is_single(bd):
                raise ValueError("Cannot run PCA on a single image")

            atlas_mask = mask_brain.copy()
            atlas_mask.data = (mask_brain.data > 0).astype(float)
            all_masked = apply_mask(bd, atlas_mask)

            # apply_mask preserves voxel ordering relative to the mask, so the
            # label vector lines up with the columns of all_masked.data.
            labels_flat = mask_brain.data[mask_brain.data > 0]
            unique_labels = np.unique(labels_flat)

            out = []
            for label in unique_labels:
                roi = shallow_copy(all_masked)
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


def icc(
    bd,
    n_subjects,
    n_sessions,
    method="icc2",
    parallel=None,
    n_jobs=-1,
    max_gpu_memory_gb=4.0,
):
    """Calculate voxel-wise intraclass correlation coefficient for data within
        BrainData class.

    Computes ICC for each voxel independently, making it highly parallelizable.
    Supports GPU acceleration for large voxel counts.

    ICC Formulas are based on:
    Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
    assessing rater reliability. Psychological bulletin, 86(2), 420.

    icc1:  x_ij = mu + beta_j + w_ij
    icc2/3:  x_ij = mu + alpha_i + beta_j + (ab)_ij + epsilon_ij

    Args:
        bd: BrainData instance.
        n_subjects: Number of subjects in the data
        n_sessions: Number of sessions per subject
        method: Type of ICC to calculate
                - 'icc1': One-way random effects (subjects random, sessions treated as interchangeable)
                - 'icc2': Two-way random effects (subjects and sessions random) (default)
                - 'icc3': Two-way mixed effects (subjects random, sessions fixed)
        parallel: Parallelization method
                - None: Single-threaded vectorized NumPy (default, memory efficient)
                - 'cpu': CPU parallelization via joblib (for medium-sized problems, 1K-10K voxels)
                - 'gpu': GPU acceleration via PyTorch (recommended for large voxel counts >10K, 10-50x speedup)
        n_jobs: Number of CPU cores (-1 = all cores). Only used when parallel='cpu'
        max_gpu_memory_gb: GPU memory budget in GB. Only used when parallel='gpu'

    Returns:
        BrainData: BrainData instance with ICC map (shape: (1, n_voxels))

    Examples:
        >>> # Typical test-retest reliability analysis
        >>> data = BrainData(...)  # Shape: (60, 238955) = 20 subjects x 3 sessions
        >>> icc_map = data.icc(n_subjects=20, n_sessions=3, method='icc2')
        >>> icc_map.shape
        (1, 238955)
        >>> # Visualize ICC map
        >>> icc_map.plot()

    Notes:
        Data must be organized such that n_images = n_subjects * n_sessions.
        Images should be ordered as: [subject1_session1, subject1_session2, ...,
        subject2_session1, ...]
    """
    from nltools.algorithms.inference.icc import compute_icc_voxelwise

    # Validate data shape
    n_images = bd.shape[0]

    if n_images != n_subjects * n_sessions:
        raise ValueError(
            f"Number of images ({n_images}) must equal n_subjects * n_sessions "
            f"({n_subjects} * {n_sessions} = {n_subjects * n_sessions}). "
            f"Make sure images are organized as: "
            f"[subject1_session1, subject1_session2, ..., subject2_session1, ...]"
        )

    # Compute voxel-wise ICC
    icc_map = compute_icc_voxelwise(
        bd.data,
        n_subjects=n_subjects,
        n_sessions=n_sessions,
        icc_type=method,
        parallel=parallel,
        n_jobs=n_jobs,
        max_gpu_memory_gb=max_gpu_memory_gb,
    )

    # Return as BrainData object (shape: (1, n_voxels))
    out = shallow_copy(bd)
    out.data = icc_map[np.newaxis, :]  # (1, n_voxels)
    out.X = pl.DataFrame()
    out.Y = pl.DataFrame()
    return out


def detrend_data(bd, method="linear"):
    """Remove linear trend from each voxel

    Args:
        bd: BrainData instance.
        method: ('linear','constant', optional) type of detrending

    Returns:
        out: (BrainData) detrended BrainData instance

    """
    from scipy.signal import detrend

    if len(bd.shape) == 1:
        raise ValueError("Make sure there is more than one image in order to detrend.")

    out = shallow_copy(bd)
    # Copy data and detrend
    out.data = detrend(bd.data.copy(), type=method, axis=0)
    return out


def r_to_z(bd):
    """Apply Fisher's r to z transformation to each element of the data
    object.

    Args:
        bd: BrainData instance.

    Returns:
        BrainData: Transformed BrainData instance.
    """
    from nltools.stats import fisher_r_to_z

    out = shallow_copy(bd)
    # fisher_r_to_z creates a new array
    out.data = fisher_r_to_z(bd.data)
    return out


def z_to_r(bd):
    """Convert z score back into r value for each element of data object.

    Args:
        bd: BrainData instance.

    Returns:
        BrainData: Transformed BrainData instance.
    """
    from nltools.stats import fisher_z_to_r

    out = shallow_copy(bd)
    # fisher_z_to_r creates a new array
    out.data = fisher_z_to_r(bd.data)
    return out


def filter_data(bd, sampling_freq=None, high_pass=None, low_pass=None, **kwargs):
    """Apply butterworth filter to data. Wraps nilearn.signal.clean.

    Does not default to detrending and standardizing like nilearn
    implementation, but this can be overridden using kwargs.

    Args:
        bd: BrainData instance.
        sampling_freq: Sampling freq in hertz (i.e. 1 / TR). Default: None.
        high_pass: High pass cutoff frequency. Default: None.
        low_pass: Low pass cutoff frequency. Default: None.
        **kwargs: Additional arguments passed to nilearn.signal.clean
                  Common options:
                  - confounds: Confound timeseries to remove
                  - sample_mask: Volumes to exclude (scrubbing)
                  - detrend: Enable detrending (default False)
                  - standardize: Enable standardization (default False)
                  - ensure_finite: Replace NaN/inf (default False)

    Returns:
        BrainData: Filtered BrainData instance

    See Also:
        nilearn.signal.clean documentation for all available options
    """
    from nilearn.signal import clean

    if sampling_freq is None:
        raise ValueError("Need to provide sampling rate (TR)!")
    if high_pass is None and low_pass is None:
        raise ValueError("high_pass and/or low_pass cutoff must be provided!")

    standardize = kwargs.get("standardize", False)
    detrend = kwargs.get("detrend", False)

    # Optimized: Use shallow copy instead of deepcopy
    out = shallow_copy(bd)
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


def standardize(bd, axis=0, method="center", verbose=True):
    """Standardize BrainData() instance.

    Args:
        bd: BrainData instance.
        axis: 0 for observations 1 for voxels (default: 0)
        method: ['center','zscore'] (default: 'center')
        verbose: If False, suppress sklearn numerical warnings that occur
            when voxels have near-zero variance. (default: True)

    Returns:
        BrainData: Standardized BrainData instance.

    """
    import warnings

    from sklearn.preprocessing import scale

    if axis == 1 and len(bd.shape) == 1:
        raise IndexError(
            "BrainData is only 3d but standardization was requested over observations"
        )

    # Optimized: Use shallow copy instead of deepcopy
    out = shallow_copy(bd)
    if method == "zscore":
        with_std = True
    elif method == "center":
        with_std = False
    else:
        raise ValueError('method must be ["center","zscore"')

    if verbose:
        out.data = scale(bd.data, axis=axis, with_std=with_std)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Numerical issues", category=UserWarning
            )
            out.data = scale(bd.data, axis=axis, with_std=with_std)
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
        bd: BrainData instance.
        scale_val: (int/float) Target value for the mean after scaling.
            Default 100.
        axis: (int or None) Axis along which to compute the mean.
            None for grand-mean scaling (default, FSL/SPM style).
            0 for voxel-wise scaling (AFNI style, each voxel scaled
            by its own temporal mean).

    Returns:
        BrainData: New BrainData instance with scaled data.

    Examples:
        >>> # Grand-mean scaling (default)
        >>> scaled = brain.scale(100.0)
        >>>
        >>> # Voxel-wise scaling (AFNI style)
        >>> scaled = brain.scale(100.0, axis=0)

    """
    out = shallow_copy(bd)
    out.data = bd.data.copy()

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
        upper: (float or str) Upper cutoff for thresholding. If string
                will interpret as percentile; can be None for one-sided
                thresholding.
        lower: (float or str) Lower cutoff for thresholding. If string
                will interpret as percentile; can be None for one-sided
                thresholding.
        bd: BrainData instance.
        binarize (bool): return binarized image respecting thresholds if
                provided, otherwise binarize on every non-zero value;
                default False
        coerce_nan (bool): coerce nan values to 0s; default True
        cluster_threshold (int): Minimum cluster size in voxels. If > 0, uses
                nilearn.image.threshold_img with cluster filtering.
                Band-pass filtering (both upper AND lower) not supported
                with cluster thresholding. Default 0 (disabled).

    Returns:
        Thresholded BrainData object.

    Note:
        When cluster_threshold=0 (default), uses fast path for basic thresholding.
        When cluster_threshold>0, uses nilearn for cluster filtering.
        Band-pass filtering (unique nltools feature) preserved when cluster_threshold=0.

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
        b = bd.copy()
        if coerce_nan:
            b.data = np.nan_to_num(b.data)

        if isinstance(threshold_val, str) and threshold_val[-1] == "%":
            threshold_val = np.percentile(b.data, float(threshold_val[:-1]))

        # Use nilearn's cluster thresholding
        out = shallow_copy(bd)
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
    b = bd.copy()

    if coerce_nan:
        b.data = np.nan_to_num(b.data)

    if isinstance(upper, str) and upper[-1] == "%":
        upper = np.percentile(b.data, float(upper[:-1]))

    if isinstance(lower, str) and lower[-1] == "%":
        lower = np.percentile(b.data, float(lower[:-1]))

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
    min_region_size=1350,
    method="local_regions",
    smoothing_fwhm=6,
    is_mask=False,
):
    """Extract brain connected regions into separate regions.

    Args:
        bd: BrainData instance.
        min_region_size (int): Minimum volume in mm3 for a region to be
                            kept.
        method (str): Type of extraction method
                            ['connected_components', 'local_regions'].
                            If 'connected_components', each component/region
                            in the image is extracted automatically by
                            labelling each region based upon the presence of
                            unique features in their respective regions.
                            If 'local_regions', each component/region is
                            extracted based on their maximum peak value to
                            define a seed marker and then using random
                            walker segementation algorithm on these
                            markers for region separation.
        smoothing_fwhm (scalar): Smooth an image to extract more sparser
                            regions. Only works for method='local_regions'.
        is_mask (bool): Whether the BrainData instance should be treated
                        as a boolean mask and if so, calls
                        connected_label_regions instead. Default: False.

    Returns:
        BrainData: BrainData instance with extracted ROIs as data.
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
        bd: BrainData instance.

    Returns:
        BrainData: BrainData instance transformed into pairwise comparisons.
    """
    from nltools.stats import transform_pairwise

    out = shallow_copy(bd)
    out.data, new_Y = transform_pairwise(bd.data, bd.Y.to_numpy())
    new_Y = np.where(np.asarray(new_Y) == -1, 0, new_Y)
    out.Y = pl.DataFrame(new_Y)
    return out


def decompose(bd, method="pca", axis="voxels", n_components=None, *args, **kwargs):
    """Decompose BrainData object

    Args:
        bd: BrainData instance.
        method: (str) Algorithm to perform decomposition
                    types=['pca','ica','nnmf','fa','dictionary','kernelpca']
        axis: dimension to decompose ['voxels','images']
        n_components: (int) number of components. If None then retain
                    as many as possible (default: None).
        **kwargs: Additional keyword arguments passed to the decomposition algorithm.

    Returns:
        output: a dictionary of decomposition parameters
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
    """Align BrainData instance to target object using functional alignment

    Alignment type can be hyperalignment or Shared Response Model. When
    using hyperalignment, `target` image can be another subject or an
    already estimated common model. When using SRM, `target` must be a previously
    estimated common model stored as a numpy array. Transformed data can be back
    projected to original data using Transformation matrix.

    See nltools.stats.align for aligning multiple BrainData instances

    Args:
        bd: BrainData instance.
        target: (BrainData) object to align to.
        method: (str) alignment method to use
            ['probabilistic_srm','deterministic_srm','procrustes']
        axis: (int) axis to align on (default: 0)

    Returns:
        out: (dict) a dictionary containing transformed object,
            transformation matrix, and the shared response matrix

    Examples:
        - Hyperalign using procrustes transform:
            >>> out = data.align(target, method='procrustes')
        - Align using shared response model:
            >>> out = data.align(target, method='probabilistic_srm', n_features=None)
        - Project aligned data into original data:
            >>> original_data = np.dot(out['transformed'].data,out['transformation_matrix'].T)
    """
    from nltools.stats import procrustes
    from .utils import check_brain_data

    if method not in ["probabilistic_srm", "deterministic_srm", "procrustes"]:
        raise ValueError(
            "Method must be ['probabilistic_srm','deterministic_srm','procrustes']"
        )

    source = bd.copy()
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

        out["transformation_matrix"] = source
        out["transformation_matrix"].data = U.dot(V).T

        out["transformed"] = data1.dot(out["transformation_matrix"].data.T)
        out["common_model"] = target
    elif method == "procrustes":
        _, transformed, out["disparity"], tf_mtx, out["scale"] = procrustes(
            data2, data1
        )
        source.data = transformed
        out["transformed"] = source
        out["common_model"] = target
        out["transformation_matrix"] = source.copy()
        out["transformation_matrix"].data = tf_mtx
    if axis == 1:
        if method == "procrustes":
            out["transformed"].data = out["transformed"].data.T
        else:
            out["transformed"] = out["transformed"].T

    return out


def smooth(bd, fwhm):
    """Apply spatial smoothing using nilearn smooth_img()

    Args:
        bd: BrainData instance.
        fwhm: (float) full width half maximum of gaussian spatial filter

    Returns:
        BrainData instance (copy with smoothed data)
    """
    from nilearn.image import smooth_img
    from nilearn.masking import apply_mask as nilearn_apply_mask

    from .utils import check_brain_data_is_single

    # Optimized: Use shallow copy instead of deepcopy, single conversion path
    out = shallow_copy(bd)

    # Single conversion: data -> nifti -> smooth -> data
    nifti = bd.to_nifti()
    smoothed_nifti = smooth_img(nifti, fwhm)
    smoothed_data = nilearn_apply_mask(smoothed_nifti, bd.mask)

    # Ensure single images remain 1D
    if check_brain_data_is_single(bd):
        out.data = smoothed_data.flatten()
    else:
        out.data = smoothed_data

    return out


def find_spikes_data(bd, global_spike_cutoff=3, diff_spike_cutoff=3):
    """Function to identify spikes from Time Series Data

    Args:
        bd: BrainData instance.
        global_spike_cutoff: (int,None) cutoff to identify spikes in global signal
                             in standard deviations, None indicates do not calculate.
        diff_spike_cutoff: (int,None) cutoff to identify spikes in average frame difference
                             in standard deviations, None indicates do not calculate.

    Returns:
        pl.DataFrame: DataFrame with spikes as indicator variables.
    """
    from nltools.stats import find_spikes

    return find_spikes(
        bd,
        global_spike_cutoff=global_spike_cutoff,
        diff_spike_cutoff=diff_spike_cutoff,
    )


def temporal_resample(bd, sampling_freq=None, target=None, target_type="hz"):
    """
    Resample BrainData timeseries to a new target frequency or number of samples
    using Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) interpolation.
    This function can up- or down-sample data.

    Note: this function can use quite a bit of RAM.

    Args:
        bd: BrainData instance.
        sampling_freq: (float) sampling frequency of data in hertz (default: None)
        target: (float) upsampling target (default: None)
        target_type: (str) type of target can be [samples,seconds,hz] (default: 'hz')

    Returns:
        upsampled BrainData instance
    """
    from scipy.interpolate import pchip

    # Optimized: Use shallow copy instead of deepcopy
    out = shallow_copy(bd)

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

    out.data = np.zeros([len(new_spacing), bd.shape[1]])
    for i in range(bd.shape[1]):
        interpolate = pchip(orig_spacing, bd.data[:, i])
        out.data[:, i] = interpolate(new_spacing)
    return out
