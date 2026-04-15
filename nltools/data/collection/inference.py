"""
BrainCollection inference functions.

Extracted from BrainCollection methods — each function takes a BrainCollection
as its first argument (``bc``) instead of ``self``.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import nibabel as nib
    from . import BrainCollection


def _device_to_parallel(device: str, n_jobs: int) -> str | None:
    """Map user-facing ``device`` + ``n_jobs`` to algorithm-level ``parallel``.

    - device='gpu' -> 'gpu'
    - device='cpu', n_jobs == 1 -> None  (single-threaded)
    - device='cpu', n_jobs != 1 -> 'cpu'
    """
    if device == "gpu":
        return "gpu"
    if device == "cpu":
        return None if n_jobs == 1 else "cpu"
    raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")


def ttest(
    bc: "BrainCollection",
    popmean: float = 0.0,
    axis: int | str = 0,
) -> tuple:
    """
    One-sample t-test across images.

    Tests whether the mean across images is significantly different from
    a population mean (default: 0). This is the voxel-wise equivalent of
    scipy.stats.ttest_1samp.

    Args:
        bc: BrainCollection to test.
        popmean: Population mean to test against (default: 0).
        axis: Axis to test across. Only axis=0 (images) supported.

    Returns:
        Tuple of (t_stat, p_value) as BrainData objects.
        Both have shape (n_obs, n_voxels) if uniform obs counts.

    Raises:
        ValueError: If images have variable observation counts.

    Examples:
        >>> t_stat, p_val = bc.ttest()  # Test mean != 0
        >>> t_stat, p_val = bc.ttest(popmean=0.5)  # Test mean != 0.5

        >>> # Threshold significant voxels
        >>> sig_mask = p_val.data < 0.05
    """
    from scipy import stats
    from ..braindata import BrainData

    axis = bc._normalize_axis(axis)
    if axis != 0:
        raise ValueError(
            "ttest only supports axis=0 (across images). "
            "For per-image tests, use apply()."
        )

    # Get tensor - requires uniform observation counts
    tensor = bc.to_tensor()  # (n_images, n_obs, n_voxels)

    # Compute t-test across axis 0
    t_stat_arr, p_val_arr = stats.ttest_1samp(tensor, popmean, axis=0)

    # Package as BrainData
    t_stat = BrainData(mask=bc._mask)
    t_stat.data = t_stat_arr

    p_val = BrainData(mask=bc._mask)
    p_val.data = p_val_arr

    return t_stat, p_val


def ttest2(
    bc: "BrainCollection",
    other: "BrainCollection",
    equal_var: bool = True,
) -> tuple:
    """
    Two-sample t-test between collections.

    Tests whether two collections have different means. This is the
    voxel-wise equivalent of scipy.stats.ttest_ind.

    Args:
        bc: First BrainCollection.
        other: Another BrainCollection to compare against.
        equal_var: If True (default), perform standard t-test assuming
            equal variances. If False, use Welch's t-test.

    Returns:
        Tuple of (t_stat, p_value) as BrainData objects.

    Raises:
        ValueError: If collections have different masks or variable obs counts.

    Examples:
        >>> t_stat, p_val = patients.ttest2(controls)
        >>> t_stat, p_val = group1.ttest2(group2, equal_var=False)  # Welch's
    """
    from scipy import stats
    from ..braindata import BrainData

    # Validate mask compatibility
    if bc._mask.shape != other._mask.shape:
        raise ValueError(
            f"Collections must have same mask shape. "
            f"Got {bc._mask.shape} and {other._mask.shape}."
        )

    # Get tensors
    tensor1 = bc.to_tensor()  # (n1, n_obs, n_voxels)
    tensor2 = other.to_tensor()  # (n2, n_obs, n_voxels)

    # Check obs counts match
    if tensor1.shape[1] != tensor2.shape[1]:
        raise ValueError(
            f"Collections must have same observation count per image. "
            f"Got {tensor1.shape[1]} and {tensor2.shape[1]}."
        )

    # Compute two-sample t-test across axis 0
    t_stat_arr, p_val_arr = stats.ttest_ind(
        tensor1, tensor2, axis=0, equal_var=equal_var
    )

    # Package as BrainData
    t_stat = BrainData(mask=bc._mask)
    t_stat.data = t_stat_arr

    p_val = BrainData(mask=bc._mask)
    p_val.data = p_val_arr

    return t_stat, p_val


def permutation_test(
    bc: "BrainCollection",
    n_permute: int = 5000,
    tail: int = 2,
    device: str = "cpu",
    max_gpu_memory_gb: float = 4.0,
    return_null: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
) -> dict:
    """
    One-sample permutation test across images (sign-flipping).

    Tests whether the mean across images is significantly different from
    zero using sign-flipping permutation. More robust than parametric
    t-test for non-normal distributions.

    This is a collection-level interface to
    nltools.algorithms.inference.one_sample_permutation_test.

    Args:
        bc: BrainCollection to test.
        n_permute: Number of permutations (default: 5000).
        tail: Test type - 1 for one-tailed, 2 for two-tailed (default).
        device: Compute device: 'cpu' (default) or 'gpu' (via PyTorch).
        max_gpu_memory_gb: GPU memory budget (default: 4.0 GB).
        return_null: If True, include null distribution in result.
        n_jobs: Number of CPU jobs (-1 = all cores, 1 = single-threaded).
        random_state: Random seed for reproducibility.

    Returns:
        dict with keys:
            - 'mean': BrainData with observed mean across images
            - 'p': BrainData with p-values
            - 'null_dist': np.ndarray (if return_null=True)
            - 'device': compute device used

    Raises:
        ValueError: If images have variable observation counts.

    Examples:
        >>> result = bc.permutation_test(n_permute=5000)
        >>> mean_bd, p_bd = result['mean'], result['p']

        >>> # With GPU acceleration
        >>> result = bc.permutation_test(device='gpu')
    """
    parallel = _device_to_parallel(device, n_jobs)
    from nltools.stats import one_sample_permutation_test
    from nltools.utils import attempt_to_import
    from ..braindata import BrainData

    tqdm = attempt_to_import("tqdm", "tqdm")

    # Get tensor - requires uniform observation counts
    tensor = bc.to_tensor()  # (n_images, n_obs, n_voxels)
    n_images, n_obs, n_voxels = tensor.shape

    # For each observation/timepoint, run permutation test across images
    mean_results = []
    p_results = []
    null_dists = [] if return_null else None

    iterator = range(n_obs)
    if tqdm is not None:
        iterator = tqdm.tqdm(iterator, desc="Permutation tests")

    for obs_idx in iterator:
        # Data for this observation: (n_images, n_voxels)
        data = tensor[:, obs_idx, :]

        result = one_sample_permutation_test(
            data,
            n_permute=n_permute,
            tail=tail,
            return_null=return_null,
            parallel=parallel,
            n_jobs=n_jobs,
            max_gpu_memory_gb=max_gpu_memory_gb,
            random_state=random_state,
        )

        mean_results.append(result["mean"])
        p_results.append(result["p"])
        if return_null:
            null_dists.append(result["null_dist"])

    # Stack results: (n_obs, n_voxels)
    mean_arr = np.vstack(mean_results)
    p_arr = np.vstack(p_results)

    # Package as BrainData
    mean_bd = BrainData(mask=bc._mask)
    mean_bd.data = mean_arr if n_obs > 1 else mean_arr.squeeze()

    p_bd = BrainData(mask=bc._mask)
    p_bd.data = p_arr if n_obs > 1 else p_arr.squeeze()

    result_dict = {
        "mean": mean_bd,
        "p": p_bd,
        "device": device,
    }

    if return_null:
        result_dict["null_dist"] = np.array(null_dists)

    return result_dict


def permutation_test2(
    bc: "BrainCollection",
    other: "BrainCollection",
    n_permute: int = 5000,
    tail: int = 2,
    device: str = "cpu",
    max_gpu_memory_gb: float = 4.0,
    return_null: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
) -> dict:
    """
    Two-sample permutation test between collections.

    Tests whether two collections have different means using group
    label permutation. More robust than parametric t-test.

    Args:
        bc: First BrainCollection.
        other: Another BrainCollection to compare against.
        n_permute: Number of permutations (default: 5000).
        tail: Test type - 1 for one-tailed, 2 for two-tailed (default).
        device: Compute device: 'cpu' (default) or 'gpu' (via PyTorch).
        max_gpu_memory_gb: GPU memory budget (default: 4.0 GB).
        return_null: If True, include null distribution in result.
        n_jobs: Number of CPU jobs (-1 = all cores, 1 = single-threaded).
        random_state: Random seed for reproducibility.

    Returns:
        dict with keys:
            - 'mean_diff': BrainData with observed mean difference
            - 'p': BrainData with p-values
            - 'null_dist': np.ndarray (if return_null=True)
            - 'device': compute device used

    Examples:
        >>> result = patients.permutation_test2(controls)
        >>> diff_bd, p_bd = result['mean_diff'], result['p']
    """
    parallel = _device_to_parallel(device, n_jobs)
    from nltools.stats import two_sample_permutation_test
    from nltools.utils import attempt_to_import
    from ..braindata import BrainData

    tqdm = attempt_to_import("tqdm", "tqdm")

    # Validate mask compatibility
    if bc._mask.shape != other._mask.shape:
        raise ValueError(
            f"Collections must have same mask shape. "
            f"Got {bc._mask.shape} and {other._mask.shape}."
        )

    # Get tensors
    tensor1 = bc.to_tensor()  # (n1, n_obs, n_voxels)
    tensor2 = other.to_tensor()  # (n2, n_obs, n_voxels)

    if tensor1.shape[1] != tensor2.shape[1]:
        raise ValueError(
            f"Collections must have same observation count. "
            f"Got {tensor1.shape[1]} and {tensor2.shape[1]}."
        )

    n_obs = tensor1.shape[1]

    diff_results = []
    p_results = []
    null_dists = [] if return_null else None

    iterator = range(n_obs)
    if tqdm is not None:
        iterator = tqdm.tqdm(iterator, desc="Two-sample permutation tests")

    for obs_idx in iterator:
        data1 = tensor1[:, obs_idx, :]  # (n1, n_voxels)
        data2 = tensor2[:, obs_idx, :]  # (n2, n_voxels)

        result = two_sample_permutation_test(
            data1,
            data2,
            n_permute=n_permute,
            tail=tail,
            return_null=return_null,
            parallel=parallel,
            n_jobs=n_jobs,
            max_gpu_memory_gb=max_gpu_memory_gb,
            random_state=random_state,
        )

        diff_results.append(result["mean_diff"])
        p_results.append(result["p"])
        if return_null:
            null_dists.append(result["null_dist"])

    # Stack results
    diff_arr = np.vstack(diff_results)
    p_arr = np.vstack(p_results)

    diff_bd = BrainData(mask=bc._mask)
    diff_bd.data = diff_arr if n_obs > 1 else diff_arr.squeeze()

    p_bd = BrainData(mask=bc._mask)
    p_bd.data = p_arr if n_obs > 1 else p_arr.squeeze()

    result_dict = {
        "mean_diff": diff_bd,
        "p": p_bd,
        "device": device,
    }

    if return_null:
        result_dict["null_dist"] = np.array(null_dists)

    return result_dict


def anova(
    bc: "BrainCollection",
    groups: str | list | np.ndarray,
) -> tuple:
    """
    One-way ANOVA across groups defined by metadata.

    Tests whether group means differ significantly. This is the
    voxel-wise equivalent of scipy.stats.f_oneway.

    Args:
        bc: BrainCollection to test.
        groups: Group assignment for each image. Can be:
            - str: Column name in metadata
            - list/array: Group labels of length n_images

    Returns:
        Tuple of (F_stat, p_value) as BrainData objects.

    Raises:
        ValueError: If groups length doesn't match n_images.
        KeyError: If group column not found in metadata.

    Examples:
        >>> # Groups from metadata column
        >>> f_stat, p_val = bc.anova('condition')

        >>> # Explicit group labels
        >>> groups = ['control'] * 10 + ['patient'] * 15
        >>> f_stat, p_val = bc.anova(groups)
    """
    from scipy import stats
    from ..braindata import BrainData

    # Resolve groups
    if isinstance(groups, str):
        if groups not in bc._metadata.columns:
            raise KeyError(
                f"Column '{groups}' not found in metadata. "
                f"Available: {list(bc._metadata.columns)}"
            )
        group_labels = bc._metadata[groups].to_numpy()
    else:
        group_labels = np.asarray(groups)
        if len(group_labels) != bc.n_images:
            raise ValueError(
                f"groups length ({len(group_labels)}) must match "
                f"n_images ({bc.n_images})"
            )

    # Get tensor
    tensor = bc.to_tensor()  # (n_images, n_obs, n_voxels)
    n_images, n_obs, n_voxels = tensor.shape

    # Get unique groups
    unique_groups = np.unique(group_labels)
    if len(unique_groups) < 2:
        raise ValueError("ANOVA requires at least 2 groups")

    # Compute F-test for each observation
    f_results = []
    p_results = []

    for obs_idx in range(n_obs):
        data = tensor[:, obs_idx, :]  # (n_images, n_voxels)

        # Split by groups
        group_data = [data[group_labels == g] for g in unique_groups]

        # F-test across groups
        f_stat_arr, p_val_arr = stats.f_oneway(*group_data)

        f_results.append(f_stat_arr)
        p_results.append(p_val_arr)

    # Stack results
    f_arr = np.vstack(f_results) if n_obs > 1 else np.array(f_results[0])
    p_arr = np.vstack(p_results) if n_obs > 1 else np.array(p_results[0])

    # Package as BrainData
    f_stat = BrainData(mask=bc._mask)
    f_stat.data = f_arr

    p_val = BrainData(mask=bc._mask)
    p_val.data = p_arr

    return f_stat, p_val


def isc(
    bc: "BrainCollection",
    method: str = "loo",
    roi_mask: "nib.Nifti1Image | Path | str | None" = None,
    radius: float | None = 6.0,
    metric: str = "median",
    device: str = "cpu",
    n_jobs: int = -1,
    progress_bar: bool = False,
) -> dict:
    """
    Compute intersubject correlation (ISC) across the collection.

    ISC measures the similarity of brain responses across subjects,
    computed by correlating each subject's timeseries with others.

    Args:
        bc: BrainCollection to compute ISC on.
        method: ISC computation method:
            - 'loo': Leave-one-out (correlate each subject with mean of others)
            - 'pairwise': All pairwise correlations between subjects
        roi_mask: If provided, compute ISC per ROI. Can be:
            - NIfTI image with integer labels (atlas/parcellation)
            - Path to parcellation file
        radius: Searchlight radius in mm. If None, use voxelwise mode.
            Ignored if roi_mask is provided.
        metric: Summary statistic for aggregating ISC values:
            - 'median': Robust to outliers (default)
            - 'mean': Fisher z-transformed mean
        device: Compute device: 'cpu' (default) or 'gpu' (via PyTorch).
        n_jobs: Number of CPU jobs (-1 = all cores, 1 = single-threaded).
        progress_bar: Show progress bar during extraction.

    Returns:
        Dictionary with:
            - 'isc': BrainData with ISC values
            - 'method': ISC method used ('loo' or 'pairwise')
            - 'extraction': Extraction mode ('roi', 'searchlight', 'voxelwise')
            - 'n_subjects': Number of subjects
            - 'extraction_info': Dict with extraction metadata

    Examples:
        >>> # ROI-based ISC using atlas
        >>> result = bc.isc(roi_mask="atlas.nii.gz")
        >>> result['isc'].plot()

        >>> # Searchlight ISC
        >>> result = bc.isc(radius=10.0)

        >>> # Voxelwise ISC
        >>> result = bc.isc(radius=None)

    Notes:
        For permutation testing, see BrainCollection.isc_test() (requires
        discussion of statistical methodology first).
    """
    from nltools.algorithms.inference.isc import (
        _compute_loo_isc,
        _compute_pairwise_isc,
    )

    # Extract data
    extracted, extraction_info = extract_for_isc(
        bc,
        roi_mask=roi_mask,
        radius=radius,
        progress_bar=progress_bar,
    )

    # Data is (n_obs, n_subjects, n_features)
    # ISC functions expect this shape

    # Compute ISC
    if method == "loo":
        # LOO ISC: (n_subjects, n_features)
        backend = "torch" if device == "gpu" else "numpy"
        loo_values = _compute_loo_isc(extracted, backend=backend)

        # Aggregate across subjects
        if metric == "median":
            isc_values = np.median(loo_values, axis=0)
        elif metric == "mean":
            z = np.arctanh(np.clip(loo_values, -0.9999, 0.9999))
            isc_values = np.tanh(np.mean(z, axis=0))
        else:
            raise ValueError(f"metric must be 'median' or 'mean', got {metric}")

    elif method == "pairwise":
        # Pairwise ISC: (n_pairs, n_features)
        pairwise = _compute_pairwise_isc(extracted, backend="numpy")

        # Aggregate across pairs
        if metric == "median":
            isc_values = np.nanmedian(pairwise, axis=0)
        elif metric == "mean":
            z = np.arctanh(np.clip(pairwise, -0.9999, 0.9999))
            isc_values = np.tanh(np.nanmean(z, axis=0))
        else:
            raise ValueError(f"metric must be 'median' or 'mean', got {metric}")

    else:
        raise ValueError(f"method must be 'loo' or 'pairwise', got {method}")

    # Project back to brain space
    isc_brain = project_to_brain(bc, isc_values, extraction_info)

    return {
        "isc": isc_brain,
        "method": method,
        "extraction": extraction_info["mode"],
        "n_subjects": len(bc),
        "extraction_info": extraction_info,
    }


def isc_test(
    bc: "BrainCollection",
    method: str = "loo",
    roi_mask: "nib.Nifti1Image | Path | str | None" = None,
    radius: float | None = 6.0,
    n_permute: int = 5000,
    permutation_method: str = "bootstrap",
    metric: str = "median",
    tail: int = 2,
    ci_percentile: float = 95,
    device: str = "cpu",
    return_null: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
    progress_bar: bool = False,
) -> dict:
    """
    Compute ISC with permutation testing for statistical inference.

    This method combines ISC computation with permutation testing to
    determine statistical significance. It uses the same extraction
    pipeline as isc() and wraps isc_permutation_test().

    Args:
        bc: BrainCollection to test.
        method: ISC computation method:
            - 'loo': Leave-one-out (correlate each subject with mean of others)
            - 'pairwise': All pairwise correlations between subjects
        roi_mask: If provided, compute ISC per ROI. Can be:
            - NIfTI image with integer labels (atlas/parcellation)
            - Path to parcellation file
        radius: Searchlight radius in mm. If None, use voxelwise mode.
            Ignored if roi_mask is provided.
        n_permute: Number of permutations/bootstrap iterations. Default 5000.
        permutation_method: Method for null distribution:

            - 'bootstrap': Subject-wise bootstrap (default, Chen et al. 2016).
              Tests whether observed ISC differs from random groupings.
            - 'circle_shift': Circular time-shift (preserves autocorrelation).
              Tests for temporally-locked shared signal.
            - 'phase_randomize': FFT phase randomization (preserves power spectrum).
              Tests for nonlinear temporal coupling.
        metric: Summary statistic for aggregating ISC values:
            - 'median': Robust to outliers (default)
            - 'mean': Fisher z-transformed mean
        tail: One-tailed (1) or two-tailed (2) test. Default 2.
        ci_percentile: Confidence interval percentile (e.g., 95). Default 95.
        device: Compute device: 'cpu' (default) or 'gpu' (via PyTorch).
        n_jobs: Number of CPU jobs (-1 = all cores, 1 = single-threaded).
        random_state: Random seed for reproducibility.
        return_null: If True, include null distribution in results.
        progress_bar: Show progress bar during extraction and permutation.

    Returns:
        Dictionary with:
            - 'isc': BrainData with ISC values
            - 'p': BrainData with p-values (Phipson-Smyth corrected)
            - 'ci': Tuple of (lower, upper) BrainData confidence intervals
            - 'method': ISC method used ('loo' or 'pairwise')
            - 'permutation_method': Permutation method used
            - 'extraction': Extraction mode ('roi', 'searchlight', 'voxelwise')
            - 'n_subjects': Number of subjects
            - 'n_permute': Number of permutations
            - 'null_dist': (optional) Null distribution array if return_null=True

    Examples:
        >>> # ROI-based ISC with permutation testing
        >>> result = bc.isc_test(roi_mask="atlas.nii.gz", n_permute=5000)
        >>> sig_mask = result['p'].data < 0.05
        >>> print(f"Significant ROIs: {sig_mask.sum()}")

        >>> # Searchlight ISC testing
        >>> result = bc.isc_test(radius=10.0)
        >>> result['isc'].plot()  # Show ISC values
        >>> result['p'].plot()    # Show p-values

        >>> # Voxelwise with phase randomization (tests temporal coupling)
        >>> result = bc.isc_test(
        ...     radius=None,
        ...     permutation_method='phase_randomize',
        ...     device='gpu'
        ... )

    Notes:
        - Bootstrap (default) is recommended for standard ISC inference
          (Chen et al. 2016). It tests whether ISC is significant at
          the group level.
        - Circle_shift and phase_randomize are more specialized - they
          test for temporally-structured shared signal beyond what's
          explained by autocorrelation or spectral structure alone.
        - For large voxelwise analyses, bootstrap is much faster as it
          resamples pre-computed values rather than recomputing ISC.

    References:
        Chen, G., et al. (2016). Untangling the relatedness among
        correlations, part I: nonparametric approaches to inter-subject
        correlation analysis at the group level. NeuroImage, 142, 248-259.
    """
    from nltools.algorithms.inference.isc import isc_permutation_test

    parallel = _device_to_parallel(device, n_jobs)
    # Map method names
    summary_statistic = "leave-one-out" if method == "loo" else method

    # Extract data
    extracted, extraction_info = extract_for_isc(
        bc,
        roi_mask=roi_mask,
        radius=radius,
        progress_bar=progress_bar,
    )

    # Run permutation test
    result = isc_permutation_test(
        data=extracted,
        n_permute=n_permute,
        metric=metric,
        summary_statistic=summary_statistic,
        method=permutation_method,
        ci_percentile=ci_percentile,
        tail=tail,
        return_null=return_null,
        progress_bar=progress_bar,
        parallel=parallel,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    # Project results to brain space
    isc_brain = project_to_brain(bc, result["isc"], extraction_info)
    p_brain = project_to_brain(bc, result["p"], extraction_info)
    ci_lower = project_to_brain(bc, result["ci"][0], extraction_info)
    ci_upper = project_to_brain(bc, result["ci"][1], extraction_info)

    output = {
        "isc": isc_brain,
        "p": p_brain,
        "ci": (ci_lower, ci_upper),
        "method": method,
        "permutation_method": permutation_method,
        "extraction": extraction_info["mode"],
        "n_subjects": len(bc),
        "n_permute": n_permute,
    }

    if return_null:
        output["null_dist"] = result.get("null_dist")

    return output


def extract_for_isc(
    bc: "BrainCollection",
    roi_mask: "nib.Nifti1Image | Path | str | None" = None,
    radius: float | None = 6.0,
    progress_bar: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Extract data for ISC computation.

    Memory-efficient extraction that processes one subject at a time.
    Returns data in ISC-compatible format: (n_obs, n_subjects, n_features).

    Args:
        bc: BrainCollection to extract from.
        roi_mask: If provided, extract mean per ROI. Can be:
            - NIfTI image with integer labels (atlas/parcellation)
            - Path to parcellation file
        radius: Searchlight radius in mm. If None, use voxelwise mode.
            Ignored if roi_mask is provided.
        progress_bar: Show progress bar during extraction.

    Returns:
        Tuple of:
            - extracted_data: Array of shape (n_obs, n_subjects, n_features)
            - extraction_info: Dict with metadata for projection back:
                - 'mode': 'roi', 'searchlight', or 'voxelwise'
                - 'n_features': Number of features
                - 'roi_mask': ROI mask if mode='roi'
                - 'neighborhoods': SphereNeighborhoods if mode='searchlight'
    """
    n_obs = bc.shape[1]
    if n_obs is None:
        raise ValueError(
            "ISC requires uniform observation counts across subjects. "
            f"Got variable counts: {[bd.shape[0] for bd in bc]}"
        )

    # Determine extraction mode
    if roi_mask is not None:
        return extract_roi(bc, roi_mask, progress_bar)
    elif radius is not None:
        return extract_searchlight(bc, radius, progress_bar)
    else:
        return extract_voxelwise(bc, progress_bar)


def extract_roi(
    bc: "BrainCollection",
    roi_mask: "nib.Nifti1Image | Path | str",
    progress_bar: bool = False,
) -> tuple[np.ndarray, dict]:
    """Extract mean signal per ROI."""
    import nibabel as nib
    from nilearn.maskers import NiftiLabelsMasker
    from nltools.utils import attempt_to_import

    tqdm = attempt_to_import("tqdm", "tqdm")

    # Load ROI mask if path
    if isinstance(roi_mask, (str, Path)):
        roi_mask = nib.load(roi_mask)

    # Create masker
    masker = NiftiLabelsMasker(
        labels_img=roi_mask,
        standardize=False,
        resampling_target="data",
    )

    n_subjects = len(bc)
    n_obs = bc.shape[1]

    # Get number of ROIs from first subject
    first_img = bc[0].to_nifti()
    first_signals = masker.fit_transform(first_img)
    n_rois = first_signals.shape[1]

    # Preallocate output: (n_obs, n_subjects, n_rois)
    extracted = np.zeros((n_obs, n_subjects, n_rois), dtype=np.float32)
    extracted[:, 0, :] = first_signals

    # Extract remaining subjects
    iterator = range(1, n_subjects)
    if progress_bar and tqdm is not None:
        iterator = tqdm.tqdm(iterator, desc="Extracting ROIs", unit="subjects")

    for i in iterator:
        img = bc[i].to_nifti()
        signals = masker.transform(img)
        extracted[:, i, :] = signals

    extraction_info = {
        "mode": "roi",
        "n_features": n_rois,
        "roi_mask": roi_mask,
        "masker": masker,
    }

    return extracted, extraction_info


def extract_searchlight(
    bc: "BrainCollection",
    radius: float,
    progress_bar: bool = False,
) -> tuple[np.ndarray, dict]:
    """Extract mean signal per searchlight sphere."""
    from nltools.data.braindata.neighborhoods import compute_searchlight_neighborhoods
    from nltools.utils import attempt_to_import

    tqdm = attempt_to_import("tqdm", "tqdm")

    n_subjects = len(bc)
    n_obs = bc.shape[1]
    n_voxels = bc.n_voxels

    # Get cached neighborhoods
    neighborhoods = compute_searchlight_neighborhoods(
        bc._mask, radius_mm=radius, use_cache=True
    )

    # Preallocate output: (n_obs, n_subjects, n_voxels)
    extracted = np.zeros((n_obs, n_subjects, n_voxels), dtype=np.float32)

    # Extract each subject
    iterator = range(n_subjects)
    if progress_bar and tqdm is not None:
        iterator = tqdm.tqdm(iterator, desc="Extracting searchlight", unit="subjects")

    for subj_idx in iterator:
        bd = bc[subj_idx]
        data = bd.data  # (n_obs, n_voxels)

        # Compute mean per sphere neighborhood
        for voxel_idx, neighbor_indices in neighborhoods.iter_neighborhoods():
            extracted[:, subj_idx, voxel_idx] = data[:, neighbor_indices].mean(axis=1)

    extraction_info = {
        "mode": "searchlight",
        "n_features": n_voxels,
        "radius": radius,
        "neighborhoods": neighborhoods,
    }

    return extracted, extraction_info


def extract_voxelwise(
    bc: "BrainCollection",
    progress_bar: bool = False,
) -> tuple[np.ndarray, dict]:
    """Extract raw voxel data."""
    from nltools.utils import attempt_to_import

    tqdm = attempt_to_import("tqdm", "tqdm")

    n_subjects = len(bc)
    n_obs = bc.shape[1]
    n_voxels = bc.n_voxels

    # Preallocate output: (n_obs, n_subjects, n_voxels)
    extracted = np.zeros((n_obs, n_subjects, n_voxels), dtype=np.float32)

    # Extract each subject
    iterator = range(n_subjects)
    if progress_bar and tqdm is not None:
        iterator = tqdm.tqdm(iterator, desc="Extracting voxels", unit="subjects")

    for subj_idx in iterator:
        bd = bc[subj_idx]
        extracted[:, subj_idx, :] = bd.data

    extraction_info = {
        "mode": "voxelwise",
        "n_features": n_voxels,
    }

    return extracted, extraction_info


def project_to_brain(
    bc: "BrainCollection",
    values: np.ndarray,
    extraction_info: dict,
):
    """
    Project ISC values back to brain space.

    Args:
        bc: BrainCollection (used for mask).
        values: ISC values, shape depends on extraction mode:
            - ROI mode: (n_rois,)
            - Searchlight/voxelwise: (n_voxels,)
        extraction_info: Dict from extract_for_isc with mode info.

    Returns:
        BrainData with ISC values in brain space.
    """
    from ..braindata import BrainData

    mode = extraction_info["mode"]

    if mode == "roi":
        # For ROI mode, values are per-ROI, not per-voxel
        # Return a BrainData with ROI values directly (not expanded to voxels)
        result = BrainData(mask=bc._mask)
        result.data = values
        return result

    elif mode in ("searchlight", "voxelwise"):
        # Direct assignment
        result = BrainData(mask=bc._mask)
        result.data = values
        return result

    else:
        raise ValueError(f"Unknown extraction mode: {mode}")
