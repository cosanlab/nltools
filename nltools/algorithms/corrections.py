"""Multiple comparison corrections and thresholding."""

import numpy as np


def fdr(p, q=0.05):
    """Determine an FDR threshold for an array of p-values.

    Benjamini-Hochberg procedure at false discovery rate `q` (valid under
    independence or positive dependence). Written by Tal Yarkoni.

    Args:
        p (np.ndarray): Vector of p-values.
        q (float): False discovery rate level. Defaults to 0.05.

    Returns:
        float: The p-value threshold; `-1` when no p-value survives correction.
    """

    if not isinstance(p, np.ndarray):
        raise ValueError("Make sure vector of p-values is a numpy array")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("array contains p-values that are outside the range 0-1")

    s = np.sort(p)
    nvox = p.shape[0]
    null = np.array(range(1, nvox + 1), dtype="float") * q / nvox
    below = np.where(s <= null)[0]
    return s[max(below)] if len(below) else -1


def holm_bonf(p, alpha=0.05):
    """Determine a Holm-Bonferroni (step-down) threshold for an array of p-values.

    The step-down procedure applies progressively less correction to larger
    p-values. It is more conservative than FDR but much more powerful than plain
    Bonferroni correction.

    Args:
        p (np.ndarray): Vector of p-values.
        alpha (float): Family-wise alpha level. Defaults to 0.05.

    Returns:
        float: The p-value threshold; `-1` when no p-value survives correction.
    """

    if not isinstance(p, np.ndarray):
        raise ValueError("Make sure vector of p-values is a numpy array")

    s = np.sort(p)
    nvox = p.shape[0]
    null = alpha / (nvox - np.arange(1, nvox + 1) + 1)
    # Step *down*: walk up the sorted p-values and stop at the first one above
    # its own boundary. Everything after it is rejected too, so the largest
    # index below its boundary (the step-up rule `fdr` uses) is not the answer
    # here — it would report a surviving threshold for a family in which the
    # walk had already stopped.
    failed = np.where(s > null)[0]
    n_rejected = int(failed[0]) if len(failed) else nvox
    return s[n_rejected - 1] if n_rejected else -1


def _check_voxel_correspondence(stat, p, stat_name, p_name):
    """Raise unless two images describe the same voxels in the same order.

    Both functions in this module apply one image's p-values to another image
    by array position, which is only meaningful when the two arrays index the
    same voxels. v0.5.1 resolved the two images spatially, through the NIfTI
    grid; the masked-array rewrite has to check the correspondence instead.

    Args:
        stat (BrainData): Statistic image.
        p (BrainData): P-value image.
        stat_name (str): Name of the statistic argument, for the message.
        p_name (str): Name of the p-value argument, for the message.

    Raises:
        ValueError: If the data shapes, the grids, or the mask support differ.
    """
    from nltools.data.braindata.io import _check_space_match

    if stat.data.shape != p.data.shape:
        raise ValueError(
            f"{stat_name} and {p_name} must have the same shape. "
            f"Got {stat.data.shape} and {p.data.shape}"
        )

    same_grid = _check_space_match(stat.mask, p.mask)
    # Compare the support, not the raw arrays: a mask saved as float and one
    # saved as int describe the same voxels.
    same_support = same_grid and np.array_equal(
        stat.mask.get_fdata() > 0, p.mask.get_fdata() > 0
    )
    if not same_support:
        raise ValueError(
            f"{stat_name} and {p_name} must cover the same voxels: the "
            f"p-values are applied position by position. {stat_name} is "
            f"{stat.mask.shape} with affine\n{stat.mask.affine}\nand {p_name} is "
            f"{p.mask.shape} with affine\n{p.mask.affine}\n"
            "Bring them onto a common grid and mask with resample() first."
        )


def threshold(stat, p, thr=0.05, return_mask=False):
    """Threshold a statistic image by the p-values in a separate image.

    Voxels whose p-value is at or above `thr` are set to zero in a copy of `stat`.

    Args:
        stat (BrainData): Statistic image (e.g. betas or t-values).
        p (BrainData): P-value image with the same voxels as `stat`.
        thr (float): P-value threshold; voxels with `p < thr` are kept. Defaults to 0.05.
        return_mask (bool): Also return the binary thresholding mask. Defaults to False.

    Returns:
        BrainData | tuple[BrainData, BrainData]: The thresholded image, or the
            tuple `(thresholded, mask)` when `return_mask=True`.

    Raises:
        ValueError: If either argument is not a `BrainData`, or if the two
            images do not cover the same voxels on the same grid.

    Note:
        `BrainData.threshold` and `nilearn.image.threshold_img` threshold an image
        by its own values; this function is the only one that thresholds one
        image by the p-values of another.
    """
    from nltools.data import BrainData
    from nltools.data.braindata.utils import _result_from_array

    if not isinstance(stat, BrainData):
        raise ValueError("Make sure stat is a BrainData instance")

    if not isinstance(p, BrainData):
        raise ValueError("Make sure p is a BrainData instance")

    _check_voxel_correspondence(stat, p, "stat", "p")

    # Work with masked data arrays directly
    # Create binary mask: p < thr
    if thr > 0:
        p_mask = (p.data < thr).astype(float)
    else:
        p_mask = np.zeros_like(p.data, dtype=float)

    # Apply mask to stat data
    if np.sum(p_mask) > 0:
        # Threshold stat: keep only voxels where p < thr
        thresholded_data = stat.data.copy()
        thresholded_data[p_mask == 0] = 0.0
    else:
        # No voxels pass threshold - return zeros
        thresholded_data = np.zeros_like(stat.data, dtype=float)

    # Create output BrainData with same mask as stat
    out = _result_from_array(stat, thresholded_data, rows="clear")

    if return_mask:
        # Create mask BrainData with same mask as p
        mask = _result_from_array(p, p_mask, rows="clear")
        return out, mask
    return out


def multi_threshold(t_map, p_map, thresh):
    """Threshold a statistic image at several p-values and count the passes per voxel.

    Args:
        t_map (BrainData): Statistic image (e.g. t-values or betas).
        p_map (BrainData): P-value image with the same voxels as `t_map`.
        thresh (list[float]): P-value thresholds to apply.

    Returns:
        BrainData: Cumulative map. Positive values count how many thresholds a
            positive statistic passed; negative values count the same for negative
            statistics.

    Raises:
        ValueError: If either image is not a `BrainData`, if `thresh` is not a
            list, or if the two images do not cover the same voxels on the same
            grid.

    Note:
        Calling `threshold` once per level gives separate images; this returns a
        single map of the threshold hierarchy, which `nilearn.image.threshold_img`
        cannot produce.
    """
    from nltools.data import BrainData

    if not isinstance(t_map, BrainData):
        raise ValueError("Make sure t_map is a BrainData instance")

    if not isinstance(p_map, BrainData):
        raise ValueError("Make sure p_map is a BrainData instance")

    if not isinstance(thresh, list):
        raise ValueError("Make sure thresh is a list of p-values")

    _check_voxel_correspondence(t_map, p_map, "t_map", "p_map")

    # Initialize cumulative maps (working with masked data arrays)
    pos_out = np.zeros_like(t_map.data, dtype=float)
    neg_out = np.zeros_like(t_map.data, dtype=float)

    # Accumulate threshold contributions for each threshold level
    for thr in thresh:
        # Use threshold() to get thresholded image at this level
        t_thresh = threshold(t_map, p_map, thr=thr)

        # Count positive and negative contributions at this threshold level
        pos_out += (t_thresh.data > 0).astype(float)
        neg_out += (t_thresh.data < 0).astype(float)

    # Combine positive and negative cumulative maps
    # Positive values show positive threshold counts, negative show negative counts
    cumulative_data = pos_out - neg_out

    # Create output BrainData with cumulative map
    from nltools.data.braindata.utils import _result_from_array

    out = _result_from_array(t_map, cumulative_data, rows="clear")

    return out
