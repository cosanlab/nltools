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
    below = np.where(s <= null)[0]
    return s[max(below)] if len(below) else -1


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

    # Ensure stat and p have compatible shapes
    if len(stat.data) != len(p.data):
        raise ValueError(
            f"stat and p must have the same number of voxels. "
            f"Got {len(stat.data)} and {len(p.data)}"
        )

    # Work with masked data arrays directly
    # Create binary mask: p < thr
    if thr > 0:
        p_mask = (p.data < thr).astype(float)
    else:
        p_mask = np.zeros(len(p.data), dtype=float)

    # Apply mask to stat data
    if np.sum(p_mask) > 0:
        # Threshold stat: keep only voxels where p < thr
        thresholded_data = stat.data.copy()
        thresholded_data[p_mask == 0] = 0.0
    else:
        # No voxels pass threshold - return zeros
        thresholded_data = np.zeros(len(stat.data), dtype=float)

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

    # Ensure compatible shapes
    if len(t_map.data) != len(p_map.data):
        raise ValueError(
            f"t_map and p_map must have the same number of voxels. "
            f"Got {len(t_map.data)} and {len(p_map.data)}"
        )

    # Initialize cumulative maps (working with masked data arrays)
    pos_out = np.zeros(len(t_map.data), dtype=float)
    neg_out = np.zeros(len(t_map.data), dtype=float)

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
