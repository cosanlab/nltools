"""Multiple comparison corrections and thresholding."""

import numpy as np

__all__ = [
    "fdr",
    "holm_bonf",
    "multi_threshold",
    "threshold",
]


def fdr(p, q=0.05):
    """Determine an FDR threshold for an array of p-values.

    Uses the desired false discovery rate ``q``. Written by Tal Yarkoni.

    Args:
        p: (np.array) vector of p-values
        q: (float) false discovery rate level

    Returns:
        fdr_p: (float) p-value threshold based on independence or positive
                dependence

    """

    if not isinstance(p, np.ndarray):
        raise ValueError("Make sure vector of p-values is a numpy array")
    if any(p < 0) or any(p > 1):
        raise ValueError("array contains p-values that are outside the range 0-1")

    if np.any(p > 1) or np.any(p < 0):
        raise ValueError("Does not include valid p-values.")

    s = np.sort(p)
    nvox = p.shape[0]
    null = np.array(range(1, nvox + 1), dtype="float") * q / nvox
    below = np.where(s <= null)[0]
    return s[max(below)] if len(below) else -1


def holm_bonf(p, alpha=0.05):
    """Compute Holm-Bonferroni-corrected p-values.

    This step-down procedure applies iteratively less correction to the highest
    p-values. It is a bit more conservative than FDR, but much more powerful than
    vanilla Bonferroni correction.

    Args:
        p: (np.array) vector of p-values
        alpha: (float) alpha level

    Returns:
        bonf_p: (float) p-value threshold based on bonferroni
                step-down procedure

    """

    if not isinstance(p, np.ndarray):
        raise ValueError("Make sure vector of p-values is a numpy array")

    s = np.sort(p)
    nvox = p.shape[0]
    null = alpha / (nvox - np.arange(1, nvox + 1) + 1)
    below = np.where(s <= null)[0]
    return s[max(below)] if len(below) else -1


def threshold(stat, p, thr=0.05, return_mask=False):
    """Threshold test image by p-value from p image.

    Args:
        stat: (BrainData) BrainData instance of arbitrary statistic metric
              (e.g., beta, t, etc)
        p: (BrainData) BrainData instance of p-values
        thr: (float) p-value threshold to apply
        return_mask: (bool) optionally return the thresholding mask; default False

    Returns:
        out: Thresholded BrainData instance
        mask: (optional) BrainData instance of thresholding mask if return_mask=True

    Note:
        This function provides unique functionality not available in nilearn:
        - Thresholds stat image based on p-values from separate p-value image
        - Neither nilearn.threshold_img nor BrainData.threshold() support this
        - BrainData.threshold() thresholds based on stat values themselves
        - nilearn.threshold_img() thresholds based on image intensity values

    """
    from nltools.data import BrainData

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
    out = stat.copy()
    out.data = thresholded_data

    if return_mask:
        # Create mask BrainData with same mask as p
        mask = p.copy()
        mask.data = p_mask
        return out, mask
    return out


def multi_threshold(t_map, p_map, thresh):
    """Threshold test image by multiple p-values from p image.

    Args:
        t_map: (BrainData) BrainData instance of statistic metric
            (e.g., t-statistic, beta, etc)
        p_map: (BrainData) BrainData instance of p-values
        thresh: (list) list of p-values to threshold stat image

    Returns:
        out: Thresholded BrainData instance with cumulative map
            - Positive values indicate how many thresholds were passed for positive stats
            - Negative values indicate how many thresholds were passed for negative stats

    Note:
        This function provides unique cumulative threshold map functionality:
        - Creates a single map showing which thresholds were passed
        - Different from calling threshold() multiple times (which would give separate images)
        - Useful for visualizing threshold hierarchies
        - nilearn.threshold_img() does not support cumulative multi-threshold maps

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
    out = t_map.copy()
    out.data = cumulative_data

    return out
