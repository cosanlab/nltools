"""Intersubject correlation, functional connectivity, and phase synchrony."""

__all__ = ["isc", "isc_group", "isfc", "isps"]

import numpy as np
import polars as pl
from scipy.signal import hilbert
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state

from nltools.algorithms.inference import isc_permutation_test
from nltools.algorithms.inference.matrix import _compute_cross_correlation

from .timeseries import (
    _butter_bandpass_filter,
    _phase_mean_angle,
    _phase_rayleigh_p,
    _phase_vector_length,
)


def _as_ndarray(data, name="data"):
    """Coerce a numpy array, polars DataFrame, or pandas DataFrame to a numpy array."""
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, pl.DataFrame):
        return data.to_numpy()
    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None and isinstance(data, pd.DataFrame):
        return data.values
    raise ValueError(
        f"{name} must be a numpy array, polars DataFrame, or pandas DataFrame"
    )


def _bootstrap_isc(
    similarity_matrix, metric="median", exclude_self_corr=True, random_state=None
):
    """Helper function to compute bootstrapped ISC from Adjacency Instance

    This function implements the subject-wise bootstrap method discussed in Chen et al., 2016.

    Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C., Israel, R. B.,
    & Cox, R. W. (2016). Untangling the relatedness among correlations, part I:
    nonparametric approaches to inter-subject correlation analysis at the group level.
    NeuroImage, 142, 248-259.

    Args:

        similarity_matrix: (Adjacency) Adjacency matrix of pairwise correlation values
        metric: (str) type of summary statistic (Default: median)
        exclude_self_corr: (bool) set correlations with random draws of same subject to NaN (Default: True)
        random_state: random_state instance for permutation

    Returns:

        isc: summary statistic of bootstrapped similarity matrix

    """
    from nltools.data import Adjacency

    if not isinstance(similarity_matrix, Adjacency):
        raise ValueError("similarity_matrix must be an Adjacency instance.")

    random_state = check_random_state(random_state)

    square = similarity_matrix.squareform()
    n_sub = square.shape[0]
    np.fill_diagonal(square, 1)

    bootstrap_subject = sorted(
        random_state.choice(np.arange(n_sub), size=n_sub, replace=True)
    )
    bootstrap_sample = Adjacency(
        square[bootstrap_subject, :][:, bootstrap_subject], matrix_type="similarity"
    )

    if exclude_self_corr:
        bootstrap_sample.data[bootstrap_sample.data == 1] = np.nan

    if metric == "mean":
        return np.tanh(bootstrap_sample.r_to_z().mean())
    if metric == "median":
        return bootstrap_sample.median()
    raise ValueError(f"metric must be 'mean' or 'median', got {metric!r}")


def isc(
    data,
    n_samples=5000,
    metric="median",
    method="bootstrap",
    ci_percentile=95,
    exclude_self_corr=True,
    return_null=False,
    tail=2,
    n_jobs=-1,
    random_state=None,
    sim_metric="correlation",
):
    """Compute pairwise intersubject correlation from observations by subjects array.

    This function computes pairwise intersubject correlations (ISC) using the median as recommended by Chen
    et al., 2016). However, if the mean is preferred, we compute the mean correlation after performing
    the fisher r-to-z transformation and then convert back to correlations to minimize artificially
    inflating the correlation values.

    There are currently three different methods to compute p-values. These include the classic methods for
    computing permuted time-series by either circle-shifting the data or phase-randomizing the data
    (see Lancaster et al., 2018). These methods create random surrogate data while preserving the temporal
    autocorrelation inherent to the signal. By default, we use the subject-wise bootstrap method from
    Chen et al., 2016. Instead of recomputing the pairwise ISC using circle_shift or phase_randomization methods,
    this approach uses the computationally more efficient method of bootstrapping the subjects
    and computing a new pairwise similarity matrix with randomly selected subjects with replacement.
    If the same subject is selected multiple times, we set the perfect correlation to a nan with
    (exclude_self_corr=True). We compute the p-values using the percentile method using the same
    method in Brainiak.

    Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C., Israel, R. B.,
    & Cox, R. W. (2016). Untangling the relatedness among correlations, part I:
    nonparametric approaches to inter-subject correlation analysis at the group level.
    NeuroImage, 142, 248-259.

    Hall, P., & Wilson, S. R. (1991). Two guidelines for bootstrap hypothesis testing.
    Biometrics, 757-762.

    Lancaster, G., Iatsenko, D., Pidde, A., Ticcinelli, V., & Stefanovska, A. (2018).
    Surrogate data for hypothesis testing of physical systems. Physics Reports, 748, 1-60.

    This function is a wrapper around `isc_permutation_test` from the inference module,
    which provides optimized implementations with CPU-parallel and GPU acceleration support.

    Args:
        data: (pd.DataFrame, np.array) observations by subjects where isc is computed across subjects
        n_samples: (int) number of random samples/bootstraps
        metric: (str) type of isc summary metric ['mean','median']
        method: (str) method to compute p-values ['bootstrap', 'circle_shift','phase_randomize'] (default: bootstrap)
        tail: (int) either 1 for one-tail or 2 for two-tailed test (default: 2)
        n_jobs: (int) The number of CPUs to use to do the computation. -1 means all CPUs.
        return_null: (bool) Return the permutation distribution along with the p-value; default False
        sim_metric: (str) pairwise distance metric. See sklearn's pairwise_distances for valid inputs (default: correlation)

    Returns:
        stats: (dict) dictionary of permutation results ['isc', 'p', 'ci', 'null_distribution']

    """
    data = _as_ndarray(data)

    if metric not in ["mean", "median"]:
        raise ValueError("metric must be ['mean', 'median']")

    # Call inference module function with parameter mapping
    result = isc_permutation_test(
        data,
        n_permute=n_samples,  # Map n_samples -> n_permute
        metric=metric,
        summary_statistic="pairwise",  # Explicitly set to match original behavior
        method=method,
        ci_percentile=ci_percentile,
        tail=tail,
        n_jobs=n_jobs,
        random_state=random_state,
        return_null=return_null,
        exclude_self_corr=exclude_self_corr,
        sim_metric=sim_metric,
        progress_bar=False,  # Disable progress bar for backward compatibility
    )

    # Map return keys to match original function signature
    # Inference module returns 'null_dist', but original function returned 'null_distribution'
    if return_null and "null_dist" in result:
        result["null_distribution"] = result.pop("null_dist")

    # Return dict with same keys as original function
    return result


def _compute_isc_group(group1, group2, metric="median"):
    """Helper function to compute intersubject correlation difference between two groups from either:
    1) an observations by subjects array
    2) or an Adjacency instance of a similarity matrix.

    Args:
        group1: (pd.DataFrame, np.array, Adjacency) group1 data or similarity matrix
        group2: (pd.DataFrame, np.array,Adjacency)  group2 data or similarity matrix
        metric: (str) type of isc metric ['mean','median']

    Returns:
        isc: (float) intersubject correlation coefficient difference across groups

    """
    from nltools.data import Adjacency

    def _is_matrix_like(x):
        if isinstance(x, np.ndarray):
            return True
        if isinstance(x, pl.DataFrame):
            return True
        try:
            import pandas as pd
        except ImportError:
            return False
        return isinstance(x, pd.DataFrame)

    if _is_matrix_like(group1) and _is_matrix_like(group2):
        group1 = _as_ndarray(group1, name="group1")
        group2 = _as_ndarray(group2, name="group2")
        if group1.shape[0] != group2.shape[0]:
            raise ValueError(
                "group1 has a different number of observations from group2."
            )

        similarity_group1 = Adjacency(
            1 - pairwise_distances(group1.T, metric="correlation"),
            matrix_type="similarity",
        )
        similarity_group2 = Adjacency(
            1 - pairwise_distances(group2.T, metric="correlation"),
            matrix_type="similarity",
        )
    elif isinstance(group1, (Adjacency)) and isinstance(group2, (Adjacency)):
        similarity_group1 = group1
        similarity_group2 = group2
    else:
        raise ValueError(
            "group1 and group2 data must either be a observation by feature matrix or Adjacency instances."
        )

    if metric == "mean":
        isc_group1 = np.tanh(similarity_group1.r_to_z().mean())
        isc_group2 = np.tanh(similarity_group2.r_to_z().mean())
    elif metric == "median":
        isc_group1 = similarity_group1.median()
        isc_group2 = similarity_group2.median()
    return isc_group1 - isc_group2


def _permute_isc_group(similarity_matrix, group, metric="median", random_state=None):
    """Helper function to compute ISC differences between groups from Adjacency instance

    This function implements the subject-wise permutation method discussed in Chen et al., 2016.

    Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C., Israel, R. B.,
    & Cox, R. W. (2016). Untangling the relatedness among correlations, part I:
    nonparametric approaches to inter-subject correlation analysis at the group level.
    NeuroImage, 142, 248-259.

    Args:

        similarity_matrix: (Adjacency) Adjacency matrix of pairwise correlation values
        group: (numpy array) Array indicating group 1 and group 2 order (i.e., np.array([1,1,1,2,2,2]))
        metric: (str) type of summary statistic (Default: median)
        exclude_self_corr: (bool) set correlations with random draws of same subject to NaN (Default: True)
        random_state: random_state instance for permutation

    Returns:

        isc: summary statistic of bootstrapped similarity matrix

    """
    from nltools.data import Adjacency

    if not isinstance(similarity_matrix, Adjacency):
        raise ValueError("similarity_matrix must be an Adjacency instance.")

    if not isinstance(group, np.ndarray):
        raise ValueError("group must be a numpy array.")

    if len(group) != similarity_matrix.n_nodes:
        raise ValueError(
            "Group array must be the same length as the similarity matrix."
        )

    if len(np.unique(group)) != 2:
        raise ValueError("There must only be 2 unique group ids in the group array.")

    random_state = check_random_state(random_state)

    group1_id, group2_id = np.unique(group)
    permute_group = permute_group = random_state.permutation(group)
    permute_order = np.concatenate(
        [
            np.where(permute_group == group1_id)[0],
            np.where(permute_group == group2_id)[0],
        ]
    )

    permuted_matrix = similarity_matrix.squareform()[permute_order, :][:, permute_order]
    group1_similarity_permuted = Adjacency(
        permuted_matrix[group == group1_id, :][:, group == group1_id],
        matrix_type="similarity",
    )
    group2_similarity_permuted = Adjacency(
        permuted_matrix[group == group2_id, :][:, group == group2_id],
        matrix_type="similarity",
    )

    return _compute_isc_group(
        group1_similarity_permuted, group2_similarity_permuted, metric=metric
    )


def isc_group(
    group1,
    group2,
    n_samples=5000,
    metric="median",
    method="permute",
    ci_percentile=95,
    exclude_self_corr=True,
    return_null=False,
    tail=2,
    n_jobs=-1,
    random_state=None,
):
    """Compute difference in intersubject correlation between groups.

    This function computes pairwise intersubject correlations (ISC) using the median as recommended by Chen
    et al., 2016). However, if the mean is preferred, we compute the mean correlation after performing
    the fisher r-to-z transformation and then convert back to correlations to minimize artificially
    inflating the correlation values.

    There are currently two different methods to compute p-values. By default, we use the subject-wise permutation
    method recommended Chen et al., 2016. This method combines the two groups and computes pairwise similarity both
    within and between the groups. Then the group labels are permuted and the mean difference between the two groups
    are recomputed to generate a null distribution. The second method uses subject-wise bootstrapping, where a new
    pairwise similarity matrix with randomly selected subjects with replacement is created separately for each group
    and the ISC difference between these groups is used to generate a null distribution. If the same subject is
    selected multiple times, we set the perfect correlation to a nan with (exclude_self_corr=True). We compute the
    p-values using the percentile method (Hall & Wilson, 1991).

    Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C., Israel, R. B.,
    & Cox, R. W. (2016). Untangling the relatedness among correlations, part I:
    nonparametric approaches to inter-subject correlation analysis at the group level.
    NeuroImage, 142, 248-259.

    Hall, P., & Wilson, S. R. (1991). Two guidelines for bootstrap hypothesis testing.
    Biometrics, 757-762.

    This function is a wrapper around `nltools.algorithms.inference.isc.isc_group_permutation_test`
    for backward compatibility. The underlying implementation provides optimized CPU parallelization
    and optional GPU acceleration. For new code, consider using `isc_group_permutation_test` directly.

    Args:
        group1: (pd.DataFrame, np.array) observations by subjects where isc is computed across subjects
        group2: (pd.DataFrame, np.array) observations by subjects where isc is computed across subjects
        n_samples: (int) number of samples for permutation or bootstrapping
        metric: (str) type of isc summary metric ['mean','median']
        method: (str) method to compute p-values ['permute', 'bootstrap'] (default: permute)
        ci_percentile: (float) confidence interval percentile (default: 95)
        exclude_self_corr: (bool) exclude self-correlations in bootstrap (default: True)
        return_null: (bool) Return the permutation distribution along with the p-value; default False
        tail: (int) either 1 for one-tail or 2 for two-tailed test (default: 2)
        n_jobs: (int) The number of CPUs to use to do the computation. -1 means all CPUs.
        random_state: (int or RandomState) Random seed for reproducibility

    Returns:
        stats: (dict) dictionary of permutation results with keys:
            - 'isc_group_difference': Observed ISC difference (float or array)
            - 'p': P-value (float or array)
            - 'ci': Confidence interval tuple (lower, upper)
            - 'null_distribution': Null distribution (if return_null=True)

    """
    from nltools.algorithms.inference.isc import isc_group_permutation_test

    group1 = _as_ndarray(group1, name="group1")
    group2 = _as_ndarray(group2, name="group2")

    if metric not in ["mean", "median"]:
        raise ValueError("metric must be ['mean', 'median']")

    if group1.shape[0] != group2.shape[0]:
        raise ValueError("group1 has a different number of observations from group2.")

    if method not in ["permute", "bootstrap"]:
        raise NotImplementedError("method can only be ['permute', 'bootstrap']")

    # Call inference module function
    result = isc_group_permutation_test(
        group1,
        group2,
        n_permute=n_samples,  # Map parameter name
        metric=metric,
        method=method,
        ci_percentile=ci_percentile,
        tail=tail,
        n_jobs=n_jobs,
        random_state=random_state,
        return_null=return_null,
        exclude_self_corr=exclude_self_corr,
        progress_bar=False,  # Disable progress bar for backward compatibility
        summary_statistic="pairwise",  # Match old behavior (always pairwise)
    )

    # Map return keys to match original function signature
    # Inference module returns 'null_dist', but original function returned 'null_distribution'
    if return_null and "null_dist" in result:
        result["null_distribution"] = result.pop("null_dist")

    # Return dict with expected keys: ['isc_group_difference', 'p', 'ci', 'null_distribution']
    return result


def isfc(data, method="average", n_jobs=-1):
    """Compute intersubject functional connectivity (ISFC) from a list of observation x feature matrices

    This function uses the leave one out approach to compute ISFC (Simony et al., 2016).
    For each subject, compute the cross-correlation between each voxel/roi
    with the average of the rest of the subjects data. In other words,
    compute the mean voxel/ROI response for all participants except the
    target subject. Then compute the correlation between each ROI within
    the target subject with the mean ROI response in the group average.

    Simony, E., Honey, C. J., Chen, J., Lositsky, O., Yeshurun, Y., Wiesel, A., & Hasson, U. (2016).
    Dynamic reconfiguration of the default mode network during narrative comprehension.
    Nature communications, 7, 12141.

    This function now uses the optimized implementation from the inference module,
    which provides efficient cross-correlation computation between matrix columns.
    CPU parallelization is available via joblib when n_jobs > 1 or n_jobs=-1.
    Each subject's ISFC computation is independent and can be parallelized efficiently.

    Args:
        data: list of subject matrices (observations x voxels/rois)
        method: approach to computing ISFC. 'average' uses leave one out
        n_jobs: (int) Number of parallel jobs to use. -1 means all available cores.
                Default is -1 (parallel execution by default, consistent with other stats functions).

    Returns:
        list of subject ISFC matrices

    """
    if method != "average":
        raise NotImplementedError(
            "Only average method is implemented. Pairwise will be added at some point."
        )

    # Convert to numpy arrays if needed (for efficiency)
    data_arrays = [np.asarray(subject_data) for subject_data in data]
    n_subjects = len(data_arrays)
    subjects = np.arange(n_subjects)

    # Validate all subjects have same shape
    reference_shape = data_arrays[0].shape
    for i, subject_data in enumerate(data_arrays):
        if subject_data.shape != reference_shape:
            raise ValueError(
                f"All subject matrices must have the same shape. "
                f"Subject 0 has shape {reference_shape}, subject {i} has shape {subject_data.shape}"
            )

    if n_jobs == 1:
        # Serial execution (for explicit serial control)
        sub_isfc = []
        for target in subjects:
            m1 = data_arrays[target]
            sub_mean = np.zeros(m1.shape)
            for y in (y for y in subjects if y != target):
                sub_mean += data_arrays[y]
            # Use inference module function for cross-correlation computation
            sub_isfc.append(_compute_cross_correlation(m1, sub_mean / (n_subjects - 1)))
    else:
        # Parallel execution using joblib (default: n_jobs=-1 uses all cores)
        from joblib import Parallel, delayed

        def _compute_one_subject_isfc(target_idx):
            """Compute ISFC for one subject (worker function)."""
            m1 = data_arrays[target_idx]
            sub_mean = np.zeros(m1.shape, dtype=m1.dtype)
            for y in (y for y in subjects if y != target_idx):
                sub_mean += data_arrays[y]
            return _compute_cross_correlation(m1, sub_mean / (n_subjects - 1))

        # Parallelize across subjects
        sub_isfc = Parallel(n_jobs=n_jobs)(
            delayed(_compute_one_subject_isfc)(target) for target in subjects
        )

    return sub_isfc


def isps(data, sampling_freq=0.5, low_cut=0.04, high_cut=0.07, order=5, pairwise=False):
    """Compute Dynamic Intersubject Phase Synchrony (ISPS from a observation by subject array)

    This function computes the instantaneous intersubject phase synchrony for a single voxel/roi
    timeseries. Requires multiple subjects. This method is largely based on that described by Glerean
    et al., 2012 and performs a hilbert transform on narrow bandpass filtered timeseries (butterworth)
    data to get the instantaneous phase angle. The function returns a dictionary containing the
    average phase angle, the average vector length, and parametric p-values computed using the rayleigh test using circular
    statistics (Fisher, 1993). If pairwise=True, then it will compute these on the pairwise phase angle differences,
    if pairwise=False, it will compute these on the actual phase angles. This is called inter-site phase coupling
    or inter-trial phase coupling respectively in the EEG literatures.

    This function requires narrow band filtering your data. As a default we use the recommendations
    by (Glerean et al., 2012) of .04-.07Hz. This is similar to the "slow-4" band (0.025–0.067 Hz)
    described by (Zuo et al., 2010; Penttonen & Buzsáki, 2003), but excludes the .03 band, which has been
    demonstrated to contain aliased respiration signals (Birn, 2006).

    Birn RM, Smith MA, Bandettini PA, Diamond JB. 2006. Separating respiratory-variation-related
    fluctuations from neuronal-activity- related fluctuations in fMRI. Neuroimage 31:1536–1548.

    Buzsáki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical networks. Science,
    304(5679), 1926-1929.

    Fisher, N. I. (1995). Statistical analysis of circular data. cambridge university press.

    Glerean, E., Salmi, J., Lahnakoski, J. M., Jääskeläinen, I. P., & Sams, M. (2012).
    Functional magnetic resonance imaging phase synchronization as a measure of dynamic
    functional connectivity. Brain connectivity, 2(2), 91-101.

    Args:
        data: (pd.DataFrame, np.ndarray) observations x subjects data
        sampling_freq: (float) sampling freqency of data in Hz
        low_cut: (float) lower bound cutoff for high pass filter
        high_cut: (float) upper bound cutoff for low pass filter
        order: (int) filter order for butterworth bandpass
        pairwise: (bool) compute phase angle coherence on pairwise phase angle differences
                or on raw phase angle.

    Returns:
        dictionary with mean phase angle, vector length, and rayleigh statistic

    """
    data_array = _as_ndarray(data)
    phase = np.angle(
        hilbert(
            _butter_bandpass_filter(
                data_array, low_cut, high_cut, sampling_freq, order=order
            ),
            axis=0,
        )
    )

    if pairwise:
        phase = np.array(
            [
                phase[:, i] - phase[:, j]
                for i in range(phase.shape[1])
                for j in range(phase.shape[1])
                if i < j
            ]
        ).T

    out = {"average_angle": _phase_mean_angle(phase)}
    out["vector_length"] = _phase_vector_length(phase)
    out["p"] = _phase_rayleigh_p(phase)
    return out
