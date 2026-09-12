"""Intersubject correlation, functional connectivity, and phase synchrony."""

__all__ = ["isc", "isc_group", "isfc", "isps"]

import numpy as np
import polars as pl
from scipy.signal import hilbert

from .isc import isc_permutation_test
from .matrix import _compute_cross_correlation
from .utils import maybe_tqdm

from ..signal import (
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


def isc(
    data,
    *,
    n_samples=5000,
    summary="median",
    method="bootstrap",
    ci_percentile=95,
    exclude_self_corr=True,
    tail=2,
    metric="correlation",
    return_null=False,
    n_jobs=-1,
    random_state=None,
    progress_bar=False,
):
    """Compute pairwise intersubject correlation from an observations-by-subjects array.

    Pairwise ISC is summarized with the median, as Chen et al. (2016) recommend;
    `summary='mean'` instead averages after the Fisher r-to-z transform and
    converts back, which avoids inflating the estimate.

    Three null distributions are available. The default subject-wise bootstrap
    (Chen et al., 2016) resamples subjects with replacement and recomputes the
    pairwise similarity matrix; a subject drawn twice correlates perfectly with
    itself, so those entries are set to NaN when `exclude_self_corr=True`.
    P-values use the percentile method, as in Brainiak. The classic surrogate
    methods instead circle-shift or phase-randomize each time series (Lancaster
    et al., 2018), preserving its temporal autocorrelation, and recompute ISC.

    Runs on plain arrays with observations aligned across subjects.
    `isc_permutation_test` exposes the same engine with leave-one-out ISC.

    Args:
        data (np.ndarray | pl.DataFrame | pd.DataFrame): Observations by
            subjects; ISC is computed across the columns.
        n_samples (int): Number of bootstrap draws or surrogate permutations.
            Defaults to 5000.
        summary (str): `'median'` (default) or `'mean'`.
        method (str): `'bootstrap'` (default), `'circle_shift'`, or
            `'phase_randomize'`.
        ci_percentile (int): Confidence-interval width in percent. Defaults to 95.
        exclude_self_corr (bool): Set self-correlations (the same subject
            bootstrapped twice) to NaN. Defaults to True.
        tail (int | str): `2` or `'two'` (two-tailed, default) or `1` or `'one'`
            (one-tailed, ISC > 0).
        metric (str): Pairwise similarity metric; any metric accepted by
            sklearn's `pairwise_distances`. Defaults to `'correlation'`.
        return_null (bool): Include the null distribution in the result.
            Defaults to False.
        n_jobs (int): CPU workers for the resamples; -1 (default) picks the
            count from available memory.
        random_state (int | np.random.RandomState | None): Seed or generator for
            the resampling.
        progress_bar (bool): Display a progress bar. Defaults to False.

    Returns:
        dict: Keys `'isc'` (float, observed ISC), `'p'` (float), `'ci'` (tuple
            `(lower, upper)`), and — when `return_null=True` — `'null_dist'`
            (np.ndarray).

    References:
        Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
        Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
        correlations, part I: nonparametric approaches to inter-subject
        correlation analysis at the group level. NeuroImage, 142, 248-259.

        Hall, P., & Wilson, S. R. (1991). Two guidelines for bootstrap
        hypothesis testing. Biometrics, 757-762.

        Lancaster, G., Iatsenko, D., Pidde, A., Ticcinelli, V., & Stefanovska,
        A. (2018). Surrogate data for hypothesis testing of physical systems.
        Physics Reports, 748, 1-60.
    """
    data = _as_ndarray(data)

    if summary not in ["mean", "median"]:
        raise ValueError("summary must be ['mean', 'median']")

    # The engine speaks the same canonical vocabulary (summary=, metric=), so
    # this wrapper only maps n_samples -> n_permute and pins the classic
    # pairwise behavior.
    return isc_permutation_test(
        data,
        n_permute=n_samples,  # Map n_samples -> n_permute
        summary=summary,
        summary_statistic="pairwise",  # Explicitly set to match original behavior
        method=method,
        ci_percentile=ci_percentile,
        tail=tail,
        n_jobs=n_jobs,
        random_state=random_state,
        return_null=return_null,
        exclude_self_corr=exclude_self_corr,
        metric=metric,
        progress_bar=progress_bar,
    )


def isc_group(
    group1,
    group2,
    *,
    n_samples=5000,
    summary="median",
    method="permute",
    ci_percentile=95,
    exclude_self_corr=True,
    return_null=False,
    tail=2,
    metric="correlation",
    n_jobs=-1,
    random_state=None,
    progress_bar=False,
):
    """Test the difference in pairwise intersubject correlation between two groups.

    ISC within each group is summarized with the median, as Chen et al. (2016)
    recommend (`summary='mean'` averages after the Fisher r-to-z transform), and
    the observed statistic is `group1 - group2`.

    Two null distributions are available. The default subject-wise permutation
    (Chen et al., 2016) pools the subjects, computes pairwise similarity within
    and between groups, then reshuffles the group labels and recomputes the
    difference. The subject-wise bootstrap instead resamples subjects with
    replacement within each group; a subject drawn twice correlates perfectly
    with itself, so those entries are set to NaN when `exclude_self_corr=True`.
    P-values use the percentile method (Hall & Wilson, 1991).

    Runs on plain arrays; `isc_group_permutation_test` exposes the same engine
    with leave-one-out ISC.

    Args:
        group1 (np.ndarray | pl.DataFrame | pd.DataFrame): Observations by
            subjects for the first group.
        group2 (np.ndarray | pl.DataFrame | pd.DataFrame): Observations by
            subjects for the second group (same number of observations).
        n_samples (int): Number of permutations or bootstrap draws. Defaults to
            5000.
        summary (str): `'median'` (default) or `'mean'`.
        method (str): `'permute'` (default) or `'bootstrap'`.
        ci_percentile (float): Confidence-interval width in percent. Defaults to
            95.
        exclude_self_corr (bool): In the bootstrap, set self-correlations to NaN.
            Defaults to True.
        return_null (bool): Include the null distribution in the result.
            Defaults to False.
        tail (int | str): `2` or `'two'` (two-tailed, default) or `1` or `'one'`
            (one-tailed, group1 > group2).
        metric (str): Pairwise similarity metric; any metric accepted by
            sklearn's `pairwise_distances`. Defaults to `'correlation'`.
        n_jobs (int): CPU workers for the resamples; -1 (default) picks the
            count from available memory.
        random_state (int | np.random.RandomState | None): Random seed for
            reproducibility.
        progress_bar (bool): Display a progress bar. Defaults to False.

    Returns:
        dict: Keys `'isc_group_difference'` (float, observed difference), `'p'`
            (float), `'ci'` (tuple `(lower, upper)`), and — when
            `return_null=True` — `'null_dist'` (np.ndarray).

    References:
        Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
        Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
        correlations, part I: nonparametric approaches to inter-subject
        correlation analysis at the group level. NeuroImage, 142, 248-259.

        Hall, P., & Wilson, S. R. (1991). Two guidelines for bootstrap
        hypothesis testing. Biometrics, 757-762.
    """
    from .isc import isc_group_permutation_test

    group1 = _as_ndarray(group1, name="group1")
    group2 = _as_ndarray(group2, name="group2")

    if summary not in ["mean", "median"]:
        raise ValueError("summary must be ['mean', 'median']")

    if group1.shape[0] != group2.shape[0]:
        raise ValueError("group1 has a different number of observations from group2.")

    if method not in ["permute", "bootstrap"]:
        raise NotImplementedError("method can only be ['permute', 'bootstrap']")

    # The engine speaks the same canonical vocabulary; only n_samples ->
    # n_permute is mapped here.
    return isc_group_permutation_test(
        group1,
        group2,
        n_permute=n_samples,  # Map parameter name
        summary=summary,
        method=method,
        ci_percentile=ci_percentile,
        tail=tail,
        metric=metric,
        n_jobs=n_jobs,
        random_state=random_state,
        return_null=return_null,
        exclude_self_corr=exclude_self_corr,
        progress_bar=progress_bar,
        summary_statistic="pairwise",  # Match old behavior (always pairwise)
    )


def isfc(data, *, method="average", n_jobs=-1, random_state=None, progress_bar=False):
    """Compute intersubject functional connectivity (ISFC) from per-subject matrices.

    Uses the leave-one-out approach of Simony et al. (2016): for each subject,
    average the other subjects' data and correlate every voxel/ROI time series
    of the target subject with every voxel/ROI time series of that average.
    Subjects are independent, so they are processed in parallel with joblib
    unless `n_jobs=1`.

    Args:
        data (list[np.ndarray]): One matrix per subject, each
            `(n_observations, n_features)` with identical shapes.
        method (str): Only `'average'` (leave-one-out) is implemented.
        n_jobs (int): Parallel workers; -1 (default) uses all cores, 1 runs
            serially.
        random_state (int | np.random.RandomState | None): Unused. ISFC's
            leave-one-out computation is deterministic and draws no random
            samples; the parameter exists for signature parity with the rest
            of the ISC family (`isc`, `isc_group`).
        progress_bar (bool): Display a progress bar over subjects. Defaults to
            False.

    Returns:
        list[np.ndarray]: One `(n_features, n_features)` ISFC matrix per
            subject.

    References:
        Simony, E., Honey, C. J., Chen, J., Lositsky, O., Yeshurun, Y., Wiesel,
        A., & Hasson, U. (2016). Dynamic reconfiguration of the default mode
        network during narrative comprehension. Nature Communications, 7, 12141.
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

    progress_kwargs = {
        "progress_bar": progress_bar,
        "desc": "ISFC subjects",
        "unit": "subject",
    }

    if n_jobs == 1:
        # Serial execution (for explicit serial control)
        sub_isfc = []
        for target in maybe_tqdm(subjects, **progress_kwargs):
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
            delayed(_compute_one_subject_isfc)(target)
            for target in maybe_tqdm(subjects, **progress_kwargs)
        )

    return sub_isfc


def isps(
    data, *, sampling_freq=0.5, low_cut=0.04, high_cut=0.07, order=5, pairwise=False
):
    """Compute dynamic intersubject phase synchrony (ISPS) from an observations-by-subjects array.

    Instantaneous phase synchrony across subjects for a single voxel/ROI time
    series, after Glerean et al. (2012): the data are narrow-band filtered
    (Butterworth) and Hilbert-transformed to get each subject's instantaneous
    phase angle at every time point. Across subjects, the result gives the
    mean phase angle, the mean resultant vector length, and a parametric
    p-value from the Rayleigh test for circular uniformity (Fisher, 1995).
    With `pairwise=True` these are computed on pairwise phase-angle differences
    (inter-site phase coupling in the EEG literature) rather than on the raw
    angles (inter-trial phase coupling).

    The default band, 0.04-0.07 Hz, follows Glerean et al. (2012). It is close
    to the "slow-4" band (0.025-0.067 Hz; Zuo et al., 2010; Penttonen &
    Buzsáki, 2003) but excludes ~0.03 Hz, which carries aliased respiration
    (Birn et al., 2006).

    Args:
        data (np.ndarray | pl.DataFrame | pd.DataFrame): Observations by
            subjects.
        sampling_freq (float): Sampling frequency in Hz. Defaults to 0.5.
        low_cut (float): Lower band-pass cutoff in Hz. Defaults to 0.04.
        high_cut (float): Upper band-pass cutoff in Hz. Defaults to 0.07.
        order (int): Butterworth filter order. Defaults to 5.
        pairwise (bool): Compute on pairwise phase-angle differences instead of
            the raw phase angles. Defaults to False.

    Returns:
        dict: Keys `'average_angle'` (np.ndarray, mean phase angle per time
            point), `'vector_length'` (np.ndarray, mean resultant length per
            time point), and `'p'` (np.ndarray, Rayleigh-test p-value per time
            point).

    References:
        Birn, R. M., Smith, M. A., Bandettini, P. A., & Diamond, J. B. (2006).
        Separating respiratory-variation-related fluctuations from
        neuronal-activity-related fluctuations in fMRI. NeuroImage, 31,
        1536-1548.

        Buzsáki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical
        networks. Science, 304(5679), 1926-1929.

        Fisher, N. I. (1995). Statistical analysis of circular data. Cambridge
        University Press.

        Glerean, E., Salmi, J., Lahnakoski, J. M., Jääskeläinen, I. P., & Sams,
        M. (2012). Functional magnetic resonance imaging phase synchronization
        as a measure of dynamic functional connectivity. Brain Connectivity,
        2(2), 91-101.
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
