"""
Standalone statistical functions for Adjacency matrices.

Each function takes an Adjacency instance as its first argument (`adj`).
"""

import numpy as np
import warnings


def similarity(
    adj,
    data,
    plot=False,
    perm_type="2d",
    n_permute=5000,
    metric="spearman",
    ignore_diagonal=False,
    nan_policy="omit",
    **kwargs,
):
    """
    Calculate similarity between two Adjacency matrices. Default is to use spearman
    correlation and permutation test.

    Args:
        adj (Adjacency): Adjacency instance.
        data (Adjacency or array): Adjacency data, or 1-d array same size as adj.data.
        plot (bool): If True, plot stacked adjacency matrices. Default False.
        perm_type (str): '1d', '2d', or None.
        n_permute (int): Number of permutations. Default 5000.
        metric (str): 'spearman', 'pearson', or 'kendall'.
        ignore_diagonal (bool): Only applies to 'directed' Adjacency types using
            perm_type=None or perm_type='1d'.
        nan_policy (str): How to handle NaN values. Options:
            - 'omit': Remove NaN values pairwise before computing correlation (default)
            - 'propagate': Allow NaN to propagate through calculations
            - 'raise': Raise an error if NaN values are present

    Returns:
        dict or list: Correlation result dict with keys 'r' and 'p', or a list of
            such dicts when adj contains multiple matrices.

    """
    from nltools.data.adjacency import Adjacency
    from nltools.stats import correlation_permutation, matrix_permutation
    from nltools.plotting import plot_stacked_adjacency

    if nan_policy not in ("omit", "propagate", "raise"):
        raise ValueError(
            f"nan_policy must be 'omit', 'propagate', or 'raise', got {nan_policy!r}"
        )

    def _handle_nans(arr1, arr2, nan_policy):
        """Handle NaN values according to policy.

        For 1D arrays: masks out positions where either array has NaN.
        For 2D arrays: flattens and masks, then reshapes (for matrix perm).
        """
        arr1 = np.asarray(arr1)
        arr2 = np.asarray(arr2)

        # Check for NaN presence
        has_nan1 = np.any(np.isnan(arr1))
        has_nan2 = np.any(np.isnan(arr2))

        if not has_nan1 and not has_nan2:
            return arr1, arr2

        if nan_policy == "raise":
            if has_nan1 or has_nan2:
                raise ValueError(
                    "Input contains NaN values. Use nan_policy='omit' to ignore them "
                    "or nan_policy='propagate' to allow NaN in results."
                )
        elif nan_policy == "propagate":
            return arr1, arr2
        elif nan_policy == "omit":
            # For 2D matrix permutation, we can't easily mask individual elements
            # because the permutation test permutes rows/columns together.
            # Instead, we warn and use propagate for 2D.
            if arr1.ndim == 2:
                warnings.warn(
                    "NaN values detected in 2D matrix data. For perm_type='2d', "
                    "NaN handling is limited. Consider using perm_type='1d' or None, "
                    "or removing NaN values before calling similarity().",
                    UserWarning,
                )
                return arr1, arr2

            # For 1D: mask out NaN positions from both arrays
            mask = ~(np.isnan(arr1) | np.isnan(arr2))
            if not np.any(mask):
                raise ValueError(
                    "All values are NaN after pairwise removal. Cannot compute similarity."
                )
            return arr1[mask], arr2[mask]

        return arr1, arr2

    data1 = adj.copy()
    if not isinstance(data, Adjacency):
        data2 = Adjacency(data)
    else:
        data2 = data.copy()

    if perm_type is None:
        n_permute = 0
        similarity_func = correlation_permutation
    elif perm_type == "1d":
        similarity_func = correlation_permutation
    elif perm_type == "2d":
        similarity_func = matrix_permutation
    else:
        raise ValueError("perm_type must be ['1d','2d', or None']")

    def _convert_data_similarity(data, perm_type=None, ignore_diagonal=ignore_diagonal):
        """Helper function to convert data correctly"""
        if (perm_type is None) or (perm_type == "1d"):
            if ignore_diagonal and (not data.issymmetric):
                d = data.squareform()
                data = d[~np.eye(d.shape[0]).astype(bool)]
            else:
                data = data.data
        elif perm_type == "2d":
            if not data.issymmetric:
                raise TypeError(f"data must be symmetric to do {perm_type} permutation")
            else:
                data = data.squareform()
        else:
            raise ValueError("perm_type must be ['1d','2d', or None']")
        return data

    if adj.is_single_matrix:
        if plot:
            plot_stacked_adjacency(adj, data)
        arr1 = _convert_data_similarity(data1, perm_type=perm_type)
        arr2 = _convert_data_similarity(data2, perm_type=perm_type)
        arr1, arr2 = _handle_nans(arr1, arr2, nan_policy)
        return similarity_func(
            arr1,
            arr2,
            metric=metric,
            n_permute=n_permute,
            **kwargs,
        )
    else:
        if plot:
            import matplotlib.pyplot as plt

            _, a = plt.subplots(len(adj))
            for i in a:
                plot_stacked_adjacency(adj, data, ax=i)
        results = []
        arr2_base = _convert_data_similarity(data2, perm_type=perm_type)
        for x in adj:
            arr1 = _convert_data_similarity(x, perm_type=perm_type)
            arr1_clean, arr2_clean = _handle_nans(arr1, arr2_base, nan_policy)
            results.append(
                similarity_func(
                    arr1_clean,
                    arr2_clean,
                    metric=metric,
                    n_permute=n_permute,
                    **kwargs,
                )
            )
        return results


def r_to_z(adj):
    """Apply Fisher's r to z transformation to each element of the data object.

    Args:
        adj (Adjacency): Adjacency instance.

    Returns:
        Adjacency: New Adjacency with z-transformed values.
    """
    from nltools.stats import fisher_r_to_z

    out = adj.copy()
    out.data = fisher_r_to_z(out.data)
    return out


def z_to_r(adj):
    """Convert z score back into r value for each element of data object.

    Args:
        adj (Adjacency): Adjacency instance.

    Returns:
        Adjacency: New Adjacency with r values.
    """
    from nltools.stats import fisher_z_to_r

    out = adj.copy()
    out.data = fisher_z_to_r(out.data)
    return out


def threshold(adj, upper=None, lower=None, binarize=False):
    """Threshold Adjacency instance. Provide upper and lower values or
       percentages to perform two-sided thresholding. Binarize will return
       a mask image respecting thresholds if provided, otherwise respecting
       every non-zero value.

    Args:
        adj (Adjacency): Adjacency instance
        upper: (float or str) Upper cutoff for thresholding. If string
                will interpret as percentile; can be None for one-sided
                thresholding.
        lower: (float or str) Lower cutoff for thresholding. If string
                will interpret as percentile; can be None for one-sided
                thresholding.
        binarize (bool): return binarized image respecting thresholds if
                provided, otherwise binarize on every non-zero value;
                default False

    Returns:
        Adjacency: thresholded Adjacency instance

    """

    b = adj.copy()
    if isinstance(upper, str) and upper[-1] == "%":
        upper = np.percentile(b.data, float(upper[:-1]))
    if isinstance(lower, str) and lower[-1] == "%":
        lower = np.percentile(b.data, float(lower[:-1]))

    if upper and lower:
        b.data[(b.data < upper) & (b.data > lower)] = 0
    elif upper:
        b.data[b.data < upper] = 0
    elif lower:
        b.data[b.data > lower] = 0
    if binarize:
        b.data[b.data != 0] = 1
    return b


def ttest(adj, permutation=False, **kwargs):
    """Calculate ttest across samples.

    Args:
        adj (Adjacency): Adjacency instance (must contain multiple matrices)
        permutation: (bool) Run ttest as permutation. Note this can be very slow.

    Returns:
        out: (dict) contains Adjacency instances of t values (or mean if
             running permutation) and Adjacency instance of p values.

    """
    from copy import deepcopy

    from nltools.data.adjacency import Adjacency
    from nltools.stats import one_sample_permutation

    if adj.is_single_matrix:
        raise ValueError("t-test cannot be run on single matrices.")

    if permutation:
        t = []
        p = []
        for i in range(adj.data.shape[1]):
            stats = one_sample_permutation(adj.data[:, i], **kwargs)
            t.append(stats["mean"])
            p.append(stats["p"])
        t = Adjacency(np.array(t))
        p = Adjacency(np.array(p))
    else:
        from scipy.stats import ttest_1samp

        t = adj.mean().copy()
        p = deepcopy(t)
        t.data, p.data = ttest_1samp(adj.data, 0, 0)

    return {"t": t, "p": p}


def plot_label_distance(adj, labels=None, ax=None):
    """Create a violin plot indicating within and between label distance

    Args:
        adj (Adjacency): Adjacency instance (must be a single matrix)
        labels (np.array):  numpy array of labels to plot

    Returns:
        None

    """
    import pandas as pd
    import seaborn as sns
    from copy import deepcopy

    if not adj.is_single_matrix:
        raise ValueError("This function only works on single adjacency matrices.")

    distance = pd.DataFrame(adj.squareform())

    if labels is None:
        labels = np.array(deepcopy(adj.labels))
    else:
        if len(labels) != distance.shape[0]:
            raise ValueError("Labels must be same length as distance matrix")

    frames = []
    for i in np.unique(labels):
        # Within-group distances (upper triangle of group i x group i)
        within_vals = distance.loc[labels == i, labels == i].values[
            np.triu_indices(sum(labels == i), k=1)
        ]
        tmp_w = pd.DataFrame({"Distance": within_vals, "Type": "Within", "Group": i})
        # Between-group distances (group i to all other groups)
        between_vals = distance.loc[labels == i, labels != i].values.flatten()
        tmp_b = pd.DataFrame({"Distance": between_vals, "Type": "Between", "Group": i})
        frames.extend([tmp_w, tmp_b])
    out = pd.concat(frames, ignore_index=True)
    f = sns.violinplot(
        x="Group",
        y="Distance",
        hue="Type",
        data=out,
        split=True,
        inner="quartile",
        palette={"Within": "lightskyblue", "Between": "red"},
        ax=ax,
    )
    f.set_ylabel("Average Distance")
    f.set_title("Average Group Distance")
    return


def stats_label_distance(adj, labels=None, n_permute=5000, n_jobs=-1):
    """Calculate permutation tests on within and between label distance.

    Args:
        adj (Adjacency): Adjacency instance (must be a single matrix)
        labels (np.array):  numpy array of labels to plot
        n_permute (int): number of permutations to run (default=5000)

    Returns:
        dict:  dictionary of within and between group differences
                and p-values

    """
    import pandas as pd
    from copy import deepcopy

    from nltools.stats import two_sample_permutation

    if not adj.is_single_matrix:
        raise ValueError("This function only works on single adjacency matrices.")

    distance = pd.DataFrame(adj.squareform())

    if labels is None:
        labels = deepcopy(adj.labels)
    else:
        if len(labels) != distance.shape[0]:
            raise ValueError("Labels must be same length as distance matrix")

    frames = []
    for i in np.unique(labels):
        # Within-group distances (upper triangle of group i x group i)
        within_vals = distance.loc[labels == i, labels == i].values[
            np.triu_indices(sum(labels == i), k=1)
        ]
        tmp_w = pd.DataFrame({"Distance": within_vals, "Type": "Within", "Group": i})
        # Between-group distances (group i to all other groups)
        between_vals = distance.loc[labels == i, labels != i].values.flatten()
        tmp_b = pd.DataFrame({"Distance": between_vals, "Type": "Between", "Group": i})
        frames.extend([tmp_w, tmp_b])
    out = pd.concat(frames, ignore_index=True)
    stats = {}
    for i in np.unique(labels):
        # Within group test
        tmp1 = out.loc[(out["Group"] == i) & (out["Type"] == "Within"), "Distance"]
        tmp2 = out.loc[(out["Group"] == i) & (out["Type"] == "Between"), "Distance"]
        stats[str(i)] = two_sample_permutation(
            tmp1, tmp2, n_permute=n_permute, n_jobs=n_jobs
        )
    return stats


def plot_silhouette(
    adj, labels=None, ax=None, permutation_test=True, n_permute=5000, **kwargs
):
    """Create a silhouette plot.

    Args:
        adj (Adjacency): Adjacency instance (must be a single matrix).
        labels (np.array): Numpy array of cluster/group labels.
        ax: Matplotlib axis handle.
        permutation_test (bool): Whether to run a permutation test. Default True.
        n_permute (int): Number of permutations for the test. Default 5000.

    Returns:
        dict: Silhouette plot results including scores and optional permutation p-value.
    """
    import pandas as pd
    from copy import deepcopy

    from nltools.plotting import plot_silhouette as _plot_silhouette

    distance = pd.DataFrame(adj.squareform())

    if labels is None:
        labels = np.array(deepcopy(adj.labels))
    else:
        if len(labels) != distance.shape[0]:
            raise ValueError("Labels must be same length as distance matrix")

    return _plot_silhouette(
        distance,
        pd.Series(labels),
        ax=ax,
        permutation_test=permutation_test,
        n_permute=n_permute,
        **kwargs,
    )


def cluster_summary(adj, clusters=None, metric="mean", summary="within"):
    """This function provides summaries of clusters within Adjacency matrices.

    It can compute mean/median of within and between cluster values. Requires a
    list of cluster ids indicating the row/column of each cluster.

    Args:
        adj (Adjacency): Adjacency instance
        clusters: (list) list of cluster labels
        metric: (str) method to summarize mean or median. If 'None" then return all r values
        summary: (str) summarize within cluster or between clusters

    Returns:
        dict: (dict) within cluster means

    """
    import pandas as pd

    if metric not in ["mean", "median", None]:
        raise ValueError("metric must be ['mean','median', None]")

    distance = pd.DataFrame(adj.squareform())
    clusters = np.array(clusters)

    if len(clusters) != distance.shape[0]:
        raise ValueError("Cluster labels must be same length as distance matrix")

    out = {}
    for i in list(set(clusters)):
        if summary == "within":
            if metric == "mean":
                out[i] = np.mean(
                    distance.loc[clusters == i, clusters == i].values[
                        np.triu_indices(sum(clusters == i), k=1)
                    ]
                )
            elif metric == "median":
                out[i] = np.median(
                    distance.loc[clusters == i, clusters == i].values[
                        np.triu_indices(sum(clusters == i), k=1)
                    ]
                )
            elif metric is None:
                out[i] = distance.loc[clusters == i, clusters == i].values[
                    np.triu_indices(sum(clusters == i), k=1)
                ]
        elif summary == "between":
            if metric == "mean":
                out[i] = distance.loc[clusters == i, clusters != i].mean().mean()
            elif metric == "median":
                out[i] = distance.loc[clusters == i, clusters != i].median().median()
            elif metric is None:
                out[i] = distance.loc[clusters == i, clusters != i]
    return out
