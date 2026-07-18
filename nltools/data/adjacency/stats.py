"""Provide standalone statistical functions for Adjacency matrices.

Each function takes an Adjacency instance as its first argument (`adj`).
"""

import numpy as np
import warnings


def similarity(
    adj,
    data,
    plot=False,
    method="2d",
    n_permute=5000,
    metric="spearman",
    include_diag=False,
    nan_policy="omit",
    tail=2,
    return_null=False,
    n_jobs=-1,
    random_state=None,
    *,
    project: bool = False,
):
    """Calculate similarity between two Adjacency matrices.

    The default uses Spearman correlation and a permutation test.

    Args:
        adj (Adjacency): Adjacency instance.
        data (Adjacency or array): Adjacency data, or 1-d array same size as adj.data.
        plot (bool): If True, plot stacked adjacency matrices. Default False.
        method (str): permutation scheme '1d', '2d', or None.
        n_permute (int): Number of permutations. Default 5000.
        metric (str): 'spearman', 'pearson', or 'kendall'.
        include_diag (bool): Only applies to 'directed' Adjacency types using
            method=None or method='1d'. Default False
            (self-similarity is uninformative). Symmetric matrices never store
            the diagonal, so this flag is a no-op for them.
        nan_policy (str): How to handle NaN values. Options:
            - 'omit': Remove NaN values pairwise before computing correlation (default)
            - 'propagate': Allow NaN to propagate through calculations
            - 'raise': Raise an error if NaN values are present
        tail (int): Tail of the test (1 or 2). Default 2.
        return_null (bool): If True, also return the null distribution. Default False.
        n_jobs (int): Number of parallel jobs. -1 means all cores. Default -1.
        random_state (int, optional): Random seed for reproducibility.
        project (bool): If True and adj has a spatial_scale, project the per-matrix
            correlations back into brain space. Default False.

    Returns:
        dict | list | BrainData: A correlation result dict with keys 'correlation',
            'p', and 'parallel' (or a list of such dicts when adj contains multiple
            matrices); a `BrainData` when `project=True`, holding the per-matrix
            correlations projected back into brain space via the spatial_scale.

    """
    from nltools.data.adjacency import Adjacency
    from nltools.stats import (
        correlation_permutation_test,
        matrix_permutation_test,
    )
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
                    "NaN values detected in 2D matrix data. For method='2d', "
                    "NaN handling is limited. Consider using method='1d' or None, "
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

    if method is None:
        n_permute = 0
        similarity_func = correlation_permutation_test
    elif method == "1d":
        similarity_func = correlation_permutation_test
    elif method == "2d":
        similarity_func = matrix_permutation_test
    else:
        raise ValueError("method must be ['1d','2d', or None']")

    def _convert_data_similarity(
        data, permutation_method=None, include_diag=include_diag
    ):
        """Convert data to the representation required for similarity."""
        if (permutation_method is None) or (permutation_method == "1d"):
            if not include_diag and (not data.issymmetric):
                d = data.squareform()
                data = d[~np.eye(d.shape[0]).astype(bool)]
            else:
                data = data.data
        elif permutation_method == "2d":
            if not data.issymmetric:
                raise TypeError(
                    f"data must be symmetric to do {permutation_method} permutation"
                )
            data = data.squareform()
        else:
            raise ValueError("permutation_method must be ['1d','2d', or None']")
        return data

    if project and adj.spatial_scale is None:
        raise ValueError(
            "similarity(project=True) requires the calling Adjacency to have "
            "a spatial_scale set (i.e. produced by a spatial-scale-aware "
            "operation like BrainData.distance(spatial_scale='roi'))."
        )

    if adj.is_single_matrix:
        if plot:
            plot_stacked_adjacency(adj, data)
        arr1 = _convert_data_similarity(data1, permutation_method=method)
        arr2 = _convert_data_similarity(data2, permutation_method=method)
        arr1, arr2 = _handle_nans(arr1, arr2, nan_policy)
        return similarity_func(
            arr1,
            arr2,
            metric=metric,
            n_permute=n_permute,
            tail=tail,
            return_null=return_null,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if plot:
        import matplotlib.pyplot as plt

        _, a = plt.subplots(len(adj))
        for i in a:
            plot_stacked_adjacency(adj, data, ax=i)
    results = []
    arr2_base = _convert_data_similarity(data2, permutation_method=method)
    for x in adj:
        arr1 = _convert_data_similarity(x, permutation_method=method)
        arr1_clean, arr2_clean = _handle_nans(arr1, arr2_base, nan_policy)
        results.append(
            similarity_func(
                arr1_clean,
                arr2_clean,
                metric=metric,
                n_permute=n_permute,
                tail=tail,
                return_null=return_null,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        )
    if project:
        per_matrix = np.array([r["correlation"] for r in results])
        return adj.to_brain(per_matrix)
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


def threshold(adj, *, upper=None, lower=None, binarize=False):
    """Threshold an Adjacency instance.

    Provide upper and lower values or percentages to perform two-sided
    thresholding. Binarize will return a mask image respecting thresholds if
    provided, otherwise respecting every non-zero value.

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

    if upper is not None and lower is not None:
        b.data[(b.data < upper) & (b.data > lower)] = 0
    elif upper is not None:
        b.data[b.data < upper] = 0
    elif lower is not None:
        b.data[b.data > lower] = 0
    if binarize:
        b.data[b.data != 0] = 1
    return b


def ttest(
    adj,
    *,
    permutation=False,
    n_permute=5000,
    tail=2,
    return_null=False,
    n_jobs=-1,
    random_state=None,
):
    """Calculate ttest across samples.

    Args:
        adj (Adjacency): Adjacency instance (must contain multiple matrices)
        permutation: (bool) Run ttest as permutation. Note this can be very slow.
        n_permute: Number of permutations (used only when
            ``permutation=True``). Default 5000.
        tail: Tail of the test (1 or 2). Default 2.
        return_null: If True, also return the null distribution. Default False.
        n_jobs: Number of parallel jobs. Default -1 (all cores).
        random_state: Random seed for reproducibility.

    Returns:
        out: (dict) contains Adjacency instances of t values (or mean if
             running permutation) and Adjacency instance of p values.

    """
    from copy import deepcopy

    from nltools.data.adjacency import Adjacency
    from nltools.stats import one_sample_permutation_test

    if adj.is_single_matrix:
        raise ValueError("t-test cannot be run on single matrices.")

    if permutation:
        t = []
        p = []
        for i in range(adj.data.shape[1]):
            stats = one_sample_permutation_test(
                adj.data[:, i],
                n_permute=n_permute,
                tail=tail,
                return_null=return_null,
                n_jobs=n_jobs,
                random_state=random_state,
            )
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


def _label_distance_long(adj, labels):
    """Build long-format within/between distance arrays for a labelled adjacency.

    Returns:
        dict with keys ``Distance`` (1-D float array), ``Type`` (1-D object
        array of "Within"/"Between"), ``Group`` (1-D array of label values).
    """
    distance = np.asarray(adj.squareform())
    labels = np.asarray(labels)
    if len(labels) != distance.shape[0]:
        raise ValueError("Labels must be same length as distance matrix")

    dist_parts, type_parts, group_parts = [], [], []
    for i in np.unique(labels):
        mask_i = labels == i
        sub = distance[np.ix_(mask_i, mask_i)]
        within_vals = sub[np.triu_indices(mask_i.sum(), k=1)]
        between_vals = distance[np.ix_(mask_i, ~mask_i)].ravel()

        dist_parts.append(within_vals)
        type_parts.append(np.full(within_vals.shape, "Within", dtype=object))
        group_parts.append(np.full(within_vals.shape, i))

        dist_parts.append(between_vals)
        type_parts.append(np.full(between_vals.shape, "Between", dtype=object))
        group_parts.append(np.full(between_vals.shape, i))

    return {
        "Distance": np.concatenate(dist_parts),
        "Type": np.concatenate(type_parts),
        "Group": np.concatenate(group_parts),
    }


def plot_label_distance(adj, labels=None, ax=None):
    """Create a violin plot of within- and between-label distances.

    Args:
        adj (Adjacency): Adjacency instance (must be a single matrix)
        labels (np.array):  numpy array of labels to plot

    Returns:
        None

    """
    from copy import deepcopy

    import pandas as pd
    import seaborn as sns

    if not adj.is_single_matrix:
        raise ValueError("This function only works on single adjacency matrices.")

    if labels is None:
        labels = np.array(deepcopy(adj.labels))

    long = _label_distance_long(adj, labels)
    # Pandas boundary: seaborn requires a pandas DataFrame input.
    out = pd.DataFrame(long)
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


def stats_label_distance(adj, *, labels=None, n_permute=5000, n_jobs=-1):
    """Calculate permutation tests on within and between label distance.

    Args:
        adj (Adjacency): Adjacency instance (must be a single matrix)
        labels (np.array):  numpy array of labels to plot
        n_permute (int): number of permutations to run (default=5000)

    Returns:
        dict:  dictionary of within and between group differences
                and p-values

    """
    from copy import deepcopy

    from nltools.stats import two_sample_permutation_test

    if not adj.is_single_matrix:
        raise ValueError("This function only works on single adjacency matrices.")

    if labels is None:
        labels = deepcopy(adj.labels)

    long = _label_distance_long(adj, labels)
    distances = long["Distance"]
    types = long["Type"]
    groups = long["Group"]

    stats = {}
    for i in np.unique(groups):
        within = distances[(groups == i) & (types == "Within")]
        between = distances[(groups == i) & (types == "Between")]
        stats[str(i)] = two_sample_permutation_test(
            within, between, n_permute=n_permute, n_jobs=n_jobs
        )
    return stats


def plot_silhouette(
    adj,
    *,
    labels=None,
    ax=None,
    permutation_test=True,
    n_permute=5000,
    colors=None,
    figsize=(6, 4),
):
    """Create a silhouette plot.

    Args:
        adj (Adjacency): Adjacency instance (must be a single matrix).
        labels (np.array): Numpy array of cluster/group labels.
        ax: Matplotlib axis handle.
        permutation_test (bool): Whether to run a permutation test. Default True.
        n_permute (int): Number of permutations for the test. Default 5000.
        colors: Optional list of RGB triplets, one per cluster (default: seaborn 'hls' palette).
        figsize: Figure size tuple. Default (6, 4).

    Returns:
        dict: Silhouette plot results including scores and optional permutation p-value.
    """
    from copy import deepcopy

    from nltools.plotting import plot_silhouette as _plot_silhouette

    distance = adj.squareform()

    if labels is None:
        labels = np.array(deepcopy(adj.labels))
    else:
        if len(labels) != distance.shape[0]:
            raise ValueError("Labels must be same length as distance matrix")

    return _plot_silhouette(
        distance,
        np.asarray(labels),
        ax=ax,
        permutation_test=permutation_test,
        n_permute=n_permute,
        colors=colors,
        figsize=figsize,
    )


def cluster_summary(adj, *, clusters=None, method="mean", summary="within"):
    """This function provides summaries of clusters within Adjacency matrices.

    It can compute mean/median of within and between cluster values. Requires a
    list of cluster ids indicating the row/column of each cluster.

    Args:
        adj (Adjacency): Adjacency instance
        clusters: (list) list of cluster labels
        method: (str) how to summarize, 'mean' or 'median'. If `None` then return all r values
        summary: (str) summarize within cluster or between clusters

    Returns:
        dict: (dict) within cluster means

    """
    if method not in ["mean", "median", None]:
        raise ValueError("method must be ['mean','median', None]")

    distance = np.asarray(adj.squareform())
    clusters = np.asarray(clusters)

    if len(clusters) != distance.shape[0]:
        raise ValueError("Cluster labels must be same length as distance matrix")

    out = {}
    for i in list(set(clusters.tolist())):
        mask_i = clusters == i
        if summary == "within":
            within_vals = distance[np.ix_(mask_i, mask_i)][
                np.triu_indices(mask_i.sum(), k=1)
            ]
            if method == "mean":
                out[i] = float(np.mean(within_vals))
            elif method == "median":
                out[i] = float(np.median(within_vals))
            else:
                out[i] = within_vals
        elif summary == "between":
            between_block = distance[np.ix_(mask_i, ~mask_i)]
            if method == "mean":
                out[i] = float(np.mean(between_block))
            elif method == "median":
                out[i] = float(np.median(between_block))
            else:
                out[i] = between_block
    return out
