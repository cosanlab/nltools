"""Provide standalone statistical functions for Adjacency matrices.

Each function takes an Adjacency instance as its first argument (`adj`).
"""

import numpy as np


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
    progress_bar: bool = False,
):
    """Calculate similarity between two Adjacency matrices.

    The default uses Spearman correlation and a permutation test.

    Args:
        adj (Adjacency): Adjacency instance.
        data (Adjacency | np.ndarray): Adjacency to compare against, or a 1-D array the
            same size as `adj.data`.
        plot (bool): If True, plot stacked adjacency matrices. Default False.
        method (str | None): Permutation scheme, `'1d'`, `'2d'`, or None (no
            permutation test).
        n_permute (int): Number of permutations. Default 5000.
        metric (str): `'spearman'`, `'pearson'`, or `'kendall'`.
        include_diag (bool): Only applies to `'directed'` matrices with `method=None`
            or `method='1d'`. Default False (self-similarity is uninformative).
            Symmetric matrices never store the diagonal, so this flag is a no-op
            for them.
        nan_policy (str): How to handle NaN values on the 1-D paths
            (`method='1d'` or `method=None`): `'omit'` removes NaN pairwise before
            computing the correlation (default), `'propagate'` lets NaN flow
            through, `'raise'` errors if any NaN is present. `method='2d'` raises
            on any NaN whatever the policy.
        tail (int | str): `2`/`'two'` (two-tailed, default) or `1`/`'one'` (one-tailed, positive direction).
        return_null (bool): If True, also return the null distribution. Default False.
        n_jobs (int): Number of parallel jobs. -1 means all cores. Default -1.
        random_state (int, optional): Random seed for reproducibility.
        progress_bar (bool): If True, show a progress bar. Default False.

    Returns:
        dict | list[dict]: A correlation result dict with keys 'correlation'
            and 'p', or a list of these dicts for a stack.
    """
    from nltools.data.adjacency import Adjacency
    from nltools.algorithms.inference import (
        correlation_permutation_test,
        matrix_permutation_test,
    )
    from nltools.plotting import plot_stacked_adjacency

    if nan_policy not in ("omit", "propagate", "raise"):
        raise ValueError(
            f"nan_policy must be 'omit', 'propagate', or 'raise', got {nan_policy!r}"
        )

    def _handle_nans(arr1, arr2, nan_policy):
        """Apply `nan_policy` to the 1-D inputs; reject NaN outright for 2-D."""
        arr1 = np.asarray(arr1)
        arr2 = np.asarray(arr2)

        if not np.any(np.isnan(arr1)) and not np.any(np.isnan(arr2)):
            return arr1, arr2

        if arr1.ndim == 2:
            # The matrix permutation shuffles whole rows and columns together,
            # so no policy makes a 2-D correlation over NaN edges meaningful.
            raise ValueError(
                "Input contains NaN values, which method='2d' cannot handle. "
                "Use method='1d' (or method=None), which masks NaN pairwise, "
                "or remove the NaN values before calling similarity()."
            )

        if nan_policy == "raise":
            raise ValueError(
                "Input contains NaN values. Use nan_policy='omit' to ignore them "
                "or nan_policy='propagate' to allow NaN in results."
            )
        if nan_policy == "propagate":
            return arr1, arr2

        mask = ~(np.isnan(arr1) | np.isnan(arr2))
        if not np.any(mask):
            raise ValueError(
                "All values are NaN after pairwise removal. Cannot compute similarity."
            )
        return arr1[mask], arr2[mask]

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
            progress_bar=progress_bar,
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
                progress_bar=progress_bar,
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
    from nltools.algorithms.similarity import fisher_r_to_z

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
    from nltools.algorithms.similarity import fisher_z_to_r

    out = adj.copy()
    out.data = fisher_z_to_r(out.data)
    return out


def threshold(adj, *, upper=None, lower=None, binarize=False):
    """Threshold an Adjacency instance.

    Provide upper and lower values or percentages to perform two-sided
    thresholding. Binarize will return a mask image respecting thresholds if
    provided, otherwise respecting every non-zero value.

    Args:
        adj (Adjacency): Adjacency instance.
        upper (float | str, optional): Upper cutoff. A string such as `'95%'` is
            interpreted as a percentile; None for one-sided thresholding.
        lower (float | str, optional): Lower cutoff. A string such as `'5%'` is
            interpreted as a percentile; None for one-sided thresholding.
        binarize (bool): Return a binarized matrix respecting the thresholds if
            provided, otherwise binarize on every non-zero value. Default False.

    Returns:
        Adjacency: Thresholded Adjacency instance.
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
    popmean=0.0,
    permutation=False,
    n_permute=5000,
    tail=2,
    return_null=False,
    n_jobs=-1,
    random_state=None,
    progress_bar=False,
):
    """Run a one-sample t-test across stacked matrices.

    Tests every stored edge against `popmean` across the matrices in the stack.
    Delegates the statistics to the shared one-sample contract in
    `nltools.algorithms.inference.one_sample`.

    Args:
        adj (Adjacency): Stack of two or more matrices with the same node order
            and storage kind.
        popmean (float): Population mean to test against. Default 0.0.
        permutation (bool): If True, take p from a sign-flip permutation test on
            `matrices - popmean`. The reported `t` stays the observed parametric
            statistic. Default False.
        n_permute (int): Number of permutations, used only when
            `permutation=True`. Default 5000.
        tail (int | str): `2`/`'two'` (two-tailed, default) or `1`/`'one'`
            (one-tailed: mean > `popmean`).
        return_null (bool): If True, also return the permutation null. Has no
            effect on the parametric path, which computes no null. Default False.
        n_jobs (int): Number of parallel jobs. Default -1 (all cores).
        random_state (int, optional): Random seed for reproducibility.
        progress_bar (bool): If True, show a progress bar. Default False.

    Returns:
        dict: `'mean'`, `'t'`, `'z'` and `'p'` as independent single-matrix
            `Adjacency` results that retain the node count, storage kind
            (including directed) and shared node labels, with matrix metadata
            cleared. `'mean'` is the edgewise mean minus `popmean`; `'t'` is the
            observed one-sample t-statistic on both paths; `'p'` is parametric,
            or the empirical sign-flip p-value when `permutation=True`; `'z'` is
            the tail-aware normal score of `p`. With `permutation=True` and
            `return_null=True` the dict also holds `'null_dist'`, an owned
            `(n_permute, n_edges)` array of centered means in flat storage
            order and in the units of `'mean'`. Maps are unthresholded. Apply a
            cutoff or a multiple-comparison correction afterwards.

    Raises:
        ValueError: If `adj` holds fewer than two matrices.
    """
    import polars as pl

    from nltools.algorithms.inference.one_sample import _one_sample_statistics

    from .state import common_labels, result

    if adj.is_single_matrix or adj.data.shape[0] < 2:
        raise ValueError(
            "t-test requires multiple matrices (got fewer than 2). "
            "Stack matrices into a single Adjacency first."
        )

    stats = _one_sample_statistics(
        adj.data,
        popmean=popmean,
        permutation=permutation,
        n_permute=n_permute,
        tail=tail,
        return_null=return_null,
        n_jobs=n_jobs,
        random_state=random_state,
        progress_bar=progress_bar,
    )
    labels = common_labels(adj)
    results = {
        key: result(adj, stats[key], labels=labels, Y=pl.DataFrame())
        for key in ("mean", "t", "z", "p")
    }
    if "null_dist" in stats:
        results["null_dist"] = stats["null_dist"]
    return results


def _label_distance_long(adj, labels):
    """Build long-format within/between distance arrays for a labelled adjacency.

    Returns:
        dict: Keys ``Distance`` (1-D float array), ``Type`` (1-D object array of
            "Within"/"Between"), ``Group`` (1-D array of label values).
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


def _label_distance_inputs(adj, labels):
    """Return the square distance matrix and node labels the label plots take.

    Both label-distance plots are drawn by `nltools.plotting.adjacency` from a
    square matrix and a label vector, so the squareform, the stored-label
    fallback, and the single-matrix rule live here once.
    """
    from copy import deepcopy

    if not adj.is_single_matrix:
        raise ValueError("This function only works on single adjacency matrices.")

    distance = adj.squareform()
    if labels is None:
        labels = np.array(deepcopy(adj.labels))
    labels = np.asarray(labels)
    if len(labels) != distance.shape[0]:
        raise ValueError("Labels must be same length as distance matrix")
    return distance, labels


def plot_label_distance(  # nosemgrep: kwargs-internal-forwarding  # forwards to seaborn via plot_mean_label_distance
    adj, labels=None, ax=None, *, permutation_test=False, n_permute=5000, **kwargs
):
    """Create a violin plot of within- and between-label distances.

    Args:
        adj (Adjacency): Adjacency instance (must be a single matrix).
        labels (np.ndarray, optional): Group label per node; defaults to `adj.labels`.
        ax (matplotlib.axes.Axes, optional): Axis to draw on.
        permutation_test (bool): Run a two-sample permutation test of within
            against between distance for each group. Default False.
        n_permute (int): Number of permutations for the test. Default 5000.
        **kwargs (dict): Forwarded to `seaborn.violinplot`, plus `fontsize` for
            the axis label and title (default 18).

    Returns:
        pl.DataFrame | tuple[pl.DataFrame, dict]: The long-format frame with
            columns `Distance`, `Type`, `Group`, or `(long_df, stats)` when
            `permutation_test=True`, where `stats` maps each group label to its
            permutation-test result.
    """
    from nltools.plotting import plot_mean_label_distance

    distance, labels = _label_distance_inputs(adj, labels)
    return plot_mean_label_distance(
        distance,
        labels,
        ax=ax,
        permutation_test=permutation_test,
        n_permute=n_permute,
        **kwargs,
    )


def plot_between_label_distance(  # nosemgrep: kwargs-internal-forwarding  # forwards to seaborn via plot_between_label_distance
    adj, *, labels=None, ax=None, permutation_test=True, n_permute=5000, **kwargs
):
    """Create a heatmap of the average distance between every pair of labels.

    Args:
        adj (Adjacency): Adjacency instance (must be a single matrix).
        labels (np.ndarray, optional): Group label per node; defaults to `adj.labels`.
        ax (matplotlib.axes.Axes, optional): Axis to draw on.
        permutation_test (bool): Also compute the mean-difference and p-value
            matrices from a two-sample permutation test. Default True.
        n_permute (int): Number of permutations for the test. Default 5000.
        **kwargs (dict): Forwarded to `seaborn.heatmap`.

    Returns:
        tuple[pl.DataFrame, ...]: `(long_df, within_mean_df)` without the
            permutation test, or `(long_df, within_mean_df, mean_diff_df, p_df)`
            with it.
    """
    from nltools.plotting import plot_between_label_distance as _plot_between

    distance, labels = _label_distance_inputs(adj, labels)
    return _plot_between(
        distance,
        labels,
        ax=ax,
        permutation_test=permutation_test,
        n_permute=n_permute,
        **kwargs,
    )


def stats_label_distance(
    adj, *, labels=None, n_permute=5000, n_jobs=-1, progress_bar=False
):
    """Calculate permutation tests on within and between label distance.

    Args:
        adj (Adjacency): Adjacency instance (must be a single matrix).
        labels (np.ndarray, optional): Group label per node; defaults to `adj.labels`.
        n_permute (int): Number of permutations to run. Default 5000.
        n_jobs (int): Number of parallel jobs. Default -1 (all cores).
        progress_bar (bool): If True, show a progress bar. Default False.

    Returns:
        dict: Per-group within-vs-between distance differences and p-values, keyed by
            group label.
    """
    from copy import deepcopy

    from nltools.algorithms.inference import two_sample_permutation_test

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
            within,
            between,
            n_permute=n_permute,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
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
        labels (np.ndarray, optional): Cluster/group label per node; defaults to
            `adj.labels`.
        ax (matplotlib.axes.Axes, optional): Axis to draw on.
        permutation_test (bool): Whether to run a permutation test. Default True.
        n_permute (int): Number of permutations for the test. Default 5000.
        colors (list, optional): RGB triplets, one per cluster. Default: seaborn
            `'hls'` palette.
        figsize (tuple): Figure size. Default (6, 4).

    Returns:
        pl.DataFrame: Columns `label` and `mean_silhouette`, plus `p` when
            `permutation_test=True`.
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


def cluster_summary(adj, *, clusters=None, summary="mean", scope="within"):
    """Provide summaries of clusters within Adjacency matrices.

    Computes the mean/median of within- or between-cluster values. Requires a
    list of cluster ids indicating the cluster of each row/column.

    Args:
        adj (Adjacency): Adjacency instance.
        clusters (list): Cluster label for each row/column.
        summary (str | None): Central tendency, `'mean'` or `'median'`. If None,
            return all values instead of a summary.
        scope (str): Summarize `'within'` cluster or `'between'` clusters.

    Returns:
        dict: Per-cluster summaries keyed by cluster label.
    """
    if summary not in ["mean", "median", None]:
        raise ValueError("summary must be ['mean','median', None]")

    distance = np.asarray(adj.squareform())
    clusters = np.asarray(clusters)

    if len(clusters) != distance.shape[0]:
        raise ValueError("Cluster labels must be same length as distance matrix")

    out = {}
    for i in list(set(clusters.tolist())):
        mask_i = clusters == i
        if scope == "within":
            within_vals = distance[np.ix_(mask_i, mask_i)][
                np.triu_indices(mask_i.sum(), k=1)
            ]
            if summary == "mean":
                out[i] = float(np.mean(within_vals))
            elif summary == "median":
                out[i] = float(np.median(within_vals))
            else:
                out[i] = within_vals
        elif scope == "between":
            between_block = distance[np.ix_(mask_i, ~mask_i)]
            if summary == "mean":
                out[i] = float(np.mean(between_block))
            elif summary == "median":
                out[i] = float(np.median(between_block))
            else:
                out[i] = between_block
    return out
