"""Adjacency matrix visualization — stacked plots, distance, and silhouette."""

__all__ = [
    "plot_stacked_adjacency",
    "plot_mean_label_distance",
    "plot_between_label_distance",
    "plot_silhouette",
]

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from nltools.stats import (
    one_sample_permutation_test,
    two_sample_permutation_test,
)


def _polars_to_pandas(df):
    """Convert a polars DataFrame to pandas without requiring pyarrow."""
    import pandas as pd

    return pd.DataFrame(df.to_dict(as_series=False))


def _as_square_ndarray(distance):
    """Accept np.ndarray, polars DataFrame, or pandas DataFrame and return a square float ndarray."""
    if isinstance(distance, np.ndarray):
        arr = distance
    elif isinstance(distance, pl.DataFrame):
        arr = distance.to_numpy()
    elif hasattr(distance, "values") and hasattr(distance, "shape"):
        arr = np.asarray(distance.values)
    else:
        arr = np.asarray(distance)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"distance must be a square matrix; got shape {arr.shape}"
        )
    return arr.astype(float)


def _within_between_values(distance_arr, labels_arr, label):
    """Return (within, between) 1-D arrays for one label."""
    mask_in = labels_arr == label
    block = distance_arr[np.ix_(mask_in, mask_in)]
    within = block[np.triu_indices(mask_in.sum(), k=1)]
    between = distance_arr[np.ix_(mask_in, ~mask_in)].ravel()
    return within, between


def plot_stacked_adjacency(adjacency1, adjacency2, normalize=True, **kwargs):
    """Create stacked adjacency to illustrate similarity.

    Args:
        adjacency1: Adjacency instance 1.
        adjacency2: Adjacency instance 2.
        normalize: Normalize matrices before stacking. Default True.
        **kwargs: Passed through to seaborn.heatmap.

    Returns:
        matplotlib axes with the stacked heatmap.
    """
    from nltools.data import Adjacency

    if not isinstance(adjacency1, Adjacency) or not isinstance(adjacency2, Adjacency):
        raise ValueError("This function requires Adjacency() instances as input.")

    upper = np.triu(adjacency2.squareform(), k=1)
    lower = np.tril(adjacency1.squareform(), k=-1)
    if normalize:
        upper = np.triu((adjacency1 - adjacency1.mean()).squareform(), k=1)
        lower = np.tril((adjacency2 - adjacency2.mean()).squareform(), k=-1)
        upper = upper / np.max(upper)
        lower = lower / np.max(lower)
    dist = upper + lower
    return sns.heatmap(
        dist, xticklabels=False, yticklabels=False, square=True, **kwargs
    )


def plot_mean_label_distance(
    distance,
    labels,
    ax=None,
    permutation_test=False,
    n_permute=5000,
    fontsize=18,
    **kwargs,
):
    """Violin plot of within- vs between-label distances.

    Args:
        distance: Square pairwise distance matrix (np.ndarray or polars DataFrame).
        labels: Array-like of length N giving a group label for each row/column.
        ax: Matplotlib axis to plot on (optional).
        permutation_test: If True, run a two-sample permutation test per group.
        n_permute: Number of permutations.
        fontsize: Font size for plot labels.
        **kwargs: Passed to seaborn.violinplot.

    Returns:
        pl.DataFrame with columns [Distance, Group, Type] in long format.
        If permutation_test=True, returns (pl.DataFrame, dict of per-group stats).
    """
    arr = _as_square_ndarray(distance)
    labels_arr = np.asarray(labels)
    if labels_arr.shape[0] != arr.shape[0]:
        raise ValueError("Labels must be same length as distance matrix")

    rows = []
    for group in np.unique(labels_arr):
        within, between = _within_between_values(arr, labels_arr, group)
        rows.append(
            pl.DataFrame(
                {"Distance": within, "Type": ["Within"] * len(within), "Group": group}
            )
        )
        rows.append(
            pl.DataFrame(
                {
                    "Distance": between,
                    "Type": ["Between"] * len(between),
                    "Group": group,
                }
            )
        )
    out = pl.concat(rows, how="vertical")

    f = sns.violinplot(
        x="Group",
        y="Distance",
        hue="Type",
        data=_polars_to_pandas(out),
        split=True,
        inner="quartile",
        palette={"Within": "lightskyblue", "Between": "red"},
        ax=ax,
        **kwargs,
    )
    f.set_ylabel("Average Distance", fontsize=fontsize)
    f.set_title("Average Group Distance", fontsize=fontsize)

    if permutation_test:
        stats = {}
        for group in np.unique(labels_arr):
            within = out.filter(
                (pl.col("Group") == group) & (pl.col("Type") == "Within")
            )["Distance"].to_numpy()
            between = out.filter(
                (pl.col("Group") == group) & (pl.col("Type") == "Between")
            )["Distance"].to_numpy()
            stats[str(group)] = two_sample_permutation_test(
                within, between, n_permute=n_permute
            )
        return out, stats
    return out


def plot_between_label_distance(
    distance,
    labels,
    ax=None,
    permutation_test=True,
    n_permute=5000,
    fontsize=18,
    **kwargs,
):
    """Heatmap of average pairwise distance between every label pair.

    Args:
        distance: Square pairwise distance matrix (np.ndarray or polars DataFrame).
        labels: Array-like of length N giving a group label for each row/column.
        ax: Matplotlib axis to plot on (optional).
        permutation_test: If True, also compute mean-difference and p-value matrices.
        n_permute: Number of permutations.
        fontsize: Reserved for future use; currently unused.
        **kwargs: Passed to seaborn.heatmap.

    Returns:
        Without permutation_test: (long_df, within_mean_df)
        With permutation_test: (long_df, within_mean_df, mean_diff_df, p_df)

        All frames are polars DataFrames. `long_df` has columns
        [Distance, Group, Comparison]. The three square-matrix-like frames
        are long format with columns [label1, label2, <value>] so they can
        be pivoted to a matrix if needed.
    """
    del fontsize  # kept for API parity, not used
    arr = _as_square_ndarray(distance)
    labels_arr = np.asarray(labels)
    if labels_arr.shape[0] != arr.shape[0]:
        raise ValueError("Labels must be same length as distance matrix")

    unique = np.unique(labels_arr)

    long_rows = []
    for i in unique:
        mask_i = labels_arr == i
        for j in unique:
            mask_j = labels_arr == j
            if i == j:
                vals = arr[np.ix_(mask_i, mask_i)][
                    np.triu_indices(mask_i.sum(), k=1)
                ]
            else:
                vals = arr[np.ix_(mask_i, mask_j)].ravel()
            long_rows.append(
                pl.DataFrame(
                    {
                        "Distance": vals,
                        "Group": i,
                        "Comparison": j,
                    }
                )
            )
    long_df = pl.concat(long_rows, how="vertical")

    within_mean_df = long_df.group_by(["Group", "Comparison"]).agg(
        pl.col("Distance").mean().alias("mean_distance")
    ).rename({"Group": "label1", "Comparison": "label2"})

    if ax is None:
        _, ax = plt.subplots(1)
    else:
        plt.figure()

    within_matrix = _long_to_matrix(
        within_mean_df, "label1", "label2", "mean_distance", unique
    )

    if permutation_test:
        mean_diff_rows = []
        p_rows = []
        for i in unique:
            within_i = long_df.filter(
                (pl.col("Group") == i) & (pl.col("Comparison") == i)
            )["Distance"].to_numpy()
            for j in unique:
                between_ij = long_df.filter(
                    (pl.col("Group") == i) & (pl.col("Comparison") == j)
                )["Distance"].to_numpy()
                if i == j or len(within_i) == 0 or len(between_ij) == 0:
                    mean_diff_rows.append(
                        {"label1": i, "label2": j, "mean_diff": 0.0}
                    )
                    p_rows.append({"label1": i, "label2": j, "p": 1.0})
                    continue
                s = two_sample_permutation_test(
                    within_i, between_ij, n_permute=n_permute
                )
                mean_diff_rows.append(
                    {"label1": i, "label2": j, "mean_diff": float(s["mean"])}
                )
                p_rows.append({"label1": i, "label2": j, "p": float(s["p"])})
        mean_diff_df = pl.DataFrame(mean_diff_rows)
        p_df = pl.DataFrame(p_rows)

        mean_diff_matrix = _long_to_matrix(
            mean_diff_df, "label1", "label2", "mean_diff", unique
        )
        p_matrix = _long_to_matrix(p_df, "label1", "label2", "p", unique)

        sns.heatmap(mean_diff_matrix, ax=ax, square=True, **kwargs)
        sns.heatmap(
            mean_diff_matrix,
            mask=p_matrix > 0.05,
            square=True,
            linewidth=2,
            annot=True,
            ax=ax,
            cbar=False,
        )
        return long_df, within_mean_df, mean_diff_df, p_df

    sns.heatmap(within_matrix, ax=ax, square=True, **kwargs)
    return long_df, within_mean_df


def _long_to_matrix(long_df, row_col, col_col, value_col, order):
    """Pivot a long-format polars frame into a numpy matrix using *order* for row/col order."""
    wide = long_df.pivot(
        values=value_col, index=row_col, on=col_col, aggregate_function="first"
    )
    out = np.zeros((len(order), len(order)), dtype=float)
    wide_dict = {row[row_col]: row for row in wide.iter_rows(named=True)}
    for i, r in enumerate(order):
        row = wide_dict.get(r, {})
        for j, c in enumerate(order):
            val = row.get(c)
            out[i, j] = float(val) if val is not None else 0.0
    return out


def plot_silhouette(
    distance, labels, ax=None, permutation_test=True, n_permute=5000, **kwargs
):
    """Silhouette plot indicating between- vs within-label distance.

    Uses the simplified silhouette definition from the original nltools
    implementation: within(i) = mean distance to other points in the same
    cluster; between(i) = mean distance to all points in other clusters
    (not the strict Rousseeuw min-over-clusters). Score is
    (between - within) / max(between, within).

    Args:
        distance: Square pairwise distance matrix (np.ndarray or polars DataFrame).
        labels: Array-like of length N giving a cluster label per row/column.
        ax: Matplotlib axis to plot on (optional).
        permutation_test: If True, run a one-sample permutation test per cluster
            on positive-mean silhouette scores.
        n_permute: Number of permutations.
        **kwargs: Optional. `colors` (list of RGB triplets, one per cluster) and
            `figsize` (tuple) control the plot appearance.

    Returns:
        pl.DataFrame with columns [label, mean_silhouette]. If permutation_test
        is True, adds a `p` column (1.0 for clusters with non-positive mean).
    """
    arr = _as_square_ndarray(distance)
    labels_arr = np.asarray(labels)
    if labels_arr.shape[0] != arr.shape[0]:
        raise ValueError("Labels must be same length as distance matrix")

    unique = np.unique(labels_arr)
    n_clusters = len(unique)
    n = arr.shape[0]

    colors = kwargs.get("colors", sns.color_palette("hls", n_clusters))
    figsize = kwargs.get("figsize", (6, 4))

    sil = np.zeros(n, dtype=float)
    for idx in range(n):
        same = (labels_arr == labels_arr[idx]) & (np.arange(n) != idx)
        other = labels_arr != labels_arr[idx]
        within_mean = arr[idx, same].mean() if same.any() else 0.0
        between_mean = arr[idx, other].mean() if other.any() else 0.0
        denom = max(within_mean, between_mean)
        sil[idx] = (between_mean - within_mean) / denom if denom > 0 else 0.0

    with sns.axes_style("white"):
        if ax is None:
            _, ax = plt.subplots(1, figsize=figsize)

    x_lower = 10
    label_x_positions = []
    for ci, cluster in enumerate(unique):
        cluster_vals = np.sort(sil[labels_arr == cluster])
        size = cluster_vals.shape[0]
        x_upper = x_lower + size
        color = colors[ci]
        with sns.axes_style("white"):
            plt.fill_between(
                np.arange(x_lower, x_upper),
                0,
                cluster_vals,
                facecolor=color,
                edgecolor=color,
            )
        label_x_positions.append(np.mean([x_lower, x_upper]))
        x_lower = x_upper + 3

    ax.set_xticks(label_x_positions)
    ax.set_xticklabels(unique)
    ax.set_title("Silhouettes", fontsize=18)
    ax.set_xlim([5, 10 + n + n_clusters * 3])

    rows = []
    for cluster in unique:
        cluster_vals = sil[labels_arr == cluster]
        rows.append(
            {"label": cluster, "mean_silhouette": float(cluster_vals.mean())}
        )
    out = pl.DataFrame(rows)

    if permutation_test:
        p_values = []
        for cluster in unique:
            cluster_vals = sil[labels_arr == cluster]
            if cluster_vals.mean() > 0:
                stats = one_sample_permutation_test(cluster_vals, n_permute=n_permute)
                p_values.append(float(stats["p"]))
            else:
                p_values.append(1.0)
        out = out.with_columns(pl.Series("p", p_values))

    return out
