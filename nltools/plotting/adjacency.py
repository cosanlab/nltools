"""Adjacency matrix visualization — stacked plots, distance, and silhouette."""

__all__ = [
    "plot_stacked_adjacency",
    "plot_mean_label_distance",
    "plot_between_label_distance",
    "plot_silhouette",
]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltools.stats import (
    one_sample_permutation_test,
    two_sample_permutation_test,
)


def plot_stacked_adjacency(adjacency1, adjacency2, normalize=True, **kwargs):
    """Create stacked adjacency to illustrate similarity.

    Args:
        matrix1:  Adjacency instance 1
        matrix2:  Adjacency instance 2
        normalize: (boolean) Normalize matrices.

    Returns:
        matplotlib figure
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
    """Create a violin plot indicating within and between label distance.

    Args:
        distance:  pandas dataframe of distance
        labels: labels indicating columns and rows to group
        ax: matplotlib axis to plot on
        permutation_test: (bool) indicates whether to run permuatation test or not
        n_permute: (int) number of permutations to run
        fontsize: (int) fontsize for plot labels
    Returns:
        f: heatmap
        stats: (optional if permutation_test=True) permutation results

    """

    if not isinstance(distance, pd.DataFrame):
        raise ValueError("distance must be a pandas dataframe")

    if distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be square.")

    if len(labels) != distance.shape[0]:
        raise ValueError("Labels must be same length as distance matrix")

    out = pd.DataFrame(columns=["Distance", "Group", "Type"], index=None)
    for i in labels.unique():
        tmp_w = pd.DataFrame(columns=out.columns, index=None)
        tmp_w["Distance"] = distance.loc[labels == i, labels == i].values[
            np.triu_indices(sum(labels == i), k=1)
        ]
        tmp_w["Type"] = "Within"
        tmp_w["Group"] = i
        tmp_b = pd.DataFrame(columns=out.columns, index=None)
        tmp_b["Distance"] = distance.loc[labels == i, labels != i].values.flatten()
        tmp_b["Type"] = "Between"
        tmp_b["Group"] = i
        out = out.append(tmp_w).append(tmp_b)
    f = sns.violinplot(
        x="Group",
        y="Distance",
        hue="Type",
        data=out,
        split=True,
        inner="quartile",
        palette={"Within": "lightskyblue", "Between": "red"},
        ax=ax,
        **kwargs,
    )
    f.set_ylabel("Average Distance", fontsize=fontsize)
    f.set_title("Average Group Distance", fontsize=fontsize)
    if permutation_test:
        stats = dict()
        for i in labels.unique():
            # Between group test
            tmp1 = out.loc[(out["Group"] == i) & (out["Type"] == "Within"), "Distance"]
            tmp2 = out.loc[(out["Group"] == i) & (out["Type"] == "Between"), "Distance"]
            stats[str(i)] = two_sample_permutation_test(tmp1, tmp2, n_permute=n_permute)
        return (f, stats)
    else:
        return f


def plot_between_label_distance(
    distance,
    labels,
    ax=None,
    permutation_test=True,
    n_permute=5000,
    fontsize=18,
    **kwargs,
):
    """Create a heatmap indicating average between label distance


    Args:
        distance: (pandas dataframe) brain_distance matrix
        labels: (pandas dataframe) group labels
        ax: axis to plot (default=None)
        permutation_test: (boolean)
        n_permute: (int) number of samples for permuation test
        fontsize: (int) size of font for plot
    Returns:
        f: heatmap
        out: pandas dataframe of pairwise distance between conditions
        within_dist_out: average pairwise distance matrix
        mn_dist_out: (optional if permutation_test=True) average difference in distance between conditions
        p_dist_out: (optional if permutation_test=True) p-value for difference in distance between conditions
    """

    labels = np.unique(np.array(labels))

    out = pd.DataFrame(columns=["Distance", "Group", "Comparison"], index=None)
    for i in labels:
        for j in labels:
            tmp_b = pd.DataFrame(columns=out.columns, index=None)
            if (
                distance.loc[labels == i, labels == j].shape[0]
                == distance.loc[labels == i, labels == j].shape[1]
            ):
                tmp_b["Distance"] = distance.loc[labels == i, labels == i].values[
                    np.triu_indices(sum(labels == i), k=1)
                ]
            else:
                tmp_b["Distance"] = distance.loc[
                    labels == i, labels == j
                ].values.flatten()
            tmp_b["Comparison"] = j
            tmp_b["Group"] = i
            out = out.append(tmp_b)

    within_dist_out = pd.DataFrame(
        np.zeros((len(out["Group"].unique()), len(out["Group"].unique()))),
        columns=out["Group"].unique(),
        index=out["Group"].unique(),
    )
    for i in out["Group"].unique():
        for j in out["Comparison"].unique():
            within_dist_out.loc[i, j] = out.loc[
                (out["Group"] == i) & (out["Comparison"] == j)
            ]["Distance"].mean()

    if ax is None:
        _, ax = plt.subplots(1)
    else:
        plt.figure()

    if permutation_test:
        mn_dist_out = pd.DataFrame(
            np.zeros((len(out["Group"].unique()), len(out["Group"].unique()))),
            columns=out["Group"].unique(),
            index=out["Group"].unique(),
        )
        p_dist_out = pd.DataFrame(
            np.zeros((len(out["Group"].unique()), len(out["Group"].unique()))),
            columns=out["Group"].unique(),
            index=out["Group"].unique(),
        )
        for i in out["Group"].unique():
            for j in out["Comparison"].unique():
                tmp1 = out.loc[
                    (out["Group"] == i) & (out["Comparison"] == i), "Distance"
                ]
                tmp2 = out.loc[
                    (out["Group"] == i) & (out["Comparison"] == j), "Distance"
                ]
                s = two_sample_permutation_test(tmp1, tmp2, n_permute=n_permute)
                mn_dist_out.loc[i, j] = s["mean_diff"]
                p_dist_out.loc[i, j] = s["p"]
        sns.heatmap(mn_dist_out, ax=ax, square=True, **kwargs)
        sns.heatmap(
            mn_dist_out,
            mask=p_dist_out > 0.05,
            square=True,
            linewidth=2,
            annot=True,
            ax=ax,
            cbar=False,
        )
        return (out, within_dist_out, mn_dist_out, p_dist_out)
    else:
        sns.heatmap(within_dist_out, ax=ax, square=True, **kwargs)
        return (out, within_dist_out)


def plot_silhouette(
    distance, labels, ax=None, permutation_test=True, n_permute=5000, **kwargs
):
    """Create a silhouette plot indicating between relative to within label distance

    Args:
        distance: (pandas dataframe) brain_distance matrix
        labels: (pandas dataframe) group labels
        ax: axis to plot (default=None)
        permutation_test: (boolean)
        n_permute: (int) number of samples for permuation test

    Optional keyword args:
        figsize: (list) dimensions of silhouette plot
        colors: (list) color triplets for silhouettes. Length must equal number of unique labels

    Returns:
        # f: heatmap
        # out: pandas dataframe of pairwise distance between conditions
        # within_dist_out: average pairwise distance matrix
        # mn_dist_out: (optional if permutation_test=True) average difference in distance between conditions
        # p_dist_out: (optional if permutation_test=True) p-value for difference in distance between conditions
    """

    # Define label set
    labelSet = np.unique(np.array(labels))
    n_clusters = len(labelSet)

    # Set defaults for plot design
    if "colors" not in kwargs.keys():
        colors = sns.color_palette("hls", n_clusters)
    if "figsize" not in kwargs.keys():
        figsize = (6, 4)

    # Compute silhouette scores
    out = pd.DataFrame(columns=("Label", "MeanWit", "MeanBet", "Sil"))
    for index in range(len(labels)):
        label = labels.iloc[index]
        sameIndices = [
            i for i, labelcur in enumerate(labels) if (labelcur == label) & (i != index)
        ]
        within = distance.iloc[index, sameIndices].values.flatten()
        otherIndices = [i for i, labelcur in enumerate(labels) if (labelcur != label)]
        between = distance.iloc[index, otherIndices].values.flatten()
        silhouetteScore = (np.mean(between) - np.mean(within)) / max(
            np.mean(between), np.mean(within)
        )
        out_tmp = pd.DataFrame(columns=out.columns)
        out_tmp.at[index] = index
        out_tmp["Label"] = label
        out_tmp["MeanWit"] = np.mean(within)
        out_tmp["MeanBet"] = np.mean(between)
        out_tmp["Sil"] = silhouetteScore
        out = out.append(out_tmp)
    sample_silhouette_values = out["Sil"]

    # Plot
    with sns.axes_style("white"):
        if ax is None:
            _, ax = plt.subplots(1, figsize=figsize)
        else:
            plt.plot(figsize=figsize)
    x_lower = 10
    labelX = []
    for labelInd in range(n_clusters):
        label = labelSet[labelInd]
        ith_cluster_silhouette_values = sample_silhouette_values[labels == label]
        ith_cluster_silhouette_values.sort_values(inplace=True)
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        x_upper = x_lower + size_cluster_i

        color = colors[labelInd]
        with sns.axes_style("white"):
            plt.fill_between(
                np.arange(x_lower, x_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
            )

        labelX = np.hstack((labelX, np.mean([x_lower, x_upper])))
        x_lower = x_upper + 3

    # Format plot
    ax.set_xticks(labelX)
    ax.set_xticklabels(labelSet)
    ax.set_title("Silhouettes", fontsize=18)
    ax.set_xlim([5, 10 + len(labels) + n_clusters * 3])

    # Permutation test on mean silhouette score per label
    if permutation_test:
        outAll = pd.DataFrame(columns=["label", "mean", "p"])
        for labelInd in range(n_clusters):
            temp = pd.DataFrame(columns=outAll.columns)
            label = labelSet[labelInd]
            data = sample_silhouette_values[labels == label]
            temp.loc[labelInd, "label"] = label
            temp.loc[labelInd, "mean"] = np.mean(data)
            if np.mean(data) > 0:  # Only test positive mean silhouette scores
                statsout = one_sample_permutation_test(data, n_permute=n_permute)
                temp["p"] = statsout["p"]
            else:
                temp["p"] = 999
            outAll = outAll.append(temp)
        return outAll
    else:
        return
