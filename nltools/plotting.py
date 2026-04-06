"""
NeuroLearn Plotting Tools
=========================

Numerous functions to plot data

"""

__all__ = [
    "dist_from_hyperplane_plot",
    "scatterplot",
    "probability_plot",
    "roc_plot",
    "plot_stacked_adjacency",
    "plot_mean_label_distance",
    "plot_between_label_distance",
    "plot_silhouette",
    "plot_t_brain",
    "plot_brain",
    "plot_interactive_brain",
    "surface_plot",
    "plot_flatmap",
]

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq
from nltools.algorithms.inference import (
    one_sample_permutation_test,
    two_sample_permutation_test,
)
from nilearn.plotting import (
    plot_glass_brain,
    plot_stat_map,
    view_img,
    view_img_on_surf,
    plot_surf_stat_map,
)
from nilearn.surface import vol_to_surf
from nltools.prefs import MNI_Template
from nltools.utils import (
    attempt_to_import,
    get_resource_path,
)
import warnings
import sklearn
import os
from pathlib import Path

# Optional dependencies
ipywidgets = attempt_to_import(
    "ipywidgets",
    name="ipywidgets",
    fromlist=["interact", "fixed", "widgets", "BoundedFloatText", "BoundedIntText"],
)


def plot_interactive_brain(
    brain,
    threshold=1e-6,
    surface=False,
    percentile_threshold=False,
    anatomical=None,
    **kwargs,
):
    """
    This function leverages nilearn's new javascript based brain viewer functions to create interactive plotting functionality.

    Args:
        brain (nltools.BrainData): a BrainData instance of 1d or 2d shape (i.e. 3d or 4d volume)
        threshold (float/str): threshold to initialize the visualization, maybe be a percentile string; default 0
        surface (bool): whether to create a surface-based plot; default False
        percentile_threshold (bool): whether to interpret threshold values as percentiles
        kwargs: optional arguments to nilearn.view_img or nilearn.view_img_on_surf

    Returns:
        interactive brain viewer widget
    """

    if ipywidgets is None:
        raise ImportError(
            "ipywidgets>=5.2.2 is required for interactive plotting. Please install this package manually or install nltools with optional arguments: pip install 'nltools[interactive_plots]'"
        )

    if isinstance(threshold, str):
        if threshold[-1] != "%":
            raise ValueError("Starting threshold provided as string must end in '%'")
        percentile_threshold = True
        warnings.warn(
            "Percentile thresholding ignores brain mask. Results are likely more liberal than you expect (e.g. with non-interactive plotting)!"
        )
        threshold = int(threshold[:-1])

    if len(brain.shape) == 2:
        time_slider = True
        max_idx = brain.shape[0] - 1
    elif len(brain.shape) == 1:
        time_slider = False
    else:
        raise ValueError("BrainData object is not 1d or 2d")

    thresh_box = ipywidgets.widgets.FloatText(value=threshold, description="Threshold")

    if time_slider:
        idx = ipywidgets.widgets.IntSlider(
            min=0,
            max=max_idx,
            step=1,
            value=0,
            orientation="horizontal",
            continuous_update=False,
            description="Volume",
            readout_format="d",
        )
    else:
        idx = ipywidgets.widgets.HTML(
            value="Image is 3D", description="Volume", placeholder=""
        )
    ipywidgets.interact(
        _viewer,
        brain=ipywidgets.fixed(brain),
        thresh=thresh_box,
        idx=idx,
        percentile_threshold=percentile_threshold,
        surface=surface,
        anatomical=ipywidgets.fixed(anatomical),
        **kwargs,
    )


def _viewer(brain, thresh, idx, percentile_threshold, surface, anatomical, **kwargs):
    if thresh == 0:
        thresh = 1e-6
    else:
        if percentile_threshold:
            thresh = str(thresh) + "%"
    if isinstance(idx, int):
        b = brain[idx].to_nifti()
    else:
        b = brain.to_nifti()
    if anatomical:
        bg_img = anatomical
    else:
        bg_img = "MNI152"
    cut_coords = kwargs.get("cut_coords", [0, 0, 0])

    if surface:
        return view_img_on_surf(b, threshold=thresh, **kwargs)
    else:
        return view_img(
            b, bg_img=bg_img, threshold=thresh, cut_coords=cut_coords, **kwargs
        )


def plot_t_brain(
    objIn, how="full", thr="unc", alpha=None, nperm=None, cut_coords=[], **kwargs
):
    """
    Takes a brain data object and computes a 1 sample t-test across it's first axis. If a list is provided will compute difference between brain data objects in list (i.e. paired samples t-test).
    Args:
        objIn (list/BrainData): if list will compute difference map first
        how (list): whether to plot a glass brain 'glass', 3 view-multi-slice mni 'mni', or both 'full'
        thr (str): what method to use for multiple comparisons correction unc, fdr, or tfce
        alpha (float): p-value threshold
        nperm (int): number of permutations for tcfe; default 1000
        cut_coords (list): x,y,z coords to plot brain slice
        kwargs: optionals args to nilearn plot functions (e.g. vmax)

    """
    if thr not in ["unc", "fdr", "tfce"]:
        raise ValueError("Acceptable threshold methods are 'unc','fdr','tfce'")
    views = ["x", "y", "z"]
    if len(cut_coords) == 0:
        cut_coords = [
            range(-40, 50, 10),
            [-88, -72, -58, -38, -26, 8, 20, 34, 46],
            [-34, -22, -10, 0, 16, 34, 46, 56, 66],
        ]
    else:
        if len(cut_coords) != 3:
            raise ValueError(
                "cut_coords must be a list of coordinates like [[xs],[ys],[zs]]"
            )
    cmap = "RdBu_r"

    if isinstance(objIn, list):
        if len(objIn) == 2:
            obj = objIn[0] - objIn[1]
        else:
            raise ValueError("Contrasts should contain only 2 list items!")

    thrDict = {}
    if thr == "tfce":
        thrDict["permutation"] = thr
        if nperm is None:
            nperm = 1000
        thrDict["n_permutations"] = nperm
        print("1-sample t-test corrected using: TFCE w/ %s permutations" % nperm)
    else:
        if thr == "unc":
            if alpha is None:
                alpha = 0.001
            thrDict[thr] = alpha
            print("1-sample t-test uncorrected at p < %.3f " % alpha)
        elif thr == "fdr":
            if alpha is None:
                alpha = 0.05
            thrDict[thr] = alpha
            print("1-sample t-test corrected at q < %.3f " % alpha)
        else:
            thrDict = None
            print("1-sample test unthresholded")

    out = objIn.ttest(threshold_dict=thrDict)
    if thrDict is not None:
        obj = out["thr_t"]
    else:
        obj = out["t"]

    if how == "full":
        plot_glass_brain(
            obj.to_nifti(),
            display_mode="lzry",
            colorbar=True,
            cmap=cmap,
            plot_abs=False,
            **kwargs,
        )
        for v, c in zip(views, cut_coords):
            plot_stat_map(
                obj.to_nifti(),
                cut_coords=c,
                display_mode=v,
                cmap=cmap,
                bg_img=MNI_Template.get_bg_image(obj.nifti_masker.affine_),
                **kwargs,
            )
    elif how == "glass":
        plot_glass_brain(
            obj.to_nifti(),
            display_mode="lzry",
            colorbar=True,
            cmap=cmap,
            plot_abs=False,
            **kwargs,
        )
    elif how == "mni":
        for v, c in zip(views, cut_coords):
            plot_stat_map(
                obj.to_nifti(),
                cut_coords=c,
                display_mode=v,
                cmap=cmap,
                bg_img=MNI_Template.get_bg_image(obj.nifti_masker.affine_),
                **kwargs,
            )
    del obj
    del out
    return


def plot_brain(
    objIn,
    how="full",
    thr_upper=None,
    thr_lower=None,
    save=False,
    verbose=False,
    **kwargs,
):
    """
    More complete brain plotting of a BrainData instance

    Args:
        obj (BrainData): object to plot
        how (str): whether to plot a glass brain 'glass', 3 view-multi-slice mni 'mni', or both 'full'
        thr_upper (str/float): thresholding of image. Can be string for percentage, or float for data units (see BrainData.threshold()
        thr_lower (str/float): thresholding of image. Can be string for percentage, or float for data units (see BrainData.threshold()
        save (str): if a string file name or path is provided plots will be saved into this directory appended with the orientation they belong to
        kwargs: optionals args to nilearn plot functions (e.g. vmax)

    """
    if thr_upper or thr_lower:
        obj = objIn.threshold(upper=thr_upper, lower=thr_lower)
    else:
        obj = objIn.copy()

    views = ["x", "y", "z"]
    coords = [
        range(-50, 51, 8),
        range(-80, 50, 10),
        range(-40, 71, 9),
    ]  # [-88,-72,-58,-38,-26,8,20,34,46]
    cmap = "RdBu_r"

    if thr_upper is None and thr_lower is None:
        msg = "Plotting unthresholded image"
    else:
        if isinstance(thr_upper, str):
            msg = "Plotting top %s of voxels" % thr_upper
        elif isinstance(thr_upper, (float, int)):
            msg = "Plotting voxels with stat value >= %s" % thr_upper
        if isinstance(thr_lower, str):
            msg = "Plotting lower %s of voxels" % thr_lower
        elif isinstance(thr_lower, (float, int)):
            msg = "Plotting voxels with stat value <= %s" % thr_lower
    if verbose:
        print(msg)

    if save:
        path, filename = os.path.split(save)
        filename, extension = filename.split(".")
        glass_save = os.path.join(path, filename + "_glass." + extension)
        x_save = os.path.join(path, filename + "_x." + extension)
        y_save = os.path.join(path, filename + "_y." + extension)
        z_save = os.path.join(path, filename + "_z." + extension)
    else:
        glass_save, x_save, y_save, z_save = None, None, None, None

    saves = [x_save, y_save, z_save]

    if how == "full":
        plot_glass_brain(
            obj.to_nifti(),
            display_mode="lzry",
            colorbar=True,
            cmap=cmap,
            plot_abs=False,
            **kwargs,
        )
        if save:
            plt.savefig(glass_save, bbox_inches="tight")
        for v, c, savefile in zip(views, coords, saves):
            plot_stat_map(
                obj.to_nifti(),
                cut_coords=c,
                display_mode=v,
                cmap=cmap,
                bg_img=MNI_Template.get_bg_image(obj.nifti_masker.affine_),
                **kwargs,
            )
            if save:
                plt.savefig(savefile, bbox_inches="tight")
    elif how == "glass":
        plot_glass_brain(
            obj.to_nifti(),
            display_mode="lzry",
            colorbar=True,
            cmap=cmap,
            plot_abs=False,
            **kwargs,
        )
        if save:
            plt.savefig(glass_save, bbox_inches="tight")
    elif how == "mni":
        for v, c, savefile in zip(views, coords, saves):
            plot_stat_map(
                obj.to_nifti(),
                cut_coords=c,
                display_mode=v,
                cmap=cmap,
                bg_img=MNI_Template.get_bg_image(obj.nifti_masker.affine_),
                **kwargs,
            )
            if save:
                plt.savefig(savefile, bbox_inches="tight")
    del obj  # save memory
    return


def dist_from_hyperplane_plot(stats_output):
    """Plot SVM Classification Distance from Hyperplane

    Args:
        stats_output: a pandas file with prediction output

    Returns:
        fig: Will return a seaborn plot of distance from hyperplane

    """

    if "dist_from_hyperplane_xval" in stats_output.columns:
        sns.catplot(
            "subject_id",
            "dist_from_hyperplane_xval",
            hue="Y",
            data=stats_output,
            kind="point",
        )
    else:
        sns.catplot(
            "subject_id",
            "dist_from_hyperplane_all",
            hue="Y",
            data=stats_output,
            kind="point",
        )
    plt.xlabel("Subject", fontsize=16)
    plt.ylabel("Distance from Hyperplane", fontsize=16)
    plt.title("Classification", fontsize=18)
    return


def scatterplot(stats_output):
    """Plot Prediction Scatterplot

    Args:
        stats_output: a pandas file with prediction output

    Returns:
        fig: Will return a seaborn scatterplot

    """

    if "yfit_xval" in stats_output.columns:
        sns.lmplot(x="Y", y="yfit_xval", data=stats_output)
    else:
        sns.lmplot(x="Y", y="yfit_all", data=stats_output)
    plt.xlabel("Y", fontsize=16)
    plt.ylabel("Predicted Value", fontsize=16)
    plt.title("Prediction", fontsize=18)
    return


def probability_plot(stats_output):
    """Plot Classification Probability

    Args:
        stats_output: a pandas file with prediction output

    Returns:
        fig: Will return a seaborn scatterplot

    """
    if "Probability_xval" in stats_output.columns:
        sns.lmplot("Y", "Probability_xval", data=stats_output, logistic=True)
    else:
        sns.lmplot("Y", "Probability_all", data=stats_output, logistic=True)
    plt.xlabel("Y", fontsize=16)
    plt.ylabel("Predicted Probability", fontsize=16)
    plt.title("Prediction", fontsize=18)
    return

    # # and plot the result
    # plt.figure(1, figsize=(4, 3))
    # plt.clf()
    # plt.scatter(X.ravel(), y, color='black', zorder=20)
    # X_test = np.linspace(-5, 10, 300)

    # def model(x):
    #     return 1 / (1 + np.exp(-x))
    # loss = model(X_test * clf.coef_ + clf.intercept_).ravel()
    # plt.plot(X_test, loss, color='blue', linewidth=3)


def roc_plot(fpr, tpr):
    """Plot 1-Specificity by Sensitivity

    Args:
        fpr: false positive rate from Roc.calculate
        tpr: true positive rate from Roc.calculate

    Returns:
        fig: Will return a matplotlib ROC plot

    """

    plt.figure()
    plt.plot(fpr, tpr, color="red", linewidth=3)
    # fig = sns.tsplot(tpr,fpr,color='red',linewidth=3)
    plt.xlabel("(1 - Specificity)", fontsize=16)
    plt.ylabel("Sensitivity", fontsize=16)
    plt.title("ROC Plot", fontsize=18)
    return


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


def component_viewer(output, tr=2.0):
    """This a function to interactively view the results of a decomposition analysis

    Args:
        output: (dict) output dictionary from running Brain_data.decompose()
        tr: (float) repetition time of data
    """

    if ipywidgets is None:
        raise ImportError(
            "ipywidgets is required for interactive plotting. Please install this package manually or install nltools with optional arguments: pip install 'nltools[interactive_plots]'"
        )

    def component_inspector(component, threshold):
        """This a function to be used with ipywidgets to interactively view a decomposition analysis

        Make sure you have tr and output assigned to variables.

        Example:

            from ipywidgets import BoundedFloatText, BoundedIntText
            from ipywidgets import interact

            tr = 2.4
            output = data_filtered_smoothed.decompose(algorithm='ica', n_components=30, axis='images', whiten=True)

            interact(component_inspector, component=BoundedIntText(description='Component', value=0, min=0, max=len(output['components'])-1),
                  threshold=BoundedFloatText(description='Threshold', value=2.0, min=0, max=4, step=.1))

        """
        _, ax = plt.subplots(nrows=3, figsize=(12, 8))
        thresholded = (
            output["components"][component] - output["components"][component].mean()
        ) * (1 / output["components"][component].std())
        thresholded.data[np.abs(thresholded.data) <= threshold] = 0
        plot_stat_map(
            thresholded.to_nifti(),
            cut_coords=range(-40, 70, 10),
            display_mode="z",
            black_bg=True,
            colorbar=True,
            annotate=False,
            draw_cross=False,
            axes=ax[0],
        )
        if isinstance(output["decomposition_object"], (sklearn.decomposition.PCA)):
            var_exp = output["decomposition_object"].explained_variance_ratio_[
                component
            ]
            ax[0].set_title(
                f"Component: {component}/{len(output['components'])}, Variance Explained: {var_exp:2.2}",
                fontsize=18,
            )
        else:
            ax[0].set_title(
                f"Component: {component}/{len(output['components'])}", fontsize=18
            )

        ax[1].plot(output["weights"][:, component], linewidth=2, color="red")
        ax[1].set_ylabel("Intensity (AU)", fontsize=18)
        ax[1].set_title(f"Timecourse (TR={tr})", fontsize=16)
        y = fft(output["weights"][:, component])
        f = fftfreq(len(y), d=tr)
        ax[2].plot(f[f > 0], np.abs(y)[f > 0] ** 2)
        ax[2].set_ylabel("Power", fontsize=18)
        ax[2].set_xlabel("Frequency (Hz)", fontsize=16)

    ipywidgets.interact(
        component_inspector,
        component=ipywidgets.BoundedIntText(
            description="Component", value=0, min=0, max=len(output["components"]) - 1
        ),
        threshold=ipywidgets.BoundedFloatText(
            description="Threshold", value=2.0, min=0, max=4, step=0.1
        ),
    )


def _get_surface_paths():
    """Get paths to included surface files.

    Returns:
        dict: Dictionary with keys for surface meshes and background maps.
            Keys include: 'pial_left', 'pial_right', 'inflated_left',
            'inflated_right', 'midthickness_left', 'midthickness_right',
            'white_left', 'white_right', 'curv_left', 'curv_right',
            'sulc_left', 'sulc_right', 'thickness_left', 'thickness_right'.
    """
    from os.path import join

    resource_path = get_resource_path().rstrip(os.pathsep)
    surfaces_dir = join(resource_path, "surfaces")

    paths = {
        # Surface meshes
        "pial_left": join(surfaces_dir, "sub-colin_hemi-L_pial.surf.gii"),
        "pial_right": join(surfaces_dir, "sub-colin_hemi-R_pial.surf.gii"),
        "inflated_left": join(surfaces_dir, "sub-colin_hemi-L_inflated.surf.gii"),
        "inflated_right": join(surfaces_dir, "sub-colin_hemi-R_inflated.surf.gii"),
        "midthickness_left": join(
            surfaces_dir, "sub-colin_hemi-L_midthickness.surf.gii"
        ),
        "midthickness_right": join(
            surfaces_dir, "sub-colin_hemi-R_midthickness.surf.gii"
        ),
        "white_left": join(surfaces_dir, "sub-colin_hemi-L_white.surf.gii"),
        "white_right": join(surfaces_dir, "sub-colin_hemi-R_white.surf.gii"),
        # Background maps
        "curv_left": join(surfaces_dir, "sub-colin_hemi-L_curv.shape.gii"),
        "curv_right": join(surfaces_dir, "sub-colin_hemi-R_curv.shape.gii"),
        "sulc_left": join(surfaces_dir, "sub-colin_hemi-L_sulc.shape.gii"),
        "sulc_right": join(surfaces_dir, "sub-colin_hemi-R_sulc.shape.gii"),
        "thickness_left": join(surfaces_dir, "sub-colin_hemi-L_thickness.shape.gii"),
        "thickness_right": join(surfaces_dir, "sub-colin_hemi-R_thickness.shape.gii"),
    }

    return paths


def _resolve_brain_input(brain):
    """Convert various input types to nibabel Nifti1Image.

    Args:
        brain: BrainData, nibabel Nifti1Image, or file path to NIfTI image.

    Returns:
        nibabel.Nifti1Image: Nifti image object.

    Raises:
        ValueError: If input cannot be converted to Nifti1Image (e.g., empty
            BrainData or file not found).
        TypeError: If input type is not supported.
    """
    import nibabel as nib
    from nltools.data import BrainData

    if isinstance(brain, BrainData):
        if len(brain) == 0:
            raise ValueError("Cannot plot empty BrainData object")
        # If multiple images, use first one
        if len(brain.shape) == 2 and brain.shape[0] > 1:
            brain = brain[0]
        return brain.to_nifti()
    elif isinstance(brain, nib.Nifti1Image):
        return brain
    elif isinstance(brain, (str, Path)):
        if not os.path.exists(brain):
            raise ValueError(f"File not found: {brain}")
        return nib.load(brain)
    else:
        raise TypeError(
            f"Input must be BrainData, nibabel Nifti1Image, or file path, got {type(brain)}"
        )


def _get_background_map(bg_map, hemi):
    """Resolve background map path.

    Args:
        bg_map (str or None): Background map type ('curvature', 'sulc', None,
            or file path to custom background map).
        hemi (str): Hemisphere ('left' or 'right').

    Returns:
        str or None: Path to background map file, or None if bg_map is None.

    Raises:
        ValueError: If bg_map is not recognized or file not found.
    """
    if bg_map is None:
        return None

    if isinstance(bg_map, str) and os.path.exists(bg_map):
        # User provided a file path
        return bg_map

    paths = _get_surface_paths()

    # Map common names to file keys
    bg_map_map = {
        "curvature": "curv",
        "curv": "curv",
        "sulc": "sulc",
        "sulcal": "sulc",
        "thickness": "thickness",
    }

    bg_key = bg_map_map.get(bg_map.lower() if isinstance(bg_map, str) else None)
    if bg_key is None:
        raise ValueError(
            f"Unknown background map: {bg_map}. "
            f"Supported options: {list(bg_map_map.keys())}, None, or file path"
        )

    key = f"{bg_key}_{hemi}"
    if key not in paths:
        raise ValueError(f"Background map not found: {key}")

    return paths[key]


def surface_plot(
    brain,
    surface="inflated",
    bg_map="curvature",
    hemi="both",
    view="montage",
    threshold=None,
    cmap="RdBu_r",
    vmax=None,
    vmin=None,
    darkness=None,
    bg_on_data=False,
    colorbar=False,
    figsize=(10, 10),
    n_samples=1,
    radius=0.0,
    interpolation="linear",
    engine="matplotlib",
    axes=None,
    save=None,
    **kwargs,
):
    """Plot neuroimaging data on cortical surface.

    Intelligently projects volumetric NIfTI data onto cortical surfaces
    and displays in customizable montage layouts. Automatically handles
    hemispheric parsing and uses included MNI152 template surfaces.

    Args:
        brain: BrainData, nibabel Nifti1Image, or file path to NIfTI image.
            If BrainData has multiple images, plots the first one.
        surface (str, optional): Surface mesh type. Options: 'pial',
            'inflated', 'midthickness', 'white'. Defaults to 'inflated'.
        bg_map (str or None, optional): Background map. Options: 'curvature',
            'sulc', None, or file path to custom background map.
            Defaults to 'curvature'.
        hemi (str, optional): Hemisphere to plot. Options: 'left', 'right',
            'both'. Defaults to 'both'.
        view (str or list, optional): View type. Options: 'lateral', 'medial',
            'montage', or list of views. Defaults to 'montage' (2×2 grid).
        threshold (float or str, optional): Threshold value. Can be a float
            or percentile string like '95%'. Defaults to None.
        cmap (str, optional): Colormap name. Defaults to 'RdBu_r'.
        vmax (float, optional): Maximum value for colormap scaling.
        vmin (float, optional): Minimum value for colormap scaling.
        darkness (float or None, optional): Background darkness (0-1). Defaults to None.
        bg_on_data (bool, optional): Overlay background on data. Defaults to False.
        colorbar (bool, optional): Show colorbar. Defaults to False.
        figsize (tuple, optional): Figure size tuple (width, height).
            Defaults to (10, 10).
        n_samples (int, optional): Number of samples for vol_to_surf projection.
            Defaults to 1.
        radius (float, optional): Sampling radius for vol_to_surf projection.
            Defaults to 0.0.
        interpolation (str, optional): Interpolation method for projection.
            Options: 'linear', 'nearest_most_frequent'. Defaults to 'linear'.
        engine (str, optional): Rendering engine. Options: 'matplotlib',
            'plotly'. Defaults to 'matplotlib'.
        axes (matplotlib.axes.Axes or list, optional): Custom matplotlib axes
            for montage layout. If None, creates new figure. Defaults to None.
        save (str or None, optional): File path to save plot. If None, plot
            is displayed but not saved. Defaults to None.
        **kwargs: Additional arguments passed to plot_surf_stat_map.

    Returns:
        matplotlib.figure.Figure or plotly.graph_objects.Figure: Figure object
        containing the surface plot(s).

    Raises:
        ValueError: If input is empty BrainData, invalid surface/view/hemi,
            or surface files not found.
        TypeError: If input type is not supported.

    Examples:
        Plot BrainData with default 2×2 montage:

        >>> from nltools.plotting import surface_plot
        >>> from nltools.data import BrainData
        >>> brain = BrainData('data.nii.gz')
        >>> fig = surface_plot(brain)

        Single hemisphere, lateral view:

        >>> fig = surface_plot(brain, hemi='left', view='lateral')

        Custom colormap and threshold:

        >>> fig = surface_plot(brain, cmap='hot', threshold=0.5)

        Percentile threshold with custom background:

        >>> fig = surface_plot(brain, threshold='95%', bg_map='sulc')
    """
    # Resolve input to nibabel image
    nifti_img = _resolve_brain_input(brain)

    # Get surface paths
    paths = _get_surface_paths()

    # Validate surface type
    valid_surfaces = ["pial", "inflated", "midthickness", "white"]
    if surface not in valid_surfaces:
        raise ValueError(
            f"Invalid surface type: {surface}. Must be one of {valid_surfaces}"
        )

    # Determine which hemispheres to plot
    if hemi == "both":
        hemispheres = ["left", "right"]
    elif hemi in ["left", "right"]:
        hemispheres = [hemi]
    else:
        raise ValueError(f"Invalid hemi: {hemi}. Must be 'left', 'right', or 'both'")

    # Determine views
    if view == "montage":
        views = ["lateral", "medial"]
    elif isinstance(view, str):
        views = [view]
    elif isinstance(view, list):
        views = view
    else:
        raise ValueError(
            f"Invalid view: {view}. Must be 'lateral', 'medial', 'montage', or list"
        )

    # Validate views
    valid_views = ["lateral", "medial"]
    for v in views:
        if v not in valid_views:
            raise ValueError(f"Invalid view: {v}. Must be one of {valid_views}")

    # Project volume to surface textures for each hemisphere
    textures = {}
    for h in hemispheres:
        surface_key = f"{surface}_{h}"
        if surface_key not in paths:
            raise ValueError(f"Surface file not found: {surface_key}")

        textures[h] = vol_to_surf(
            nifti_img,
            paths[surface_key],
            interpolation=interpolation,
            n_samples=n_samples,
            radius=radius,
        )

    # Prepare background maps
    bg_maps = {}
    for h in hemispheres:
        bg_maps[h] = _get_background_map(bg_map, h)

    # Handle threshold (percentile string)
    if isinstance(threshold, str) and threshold.endswith("%"):
        # Convert percentile to actual threshold value
        percentile = float(threshold[:-1])
        all_values = np.concatenate([textures[h] for h in hemispheres])
        all_values = all_values[~np.isnan(all_values)]
        if len(all_values) > 0:
            threshold = np.percentile(np.abs(all_values), percentile)

    # Create figure and axes if not provided
    if axes is None:
        if hemi == "both" and view == "montage":
            # Default 2×2 montage: LH lateral, RH lateral, LH medial, RH medial
            fig, axes = plt.subplots(
                2, 2, figsize=figsize, subplot_kw={"projection": "3d"}
            )
            axes = axes.flatten()
        elif len(hemispheres) == 1 and len(views) == 1:
            # Single plot
            fig, axes = plt.subplots(
                1, 1, figsize=figsize, subplot_kw={"projection": "3d"}
            )
            axes = [axes]
        else:
            # Custom layout
            n_plots = len(hemispheres) * len(views)
            fig, axes = plt.subplots(
                1, n_plots, figsize=figsize, subplot_kw={"projection": "3d"}
            )
            if n_plots == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
    else:
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        elif not isinstance(axes, list):
            axes = [axes]
        fig = axes[0].figure

    # Plot each combination
    plot_idx = 0
    for view_name in views:
        for h in hemispheres:
            if plot_idx >= len(axes):
                break

            surface_key = f"{surface}_{h}"
            mesh = paths[surface_key]
            texture = textures[h]
            bg = bg_maps[h]

            # Plot on specified axis
            plot_surf_stat_map(
                mesh,
                texture,
                hemi=h,
                view=view_name,
                bg_map=bg,
                bg_on_data=bg_on_data,
                darkness=darkness,
                colorbar=colorbar,
                cmap=cmap,
                threshold=threshold,
                vmax=vmax,
                vmin=vmin,
                axes=axes[plot_idx],
                engine=engine,
                **kwargs,
            )
            plot_idx += 1

    # Adjust layout
    if hemi == "both" and view == "montage":
        plt.subplots_adjust(wspace=-0.05, hspace=-0.1)

    # Save if requested
    if save is not None:
        fig.savefig(save, bbox_inches="tight", transparent=True, dpi=300)

    return fig


def plot_flatmap(
    brain,
    threshold=None,
    cmap="RdBu_r",
    vmax=None,
    vmin=None,
    template="fsaverage5",
    with_curvature=True,
    curvature_contrast=0.5,
    curvature_brightness=0.5,
    colorbar=True,
    colorbar_orientation="horizontal",
    figsize=(12, 6),
    title=None,
    radius=3.0,
    interpolation="linear",
    axes=None,
    save=None,
):
    """Plot brain data on cortical flatmap.

    Projects MNI152 volumetric data onto an fsaverage surface and renders
    as a 2D flattened cortical map. Uses nilearn's vol_to_surf for projection
    and matplotlib's tripcolor for rendering.

    This function provides publication-quality flatmap visualizations without
    requiring external dependencies like pycortex.

    Args:
        brain: BrainData, nibabel Nifti1Image, or file path to NIfTI image.
            Data must be in MNI152 space.
        threshold (float or str, optional): Values below this absolute
            threshold are masked. Can be a float or percentile string
            like '95%'. Defaults to None (no threshold).
        cmap (str, optional): Matplotlib colormap for data. Defaults to
            'RdBu_r' (diverging red-blue).
        vmax (float, optional): Maximum value for colormap. If None,
            uses symmetric max of absolute values.
        vmin (float, optional): Minimum value for colormap. If None
            and vmax is set, uses -vmax for diverging maps.
        template (str, optional): fsaverage resolution. Options:
            'fsaverage3' (642 vertices), 'fsaverage4' (2562),
            'fsaverage5' (10242, default), 'fsaverage6' (40962),
            'fsaverage' (163842, full resolution).
        with_curvature (bool, optional): Show sulcal/gyral pattern as
            grayscale background. Defaults to True.
        curvature_contrast (float, optional): Contrast of curvature
            (0=flat gray, 1=full contrast). Defaults to 0.5.
        curvature_brightness (float, optional): Mean brightness of
            curvature (0=dark, 1=bright). Defaults to 0.5.
        colorbar (bool, optional): Show colorbar. Defaults to True.
        colorbar_orientation (str, optional): 'horizontal' or 'vertical'.
            Defaults to 'horizontal'.
        figsize (tuple, optional): Figure size (width, height).
            Defaults to (12, 6).
        title (str, optional): Figure title. Defaults to None.
        radius (float, optional): Sampling radius in mm for vol_to_surf
            projection. Larger values provide smoother projections.
            Defaults to 3.0.
        interpolation (str, optional): Interpolation for vol_to_surf.
            Options: 'linear', 'nearest'. Defaults to 'linear'.
        axes (matplotlib.axes.Axes, optional): Existing axes to plot on.
            If None, creates new figure. Defaults to None.
        save (str, optional): File path to save figure. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The figure containing the flatmap.

    Examples:
        Basic flatmap with default settings:

        >>> from nltools.plotting import plot_flatmap
        >>> from nltools.data import BrainData
        >>> brain = BrainData('stats.nii.gz')
        >>> fig = plot_flatmap(brain)

        Thresholded with custom colormap:

        >>> fig = plot_flatmap(brain, threshold=2.5, cmap='hot')

        Percentile threshold, no curvature:

        >>> fig = plot_flatmap(brain, threshold='95%', with_curvature=False)

        High resolution for publication:

        >>> fig = plot_flatmap(brain, template='fsaverage6', figsize=(16, 8))
        >>> fig.savefig('flatmap.pdf', dpi=300)

    Notes:
        - Data is projected from MNI152 space to fsaverage surface space.
          Small alignment differences are expected at boundaries.
        - Higher resolution templates (fsaverage6, fsaverage) produce
          sharper images but take longer to render.
        - The flat surfaces are cached by nilearn after first download
          (~50MB for fsaverage5).
    """
    from nilearn import datasets, surface
    import nibabel as nib
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    # Resolve input to nibabel image
    nifti_img = _resolve_brain_input(brain)

    # Fetch fsaverage surfaces (cached after first download)
    fs = datasets.fetch_surf_fsaverage(template)

    # Project volume to surface for both hemispheres
    texture_left = surface.vol_to_surf(
        nifti_img,
        fs["pial_left"],
        radius=radius,
        interpolation=interpolation,
    )
    texture_right = surface.vol_to_surf(
        nifti_img,
        fs["pial_right"],
        radius=radius,
        interpolation=interpolation,
    )

    # Load flat surface meshes
    flat_left = nib.load(fs["flat_left"])
    flat_right = nib.load(fs["flat_right"])

    coords_left = flat_left.darrays[0].data[:, :2]  # Only X, Y for flatmap
    coords_right = flat_right.darrays[0].data[:, :2]
    faces_left = flat_left.darrays[1].data
    faces_right = flat_right.darrays[1].data

    # Offset right hemisphere to the right of left hemisphere
    gap = 20  # Gap between hemispheres in surface units
    coords_right = coords_right.copy()
    coords_right[:, 0] += coords_left[:, 0].max() - coords_right[:, 0].min() + gap

    # Load curvature for background
    if with_curvature:
        curv_left = nib.load(fs["curv_left"]).darrays[0].data
        curv_right = nib.load(fs["curv_right"]).darrays[0].data

    # Handle threshold (percentile string)
    if isinstance(threshold, str) and threshold.endswith("%"):
        percentile = float(threshold[:-1])
        all_values = np.concatenate([texture_left, texture_right])
        all_values = all_values[~np.isnan(all_values)]
        if len(all_values) > 0:
            threshold = np.percentile(np.abs(all_values), percentile)

    # Determine colormap range
    if vmax is None:
        all_values = np.concatenate([texture_left, texture_right])
        vmax = np.nanmax(np.abs(all_values))
    if vmin is None:
        vmin = -vmax  # Symmetric for diverging colormaps

    # Apply threshold masking
    if threshold is not None:
        texture_left_masked = np.where(
            np.abs(texture_left) >= threshold, texture_left, np.nan
        )
        texture_right_masked = np.where(
            np.abs(texture_right) >= threshold, texture_right, np.nan
        )
    else:
        texture_left_masked = texture_left
        texture_right_masked = texture_right

    # Create figure and axes
    if axes is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        ax = axes
        fig = ax.figure

    # Plot curvature background
    if with_curvature:
        # Normalize and adjust curvature for display
        curv_norm = Normalize(vmin=-0.5, vmax=0.5)

        # Scale by contrast and shift by brightness
        curv_left_display = (curv_norm(curv_left) - 0.5) * curvature_contrast
        curv_left_display = curv_left_display + curvature_brightness
        curv_right_display = (curv_norm(curv_right) - 0.5) * curvature_contrast
        curv_right_display = curv_right_display + curvature_brightness

        # Plot curvature as background (zorder=0)
        ax.tripcolor(
            coords_left[:, 0],
            coords_left[:, 1],
            faces_left,
            curv_left_display,
            cmap="gray",
            shading="gouraud",
            vmin=0,
            vmax=1,
            zorder=0,
        )
        ax.tripcolor(
            coords_right[:, 0],
            coords_right[:, 1],
            faces_right,
            curv_right_display,
            cmap="gray",
            shading="gouraud",
            vmin=0,
            vmax=1,
            zorder=0,
        )

    # Plot data overlay (zorder=1)
    ax.tripcolor(
        coords_left[:, 0],
        coords_left[:, 1],
        faces_left,
        texture_left_masked,
        cmap=cmap,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )
    ax.tripcolor(
        coords_right[:, 0],
        coords_right[:, 1],
        faces_right,
        texture_right_masked,
        cmap=cmap,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )

    # Style the axes
    ax.set_aspect("equal")
    ax.axis("off")

    if title is not None:
        ax.set_title(title, fontsize=14)

    # Add colorbar
    if colorbar:
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin, vmax))
        sm.set_array([])

        if colorbar_orientation == "horizontal":
            fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04)
        else:
            fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Save if requested
    if save is not None:
        fig.savefig(save, bbox_inches="tight", facecolor="white", dpi=300)

    return fig
