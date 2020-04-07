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
]
__author__ = ["Luke Chang"]
__license__ = "MIT"

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq
from nltools.stats import two_sample_permutation, one_sample_permutation
from nilearn.plotting import plot_glass_brain, plot_stat_map, view_img, view_img_on_surf
from nltools.prefs import MNI_Template, resolve_mni_path
from nltools.utils import attempt_to_import
from ipywidgets import BoundedFloatText, BoundedIntText
from ipywidgets import interact
import sklearn
import warnings
import os

# Optional dependencies
ipywidgets = attempt_to_import(
    "ipywidgets", name="ipywidgets", fromlist=["interact", "fixed", "widgets"]
)


def plot_interactive_brain(
    brain,
    threshold=1e-6,
    surface=False,
    percentile_threshold=False,
    anatomical=None,
    **kwargs
):
    """
    This function leverages nilearn's new javascript based brain viewer functions to create interactive plotting functionality.

    Args:
        brain (nltools.Brain_Data): a Brain_Data instance of 1d or 2d shape (i.e. 3d or 4d volume)
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
        threshold = int(threshold[:-1])

    if len(brain.shape()) == 2:
        time_slider = True
        max_idx = brain.shape()[0] - 1
    elif len(brain.shape()) == 1:
        time_slider = False
    else:
        raise ValueError("Brain_Data object is not 1d or 2d")

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
        **kwargs
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
        objIn:(list/Brain_Data) if list will compute difference map first
        how: (list) whether to plot a glass brain 'glass', 3 view-multi-slice mni 'mni', or both 'full'
        thr: (str) what method to use for multiple comparisons correction unc, fdr, or tfce
        alpha: (float) p-value threshold
        nperm: (int) number of permutations for tcfe; default 1000
        cut_coords: (list) x,y,z coords to plot brain slice
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
            **kwargs
        )
        for v, c in zip(views, cut_coords):
            plot_stat_map(
                obj.to_nifti(),
                cut_coords=c,
                display_mode=v,
                cmap=cmap,
                bg_img=resolve_mni_path(MNI_Template)["brain"],
                **kwargs
            )
    elif how == "glass":
        plot_glass_brain(
            obj.to_nifti(),
            display_mode="lzry",
            colorbar=True,
            cmap=cmap,
            plot_abs=False,
            **kwargs
        )
    elif how == "mni":
        for v, c in zip(views, cut_coords):
            plot_stat_map(
                obj.to_nifti(),
                cut_coords=c,
                display_mode=v,
                cmap=cmap,
                bg_img=resolve_mni_path(MNI_Template)["brain"],
                **kwargs
            )
    del obj
    del out
    return


def plot_brain(objIn, how="full", thr_upper=None, thr_lower=None, save=False, **kwargs):
    """
    More complete brain plotting of a Brain_Data instance
    Args:
        obj: (Brain_Data) object to plot
        how: (str) whether to plot a glass brain 'glass', 3 view-multi-slice mni 'mni', or both 'full'
        thr_upper: (str/float) thresholding of image. Can be string for percentage, or float for data units (see Brain_Data.threshold()
        thr_lower: (str/float) thresholding of image. Can be string for percentage, or float for data units (see Brain_Data.threshold()
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
        print("Plotting unthresholded image")
    else:
        if isinstance(thr_upper, str):
            print("Plotting top %s of voxels" % thr_upper)
        elif isinstance(thr_upper, (float, int)):
            print("Plotting voxels with stat value >= %s" % thr_upper)
        if isinstance(thr_lower, str):
            print("Plotting lower %s of voxels" % thr_lower)
        elif isinstance(thr_lower, (float, int)):
            print("Plotting voxels with stat value <= %s" % thr_lower)

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
            **kwargs
        )
        if save:
            plt.savefig(glass_save, bbox_inches="tight")
        for v, c, savefile in zip(views, coords, saves):
            plot_stat_map(
                obj.to_nifti(),
                cut_coords=c,
                display_mode=v,
                cmap=cmap,
                bg_img=resolve_mni_path(MNI_Template)["brain"],
                **kwargs
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
            **kwargs
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
                bg_img=resolve_mni_path(MNI_Template)["brain"],
                **kwargs
            )
            if save:
                plt.savefig(savefile, bbox_inches="tight")
    del obj  # save memory
    return


def dist_from_hyperplane_plot(stats_output):
    """ Plot SVM Classification Distance from Hyperplane

    Args:
        stats_output: a pandas file with prediction output

    Returns:
        fig: Will return a seaborn plot of distance from hyperplane

    """

    if "dist_from_hyperplane_xval" in stats_output.columns:
        sns.factorplot(
            "subject_id",
            "dist_from_hyperplane_xval",
            hue="Y",
            data=stats_output,
            kind="point",
        )
    else:
        sns.factorplot(
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
    """ Plot Prediction Scatterplot

    Args:
        stats_output: a pandas file with prediction output

    Returns:
        fig: Will return a seaborn scatterplot

    """

    if "yfit_xval" in stats_output.columns:
        sns.lmplot("Y", "yfit_xval", data=stats_output)
    else:
        sns.lmplot("Y", "yfit_all", data=stats_output)
    plt.xlabel("Y", fontsize=16)
    plt.ylabel("Predicted Value", fontsize=16)
    plt.title("Prediction", fontsize=18)
    return 


def probability_plot(stats_output):
    """ Plot Classification Probability

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
    """ Plot 1-Specificity by Sensitivity

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
    """ Create stacked adjacency to illustrate similarity.

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
    **kwargs
):
    """ Create a violin plot indicating within and between label distance.

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
        **kwargs
    )
    f.set_ylabel("Average Distance", fontsize=fontsize)
    f.set_title("Average Group Distance", fontsize=fontsize)
    if permutation_test:
        stats = dict()
        for i in labels.unique():
            # Between group test
            tmp1 = out.loc[(out["Group"] == i) & (out["Type"] == "Within"), "Distance"]
            tmp2 = out.loc[(out["Group"] == i) & (out["Type"] == "Between"), "Distance"]
            stats[str(i)] = two_sample_permutation(tmp1, tmp2, n_permute=n_permute)
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
    **kwargs
):
    """ Create a heatmap indicating average between label distance


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
                s = two_sample_permutation(tmp1, tmp2, n_permute=n_permute)
                mn_dist_out.loc[i, j] = s["mean"]
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
    """ Create a silhouette plot indicating between relative to within label distance

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
                statsout = one_sample_permutation(data, n_permute=n_permute)
                temp["p"] = statsout["p"]
            else:
                temp["p"] = 999
            outAll = outAll.append(temp)
        return outAll
    else:
        return

def component_viewer(output, tr=2.0):
    ''' This a function to interactively view the results of a decomposition analysis
    
    Args:
        output: (dict) output dictionary from running Brain_data.decompose()
        tr: (float) repetition time of data
    '''

    def component_inspector(component, threshold):
        '''This a function to be used with ipywidgets to interactively view a decomposition analysis

            Make sure you have tr and output assigned to variables.

            Example:

                from ipywidgets import BoundedFloatText, BoundedIntText
                from ipywidgets import interact

                tr = 2.4
                output = data_filtered_smoothed.decompose(algorithm='ica', n_components=30, axis='images', whiten=True)

                interact(component_inspector, component=BoundedIntText(description='Component', value=0, min=0, max=len(output['components'])-1),
                      threshold=BoundedFloatText(description='Threshold', value=2.0, min=0, max=4, step=.1))

        '''
        _, ax = plt.subplots(nrows=3, figsize=(12,8))
        thresholded = (output['components'][component] - output['components'][component].mean())*(1/output['components'][component].std())
        thresholded.data[np.abs(thresholded.data) <= threshold] = 0
        plot_stat_map(thresholded.to_nifti(), cut_coords=range(-40, 70, 10),
                      display_mode='z', black_bg=True, colorbar=True, annotate=False,
                      draw_cross=False, axes=ax[0])
        if isinstance(output['decomposition_object'], (sklearn.decomposition.PCA)):
            var_exp = output['decomposition_object'].explained_variance_ratio_[component]
            ax[0].set_title(f"Component: {component}/{len(output['components'])}, Variance Explained: {var_exp:2.2}", fontsize=18)
        else:
            ax[0].set_title(f"Component: {component}/{len(output['components'])}", fontsize=18)

        ax[1].plot(output['weights'][:, component], linewidth=2, color='red')
        ax[1].set_ylabel('Intensity (AU)', fontsize=18)
        ax[1].set_title(f'Timecourse (TR={tr})', fontsize=16)
        y = fft(output['weights'][:, component])
        f = fftfreq(len(y), d=tr)
        ax[2].plot(f[f > 0], np.abs(y)[f > 0]**2)
        ax[2].set_ylabel('Power', fontsize=18)
        ax[2].set_xlabel('Frequency (Hz)', fontsize=16)

    interact(component_inspector, component=BoundedIntText(description='Component', value=0, min=0, max=len(output['components'])-1),
              threshold=BoundedFloatText(description='Threshold', value=2.0, min=0, max=4, step=.1))

