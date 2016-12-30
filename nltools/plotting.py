'''
NeuroLearn Plotting Tools
=========================

Numerous functions to plot data

'''

__all__ = ['dist_from_hyperplane_plot',
            'scatterplot',
            'probability_plot',
            'roc_plot',
            'decode_radar_plot']
__author__ = ["Luke Chang"]
__license__ = "MIT"

import pandas as pd
import seaborn as sns    
import matplotlib.pyplot as plt

def dist_from_hyperplane_plot(stats_output):
    """ Plot SVM Classification Distance from Hyperplane

    Args:
        stats_output: a pandas file with prediction output

    Returns:
        fig: Will return a seaborn plot of distance from hyperplane

    """

    if "dist_from_hyperplane_xval" in stats_output.columns:
        fig = sns.factorplot("subject_id", "dist_from_hyperplane_xval", hue="Y", data=stats_output,
                        kind='point')
    else:
        fig = sns.factorplot("subject_id", "dist_from_hyperplane_all", hue="Y", data=stats_output,
                        kind='point')
    plt.xlabel('Subject', fontsize=16)
    plt.ylabel('Distance from Hyperplane', fontsize=16)
    plt.title('Classification', fontsize=18)
    return fig


def scatterplot(stats_output):
    """ Plot Prediction Scatterplot

    Args:
        stats_output: a pandas file with prediction output

    Returns:
        fig: Will return a seaborn scatterplot

    """

    if "yfit_xval" in stats_output.columns:
        fig = sns.lmplot("Y", "yfit_xval", data=stats_output)
    else:
        fig = sns.lmplot("Y", "yfit_all", data=stats_output)
    plt.xlabel('Y', fontsize=16)
    plt.ylabel('Predicted Value', fontsize=16)
    plt.title('Prediction', fontsize=18)
    return fig


def probability_plot(stats_output):
    """ Plot Classification Probability

    Args:
        stats_output: a pandas file with prediction output

    Returns:
        fig: Will return a seaborn scatterplot

    """
    if "Probability_xval" in stats_output.columns:
        fig = sns.lmplot("Y", "Probability_xval", data=stats_output,logistic=True)
    else:
        fig = sns.lmplot("Y", "Probability_all", data=stats_output,logistic=True)
    plt.xlabel('Y', fontsize=16)
    plt.ylabel('Predicted Probability', fontsize=16)
    plt.title('Prediction', fontsize=18)
    return fig

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

    fig = plt.figure()
    plt.plot(fpr,tpr,color='red',linewidth=3)
    # fig = sns.tsplot(tpr,fpr,color='red',linewidth=3)
    plt.xlabel('(1 - Specificity)', fontsize=16);
    plt.ylabel('Sensitivity', fontsize=16)
    plt.title('ROC Plot', fontsize=18)
    return fig

def decode_radar_plot(data, n_top=3, overplot=False, labels=None, palette='husl'):
    """ Create a radar plot for displaying decoding results

    Args:
        data: pandas object with labels as indices
        n_top: number of top results to display
        overplot: overlay multiple decoding results
        labels: Decoding labels
        palette: seaborn color palette

    Returns:
        plt: Will return a matplotlib plot

    """     

    r = np.linspace(0, 10, num=100)
    n_panels = data.shape[1]

    if labels is None:
        labels = []
        for i in range(n_panels):
            labels.extend(data.iloc[:, i].order(ascending=False)
                          .index[:n_top])
        labels = np.unique(labels)

    data = data.loc[labels, :]

    # Use hierarchical clustering to order
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, leaves_list
    dists = pdist(data, metric='correlation')
    pairs = linkage(dists)
    order = leaves_list(pairs)
    data = data.iloc[order, :]
    labels = [labels[i] for i in order]

    theta = np.linspace(0.0, 2 * np.pi, len(labels), endpoint=False)
    import matplotlib.pyplot as plt
    if overplot:
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(polar=True))
        fig.set_size_inches(10, 10)
    else:
        fig, ax = plt.subplots(1, n_panels, sharex=False, sharey=False,
                                 subplot_kw=dict(polar=True))
        fig.set_size_inches((6 * n_panels, 6))
    # A bit silly to import seaborn just for this...
    # should extract just the color_palette functionality.
    import seaborn as sns
    colors = sns.color_palette(palette, n_panels)
    for i in range(n_panels):
        if overplot:
            alpha = 0.2
        else:
            ax = axes[i]
            alpha = 0.8
        ax.set_ylim(data.values.min(), data.values.max())
        d = data.iloc[:, i].values
        ax.fill(theta, d, color=colors[i], alpha=alpha, ec='k',
                linewidth=0)
        ax.fill(theta, d, alpha=1.0, ec=colors[i],
                linewidth=2, fill=False)
        ax.set_xticks(theta)
        ax.set_xticklabels(labels, fontsize=18)
        [lab.set_fontsize(18) for lab in ax.get_yticklabels()]
        ax.set_title('Cluster %d' % i, fontsize=22, y=1.12)
    plt.tight_layout()
    return plt

