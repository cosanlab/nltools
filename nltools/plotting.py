'''
    NeuroLearn Plotting Tools
    =========================
    Numerous functions to plot data

    Author: Luke Chang
    License: MIT
'''

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
        fig: Will return an ROC plot
    """     

    fig = plt.plot(fpr, tpr, linewidth=3.0, color='red')
    plt.xlabel('(1 - Specificity)', fontsize=16);
    plt.ylabel('Sensitivity', fontsize=16)
    plt.title('ROC Plot', fontsize=18)
    return fig


