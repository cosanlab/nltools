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

    fig = sns.factorplot("SubID", "xval_dist_from_hyperplane", hue="Y", data=stats_output,
                        kind='point')
    plt.xlabel('Subject')
    plt.ylabel('Distance from Hyperplane')
    plt.title('Classification')
    return fig


def scatterplot(stats_output):
    """ Plot Prediction Scatterplot
    Args:
        stats_output: a pandas file with prediction output
    Returns:
        fig: Will return a seaborn scatterplot
    """

    fig = sns.lmplot("Y", "yfit", data=stats_output)
    plt.xlabel('Y')
    plt.ylabel('Predicted Value')
    plt.title('Prediction')
    return fig


def probability_plot(stats_output):
    """ Plot Classification Probability
    Args:
        stats_output: a pandas file with prediction output
    Returns:
        fig: Will return a seaborn scatterplot
    """

    fig = sns.lmplot("Y", "Probability", data=stats_output,logistic=True)
    plt.xlabel('Y')
    plt.ylabel('Predicted Probability')
    plt.title('Prediction')
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


