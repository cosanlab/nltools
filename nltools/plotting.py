'''
NeuroLearn Plotting Tools
=========================

Numerous functions to plot data

'''

__all__ = ['dist_from_hyperplane_plot',
            'scatterplot',
            'probability_plot',
            'roc_plot',
            'decode_radar_plot',
            'plot_stacked_adjacency',
            'plot_mean_label_distance',
            'plot_between_label_distance']
__author__ = ["Luke Chang"]
__license__ = "MIT"

import pandas as pd
import seaborn as sns    
import matplotlib.pyplot as plt
import numpy as np
from nltools.stats import two_sample_permutation

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

def plot_stacked_adjacency(adjacency1,adjacency2, normalize=True, **kwargs):
    ''' Create stacked adjacency to illustrate similarity.
    
    Args:
        matrix1:  Adjacency instance 1
        matrix2:  Adjacency instance 2
        normalize: (boolean) Normalize matrices.
        
    Returns:
        matplotlib figure
    '''
    from nltools.data import Adjacency

    if not isinstance(adjacency1,Adjacency) or not isinstance(adjacency2,Adjacency):
        raise ValueError('This function requires Adjacency() instances as input.')
    
    upper = np.triu(adjacency2.squareform(),k=1)
    lower = np.tril(adjacency1.squareform(),k=-1)
    if normalize:
        upper = upper/np.max(upper)
        lower = lower/np.max(lower)
    dist = upper+lower
    return sns.heatmap(dist,xticklabels=False,yticklabels=False, square=True,**kwargs)

def plot_mean_label_distance(distance, labels, ax=None, permutation_test=False,
                            n_permute=5000, fontsize=18, **kwargs):
    ''' Create a violin plot indicating within and between label distance.
    
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
        
    '''

    if not isinstance(distance, pd.DataFrame):
        raise ValueError('distance must be a pandas dataframe')

    if distance.shape[0] != distance.shape[1]:
        raise ValueError('distance must be square.')

    if len(labels) != distance.shape[0]:
        raise ValueError('Labels must be same length as distance matrix')

    within = []; between = []
    out = pd.DataFrame(columns=['Distance','Group','Type'],index=None)
    for i in labels.unique():
        tmp_w = pd.DataFrame(columns=out.columns,index=None)
        tmp_w['Distance'] = distance.loc[labels==i,labels==i].values[np.triu_indices(sum(labels==i),k=1)]
        tmp_w['Type'] = 'Within'
        tmp_w['Group'] = i
        tmp_b = pd.DataFrame(columns=out.columns,index=None)
        tmp_b['Distance'] = distance.loc[labels==i,labels!=i].values.flatten()
        tmp_b['Type'] = 'Between'
        tmp_b['Group'] = i
        out = out.append(tmp_w).append(tmp_b)
    f = sns.violinplot(x="Group", y="Distance", hue="Type", data=out, split=True, inner='quartile',
          palette={"Within": "lightskyblue", "Between": "red"},ax=ax,**kwargs)
    f.set_ylabel('Average Distance',fontsize=fontsize)
    f.set_title('Average Group Distance',fontsize=fontsize)
    if permutation_test:
        stats = dict()
        for i in labels.unique():
            # Between group test
            tmp1 = out.loc[(out['Group']==i) & (out['Type']=='Within'),'Distance']
            tmp2 = out.loc[(out['Group']==i) & (out['Type']=='Between'),'Distance']
            stats[str(i)] = two_sample_permutation(tmp1,tmp2,n_permute=n_permute)
        return (f,stats)
    else:
        return f

def plot_between_label_distance(distance, labels, ax=None, permutation_test=True,
                                n_permute=5000, fontsize=18, **kwargs):
    ''' Create a heatmap indicating average between label distance
    
    
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
    '''

    out = pd.DataFrame(columns=['Distance','Group','Comparison'],index=None)
    for i in labels.unique():
        for j in labels.unique():
            tmp_b = pd.DataFrame(columns=out.columns,index=None)
            if distance.loc[labels==i,labels==j].shape[0]==distance.loc[labels==i,labels==j].shape[1]:
                tmp_b['Distance'] = distance.loc[labels==i,labels==i].values[np.triu_indices(sum(labels==i),k=1)]
            else:
                tmp_b['Distance'] = distance.loc[labels==i,labels==j].values.flatten()
            tmp_b['Comparison'] = j
            tmp_b['Group'] = i
            out = out.append(tmp_b)

    within_dist_out = pd.DataFrame(np.zeros((len(out['Group'].unique()),len(out['Group'].unique()))),
                               columns=out['Group'].unique(),index=out['Group'].unique())
    for i in out['Group'].unique():
        for j in out['Comparison'].unique():
            within_dist_out.loc[i,j] = out.loc[(out['Group']==i) & (out['Comparison']==j)]['Distance'].mean()  
    
    if ax is None:
        f,ax = plt.subplots(1)
    else:
        f = plt.figure()
    
    if permutation_test:
        mn_dist_out = pd.DataFrame(np.zeros((len(out['Group'].unique()),len(out['Group'].unique()))),
                               columns=out['Group'].unique(),index=out['Group'].unique())
        p_dist_out = pd.DataFrame(np.zeros((len(out['Group'].unique()),len(out['Group'].unique()))),
                               columns=out['Group'].unique(),index=out['Group'].unique())
        for i in out['Group'].unique():
            for j in out['Comparison'].unique():
                tmp1 = out.loc[(out['Group']==i) & (out['Comparison']==i),'Distance']
                tmp2 = out.loc[(out['Group']==i) & (out['Comparison']==j),'Distance']
                s = two_sample_permutation(tmp1,tmp2,n_permute=n_permute)
                mn_dist_out.loc[i,j] = s['mean']
                p_dist_out.loc[i,j] = s['p']
        sns.heatmap(mn_dist_out,ax=ax,square=True,**kwargs)
        sns.heatmap(mn_dist_out,mask=p_dist_out>.05,square=True,linewidth=2,annot=True,ax=ax,cbar=False)
        return (f, out, within_dist_out,mn_dist_out,p_dist_out)
    else:
        f = sns.heatmap(within_dist_out,ax=ax,square=True,**kwargs)
        return (f, out, within_dist_out)
