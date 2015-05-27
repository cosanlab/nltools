"""Various statistical helper functions"""

import numpy as np
from scipy.stats import ss

def pearson(x, y):
    """ Correlates row vector x with each row vector in 2D array y. 
    From neurosynth.stats.py - author: Tal Yarkoni
    """
    data = np.vstack((x, y))
    ms = data.mean(axis=1)[(slice(None, None, None), None)]
    datam = data - ms
    datass = np.sqrt(ss(datam, axis=1))
    temp = np.dot(datam[1:], datam[0].T)
    rs = temp / (datass[1:] * datass[0])
    return rs

def auc(x, y):
    """ Calculate area under the curve using triangle method. 
    From Matlab Canlab Core roc_calc - author: Tor Wager
    Args:
        x: false positive rate on x axis
        y: true positive rate on y axis
    Returns:
        auc: will return area under the curve
    """
    fpr_unique = np.unique(x)
    tpr_unique = np.unique(y)

    if any((fpr_unique == 0) & (tpr_unique == 1)):
        # Fix for AUC = 1 if no overlap; triangle method not perfectly accurate here.
        auc = 1
    else:
        a = np.zeros(len(fpr_unique))
        for i in range(1,len(fpr_unique)):
            xdiff = fpr_unique[i] - fpr_unique[i - 1]
            ydiff = tpr_unique[i] - tpr_unique[i - 1]
            a[i] = xdiff * tpr_unique[i - 1] + xdiff * ydiff / 2;  # area of rect + area of triangle
            auc = sum(a);
    return auc

    