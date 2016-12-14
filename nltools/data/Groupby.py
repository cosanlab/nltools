from __future__ import division

'''
    NeuroLearn Groupby Class
    ==========================================
    Class to represent groupby objects

'''

__all__ = ['Groupby']
__author__ = ["Luke Chang"]
__license__ = "MIT"

import os
import cPickle 
import nibabel as nib
from nltools.utils import get_resource_path, set_algorithm, get_anatomical
from nltools.cross_validation import set_cv
from nltools.plotting import dist_from_hyperplane_plot, scatterplot, probability_plot, roc_plot
from nltools.stats import pearson,fdr,threshold, fisher_r_to_z, correlation_permutation,one_sample_permutation,two_sample_permutation
from nltools.mask import expand_mask,collapse_mask
from nltools.analysis import Roc
from nltools.data import Brain_Data
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_img
from nilearn.masking import intersect_masks
from nilearn.plotting.img_plotting import plot_epi, plot_roi, plot_stat_map
from copy import deepcopy
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp, t, norm
from scipy.signal import detrend
from scipy.spatial.distance import squareform
import six
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import pairwise_distances
from nltools.pbs_job import PBS_Job
import seaborn as sns


class Groupby(object):
    def __init__(self, data, mask):
        
        if not isinstance(data,Brain_Data):
            raise ValueError('Groupby requires a Brain_Data instance.')
        if not isinstance(mask,Brain_Data):
            if isinstance(mask, nib.Nifti1Image):
                mask = Brain_Data(mask)
            else:
                raise ValueError('mask must be a Brain_Data instance.')
        
        mask.data = np.round(mask.data).astype(int)
        if len(mask.shape()) <= 1:
            if len(np.unique(mask.data)) > 2:
                mask = expand_mask(mask)
            else:
                raise ValueError('mask does not have enough groups.')
        
        self.mask = mask
        self.split(data,mask)

    def __repr__(self):
        return '%s.%s(len=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            len(self),
            )
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        for x in self.data:
            yield (x,self.data[x])
            
    def __getitem__(self,index):
        if isinstance(index, int):
            return self.data[index]
        else:
            raise ValueError('Groupby currently only supports integer indexing')
            
    def split(self, data, mask):
        '''Split Brain_Data instance into separate masks and store as a dictionary.'''
        
        self.data = {}
        for i,m in enumerate(mask):
            self.data[i] = data.apply_mask(m)
            
    def apply(self, method):
        '''Apply Brain_Data instance methods to each element of Groupby object.'''
        return dict([(i,getattr(x,method)()) for i,x in self])
    
    def combine(self, value_dict):
        '''Combine value dictionary back into masks'''
        out = self.mask.copy().astype(float)
        for i in value_dict.iterkeys():
            if isinstance(value_dict[i],Brain_Data):
                if value_dict[i].shape()[0]==np.sum(self.mask[i].data):
                    out.data[i,out.data[i,:]==1] = value_dict[i].data
                else:
                    raise ValueError('Brain_Data instances are different shapes.')
            elif isinstance(value_dict[i],(float,int,bool,np.number)):
                out.data[i,:] = out.data[i,:]*value_dict[i]
            else:
                raise ValueError('No method for aggregation implented for %s yet.' % type(value_dict[i]))
        return out.sum()