from __future__ import division

'''
    NeuroLearn Adjacency Class
    ================================================================
    Classes to represent similarity, distance and adjacency matrices

'''

__all__ = ['Adjacency']
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

# Optional dependencies
try:
    from mne.stats import spatio_temporal_cluster_1samp_test, ttest_1samp_no_p
except ImportError:
    pass

try:
    from pyneurovault_upload import Client
except ImportError:
    pass

try:
    import requests
except ImportError:
    pass

def _all_same(items):
    return np.all(x == items[0] for x in items)

class Adjacency(object):
    """
    Adjacency is a class to...  
    This makes it easier to perform data manipulation and analyses.

    Args:
        data: squareform matrix or flattened upper triangle of a square matrix
        Y: Pandas DataFrame of training labels
        X: Pandas DataFrame Design Matrix for running univariate models 
        mask: binary nifiti file to mask brain data
        output_file: Name to write out to nifti file
        **kwargs: Additional keyword arguments to pass to the prediction algorithm

    """
    def __init__(self, data=None, Y = None, matrix_type=None, **kwargs):
        
        if matrix_type is not None:
            if matrix_type.lower() not in ['distance','similarity','directed','distance_flat','similarity_flat','directed_flat']:
                raise ValueError("matrix_type must be [None,'distance','similarity','weighted','distance_flat','similarity_flat','directed_flat']")
        
        if isinstance(data,list):
            d_all = []; symmetric_all = []; matrix_type_all = []
            for d in data:
                data_tmp, issymmetric_tmp, matrix_type_tmp, is_single_matrix = self._import_single_data(d,matrix_type=matrix_type)
                d_all.append(data_tmp)
                symmetric_all.append(issymmetric_tmp)
                matrix_type_all.append(matrix_type_tmp) 
            if not _all_same(symmetric_all):
                raise ValueError('Not all matrices are of the same symmetric type.')
            if not _all_same(matrix_type_all):
                raise ValueError('Not all matrices are of the same matrix type.')
            self.data = np.array(d_all)
            self.issymmetric = symmetric_all[0]
            self.matrix_type = matrix_type_all[0]
            self.is_single_matrix = False
        else:
            self.data,self.issymmetric,self.matrix_type,self.is_single_matrix = self._import_single_data(data, matrix_type=matrix_type)

        if Y is not None:
            if type(Y) is str:
                if os.path.isfile(Y):
                    Y=pd.read_csv(Y,header=None,index_col=None)
            if isinstance(Y, pd.DataFrame):
                if self.data.shape[0]!= len(Y):
                    raise ValueError("Y does not match the correct size of data")
                self.Y = Y
            else:
                raise ValueError("Make sure Y is a pandas data frame.")
        else:
            self.Y = pd.DataFrame()

    def __repr__(self):
        return '%s.%s(shape=%s, square_shape=%s, Y=%s, is_symmetric=%s, matrix_type=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.shape(),
            self.square_shape(),
            len(self.Y),
            self.issymmetric,
            self.matrix_type
            )

    def __getitem__(self,index):
        new = self.copy()
        if isinstance(index, int):
            new.data = np.array(self.data[index,:]).flatten()
            new.is_single_matrix = True
        else:
            new.data = np.array(self.data[index,:])           
        if not self.Y.empty:
            new.Y = self.Y.iloc[index] 
        return new
    
    def __len__(self):
        if self.is_single_matrix:
            return 1
        else:
            return self.data.shape[0]

    def __iter__(self):
        for x in range(len(self)):
            yield self[x]  

    def _import_single_data(self, data, matrix_type=None):
        ''' Helper function to import single data matrix.'''
        
        if isinstance(data,str):
            if os.path.isfile(data):
                data = pd.read_csv(data)
            else:
                raise ValueError('Make sure you have specified a valid file path.')

        def test_is_single_matrix(data):
            if len(data.shape)==1:
                return True
            else:
                return False
            
        if matrix_type is not None:
            if matrix_type.lower() == 'distance_flat':
                matrix_type = 'distance'
                data = np.array(data)
                issymmetric = True
                is_single_matrix = test_is_single_matrix(data)
            elif matrix_type.lower() == 'similarity_flat':
                matrix_type = 'similarity'
                data = np.array(data)
                issymmetric = True
                is_single_matrix = test_is_single_matrix(data)
            elif matrix_type.lower() == 'directed_flat':
                matrix_type = 'directed'
                data = np.array(data)
                issymmetric = False
                is_single_matrix = test_is_single_matrix(data)
            elif matrix_type.lower() in ['distance','similarity','directed']:
                if data.shape[0]!=data.shape[1]:
                    raise ValueError('Data matrix must be square')
                data = np.array(data)
                matrix_type = matrix_type.lower()
                if matrix_type in ['distance','similarity']:
                    issymmetric = True
                    data = data[np.triu_indices(data.shape[0],k=1)]
                else:
                    issymmetric = False
                    data = data.flatten()
                is_single_matrix = True
        else:
            if len(data.shape)==1: # Single Vector
                try:
                    data = squareform(data)
                except:
                    raise ValueError('Data is not flattened upper triangle from similarity/distance matrix or flattened directed matrix.')
                is_single_matrix = True
            elif data.shape[0]==data.shape[1]: # Square Matrix
                is_single_matrix = True
            else: # Rectangular Matrix
                data_all = deepcopy(data)
                try:
                    data = squareform(data_all[0,:])
                except:
                    raise ValueError('Data is not flattened upper triangle from multiple similarity/distance matrices or flattened directed matrices.')
                is_single_matrix = False
            
            # Test if matrix is symmetrical
            if np.all(data[np.triu_indices(data.shape[0],k=1)]==data.T[np.triu_indices(data.shape[0],k=1)]):
                issymmetric = True
            else:
                issymmetric = False 
                
            # Determine matrix type
            if issymmetric:
                if np.sum(np.diag(data)) == 0:
                    matrix_type = 'distance'
                elif np.sum(np.diag(data)) == data.shape[0]:
                    matrix_type = 'similarity'
                data = data[np.triu_indices(data.shape[0],k=1)]
            else:
                matrix_type = 'directed'
                data = data.flatten()
            
            if not is_single_matrix:
                data = data_all
                
        return (data, issymmetric, matrix_type, is_single_matrix)

    def squareform(self):
        '''Convert adjacency back to squareform'''
        if self.is_single_matrix:
            return squareform(self.data)  
        else:
            return [squareform(x.data) for x in self]

    def plot(self, limit=3, **kwargs):
        ''' Create Heatmap of Adjacency Matrix'''
        if self.is_single_matrix:
            return sns.heatmap(self.squareform(),square=True,**kwargs)
        else:
            f,a = plt.subplots(ncols=limit,figsize=(12,5))
            for i in range(limit):
                sns.heatmap(self[i].squareform(),square=True,ax=a[i],**kwargs)
            return f

    def mean(self, axis=0):
        ''' Calculate mean of upper triangle

        Args:
            axis:  calculate mean over features (0) or data (1)

        '''
        if self.is_single_matrix:
            return np.mean(self.data)
        else:
            return np.mean(self.data,axis=axis)
    
    def std(self, axis=0):
        ''' Calculate standard deviation of upper triangle

        Args:
            axis:  calculate standard deviation over features (0) or data (1)

        '''
        if self.is_single_matrix:
            return np.mean(self.data)
        else:
            return np.mean(self.data,axis=axis)
    
    def shape(self):
        ''' Calculate shape of data. '''
        return self.data.shape

    def square_shape(self):
        ''' Calculate shape of squareform data. '''
        if self.is_single_matrix:
            return self.squareform().shape  
        else:
            return self[0].squareform().shape

    def copy(self):
        ''' Create a copy of Adjacency object.'''
        return deepcopy(self)
    
    def append(self, data):
        ''' Append data to Adjacency instance

        Args:
            data:  Adjacency instance to append

        Returns:
            out: new appended Adjacency instance

        '''

        if not isinstance(data, Adjacency):
            raise ValueError('Make sure data is a Adjacency instance.')

        if self.isempty():
            out = data.copy() 
        else:
            out = self.copy()
        
        if self.square_shape() != data.square_shape():
            raise ValueError('Data is not the same shape as Adjacency instance.')

        out.data = np.vstack([self.data,data.data])
        if out.Y.size:
            out.Y = self.Y.append(data.Y)
        
        return out

    def write(self, file_name, method='long'):
        ''' Write out Adjacency object to csv file. 
        
            Args:
                file_name (str):  name of file name to write
                method (str):     method to write out data ['long','square']
        
        '''
        if method not in ['long','square']:
            raise ValueError('Make sure method is ["long","square"].')
        if self.is_single_matrix:
            if method is 'long':
                out = pd.DataFrame(self.data).to_csv(file_name,index=None)
            elif method is 'square':
                out = pd.DataFrame(self.squareform()).to_csv(file_name,index=None)
        else:
            if method is 'long':
                out = pd.DataFrame(self.data).to_csv(file_name,index=None)
            elif method is 'square':
                raise NotImplementedError('Need to decide how we should write out multiple matrices.  As separate files?')

    def similarity(self, data, **kwargs):
        ''' Calculate similarity between two Adjacency matrices.  
        Default is to use spearman correlation and permutation test.'''
        if not isinstance(data,Adjacency):
            data2 = Adjacency(data)
        else:
            data2 = data.copy()
        if self.is_single_matrix:
            return correlation_permutation(self.data, data2.data,**kwargs)
        else:
            return [correlation_permutation(x.data, data2.data,**kwargs) for x in self]

    def distance(self, method='correlation', **kwargs):
        """ Calculate distance between images within an Adjacency() instance.

            Args:
                method: type of distance metric (can use any scikit learn or sciypy metric)

            Returns:
                dist: Outputs a 2D distance matrix.

        """

        return pairwise_distances(self.data, metric = method, **kwargs)

    def ttest(self, **kwargs):
        ''' Calculate ttest across samples. '''
        if self.is_single_matrix:
            raise ValueError('t-test cannot be run on single matrices.')
        m = []; p = []
        for i in range(self.data.shape[1]):
            stats = one_sample_permutation(self.data[:,i],**kwargs)
            m.append(stats['mean'])
            p.append(stats['p'])
        mn = Adjacency(np.array(m))
        pval = Adjacency(np.array(p))
        return (mn,pval)

    def plot_label_distance(self, labels, ax=None):
        ''' Create a violin plot indicating within and between label distance

            Args:
                labels (np.array):  numpy array of labels to plot
        
            Returns:
                violin plot handles
        
        '''

        if not self.is_single_matrix:
            raise ValueError('This function only works on single adjacency matrices.')

        distance = pd.DataFrame(self.squareform())

        if len(labels) != distance.shape[0]:
            raise ValueError('Labels must be same length as distance matrix')

        within = []; between = []
        out = pd.DataFrame(columns=['Distance','Group','Type'],index=None)
        for i in np.unique(labels):
            tmp_w = pd.DataFrame(columns=out.columns,index=None)
            tmp_w['Distance'] = distance.loc[labels==i,labels==i].values[np.triu_indices(sum(labels==i),k=1)]
            tmp_w['Type'] = 'Within'
            tmp_w['Group'] = i
            tmp_b = pd.DataFrame(columns=out.columns,index=None)
            tmp_b['Distance'] = distance.loc[labels!=i,labels!=i].values[np.triu_indices(sum(labels==i),k=1)]
            tmp_b['Type'] = 'Between'
            tmp_b['Group'] = i
            out = out.append(tmp_w).append(tmp_b)
        f = sns.violinplot(x="Group", y="Distance", hue="Type", data=out, split=True, inner='quartile',
              palette={"Within": "lightskyblue", "Between": "red"},ax=ax)
        f.set_ylabel('Average Distance')
        f.set_title('Average Group Distance')
        return f
        
    def stats_label_distance(self, labels, n_permute=5000):
        ''' Calculate permutation tests on within and between label distance.

            Args:
                labels (np.array):  numpy array of labels to plot
                n_permute (int): number of permutations to run (default=5000)

            Returns:
                dict:  dictionary of within and between group differences and p-values
        
        '''

        if not self.is_single_matrix:
            raise ValueError('This function only works on single adjacency matrices.')

        distance = pd.DataFrame(self.squareform())

        if len(labels) != distance.shape[0]:
            raise ValueError('Labels must be same length as distance matrix')

        within = []; between = []
        out = pd.DataFrame(columns=['Distance','Group','Type'],index=None)
        for i in np.unique(labels):
            tmp_w = pd.DataFrame(columns=out.columns,index=None)
            tmp_w['Distance'] = distance.loc[labels==i,labels==i].values[np.triu_indices(sum(labels==i),k=1)]
            tmp_w['Type'] = 'Within'
            tmp_w['Group'] = i
            tmp_b = pd.DataFrame(columns=out.columns,index=None)
            tmp_b['Distance'] = distance.loc[labels==i,labels!=i].values.flatten()
            tmp_b['Type'] = 'Between'
            tmp_b['Group'] = i
            out = out.append(tmp_w).append(tmp_b)
        stats = dict()
        for i in np.unique(labels):
            # Within group test
            tmp1 = out.loc[(out['Group']==i) & (out['Type']=='Within'),'Distance']
            tmp2 = out.loc[(out['Group']==i) & (out['Type']=='Between'),'Distance']
            stats[str(i)] = two_sample_permutation(tmp1,tmp2,n_permute=n_permute)
        return stats
