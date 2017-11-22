from __future__ import division

'''
This data class is for working with similarity/dissimilarity matrices
'''

__author__ = ["Luke Chang"]
__license__ = "MIT"

import os
import pandas as pd
import numpy as np
import six
from copy import deepcopy
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
import seaborn as sns
import matplotlib.pyplot as plt
from nltools.stats import (correlation_permutation,
                           one_sample_permutation,
                           two_sample_permutation,
                           summarize_bootstrap)
from nltools.plotting import (plot_stacked_adjacency,
                              plot_silhouette)
from nltools.utils import (all_same,
                           attempt_to_import,
                           concatenate,
                           _bootstrap_apply_func)
from joblib import Parallel, delayed

# Optional dependencies
nx = attempt_to_import('networkx', 'nx')

class Adjacency(object):

    '''
    Adjacency is a class to represent Adjacency matrices as a vector rather
    than a 2-dimensional matrix. This makes it easier to perform data
    manipulation and analyses.

    Args:
        data: pandas data instance or list of files
        matrix_type: (str) type of matrix.  Possible values include:
                    ['distance','similarity','directed','distance_flat',
                    'similarity_flat','directed_flat']
        Y: Pandas DataFrame of training labels
        **kwargs: Additional keyword arguments

    '''

    def __init__(self, data=None, Y=None, matrix_type=None, **kwargs):
        if matrix_type is not None:
            if matrix_type.lower() not in ['distance','similarity','directed',
                                            'distance_flat','similarity_flat',
                                            'directed_flat']:
                raise ValueError("matrix_type must be [None,'distance', "
                                "'similarity','directed','distance_flat', "
                                "'similarity_flat','directed_flat']")

        if data is None:
            self.data = np.array([])
            self.matrix_type = 'empty'
            self.is_single_matrix = np.nan
            self.issymmetric = np.nan
        elif isinstance(data, list):
            if isinstance(data[0], Adjacency):
                tmp = concatenate(data)
                for item in ['data', 'matrix_type', 'Y','issymmetric']:
                    setattr(self, item, getattr(tmp,item))
            else:
                d_all = []; symmetric_all = []; matrix_type_all = []
                for d in data:
                    data_tmp, issymmetric_tmp, matrix_type_tmp, _ = self._import_single_data(d, matrix_type=matrix_type)
                    d_all.append(data_tmp)
                    symmetric_all.append(issymmetric_tmp)
                    matrix_type_all.append(matrix_type_tmp)
                if not all_same(symmetric_all):
                    raise ValueError('Not all matrices are of the same '
                                    'symmetric type.')
                if not all_same(matrix_type_all):
                    raise ValueError('Not all matrices are of the same matrix '
                                    'type.')
                self.data = np.array(d_all)
                self.issymmetric = symmetric_all[0]
                self.matrix_type = matrix_type_all[0]
            self.is_single_matrix = False
        else:
            self.data, self.issymmetric, self.matrix_type, self.is_single_matrix = self._import_single_data(data, matrix_type=matrix_type)

        if Y is not None:
            if isinstance(Y, six.string_types):
                if os.path.isfile(Y):
                    Y = pd.read_csv(Y, header=None, index_col=None)
            if isinstance(Y, pd.DataFrame):
                if self.data.shape[0] != len(Y):
                    raise ValueError("Y does not match the correct size of "
                                     "data")
                self.Y = Y
            else:
                raise ValueError("Make sure Y is a pandas data frame.")
        else:
            self.Y = pd.DataFrame()

    def __repr__(self):
        return ("%s.%s(shape=%s, square_shape=%s, Y=%s, is_symmetric=%s,"
                "matrix_type=%s)") % (
                    self.__class__.__module__,
                    self.__class__.__name__,
                    self.shape(),
                    self.square_shape(),
                    len(self.Y),
                    self.issymmetric,
                    self.matrix_type)

    def __getitem__(self,index):
        new = self.copy()
        if isinstance(index, int):
            new.data = np.array(self.data[index, :]).flatten()
            new.is_single_matrix = True
        else:
            new.data = np.array(self.data[index, :])
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

    def __add__(self, y):
        new = deepcopy(self)
        if isinstance(y, (int, float)):
            new.data = new.data + y
        if isinstance(y, Adjacency):
            if self.shape() != y.shape():
                raise ValueError('Both Adjacency() instances need to be the '
                                 'same shape.')
            new.data = new.data + y.data
        return new

    def __sub__(self, y):
        new = deepcopy(self)
        if isinstance(y, (int, float)):
            new.data = new.data - y
        if isinstance(y, Adjacency):
            if self.shape() != y.shape():
                raise ValueError('Both Adjacency() instances need to be the '
                                 'same shape.')
            new.data = new.data - y.data
        return new

    def __mul__(self, y):
        new = deepcopy(self)
        if isinstance(y, (int, float)):
            new.data = new.data * y
        if isinstance(y, Adjacency):
            if self.shape() != y.shape():
                raise ValueError('Both Adjacency() instances need to be the '
                                 'same shape.')
            new.data = np.multiply(new.data, y.data)
        return new

    def _import_single_data(self, data, matrix_type=None):
        ''' Helper function to import single data matrix.'''

        if isinstance(data, six.string_types):
            if os.path.isfile(data):
                data = pd.read_csv(data)
            else:
                raise ValueError('Make sure you have specified a valid file '
                                 'path.')

        def test_is_single_matrix(data):
            if len(data.shape) == 1:
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
                data = np.array(data).flatten()
                issymmetric = False
                is_single_matrix = test_is_single_matrix(data)
            elif matrix_type.lower() in ['distance', 'similarity', 'directed']:
                if data.shape[0] != data.shape[1]:
                    raise ValueError('Data matrix must be square')
                data = np.array(data)
                matrix_type = matrix_type.lower()
                if matrix_type in ['distance', 'similarity']:
                    issymmetric = True
                    data = data[np.triu_indices(data.shape[0], k=1)]
                else:
                    issymmetric = False
                    if isinstance(data, pd.DataFrame):
                        data = data.values.flatten()
                    elif isinstance(data, np.ndarray):
                        data = data.flatten()
                is_single_matrix = True
        else:
            if len(data.shape) == 1:  # Single Vector
                try:
                    data = squareform(data)
                except ValueError:
                    print('Data is not flattened upper triangle from '
                          'similarity/distance matrix or flattened directed '
                          'matrix.')
                is_single_matrix = True
            elif data.shape[0] == data.shape[1]:  # Square Matrix
                is_single_matrix = True
            else:  # Rectangular Matrix
                data_all = deepcopy(data)
                try:
                    data = squareform(data_all[0, :])
                except ValueError:
                    print('Data is not flattened upper triangle from multiple '
                          'similarity/distance matrices or flattened directed '
                          'matrices.')
                is_single_matrix = False

            # Test if matrix is symmetrical
            if np.all(data[np.triu_indices(data.shape[0], k=1)] == data.T[np.triu_indices(data.shape[0], k=1)]):
                issymmetric = True
            else:
                issymmetric = False

            # Determine matrix type
            if issymmetric:
                if np.sum(np.diag(data)) == 0:
                    matrix_type = 'distance'
                elif np.sum(np.diag(data)) == data.shape[0]:
                    matrix_type = 'similarity'
                data = data[np.triu_indices(data.shape[0], k=1)]
            else:
                matrix_type = 'directed'
                data = data.flatten()

            if not is_single_matrix:
                data = data_all

        return (data, issymmetric, matrix_type, is_single_matrix)

    def isempty(self):
        '''Check if Adjacency object is empty'''
        return bool(self.matrix_type is 'empty')

    def squareform(self):
        '''Convert adjacency back to squareform'''
        if self.issymmetric:
            if self.is_single_matrix:
                return squareform(self.data)
            else:
                return [squareform(x.data) for x in self]
        else:
            if self.is_single_matrix:
                return self.data.reshape(int(np.sqrt(self.data.shape[0])),
                                         int(np.sqrt(self.data.shape[0])))
            else:
                return [x.data.reshape(int(np.sqrt(x.data.shape[0])),
                            int(np.sqrt(x.data.shape[0]))) for x in self]

    def plot(self, limit=3, **kwargs):
        ''' Create Heatmap of Adjacency Matrix'''
        if self.is_single_matrix:
            return sns.heatmap(self.squareform(), square=True, **kwargs)
        else:
            f, a = plt.subplots(limit)
            for i in range(limit):
                sns.heatmap(self[i].squareform(), square=True, ax=a[i],
                            **kwargs)
            return f

    def mean(self, axis=0):
        ''' Calculate mean of Adjacency

        Args:
            axis:  calculate mean over features (0) or data (1).
                    For data it will be on upper triangle.

        Returns:
            mean:  float if single, adjacency if axis=0, np.array if axis=1
                    and multiple

        '''

        if self.is_single_matrix:
            return np.mean(self.data)
        else:
            if axis == 0:
                return Adjacency(data=np.mean(self.data, axis=axis),
                                 matrix_type=self.matrix_type + '_flat')
            elif axis == 1:
                return np.mean(self.data, axis=axis)

    def std(self, axis=0):
        ''' Calculate standard deviation of Adjacency

        Args:
            axis:  calculate std over features (0) or data (1).
                    For data it will be on upper triangle.

        Returns:
            std:  float if single, adjacency if axis=0, np.array if axis=1 and
                    multiple

        '''

        if self.is_single_matrix:
            return np.std(self.data)
        else:
            if axis == 0:
                return Adjacency(data=np.std(self.data, axis=axis),
                                 matrix_type=self.matrix_type + '_flat')
            elif axis == 1:
                return np.std(self.data, axis=axis)

    def shape(self):
        ''' Calculate shape of data. '''
        return self.data.shape

    def square_shape(self):
        ''' Calculate shape of squareform data. '''
        if self.matrix_type is 'empty':
            return np.array([])
        else:
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
                raise ValueError('Data is not the same shape as Adjacency '
                                 'instance.')

            out.data = np.vstack([self.data, data.data])
            out.is_single_matrix = False
            if out.Y.size:
                out.Y = self.Y.append(data.Y)

        return out

    def write(self, file_name, method='long'):
        ''' Write out Adjacency object to csv file.

            Args:
                file_name (str):  name of file name to write
                method (str):     method to write out data ['long','square']

        '''
        if method not in ['long', 'square']:
            raise ValueError('Make sure method is ["long","square"].')
        if self.is_single_matrix:
            if method is 'long':
                out = pd.DataFrame(self.data).to_csv(file_name, index=None)
            elif method is 'square':
                out = pd.DataFrame(self.squareform()).to_csv(file_name,
                                                             index=None)
        else:
            if method is 'long':
                out = pd.DataFrame(self.data).to_csv(file_name, index=None)
            elif method is 'square':
                raise NotImplementedError('Need to decide how we should write '
                                          'out multiple matrices.  As separate '
                                          'files?')

    def similarity(self, data, plot=False, **kwargs):
        ''' Calculate similarity between two Adjacency matrices.
        Default is to use spearman correlation and permutation test.'''
        if not isinstance(data, Adjacency):
            data2 = Adjacency(data)
        else:
            data2 = data.copy()
        if self.is_single_matrix:
            if plot:
                plot_stacked_adjacency(self, data)
            return correlation_permutation(self.data, data2.data, **kwargs)
        else:
            if plot:
                _, a = plt.subplots(len(self))
                for i in a:
                    plot_stacked_adjacency(self, data, ax=i)
            return [correlation_permutation(x.data, data2.data, **kwargs) for x in self]

    def distance(self, method='correlation', **kwargs):
        ''' Calculate distance between images within an Adjacency() instance.

        Args:
            method: type of distance metric (can use any scikit learn or
                    sciypy metric)

        Returns:
            dist: Outputs a 2D distance matrix.

        '''
        return Adjacency(pairwise_distances(self.data, metric=method, **kwargs),
                         matrix_type='distance')

    def threshold(self, thresh=0, binarize=False):
        '''Threshold Adjacency instance

        Args:
            thresh: cutoff to threshold image (float).  if 'threshold'=50%,
                    will calculate percentile.
            binarize (bool): if 'binarize'=True then binarize output
        Returns:
            Brain_Data: thresholded Brain_Data instance

        '''

        b = self.copy()
        if isinstance(thresh, six.string_types):
            if thresh[-1] is '%':
                thresh = np.percentile(b.data, float(thresh[:-1]))
        if binarize:
            b.data = b.data > thresh
        else:
            b.data[b.data < thresh] = 0
        return b

    def to_graph(self):
        ''' Convert Adjacency into networkx graph.  only works on
            single_matrix for now.'''

        if self.is_single_matrix:
            if self.matrix_type == 'directed':
                return nx.DiGraph(self.squareform())
            else:
                return nx.Graph(self.squareform())
        else:
            raise NotImplementedError('This function currently only works on '
                                      'single matrices.')

    def ttest(self, **kwargs):
        ''' Calculate ttest across samples. '''
        if self.is_single_matrix:
            raise ValueError('t-test cannot be run on single matrices.')
        m = []; p = []
        for i in range(self.data.shape[1]):
            stats = one_sample_permutation(self.data[:, i], **kwargs)
            m.append(stats['mean'])
            p.append(stats['p'])
        mn = Adjacency(np.array(m))
        pval = Adjacency(np.array(p))
        return (mn, pval)

    def plot_label_distance(self, labels, ax=None):
        ''' Create a violin plot indicating within and between label distance

            Args:
                labels (np.array):  numpy array of labels to plot

            Returns:
                violin plot handles

        '''

        if not self.is_single_matrix:
            raise ValueError('This function only works on single adjacency '
                             'matrices.')

        distance = pd.DataFrame(self.squareform())

        if len(labels) != distance.shape[0]:
            raise ValueError('Labels must be same length as distance matrix')

        within = []; between = []
        out = pd.DataFrame(columns=['Distance', 'Group', 'Type'], index=None)
        for i in np.unique(labels):
            tmp_w = pd.DataFrame(columns=out.columns, index=None)
            tmp_w['Distance'] = distance.loc[labels == i, labels == i].values[np.triu_indices(sum(labels == i), k=1)]
            tmp_w['Type'] = 'Within'
            tmp_w['Group'] = i
            tmp_b = pd.DataFrame(columns=out.columns, index=None)
            tmp_b['Distance'] = distance.loc[labels != i, labels != i].values[np.triu_indices(sum(labels == i), k=1)]
            tmp_b['Type'] = 'Between'
            tmp_b['Group'] = i
            out = out.append(tmp_w).append(tmp_b)
        f = sns.violinplot(x="Group", y="Distance", hue="Type", data=out, split=True, inner='quartile',
              palette={"Within": "lightskyblue", "Between": "red"}, ax=ax)
        f.set_ylabel('Average Distance')
        f.set_title('Average Group Distance')
        return f

    def stats_label_distance(self, labels, n_permute=5000, n_jobs=-1):
        ''' Calculate permutation tests on within and between label distance.

            Args:
                labels (np.array):  numpy array of labels to plot
                n_permute (int): number of permutations to run (default=5000)

            Returns:
                dict:  dictionary of within and between group differences
                        and p-values

        '''

        if not self.is_single_matrix:
            raise ValueError('This function only works on single adjacency '
                             'matrices.')

        distance = pd.DataFrame(self.squareform())

        if len(labels) != distance.shape[0]:
            raise ValueError('Labels must be same length as distance matrix')

        within = []; between = []
        out = pd.DataFrame(columns=['Distance', 'Group', 'Type'], index=None)
        for i in np.unique(labels):
            tmp_w = pd.DataFrame(columns=out.columns, index=None)
            tmp_w['Distance'] = distance.loc[labels == i, labels == i].values[np.triu_indices(sum(labels == i), k=1)]
            tmp_w['Type'] = 'Within'
            tmp_w['Group'] = i
            tmp_b = pd.DataFrame(columns=out.columns, index=None)
            tmp_b['Distance'] = distance.loc[labels == i, labels != i].values.flatten()
            tmp_b['Type'] = 'Between'
            tmp_b['Group'] = i
            out = out.append(tmp_w).append(tmp_b)
        stats = dict()
        for i in np.unique(labels):
            # Within group test
            tmp1 = out.loc[(out['Group'] == i) & (out['Type'] == 'Within'), 'Distance']
            tmp2 = out.loc[(out['Group'] == i) & (out['Type'] == 'Between'), 'Distance']
            stats[str(i)] = two_sample_permutation(tmp1, tmp2,
                                        n_permute=n_permute, n_jobs=n_jobs)
        return stats

    def plot_silhouette(self, labels, ax=None, permutation_test=True,
                        n_permute=5000, **kwargs):
        '''Create a silhouette plot'''
        distance = pd.DataFrame(self.squareform())

        if len(labels) != distance.shape[0]:
            raise ValueError('Labels must be same length as distance matrix')

        (f,outAll) = plot_silhouette(distance, labels, ax=None,
                                    permutation_test=True,
                                    n_permute=5000, **kwargs)
        return (f,outAll)

    def bootstrap(self, function, n_samples=5000, save_weights=False,
                    n_jobs=-1, *args, **kwargs):
        '''Bootstrap an Adjacency method.

            Example Useage:
            b = dat.bootstrap('mean', n_samples=5000)
            b = dat.bootstrap('predict', n_samples=5000, algorithm='ridge')
            b = dat.bootstrap('predict', n_samples=5000, save_weights=True)

        Args:
            function: (str) method to apply to data for each bootstrap
            n_samples: (int) number of samples to bootstrap with replacement
            save_weights: (bool) Save each bootstrap iteration
                        (useful for aggregating many bootstraps on a cluster)
            n_jobs: (int) The number of CPUs to use to do the computation.
                        -1 means all CPUs.Returns:
        output: summarized studentized bootstrap output

        '''

        bootstrapped = Parallel(n_jobs=n_jobs)(
                        delayed(_bootstrap_apply_func)(self,
                        function, *args, **kwargs) for i in range(n_samples))
        bootstrapped = Adjacency(bootstrapped)
        return summarize_bootstrap(bootstrapped, save_weights=save_weights)
