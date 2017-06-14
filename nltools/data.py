from __future__ import division

'''
NeuroLearn Data Classes
=======================

Classes to represent various types of data

'''

# Notes:
# Need to figure out how to speed up loading and resampling of data

__all__ = ['Brain_Data',
            'Adjacency',
            'Groupby',
            'Design_Matrix',
            'Design_Matrix_Series']
__author__ = ["Luke Chang"]
__license__ = "MIT"

import os
import cPickle
import nibabel as nib
from nltools.utils import get_resource_path, set_algorithm, get_anatomical, make_cosine_basis, glover_hrf
from nltools.cross_validation import set_cv
from nltools.plotting import (dist_from_hyperplane_plot,
                              scatterplot,
                              probability_plot,
                              roc_plot,
                              plot_stacked_adjacency,
                              plot_silhouette)
from nltools.stats import (pearson,
                           fdr,
                           threshold,
                           fisher_r_to_z,
                           correlation_permutation,
                           one_sample_permutation,
                           two_sample_permutation)
from nltools.mask import expand_mask, collapse_mask
from nltools.stats import downsample, zscore, upsample
from nltools.analysis import Roc
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_img
from nilearn.masking import intersect_masks
from nilearn.regions import connected_regions
from nilearn.plotting.img_plotting import plot_epi, plot_roi, plot_stat_map
from nilearn.signal import clean
from copy import deepcopy
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from scipy.stats import ttest_1samp, t, norm
from scipy.signal import detrend
from scipy.spatial.distance import squareform
import six
from sklearn.metrics.pairwise import pairwise_distances
from nltools.pbs_job import PBS_Job
import warnings
import shutil
import tempfile
import seaborn as sns
from pynv import Client
import matplotlib.pyplot as plt

# Optional dependencies
try:
    from mne.stats import (spatio_temporal_cluster_1samp_test,
                           ttest_1samp_no_p)
except ImportError:
    pass

try:
    import networkx as nx
except ImportError:
    pass


class Brain_Data(object):

    """
    Brain_Data is a class to represent neuroimaging data in python as a vector
    rather than a 3-dimensional matrix.This makes it easier to perform data
    manipulation and analyses.

    Args:
        data: nibabel data instance or list of files
        Y: Pandas DataFrame of training labels
        X: Pandas DataFrame Design Matrix for running univariate models
        mask: binary nifiti file to mask brain data
        output_file: Name to write out to nifti file
        **kwargs: Additional keyword arguments to pass to the prediction
                algorithm

    """

    def __init__(self, data=None, Y=None, X=None, mask=None, output_file=None,
                 **kwargs):
        if mask is not None:
            if not isinstance(mask, nib.Nifti1Image):
                if isinstance(mask, six.string_types):
                    if os.path.isfile(mask):
                        mask = nib.load(mask)
                else:
                    raise ValueError("mask is not a nibabel instance or a "
                                     "valid file name")
            self.mask = mask
        else:
            self.mask = nib.load(os.path.join(get_resource_path(),
                                'MNI152_T1_2mm_brain_mask.nii.gz'))
        self.nifti_masker = NiftiMasker(mask_img=self.mask)

        if data is not None:
            if isinstance(data, (str, unicode)):
                if 'http://' in data:
                    from nltools.datasets import download_nifti
                    tmp_dir = os.path.join(tempfile.gettempdir(),
                                            str(os.times()[-1]))
                    os.makedirs(tmp_dir)
                    data = nib.load(download_nifti(data, data_dir=tmp_dir))
                else:
                    data = nib.load(data)
                self.data = self.nifti_masker.fit_transform(data)
            elif isinstance(data, list):
                self.data = []
                for i in data:
                    if isinstance(i, six.string_types):
                        self.data.append(self.nifti_masker.fit_transform(
                                         nib.load(i)))
                    elif isinstance(i,nib.Nifti1Image):
                        self.data.append(self.nifti_masker.fit_transform(i))
                self.data = np.array(self.data)
            elif isinstance(data, nib.Nifti1Image):
                self.data = np.array(self.nifti_masker.fit_transform(data))
            else:
                raise ValueError("data is not a nibabel instance")

            # Collapse any extra dimension
            if any([x == 1 for x in self.data.shape]):
                self.data = self.data.squeeze()
        else:
            self.data = np.array([])

        if Y is not None:
            if isinstance(Y, six.string_types):
                if os.path.isfile(Y):
                    Y = pd.read_csv(Y, header=None, index_col=None)
            if isinstance(Y, pd.DataFrame):
                if self.data.shape[0] != len(Y):
                    raise ValueError("Y does not match the correct size "
                                     "of data")
                self.Y = Y
            else:
                raise ValueError("Make sure Y is a pandas data frame.")
        else:
            self.Y = pd.DataFrame()

        if X is not None:
            if isinstance(X, six.string_types):
                if os.path.isfile(X):
                    X = pd.read_csv(X, header=None, index_col=None)
            if isinstance(X, pd.DataFrame):
                if self.data.shape[0] != X.shape[0]:
                    raise ValueError("X does not match the correct size "
                                     "of data")
                self.X = X
            else:
                raise ValueError("Make sure X is a pandas data frame.")
        else:
            self.X = pd.DataFrame()

        if output_file is not None:
            self.file_name = output_file
        else:
            self.file_name = []

    def __repr__(self):
        return '%s.%s(data=%s, Y=%s, X=%s, mask=%s, output_file=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.shape(),
            len(self.Y),
            self.X.shape,
            os.path.basename(self.mask.get_filename()),
            self.file_name
            )

    def __getitem__(self, index):
        new = deepcopy(self)
        if isinstance(index, int):
            new.data = np.array(self.data[index, :]).flatten()
        else:
            new.data = np.array(self.data[index, :])
        if not self.Y.empty:
            new.Y = self.Y.iloc[index]
        if not self.X.empty:
            new.X = self.X.iloc[index]
        return new

    def __setitem__(self, index, value):
        if not isinstance(value, Brain_Data):
            raise ValueError("Make sure the value you are trying to set is a "
                             "Brain_Data() instance.")
        self.data[index, :] = value.data
        if not value.Y.empty:
            self.Y.values[index] = value.Y
        if not value.X.empty:
            if self.X.shape[1] != value.X.shape[1]:
                raise ValueError("Make sure self.X is the same size as "
                                 "value.X.")
            self.X.values[index] = value.X

    def __len__(self):
        return self.shape()[0]

    def __add__(self, y):
        new = deepcopy(self)
        if isinstance(y, (int, float)):
            new.data = new.data + y
        if isinstance(y, Brain_Data):
            if self.shape() != y.shape():
                raise ValueError("Both Brain_Data() instances need to be the "
                                 "same shape.")
            new.data = new.data + y.data
        return new

    def __sub__(self, y):
        new = deepcopy(self)
        if isinstance(y, (int, float)):
            new.data = new.data - y
        if isinstance(y, Brain_Data):
            if self.shape() != y.shape():
                raise ValueError('Both Brain_Data() instances need to be the '
                                 'same shape.')
            new.data = new.data - y.data
        return new

    def __mul__(self, y):
        new = deepcopy(self)
        if isinstance(y, (int, float)):
            new.data = new.data * y
        if isinstance(y, Brain_Data):
            if self.shape() != y.shape():
                raise ValueError("Both Brain_Data() instances need to be the "
                                 "same shape.")
            new.data = np.multiply(new.data, y.data)
        return new

    def __iter__(self):
        for x in range(len(self)):
            yield self[x]

    def shape(self):
        """ Get images by voxels shape. """

        return self.data.shape

    def mean(self):
        """ Get mean of each voxel across images. """

        out = deepcopy(self)
        if len(self.shape()) > 1:
            out.data = np.mean(self.data, axis=0)
        else:
            out = np.mean(self.data)
        return out

    def std(self):
        """ Get standard deviation of each voxel across images. """

        out = deepcopy(self)
        if len(self.shape()) > 1:
            out.data = np.std(self.data, axis=0)
        else:
            out = np.std(self.data)
        return out

    def sum(self):
        """ Sum over voxels."""

        out = deepcopy(self)
        if len(self.shape()) > 1:
            out.data = np.sum(out.data, axis=0)
        else:
            out = np.sum(self.data)
        return out

    def to_nifti(self):
        """ Convert Brain_Data Instance into Nifti Object """

        return self.nifti_masker.inverse_transform(self.data)

    def write(self, file_name=None):
        """ Write out Brain_Data object to Nifti File.

        Args:
            file_name: name of nifti file

        """

        self.to_nifti().to_filename(file_name)

    def plot(self, limit=5, anatomical=None, **kwargs):
        """ Create a quick plot of self.data.  Will plot each image separately

        Args:
            limit: max number of images to return
            anatomical: nifti image or file name to overlay

        """

        if anatomical is not None:
            if not isinstance(anatomical, nib.Nifti1Image):
                if isinstance(anatomical, str):
                    anatomical = nib.load(anatomical)
                else:
                    raise ValueError("anatomical is not a nibabel instance")
        else:
            anatomical = get_anatomical()

        if self.data.ndim == 1:
            plot_stat_map(self.to_nifti(), anatomical,
                          cut_coords=range(-40, 50, 10), display_mode='z',
                          black_bg=True, colorbar=True, draw_cross=False,**kwargs)
        else:
            for i in xrange(self.data.shape[0]):
                if i < limit:
                     plot_stat_map(self[i].to_nifti(), anatomical,
                                   cut_coords=range(-40, 50, 10),
                                   display_mode='z',
                                   black_bg=True,
                                   colorbar=True,
                                   draw_cross=False,
                                   **kwargs)

    def regress(self):
        """ run vectorized OLS regression across voxels.

        Returns:
            out: dictionary of regression statistics in Brain_Data instances
                {'beta','t','p','df','residual'}

        """

        if not isinstance(self.X, pd.DataFrame):
            raise ValueError('Make sure self.X is a pandas DataFrame.')

        if self.X.empty:
            raise ValueError('Make sure self.X is not empty.')

        if self.data.shape[0] != self.X.shape[0]:
            raise ValueError("self.X does not match the correct size of "
                             "self.data")

        b = np.dot(np.linalg.pinv(self.X), self.data)
        res = self.data - np.dot(self.X, b)
        sigma = np.std(res, axis=0, ddof=self.X.shape[1])
        stderr = np.dot(np.matrix(np.diagonal(np.linalg.inv(np.dot(self.X.T,
                        self.X)))**.5).T, np.matrix(sigma))
        b_out = deepcopy(self)
        b_out.data = b
        t_out = deepcopy(self)
        t_out.data = b / stderr
        df = np.array([self.X.shape[0]-self.X.shape[1]] * t_out.data.shape[1])
        p_out = deepcopy(self)
        p_out.data = 2*(1-t.cdf(np.abs(t_out.data), df))

        # Might want to not output this info
        df_out = deepcopy(self)
        df_out.data = df
        sigma_out = deepcopy(self)
        sigma_out.data = sigma
        res_out = deepcopy(self)
        res_out.data = res

        return {'beta': b_out, 't': t_out, 'p': p_out, 'df': df_out,
                'sigma': sigma_out, 'residual': res_out}

    def ttest(self, threshold_dict=None):
        """ Calculate one sample t-test across each voxel (two-sided)

        Args:
            threshold_dict: a dictionary of threshold parameters {'unc':.001}
                            or {'fdr':.05} or {'permutation':tcfe,
                            n_permutation:5000}

        Returns:
            out: dictionary of regression statistics in Brain_Data instances
                {'t','p'}

        """

        t = deepcopy(self)
        p = deepcopy(self)

        if threshold_dict is not None:
            if 'permutation' in threshold_dict:
                # Convert data to correct shape (subjects, time, space)
                data_convert_shape = deepcopy(self.data)
                data_convert_shape = np.expand_dims(data_convert_shape, axis=1)
                if 'n_permutations' in threshold_dict:
                    n_permutations = threshold_dict['n_permutations']
                else:
                    n_permutations = 1000
                    warnings.warn("n_permutations not set:  running with 1000 "
                                  "permutations")

                if 'connectivity' in threshold_dict:
                    connectivity = threshold_dict['connectivity']
                else:
                    connectivity = None

                if 'n_jobs' in threshold_dict:
                    n_jobs = threshold_dict['n_jobs']
                else:
                    n_jobs = 1

                if threshold_dict['permutation'] is 'tfce':
                    perm_threshold = dict(start=0, step=0.2)
                else:
                    perm_threshold = None

                if 'stat_fun' in threshold_dict:
                    stat_fun = threshold_dict['stat_fun']
                else:
                    stat_fun = ttest_1samp_no_p

                t.data, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
                    data_convert_shape, tail=0, threshold=perm_threshold, stat_fun=stat_fun,
                    connectivity=connectivity, n_permutations=n_permutations, n_jobs=n_jobs)

                t.data = t.data.squeeze()

                p = deepcopy(t)
                for cl, pval in zip(clusters, p_values):
                    p.data[cl[1][0]] = pval
            else:
                t.data, p.data = ttest_1samp(self.data, 0, 0)
        else:
            t.data, p.data = ttest_1samp(self.data, 0, 0)

        if threshold_dict is not None:
            if isinstance(threshold_dict, dict):
                if 'unc' in threshold_dict:
                    thr = threshold_dict['unc']
                elif 'fdr' in threshold_dict:
                    thr = fdr(p.data, q=threshold_dict['fdr'])
                elif 'permutation' in threshold_dict:
                    thr = .05
                thr_t = threshold(t, p, thr)
                out = {'t': t, 'p': p, 'thr_t': thr_t}
            else:
                raise ValueError("threshold_dict is not a dictionary. "
                                 "Make sure it is in the form of {'unc': .001} "
                                 "or {'fdr': .05}")
        else:
            out = {'t': t, 'p': p}

        return out

    def append(self, data):
        """ Append data to Brain_Data instance

        Args:
            data: Brain_Data instance to append

        Returns:
            out: new appended Brain_Data instance
        """

        if not isinstance(data, Brain_Data):
            raise ValueError('Make sure data is a Brain_Data instance')

        if self.isempty():
            out = deepcopy(data)
        else:
            out = deepcopy(self)
            if len(self.shape()) == 1 & len(data.shape()) == 1:
                if self.shape()[0] != data.shape()[0]:
                    raise ValueError("Data is a different number of voxels "
                                     "then the weight_map.")
            elif len(self.shape()) == 1 & len(data.shape()) > 1:
                if self.shape()[0] != data.shape()[1]:
                    raise ValueError("Data is a different number of voxels "
                                     "then the weight_map.")
            elif len(self.shape()) > 1 & len(data.shape()) == 1:
                if self.shape()[1] != data.shape()[0]:
                    raise ValueError("Data is a different number of voxels "
                                     "then the weight_map.")
            elif self.shape()[1] != data.shape()[1]:
                raise ValueError("Data is a different number of voxels then "
                                 "the weight_map.")

            out.data = np.vstack([self.data, data.data])
            if out.Y.size:
                out.Y = self.Y.append(data.Y)
            if self.X.size:
                if isinstance(self.X, pd.DataFrame):
                    out.X = self.X.append(data.X)
                else:
                    out.X = np.vstack([self.X, data.X])
        return out

    def empty(self, data=True, Y=True, X=True):
        """ Initalize Brain_Data.data as empty """

        tmp = deepcopy(self)
        if data:
            tmp.data = np.array([])
        if Y:
            tmp.Y = pd.DataFrame()
        if X:
            tmp.X = pd.DataFrame()
        return tmp

    def isempty(self):
        """ Check if Brain_Data.data is empty """

        if isinstance(self.data, np.ndarray):
            if self.data.size:
                boolean = False
            else:
                boolean = True

        if isinstance(self.data, list):
            if not self.data:
                boolean = True
            else:
                boolean = False

        return boolean

    def similarity(self, image, method='correlation'):
        """ Calculate similarity of Brain_Data() instance with single
            Brain_Data or Nibabel image

            Args:
                image: Brain_Data or Nibabel instance of weight map

            Returns:
                pexp: Outputs a vector of pattern expression values

        """

        if not isinstance(image, Brain_Data):
            if isinstance(image, nib.Nifti1Image):
                image = Brain_Data(image)
            else:
                raise ValueError("Image is not a Brain_Data or nibabel "
                                 "instance")
        dim = image.shape()

        # Check to make sure masks are the same for each dataset and if not create a union mask
        # This might be handy code for a new Brain_Data method
        if np.sum(self.nifti_masker.mask_img.get_data() == 1) != np.sum(image.nifti_masker.mask_img.get_data()==1):
            new_mask = intersect_masks([self.nifti_masker.mask_img,
                                        image.nifti_masker.mask_img],
                                        threshold=1, connected=False)
            new_nifti_masker = NiftiMasker(mask_img=new_mask)
            data2 = new_nifti_masker.fit_transform(self.to_nifti())
            image2 = new_nifti_masker.fit_transform(image.to_nifti())
        else:
            data2 = self.data
            image2 = image.data

        # Calculate pattern expression
        if method is 'dot_product':
            if len(image2.shape) > 1:
                if image2.shape[0] > 1:
                    pexp = []
                    for i in range(image2.shape[0]):
                        pexp.append(np.dot(data2, image2[i, :]))
                    pexp = np.array(pexp)
                else:
                    pexp = np.dot(data2, image2)
            else:
                pexp = np.dot(data2, image2)
        elif method is 'correlation':
            if len(image2.shape) > 1:
                if image2.shape[0] > 1:
                    pexp = []
                    for i in range(image2.shape[0]):
                        pexp.append(pearson(image2[i, :], data2))
                    pexp = np.array(pexp)
                else:
                    pexp = pearson(image2, data2)
            else:
                pexp = pearson(image2, data2)
        else:
            raise ValueError("Method must be one of: correlation, dot_product")
        return pexp

    def distance(self, method='euclidean', **kwargs):
        """ Calculate distance between images within a Brain_Data() instance.

            Args:
                method: type of distance metric (can use any scikit learn or
                        sciypy metric)

            Returns:
                dist: Outputs a 2D distance matrix.

        """

        return Adjacency(pairwise_distances(self.data, metric=method, **kwargs),
                         matrix_type='Distance')

    def multivariate_similarity(self, images, method='ols'):
        """ Predict spatial distribution of Brain_Data() instance from linear
            combination of other Brain_Data() instances or Nibabel images

            Args:
                self: Brain_Data instance of data to be applied
                images: Brain_Data instance of weight map

            Returns:
                out: dictionary of regression statistics in Brain_Data
                    instances {'beta','t','p','df','residual'}

        """
        # Notes:  Should add ridge, and lasso, elastic net options options

        if len(self.shape()) > 1:
            raise ValueError("This method can only decompose a single brain "
                             "image.")

        if not isinstance(images, Brain_Data):
            raise ValueError("Images are not a Brain_Data instance")

        # Check to make sure masks are the same for each dataset and if not create a union mask
        # This might be handy code for a new Brain_Data method
        if np.sum(self.nifti_masker.mask_img.get_data() == 1) != np.sum(images.nifti_masker.mask_img.get_data()==1):
            new_mask = intersect_masks([self.nifti_masker.mask_img,
                                        images.nifti_masker.mask_img],
                                        threshold=1, connected=False)
            new_nifti_masker = NiftiMasker(mask_img=new_mask)
            data2 = new_nifti_masker.fit_transform(self.to_nifti())
            image2 = new_nifti_masker.fit_transform(images.to_nifti())
        else:
            data2 = self.data
            image2 = images.data

        # Add intercept and transpose
        image2 = np.vstack((np.ones(image2.shape[1]), image2)).T

        # Calculate pattern expression
        if method is 'ols':
            b = np.dot(np.linalg.pinv(image2), data2)
            res = data2 - np.dot(image2, b)
            sigma = np.std(res, axis=0)
            stderr = np.dot(np.matrix(np.diagonal(np.linalg.inv(np.dot(image2.T,
                            image2)))**.5).T, np.matrix(sigma))
            t_out = b / stderr
            df = image2.shape[0]-image2.shape[1]
            p = 2*(1-t.cdf(np.abs(t_out), df))
        else:
            raise NotImplementedError

        return {'beta': b, 't': t_out, 'p': p, 'df': df, 'sigma': sigma,
                'residual': res}

    def predict(self, algorithm=None, cv_dict=None, plot=True, **kwargs):

        """ Run prediction

        Args:
            algorithm: Algorithm to use for prediction.  Must be one of 'svm',
                    'svr', 'linear', 'logistic', 'lasso', 'ridge',
                    'ridgeClassifier','randomforest', or
                    'randomforestClassifier'
            cv_dict: Type of cross_validation to use. A dictionary of
                    {'type': 'kfolds', 'n_folds': n},
                    {'type': 'kfolds', 'n_folds': n, 'stratified': Y},
                    {'type': 'kfolds', 'n_folds': n, 'subject_id': holdout}, or
                    {'type': 'loso', 'subject_id': holdout}
                    where 'n' = number of folds, and 'holdout' = vector of
                    subject ids that corresponds to self.Y
            plot: Boolean indicating whether or not to create plots.
            **kwargs: Additional keyword arguments to pass to the prediction
                    algorithm

        Returns:
            output: a dictionary of prediction parameters

        """

        # Set algorithm
        if algorithm is not None:
            predictor_settings = set_algorithm(algorithm, **kwargs)
        else:
            # Use SVR as a default
            predictor_settings = set_algorithm('svr', **{'kernel': "linear"})

        # Initialize output dictionary
        output = {}
        output['Y'] = np.array(self.Y).flatten()

        # Overall Fit for weight map
        predictor = predictor_settings['predictor']
        predictor.fit(self.data, output['Y'])
        output['yfit_all'] = predictor.predict(self.data)
        if predictor_settings['prediction_type'] == 'classification':
            if predictor_settings['algorithm'] not in ['svm', 'ridgeClassifier',
                                  'ridgeClassifierCV']:
                output['prob_all'] = predictor.predict_proba(self.data)[:, 1]
            else:
                output['dist_from_hyperplane_all'] = predictor.decision_function(self.data)
                if predictor_settings['algorithm'] == 'svm' and predictor.probability:
                    output['prob_all'] = predictor.predict_proba(self.data)[:, 1]

        # Intercept
        if predictor_settings['algorithm'] == 'pcr':
            output['intercept'] = predictor_settings['_regress'].intercept_
        elif predictor_settings['algorithm'] == 'lassopcr':
            output['intercept'] = predictor_settings['_lasso'].intercept_
        else:
            output['intercept'] = predictor.intercept_

        # Weight map
        output['weight_map'] = self.empty()
        if predictor_settings['algorithm'] == 'lassopcr':
            output['weight_map'].data = np.dot(predictor_settings['_pca'].components_.T, predictor_settings['_lasso'].coef_)
        elif predictor_settings['algorithm'] == 'pcr':
            output['weight_map'].data = np.dot(predictor_settings['_pca'].components_.T, predictor_settings['_regress'].coef_)
        else:
            output['weight_map'].data = predictor.coef_.squeeze()

        # Cross-Validation Fit
        if cv_dict is not None:
            cv = set_cv(Y=self.Y, cv_dict=cv_dict)

            predictor_cv = predictor_settings['predictor']
            output['yfit_xval'] = output['yfit_all'].copy()
            output['intercept_xval'] = []
            output['weight_map_xval'] = output['weight_map'].copy()
            output['cv_idx'] = []
            wt_map_xval = []
            if predictor_settings['prediction_type'] == 'classification':
                if predictor_settings['algorithm'] not in ['svm', 'ridgeClassifier', 'ridgeClassifierCV']:
                    output['prob_xval'] = np.zeros(len(self.Y))
                else:
                    output['dist_from_hyperplane_xval'] = np.zeros(len(self.Y))
                    if predictor_settings['algorithm'] == 'svm' and predictor_cv.probability:
                        output['prob_xval'] = np.zeros(len(self.Y))

            for train, test in cv:
                predictor_cv.fit(self.data[train], self.Y.loc[train])
                output['yfit_xval'][test] = predictor_cv.predict(self.data[test]).ravel()
                if predictor_settings['prediction_type'] == 'classification':
                    if predictor_settings['algorithm'] not in ['svm', 'ridgeClassifier', 'ridgeClassifierCV']:
                        output['prob_xval'][test] = predictor_cv.predict_proba(self.data[test])[:, 1]
                    else:
                        output['dist_from_hyperplane_xval'][test] = predictor_cv.decision_function(self.data[test])
                        if predictor_settings['algorithm'] == 'svm' and predictor_cv.probability:
                            output['prob_xval'][test] = predictor_cv.predict_proba(self.data[test])[:, 1]
                # Intercept
                if predictor_settings['algorithm'] == 'pcr':
                    output['intercept_xval'].append(predictor_settings['_regress'].intercept_)
                elif predictor_settings['algorithm'] == 'lassopcr':
                    output['intercept_xval'].append(predictor_settings['_lasso'].intercept_)
                else:
                    output['intercept_xval'].append(predictor_cv.intercept_)
                output['cv_idx'].append((train,test))

                # Weight map
                if predictor_settings['algorithm'] == 'lassopcr':
                    wt_map_xval.append(np.dot(predictor_settings['_pca'].components_.T, predictor_settings['_lasso'].coef_))
                elif predictor_settings['algorithm'] == 'pcr':
                    wt_map_xval.append(np.dot(predictor_settings['_pca'].components_.T, predictor_settings['_regress'].coef_))
                else:
                    wt_map_xval.append(predictor_cv.coef_.squeeze())
                output['weight_map_xval'].data = np.array(wt_map_xval)

        # Print Results
        if predictor_settings['prediction_type'] == 'classification':
            output['mcr_all'] = np.mean(output['yfit_all'] == np.array(self.Y).flatten())
            print('overall accuracy: %.2f' % output['mcr_all'])
            if cv_dict is not None:
                output['mcr_xval'] = np.mean(output['yfit_xval'] == np.array(self.Y).flatten())
                print('overall CV accuracy: %.2f' % output['mcr_xval'])
        elif predictor_settings['prediction_type'] == 'prediction':
            output['rmse_all'] = np.sqrt(np.mean((output['yfit_all']-output['Y'])**2))
            output['r_all'] = np.corrcoef(output['Y'], output['yfit_all'])[0, 1]
            print('overall Root Mean Squared Error: %.2f' % output['rmse_all'])
            print('overall Correlation: %.2f' % output['r_all'])
            if cv_dict is not None:
                output['rmse_xval'] = np.sqrt(np.mean((output['yfit_xval']-output['Y'])**2))
                output['r_xval'] = np.corrcoef(output['Y'],output['yfit_xval'])[0, 1]
                print('overall CV Root Mean Squared Error: %.2f' % output['rmse_xval'])
                print('overall CV Correlation: %.2f' % output['r_xval'])

        # Plot
        if plot:
            if cv_dict is not None:
                if predictor_settings['prediction_type'] == 'prediction':
                    scatterplot(pd.DataFrame({'Y': output['Y'], 'yfit_xval': output['yfit_xval']}))
                elif predictor_settings['prediction_type'] == 'classification':
                    if predictor_settings['algorithm'] not in ['svm', 'ridgeClassifier', 'ridgeClassifierCV']:
                        output['roc'] = Roc(input_values=output['prob_xval'], binary_outcome=output['Y'].astype('bool'))
                    else:
                        output['roc'] = Roc(input_values=output['dist_from_hyperplane_xval'], binary_outcome=output['Y'].astype('bool'))
                        if predictor_settings['algorithm'] == 'svm' and predictor_cv.probability:
                            output['roc'] = Roc(input_values=output['prob_xval'], binary_outcome=output['Y'].astype('bool'))
                    output['roc'].plot()
            output['weight_map'].plot()

        return output

    def bootstrap(self, analysis_type=None, n_samples=10, save_weights=False,
                  **kwargs):
        """ Bootstrap various Brain_Data analaysis methods (e.g., mean, std,
            regress, predict).  Currently

        Args:
            analysis_type: Type of analysis to bootstrap (mean,std,regress,
                        predict)
            n_samples: Number of samples to boostrap
            **kwargs: Additional keyword arguments to pass to the analysis
                    method

        Returns:
            output: a dictionary of prediction parameters

        """

        # Notes:
        # might want to add options for [studentized, percentile,
        # bias corrected, bias corrected accelerated] methods

        # Regress method is pretty convoluted and slow, this should be
        # optimized better.

        def summarize_bootstrap(sample):
            """ Calculate summary of bootstrap samples

            Args:
                sample: Brain_Data instance of samples

            Returns:
                output: dictionary of Brain_Data summary images

            """

            output = {}

            # Calculate SE of bootstraps
            wstd = sample.std()
            wmean = sample.mean()
            wz = deepcopy(wmean)
            wz.data = wmean.data / wstd.data
            wp = deepcopy(wmean)
            wp.data = 2*(1-norm.cdf(np.abs(wz.data)))

            # Create outputs
            output['Z'] = wz
            output['p'] = wp
            output['mean'] = wmean
            if save_weights:
                output['samples'] = sample

            return output

        analysis_list = ['mean', 'std', 'regress', 'predict']

        if analysis_type in analysis_list:
            data_row_id = range(self.shape()[0])
            sample = self.empty()
            if analysis_type is 'regress': #initialize dictionary of empty betas
                beta = {}
                for i in range(self.X.shape[1]):
                    beta['b' + str(i)] = self.empty()
            for i in range(n_samples):
                this_sample = np.random.choice(data_row_id,
                                               size=len(data_row_id),
                                               replace=True)
                if analysis_type is 'mean':
                    sample = sample.append(self[this_sample].mean())
                elif analysis_type is 'std':
                    sample = sample.append(self[this_sample].std())
                elif analysis_type is 'regress':
                    out = self[this_sample].regress()
                    # Aggegate bootstraps for each beta separately
                    for i, b in enumerate(beta.iterkeys()):
                        beta[b] = beta[b].append(out['beta'][i])
                elif analysis_type is 'predict':
                    if 'algorithm' in kwargs:
                        algorithm = kwargs['algorithm']
                        del kwargs['algorithm']
                    else:
                        algorithm = 'ridge'
                    if 'cv_dict' in kwargs:
                        cv_dict = kwargs['cv_dict']
                        del kwargs['cv_dict']
                    else:
                        cv_dict = None
                    if 'plot' in ['kwargs']:
                        plot = kwargs['plot']
                        del kwargs['plot']
                    else:
                        plot = False
                    out = self[this_sample].predict(algorithm=algorithm,
                                                    cv_dict=cv_dict,
                                                    plot=plot,
                                                    **kwargs)
                    sample = sample.append(out['weight_map'])
        else:
            raise ValueError("The analysis_type you specified (%s) is not yet "
                             "implemented." % (analysis_type))

        # Save outputs
        if analysis_type is 'regress':
            reg_out = {}
            for i, b in enumerate(beta.iterkeys()):
                reg_out[b] = summarize_bootstrap(beta[b])
            output = {}
            for b in reg_out.iteritems():
                for o in b[1].iteritems():
                    if o[0] in output:
                        output[o[0]] = output[o[0]].append(o[1])
                    else:
                        output[o[0]] = o[1]
        else:
            output = summarize_bootstrap(sample)
        return output

    def apply_mask(self, mask):
        """ Mask Brain_Data instance

        Args:
            mask: mask (Brain_Data or nifti object)

        """

        if isinstance(mask, Brain_Data):
            mask = mask.to_nifti() # convert to nibabel
        if not isinstance(mask, nib.Nifti1Image):
            if isinstance(mask, six.string_types):
                if os.path.isfile(mask):
                    mask = nib.load(mask)
                if not ((self.mask.get_affine() == mask.get_affine()).all()) & (self.mask.shape[0:3] == mask.shape[0:3]):
                    mask = resample_img(mask, target_affine=self.mask.get_affine(), target_shape=self.mask.shape)
            else:
                raise ValueError("Mask is not a nibabel instance, Brain_Data "
                                 "instance, or a valid file name.")

        masked = deepcopy(self)
        nifti_masker = NiftiMasker(mask_img=mask)
        masked.data = nifti_masker.fit_transform(self.to_nifti())
        masked.nifti_masker = nifti_masker
        if (len(masked.shape()) > 1) & (masked.shape()[0] == 1):
            masked.data = masked.data.flatten()
        return masked

    def searchlight(self, ncores, process_mask=None, parallel_out=None,
                    radius=3, walltime='24:00:00', email=None,
                    algorithm='svr', cv_dict=None, kwargs={}):

        if len(kwargs) is 0:
            kwargs['kernel']= 'linear'

        # new parallel job
        pbs_kwargs = {'algorithm': algorithm,
                  'cv_dict': cv_dict,
                  'predict_kwargs': kwargs}
        #cv_dict={'type': 'kfolds','n_folds': 5,'stratified':dat.Y}

        parallel_job = PBS_Job(self, parallel_out=parallel_out,
                                process_mask=process_mask, radius=radius,
                                kwargs=pbs_kwargs)

        # make and store data we will need to access on the worker core level
        parallel_job.make_searchlight_masks()
        cPickle.dump(parallel_job, open(
                        os.path.join(parallel_out, "pbs_searchlight.pkl"), "w"))

        #make core startup script (python)
        parallel_job.make_startup_script("core_startup.py")

        # make email notification script (pbs)
        if type(email) is str:
            parallel_job.make_pbs_email_alert(email)

        # make pbs job submission scripts (pbs)
        for core_i in range(ncores):
            script_name = "core_pbs_script_" + str(core_i) + ".pbs"
            parallel_job.make_pbs_scripts(script_name, core_i, ncores, walltime)  # create a script
            print("python " + os.path.join(parallel_out, script_name))
            os.system("qsub " + os.path.join(parallel_out, script_name))  # run it on a core

    def extract_roi(self, mask, method='mean'):
        """ Extract activity from mask

        Args:
            mask: nibabel mask can be binary or numbered for different rois
            method: type of extraction method (default=mean)

        Returns:
            out: mean within each ROI across images

        """

        if not isinstance(mask, Brain_Data):
            if isinstance(mask, nib.Nifti1Image):
                mask = Brain_Data(mask)
            else:
                raise ValueError('Make sure mask is a Brain_Data or nibabel '
                                 'instance')

        ma = mask.copy()

        if len(np.unique(ma.data)) == 2:
            if method is 'mean':
                out = np.mean(self.data[:, np.where(ma.data)].squeeze(), axis=1)
        elif len(np.unique(ma.data)) > 2:
            # make sure each ROI id is an integer
            ma.data = np.round(ma.data).astype(int)
            all_mask = expand_mask(ma)
            out = []
            for i in range(all_mask.shape()[0]):
                if method is 'mean':
                    out.append(np.mean(self.data[:, np.where(all_mask[i].data)].squeeze(),axis=1))
            out = np.array(out)
        return out

    def icc(self, icc_type='icc2'):
        ''' Calculate intraclass correlation coefficient for data within
            Brain_Data class

        ICC Formulas are based on:
        Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
        assessing rater reliability. Psychological bulletin, 86(2), 420.

        icc1:  x_ij = mu + beta_j + w_ij
        icc2/3:  x_ij = mu + alpha_i + beta_j + (ab)_ij + epsilon_ij

        Code modifed from nipype algorithms.icc
        https://github.com/nipy/nipype/blob/master/nipype/algorithms/icc.py

        Args:
            icc_type: type of icc to calculate (icc: voxel random effect,
                    icc2: voxel and column random effect, icc3: voxel and
                    column fixed effect)

        Returns:
            ICC: intraclass correlation coefficient

        '''

        Y = self.data.T
        [n, k] = Y.shape

        # Degrees of Freedom
        dfc = k - 1
        dfe = (n - 1) * (k-1)
        dfr = n - 1

        # Sum Square Total
        mean_Y = np.mean(Y)
        SST = ((Y - mean_Y) ** 2).sum()

        # create the design matrix for the different levels
        x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
        x0 = np.tile(np.eye(n), (k, 1))  # subjects
        X = np.hstack([x, x0])

        # Sum Square Error
        predicted_Y = np.dot(np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))),
                             X.T), Y.flatten('F'))
        residuals = Y.flatten('F') - predicted_Y
        SSE = (residuals ** 2).sum()

        MSE = SSE / dfe

        # Sum square column effect - between colums
        SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
        MSC = SSC / dfc / n

        # Sum Square subject effect - between rows/subjects
        SSR = SST - SSC - SSE
        MSR = SSR / dfr

        if icc_type == 'icc1':
            # ICC(2,1) = (mean square subject - mean square error) /
            # (mean square subject + (k-1)*mean square error +
            # k*(mean square columns - mean square error)/n)
            # ICC = (MSR - MSRW) / (MSR + (k-1) * MSRW)
            NotImplementedError("This method isn't implemented yet.")

        elif icc_type == 'icc2':
            # ICC(2,1) = (mean square subject - mean square error) /
            # (mean square subject + (k-1)*mean square error +
            # k*(mean square columns - mean square error)/n)
            ICC = (MSR - MSE) / (MSR + (k-1) * MSE + k * (MSC - MSE) / n)

        elif icc_type == 'icc3':
            # ICC(3,1) = (mean square subject - mean square error) /
            # (mean square subject + (k-1)*mean square error)
            ICC = (MSR - MSE) / (MSR + (k-1) * MSE)

        return ICC

    def detrend(self, method='linear'):
        """ Remove linear trend from each voxel

        Args:
            type: {'linear','constant'} optional

        Returns:
            out: detrended Brain_Data instance

        """

        if len(self.shape()) == 1:
            raise ValueError('Make sure there is more than one image in order '
                             'to detrend.')

        out = deepcopy(self)
        out.data = detrend(out.data, type=method, axis=0)
        return out

    def copy(self):
        """ Create a copy of a Brain_Data instance.  """

        return deepcopy(self)

    def upload_neurovault(self, access_token=None, collection_name=None,
                          collection_id=None, img_type=None, img_modality=None,
                          **kwargs):
        """ Upload Data to Neurovault.  Will add any columns in self.X to image
            metadata. Index will be used as image name.

        Args:
            access_token: (Required) Neurovault api access token
            collection_name: (Optional) name of new collection to create
            collection_id: (Optional) neurovault collection_id if adding images
                            to existing collection
            img_type: (Required) Neurovault map_type
            img_modality: (Required) Neurovault image modality

        Returns:
            collection: neurovault collection information

        """

        if access_token is None:
            raise ValueError('You must supply a valid neurovault access token')

        api = Client(access_token=access_token)

        # Check if collection exists
        if collection_id is not None:
            collection = api.get_collection(collection_id)
        else:
            try:
                collection = api.create_collection(collection_name)
            except ValueError:
                print('Collection Name already exists.  Pick a '
                      'different name or specify an existing collection id')

        tmp_dir = os.path.join(tempfile.gettempdir(), str(os.times()[-1]))
        os.makedirs(tmp_dir)

        def add_image_to_collection(api, collection, dat, tmp_dir, index_id=0,
                                    **kwargs):
            '''Upload image to collection
            Args:
                api: pynv Client instance
                collection: collection information
                dat: Brain_Data instance to upload
                tmp_dir: temporary directory
                index_id: (int) index for file naming
            '''
            if (len(dat.shape()) > 1) & (dat.shape()[0] > 1):
                raise ValueError('"dat" must be a single image.')
            if not dat.X.empty:
                if isinstance(dat.X.name, six.string_types):
                    img_name = dat.X.name
                else:
                    img_name = collection['name'] + '_' + str(index_id) + '.nii.gz'
            else:
                img_name = collection['name'] + '_' + str(index_id) + '.nii.gz'
            f_path = os.path.join(tmp_dir, img_name)
            dat.write(f_path)
            if not dat.X.empty:
                kwargs.update(dict([(k, dat.X.loc[k]) for k in dat.X.keys()]))
            api.add_image(collection['id'],
                          f_path,
                          name=img_name,
                          modality=img_modality,
                          map_type=img_type,
                          **kwargs)

        if len(self.shape()) == 1:
            add_image_to_collection(api, collection, self, tmp_dir, index_id=0,
                                    **kwargs)
        else:
            for i, x in enumerate(self):
                add_image_to_collection(api, collection, x, tmp_dir,
                                        index_id=i, **kwargs)

        shutil.rmtree(tmp_dir, ignore_errors=True)
        return collection

    def r_to_z(self):
        ''' Apply Fisher's r to z transformation to each element of the data
            object.'''

        out = self.copy()
        out.data = fisher_r_to_z(out.data)
        return out

    def filter(self,sampling_rate=None, high_pass=None,low_pass=None,TR=None,**kwargs):
        ''' Apply 5th order butterworth filter to data. Wraps nilearn functionality. Does not default to detrending and standardizing like nilearn implementation, but this can be overridden using kwargs.

        Args:
            sampling_rate: sampling frequence (e.g. TR) in seconds
            high_pass: high pass cutoff frequency
            low_pass: low pass cutoff frequency
            kwargs: other keyword arguments to nilearn.signal.clean

        Returns:
            Brain_Data: Filtered Brain_Data instance
        '''

        if sampling_rate is None:
            raise ValueError("Need to provide sampling rate!")
        if high_pass is None and low_pass is None:
            raise ValueError("high_pass and/or low_pass cutoff must be provided!")
        if TR is None:
            raise ValueError("Need to provide TR!")

        standardize = kwargs.get('standardize',False)
        detrend = kwargs.get('detrend',False)
        out = self.copy()
        out.data = clean(out.data,t_r=TR,detrend=detrend,standardize=standardize,high_pass=high_pass,low_pass=low_pass,**kwargs)
        return out


    def dtype(self):
        ''' Get data type of Brain_Data.data.'''
        return self.data.dtype

    def astype(self, dtype):
        ''' Cast Brain_Data.data as type.

        Args:
            dtype: datatype to convert

        Returns:
            Brain_Data: Brain_Data instance with new datatype

        '''

        out = self.copy()
        out.data = out.data.astype(dtype)
        return out

    def standardize(self, method='center'):
        ''' Standardize Brain_Data() instance.

        Args:
            method: ['center','zscore']

        Returns:
            Brain_Data Instance

        '''

        out = self.copy()
        if method is 'center':
            out.data = out.data - np.repeat(np.array([np.mean(out.data, axis=0)]).T, len(out), axis=1).T
        elif method is 'zscore':
            out.data = out.data - np.repeat(np.array([np.mean(out.data, axis=0)]).T, len(out), axis=1).T
            out.data = out.data/np.repeat(np.array([np.std(out.data, axis=0)]).T, len(out), axis=1).T
        else:
            raise ValueError('method must be ["center","zscore"')
        return out

    def groupby(self, mask):
        '''Create groupby instance'''
        return Groupby(self, mask)

    def aggregate(self, mask, func):
        '''Create new Brain_Data instance that aggregages func over mask'''
        dat = self.groupby(mask)
        values = dat.apply(func)
        return dat.combine(values)

    def threshold(self, thresh=0, binarize=False):
        '''Threshold Brain_Data instance

        Args:
            thresh: cutoff to threshold image (float).  if 'threshold'=50%,
                    will calculate percentile.
            binarize (bool): if 'binarize'=True then binarize output

        Returns:
            Brain_Data: thresholded Brain_Data instance

        '''

        b = self.copy()
        if isinstance(thresh, str):
            if thresh[-1] is '%':
                thresh = np.percentile(b.data, float(thresh[:-1]))
        if binarize:
            b.data = b.data > thresh
        else:
            b.data[b.data < thresh] = 0
        return b

    def regions(self, min_region_size=1350, extract_type='local_regions',
                smoothing_fwhm=6):
        ''' Extract brain connected regions into separate regions.

        Args:
            min_region_size (int): Minimum volume in mm3 for a region to be
                                kept.
            extract_type (str): Type of extraction method
                                ['connected_components', 'local_regions'].
                                If 'connected_components', each component/region
                                in the image is extracted automatically by
                                labelling each region based upon the presence of
                                unique features in their respective regions.
                                If 'local_regions', each component/region is
                                extracted based on their maximum peak value to
                                define a seed marker and then using random
                                walker segementation algorithm on these
                                markers for region separation.
            smoothing_fwhm (scalar): Smooth an image to extract more sparser
                                regions. Only works for extract_type
                                'local_regions'.

        Returns:
            Brain_Data: Brain_Data instance with extracted ROIs as data.
        '''

        regions, _ = connected_regions(self.to_nifti(),
                                       min_region_size, extract_type,
                                       smoothing_fwhm)
        return Brain_Data(regions)


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
            if matrix_type.lower() not in ['distance','similarity','directed','distance_flat','similarity_flat','directed_flat']:
                raise ValueError("matrix_type must be [None,'distance', "
                                "'similarity','directed','distance_flat', "
                                "'similarity_flat','directed_flat']")

        if data is None:
            self.data = np.array([])
            self.matrix_type = 'empty'
            self.is_single_matrix = np.nan
            self.issymmetric = np.nan
        elif isinstance(data, list):
            d_all = []; symmetric_all = []; matrix_type_all = []
            for d in data:
                data_tmp, issymmetric_tmp, matrix_type_tmp, _ = self._import_single_data(d,matrix_type=matrix_type)
                d_all.append(data_tmp)
                symmetric_all.append(issymmetric_tmp)
                matrix_type_all.append(matrix_type_tmp)
            if not all_same(symmetric_all):
                raise ValueError('Not all matrices are of the same symmetric '
                                'type.')
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
            if isinstance(Y, str):
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

        if isinstance(data, str):
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
        if isinstance(thresh, str):
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

    def stats_label_distance(self, labels, n_permute=5000):
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
            stats[str(i)] = two_sample_permutation(tmp1, tmp2, n_permute=n_permute)
        return stats

    def plot_silhouette(self,labels,ax=None,permutation_test=True,n_permute=5000,**kwargs):

        distance = pd.DataFrame(self.squareform())

        if len(labels) != distance.shape[0]:
            raise ValueError('Labels must be same length as distance matrix')

        (f,outAll) = plot_silhouette(distance,labels,ax=None,permutation_test=True,n_permute=5000,**kwargs)

        return (f,outAll)


class Groupby(object):
    def __init__(self, data, mask):

        if not isinstance(data, Brain_Data):
            raise ValueError('Groupby requires a Brain_Data instance.')
        if not isinstance(mask, Brain_Data):
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
        self.split(data, mask)

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
            yield (x, self.data[x])

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.data[index]
        else:
            raise ValueError('Groupby currently only supports integer indexing')

    def split(self, data, mask):
        '''Split Brain_Data instance into separate masks and store as a
            dictionary.
        '''

        self.data = {}
        for i, m in enumerate(mask):
            self.data[i] = data.apply_mask(m)

    def apply(self, method):
        '''Apply Brain_Data instance methods to each element of Groupby
            object.
        '''
        return dict([(i, getattr(x, method)()) for i, x in self])

    def combine(self, value_dict):
        '''Combine value dictionary back into masks'''
        out = self.mask.copy().astype(float)
        for i in value_dict.iterkeys():
            if isinstance(value_dict[i], Brain_Data):
                if value_dict[i].shape()[0] == np.sum(self.mask[i].data):
                    out.data[i, out.data[i, :] == 1] = value_dict[i].data
                else:
                    raise ValueError('Brain_Data instances are different '
                                     'shapes.')
            elif isinstance(value_dict[i], (float, int, bool, np.number)):
                out.data[i, :] = out.data[i, :]*value_dict[i]
            else:
                raise ValueError('No method for aggregation implented for %s '
                                 'yet.' % type(value_dict[i]))
        return out.sum()


class Design_Matrix_Series(Series):

    """
    This is a sub-class of pandas series. While not having additional methods
    of it's own required to retain normal slicing functionality for the
    Design_Matrix class, i.e. how slicing is typically handled in pandas.
    All methods should be called on Design_Matrix below.
    """

    @property
    def _constructor(self):
        return Design_Matrix_Series

    @property
    def _constructor_expanddim(self):
        return Design_Matrix


class Design_Matrix(DataFrame):

    """Design_Matrix is a class to represent design matrices with convenience
        functionality for convolution, upsampling and downsampling. It plays
        nicely with Brain_Data and can be used to build an experimental design
        to pass to Brain_Data's X attribute. It is essentially an enhanced
        pandas df, with extra attributes and methods. Methods always return a
        new design matrix instance.

    Args:
        convolved (list, optional): on what columns convolution has been performed; defaults to None
        hasIntercept (bool, optional): whether the design matrix has an
                            intercept column; defaults to False
        sampling_rate (float, optional): sampling rate of each row in seconds (e.g. TR in neuroimaging); defaults to None

    """

    _metadata = ['sampling_rate', 'convolved', 'hasIntercept']

    def __init__(self, *args, **kwargs):

        sampling_rate = kwargs.pop('sampling_rate',None)
        convolved = kwargs.pop('convolved', None)
        hasIntercept = kwargs.pop('hasIntercept', False)
        self.sampling_rate = sampling_rate
        self.convolved = convolved
        self.hasIntercept = hasIntercept

        super(Design_Matrix, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return Design_Matrix

    @property
    def _constructor_sliced(self):
        return Design_Matrix_Series


    def info(self):
        """Print class meta data.

        """
        return '%s.%s(sampling_rate=%s, shape=%s, convolved=%s, hasIntercept=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.sampling_rate,
            self.shape,
            self.convolved,
            self.hasIntercept
            )

    def append(self, df, axis, **kwargs):
        """Method for concatenating another design matrix row or column-wise.
            Can "uniquify" certain columns when appending row-wise, and by
            default will attempt to do that with the intercept.

        Args:
            axis (int): 0 for row-wise (vert-cat), 1 for column-wise (horz-cat)
            separate (bool,optional): whether try and uniquify columns;
                                        defaults to True; only applies
                                        when axis==0
            addIntercept (bool,optional): whether to add intercepts to matrices
                                        before appending; defaults to False;
                                        only applies when axis==0
            uniqueCols (list,optional): what additional columns to try to keep
                                        separated by uniquifying; defaults to
                                        intercept only; only applies when
                                        axis==0

        """
        if axis == 1:
            return self.horzcat(df)
        elif axis == 0:
            return self.vertcat(df, **kwargs)
        else:
            raise ValueError("Axis must be 0 (row) or 1 (column)")


    def horzcat(self, df):
        """Used by .append(). Append another design matrix, column-wise
            (horz cat). Always returns a new design_matrix.

        """
        if self.shape[0] != df.shape[0]:
            raise ValueError("Can't append differently sized design matrices! "
                             "Mat 1 has %s rows and Mat 2 has %s rows."
                            % (self.shape[0]), df.shape[0])
        out = pd.concat([self, df], axis=1)
        out.sampling_rate = self.sampling_rate
        out.convolved = self.convolved
        out.hasIntercept = self.hasIntercept
        return out


    def vertcat(self, df, separate=True, addIntercept=False, uniqueCols=[]):
        """Used by .append(). Append another design matrix row-wise (vert cat).
            Always returns a new design matrix.

        """

        outdf = df.copy()
        assert self.hasIntercept == outdf.hasIntercept, ("Intercepts are "
                "ambigious. Both design matrices should match in whether "
                "they do or don't have intercepts.")

        if addIntercept:
            self['intercept'] = 1
            self.hasIntercept = True
            outdf['intercept'] = 1
            outdf.hasIntercept = True
        if separate:
            if self.hasIntercept:
                uniqueCols += ['intercept']
            idx_1 = []
            idx_2 = []
            for col in uniqueCols:
                # To match substrings within column names, we loop over each
                # element and search for the uniqueCol as a substring within it;
                # first we do to check if the uniqueCol actually occurs in both
                # design matrices, if so then we change it's name before
                # concatenating
                joint = set(self.columns) & set(outdf.columns)
                shared = [elem for elem in joint if col in elem]
                if shared:
                    idx_1 += [i for i, elem in enumerate(self.columns) if col in elem]
                    idx_2 += [i for i, elem in enumerate(outdf.columns) if col in elem]
            aRename = {self.columns[elem]: '0_'+self.columns[elem] for i, elem in enumerate(idx_1)}
            bRename = {outdf.columns[elem]: '1_'+outdf.columns[elem] for i, elem in enumerate(idx_2)}
            if aRename:
                out = self.rename(columns=aRename)
                outdf = outdf.rename(columns=bRename)
                colOrder = []
                #retain original column order as closely as possible
                for colA,colB in zip(out.columns, outdf.columns):
                    colOrder.append(colA)
                    if colA != colB:
                        colOrder.append(colB)
                out = super(Design_Matrix, out).append(outdf, ignore_index=True)
                # out = out.append(outdf,separate=False,axis=0,ignore_index=True).fillna(0)
                out = out[colOrder]
            else:
                raise ValueError("Separate concatentation impossible. None of "
                                "the requested unique columns were found in "
                                "both design_matrices.")
        else:
            out = super(Design_Matrix, self).append(outdf, ignore_index=True)
            # out = self.append(df,separate=False,axis=0,ignore_index=True).fillna(0)

        out.convolved = self.convolved
        out.sampling_rate = self.sampling_rate
        out.hasIntercept = self.hasIntercept

        return out

    def vif(self):
        """Compute variance inflation factor amongst columns of design matrix,
            ignoring the intercept. Much faster that statsmodels and more
            reliable too. Uses the same method as Matlab and R (diagonal
            elements of the inverted correlation matrix).

        Returns:
            vifs (list): list with length == number of columns - intercept

        """
        assert self.shape[1] > 1, "Can't compute vif with only 1 column!"
        if self.hasIntercept:
            idx = [i for i, elem in enumerate(self.columns) if 'intercept' not in elem]
            out = self[self.columns[np.r_[idx]]]
        else:
            out = self[self.columns]

        if any(out.corr() < 0):
            warnings.warn("Correlation matrix has negative values!")

        return np.diag(np.linalg.inv(out.corr()), 0)

    def heatmap(self, figsize=(8, 6), **kwargs):
        """Visualize Design Matrix spm style. Use .plot() for typical pandas
            plotting functionality. Can pass optional keyword args to seaborn
            heatmap.

        """
        fig, ax = plt.subplots(1, figsize=figsize)
        ax = sns.heatmap(self, cmap='gray', cbar=False, ax=ax, **kwargs)
        for _, spine in ax.spines.items():
            spine.set_visible(True)
        for i, label in enumerate(ax.get_yticklabels()):
            if i == 0 or i == self.shape[0]-1:
                label.set_visible(True)
            else:
                label.set_visible(False)
        ax.axhline(linewidth=4, color="k")
        ax.axvline(linewidth=4, color="k")
        ax.axhline(y=self.shape[0], color='k', linewidth=4)
        ax.axvline(x=self.shape[1], color='k', linewidth=4)
        plt.yticks(rotation=0)

    def convolve(self, conv_func='hrf', colNames=None):
        """Perform convolution using an arbitrary function.

        Args:
            conv_func (ndarray or string): either a 1d numpy array containing output of a function that you want to convolve; a samples by kernel 2d array of several kernels to convolve; or th string 'hrf' which defaults to a glover HRF function at the Design_matrix's sampling_freq
            colNames (list): what columns to perform convolution on; defaults
                            to all skipping intercept, and columns containing 'poly' or 'cosine'

        """
        assert self.sampling_rate is not None, "Design_matrix has no sampling_rate set!"

        if colNames is None:
            colNames = [col for col in self.columns if 'intercept' not in col and 'poly' not in col and 'cosine' not in col]
        nonConvolved = [col for col in self.columns if col not in colNames]

        if type(conv_func) == str:
            assert conv_func == 'hrf',"Did you mean 'hrf'? 'hrf' can generate a kernel for you, otherwise custom kernels should be passed in as 1d or 2d arrays."

            assert self.sampling_rate is not None, "Design_matrix sampling rate not set. Can't figure out how to generate HRF!"
            conv_func = glover_hrf(self.sampling_rate,oversampling=1)

        else:
            assert type(conv_func) == np.ndarray, 'Must provide a function for convolution!'

        if len(conv_func.shape) > 1:
            assert conv_func.shape[0] > conv_func.shape[1], '2d conv_func must be formatted as, samples X kernels!'
            conv_mats = []
            for i in xrange(conv_func.shape[1]):
                c = self[colNames].apply(lambda x: np.convolve(x, conv_func[:,i])[:self.shape[0]])
                c.columns = [str(col)+'_c'+str(i) for col in c.columns]
                conv_mats.append(c)
                out = pd.concat(conv_mats+ [self[nonConvolved]], axis=1)
        else:
            c = self[colNames].apply(lambda x: np.convolve(x, conv_func)[:self.shape[0]])
            c.columns = [str(col)+'_c0' for col in c.columns]
            out = pd.concat([c,self[nonConvolved]], axis=1)

        out.convolved = colNames
        out.sampling_rate = self.sampling_rate
        out.hasIntercept = self.hasIntercept
        return out

    def downsample(self, target,**kwargs):
        """Downsample columns of design matrix. Relies on
            nltools.stats.downsample, but ensures that returned object is a
            design matrix.

        Args:
            target(float): downsampling target, typically in samples not
                            seconds
            kwargs: additional inputs to nltools.stats.downsample

        """
        df = downsample(self, sampling_freq=self.sampling_rate, target=target, **kwargs)

        # convert df to a design matrix
        newMat = Design_Matrix(df, sampling_rate=target,
                               convolved=self.convolved,
                               hasIntercept=self.hasIntercept)
        return newMat

    def upsample(self, target,**kwargs):
        """Upsample columns of design matrix. Relies on
            nltools.stats.upsample, but ensures that returned object is a
            design matrix.

        Args:
            target(float): downsampling target, typically in samples not
                            seconds
            kwargs: additional inputs to nltools.stats.downsample

        """
        df = upsample(self, sampling_freq=self.sampling_rate, target=target, **kwargs)

        # convert df to a design matrix
        newMat = Design_Matrix(df, sampling_rate=target,
                               convolved=self.convolved,
                               hasIntercept=self.hasIntercept)
        return newMat

    def zscore(self, colNames=[]):
        """Z-score specific columns of design matrix. Relies on
            nltools.stats.downsample, but ensures that returned object is a
            design matrix.

        Args:
            colNames (list): columns to z-score; defaults to all columns

        """
        colOrder = self.columns
        if not list(colNames):
            colNames = self.columns
        nonZ = [col for col in self.columns if col not in colNames]
        df = zscore(self[colNames])
        df = pd.concat([df, self[nonZ]], axis=1)
        df = df[colOrder]
        newMat = Design_Matrix(df, sampling_rate=self.sampling_rate,
                               convolved=self.convolved,
                               hasIntercept=self.hasIntercept)

        return newMat

    def addpoly(self, order=0, include_lower=True):
        """Add nth order polynomial terms as columns to design matrix.

        Args:
            order (int): what order terms to add; 0 = constant/intercept
                        (default), 1 = linear, 2 = quadratic, etc
            include_lower: (bool) whether to add lower order terms if order > 0

        """

        #This method is kind of ugly
        polyDict = {}
        if include_lower:
            if order > 0:
                for i in xrange(0, order+1):
                    if i == 0:
                        if self.hasIntercept:
                            warnings.warn("Design Matrix already has "
                                          "intercept...skipping")
                        else:
                            polyDict['intercept'] = np.repeat(1, self.shape[0])
                    else:
                        polyDict['poly_' + str(i)] = (range(self.shape[0]) - np.mean(range(self.shape[0]))) ** i
            else:
                if self.hasIntercept:
                    raise ValueError("Design Matrix already has intercept!")
                else:
                    polyDict['intercept'] = np.repeat(1, self.shape[0])
        else:
            if order == 0:
                if self.hasIntercept:
                    raise ValueError("Design Matrix already has intercept!")
                else:
                    polyDict['intercept'] = np.repeat(1, self.shape[0])
            else:
                polyDict['poly_'+str(order)] = (range(self.shape[0]) - np.mean(range(self.shape[0])))**order

        toAdd = Design_Matrix(polyDict)
        out = self.append(toAdd, axis=1)
        if 'intercept' in polyDict.keys() or self.hasIntercept:
            out.hasIntercept = True
        return out

    def add_dct_basis(self,duration=180):
        """Adds cosine basis functions to Design_Matrix columns, based on spm-style discrete cosine transform for use in high-pass filtering.

        Args:
            duration (int): length of filter in seconds

        """
        assert self.sampling_rate is not None, "Design_Matrix has no sampling_rate set!"
        basis_mat = make_cosine_basis(self.shape[0],self.sampling_rate,duration)

        basis_frame = Design_Matrix(basis_mat,sampling_rate=self.sampling_rate,hasIntercept=False,convolved=False)

        basis_frame.columns = ['cosine_'+str(i+1) for i in xrange(basis_frame.shape[1])]

        out = self.append(basis_frame,axis=1)

        return out


def all_same(items):
    return np.all(x == items[0] for x in items)

def _vif(X, y):
    """
        DEPRECATED
        Helper function to compute variance inflation factor. Unclear whether
        there are errors with this method relative to stats.models. Seems like
        stats.models is sometimes inconsistent with R. R always uses the
        diagonals of the inverted covariance matrix which is what's implemented
        instead of this.

        Args:
            X: (Dataframe) explanatory variables
            y: (Dataframe/Series) outcome variable

    """

    b,resid, _, _ = np.linalg.lstsq(X, y)
    SStot = y.var(ddof=0)*len(y)
    if SStot == 0:
        SStot = .0001  # to prevent divide by 0 errors
    r2 = 1.0 - (resid/SStot)
    if r2 == 1:
        r2 = 0.9999  # to prevent divide by 0 errors
    return (1.0/(1.0-r2))[0]
