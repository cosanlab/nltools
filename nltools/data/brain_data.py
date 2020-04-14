from __future__ import division

'''
NeuroLearn Brain Data
=====================

Classes to represent brain image data.

'''

# Notes:
# Need to figure out how to speed up loading and resampling of data

__author__ = ["Luke Chang"]
__license__ = "MIT"

from nilearn.signal import clean
from scipy.stats import ttest_1samp, pearsonr
from scipy.stats import t as t_dist
from scipy.signal import detrend
import os
import shutil
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import tempfile
from copy import deepcopy
import six
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.utils import check_random_state
from sklearn.preprocessing import scale
from pynv import Client
from joblib import Parallel, delayed
from nltools.mask import expand_mask
from nltools.analysis import Roc
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map
from nilearn.image import smooth_img, resample_to_img
from nilearn.masking import intersect_masks
from nilearn.regions import connected_regions, connected_label_regions
from nltools.utils import (set_algorithm,
                           attempt_to_import,
                           concatenate,
                           _bootstrap_apply_func,
                           set_decomposition_algorithm,
                           check_brain_data,
                           check_brain_data_is_single,
                           _roi_func,
                           get_mni_from_img_resolution,
                           _df_meta_to_arr)
from nltools.cross_validation import set_cv
from nltools.plotting import scatterplot
from nltools.stats import (pearson,
                           fdr,
                           holm_bonf,
                           threshold,
                           fisher_r_to_z,
                           transform_pairwise,
                           summarize_bootstrap,
                           procrustes,
                           find_spikes,
                           regress_permutation)
from nltools.stats import regress as regression
from .adjacency import Adjacency
from nltools.prefs import MNI_Template, resolve_mni_path
from nltools.external.srm import DetSRM, SRM
from nltools.plotting import plot_interactive_brain, plot_brain
from nilearn.decoding import SearchLight
import deepdish as dd


# Optional dependencies
nx = attempt_to_import('networkx', 'nx')
mne_stats = attempt_to_import('mne.stats', name='mne_stats', fromlist=['spatio_temporal_cluster_1samp_test', 'ttest_1samp_no_p'])
MAX_INT = np.iinfo(np.int32).max


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
            self.mask = nib.load(resolve_mni_path(MNI_Template)['mask'])
        self.nifti_masker = NiftiMasker(mask_img=self.mask)

        if data is not None:
            if isinstance(data, six.string_types):
                if 'http://' in data:
                    from nltools.datasets import download_nifti
                    tmp_dir = os.path.join(tempfile.gettempdir(),
                                           str(os.times()[-1]))
                    os.makedirs(tmp_dir)
                    data = nib.load(download_nifti(data, data_dir=tmp_dir))
                elif ('.h5' in data) or ('.hdf5' in data):
                    f = dd.io.load(data)
                    self.data = f['data']
                    self.X = pd.DataFrame(f['X'], columns=[e.decode('utf-8') if isinstance(e, bytes) else e for e in f['X_columns']], index=[e.decode('utf-8') if isinstance(e, bytes) else e for e in f['X_index']])
                    self.Y = pd.DataFrame(f['Y'], columns=[e.decode('utf-8') if isinstance(e, bytes) else e for e in f['Y_columns']], index=[e.decode('utf-8') if isinstance(e, bytes) else e for e in f['Y_index']])
                    self.mask = nib.Nifti1Image(f['mask_data'], affine=f['mask_affine'], file_map={'image': nib.FileHolder(filename=f['mask_file_name'])})
                    nifti_masker = NiftiMasker(self.mask)
                    self.nifti_masker = nifti_masker.fit(self.mask)
                    self.file_name = f['file_name']
                    return

                else:
                    data = nib.load(data)
                self.data = self.nifti_masker.fit_transform(data)
            elif isinstance(data, list):
                if isinstance(data[0], Brain_Data):
                    tmp = concatenate(data)
                    for item in ['data', 'Y', 'X', 'mask', 'nifti_masker',
                                 'file_name']:
                        setattr(self, item, getattr(tmp, item))
                else:
                    if all([isinstance(x, data[0].__class__) for x in data]):
                        self.data = []
                        for i in data:
                            if isinstance(i, six.string_types):
                                self.data.append(self.nifti_masker.fit_transform(
                                                 nib.load(i)))
                            elif isinstance(i, nib.Nifti1Image):
                                self.data.append(self.nifti_masker.fit_transform(i))
                        self.data = np.concatenate(self.data)
                    else:
                        raise ValueError('Make sure all objects in the list are the same type.')
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
        if isinstance(index, (int, np.integer)):
            new.data = np.array(self.data[index, :]).squeeze()
        else:
            if isinstance(index, slice):
                new.data = self.data[index, :]
            else:
                index = np.array(index).flatten()
                new.data = np.array(self.data[index, :]).squeeze()
        if not self.Y.empty:
            new.Y = self.Y.iloc[index]
            if isinstance(new.Y, pd.Series):
                new.Y.reset_index(inplace=True, drop=True)
        if not self.X.empty:
            new.X = self.X.iloc[index]
            if len(new.X) > 1:
                new.X.reset_index(inplace=True, drop=True)
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
        if isinstance(y, (int, np.integer, float, np.floating)):
            new.data = new.data + y
        elif isinstance(y, Brain_Data):
            if self.shape() != y.shape():
                raise ValueError("Both Brain_Data() instances need to be the "
                                 "same shape.")
            new.data = new.data + y.data
        else:
            raise ValueError('Can only add int, float, or Brain_Data')
        return new

    def __radd__(self, y):
        new = deepcopy(self)
        if isinstance(y, (int, np.integer, float, np.floating)):
            new.data = y + new.data
        elif isinstance(y, Brain_Data):
            if self.shape() != y.shape():
                raise ValueError("Both Brain_Data() instances need to be the "
                                 "same shape.")
            new.data = y.data + new.data
        else:
            raise ValueError('Can only add int, float, or Brain_Data')
        return new

    def __sub__(self, y):
        new = deepcopy(self)
        if isinstance(y, (int, np.integer, float, np.floating)):
            new.data = new.data - y
        elif isinstance(y, Brain_Data):
            if self.shape() != y.shape():
                raise ValueError('Both Brain_Data() instances need to be the '
                                 'same shape.')
            new.data = new.data - y.data
        else:
            raise ValueError('Can only add int, float, or Brain_Data')
        return new

    def __rsub__(self, y):
        new = deepcopy(self)
        if isinstance(y, (int, np.integer, float, np.floating)):
            new.data = y - new.data
        elif isinstance(y, Brain_Data):
            if self.shape() != y.shape():
                raise ValueError('Both Brain_Data() instances need to be the '
                                 'same shape.')
            new.data = y.data - new.data
        else:
            raise ValueError('Can only add int, float, or Brain_Data')
        return new

    def __mul__(self, y):
        new = deepcopy(self)
        if isinstance(y, (int, np.integer, float, np.floating)):
            new.data = new.data * y
        elif isinstance(y, Brain_Data):
            if self.shape() != y.shape():
                raise ValueError("Both Brain_Data() instances need to be the "
                                 "same shape.")
            new.data = np.multiply(new.data, y.data)
        elif isinstance(y, (list, np.ndarray)):
            if len(y) != len(self):
                raise ValueError('Vector multiplication requires that the '
                                 'length of the vector match the number of '
                                 'images in Brain_Data instance.')
            else:
                new.data = np.dot(new.data.T, y).T
        else:
            raise ValueError('Can only multiply int, float, list, or Brain_Data')
        return new

    def __rmul__(self, y):
        new = deepcopy(self)
        if isinstance(y, (int, np.integer, float, np.floating)):
            new.data = y * new.data
        elif isinstance(y, Brain_Data):
            if self.shape() != y.shape():
                raise ValueError("Both Brain_Data() instances need to be the "
                                 "same shape.")
            new.data = np.multiply(y.data, new.data)
        else:
            raise ValueError('Can only multiply int, float, or Brain_Data')
        return new

    def __iter__(self):
        for x in range(len(self)):
            yield self[x]

    def shape(self):
        """ Get images by voxels shape. """

        return self.data.shape

    def mean(self, axis=0):
        ''' Get mean of each voxel or image
            
            Args:
                axis: (int) across images=0 (default), within images=1
            
            Returns:
                out: (float/np.array/Brain_Data)

        '''

        out = deepcopy(self)
        if check_brain_data_is_single(self):
            out = np.mean(self.data)
        else:
            if axis == 0:
                out.data = np.mean(self.data, axis=0)
                out.X = pd.DataFrame()
                out.Y = pd.DataFrame()
            elif axis == 1:
                out = np.mean(self.data, axis=1)
            else:
                raise ValueError('axis must be 0 or 1')
        return out

    def median(self, axis=0):
        ''' Get median of each voxel or image
            
            Args:
                axis: (int) across images=0 (default), within images=1
            
            Returns:
                out: (float/np.array/Brain_Data)
                
        '''

        out = deepcopy(self)
        if check_brain_data_is_single(self):
            out = np.median(self.data)
        else:
            if axis == 0:
                out.data = np.median(self.data, axis=0)
                out.X = pd.DataFrame()
                out.Y = pd.DataFrame()
            elif axis == 1:
                out = np.median(self.data, axis=1)
            else:
                raise ValueError('axis must be 0 or 1')
        return out

    def std(self):
        """ Get standard deviation of each voxel across images. """

        out = deepcopy(self)
        if len(self.shape()) > 1:
            out.data = np.std(self.data, axis=0)
            out.X = pd.DataFrame()
            out.Y = pd.DataFrame()
        else:
            out = np.std(self.data)
        return out

    def sum(self):
        """ Sum over voxels."""

        out = deepcopy(self)
        if len(self.shape()) > 1:
            out.data = np.sum(out.data, axis=0)
            out.X = pd.DataFrame()
            out.Y = pd.DataFrame()
        else:
            out = np.sum(self.data)
        return out

    def to_nifti(self):
        """ Convert Brain_Data Instance into Nifti Object """

        return self.nifti_masker.inverse_transform(self.data)

    def write(self, file_name=None, **kwargs):
        """ Write out Brain_Data object to Nifti or HDF5 File.

        Args:
            file_name: (str) name of nifti file including path
            kwargs: optional arguments to deepdish.io.save

        """

        if ('.h5' in file_name) or ('.hdf5' in file_name):
            x_columns, x_index = _df_meta_to_arr(self.X)
            y_columns, y_index = _df_meta_to_arr(self.Y)
            dd.io.save(file_name, {
                'data': self.data,
                'X': self.X.values,
                'X_columns': x_columns,
                'X_index': x_index,
                'Y': self.Y.values,
                'Y_columns': y_columns,
                'Y_index': y_index,
                'mask_affine': self.mask.affine,
                'mask_data': self.mask.get_data(),
                'mask_file_name': self.mask.get_filename(),
                'file_name': self.file_name
            }, compression=kwargs.get('compression', 'blosc'))
        else:
            self.to_nifti().to_filename(file_name)

    def scale(self, scale_val=100.):
        """ Scale all values such that they are on the range [0, scale_val],
            via grand-mean scaling. This is NOT global-scaling/intensity
            normalization. This is useful for ensuring that data is on a
            common scale (e.g. good for multiple runs, participants, etc)
            and if the default value of 100 is used, can be interpreted as
            something akin to (but not exactly) "percent signal change."
            This is consistent with default behavior in AFNI and SPM.
            Change this value to 10000 to make consistent with FSL.

        Args:
            scale_val: (int/float) what value to send the grand-mean to;
                        default 100

        """

        out = deepcopy(self)
        out.data = out.data / out.data.mean() * scale_val

        return out

    def plot(self, limit=5, anatomical=None, view='axial', threshold_upper=None, threshold_lower=None, **kwargs):
        """ Create a quick plot of self.data.  Will plot each image separately

        Args:
            limit: (int) max number of images to return
            anatomical: (nifti, str) nifti image or file name to overlay
            view: (str) 'axial' for limit number of axial slices;
                        'glass' for ortho-view glass brain; 'mni' for
                        multi-slice view mni brain; 'full' for both glass and
                        mni views
            threshold_upper: (str/float) threshold if view is 'glass',
                             'mni', or 'full'
            threshold_lower: (str/float)threshold if view is 'glass',
                             'mni', or 'full'
            save: (str/bool): optional string file name or path for saving; only applies if view is 'mni', 'glass', or 'full'. Filenames will appended with the orientation they belong to

        """

        if view == 'axial':
            if threshold is not None:
                print("threshold is ignored for simple axial plots")
            if anatomical is not None:
                if not isinstance(anatomical, nib.Nifti1Image):
                    if isinstance(anatomical, six.string_types):
                        anatomical = nib.load(anatomical)
                    else:
                        raise ValueError("anatomical is not a nibabel instance")
            else:
                # anatomical = nib.load(resolve_mni_path(MNI_Template)['plot'])
                anatomical = get_mni_from_img_resolution(self, img_type='plot')

            if self.data.ndim == 1:
                _, a = plt.subplots(nrows=1, figsize=(15, 2))
                plot_stat_map(self.to_nifti(), anatomical,
                              cut_coords=range(-40, 50, 10), display_mode='z',
                              black_bg=True, colorbar=True, draw_cross=False,
                              axes=a, **kwargs)
            else:
                n_subs = np.minimum(self.data.shape[0], limit)
                _, a = plt.subplots(nrows=n_subs, figsize=(15, len(self) * 2))
                for i in range(n_subs):
                    plot_stat_map(self[i].to_nifti(), anatomical,
                                  cut_coords=range(-40, 50, 10),
                                  display_mode='z',
                                  black_bg=True,
                                  colorbar=True,
                                  draw_cross=False,
                                  axes=a[i],
                                  **kwargs)
            return
        elif view in ['glass', 'mni', 'full']:
            if self.data.ndim == 1:
                return plot_brain(self, how=view, thr_upper=threshold_upper, thr_lower=threshold_lower, **kwargs)
            else:
                raise ValueError("Plotting in 'glass', 'mni', or 'full' views only works with a 3D image")
        else:
            raise ValueError("view must be one of: 'axial', 'glass', 'mni', 'full'.")

    def iplot(self, threshold=0, surface=False, anatomical=None, **kwargs):
        """ Create an interactive brain viewer for the current brain data instance.

        Args:
            threshold: (float/str) two-sided threshold to initialize the
                        visualization, maybe be a percentile string; default 0
            surface: (bool) whether to create a surface-based plot; default False
            anatomical: nifti image or filename to overlay
            kwargs: optional arguments to nilearn.view_img or
                    nilearn.view_img_on_surf

        Returns:
            interactive brain viewer widget

        """
        if anatomical is not None:
            if not isinstance(anatomical, nib.Nifti1Image):
                if isinstance(anatomical, six.string_types):
                    anatomical = nib.load(anatomical)
                else:
                    raise ValueError("anatomical is not a nibabel instance")
        else:
            # anatomical = nib.load(resolve_mni_path(MNI_Template)['brain'])
            anatomical = get_mni_from_img_resolution(self, img_type='brain')
        return plot_interactive_brain(self, threshold=threshold, surface=surface, anatomical=anatomical, **kwargs)

    def regress(self, mode='ols', **kwargs):
        """ Run a mass-univariate regression across voxels. Three types of regressions can be run:
        1) Standard OLS (default)
        2) Robust OLS (heteroscedasticty and/or auto-correlation robust errors), i.e. OLS with "sandwich estimators"
        3) ARMA (auto-regressive and moving-average lags = 1 by default; experimental)

        For more information see the help for nltools.stats.regress

        ARMA notes: This experimental mode is similar to AFNI's 3dREMLFit but without spatial smoothing of voxel auto-correlation estimates. It can be **very computationally intensive** so parallelization is used by default to try to speed things up. Speed is limited because a unique ARMA model is fit to *each voxel* (like AFNI/FSL), but unlike SPM, which assumes the same AR parameters (~0.2) at each voxel. While coefficient results are typically very similar to OLS, std-errors and so t-stats, dfs and and p-vals can differ greatly depending on how much auto-correlation is explaining the response in a voxel
        relative to other regressors in the design matrix.

        Args:
            mode (str): kind of model to fit; must be one of 'ols' (default), 'robust', or 'arma'
            kwargs (dict): keyword arguments to nltools.stats.regress

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

        b, t, p, _, res = regression(self.X, self.data, mode=mode, **kwargs)

        # Prevent copy of all data in self multiple times; instead start with an empty instance and copy only needed attributes from self, and use this as a template for other outputs
        b_out = self.__class__()
        b_out.mask = deepcopy(self.mask)
        b_out.nifti_masker = deepcopy(self.nifti_masker)

        # Use this as template for other outputs before setting data
        t_out = b_out.copy()
        p_out = b_out.copy()
        sigma_out = b_out.copy()
        res_out = b_out.copy()
        b_out.data, t_out.data, p_out.data, sigma_out.data, res_out.data = (b, t, p, sigma_out, res)

        return {'beta': b_out, 't': t_out, 'p': p_out,
                'sigma': sigma_out, 'residual': res_out}

    def randomise(self, n_permute=5000, threshold_dict=None, return_mask=False, **kwargs):
        """
        Run mass-univariate regression at each voxel with inference performed
        via permutation testing ala randomise in FSL. Operates just like
        .regress(), but intended to be used for second-level analyses.

        Args:
            n_permute (int): number of permutations
            threshold_dict: (dict) a dictionary of threshold parameters
                            {'unc':.001} or {'fdr':.05}
            return_mask: (bool) optionally return the thresholding mask
        Returns:
            out: dictionary of maps for betas, tstats, and pvalues
        """

        if not isinstance(self.X, pd.DataFrame):
            raise ValueError('Make sure self.X is a pandas DataFrame.')

        if self.X.empty:
            raise ValueError('Make sure self.X is not empty.')

        if self.data.shape[0] != self.X.shape[0]:
            raise ValueError("self.X does not match the correct size of "
                             "self.data")

        b, t, p = regress_permutation(self.X, self.data, n_permute=n_permute, **kwargs)

        # Prevent copy of all data in self multiple times; instead start with an empty instance and copy only needed attributes from self, and use this as a template for other outputs
        b_out = self.__class__()
        b_out.mask = deepcopy(self.mask)
        b_out.nifti_masker = deepcopy(self.nifti_masker)

        # Use this as template for other outputs before setting data
        t_out = b_out.copy()
        p_out = b_out.copy()
        b_out.data, t_out.data, p_out.data = (b, t, p)

        if threshold_dict is not None:
            if isinstance(threshold_dict, dict):
                if 'unc' in threshold_dict:
                    thr = threshold_dict['unc']
                elif 'fdr' in threshold_dict:
                    thr = fdr(p_out.data, q=threshold_dict['fdr'])
                elif 'holm-bof' in threshold_dict:
                    thr = holm_bonf(p.data, alpha=threshold_dict['holm-bonf'])
                elif 'permutation' in threshold_dict:
                    thr = .05
                if return_mask:
                    thr_t_out, thr_mask = threshold(t_out, p_out, thr, True)
                    out = {'beta': b_out, 't': t_out, 'p': p_out, 'thr_t': thr_t_out, 'thr_mask': thr_mask}
                else:
                    thr_t_out = threshold(t_out, p_out, thr)
                    out = {'beta': b_out, 't': t_out, 'p': p_out, 'thr_t': thr_t_out}
            else:
                raise ValueError("threshold_dict is not a dictionary. "
                                 "Make sure it is in the form of {'unc': .001} "
                                 "or {'fdr': .05}")
        else:
            out = {'beta': b_out, 't': t_out, 'p': p_out}

        return out

    def ttest(self, threshold_dict=None, return_mask=False):
        """ Calculate one sample t-test across each voxel (two-sided)

        Args:
            threshold_dict: (dict) a dictionary of threshold parameters
                            {'unc':.001} or {'fdr':.05} or {'permutation':tcfe,
                            n_permutation:5000}
            return_mask: (bool) if thresholding is requested, optionall return the mask of voxels that exceed threshold, e.g. for use with another map

        Returns:
            out: (dict) dictionary of regression statistics in Brain_Data
                 instances {'t','p'}

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

                if threshold_dict['permutation'] == 'tfce':
                    perm_threshold = dict(start=0, step=0.2)
                else:
                    perm_threshold = None

                if 'stat_fun' in threshold_dict:
                    stat_fun = threshold_dict['stat_fun']
                else:
                    stat_fun = mne_stats.ttest_1samp_no_p

                t.data, clusters, p_values, _ = mne_stats.spatio_temporal_cluster_1samp_test(
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
                elif 'holm-bonf' in threshold_dict:
                    thr = holm_bonf(p.data, alpha=threshold_dict['holm-bonf'])
                elif 'permutation' in threshold_dict:
                    thr = .05
                if return_mask:
                    thr_t, thr_mask = threshold(t, p, thr, True)
                    out = {'t': t, 'p': p, 'thr_t': thr_t, 'thr_mask': thr_mask}
                else:
                    thr_t = threshold(t, p, thr)
                    out = {'t': t, 'p': p, 'thr_t': thr_t}
            else:
                raise ValueError("threshold_dict is not a dictionary. "
                                 "Make sure it is in the form of {'unc': .001} "
                                 "or {'fdr': .05}")
        else:
            out = {'t': t, 'p': p}

        return out

    def append(self, data, **kwargs):
        """ Append data to Brain_Data instance

        Args:
            data: (Brain_Data) Brain_Data instance to append
            kwargs: optional inputs to Design_Matrix append

        Returns:
            out: (Brain_Data) new appended Brain_Data instance
        """

        data = check_brain_data(data)

        if self.isempty():
            out = deepcopy(data)
        else:
            error_string = ("Data to append has different number of voxels "
                            "then Brain_Data instance.")
            if len(self.shape()) == 1 & len(data.shape()) == 1:
                if self.shape()[0] != data.shape()[0]:
                    raise ValueError(error_string)
            elif len(self.shape()) == 1 & len(data.shape()) > 1:
                if self.shape()[0] != data.shape()[1]:
                    raise ValueError(error_string)
            elif len(self.shape()) > 1 & len(data.shape()) == 1:
                if self.shape()[1] != data.shape()[0]:
                    raise ValueError(error_string)
            elif self.shape()[1] != data.shape()[1]:
                raise ValueError(error_string)
            out = deepcopy(self)
            out.data = np.vstack([self.data, data.data])
            if out.Y.size:
                out.Y = self.Y.append(data.Y)
            if self.X.size:
                if isinstance(self.X, pd.DataFrame):
                    out.X = self.X.append(data.X, **kwargs)
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
                image: (Brain_Data, nifti)  image to evaluate similarity
                method: (str) Type of similarity
                        ['correlation','dot_product','cosine']
            Returns:
                pexp: (list) Outputs a vector of pattern expression values

        """

        image = check_brain_data(image)

        # Check to make sure masks are the same for each dataset and if not
        # create a union mask
        # This might be handy code for a new Brain_Data method
        if np.sum(self.nifti_masker.mask_img.get_data() == 1) != np.sum(image.nifti_masker.mask_img.get_data() == 1):
            new_mask = intersect_masks([self.nifti_masker.mask_img,
                                        image.nifti_masker.mask_img],
                                       threshold=1, connected=False)
            new_nifti_masker = NiftiMasker(mask_img=new_mask)
            data2 = new_nifti_masker.fit_transform(self.to_nifti())
            image2 = new_nifti_masker.fit_transform(image.to_nifti())
        else:
            data2 = self.data
            image2 = image.data

        def vector2array(data):
            if len(data.shape) == 1:
                return data.reshape(-1, 1).T
            else:
                return data

        def flatten_array(data):
            if np.any(np.array(data.shape) == 1):
                data = data.flatten()
                if len(data) == 1 & data.shape[0] == 1:
                    data = data[0]
                return data
            else:
                return data

        # Calculate pattern expression
        if method == 'dot_product':
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
        elif method == 'correlation':
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
        elif method == 'cosine':
            image2 = vector2array(image2)
            data2 = vector2array(data2)
            if image2.shape[1] > 1:
                pexp = []
                for i in range(image2.shape[0]):
                    pexp.append(cosine_similarity(image2[i, :].reshape(-1, 1).T, data2).flatten())
                pexp = np.array(pexp)
            else:
                pexp = cosine_similarity(image2, data2).flatten()
        else:
            raise ValueError('Method must be one of: correlation, dot_product, cosine')
        return flatten_array(pexp)

    def distance(self, metric='euclidean', **kwargs):
        """ Calculate distance between images within a Brain_Data() instance.

            Args:
                metric: (str) type of distance metric (can use any scikit learn or
                        sciypy metric)

            Returns:
                dist: (Adjacency) Outputs a 2D distance matrix.

        """

        return Adjacency(pairwise_distances(self.data, metric=metric, **kwargs),
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

        images = check_brain_data(images)

        # Check to make sure masks are the same for each dataset and if not create a union mask
        # This might be handy code for a new Brain_Data method
        if np.sum(self.nifti_masker.mask_img.get_data() == 1) != np.sum(images.nifti_masker.mask_img.get_data() == 1):
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
        if method == 'ols':
            b = np.dot(np.linalg.pinv(image2), data2)
            res = data2 - np.dot(image2, b)
            sigma = np.std(res, axis=0)
            stderr = np.dot(np.matrix(np.diagonal(np.linalg.inv(np.dot(image2.T,
                                                                       image2)))**.5).T, np.matrix(sigma))
            t_out = b / stderr
            df = image2.shape[0]-image2.shape[1]
            p = 2*(1-t_dist.cdf(np.abs(t_out), df))
        else:
            raise NotImplementedError

        return {'beta': b, 't': t_out, 'p': p, 'df': df, 'sigma': sigma,
                'residual': res}

    def predict(self, algorithm=None, cv_dict=None, plot=True, **kwargs):
        """ Run prediction

        Args:
            algorithm: Algorithm to use for prediction.  Must be one of 'svm',
                    'svr', 'linear', 'logistic', 'lasso', 'ridge',
                    'ridgeClassifier','pcr', or 'lassopcr'
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
        predictor = predictor_settings['predictor']

        # Overall Fit for weight map
        predictor.fit(self.data, np.ravel(output['Y']))
        output['yfit_all'] = predictor.predict(self.data)
        if predictor_settings['prediction_type'] == 'classification':
            if predictor_settings['algorithm'] not in ['svm', 'ridgeClassifier',
                                                       'ridgeClassifierCV']:
                output['prob_all'] = predictor.predict_proba(self.data)
            else:
                output['dist_from_hyperplane_all'] = predictor.decision_function(self.data)
                if predictor_settings['algorithm'] == 'svm' and predictor.probability:
                    output['prob_all'] = predictor.predict_proba(self.data)

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
        from sklearn.base import clone
        if cv_dict is not None:
            cv = set_cv(Y=self.Y, cv_dict=cv_dict)

            predictor_cv = predictor_settings['predictor']
            output['yfit_xval'] = output['yfit_all'].copy()
            output['intercept_xval'] = []
            # Multi-class classification, init weightmaps as list
            if ((predictor_settings['prediction_type'] == 'classification') and (len(np.unique(self.Y)) > 2)):
                output['weight_map_xval'] = []
            else:
                # Otherwise we'll have a single weightmap
                output['weight_map_xval'] = output['weight_map'].copy()
            output['cv_idx'] = []
            wt_map_xval = []

            # Initialize zero'd arrays that will be filled during cross-validation and fitting
            # These will need change shape if doing multi-class or probablistic predictions
            if (predictor_settings['algorithm'] == 'logistic') or (predictor_settings['algorithm'] == 'svm' and predictor.probability):
                # If logistic or svm prob, probs == number of classes
                probs_init = np.zeros((len(self.Y), len(np.unique(self.Y))))
            # however if num classes == 2 decision function == 1, but if num class > 2, decision function == num classes (sklearn weirdness)
            if len(np.unique(self.Y)) == 2:
                dec_init = np.zeros(len(self.Y))
            else:
                dec_init = np.zeros((len(self.Y), len(np.unique(self.Y))))
            # else:
            #
            #     if len(np.unique(self.Y)) == 2:
            #         dec_init = np.zeros(len(self.Y))
            #     else:
            #         dec_init = np.zeros((len(self.Y), len(np.unique(self.Y))))

            if predictor_settings['prediction_type'] == 'classification':
                if predictor_settings['algorithm'] not in ['svm', 'ridgeClassifier', 'ridgeClassifierCV']:
                    output['prob_xval'] = probs_init
                else:
                    output['dist_from_hyperplane_xval'] = dec_init
                    if predictor_settings['algorithm'] == 'svm' and predictor_cv.probability:
                        output['prob_xval'] = probs_init

            for train, test in cv:
                # Ensure estimators are always indepedent across folds
                predictor_cv = clone(predictor_settings['predictor'])
                predictor_cv.fit(self.data[train], np.ravel(self.Y.iloc[train]))
                output['yfit_xval'][test] = predictor_cv.predict(self.data[test]).ravel()
                if predictor_settings['prediction_type'] == 'classification':
                    if predictor_settings['algorithm'] not in ['svm', 'ridgeClassifier', 'ridgeClassifierCV']:
                        output['prob_xval'][test] = predictor_cv.predict_proba(self.data[test])
                    else:
                        output['dist_from_hyperplane_xval'][test] = predictor_cv.decision_function(self.data[test])
                        if predictor_settings['algorithm'] == 'svm' and predictor_cv.probability:
                            output['prob_xval'][test] = predictor_cv.predict_proba(self.data[test])
                # Intercept
                if predictor_settings['algorithm'] == 'pcr':
                    output['intercept_xval'].append(predictor_settings['_regress'].intercept_)
                elif predictor_settings['algorithm'] == 'lassopcr':
                    output['intercept_xval'].append(predictor_settings['_lasso'].intercept_)
                else:
                    output['intercept_xval'].append(predictor_cv.intercept_)
                output['cv_idx'].append((train, test))

                # Weight map
                # Multi-class classification, weightmaps as list
                if ((predictor_settings['prediction_type'] == 'classification') and (len(np.unique(self.Y)) > 2)):
                    tmp = output['weight_map'].empty()
                    tmp.data = predictor_cv.coef_.squeeze()
                    output['weight_map_xval'].append(tmp)
                # Regression or binary classification
                else:
                    if predictor_settings['algorithm'] == 'lassopcr':
                        wt_map_xval.append(np.dot(predictor_settings['_pca'].components_.T, predictor_settings['_lasso'].coef_))
                    elif predictor_settings['algorithm'] == 'pcr':
                        wt_map_xval.append(np.dot(predictor_settings['_pca'].components_.T, predictor_settings['_regress'].coef_))
                    else:
                        wt_map_xval.append(predictor_cv.coef_.squeeze())
                    output['weight_map_xval'].data = np.array(wt_map_xval)

        # Print Results
        if predictor_settings['prediction_type'] == 'classification':
            output['mcr_all'] = balanced_accuracy_score(self.Y.values, output['yfit_all'])
            print('overall accuracy: %.2f' % output['mcr_all'])
            if cv_dict is not None:
                output['mcr_xval'] = np.mean(output['yfit_xval'] == np.array(self.Y).flatten())
                print('overall CV accuracy: %.2f' % output['mcr_xval'])
        elif predictor_settings['prediction_type'] == 'prediction':
            output['rmse_all'] = np.sqrt(np.mean((output['yfit_all']-output['Y'])**2))
            output['r_all'] = pearsonr(output['Y'], output['yfit_all'])[0]
            print('overall Root Mean Squared Error: %.2f' % output['rmse_all'])
            print('overall Correlation: %.2f' % output['r_all'])
            if cv_dict is not None:
                output['rmse_xval'] = np.sqrt(np.mean((output['yfit_xval']-output['Y'])**2))
                output['r_xval'] = pearsonr(output['Y'], output['yfit_xval'])[0]
                print('overall CV Root Mean Squared Error: %.2f' % output['rmse_xval'])
                print('overall CV Correlation: %.2f' % output['r_xval'])

        # Plot
        if plot:
            if cv_dict is not None:
                if predictor_settings['prediction_type'] == 'prediction':
                    scatterplot(pd.DataFrame({'Y': output['Y'], 'yfit_xval': output['yfit_xval']}))
                elif predictor_settings['prediction_type'] == 'classification':
                    if len(np.unique(self.Y)) > 2:
                        print('Skipping ROC plot because num_classes > 2')
                    else:
                        if predictor_settings['algorithm'] not in ['svm', 'ridgeClassifier', 'ridgeClassifierCV']:
                            output['roc'] = Roc(input_values=output['prob_xval'][:,1], binary_outcome=output['Y'].astype('bool'))
                        else:
                            output['roc'] = Roc(input_values=output['dist_from_hyperplane_xval'], binary_outcome=output['Y'].astype('bool'))
                            if predictor_settings['algorithm'] == 'svm' and predictor_cv.probability:
                                output['roc'] = Roc(input_values=output['prob_xval'][:, 1], binary_outcome=output['Y'].astype('bool'))
                        output['roc'].plot()
            output['weight_map'].plot()

        return output

    def predict_multi(self, algorithm=None, cv_dict=None, method='searchlight', rois=None, process_mask=None, radius=2.0, scoring=None, n_jobs=1, verbose=0, **kwargs):
        """ Perform multi-region prediction. This can be a searchlight analysis or multi-roi analysis if provided a Brain_Data instance with labeled non-overlapping rois.

        Args:
            algorithm (string): algorithm to use for prediction Must be one of 'svm',
                    'svr', 'linear', 'logistic', 'lasso', 'ridge',
                    'ridgeClassifier','pcr', or 'lassopcr'
            cv_dict: Type of cross_validation to use. Default is 3-fold. A dictionary of
                    {'type': 'kfolds', 'n_folds': n},
                    {'type': 'kfolds', 'n_folds': n, 'stratified': Y},
                    {'type': 'kfolds', 'n_folds': n, 'subject_id': holdout}, or
                    {'type': 'loso', 'subject_id': holdout}
                    where 'n' = number of folds, and 'holdout' = vector of
                    subject ids that corresponds to self.Y
            method (string): one of 'searchlight' or 'roi'
            rois (string/nltools.Brain_Data): nifti file path or Brain_data instance containing non-overlapping regions-of-interest labeled by integers
            process_mask (nib.Nifti1Image/nltools.Brain_Data): mask to constrain where to perform analyses; only applied if method = 'searchlight'
            radius (float): radius of searchlight in mm; default 2mm
            scoring (function): callable scoring function; see sklearn documentation; defaults to estimator's default scoring function
            n_jobs (int): The number of CPUs to use to do permutation; default 1 because this can be very memory intensive
            verbose (int): whether parallelization progress should be printed; default 0

        Returns:
            output: image of results

        """

        if method not in ['searchlight', 'rois']:
            raise ValueError("method must be one of 'searchlight' or 'roi'")
        if method == 'roi' and rois is None:
            raise ValueError("With method = 'roi' a file path, or nibabel/nltools instance with roi labels must be provided")

        # Set algorithm
        if algorithm is not None:
            predictor_settings = set_algorithm(algorithm, **kwargs)
        else:
            # Use SVR as a default
            predictor_settings = set_algorithm('svr', **{'kernel': "linear"})
        estimator = predictor_settings['predictor']

        if cv_dict is not None:
            cv = set_cv(Y=self.Y, cv_dict=cv_dict, return_generator=False)
            if cv_dict['type'] == 'loso':
                groups = cv_dict['subject_id']
            else:
                groups = None
        else:
            cv = None
            groups = None

        if method == 'rois':
            if isinstance(rois, six.string_types):
                if os.path.isfile(rois):
                    rois_img = Brain_Data(rois, mask=self.mask)
            elif isinstance(rois, Brain_Data):
                rois_img = rois.copy()
            else:
                raise TypeError("rois must be a file path or a Brain_Data instance")
            if len(rois_img.shape()) == 1:
                rois_img = expand_mask(rois_img, custom_mask=self.mask)
            if len(rois_img.shape()) != 2:
                raise ValueError("rois cannot be coerced into a mask. Make sure nifti file or Brain_Data is 3d with non-overlapping integer labels or 4d with non-overlapping boolean masks")

            out = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(_roi_func)(self, r, algorithm, cv_dict, **kwargs) for r in rois_img)

        elif method == 'searchlight':
            # Searchlight
            if process_mask is None:
                process_mask_img = None
            elif isinstance(process_mask, nib.Nifti1Image):
                process_mask_img = process_mask
            elif isinstance(process_mask, Brain_Data):
                process_mask_img = process_mask.to_nifti()
            elif isinstance(process_mask, six.string_types):
                if os.path.isfile(process_mask):
                    process_mask_img = nib.load(process_mask)
                else:
                    raise ValueError("process mask file path specified but can't be found")
            else:
                raise TypeError("process_mask is not a valid nibabel instance, Brain_Data instance or file path")

            sl = SearchLight(mask_img=self.mask, process_mask_img=process_mask_img, estimator=estimator, n_jobs=n_jobs, scoring=scoring, cv=cv, verbose=verbose, radius=radius)
            in_image = self.to_nifti()
            sl.fit(in_image, self.Y, groups=groups)
            out = nib.Nifti1Image(sl.scores_, affine=self.nifti_masker.affine_)
            out = Brain_Data(out, mask=self.mask)
        return out

    def apply_mask(self, mask, resample_mask_to_brain=False):
        """ Mask Brain_Data instance

        Note target data will be resampled into the same space as the mask. If you would like the mask
        resampled into the Brain_Data space, then set resample_mask_to_brain=True.

        Args:
            mask: (Brain_Data or nifti object) mask to apply to Brain_Data object.
            resample_mask_to_brain: (bool) Will resample mask to brain space before applying mask (default=False).

        Returns:
            masked: (Brain_Data) masked Brain_Data object

        """

        masked = deepcopy(self)
        mask = check_brain_data(mask)
        if not check_brain_data_is_single(mask):
            raise ValueError('Mask must be a single image')

        if check_brain_data_is_single(self):
            n_vox = len(self)
        else:
            n_vox = self.shape()[1]

        if resample_mask_to_brain: 
            mask = resample_to_img(mask.to_nifti(), masked.to_nifti())
            mask = check_brain_data(mask, masked.mask)

        nifti_masker = NiftiMasker(mask_img=mask.to_nifti()).fit()

        if n_vox == len(mask):
            if check_brain_data_is_single(masked):
                masked.data = masked.data[mask.data.astype(bool)]
            else:
                masked.data = masked.data[:, mask.data.astype(bool)]
            masked.nifti_masker = nifti_masker
        else:
            masked.data = nifti_masker.fit_transform(masked.to_nifti())
            masked.nifti_masker = nifti_masker
        if (len(masked.shape()) > 1) & (masked.shape()[0] == 1):
            masked.data = masked.data.flatten()
        return masked

    def extract_roi(self, mask, metric='mean', n_components=None):
        """ Extract activity from mask

        Args:
            mask: (nifti) nibabel mask can be binary or numbered for
                  different rois
            metric: type of extraction method ['mean', 'median', 'pca'], (default=mean)
                    NOTE: Only mean currently works!
            n_components: if metric='pca', number of components to return (takes any input into sklearn.Decomposition.PCA)

        Returns:
            out: mean within each ROI across images

        """

        metrics = ['mean','median','pca']

        mask = check_brain_data(mask)
        ma = mask.copy()

        if metric not in metrics:
            raise NotImplementedError

        if len(np.unique(ma.data)) == 2:
            masked = self.apply_mask(ma)
            if check_brain_data_is_single(masked):
                if metric == 'mean':
                    out = masked.mean()
                elif metric == 'median':
                    out = masked.median()
                else:
                    raise ValueError('Not possible to run PCA on a single image')
            else:
                if metric == 'mean':
                    out = masked.mean(axis=1)
                elif metric == 'median':
                    out = masked.median(axis=1)
                else:
                    output = masked.decompose(algorithm='pca', n_components=n_components, axis='images')
                    out = output['weights'].T
        elif len(np.unique(ma.data)) > 2:
            # make sure each ROI id is an integer
            ma.data = np.round(ma.data).astype(int)
            all_mask = expand_mask(ma)
            if check_brain_data_is_single(self):
                if metric == 'mean':
                    out = np.array([self.apply_mask(m).mean() for m in all_mask])
                elif metric == 'median':
                    out = np.array([self.apply_mask(m).median() for m in all_mask])
                else:
                    raise ValueError('Not possible to run PCA on a single image')
            else:
                if metric == 'mean':
                    out = np.array([self.apply_mask(m).mean(axis=1) for m in all_mask])
                elif metric == 'median':
                    out = np.array([self.apply_mask(m).median(axis=1) for m in all_mask])
                else:
                    out = []
                    for m in all_mask:
                        masked = self.apply_mask(m)
                        output = masked.decompose(algorithm='pca', n_components=n_components, axis='images')
                        out.append(output['weights'].T)
        else:
            raise ValueError('Mask must be binary or integers')
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
            ICC: (np.array) intraclass correlation coefficient

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
            type: ('linear','constant', optional) type of detrending

        Returns:
            out: (Brain_Data) detrended Brain_Data instance

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
            access_token: (str, Required) Neurovault api access token
            collection_name: (str, Optional) name of new collection to create
            collection_id: (int, Optional) neurovault collection_id if adding images
                            to existing collection
            img_type: (str, Required) Neurovault map_type
            img_modality: (str, Required) Neurovault image modality

        Returns:
            collection: (pd.DataFrame) neurovault collection information

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

    def filter(self, sampling_freq=None, high_pass=None, low_pass=None, **kwargs):
        ''' Apply 5th order butterworth filter to data. Wraps nilearn
        functionality. Does not default to detrending and standardizing like
        nilearn implementation, but this can be overridden using kwargs.

        Args:
            sampling_freq: sampling freq in hertz (i.e. 1 / TR)
            high_pass: high pass cutoff frequency
            low_pass: low pass cutoff frequency
            kwargs: other keyword arguments to nilearn.signal.clean

        Returns:
            Brain_Data: Filtered Brain_Data instance
        '''

        if sampling_freq is None:
            raise ValueError("Need to provide sampling rate (TR)!")
        if high_pass is None and low_pass is None:
            raise ValueError("high_pass and/or low_pass cutoff must be"
                             "provided!")
        if sampling_freq is None:
            raise ValueError("Need to provide TR!")

        standardize = kwargs.get('standardize', False)
        detrend = kwargs.get('detrend', False)
        out = self.copy()
        out.data = clean(out.data, t_r=1. / sampling_freq, detrend=detrend,
                         standardize=standardize, high_pass=high_pass,
                         low_pass=low_pass, **kwargs)
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

    def standardize(self, axis=0, method='center'):
        ''' Standardize Brain_Data() instance.

        Args:
            axis: 0 for observations 1 for features
            method: ['center','zscore']

        Returns:
            Brain_Data Instance

        '''

        if axis == 1 and len(self.shape()) == 1:
            raise IndexError("Brain_Data is only 3d but standardization was requested over observations")
        out = self.copy()
        if method == 'zscore':
            with_std = True
        elif method == 'center':
            with_std = False
        else:
            raise ValueError('method must be ["center","zscore"')
        out.data = scale(out.data, axis=axis, with_std=with_std)
        return out

    def groupby(self, mask):
        '''Create groupby instance'''
        return Groupby(self, mask)

    def aggregate(self, mask, func):
        '''Create new Brain_Data instance that aggregages func over mask'''
        dat = self.groupby(mask)
        values = dat.apply(func)
        return dat.combine(values)

    def threshold(self, upper=None, lower=None, binarize=False, coerce_nan=True):
        '''Threshold Brain_Data instance. Provide upper and lower values or
           percentages to perform two-sided thresholding. Binarize will return
           a mask image respecting thresholds if provided, otherwise respecting
           every non-zero value.

        Args:
            upper: (float or str) Upper cutoff for thresholding. If string
                    will interpret as percentile; can be None for one-sided
                    thresholding.
            lower: (float or str) Lower cutoff for thresholding. If string
                    will interpret as percentile; can be None for one-sided
                    thresholding.
            binarize (bool): return binarized image respecting thresholds if
                    provided, otherwise binarize on every non-zero value;
                    default False
            coerce_nan (bool): coerce nan values to 0s; default True

        Returns:
            Thresholded Brain_Data object.

        '''

        b = self.copy()

        if coerce_nan:
            b.data = np.nan_to_num(b.data)

        if isinstance(upper, six.string_types):
            if upper[-1] == '%':
                upper = np.percentile(b.data, float(upper[:-1]))

        if isinstance(lower, six.string_types):
            if lower[-1] == '%':
                lower = np.percentile(b.data, float(lower[:-1]))

        if upper and lower:
            b.data[(b.data < upper) & (b.data > lower)] = 0
        elif upper and not lower:
            b.data[b.data < upper] = 0
        elif lower and not upper:
            b.data[b.data > lower] = 0

        if binarize:
            b.data[b.data != 0] = 1
        return b

    def regions(self, min_region_size=1350, extract_type='local_regions',
                smoothing_fwhm=6, is_mask=False):
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
            is_mask (bool): Whether the Brain_Data instance should be treated
                            as a boolean mask and if so, calls
                            connected_label_regions instead.

        Returns:
            Brain_Data: Brain_Data instance with extracted ROIs as data.
        '''

        if is_mask:
            regions, _ = connected_label_regions(self.to_nifti())
        else:
            regions, _ = connected_regions(self.to_nifti(),
                                           min_region_size, extract_type,
                                           smoothing_fwhm)

        return Brain_Data(regions, mask=self.mask)

    def transform_pairwise(self):
        ''' Extract brain connected regions into separate regions.

        Args:

        Returns:
            Brain_Data: Brain_Data instance tranformed into pairwise comparisons
        '''
        out = self.copy()
        out.data, new_Y = transform_pairwise(self.data, self.Y)
        out.Y = pd.DataFrame(new_Y)
        out.Y.replace(-1, 0, inplace=True)
        return out

    def bootstrap(self, function, n_samples=5000, save_weights=False,
                  n_jobs=-1, random_state=None, *args, **kwargs):
        '''Bootstrap a Brain_Data method.

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

        random_state = check_random_state(random_state)
        seeds = random_state.randint(MAX_INT, size=n_samples)

        bootstrapped = Parallel(n_jobs=n_jobs)(
                        delayed(_bootstrap_apply_func)(self,
                                                       function, random_state=seeds[i], *args, **kwargs)
                        for i in range(n_samples))

        if function == 'predict':
            bootstrapped = [x['weight_map'] for x in bootstrapped]
        bootstrapped = Brain_Data(bootstrapped, mask=self.mask)
        return summarize_bootstrap(bootstrapped, save_weights=save_weights)

    def decompose(self, algorithm='pca', axis='voxels', n_components=None,
                  *args, **kwargs):
        ''' Decompose Brain_Data object

        Args:
            algorithm: (str) Algorithm to perform decomposition
                        types=['pca','ica','nnmf','fa','dictionary','kernelpca']
            axis: dimension to decompose ['voxels','images']
            n_components: (int) number of components. If None then retain
                        as many as possible.
        Returns:
            output: a dictionary of decomposition parameters
        '''

        out = {}
        out['decomposition_object'] = set_decomposition_algorithm(
                                                    algorithm=algorithm,
                                                    n_components=n_components,
                                                    *args, **kwargs)
        if axis == 'images':
            out['decomposition_object'].fit(self.data.T)
            out['components'] = self.empty()
            out['components'].data = out['decomposition_object'].transform(
                                                                self.data.T).T
            out['weights'] = out['decomposition_object'].components_.T
        if axis == 'voxels':
            out['decomposition_object'].fit(self.data)
            out['weights'] = out['decomposition_object'].transform(self.data)
            out['components'] = self.empty()
            out['components'].data = out['decomposition_object'].components_
        return out

    def align(self, target, method='procrustes', n_features=None, axis=0,
              *args, **kwargs):
        ''' Align Brain_Data instance to target object

        Can be used to hyperalign source data to target data using
        Hyperalignemnt from Dartmouth (i.e., procrustes transformation; see
        nltools.stats.procrustes) or Shared Response Model from Princeton (see
        nltools.external.srm). (see nltools.stats.align for aligning many data
        objects together). Common Model is shared response model or centered
        target data.Transformed data can be back projected to original data
        using Tranformation matrix.

        Examples:
            Hyperalign using procrustes transform:
                out = data.align(target, method='procrustes')

            Align using shared response model:
                out = data.align(target, method='probabilistic_srm', n_features=None)

            Project aligned data into original data:
                original_data = np.dot(out['transformed'].data,out['transformation_matrix'].T)

        Args:
            target: (Brain_Data) object to align to.
            method: (str) alignment method to use
                ['probabilistic_srm','deterministic_srm','procrustes']
            n_features: (int) number of features to align to common space.
                If None then will select number of voxels
            axis: (int) axis to align on

        Returns:
            out: (dict) a dictionary containing transformed object,
                transformation matrix, and the shared response matrix

        '''

        source = self.copy()
        common = target.copy()

        target = check_brain_data(target)

        if method not in ['probabilistic_srm', 'deterministic_srm', 'procrustes']:
            raise ValueError("Method must be ['probabilistic_srm','deterministic_srm','procrustes']")

        data1 = source.data.T
        data2 = target.data.T

        if axis == 1:
            data1 = data1.T
            data2 = data2.T

        out = dict()
        if method in ['deterministic_srm', 'probabilistic_srm']:
            if n_features is None:
                n_features = data1.shape[0]
            if method == 'deterministic_srm':
                srm = DetSRM(features=n_features, *args, **kwargs)
            elif method == 'probabilistic_srm':
                srm = SRM(features=n_features, *args, **kwargs)
            srm.fit([data1, data2])
            source.data = srm.transform([data1, data2])[0].T
            common.data = srm.s_.T
            out['transformed'] = source
            out['common_model'] = common
            out['transformation_matrix'] = srm.w_[0]
        elif method == 'procrustes':
            if n_features != None:
                raise NotImplementedError('Currently must use all voxels.'
                                          'Eventually will add a PCA'
                                          'reduction, must do this manually'
                                          'for now.')

            mtx1, mtx2, out['disparity'], t, out['scale'] = procrustes(data2.T,
                                                                       data1.T)
            source.data = mtx2
            common.data = mtx1
            out['transformed'] = source
            out['common_model'] = common
            out['transformation_matrix'] = t
        if axis == 1:
            out['transformed'].data = out['transformed'].data.T
            out['common_model'].data = out['common_model'].data.T
        return out

    def smooth(self, fwhm):
        '''Apply spatial smoothing using nilearn smooth_img()

            Args:
                fwhm: (float) full width half maximum of gaussian spatial filter
            Returns:
                Brain_Data instance
        '''
        out = self.copy()
        out.data = out.nifti_masker.fit_transform(smooth_img(self.to_nifti(), fwhm))
        return out

    def find_spikes(self, global_spike_cutoff=3, diff_spike_cutoff=3):
        '''Function to identify spikes from Time Series Data

            Args:
                global_spike_cutoff: (int,None) cutoff to identify spikes in global signal
                                     in standard deviations, None indicates do not calculate.
                diff_spike_cutoff: (int,None) cutoff to identify spikes in average frame difference
                                     in standard deviations, None indicates do not calculate.
            Returns:
                pandas dataframe with spikes as indicator variables
        '''
        return find_spikes(self,
                            global_spike_cutoff=global_spike_cutoff,
                            diff_spike_cutoff=diff_spike_cutoff)


class Groupby(object):
    def __init__(self, data, mask):

        data = check_brain_data(data)
        mask = check_brain_data(mask)

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
        for i in iter(value_dict.keys()):
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
