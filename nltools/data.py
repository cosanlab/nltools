'''
    NeuroLearn Data Classes
    =========================
    Classes to represent various types of fdata

    Author: Luke Chang
    License: MIT
'''

## Notes:
# Might consider moving anatomical field out of object and just request when needed.  Probably only when plotting
# Need to figure out how to speed up loading and resampling of data

__all__ = ['Brain_Data',
            ]

import os
import nibabel as nib
from nltools.utils import get_resource_path
from nilearn.input_data import NiftiMasker
from copy import deepcopy
import pandas as pd
import numpy as np
from nilearn.plotting.img_plotting import plot_epi, plot_roi
from scipy.stats import ttest_1samp
from scipy.stats import t


class Brain_Data:

    def __init__(self, data=None, Y=None, X=None, mask=None, output_file=None, anatomical=None, **kwargs):
        """ Initialize Brain_Data Instance.

        Args:
            data: nibabel data instance or list of files
            Y: vector of training labels
            X: Pandas DataFrame Design Matrix for running univariate models 
            mask: binary nifiti file to mask brain data
            output_file: Name to write out to nifti file
            anatomical: anatomical image to overlay plots
            **kwargs: Additional keyword arguments to pass to the prediction algorithm

        """

        if mask is not None:
            if type(mask) is not nib.nifti1.Nifti1Image:
                raise ValueError("mask is not a nibabel instance")
            self.mask = mask
        else:
            self.mask = nib.load(os.path.join(get_resource_path(),'MNI152_T1_2mm_brain_mask.nii.gz'))

        if anatomical is not None:
            if type(anatomical) is not nib.nifti1.Nifti1Image:
                raise ValueError("anatomical is not a nibabel instance")
            self.anatomical = anatomical
        else:
            self.anatomical = nib.load(os.path.join(get_resource_path(),'MNI152_T1_2mm.nii.gz'))

        if type(data) is str:
            data=nib.load(data)
        elif type(data) is list:
            data=nib.concat_images(data)
        elif ~isinstance(data, nib.Nifti1Image):
            raise ValueError("data is not a nibabel instance")

        self.nifti_masker = NiftiMasker(mask_img=mask)
        self.data = self.nifti_masker.fit_transform(data)

        if Y is not None:
            if type(Y) is str:
                if os.path.isfile(Y):
                    Y=pd.read_csv(os.path.join(output_dir,'y.csv'),header=None,index_col=None)
                    Y=np.array(Y)[0]
            elif type(Y) is list:
                Y=np.array(Y)
            if self.data.shape[0]!= len(Y):
                raise ValueError("Y does not match the correct size of data")
            self.Y = Y
        else:
            self.Y = []

        if X is not None:
            if self.data.shape[0]!= X.shape[0]:
                raise ValueError("X does not match the correct size of data")
            self.X = X
        else:
            self.X = pd.DataFrame()

        if output_file is not None:
            self.file_name = output_file
        else:
            self.file_name = []

    def __getitem__(self, i):
        new = deepcopy(self)
        new.data = np.array(new.data[i,:]).flatten()
        return new

    # def __repr__(self):
    #   return '%s.%s(data=%size, Y=%s, X=%s, output_file=%s)' % (
    #      self.__class__.__module__,
    #      self.__class__.__name__,
    #      self.data.shape,
    #      self.Y.shape,
    #      self.X.shape,
    #      self.file_name,
    #   )

    def shape(self):
        """ Get images by voxels shape.

        Args:
            self: Brain_Data instance

        """

        return self.data.shape

    def mean(self):
        """ Get mean of each voxel across images.

        Args:
            self: Brain_Data instance

        Returns:
            out: Brain_Data instance
        
        """ 

        out = deepcopy(self)
        out.data = np.mean(out.data, axis=0)
        return out

    def std(self):
        """ Get standard deviation of each voxel across images.

        Args:
            self: Brain_Data instance

        Returns:
            out: Brain_Data instance
        
        """ 

        out = deepcopy(self)
        out.data = np.std(out.data, axis=0)
        return out

    def to_nifti(self):
        """ Convert Brain_Data Instance into Nifti Object

        Args:
            self: Brain_Data instance
        
        """
        
        nifti_dat = self.nifti_masker.inverse_transform(self.data) #add noise scaled by sigma
        return nifti_dat

    def write(self, file_name=None):
        """ Write out Brain_Data object to Nifti File.

        Args:
            self: Brain_Data instance
            file_name: name of nifti file

        """

        to_nifti(self).write(file_name)

    def plot(self, limit=5):
        """ Create a quick plot of self.data.  Will plot each image separately

        Args:
            self: Brain_Data instance
            limit: max number of images to return
            mask: Binary nifti mask to calculate mean

        """

        if self.data.ndim == 1:
            plot_roi(self.to_nifti(), self.anatomical)
        else:
            for i in xrange(self.data.shape[0]):
                if i < limit:
                    plot_roi(self.nifti_masker.inverse_transform(self.data[i,:]), self.anatomical)


    def regress(self):
        """ run vectorized OLS regression across voxels.

        Args:
            self: Brain_Data instance

        Returns:
            out: dictionary of regression statistics in Brain_Data instances {'beta','t','p','df','residual'}
        
        """ 

        if not isinstance(self.X, pd.DataFrame):
            raise ValueError('Make sure self.X is a pandas DataFrame.')

        if self.X.empty:
            raise ValueError('Make sure self.X is not empty.')

        if self.data.shape[0]!= self.X.shape[0]:
            raise ValueError("self.X does not match the correct size of self.data")

        b = np.dot(np.linalg.pinv(self.X), self.data)
        res = self.data - np.dot(self.X,b)
        sigma = np.std(res,axis=0)
        stderr = np.dot(np.matrix(np.diagonal(np.linalg.inv(np.dot(self.X.T,self.X)))**.5).T,np.matrix(sigma))
        b_out = deepcopy(self)
        b_out.data = b
        t_out = deepcopy(self)
        t_out.data = b /stderr
        df = np.array([self.X.shape[0]-self.X.shape[1]] * t_out.data.shape[1])
        p_out = deepcopy(self)
        p_out.data = 2*(1-t.cdf(np.abs(t_out.data),df))

 
        # Might want to not output this info
        df_out = deepcopy(self)
        df_out.data = df
        sigma_out = deepcopy(self)
        sigma_out.data = sigma
        res_out = deepcopy(self)
        res_out.data = res

        return {'beta':b_out, 't':t_out, 'p':p_out, 'df':df_out, 'sigma':sigma_out, 'residual':res_out}

    def ttest(self, threshold_dict=None):
        """ Calculate one sample t-test across each voxel (two-sided)

        Args:
            self: Brain_Data instance
            threshold_dict: a dictionary of threshold parameters {'unc':.001} or {'fdr':.05}

        Returns:
            out: dictionary of regression statistics in Brain_Data instances {'t','p'}
        
        """ 

        # Notes:  Need to add FDR Option

        t = deepcopy(self)
        p = deepcopy(self)
        t.data, p.data = ttest_1samp(self.data, 0, 0)

        if threshold_dict is not None:
            if type(threshold_dict) is dict:
                if 'unc' in threshold_dict:
                    #Uncorrected Thresholding
                    t.data[np.where(p.data>threshold_dict['unc'])] = np.nan
                elif 'fdr' in threshold_dict:
                    pass
            else:
                raise ValueError("threshold_dict is not a dictionary.  Make sure it is in the form of {'unc':.001} or {'fdr':.05}")

        out = {'t':t, 'p':p}

        return out

    def resample(self, target):
        """ Resample data into target space

        Args:
            self: Brain_Data instance
            target: Brain_Data instance of target space
        
        """ 

    if not isinstance(target, Brain_Data):
        raise ValueError('Make sure target is a Brain_Data instance')
 
    pass

