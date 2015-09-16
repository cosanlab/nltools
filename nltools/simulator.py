# Sam Greydanus and Luke Chang 2015
# Some code taken from nilearn searchlight implementation: https://github.com/nilearn/nilearn/blob/master/nilearn/decoding/searchlight.py

import os

import time
import sys
import warnings
from distutils.version import LooseVersion
import random

import cPickle 
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import plotting

import nibabel as nib

import sklearn
from sklearn import neighbors
from sklearn.externals.joblib import Parallel, delayed, cpu_count
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.base import BaseEstimator
from sklearn import neighbors
from sklearn.svm import SVR

from nilearn import masking
from nilearn.input_data import NiftiMasker

from scipy.stats import multivariate_normal

from nltools.analysis import Predict
import glob

class Simulator:
    def __init__(self, brain_mask=None, output_dir = None): #no scoring param
        self.resource_folder = os.path.join(os.getcwd(),'resources')
        if output_dir is None:
            self.output_dir = os.path.join(os.getcwd())
        else:
            self.output_dir = output_dir
        
        if type(brain_mask) is str:
            brain_mask = nib.load(brain_mask)
        elif brain_mask is None:
            brain_mask = nib.load(os.path.join(self.resource_folder,'MNI152_T1_2mm_brain_mask_dil.nii.gz'))
        elif type(brain_mask) is not nib.nifti1.Nifti1Image:
            print(brain_mask)
            print(type(brain_mask))
            raise ValueError("brain_mask is not a string or a nibabel instance")
        self.brain_mask = brain_mask
        self.nifti_masker = NiftiMasker(mask_img=self.brain_mask)


    def gaussian(self, mu, sigma, i_tot):
        x, y, z = np.mgrid[0:self.brain_mask.shape[0], 0:self.brain_mask.shape[1], 0:self.brain_mask.shape[2]]
        
        # Need an (N, 3) array of (x, y) pairs.
        xyz = np.column_stack([x.flat, y.flat, z.flat])

        covariance = np.diag(sigma**2)
        g = multivariate_normal.pdf(xyz, mean=mu, cov=covariance)

        # Reshape back to a 3D grid.
        g = g.reshape(x.shape).astype(float)
        
        #select only the regions within the brain mask
        g = np.multiply(self.brain_mask.get_data(),g)
        #adjust total intensity of gaussian
        g = np.multiply(i_tot/np.sum(g),g)

        return g

    def sphere(self, r, p):
        dims = self.brain_mask.shape

        x, y, z = np.ogrid[-p[0]:dims[0]-p[0], -p[1]:dims[1]-p[1], -p[2]:dims[2]-p[2]]
        mask = x*x + y*y + z*z <= r*r

        activation = np.zeros(dims)
        activation[mask] = 1
        activation = nib.Nifti1Image(activation, affine=np.eye(4))
        
        #return the 3D numpy matrix of zeros containing the sphere as a region of ones
        return activation.get_data()

    def normal_noise(self, mu, sigma):
        vmask = self.nifti_masker.fit_transform(self.brain_mask)
        
        vlength = np.sum(self.brain_mask.get_data())
        n = np.random.normal(mu, sigma, vlength)
        m = self.nifti_masker.inverse_transform(n)

        #return the 3D numpy matrix of zeros containing the brain mask filled with noise produced over a normal distribution
        return m.get_data()

    def to_nifti(self, m):
        if not (type(m) == np.ndarray and len(v.shape) == 3):
            raise ValueError("ERROR: need 3D np.ndarray matrix to create the nifti file")
        ni = nib.Nifti1Image(m, affine=np.eye(4))
        return ni


        
    # def getnifti(self, mu, sigma, i_tot=1):
    #     # coordinates of center of gaussian (center in brain mask)
    #     if mu == 'center':
    #         mu = np.array([self.brain_mask.shape[0]/2.0, self.brain_mask.shape[1]/2.0, self.brain_mask.shape[2]/2.0])

    #     g = self.gaussian(mu, sigma, i_tot)

    #     n = nib.Nifti1Image(g, affine=np.eye(4))
    #     n.to_filename(os.path.join(os.getcwd(),'data_3D.nii.gz'))

    #     return n
