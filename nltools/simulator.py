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
from nltools.utils import get_resource_path
import glob
import csv

class Simulator:
    def __init__(self, brain_mask=None, output_dir = None): #no scoring param
        # self.resource_folder = os.path.join(os.getcwd(),'resources')
        if output_dir is None:
            self.output_dir = os.path.join(os.getcwd())
        else:
            self.output_dir = output_dir
        
        if type(brain_mask) is str:
            brain_mask = nib.load(brain_mask)
        elif brain_mask is None:
            brain_mask = nib.load(os.path.join(get_resource_path(),'MNI152_T1_2mm_brain_mask.nii.gz'))
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
        activation = np.multiply(activation, self.brain_mask.get_data())
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
        if not (type(m) == np.ndarray and len(m.shape) == 4): #try 4D
        # if not (type(m) == np.ndarray and len(m.shape) == 3):
            raise ValueError("ERROR: need 3D np.ndarray matrix to create the nifti file")
        m = m.astype(np.float32)
        ni = nib.Nifti1Image(m, affine=self.brain_mask.affine)
        return ni

    def n_spheres(self, r, p_list):
        #initialize useful values
        dims = self.brain_mask.get_data().shape
        
        #generate and sum spheres of 0's and 1's
        A = np.zeros_like(self.brain_mask.get_data())
        for p in p_list:
            A = np.add(A, self.sphere(r, p))
        
        return A

    def collection_from_pattern(self, A, sigma, reps = 1, I = None, output_dir = None):
            if I is None:
                I = [sigma/10.0]

            levels = len(I)
            temp = I
            for i in xrange(reps - 1):
                I = I + temp
            
            #initialize useful values
            dims = self.brain_mask.get_data().shape
            
            #for each intensity
            A_list = []
            for i in I:
                A_list.append(np.multiply(A, i))

            #generate a different gaussian noise profile for each mask
            mu = 0 #values centered around 0
            N_list = []
            for i in xrange(len(I)):
                N_list.append(self.normal_noise(mu, sigma))
            
            #add noise and signal together, then convert to nifti files
            NF_list = []
            for i in xrange(len(I)):
                NF_list.append(self.to_nifti(np.add(N_list[i],A_list[i]) ))
                
            if output_dir is not None:
                if type(output_dir) is str:
                    for i in xrange(len(I)):
                        NF_list[i].to_filename(os.path.join(output_dir,'centered_sphere_' + str(i) + "_" + str(i%levels) + '.nii.gz'))
                else:
                    raise ValueError("ERROR. output_dir must be a string")
            
            return (NF_list, I)

    def collection_of_centered_spheres(self, r, sigma, reps = 1, I = None, output_dir = None):
        dims = self.brain_mask.get_data().shape
        p = [dims[0]/2, dims[1]/2, dims[2]/2]
        A = self.sphere(r, p)

        c = self.collection_from_pattern(A, sigma, reps = reps, I = I, output_dir = output_dir)

        return c

    def create_data(self, y, sigma, radius = 5, reps = 1, output_dir = None):
        """ create simulated data

        Args:
            y: vector of intensities or class labels
            sigma: amount of noise to add
            radius: vector of radius.  Will create multiple spheres if len(radius) > 1
            reps: number of data repetitions useful for trials or subjects 
            reps: number of data repetitions
            output_dir: string path of directory to output data.  If None, no data will be written
            **kwargs: Additional keyword arguments to pass to the prediction algorithm

        """

        # Create reps
        levels = len(y)
        temp = y
        rep_id = [1] * len(temp)
        for i in xrange(reps - 1):
            y = y + temp
            rep_id.extend([i+2] * len(temp))
        
        #initialize useful values
        dims = self.brain_mask.get_data().shape
        
        # Initialize Spheres
        if type(radius) is int:
            p = [dims[0]/2, dims[1]/2, dims[2]/2]
            A = self.sphere(radius, p)
        else:
            raise ValueError("More than one sphere not implemented yet.")

        #for each intensity
        A_list = []
        for i in y:
            A_list.append(np.multiply(A, i))

        #generate a different gaussian noise profile for each mask
        mu = 0 #values centered around 0
        N_list = []
        for i in xrange(len(y)):
            N_list.append(self.normal_noise(mu, sigma))
        
        #add noise and signal together, then convert to nifti files
        NF_list = []
        for i in xrange(len(y)):
            NF_list.append(self.to_nifti(np.add(N_list[i],A_list[i]) ))
        
        # Assign variables to object
        self.data = NF_list
        self.y = y
        self.rep_id = rep_id

        # Write Data to files if requested
        if output_dir is not None:
            if type(output_dir) is str:
                for i in xrange(len(y)):
                    NF_list[i].to_filename(os.path.join(output_dir,'centered_sphere_' + str(i) + "_" + str(i%levels) + '.nii.gz'))
                y_file = open(os.path.join(output_dir,'y.csv'), 'wb')
                wr = csv.writer(y_file, quoting=csv.QUOTE_ALL)
                wr.writerow(self.y)

                rep_id_file = open(os.path.join(output_dir,'rep_id.csv'), 'wb')
                wr = csv.writer(rep_id_file, quoting=csv.QUOTE_ALL)
                wr.writerow(self.rep_id)

    def create_cov_data(self, cor, cov, sigma, radius = 5, reps = 1, output_dir = None):
        """ create simulated data

        Args:
            cor: amount of covariance between each voxel and Y variable
            cov: amount of covariance between voxels
            sigma: amount of noise to add
            radius: vector of radius.  Will create multiple spheres if len(radius) > 1
            reps: number of data repetitions
            output_dir: string path of directory to output data.  If None, no data will be written
            **kwargs: Additional keyword arguments to pass to the prediction algorithm

        """
        
        #initialize useful values
        dims = self.brain_mask.get_data().shape
        
        # Initialize Spheres
        if type(radius) is int:
            p = [dims[0]/2, dims[1]/2, dims[2]/2]
            A = self.sphere(radius, p)
        else:
            raise ValueError("More than one sphere not implemented yet.")

        # Create n_reps with cov for each voxel within sphere
        # Build covariance matrix with each variable correlated with y amount 'cor' and each other amount 'cov'
        
        nifti_sphere = nib.Nifti1Image(A.astype(np.float32), affine=self.brain_mask.affine)
        flat_sphere = self.nifti_masker.fit_transform(nifti_sphere)

        n_vox = np.sum(flat_sphere==1)
        cov_matrix = np.ones([n_vox+1,n_vox+1]) * cov
        cov_matrix[0,:] = cor # set covariance with y
        cov_matrix[:,0] = cor # set covariance with all other voxels
        np.fill_diagonal(cov_matrix,1) # set diagonal to 1
        mv_sim = np.random.multivariate_normal(np.zeros([n_vox+1]),cov_matrix, size=reps)
        self.y = mv_sim[:,0] 
        mv_sim = mv_sim[:,1:]
        new_dat = np.ones([mv_sim.shape[0],flat_sphere.shape[1]])
        new_dat[:,np.where(flat_sphere==1)[1]] = mv_sim
        self.data = self.nifti_masker.inverse_transform(np.add(new_dat,np.random.standard_normal(size=new_dat.shape)*sigma)) #add noise scaled by sigma
        # self.rep_id = ???  # need to add this later


        # # Old method in 4 D space - much slower
        # x,y,z = np.where(A==1)
        # cov_matrix = np.ones([len(x)+1,len(x)+1]) * cov
        # cov_matrix[0,:] = cor # set covariance with y
        # cov_matrix[:,0] = cor # set covariance with all other voxels
        # np.fill_diagonal(cov_matrix,1) # set diagonal to 1
        # mv_sim = np.random.multivariate_normal(np.zeros([len(x)+1]),cov_matrix, size=reps) # simulate data from multivariate covar
        # self.y = mv_sim[:,0] 
        # mv_sim = mv_sim[:,1:]
        # A_4d = np.resize(A,(reps,A.shape[0],A.shape[1],A.shape[2]))
        # for i in xrange(len(x)):
        #     A_4d[:,x[i],y[i],z[i]]=mv_sim[:,i]
        # A_4d = np.rollaxis(A_4d,0,4) # reorder shape of matrix so that time is in 4th dimension
        # self.data = self.to_nifti(np.add(A_4d,np.random.standard_normal(size=A_4d.shape)*sigma)) #add noise scaled by sigma
        # self.rep_id = ???  # need to add this later

        # Write Data to files if requested
        if output_dir is not None:
            if type(output_dir) is str:
                self.data.to_filename(os.path.join(output_dir,'centered_sphere_cor' + str(cor) + "_cov" + str(cov) + '_sigma' + str(sigma) + '.nii.gz'))
                y_file = open(os.path.join(output_dir,'y.csv'), 'wb')
                wr = csv.writer(y_file, quoting=csv.QUOTE_ALL)
                wr.writerow(self.y)

                # rep_id_file = open(os.path.join(output_dir,'rep_id.csv'), 'wb')
                # wr = csv.writer(rep_id_file, quoting=csv.QUOTE_ALL)
                # wr.writerow(self.rep_id)


