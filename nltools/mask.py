'''
    NeuroLearn Mask Classes
    =========================
    Classes to represent masks

    Author: Luke Chang
    License: MIT
'''

## Notes:


__all__ = ['Masks']

import os
import nibabel as nib
from nltools.utils import get_resource_path
from nilearn.input_data import NiftiMasker
from copy import deepcopy
import pandas as pd
import numpy as np
# from neurosynth.masks import Masker
import importlib
import sklearn
from sklearn.pipeline import Pipeline
from nilearn.input_data import NiftiMasker


def sphere(r, p, mask=None):
    """ create a sphere of given radius at some point p in the brain mask

    Args:
        r: radius of the sphere
        p: point (in coordinates of the brain mask) of the center of the sphere

    """

    if mask is not None:
        if not isinstance(mask,nib.Nifti1Image):
            raise ValueError("mask is not a nibabel instance")
    else:
        mask = nib.load(os.path.join(get_resource_path(),'MNI152_T1_2mm_brain_mask.nii.gz'))
    dims = mask

    x, y, z = np.ogrid[-p[0]:dims[0]-p[0], -p[1]:dims[1]-p[1], -p[2]:dims[2]-p[2]]
    mask = x*x + y*y + z*z <= r*r

    activation = np.zeros(dims)
    activation[mask] = 1
    activation = np.multiply(activation, self.brain_mask.get_data())
    activation = nib.Nifti1Image(activation, affine=np.eye(4))

    #return the 3D numpy matrix of zeros containing the sphere as a region of ones
    return activation.get_data()


