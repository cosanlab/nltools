'''
NeuroLearn Mask Classes
=======================

Classes to represent masks

'''

__all__ = ['create_sphere', 'expand_mask', 'collapse_mask']
__author__ = ["Luke Chang", "Sam Greydanus"]
__license__ = "MIT"

import os
import nibabel as nib
from nltools.utils import get_resource_path
from nilearn.input_data import NiftiMasker
from copy import deepcopy
import pandas as pd
import numpy as np
import warnings
from nilearn.masking import intersect_masks
# from neurosynth.masks import Masker


def create_sphere(coordinates, radius=5, mask=None):
    """ Generate a set of spheres in the brain mask space

    Args:
        radius: vector of radius.  Will create multiple spheres if len(radius) > 1
        centers: a vector of sphere centers of the form [px, py, pz] or [[px1, py1, pz1], ..., [pxn, pyn, pzn]]

    """
    from nltools.data import Brain_Data

    if mask is not None:
        if not isinstance(mask,nib.Nifti1Image):
            if type(mask) is str:
                if os.path.isfile(mask):
                    data = nib.load(mask)
            else:
                raise ValueError("mask is not a nibabel instance or a valid file name")
    else:
        mask = nib.load(os.path.join(get_resource_path(),'MNI152_T1_2mm_brain_mask.nii.gz'))
    dims = mask.get_data().shape

    def sphere(r, p, mask):
        """ create a sphere of given radius at some point p in the brain mask

        Args:
            r: radius of the sphere
            p: point (in coordinates of the brain mask) of the center of the sphere
 
        """
        dims = mask.shape
        m = [dims[0]/2, dims[1]/2, dims[2]/2] # JC edit: default value for centers
        x, y, z = np.ogrid[-m[0]:dims[0]-m[0], -m[1]:dims[1]-m[1], -m[2]:dims[2]-m[2]] #JC edit: creates sphere
        # x, y, z = np.ogrid[-p[0]:dims[0]-p[0], -p[1]:dims[1]-p[1], -p[2]:dims[2]-p[2]]
        mask_r = x*x + y*y + z*z <= r*r

        activation = np.zeros(dims)
        activation[mask_r] = 1
        # JC edit shift mask to proper location
        translation_affine= np.array([[1, 0, 0, p[0]-m[0]],
                                [0, 1, 0, p[1]-m[1]],
                                [0, 0, 1, p[2]-m[2]],
                                 [0, 0, 0, 1]])

        # activation = np.multiply(activation, mask.get_data())
        # activation = nib.Nifti1Image(activation, affine=np.eye(4))
        activation = nib.Nifti1Image(activation,affine=translation_affine)
        #return the 3D numpy matrix of zeros containing the sphere as a region of ones
        # return activation.get_data(), translation_affine
        return activation

    # Initialize Spheres with options for multiple radii and centers of the spheres (or just an int and a 3D list)
    # return sphere(radius,coordinates,mask)
    if type(radius) is int:
        radius = [radius]
    if coordinates is None:
        coordinates = [[dims[0]/2, dims[1]/2, dims[2]/2] * len(radius)] #default value for centers
    elif type(coordinates) is list and type(coordinates[0]) is int and len(radius) is 1:
        coordinates = [coordinates]
    if (type(radius)) is list and (type(coordinates) is list) and (len(radius) == len(coordinates)):
        A = np.zeros_like(mask.get_data())
        A = Brain_Data(nib.Nifti1Image(A,affine=mask.get_affine()))
        for i in xrange(len(radius)):
            A = A + Brain_Data(sphere(radius[i], coordinates[i], mask))
            # B,translation_affine = sphere(radius[i], coordinates[i], mask)
            # A = np.add(A, B)
            # A = np.add(A, sphere(radius[i], coordinates[i], mask))
        # nifti_sphere = nib.Nifti1Image(A.astype(np.float32), affine=translation_affine)
        # return nifti_sphere
        A = A.to_nifti()
        A.get_data()[A.get_data()>0.5]=1
        A.get_data()[A.get_data()<0.5]=0
        return A
    else:
        raise ValueError("Data type for sphere or radius(ii) or center(s) not recognized.")

def expand_mask(mask):
    """ expand a mask with multiple integers into separate binary masks
    
    Args:
        mask: nibabel or Brain_Data instance
    
    Returns:
        out: Brain_Data instance of multiple binary masks

    """

    from nltools.data import Brain_Data
    if isinstance(mask,nib.Nifti1Image):
        mask = Brain_Data(mask)
    if not isinstance(mask,Brain_Data):
        raise ValueError('Make sure mask is a nibabel or Brain_Data instance.')
    mask.data = np.round(mask.data).astype(int)
    tmp = []
    for i in np.nonzero(np.unique(mask.data))[0]:
        tmp.append((mask.data==i)*1)
    out = mask.empty()
    out.data = np.array(tmp)
    return out

def collapse_mask(mask, auto_label=True):
    """ collapse separate masks into one mask with multiple integers overlapping areas are ignored

    Args:
        mask: nibabel or Brain_Data instance

    Returns:
        out: Brain_Data instance of a mask with different integers indicating different masks

    """

    from nltools.data import Brain_Data
    if not isinstance(mask,Brain_Data):
        if isinstance(mask,nib.Nifti1Image):
            mask = Brain_Data(mask)
        else:
            raise ValueError('Make sure mask is a nibabel or Brain_Data instance.')
    
    if len(mask.shape()) > 1:
        if len(mask) > 1:
            out = mask.empty()

            # Create list of masks and find any overlaps
            m_list = []
            for x in range(len(mask)):
                m_list.append(mask[x].to_nifti())
            intersect = intersect_masks(m_list, threshold=1, connected=False)
            intersect = Brain_Data(nib.Nifti1Image(np.abs(intersect.get_data()-1),intersect.get_affine()))

            merge = []
            if auto_label:
                # Combine all masks into sequential order ignoring any areas of overlap            
                for i in range(len(m_list)):
                    merge.append(np.multiply(Brain_Data(m_list[i]).data,intersect.data)*(i+1))
                out.data = np.sum(np.array(merge).T,1).astype(int)
            else:
                # Collapse masks using value as label
                for i in range(len(m_list)):
                    merge.append(np.multiply(Brain_Data(m_list[i]).data,intersect.data))
                out.data = np.sum(np.array(merge).T,1)
            return out
    else:
        warnings.warn("Doesn't need to be collapased")



