'''
NeuroLearn Mask Classes
=======================

Classes to represent masks

'''

__all__ = ['create_sphere',
           'expand_mask',
           'collapse_mask',
           'roi_to_brain']
__author__ = ["Luke Chang", "Sam Greydanus"]
__license__ = "MIT"

import os
import nibabel as nib
from nltools.prefs import MNI_Template, resolve_mni_path
import pandas as pd
import numpy as np
import six
import warnings
from nilearn.masking import intersect_masks


def create_sphere(coordinates, radius=5, mask=None):
    """ Generate a set of spheres in the brain mask space

    Args:
        radius: vector of radius.  Will create multiple spheres if
                len(radius) > 1
        centers: a vector of sphere centers of the form [px, py, pz] or
                [[px1, py1, pz1], ..., [pxn, pyn, pzn]]

    """
    from nltools.data import Brain_Data

    if mask is not None:
        if not isinstance(mask, nib.Nifti1Image):
            if isinstance(mask, six.string_types):
                if os.path.isfile(mask):
                    mask = nib.load(mask)
            else:
                raise ValueError("mask is not a nibabel instance or a valid "
                                 "file name")
    else:
        mask = nib.load(resolve_mni_path(MNI_Template)['mask'])

    def sphere(r, p, mask):
        """ create a sphere of given radius at some point p in the brain mask

        Args:
            r: radius of the sphere
            p: point (in coordinates of the brain mask) of the center of the
                sphere

        """
        dims = mask.shape
        m = [dims[0]/2, dims[1]/2, dims[2]/2]
        x, y, z = np.ogrid[-m[0]:dims[0]-m[0],
                           -m[1]:dims[1]-m[1],
                           -m[2]:dims[2]-m[2]]
        mask_r = x*x + y*y + z*z <= r*r

        activation = np.zeros(dims)
        activation[mask_r] = 1
        translation_affine = np.array([[1, 0, 0, p[0]-m[0]],
                                       [0, 1, 0, p[1]-m[1]],
                                       [0, 0, 1, p[2]-m[2]],
                                       [0, 0, 0, 1]])

        return nib.Nifti1Image(activation, affine=translation_affine)

    if any(isinstance(i, list) for i in coordinates):
        if isinstance(radius, list):
            if len(radius) != len(coordinates):
                raise ValueError('Make sure length of radius list matches'
                                 'length of coordinate list.')
        elif isinstance(radius, int):
            radius = [radius]*len(coordinates)
        out = Brain_Data(nib.Nifti1Image(np.zeros_like(mask.get_data()),
                                         affine=mask.affine), mask=mask)
        for r, c in zip(radius, coordinates):
            out = out + Brain_Data(sphere(r, c, mask), mask=mask)
    else:
        out = Brain_Data(sphere(radius, coordinates, mask), mask=mask)
    out = out.to_nifti()
    out.get_data()[out.get_data() > 0.5] = 1
    out.get_data()[out.get_data() < 0.5] = 0
    return out


def expand_mask(mask, custom_mask=None):
    """ expand a mask with multiple integers into separate binary masks

    Args:
        mask: nibabel or Brain_Data instance
        custom_mask: nibabel instance or string to file path; optional

    Returns:
        out: Brain_Data instance of multiple binary masks

    """

    from nltools.data import Brain_Data
    if isinstance(mask, nib.Nifti1Image):
        mask = Brain_Data(mask, mask=custom_mask)
    if not isinstance(mask, Brain_Data):
        raise ValueError('Make sure mask is a nibabel or Brain_Data instance.')
    mask.data = np.round(mask.data).astype(int)
    tmp = []
    for i in np.nonzero(np.unique(mask.data))[0]:
        tmp.append((mask.data == i)*1)
    out = mask.empty()
    out.data = np.array(tmp)
    return out


def collapse_mask(mask, auto_label=True, custom_mask=None):
    """ collapse separate masks into one mask with multiple integers
        overlapping areas are ignored

    Args:
        mask: nibabel or Brain_Data instance
        custom_mask: nibabel instance or string to file path; optional

    Returns:
        out: Brain_Data instance of a mask with different integers indicating
            different masks

    """

    from nltools.data import Brain_Data
    if not isinstance(mask, Brain_Data):
        if isinstance(mask, nib.Nifti1Image):
            mask = Brain_Data(mask, mask=custom_mask)
        else:
            raise ValueError('Make sure mask is a nibabel or Brain_Data '
                             'instance.')

    if len(mask.shape()) > 1:
        if len(mask) > 1:
            out = mask.empty()

            # Create list of masks and find any overlaps
            m_list = []
            for x in range(len(mask)):
                m_list.append(mask[x].to_nifti())
            intersect = intersect_masks(m_list, threshold=1, connected=False)
            intersect = Brain_Data(nib.Nifti1Image(
                            np.abs(intersect.get_data()-1),
                            intersect.get_affine()), mask=custom_mask)

            merge = []
            if auto_label:
                # Combine all masks into sequential order
                # ignoring any areas of overlap
                for i in range(len(m_list)):
                    merge.append(np.multiply(
                                Brain_Data(m_list[i], mask=custom_mask).data,
                                intersect.data)*(i+1))
                out.data = np.sum(np.array(merge).T, 1).astype(int)
            else:
                # Collapse masks using value as label
                for i in range(len(m_list)):
                    merge.append(np.multiply(
                                    Brain_Data(m_list[i], mask=custom_mask).data,
                                    intersect.data))
                out.data = np.sum(np.array(merge).T, 1)
            return out
    else:
        warnings.warn("Doesn't need to be collapased")


def roi_to_brain(data, mask_x):
    ''' This function will create convert an expanded binary mask of ROIs
    (see expand_mask) based on a vector of of values. The dataframe of values
    must correspond to ROI numbers.

    This is useful for populating a parcellation scheme by a vector of Values

    Args:
        data: Pandas series, dataframe, list, np.array of ROI by observation
        mask_x: an expanded binary mask
    Returns:
        out: (Brain_Data) Brain_Data instance where each ROI is now populated
             with a value
    '''
    from nltools.data import Brain_Data

    def series_to_brain(data, mask_x):
        '''Converts a pandas series of ROIs to a Brain_Data instance. Index must correspond to ROI index'''

        if not isinstance(data, pd.Series):
            raise ValueError('Data must be a pandas series')
        if len(mask_x) != len(data):
            raise ValueError('Data must have the same number of rows as mask has ROIs.')
        return Brain_Data([mask_x[x]*data[x] for x in data.keys()]).sum()

    if not isinstance(data, (pd.Series, pd.DataFrame)):
        if isinstance(data, list):
            if len(data) != len(mask_x):
                raise ValueError('Data must have the same number of rows as mask has ROIs.')
            else:
                data = pd.Series(data)
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                if len(data) != len(mask_x):
                    raise ValueError('Data must have the same number of rows as mask has ROIs.')
                else:
                    data = pd.Series(data)
            elif len(data.shape) == 2:
                data = pd.DataFrame(data)
                if data.shape[0] != len(mask_x):
                    if data.shape[1] == len(mask_x):
                        data = data.T
                    else:
                        raise ValueError('Data must have the same number of rows as rois in mask')
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError


    if len(mask_x) != data.shape[0]:
        raise ValueError('Data must have the same number of rows as mask has ROIs.')

    if isinstance(data, pd.Series):
        return series_to_brain(data, mask_x)
    elif isinstance(data, pd.DataFrame):
        return Brain_Data([series_to_brain(data[x], mask_x) for x in data.keys()])
    else:
        raise ValueError("Data must be a pandas series or data frame.")
