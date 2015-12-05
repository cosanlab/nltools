"""Handy utilities"""

__all__ = ['get_resource_path']

from os.path import dirname, join, pardir, sep as pathsep
import pandas as pd
import nibabel as nib

def get_resource_path():
	return join(dirname(__file__), 'resources') + pathsep

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

