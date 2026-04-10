"""
NeuroLearn Mask Classes
=======================

Classes to represent masks

"""

__all__ = ["create_sphere", "expand_mask", "collapse_mask", "roi_to_brain"]

import os
import nibabel as nib
from nltools.templates import get_brainspace
import numpy as np
import warnings
from nilearn.masking import intersect_masks


def create_sphere(coordinates, radius=5, mask=None):
    """Generate a set of spheres in the brain mask space

    Args:
        radius: vector of radius.  Will create multiple spheres if
                len(radius) > 1
        centers: a vector of sphere centers of the form [px, py, pz] or
                [[px1, py1, pz1], ..., [pxn, pyn, pzn]]

    """
    from nltools.data import BrainData

    if mask is not None:
        if not isinstance(mask, nib.Nifti1Image):
            if isinstance(mask, str):
                if os.path.isfile(mask):
                    mask = nib.load(mask)
            else:
                raise ValueError("mask is not a nibabel instance or a valid file name")

    else:
        mask = nib.load(get_brainspace().mask)

    def sphere(r, p, mask):
        """create a sphere of given radius at some point p in the brain mask

        Args:
            r: radius of the sphere
            p: point (in coordinates of the brain mask) of the center of the
                sphere

        """
        dims = mask.shape
        m = [dims[0] / 2, dims[1] / 2, dims[2] / 2]
        x, y, z = np.ogrid[
            -m[0] : dims[0] - m[0], -m[1] : dims[1] - m[1], -m[2] : dims[2] - m[2]
        ]
        mask_r = x * x + y * y + z * z <= r * r

        activation = np.zeros(dims)
        activation[mask_r] = 1
        translation_affine = np.array(
            [
                [1, 0, 0, p[0] - m[0]],
                [0, 1, 0, p[1] - m[1]],
                [0, 0, 1, p[2] - m[2]],
                [0, 0, 0, 1],
            ]
        )

        return nib.Nifti1Image(activation, affine=translation_affine)

    if any(isinstance(i, list) for i in coordinates):
        if isinstance(radius, list):
            if len(radius) != len(coordinates):
                raise ValueError(
                    "Make sure length of radius list matcheslength of coordinate list."
                )
        elif isinstance(radius, int):
            radius = [radius] * len(coordinates)
        out = BrainData(
            nib.Nifti1Image(np.zeros_like(mask.get_fdata()), affine=mask.affine),
            mask=mask,
        )
        for r, c in zip(radius, coordinates):
            out = out + BrainData(sphere(r, c, mask), mask=mask)
    else:
        out = BrainData(sphere(radius, coordinates, mask), mask=mask)
    out = out.to_nifti()
    out.get_fdata()[out.get_fdata() > 0.5] = 1
    out.get_fdata()[out.get_fdata() < 0.5] = 0
    return out


def expand_mask(mask, custom_mask=None):
    """expand a mask with multiple integers into separate binary masks

    Args:
        mask: nibabel or BrainData instance
        custom_mask: nibabel instance or string to file path; optional

    Returns:
        out: BrainData instance of multiple binary masks

    """

    from nltools.data import BrainData

    if isinstance(mask, nib.Nifti1Image):
        mask = BrainData(mask, mask=custom_mask)
    if not isinstance(mask, BrainData):
        raise ValueError("Make sure mask is a nibabel or BrainData instance.")
    mask.data = np.round(mask.data).astype(int)
    tmp = []
    for i in np.nonzero(np.unique(mask.data))[0]:
        tmp.append((mask.data == i) * 1)
    out = mask.create_empty()
    out.data = np.array(tmp)
    return out


def collapse_mask(mask, auto_label=True, custom_mask=None):
    """collapse separate masks into one mask with multiple integers
        overlapping areas are ignored

    Args:
        mask: nibabel or BrainData instance
        custom_mask: nibabel instance or string to file path; optional

    Returns:
        out: BrainData instance of a mask with different integers indicating
            different masks

    """

    from nltools.data import BrainData

    if not isinstance(mask, BrainData):
        if isinstance(mask, nib.Nifti1Image):
            mask = BrainData(mask, mask=custom_mask)
        else:
            raise ValueError("Make sure mask is a nibabel or BrainData instance.")

    if len(mask.shape) > 1:
        if len(mask) > 1:
            out = mask.create_empty()

            # Create list of masks and find any overlaps
            m_list = []
            for x in range(len(mask)):
                m_list.append(mask[x].to_nifti())
            intersect = intersect_masks(m_list, threshold=1, connected=False)
            intersect = BrainData(
                nib.Nifti1Image(np.abs(intersect.get_fdata() - 1), intersect.affine),
                mask=custom_mask,
            )

            merge = []
            if auto_label:
                # Combine all masks into sequential order
                # ignoring any areas of overlap
                for i in range(len(m_list)):
                    merge.append(
                        np.multiply(
                            BrainData(m_list[i], mask=custom_mask).data, intersect.data
                        )
                        * (i + 1)
                    )
                out.data = np.sum(np.array(merge).T, 1).astype(int)
            else:
                # Collapse masks using value as label
                for i in range(len(m_list)):
                    merge.append(
                        np.multiply(
                            BrainData(m_list[i], mask=custom_mask).data, intersect.data
                        )
                    )
                out.data = np.sum(np.array(merge).T, 1)
            return out
    else:
        warnings.warn("Doesn't need to be collapased")


def roi_to_brain(data, mask_x):
    """Populate an expanded binary ROI mask with a vector or matrix of per-ROI values.

    Accepts lists, numpy arrays, polars DataFrame/Series, or pandas
    DataFrame/Series. Internally coerces to a numpy array and operates on
    it — 1-D input produces a single BrainData image; 2-D input (ROIs by
    observations) produces a stack of BrainData images, one per
    observation.

    Args:
        data: ROI values. 1-D length must equal len(mask_x);
            2-D shape must be (n_rois, n_obs) or (n_obs, n_rois).
        mask_x: An expanded binary mask (BrainData) with one row per ROI.

    Returns:
        BrainData: A BrainData instance with each ROI populated by the
        provided value(s).
    """
    import polars as pl

    if isinstance(data, (pl.DataFrame, pl.Series)):
        arr = data.to_numpy()
    elif isinstance(data, np.ndarray):
        arr = data
    elif isinstance(data, list):
        arr = np.asarray(data)
    else:
        try:
            import pandas as pd
        except ImportError:
            pd = None
        if pd is not None and isinstance(data, (pd.Series, pd.DataFrame)):
            arr = np.asarray(data)
        else:
            raise ValueError(
                "Data must be a list, numpy array, polars DataFrame/Series, "
                "or pandas DataFrame/Series."
            )

    if arr.ndim == 1:
        if len(arr) != len(mask_x):
            raise ValueError(
                "Data must have the same number of rows as mask has ROIs."
            )
        out = mask_x[0].copy()
        out.data = np.zeros(out.data.shape)
        for roi in range(len(mask_x)):
            out.data[np.where(mask_x.data[roi, :])] = arr[roi]
        return out

    if arr.ndim == 2:
        if arr.shape[0] != len(mask_x):
            if arr.shape[1] == len(mask_x):
                arr = arr.T
            else:
                raise ValueError(
                    "Data must have the same number of rows as rois in mask"
                )
        out = mask_x.copy()
        out.data = np.ones((arr.shape[1], out.data.shape[1]))
        for roi in range(len(mask_x)):
            roi_data = arr[roi, :].reshape(-1, 1)
            out.data[:, mask_x[roi].data == 1] = np.repeat(
                roi_data.T, np.sum(mask_x[roi].data == 1), axis=0
            ).T
        return out

    raise NotImplementedError("Only 1-D and 2-D data are supported.")
