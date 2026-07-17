"""Utilities for creating and manipulating brain masks.

# NeuroLearn Mask Classes

Classes to represent masks

"""

__all__ = [
    "collapse_mask",
    "create_sphere",
    "expand_mask",
    "roi_to_brain",
    "roi_to_brain_from_atlas",
]

import os
import nibabel as nib
from nltools.templates import get_brainspace
import numpy as np
from nilearn.masking import intersect_masks


def create_sphere(coordinates, radius=5, mask=None):
    """Generate spheres in brain-mask space.

    Args:
        coordinates: a vector of sphere centers of the form `[px, py, pz]` or
            `[[px1, py1, pz1], ..., [pxn, pyn, pzn]]`
        radius: radius of the sphere(s). A scalar creates one sphere per
            center; a vector creates multiple spheres if `len(radius) > 1`
        mask: `Nifti1Image` (or path to a mask file) defining the brain space.
            Defaults to the package brain-space mask when None.

    Returns:
        Nifti1Image: A binary image with the requested spheres in mask space.
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
        """Create a sphere with a given radius and center in the brain mask.

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
        else:
            # A single scalar radius (int/float/np scalar) applies to every
            # coordinate. Broadened from the old `isinstance(radius, int)` check
            # so float / numpy-scalar radii no longer fall through to `zip`.
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
    """Expand an integer-labeled mask into separate binary masks.

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
    for i in np.unique(mask.data[mask.data != 0]):
        tmp.append((mask.data == i) * 1)
    out = mask.create_empty()
    out.data = np.array(tmp)
    return out


def collapse_mask(mask, auto_label=True, custom_mask=None):
    """Collapse separate masks into one integer-labeled mask.

    Overlapping areas are ignored.

    Args:
        mask: nibabel or BrainData instance holding 2+ separate masks
            (stacked along the first axis).
        auto_label: If True (default), label the collapsed regions with
            sequential integers (1, 2, 3, …) in mask order. If False, keep each
            mask's own values as its label.
        custom_mask: nibabel instance or string to file path; optional.

    Returns:
        out: BrainData instance of a mask with different integers indicating
            different masks.

    Raises:
        ValueError: If ``mask`` is neither a nibabel nor BrainData instance, or
            if it holds fewer than 2 masks (nothing to collapse).
    """

    from nltools.data import BrainData

    if not isinstance(mask, BrainData):
        if isinstance(mask, nib.Nifti1Image):
            mask = BrainData(mask, mask=custom_mask)
        else:
            raise ValueError("Make sure mask is a nibabel or BrainData instance.")

    if len(mask.shape) <= 1 or len(mask) <= 1:
        raise ValueError(
            "collapse_mask requires 2+ separate masks (stacked along the first "
            "axis) to collapse into an integer-labeled mask; got a single mask."
        )

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
                np.multiply(BrainData(m_list[i], mask=custom_mask).data, intersect.data)
                * (i + 1)
            )
        out.data = np.sum(np.array(merge).T, 1).astype(int)
    else:
        # Collapse masks using value as label
        for i in range(len(m_list)):
            merge.append(
                np.multiply(BrainData(m_list[i], mask=custom_mask).data, intersect.data)
            )
        out.data = np.sum(np.array(merge).T, 1)
    return out


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
            raise ValueError("Data must have the same number of rows as mask has ROIs.")
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
        out.data = np.zeros((arr.shape[1], out.data.shape[1]))
        for roi in range(len(mask_x)):
            roi_data = arr[roi, :].reshape(-1, 1)
            out.data[:, mask_x[roi].data == 1] = np.repeat(
                roi_data.T, np.sum(mask_x[roi].data == 1), axis=0
            ).T
        return out

    raise NotImplementedError("Only 1-D and 2-D data are supported.")


def roi_to_brain_from_atlas(
    values,
    atlas,
    source_mask,
    *,
    roi_labels=None,
    fill: float = np.nan,
):
    """Paint per-parcel values onto voxel space using a labeled atlas.

    Sibling of `roi_to_brain`, but accepts a *labeled* atlas (one
    integer label per voxel — the form carried by
    `SpatialScale`), not an expanded mask
    with one binary row per ROI. Voxels whose atlas label is not in
    ``roi_labels`` (or whose label is 0) receive ``fill``.

    Args:
        values: Per-parcel scalars, either 1-D `(n_parcels,)` for a single
            image or 2-D `(n_images, n_parcels)` for a stack of images. The
            trailing (parcel) axis must match `len(roi_labels)` (or the number
            of unique non-zero atlas labels when `roi_labels` is None).
        atlas: Labeled image — ``BrainData``, ``Nifti1Image``, or path-like.
            Resampled to ``source_mask`` (nearest-neighbor) if shapes/affines
            differ.
        source_mask: ``Nifti1Image`` (or path) defining the output voxel
            grid. The returned ``BrainData`` is masked to this image.
        roi_labels: Integer atlas IDs in the same order as ``values``. If
            None, defaults to ``np.unique`` of the atlas with 0 stripped
            (sorted ascending).
        fill: Value for voxels not in any provided ROI. Default ``np.nan``.

    Returns:
        BrainData: Masked to `source_mask`, with each in-atlas voxel set to its
        parcel's scalar from `values`. Holds a single image when `values` is
        1-D, or `n_images` images when `values` is 2-D `(n_images, n_parcels)`.

    Examples:
        >>> from nltools.mask import roi_to_brain_from_atlas
        >>> brain_map = roi_to_brain_from_atlas(
        ...     values=accuracies,
        ...     atlas=atlas_img,
        ...     source_mask=brain_mask,
        ...     roi_labels=[1, 2, 3],
        ... )
    """
    from pathlib import Path

    from nilearn.image import resample_to_img
    from nilearn.masking import apply_mask as nilearn_apply_mask

    from nltools.data import BrainData

    arr = np.asarray(values)
    if arr.ndim not in (1, 2):
        raise ValueError(
            f"values must be 1-D ``(n_parcels,)`` or 2-D ``(n_images, n_parcels)``; "
            f"got shape {arr.shape}"
        )

    # Coerce atlas + source_mask to nibabel images
    if isinstance(atlas, BrainData):
        atlas_img = atlas.to_nifti()
    elif isinstance(atlas, (str, Path)):
        atlas_img = nib.load(str(atlas))
    else:
        atlas_img = atlas

    if isinstance(source_mask, (str, Path)):
        mask_img = nib.load(str(source_mask))
    else:
        mask_img = source_mask

    # Resample atlas to mask space if needed (nearest-neighbor for labels)
    if atlas_img.shape != mask_img.shape or not np.allclose(
        atlas_img.affine, mask_img.affine
    ):
        atlas_img = resample_to_img(
            atlas_img,
            mask_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )

    # Per-mask-voxel atlas labels — same length as the BrainData voxel axis.
    label_vec = nilearn_apply_mask(atlas_img, mask_img).astype(np.int64)

    if roi_labels is None:
        unique_labels = np.unique(label_vec)
        unique_labels = unique_labels[unique_labels != 0]
    else:
        unique_labels = np.asarray(roi_labels)

    n_parcels_axis = arr.shape[-1] if arr.ndim == 2 else arr.shape[0]
    if n_parcels_axis != len(unique_labels):
        raise ValueError(
            f"values trailing axis ({n_parcels_axis}) must match number of "
            f"ROI labels ({len(unique_labels)})."
        )

    if arr.ndim == 1:
        out_arr = np.full(label_vec.shape, fill, dtype=float)
        for label, value in zip(unique_labels, arr):
            out_arr[label_vec == label] = value
        return BrainData(out_arr.reshape(1, -1), mask=mask_img)

    # 2-D case: shape (n_images, n_parcels) → (n_images, n_voxels) BrainData.
    n_images = arr.shape[0]
    out_arr = np.full((n_images, label_vec.shape[0]), fill, dtype=float)
    for col, label in enumerate(unique_labels):
        cols = label_vec == label
        out_arr[:, cols] = arr[:, col : col + 1]
    return BrainData(out_arr, mask=mask_img)
