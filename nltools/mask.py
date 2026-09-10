"""Utilities for creating and manipulating brain masks."""

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
    """Generate binary spheres in the space of a brain mask.

    Spheres are drawn with `nilearn.maskers.NiftiSpheresMasker`, so centers are
    world (MNI) millimeter coordinates and the radius is in millimeters — the same
    convention as nilearn's `SearchLight` and `NiftiSpheresMasker`. The result is
    resolution-independent: the same request covers the same physical volume on a
    1 mm, 2 mm, or 3 mm grid, up to voxel quantization.

    Args:
        coordinates (list): Sphere center `[x, y, z]` in world (MNI) millimeters, or
            one center per sphere `[[x1, y1, z1], ...]`.
        radius (int | float | list): Radius of the sphere(s) in millimeters. A scalar
            applies to every center; a list gives one radius per center.
        mask (nibabel.Nifti1Image | str, optional): Image (or path) defining the brain
            space. Defaults to the package brain-space mask.

    Returns:
        nibabel.Nifti1Image: A binary image with the requested spheres in mask space.

    Raises:
        ValueError: If `mask` is neither a nibabel image nor a readable file path, if
            the radius list length does not match the coordinate list length, or if a
            requested sphere contains no in-mask voxel.

    Examples:
        ```python
        from nltools.mask import create_sphere

        # A 10 mm sphere centered on an MNI coordinate
        roi = create_sphere([12, 10, -8], radius=10)

        # Two spheres with different radii
        rois = create_sphere([[12, 10, -8], [-12, 10, -8]], radius=[10, 6])
        ```
    """
    if mask is not None:
        if not isinstance(mask, nib.Nifti1Image):
            if isinstance(mask, str) and os.path.isfile(mask):
                mask = nib.load(mask)
            else:
                raise ValueError("mask is not a nibabel instance or a valid file name")
    else:
        mask = nib.load(get_brainspace().mask)

    centers, radii = _resolve_sphere_requests(coordinates, radius)

    volume = np.zeros(mask.shape, dtype=bool)
    for sphere_radius in sorted(set(radii)):
        seeds = [c for c, r in zip(centers, radii) if r == sphere_radius]
        volume |= _draw_spheres(seeds, sphere_radius, mask)

    return nib.Nifti1Image(
        volume.astype(np.float64), affine=mask.affine, header=mask.header
    )


def _resolve_sphere_requests(coordinates, radius):
    """Normalize the center/radius arguments into equal-length lists of floats."""
    if any(isinstance(c, (list, tuple, np.ndarray)) for c in coordinates):
        centers = [tuple(float(v) for v in c) for c in coordinates]
    else:
        centers = [tuple(float(v) for v in coordinates)]

    if isinstance(radius, (list, tuple, np.ndarray)):
        radii = [float(r) for r in radius]
        if len(radii) != len(centers):
            raise ValueError(
                "Make sure length of radius list matches length of coordinate list."
            )
    else:
        radii = [float(radius)] * len(centers)

    return centers, radii


def _draw_spheres(seeds, radius, mask):
    """Return a boolean volume covering every in-mask voxel within `radius` mm of a seed."""
    from nilearn.maskers import NiftiSpheresMasker

    masker = NiftiSpheresMasker(
        seeds=seeds, radius=radius, mask_img=mask, allow_overlap=True
    )
    try:
        masker.fit()
        drawn = masker.inverse_transform(np.ones((1, len(seeds))))
    except ValueError as error:
        # nilearn 0.14 raises "These spheres are empty: [...]" for a seed with no
        # in-mask voxel in range. Every other ValueError from this path (a
        # non-binary mask, a malformed signal vector) is a different problem and
        # must keep its own diagnostic.
        if "spheres are empty" not in str(error):
            raise
        raise ValueError(
            f"No in-mask voxel lies within {radius}mm of one of the requested "
            f"centers {seeds}; the center is outside the mask. Coordinates are "
            "world (MNI) millimeters, not voxel indices."
        ) from error

    return np.asarray(drawn.dataobj)[..., 0] > 0


def expand_mask(mask, custom_mask=None):
    """Expand an integer-labeled mask into separate binary masks.

    Args:
        mask (nibabel.Nifti1Image | BrainData): Integer-labeled mask.
        custom_mask (nibabel.Nifti1Image | str, optional): Brain mask (or path) used
            when converting a nibabel `mask` to `BrainData`.

    Returns:
        BrainData: One binary mask per unique non-zero label.
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
        mask (nibabel.Nifti1Image | BrainData): Two or more separate masks stacked
            along the first axis.
        auto_label (bool): If True (default), label the collapsed regions with
            sequential integers (1, 2, 3, …) in mask order. If False, keep each
            mask's own values as its label.
        custom_mask (nibabel.Nifti1Image | str, optional): Brain mask (or path) used
            when converting a nibabel `mask` to `BrainData`.

    Returns:
        BrainData: A single mask whose integer values identify the source masks.

    Raises:
        ValueError: If `mask` is neither a nibabel nor BrainData instance, or
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
        data (list | np.ndarray | pl.DataFrame | pl.Series | pd.DataFrame | pd.Series):
            ROI values. 1-D length must equal `len(mask_x)`; 2-D shape must be
            `(n_rois, n_obs)` or `(n_obs, n_rois)`.
        mask_x (BrainData): An expanded binary mask with one row per ROI.

    Returns:
        BrainData: A BrainData instance with each ROI populated by the
            provided value(s).
    """
    import polars as pl
    from nltools.data.braindata.utils import _result_from_array

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
        out = _result_from_array(
            mask_x[0], np.zeros(mask_x.data.shape[1]), rows="clear"
        )
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
        out = _result_from_array(
            mask_x, np.zeros((arr.shape[1], mask_x.data.shape[1])), rows="clear"
        )
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

    Sibling of `roi_to_brain`, but accepts a *labeled* atlas (one integer label
    per voxel), not an expanded mask with
    one binary row per ROI. Voxels whose atlas label is not in `roi_labels` (or
    whose label is 0) receive `fill`.

    Args:
        values (np.ndarray): Per-parcel scalars, either 1-D `(n_parcels,)` for a
            single image or 2-D `(n_images, n_parcels)` for a stack of images. The
            trailing (parcel) axis must match `len(roi_labels)` (or the number
            of unique non-zero atlas labels when `roi_labels` is None).
        atlas (BrainData | nibabel.Nifti1Image | str | Path): Labeled image.
            Resampled to `source_mask` (nearest-neighbor) if shapes/affines differ.
        source_mask (nibabel.Nifti1Image | str | Path): Image (or path) defining the
            output voxel grid. The returned `BrainData` is masked to this image.
        roi_labels (array-like, optional): Integer atlas IDs in the same order as
            `values`. If None, defaults to `np.unique` of the atlas with 0 stripped
            (sorted ascending).
        fill (float): Value for voxels not in any provided ROI. Default `np.nan`.

    Returns:
        BrainData: Masked to `source_mask`, with each in-atlas voxel set to its
            parcel's scalar from `values`. Holds a single image when `values` is
            1-D, or `n_images` images when `values` is 2-D `(n_images, n_parcels)`.

    Examples:
        ```python
        from nltools.mask import roi_to_brain_from_atlas

        brain_map = roi_to_brain_from_atlas(
            values=accuracies,
            atlas=atlas_img,
            source_mask=brain_mask,
            roi_labels=[1, 2, 3],
        )
        ```
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
