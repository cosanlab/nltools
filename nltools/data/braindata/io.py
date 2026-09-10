"""Loading, resampling, writing, and uploading for `BrainData`.

Functions that resolve a mask, load data (from files, lists, URLs, HDF5, or other
`BrainData` objects), resample to a target grid, write NIfTI/HDF5, and upload to
NeuroVault. `BrainData` methods delegate here.
"""

import os
import re
import shutil
import tempfile
import warnings

import numpy as np
from pathlib import Path

from nltools.utils import ResamplingWarning, find_stack_level


def _detect_interpolation(img):
    """Detect appropriate interpolation method based on image data type.

    Determines whether an image contains discrete (atlas/label) or continuous
    data by checking if values are integers and counting unique values. For a
    4-D image only the first volume is inspected: it decides label-vs-signal as
    well as the whole run does, without materializing a float64 copy of every
    volume (nearly 2 GB for a typical BOLD run).

    Args:
        img: nibabel Nifti1Image or similar image object with a ``dataobj``.

    Returns:
        str: 'nearest' for discrete/atlas data, 'continuous' for continuous data

    Notes:
        - Returns 'nearest' if all non-NaN values are integers AND unique count < 1000
        - Atlases typically have < 500 unique integer labels
        - Statistical maps have continuous floating-point values
    """
    if img.ndim >= 4:
        data = np.asanyarray(img.dataobj[..., 0])
    else:
        data = np.asanyarray(img.dataobj)

    # Handle empty or all-NaN data
    valid_data = data[~np.isnan(data)]
    if valid_data.size == 0:
        return "continuous"

    # Check if all values are effectively integers (trivially true for an
    # integer dtype; nibabel applies any header scaling, so a scaled int image
    # arrives here as float and is checked value by value).
    is_integer_valued = valid_data.dtype.kind in "iu" or np.allclose(
        valid_data, np.round(valid_data), rtol=1e-10
    )

    if is_integer_valued:
        n_unique = len(np.unique(valid_data))
        # Atlases typically have < 1000 unique labels (most have < 500)
        # Continuous data would have many more unique values
        if n_unique < 1000:
            return "nearest"

    return "continuous"


def initialize_mask(bd, mask):
    """Initialize the mask image.

    Args:
        bd (BrainData): Instance whose mask is being set.
        mask (Nifti1Image | str | Path | None): Brain mask as a nibabel image, file
            path, template name string, or None. Template name strings follow
            `'{res}mm-MNI152-2009{version}'` (e.g. `'2mm-MNI152-2009c'`,
            `'3mm-MNI152-2009a'`, `'2mm-MNI152-2009fsl'`).
    """
    import nibabel as nib
    from nltools.templates import get_brainspace

    # Store whether mask was None (for auto-detection later)
    bd._mask_was_none = mask is None

    if mask is None:
        # For empty BrainData or when data not yet loaded, use default template
        # Template will be auto-detected during data loading if data is provided
        bd.mask = nib.load(get_brainspace().mask)
        bd._detected_template = None  # Will be set during data loading if needed
    elif isinstance(mask, (str, Path)):
        mask_str = str(mask)
        # Check if it's a template name string (format: {res}mm-MNI152-2009{version})
        if re.match(r"^\d+mm-MNI152-2009[acfsl]+$", mask_str):
            # Resolve template name to file path
            from nltools.templates import resolve_template_name

            mask_path = resolve_template_name(mask_str, file_type="mask")
            bd.mask = nib.load(mask_path)
        else:
            # Regular file path
            bd.mask = nib.load(mask_str)
        bd._detected_template = None  # Explicit mask provided, no auto-detection
    elif isinstance(mask, nib.Nifti1Image):
        bd.mask = mask
        bd._detected_template = None  # Explicit mask provided, no auto-detection
    else:
        raise TypeError(
            f"mask must be a nibabel instance, file path, template name string, or None. "
            f"Received {type(mask).__name__}"
        )

    # Extract voxel resolution from mask affine matrix
    # The diagonal elements of the affine matrix (excluding translation) give voxel sizes
    affine = bd.mask.affine
    bd._voxel_resolution = np.abs(np.diag(affine[:3, :3]))

    # Determine space (MNI or native) based on mask
    bd._space = detect_space(bd.mask)


def get_interpolation(bd, img):
    """Get the interpolation method to use for a given image.

    Resolves 'auto' to either 'nearest' or 'continuous' based on data type.

    Args:
        bd (BrainData): Instance whose interpolation setting is consulted.
        img (Nifti1Image): Image to inspect when the setting is 'auto'.

    Returns:
        str: Interpolation method. When the instance setting is 'auto', resolves
            to 'nearest' or 'continuous' based on data type; otherwise the
            instance's configured interpolation setting.
    """
    if bd._interpolation == "auto":
        return _detect_interpolation(img)
    return bd._interpolation


def _resample_img_to_mask(bd, data_img):
    """Resample ``data_img`` onto ``bd.mask``'s grid with the resolved interpolation.

    Integer-typed voxel data (int16 BOLD is the common case) is cast to float32
    first when the interpolation is continuous. nilearn performs exactly that
    cast itself inside ``resample_img`` — and warns about it on every load —
    so doing it here removes the notice without changing the result. Nearest
    interpolation keeps the integer dtype (labels stay labels). A header with
    no sform (haxby's, for one) gets the same code-2 sform `resample`
    assigns, for the same reason (see `_ensure_sform`).
    """
    import nibabel as nib
    from nilearn.image import resample_to_img

    interpolation = get_interpolation(bd, data_img)
    if interpolation != "nearest":
        data = np.asanyarray(data_img.dataobj)
        if data.dtype.kind in "iu":
            data_img = nib.Nifti1Image(
                data.astype(np.float32), data_img.affine, data_img.header
            )
            data_img.set_data_dtype(np.float32)
    return resample_to_img(
        _ensure_sform(data_img), bd.mask, interpolation=interpolation
    )


def _resample_to_mask(bd, data_img, context=""):
    """Resample data_img to bd.mask if spaces differ and bd._resample is True.

    Returns data_img unchanged if spaces already match or resampling is disabled.
    """
    if check_space_match(data_img, bd.mask) or not bd._resample:
        return data_img

    warn_if_resampling(bd, context)
    return _resample_img_to_mask(bd, data_img)


def detect_and_update_mask(bd, data_img):
    """Detect best matching template from data and update mask if mask was None.

    Also handles resampling if needed based on the resample kwarg.

    This function is called during data loading to auto-detect template when mask=None.
    After detecting or falling back to a template, it checks if resampling is needed
    and resamples the data_img accordingly.

    Args:
        bd (BrainData): Instance whose mask may be updated.
        data_img (Nifti1Image): Image from which to detect the template.

    Returns:
        Nifti1Image: The input image, resampled to the mask grid if needed.
    """
    import nibabel as nib

    if not bd._mask_was_none:
        return _resample_to_mask(bd, data_img)

    try:
        from nltools.templates import match_resolution

        template_info = match_resolution(
            data_img.affine,
            prefer_exact=True,
            warn_resample=bd._resample,
        )
        bd._detected_template = template_info

        detected_mask = nib.load(template_info.mask_path)
        current_mask_path = bd.mask.get_filename()
        detected_mask_path = template_info.mask_path

        if current_mask_path != detected_mask_path:
            bd.mask = detected_mask
            affine = bd.mask.affine
            bd._voxel_resolution = np.abs(np.diag(affine[:3, :3]))
            bd._space = detect_space(bd.mask)

        return _resample_to_mask(
            bd,
            data_img,
            f"Detected template ({template_info.template} "
            f"{template_info.resolution}mm) differs from data resolution.",
        )

    except Exception as e:
        warnings.warn(
            f"Failed to auto-detect template from data: {e}. "
            f"Using default template (get_brainspace().mask).",
            UserWarning,
            stacklevel=find_stack_level(),
        )
        return _resample_to_mask(
            bd,
            data_img,
            "Template auto-detection failed; using default template.",
        )


def detect_space(mask):
    """Detect if mask is in MNI space or native space.

    Args:
        mask (Nifti1Image): Mask image to classify.

    Returns:
        str: 'mni' if the mask matches the MNI template, 'native' otherwise.
    """
    import nibabel as nib
    from nltools.templates import get_brainspace

    # Get mask filename if available
    mask_filename = mask.get_filename()

    # Check if mask is None (uses default MNI template)
    # This is handled in initialize_mask, but check here for safety
    if mask_filename is None:
        # Compare affine matrix with MNI template
        try:
            mni_mask = nib.load(get_brainspace().mask)
            if np.allclose(mask.affine, mni_mask.affine, rtol=1e-3):
                return "mni"
        except Exception:
            pass
        return "native"

    # Normalize paths for comparison
    mask_path = str(Path(mask_filename).resolve())
    mni_mask_path = str(Path(get_brainspace().mask).resolve())

    # Check if mask path matches MNI template path
    if mask_path == mni_mask_path:
        return "mni"

    # Check if affine matches MNI template affine (for cases where mask is loaded differently)
    try:
        mni_mask = nib.load(get_brainspace().mask)
        if np.allclose(mask.affine, mni_mask.affine, rtol=1e-3):
            return "mni"
    except Exception:
        pass

    # Default to native if not matching MNI
    return "native"


def check_space_match(data_img, mask_img):
    """Check if data and mask are in same space.

    Args:
        data_img (Nifti1Image): Data image.
        mask_img (Nifti1Image): Mask image.

    Returns:
        bool: True if affines and spatial shapes match (no resampling needed).
    """
    # Compare affine matrices
    affine_match = np.allclose(data_img.affine, mask_img.affine, rtol=1e-3)

    # Compare spatial shapes
    shape_match = data_img.shape[:3] == mask_img.shape[:3]

    return affine_match and shape_match


def warn_if_resampling(bd, context=""):
    """Emit a `ResamplingWarning` if ``verbose=True`` and ``resample=True``.

    Sibling of the template-mismatch notice in `match_resolution`: that one
    fires when a template is chosen for data at another resolution; this one
    fires when the data is actually resampled to the mask's grid.

    Args:
        bd (BrainData): Instance whose `verbose` and resample settings apply.
        context (str): Why the spaces differ, appended to the message.
            Default: empty string.
    """
    if bd._resample and bd.verbose:
        resolution = "x".join(f"{r:g}" for r in bd._voxel_resolution)
        msg = (
            f"Data does not match the mask space; resampling it to the mask's "
            f"{resolution}mm grid (resample=True)."
        )
        if context:
            msg = f"{msg} {context}"
        warnings.warn(msg, ResamplingWarning, stacklevel=find_stack_level())


def mask_images(mask, imgs):
    """Mask a list of space-aligned images with a single fitted masker.

    Validates ``mask`` exactly ONCE — one ``load_mask_img`` — and reuses the
    binarized mask across every image in ``imgs``, instead of re-running
    nilearn's costly ``load_mask_img`` (binarization checks + ``safe_get_data``,
    which each trigger nilearn's forced ``gc.collect``) per image.

    ``nilearn.masking.apply_mask`` is exactly ``load_mask_img`` (validate) ->
    ``new_img_like`` (build binary mask) -> ``apply_mask_fmri`` (extract), with
    ``dtype='f'``, ``smoothing_fwhm=None``, ``ensure_finite=True``. This hoists
    the first two out of the per-image loop and calls the lower-level
    ``apply_mask_fmri`` (which "assumes mask_img contains only two different
    values") per image, so the result is byte-equivalent to
    ``np.vstack([apply_mask(im, mask) for im in imgs])`` for space-aligned data.

    Images must already share ``mask``'s space (callers resample first); no
    resampling is done here. Falls back to the per-image functional
    ``apply_mask`` if the fast path raises for any reason.

    Args:
        mask (Nifti1Image): Boolean/binary mask image.
        imgs (list[Nifti1Image]): Space-aligned images to mask.

    Returns:
        np.ndarray: Masked data of shape ``(len(imgs), n_voxels)``.
    """
    from nilearn.masking import apply_mask as nilearn_apply_mask

    try:
        return _mask_images_fast(mask, imgs)
    except Exception:
        # Functional fallback — one load_mask_img per image, but always correct.
        return np.vstack([nilearn_apply_mask(im, mask) for im in imgs])


def _mask_images_fast(mask, imgs):
    """Validate ``mask`` once, then extract every image via ``apply_mask_fmri``.

    Reproduces ``nilearn.masking.apply_mask``'s internals with the
    validate-and-binarize step (``load_mask_img`` + ``new_img_like``) hoisted
    out of the per-image loop. Split out so the fallback in `mask_images`
    is testable in isolation.
    """
    from nilearn.image import new_img_like
    from nilearn.masking import apply_mask_fmri, load_mask_img

    mask_arr, mask_affine = load_mask_img(mask)  # validate + binarize ONCE
    binary_mask = new_img_like(mask, mask_arr, mask_affine)
    return np.vstack([apply_mask_fmri(im, binary_mask) for im in imgs])


def load_from_list(bd, data_list):
    """Load data from a list of BrainData objects or file paths.

    Args:
        bd (BrainData): Instance to populate.
        data_list (list[BrainData] | list[str | Path | Nifti1Image]): Items to load
            and stack.
    """
    import nibabel as nib
    from nltools.utils import concatenate
    from nltools.data.braindata.validation import validate_list_data

    list_type = validate_list_data(data_list)

    if list_type == "brain_data":
        tmp = concatenate(data_list)
        for item in ["data", "mask"]:
            setattr(bd, item, getattr(tmp, item))
        return

    bd.data = []

    # Auto-detect template from first item if mask was None
    if bd._mask_was_none and len(data_list) > 0:
        first_item = data_list[0]
        if isinstance(first_item, (str, Path)):
            first_img = nib.load(str(first_item))
        elif isinstance(first_item, nib.Nifti1Image):
            first_img = first_item
        else:
            first_img = None

        if first_img is not None:
            detect_and_update_mask(bd, first_img)

    # Prepare (load + space-align) each item, then mask them all with a single
    # fitted masker so the mask is validated once per call rather than per item.
    prepared_imgs = []
    for idx, item in enumerate(data_list):
        if isinstance(item, (str, Path)):
            item_img = nib.load(str(item))
        elif isinstance(item, nib.Nifti1Image):
            item_img = item
        else:
            raise TypeError(
                f"List items must be file paths or nibabel Nifti1Image. "
                f"Received {type(item).__name__}"
            )

        if not check_space_match(item_img, bd.mask):
            if not bd._resample:
                raise ValueError(
                    f"Data item and mask are in different spaces. "
                    f"Set resample=True to automatically resample data to mask space, "
                    f"or ensure all data items are already in the same space as the mask.\n"
                    f"Item affine:\n{item_img.affine}\n"
                    f"Mask affine:\n{bd.mask.affine}\n"
                    f"Item shape: {item_img.shape[:3]}\n"
                    f"Mask shape: {bd.mask.shape[:3]}"
                )
            if idx == 0:
                warn_if_resampling(bd)
            item_img = _resample_to_mask(bd, item_img)

        prepared_imgs.append(item_img)

    # Byte-equivalent to per-item apply_mask + vstack, but validates the mask
    # once (see mask_images). vstack for nilearn 0.12+ compat (transforms
    # 3D -> 1D instead of 3D -> 2D).
    bd.data = mask_images(bd.mask, prepared_imgs)


def load_from_brain_data(bd, brain_data, mask=None):
    """Load data from another BrainData object.

    Args:
        bd (BrainData): Instance to populate.
        brain_data (BrainData): Object to copy from.
        mask (Nifti1Image | str | Path | None): Mask to use. If None, uses the mask
            from `brain_data`.
    """
    import nibabel as nib
    from nilearn.image import resample_to_img
    from nilearn.masking import apply_mask as nilearn_apply_mask

    # Copy data array
    bd.data = brain_data.data.copy() if brain_data.data is not None else np.array([])

    # Handle mask: use provided mask if given, otherwise use source mask
    if mask is not None:
        # User provided mask - re-initialize with it
        # This will trigger mask initialization but we already have data
        # Need to handle resampling if mask differs
        if isinstance(mask, (str, Path)):
            mask_str = str(mask)
            # Check if it's a template name string
            if re.match(r"^\d+mm-MNI152-2009[acfsl]+$", mask_str):
                # Resolve template name to file path
                from nltools.templates import resolve_template_name

                new_mask = nib.load(resolve_template_name(mask_str, file_type="mask"))
            else:
                # Regular file path
                new_mask = nib.load(mask_str)
        elif isinstance(mask, nib.Nifti1Image):
            new_mask = mask
        else:
            raise TypeError(
                f"mask must be a nibabel instance, file path, template name string, or None. "
                f"Received {type(mask).__name__}"
            )

        # Check if mask differs from source
        if not check_space_match(brain_data.mask, new_mask):
            # Need to resample data to new mask space
            if bd._resample:
                warn_if_resampling(bd, "New mask differs from source BrainData mask.")
                source_nifti = brain_data.to_nifti()
                resampled_nifti = resample_to_img(
                    source_nifti,
                    new_mask,
                    interpolation=get_interpolation(bd, source_nifti),
                )
                # Update mask
                bd.mask = new_mask
                # Extract data via functional apply_mask
                bd.data = nilearn_apply_mask(resampled_nifti, bd.mask)
                # Update voxel resolution and space
                affine = bd.mask.affine
                bd._voxel_resolution = np.abs(np.diag(affine[:3, :3]))
                bd._space = detect_space(bd.mask)
            else:
                raise ValueError(
                    "Source BrainData mask and provided mask are in different spaces. "
                    "Set resample=True to automatically resample data to new mask space."
                )
        else:
            # Masks match - just update mask reference
            bd.mask = new_mask
            affine = bd.mask.affine
            bd._voxel_resolution = np.abs(np.diag(affine[:3, :3]))
            bd._space = detect_space(bd.mask)
    else:
        # Use source mask
        bd.mask = brain_data.mask
        bd._voxel_resolution = brain_data._voxel_resolution
        bd._space = brain_data._space

    # Copy detected template info if present
    if hasattr(brain_data, "_detected_template"):
        bd._detected_template = brain_data._detected_template
    if hasattr(brain_data, "_mask_was_none"):
        bd._mask_was_none = brain_data._mask_was_none


def load_from_h5(bd, file_path, mask):
    """Load data from HDF5 file.

    Args:
        bd (BrainData): Instance to populate.
        file_path (str | Path): Path to the HDF5 file.
        mask (Nifti1Image | str | Path | None): User-specified mask; when None the
            mask stored in the file is used.
    """
    from nltools.io import load_brain_data_h5

    # Load data using utility function
    h5_data = load_brain_data_h5(file_path, mask)
    bd.data = h5_data["data"]

    # Load X and Y if present (for backward compatibility)
    if "X" in h5_data:
        bd.X = h5_data["X"]
    if "Y" in h5_data:
        bd.Y = h5_data["Y"]

    # Handle mask if loaded from file
    if h5_data.get("load_mask", False):
        bd.mask = h5_data["mask"]
        # Extract voxel resolution from mask affine matrix
        affine = bd.mask.affine
        bd._voxel_resolution = np.abs(np.diag(affine[:3, :3]))
        # Determine space (MNI or native) based on mask
        bd._space = detect_space(bd.mask)
    elif mask is not None and not h5_data.get("load_mask", True):
        warnings.warn(
            "Existing mask found in HDF5 file but is being ignored because "
            "you passed a value for mask. Set mask=None to use existing "
            "mask in the HDF5 file",
            UserWarning,
            stacklevel=find_stack_level(),
        )


def load_from_url(bd, url):
    """Load data from URL.

    Args:
        bd (BrainData): Instance to populate.
        url (str): URL of a NIfTI file to download.
    """
    import nibabel as nib
    from nltools.datasets import download_nifti

    # TemporaryDirectory guarantees a unique name and removes the download
    # (avoids the os.times()-based collision + leak in the old code).
    with tempfile.TemporaryDirectory() as tmp_dir:
        downloaded_file = nib.load(download_nifti(url, data_dir=tmp_dir))
        load_from_file(bd, downloaded_file)


def load_from_file(bd, data):
    """Load data from file path or nibabel object.

    Args:
        bd (BrainData): Instance to populate.
        data (str | Path | Nifti1Image): File path or nibabel image.
    """
    import nibabel as nib
    from nilearn.masking import apply_mask as nilearn_apply_mask

    if isinstance(data, (str, Path)):
        data_img = nib.load(str(data))
    elif isinstance(data, nib.Nifti1Image):
        data_img = data
    else:
        raise TypeError(
            f"data must be a file path or nibabel Nifti1Image. "
            f"Received {type(data).__name__}"
        )

    # Auto-detect template from data if mask was None; also handles resampling.
    data_img = detect_and_update_mask(bd, data_img)

    # When resample=False but spaces still mismatch, warn and resample anyway
    # (required for correct masking).
    if not bd._resample and not check_space_match(data_img, bd.mask):
        if bd.verbose:
            warnings.warn(
                f"Data and mask are in different spaces (affine or shape mismatch). "
                f"Resampling data to match mask space despite resample=False. "
                f"Set resample=True to explicitly enable resampling, or ensure data "
                f"is already in the same space as the mask.\n"
                f"Data affine:\n{data_img.affine}\n"
                f"Mask affine:\n{bd.mask.affine}\n"
                f"Data shape: {data_img.shape[:3]}\n"
                f"Mask shape: {bd.mask.shape[:3]}",
                ResamplingWarning,
                stacklevel=find_stack_level(),
            )
        data_img = _resample_img_to_mask(bd, data_img)

    bd.data = nilearn_apply_mask(data_img, bd.mask)


def to_nifti(bd):
    """Convert BrainData instance to a nibabel NIfTI image.

    Args:
        bd (BrainData): Instance to convert.

    Returns:
        Nifti1Image: Brain data in volumetric NIfTI format.
    """
    from nilearn.masking import unmask

    img = unmask(bd.data, bd.mask)
    # unmask inherits the mask's dtype (often int8) for the output header, which
    # would scale-quantize float data to ~1 LSB on save — silently lossy for stat
    # maps and betas. Pin the on-disk dtype to the
    # data's own so writes are lossless (integer masks stay integer, maps stay float).
    img.set_data_dtype(bd.data.dtype)
    return img


def _ensure_sform(img):
    """Return a Nifti1Image with sform_code set, copying first if needed.

    nilearn emits a warning during resampling when sform_code==0. We set
    code=2 (NIFTI_XFORM_ALIGNED_ANAT) — the same value nilearn assigns to
    resampled outputs. The copy avoids mutating caller-owned objects
    (e.g. ``bd.mask`` or an image passed in by the user).
    """
    import nibabel as nib

    if img.header.get_sform(coded=True)[1] != 0:
        return img
    out = nib.Nifti1Image(img.dataobj, img.affine, img.header.copy())
    out.header.set_sform(img.affine, code=2)
    return out


def resample(bd, *, img=None, resolution=None, interpolation=None):
    """Resample BrainData onto a new voxel grid.

    Exactly one of `img` or `resolution` must be given. An `img` supplies only
    the target grid; its intensity values never define the output mask. The
    source mask is resampled onto that grid with nearest-neighbor interpolation
    and installed on the result, which preserves row-aligned `X` and `Y` and
    carries no fitted state.

    Args:
        bd (BrainData): Instance to resample.
        img (Nifti1Image | str | Path | None): Target image whose grid to match,
            as a nibabel image or a path to a `.nii`/`.nii.gz` file.
        resolution (float | int | None): Target isotropic voxel size in mm
            (e.g. `2.0` for 2 mm³ voxels).
        interpolation (str | None): Interpolation method for the data:
            `'nearest'` (atlases, masks, labels), `'linear'`, or `'continuous'`
            (higher-order spline, for stat maps). None uses the instance's
            interpolation setting.

    Returns:
        BrainData: New instance with resampled data and mask.

    Raises:
        ValueError: If both `img` and `resolution` are None, both are provided,
            `resolution` is not positive, or the instance is empty.
        TypeError: If `img` is not a valid image type.
    """
    import nibabel as nib
    from nilearn.image import resample_to_img, resample_img
    from nilearn.masking import apply_mask as nilearn_apply_mask

    from .utils import _result_with_mask

    if img is None and resolution is None:
        raise ValueError(
            "Must provide either 'img' or 'resolution' parameter. "
            "Provide exactly one of them."
        )
    if img is not None and resolution is not None:
        raise ValueError(
            "Cannot provide both 'img' and 'resolution' parameters. "
            "Provide exactly one of them."
        )

    # Check the target argument itself, then the object, before touching disk
    # or doing any resampling work.
    target_affine = None
    if resolution is not None:
        resolution = float(resolution)
        if resolution <= 0:
            raise ValueError(f"resolution must be positive. Got {resolution}")
        target_affine = np.eye(4)
        target_affine[:3, :3] = np.diag([resolution, resolution, resolution])
        target_description = f"resolution={resolution}"
    else:
        if not isinstance(img, (str, Path, nib.Nifti1Image)):
            raise TypeError(
                f"img must be nibabel Nifti1Image, file path (str/Path), or None. "
                f"Got {type(img).__name__}"
            )
        target_description = f"img={img if isinstance(img, (str, Path)) else 'image'}"

    if len(bd) == 0:
        raise ValueError("Cannot resample empty BrainData object")

    target_img = None
    if target_affine is None:
        if isinstance(img, (str, Path)):
            img = nib.load(str(img))
        # Copy-on-write via _ensure_sform avoids mutating the caller's image.
        target_img = _ensure_sform(img)

    source_nifti = to_nifti(bd)
    if interpolation is None:
        interpolation = get_interpolation(bd, source_nifti)
    source_nifti = _ensure_sform(source_nifti)

    # Both branches clip spline overshoot to the source range; nilearn's own
    # defaults disagree between the two calls, so state the rule here.
    # The mask always uses nearest interpolation so that it stays binary.
    source_mask = _ensure_sform(bd.mask)
    if target_img is not None:
        resampled_nifti = resample_to_img(
            source_nifti, target_img, interpolation=interpolation, clip=True
        )
        resampled_mask = resample_to_img(
            source_mask, target_img, interpolation="nearest", clip=True
        )
    else:
        resampled_nifti = resample_img(
            source_nifti,
            target_affine=target_affine,
            interpolation=interpolation,
            clip=True,
        )
        resampled_mask = resample_img(
            source_mask,
            target_affine=target_affine,
            interpolation="nearest",
            clip=True,
        )

    if not np.any(resampled_mask.get_fdata() > 0):
        raise ValueError(
            f"Resampling to {target_description} leaves no voxels: the mask's "
            "support does not survive on the target grid. Choose a finer "
            "resolution or a target grid that overlaps the data."
        )

    resampled_data = nilearn_apply_mask(resampled_nifti, resampled_mask)
    return _result_with_mask(bd, resampled_data, resampled_mask, rows="preserve")


def write_brain_data(bd, file_name):
    """Write out BrainData object to Nifti or HDF5 File.

    Args:
        bd (BrainData): Instance to write.
        file_name (str | Path): Output file path. Supports `.nii`/`.nii.gz` (NIfTI)
            and `.h5`/`.hdf5` (HDF5) formats.
    """
    from nltools.io import is_h5_path, to_h5

    if isinstance(file_name, Path):
        file_name = str(file_name)

    if is_h5_path(file_name):
        to_h5(
            bd,
            file_name,
            obj_type="brain_data",
            h5_compression=bd._h5_compression,
        )
    else:
        to_nifti(bd).to_filename(file_name)


def upload_neurovault(  # nosemgrep: kwargs-internal-forwarding  # forwards to the NeuroVault API
    bd,
    *,
    access_token=None,
    collection_name=None,
    collection_id=None,
    img_type=None,
    img_modality=None,
    **kwargs,
):
    """Upload data to NeuroVault.

    Adds any columns in `bd.X` to image metadata. Index will be used as image name.

    Args:
        bd (BrainData): Images to upload.
        access_token (str): NeuroVault API access token. Required.
        collection_name (str | None): Name of a new collection to create.
        collection_id (int | None): NeuroVault collection ID when adding images
            to an existing collection.
        img_type (str): NeuroVault map type (e.g. `'Z'`, `'T'`). Required.
        img_modality (str): NeuroVault image modality (e.g. `'fMRI-BOLD'`). Required.
        **kwargs (dict): Additional image metadata forwarded to
            `pynv.Client.add_image`.

    Returns:
        dict: NeuroVault collection information.
    """
    from pynv import Client

    if access_token is None:
        raise ValueError("You must supply a valid neurovault access token")

    if img_type is None:
        raise ValueError(
            "You must supply img_type (the NeuroVault map type, e.g. 'Z' or 'T')"
        )

    if img_modality is None:
        raise ValueError(
            "You must supply img_modality (the NeuroVault image modality, e.g. 'fMRI-BOLD')"
        )

    api = Client(access_token=access_token)

    # Check if collection exists
    if collection_id is not None:
        collection = api.get_collection(collection_id)
    else:
        try:
            collection = api.create_collection(collection_name)
        except ValueError as e:
            raise ValueError(
                "Collection Name already exists. Pick a "
                "different name or specify an existing collection id"
            ) from e

    # mkdtemp guarantees a unique dir (the old os.times() name could collide).
    tmp_dir = tempfile.mkdtemp()

    def add_image_to_collection(  # nosemgrep: kwargs-internal-forwarding  # forwards to the NeuroVault API
        api, collection, dat, tmp_dir, index_id=0, **kwargs
    ):
        """Upload an image to a NeuroVault collection.

        Args:
            api (pynv.Client): Authenticated NeuroVault client.
            collection (dict): Collection the image is added to.
            dat (BrainData): Single-image BrainData instance to upload.
            tmp_dir (str): Directory the image is written to before upload.
            index_id (int): Index used to name the uploaded file.
        """
        if (len(dat.shape) > 1) & (dat.shape[0] > 1):
            raise ValueError('"dat" must be a single image.')
        img_name = collection["name"] + "_" + str(index_id) + ".nii.gz"
        f_path = os.path.join(tmp_dir, img_name)
        dat.write(f_path)
        if not dat.X.is_empty():
            # .X is a 1-row polars DataFrame of per-image metadata; expand
            # its columns into the Neurovault upload kwargs.
            row = dat.X.row(0)
            kwargs.update(dict(zip(dat.X.columns, row)))
        api.add_image(
            collection["id"],
            f_path,
            name=img_name,
            modality=img_modality,
            map_type=img_type,
            **kwargs,
        )

    if len(bd.shape) == 1:
        add_image_to_collection(api, collection, bd, tmp_dir, index_id=0, **kwargs)
    else:
        for i, x in enumerate(bd):
            add_image_to_collection(api, collection, x, tmp_dir, index_id=i, **kwargs)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return collection
