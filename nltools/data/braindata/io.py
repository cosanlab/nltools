"""
BrainData I/O and loading functions.

Standalone functions extracted from BrainData class methods for mask initialization,
data loading (from files, lists, URLs, HDF5, other BrainData objects), resampling,
writing, and uploading.
"""

import os
import re
import shutil
import tempfile
import warnings

import numpy as np
from pathlib import Path


def _detect_interpolation(img):
    """Detect appropriate interpolation method based on image data type.

    Determines whether an image contains discrete (atlas/label) or continuous
    data by checking if values are integers and counting unique values.

    Args:
        img: nibabel Nifti1Image or similar image object with get_fdata() method

    Returns:
        str: 'nearest' for discrete/atlas data, 'continuous' for continuous data

    Notes:
        - Returns 'nearest' if all non-NaN values are integers AND unique count < 1000
        - Atlases typically have < 500 unique integer labels
        - Statistical maps have continuous floating-point values
    """
    data = img.get_fdata()

    # Handle empty or all-NaN data
    valid_data = data[~np.isnan(data)]
    if valid_data.size == 0:
        return "continuous"

    # Check if all values are effectively integers
    is_integer_valued = np.allclose(valid_data, np.round(valid_data), rtol=1e-10)

    if is_integer_valued:
        n_unique = len(np.unique(valid_data))
        # Atlases typically have < 1000 unique labels (most have < 500)
        # Continuous data would have many more unique values
        if n_unique < 1000:
            return "nearest"

    return "continuous"


def initialize_mask(bd, mask, **kwargs):
    """Initialize the mask and NiftiMasker.

    Args:
        bd: BrainData instance.
        mask: Brain mask as nibabel object, file path, template name string, or None.
            Template name strings supported: '{res}mm-MNI152-2009{version}'
            (e.g., '2mm-MNI152-2009c', '3mm-MNI152-2009a', '2mm-MNI152-2009fsl')
        **kwargs: Additional arguments passed to NiftiMasker.
    """
    import nibabel as nib
    from nilearn.maskers import NiftiMasker
    from nltools.prefs import MNI_Template

    # Store whether mask was None (for auto-detection later)
    bd._mask_was_none = mask is None

    if mask is None:
        # For empty BrainData or when data not yet loaded, use default template
        # Template will be auto-detected during data loading if data is provided
        bd.mask = nib.load(MNI_Template.mask)
        bd._detected_template = None  # Will be set during data loading if needed
    elif isinstance(mask, (str, Path)):
        mask_str = str(mask)
        # Check if it's a template name string (format: {res}mm-MNI152-2009{version})
        if re.match(r"^\d+mm-MNI152-2009[acfsl]+$", mask_str):
            # Resolve template name to file path
            from nltools.prefs import resolve_template_name

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

    # Learn 3d/4d -> 1d/2d transform on template/mask
    bd.nifti_masker = NiftiMasker(
        mask_img=bd.mask, verbose=kwargs.get("verbose", 0), **kwargs
    )
    bd.nifti_masker.fit()

    # Extract voxel resolution from mask affine matrix
    # The diagonal elements of the affine matrix (excluding translation) give voxel sizes
    affine = bd.mask.affine
    bd._voxel_resolution = np.abs(np.diag(affine[:3, :3]))

    # Determine space (MNI or native) based on mask
    bd._space = detect_space(bd, bd.mask)


def get_interpolation(bd, img):
    """Get the interpolation method to use for a given image.

    Resolves 'auto' to either 'nearest' or 'continuous' based on data type.

    Args:
        bd: BrainData instance.
        img: nibabel image to check (used when interpolation='auto')

    Returns:
        str: Interpolation method. When 'auto', resolves to 'nearest' or
            'continuous' based on data type. Otherwise returns the instance's
            configured interpolation setting.
    """
    if bd._interpolation == "auto":
        return _detect_interpolation(img)
    return bd._interpolation


def detect_and_update_mask(bd, data_img):
    """Detect best matching template from data and update mask if mask was None.

    Also handles resampling if needed based on the resample kwarg.

    This function is called during data loading to auto-detect template when mask=None.
    After detecting or falling back to a template, it checks if resampling is needed
    and resamples the data_img accordingly.

    Args:
        bd: BrainData instance.
        data_img: nibabel Nifti1Image object from which to detect template

    Returns:
        nibabel.Nifti1Image: The data_img, possibly resampled to match the mask
    """
    import nibabel as nib
    from nilearn.maskers import NiftiMasker

    if not bd._mask_was_none:
        # Mask was explicitly provided, don't auto-detect
        # Still check if resampling is needed based on resample kwarg
        if not check_space_match(data_img, bd.mask):
            if bd._resample:
                # Warn about resampling
                warn_if_resampling(bd)
                # Resample data to mask space
                from nilearn.image import resample_to_img
                from contextlib import redirect_stdout

                if not bd.verbose:
                    with open(os.devnull, "w") as devnull:
                        with redirect_stdout(devnull):
                            data_img = resample_to_img(
                                data_img,
                                bd.mask,
                                interpolation=get_interpolation(bd, data_img),
                                copy_header=True,
                                force_resample=True,
                            )
                else:
                    data_img = resample_to_img(
                        data_img,
                        bd.mask,
                        interpolation=get_interpolation(bd, data_img),
                        copy_header=True,
                    )
        return data_img

    try:
        from nltools.prefs import MNI_Template

        # Detect template from data
        template_info = MNI_Template.match_resolution(
            data_img.affine,
            prefer_exact=True,
            warn_resample=bd._resample,
        )

        # Store detected template info
        bd._detected_template = template_info

        # Load detected template mask
        detected_mask = nib.load(template_info["mask_path"])

        # Check if detected mask differs from current mask
        current_mask_path = bd.mask.get_filename()
        detected_mask_path = template_info["mask_path"]

        if current_mask_path != detected_mask_path:
            # Update mask to detected template
            bd.mask = detected_mask

            # Re-initialize masker with new mask
            bd.nifti_masker = NiftiMasker(
                mask_img=bd.mask,
                verbose=getattr(bd, "verbose", False),
            )
            bd.nifti_masker.fit()

            # Update voxel resolution
            affine = bd.mask.affine
            bd._voxel_resolution = np.abs(np.diag(affine[:3, :3]))

            # Update space detection
            bd._space = detect_space(bd, bd.mask)

        # Always check if resampling is needed (regardless of whether mask changed)
        # Data might be in different space even if template matches
        if not check_space_match(data_img, bd.mask):
            if bd._resample:
                # Warn about resampling
                warn_if_resampling(
                    bd,
                    f"Detected template ({template_info['template']} {template_info['resolution']}mm) differs from data resolution.",
                )
                # Resample data to detected mask space
                from nilearn.image import resample_to_img
                from contextlib import redirect_stdout

                if not bd.verbose:
                    with open(os.devnull, "w") as devnull:
                        with redirect_stdout(devnull):
                            data_img = resample_to_img(
                                data_img,
                                bd.mask,
                                interpolation=get_interpolation(bd, data_img),
                                copy_header=True,
                                force_resample=True,
                            )
                else:
                    data_img = resample_to_img(
                        data_img,
                        bd.mask,
                        interpolation=get_interpolation(bd, data_img),
                        copy_header=True,
                    )
            else:
                # resample=False but spaces don't match - error will be raised in caller
                pass

    except Exception as e:
        # If detection fails, fall back to default template
        # This maintains backward compatibility
        warnings.warn(
            f"Failed to auto-detect template from data: {e}. "
            f"Using default template (MNI_Template.mask).",
            UserWarning,
            stacklevel=3,
        )

        # After falling back to default template, check if resampling is needed
        if not check_space_match(data_img, bd.mask):
            if bd._resample:
                # Warn about resampling
                warn_if_resampling(
                    bd,
                    "Template auto-detection failed; using default template.",
                )
                # Resample data to default mask space
                from nilearn.image import resample_to_img
                from contextlib import redirect_stdout

                if not bd.verbose:
                    with open(os.devnull, "w") as devnull:
                        with redirect_stdout(devnull):
                            data_img = resample_to_img(
                                data_img,
                                bd.mask,
                                interpolation=get_interpolation(bd, data_img),
                                copy_header=True,
                                force_resample=True,
                            )
                else:
                    data_img = resample_to_img(
                        data_img,
                        bd.mask,
                        interpolation=get_interpolation(bd, data_img),
                        copy_header=True,
                    )
            else:
                # resample=False but spaces don't match - error will be raised in caller
                pass

    return data_img


def detect_space(bd, mask):
    """Detect if mask is in MNI space or native space.

    Args:
        bd: BrainData instance (unused, kept for API consistency).
        mask: nibabel Nifti1Image object

    Returns:
        str: 'mni' if mask is MNI template, 'native' otherwise
    """
    import nibabel as nib
    from nltools.prefs import MNI_Template

    # Get mask filename if available
    mask_filename = mask.get_filename()

    # Check if mask is None (uses default MNI template)
    # This is handled in initialize_mask, but check here for safety
    if mask_filename is None:
        # Compare affine matrix with MNI template
        try:
            mni_mask = nib.load(MNI_Template.mask)
            if np.allclose(mask.affine, mni_mask.affine, rtol=1e-3):
                return "mni"
        except Exception:
            pass
        return "native"

    # Normalize paths for comparison
    mask_path = str(Path(mask_filename).resolve())
    mni_mask_path = str(Path(MNI_Template.mask).resolve())

    # Check if mask path matches MNI template path
    if mask_path == mni_mask_path:
        return "mni"

    # Check if affine matches MNI template affine (for cases where mask is loaded differently)
    try:
        mni_mask = nib.load(MNI_Template.mask)
        if np.allclose(mask.affine, mni_mask.affine, rtol=1e-3):
            return "mni"
    except Exception:
        pass

    # Default to native if not matching MNI
    return "native"


def check_space_match(data_img, mask_img):
    """Check if data and mask are in same space.

    Args:
        data_img: nibabel Nifti1Image object
        mask_img: nibabel Nifti1Image object (mask)

    Returns:
        bool: True if spaces match (no resampling needed), False otherwise
    """
    # Compare affine matrices
    affine_match = np.allclose(data_img.affine, mask_img.affine, rtol=1e-3)

    # Compare spatial shapes
    shape_match = data_img.shape[:3] == mask_img.shape[:3]

    return affine_match and shape_match


def warn_if_resampling(bd, context=""):
    """Warn about resampling if verbose=True and resample=True.

    Args:
        bd: BrainData instance.
        context (str): Context string to include in warning. Default: empty string.
    """
    if bd._resample and bd.verbose:
        base_msg = "Resampling data to match mask space (resample=True)."
        if context:
            msg = f"{base_msg} {context}"
        else:
            msg = base_msg

        warnings.warn(msg, UserWarning, stacklevel=4)


def load_from_list(bd, data_list):
    """Load data from a list of BrainData objects or file paths.

    Args:
        bd: BrainData instance.
        data_list: List of BrainData objects or file paths.
    """
    import nibabel as nib
    from nilearn.image import resample_to_img
    from contextlib import redirect_stdout
    from nltools.utils import concatenate
    from nltools.data.braindata.validation import validate_list_data

    list_type = validate_list_data(data_list)

    if list_type == "brain_data":
        # Concatenate BrainData objects
        tmp = concatenate(data_list)
        for item in ["data", "mask", "nifti_masker"]:
            setattr(bd, item, getattr(tmp, item))
    else:
        # Load files
        bd.data = []

        # Auto-detect template from first item if mask was None
        if bd._mask_was_none and len(data_list) > 0:
            first_item = data_list[0]
            if isinstance(first_item, (str, Path)):
                first_img = nib.load(str(first_item))
            elif isinstance(first_item, nib.Nifti1Image):
                first_img = first_item
            else:
                # For BrainData objects, skip detection (will be handled in concatenate)
                first_img = None

            if first_img is not None:
                # Detect template and handle resampling if needed
                first_img = detect_and_update_mask(bd, first_img)

        if not bd.verbose:
            with open(os.devnull, "w") as devnull:
                with redirect_stdout(devnull):
                    for item in data_list:
                        # Load nibabel image if file path
                        if isinstance(item, (str, Path)):
                            item_img = nib.load(str(item))
                        elif isinstance(item, nib.Nifti1Image):
                            item_img = item
                        else:
                            raise TypeError(
                                f"List items must be file paths or nibabel Nifti1Image. "
                                f"Received {type(item).__name__}"
                            )

                        # Check if resampling is needed
                        if bd._resample:
                            if not check_space_match(item_img, bd.mask):
                                # Warn about resampling (only once for first item)
                                if item == data_list[0]:
                                    warn_if_resampling(bd)
                                item_img = resample_to_img(
                                    item_img,
                                    bd.mask,
                                    interpolation=get_interpolation(bd, item_img),
                                    copy_header=True,
                                    force_resample=True,
                                )
                        else:
                            # Check if spaces match when resample=False
                            if not check_space_match(item_img, bd.mask):
                                raise ValueError(
                                    f"Data item and mask are in different spaces. "
                                    f"Set resample=True to automatically resample data to mask space, "
                                    f"or ensure all data items are already in the same space as the mask.\n"
                                    f"Item affine:\n{item_img.affine}\n"
                                    f"Mask affine:\n{bd.mask.affine}\n"
                                    f"Item shape: {item_img.shape[:3]}\n"
                                    f"Mask shape: {bd.mask.shape[:3]}"
                                )

                        bd.data.append(bd.nifti_masker.transform(item_img))
        else:
            for item in data_list:
                # Load nibabel image if file path
                if isinstance(item, (str, Path)):
                    item_img = nib.load(str(item))
                elif isinstance(item, nib.Nifti1Image):
                    item_img = item
                else:
                    raise TypeError(
                        f"List items must be file paths or nibabel Nifti1Image. "
                        f"Received {type(item).__name__}"
                    )

                # Check if resampling is needed
                if bd._resample:
                    if not check_space_match(item_img, bd.mask):
                        # Warn about resampling (only once for first item)
                        if item == data_list[0]:
                            warn_if_resampling(bd)
                        item_img = resample_to_img(
                            item_img,
                            bd.mask,
                            interpolation=get_interpolation(bd, item_img),
                            copy_header=True,
                        )
                else:
                    # Check if spaces match when resample=False
                    if not check_space_match(item_img, bd.mask):
                        raise ValueError(
                            f"Data item and mask are in different spaces. "
                            f"Set resample=True to automatically resample data to mask space, "
                            f"or ensure all data items are already in the same space as the mask.\n"
                            f"Item affine:\n{item_img.affine}\n"
                            f"Mask affine:\n{bd.mask.affine}\n"
                            f"Item shape: {item_img.shape[:3]}\n"
                            f"Mask shape: {bd.mask.shape[:3]}"
                        )

                bd.data.append(bd.nifti_masker.transform(item_img))
        # Use vstack for nilearn 0.12+ compatibility (transforms 3D -> 1D instead of 3D -> 2D)
        bd.data = np.vstack(bd.data)


def load_from_brain_data(bd, brain_data, mask=None):
    """Load data from another BrainData object.

    Args:
        bd: BrainData instance.
        brain_data: BrainData object to copy from.
        mask: Optional mask to use. If None, uses mask from brain_data.
    """
    import nibabel as nib
    from nilearn.image import resample_to_img
    from nilearn.maskers import NiftiMasker

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
                from nltools.prefs import resolve_template_name

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
                # Warn about resampling
                warn_if_resampling(bd, "New mask differs from source BrainData mask.")
                # Convert data back to nifti, resample, then mask
                source_nifti = brain_data.to_nifti()
                resampled_nifti = resample_to_img(
                    source_nifti,
                    new_mask,
                    interpolation=get_interpolation(bd, source_nifti),
                    copy_header=True,
                    force_resample=True,
                )
                # Update mask
                bd.mask = new_mask
                bd.nifti_masker = NiftiMasker(mask_img=bd.mask).fit()
                # Extract and transform data
                bd.data = bd.nifti_masker.transform(resampled_nifti)
                # Update voxel resolution and space
                affine = bd.mask.affine
                bd._voxel_resolution = np.abs(np.diag(affine[:3, :3]))
                bd._space = detect_space(bd, bd.mask)
            else:
                raise ValueError(
                    "Source BrainData mask and provided mask are in different spaces. "
                    "Set resample=True to automatically resample data to new mask space."
                )
        else:
            # Masks match - just update mask reference
            bd.mask = new_mask
            bd.nifti_masker = NiftiMasker(mask_img=bd.mask).fit()
            affine = bd.mask.affine
            bd._voxel_resolution = np.abs(np.diag(affine[:3, :3]))
            bd._space = detect_space(bd, bd.mask)
    else:
        # Use source mask - copy mask and masker
        bd.mask = brain_data.mask
        bd.nifti_masker = brain_data.nifti_masker
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
        bd: BrainData instance.
        file_path: Path to HDF5 file.
        mask: User-specified mask (to determine if we should load mask from file).
    """
    from nilearn.maskers import NiftiMasker
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
        bd.nifti_masker = NiftiMasker(bd.mask).fit(bd.mask)
        # Extract voxel resolution from mask affine matrix
        affine = bd.mask.affine
        bd._voxel_resolution = np.abs(np.diag(affine[:3, :3]))
        # Determine space (MNI or native) based on mask
        bd._space = detect_space(bd, bd.mask)
    elif mask is not None and not h5_data.get("load_mask", True):
        warnings.warn(
            "Existing mask found in HDF5 file but is being ignored because "
            "you passed a value for mask. Set mask=None to use existing "
            "mask in the HDF5 file"
        )

    # Log if we used legacy format
    if h5_data.get("legacy_format", False) and bd.verbose:
        warnings.warn("Loaded data using legacy HDF5 format")


def load_from_url(bd, url):
    """Load data from URL.

    Args:
        bd: BrainData instance.
        url: URL to download data from.
    """
    import nibabel as nib
    from nltools.datasets import download_nifti

    tmp_dir = os.path.join(tempfile.gettempdir(), str(os.times()[-1]))
    os.makedirs(tmp_dir)
    downloaded_file = nib.load(download_nifti(url, data_dir=tmp_dir))
    load_from_file(bd, downloaded_file)


def load_from_file(bd, data):
    """Load data from file path or nibabel object.

    Args:
        bd: BrainData instance.
        data: File path or nibabel object.
    """
    import nibabel as nib
    from nilearn.image import resample_to_img
    from contextlib import redirect_stdout

    # Load nibabel image if file path
    if isinstance(data, (str, Path)):
        data_img = nib.load(str(data))
    elif isinstance(data, nib.Nifti1Image):
        data_img = data
    else:
        raise TypeError(
            f"data must be a file path or nibabel Nifti1Image. "
            f"Received {type(data).__name__}"
        )

    # Auto-detect template from data if mask was None
    # This also handles resampling if needed
    data_img = detect_and_update_mask(bd, data_img)

    # Check if resampling is needed (for resample=False case)
    if not bd._resample:
        # Check if spaces match when resample=False
        if not check_space_match(data_img, bd.mask):
            # Warn instead of raising error, but still resample to avoid data corruption
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
                    UserWarning,
                    stacklevel=2,
                )

            # Resample data to mask space (required for correct processing)
            if not bd.verbose:
                with open(os.devnull, "w") as devnull:
                    with redirect_stdout(devnull):
                        data_img = resample_to_img(
                            data_img,
                            bd.mask,
                            interpolation=get_interpolation(bd, data_img),
                            copy_header=True,
                            force_resample=True,
                        )
            else:
                data_img = resample_to_img(
                    data_img,
                    bd.mask,
                    interpolation=get_interpolation(bd, data_img),
                    copy_header=True,
                )

    # Transform data using masker
    if not bd.verbose:
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull):
                bd.data = bd.nifti_masker.transform(data_img)
    else:
        bd.data = bd.nifti_masker.transform(data_img)


def to_nifti(bd):
    """Convert BrainData instance to a nibabel NIfTI image.

    Args:
        bd: BrainData instance.

    Returns:
        nibabel.Nifti1Image: Brain data in volumetric NIfTI format.
    """
    return bd.nifti_masker.inverse_transform(bd.data)


def resample_to(bd, img=None, resolution=None, interpolation=None):
    """Resample BrainData to match target image or resolution.

    Args:
        bd: BrainData instance.
        img: Target image for resampling. Can be:
            - nibabel Nifti1Image object
            - str/Path to .nii/.nii.gz file
            - None (if using resolution parameter)
        resolution: Target voxel size in mm. Can be:
            - float/int: Isotropic resolution (e.g., 2.0 = 2mm^3)
            - None (if using img parameter)
        interpolation: Interpolation method for resampling. Can be:
            - None (default): Uses instance's interpolation setting
            - 'nearest': Nearest-neighbor (for atlases, masks, labels)
            - 'linear': Linear interpolation
            - 'continuous': Higher-order spline (for stat maps)

    Returns:
        BrainData: New BrainData instance with resampled data

    Raises:
        ValueError: If both img and resolution are None, or both are provided
        TypeError: If img is not a valid image type
    """
    import nibabel as nib
    from nilearn.image import resample_to_img, resample_img

    # Validate inputs
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

    # Check for empty BrainData
    if len(bd) == 0:
        raise ValueError("Cannot resample empty BrainData object")

    # Convert current BrainData to nifti
    source_nifti = to_nifti(bd)

    # Resolve interpolation: None uses instance setting with auto-detection
    if interpolation is None:
        interpolation = get_interpolation(bd, source_nifti)

    # Ensure source image has proper sform/qform in header (fixes nilearn warning)
    # If sform is not set (code=0), set it from affine to ensure correct resampling
    # Note: We use code=2 (NIFTI_XFORM_ALIGNED_ANAT) because:
    # 1. We're resampling/aligning to another image or resolution
    # 2. Nilearn always sets sform_code=2 during resampling anyway
    # 3. This matches NIfTI best practices for aligned anatomical images
    sform_code = source_nifti.header.get_sform(coded=True)[1]
    if sform_code == 0:
        source_nifti.header.set_sform(source_nifti.affine, code=2)

    if img is not None:
        # Resample to target image
        # Validate img type
        if not isinstance(img, (str, Path, nib.Nifti1Image)):
            raise TypeError(
                f"img must be nibabel Nifti1Image, file path (str/Path), or None. "
                f"Got {type(img).__name__}"
            )

        # Resample - resample_to_img can handle file paths directly
        resampled_nifti = resample_to_img(
            source_nifti,
            img,  # Can be file path or nibabel image
            interpolation=interpolation,
            copy_header=True,
            force_resample=True,
        )

        # For mask, we need to load the image if it's a file path
        # (since we need to create a masker with it)
        if isinstance(img, (str, Path)):
            target_img = nib.load(str(img))
        else:
            target_img = img

        # Ensure target image has proper sform if it's a nibabel object
        # Use code=2 (NIFTI_XFORM_ALIGNED_ANAT) as per nilearn's behavior
        if isinstance(target_img, nib.Nifti1Image):
            target_sform_code = target_img.header.get_sform(coded=True)[1]
            if target_sform_code == 0:
                target_img.header.set_sform(target_img.affine, code=2)

        # Lazy import to avoid circular dependency
        from nltools.data.braindata import BrainData

        return BrainData(resampled_nifti, mask=target_img, resample=False)

    else:  # resolution is not None
        # Resample to specified resolution
        resolution = float(resolution)
        if resolution <= 0:
            raise ValueError(f"resolution must be positive. Got {resolution}")

        # Create target affine with specified resolution (diagonal matrix)
        # resample_img automatically calculates output shape and origin
        target_affine = np.eye(4)
        target_affine[:3, :3] = np.diag([resolution, resolution, resolution])

        # Resample data
        resampled_nifti = resample_img(
            source_nifti,
            target_affine=target_affine,
            interpolation=interpolation,
            copy=True,
            copy_header=True,
            force_resample=True,
        )

        # Resample mask with nearest interpolation (preserves binary nature)
        # Ensure mask has proper sform before resampling (code=2 matches nilearn)
        mask_sform_code = bd.mask.header.get_sform(coded=True)[1]
        if mask_sform_code == 0:
            bd.mask.header.set_sform(bd.mask.affine, code=2)

        resampled_mask = resample_img(
            bd.mask,
            target_affine=target_affine,
            interpolation="nearest",  # Use nearest for binary masks
            copy=True,
            copy_header=True,
            force_resample=True,
        )

        # Preserve X and Y metadata if present
        kwargs = {"mask": resampled_mask, "resample": False}
        if hasattr(bd, "X") and bd.X is not None:
            kwargs["X"] = bd.X
        if hasattr(bd, "Y") and bd.Y is not None:
            kwargs["Y"] = bd.Y

        # Lazy import to avoid circular dependency
        from nltools.data.braindata import BrainData

        return BrainData(resampled_nifti, **kwargs)


def write_brain_data(bd, file_name):
    """Write out BrainData object to Nifti or HDF5 File.

    Args:
        bd: BrainData instance.
        file_name (str or Path): Output file path. Supports .nii/.nii.gz (NIfTI)
            and .h5/.hdf5 (HDF5) formats.
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


def upload_neurovault(
    bd,
    access_token=None,
    collection_name=None,
    collection_id=None,
    img_type=None,
    img_modality=None,
    **kwargs,
):
    """Upload data to NeuroVault.

    Adds any columns in bd.X to image metadata. Index will be used as image name.

    Args:
        bd: BrainData instance.
        access_token (str): NeuroVault API access token. Required.
        collection_name (str, optional): Name of new collection to create.
        collection_id (int, optional): NeuroVault collection ID if adding images
            to an existing collection.
        img_type (str): NeuroVault map type. Required.
        img_modality (str): NeuroVault image modality. Required.
        **kwargs: Additional keyword arguments passed to the NeuroVault API.

    Returns:
        dict: NeuroVault collection information.
    """
    from pynv import Client

    if access_token is None:
        raise ValueError("You must supply a valid neurovault access token")

    api = Client(access_token=access_token)

    # Check if collection exists
    if collection_id is not None:
        collection = api.get_collection(collection_id)
    else:
        try:
            collection = api.create_collection(collection_name)
        except ValueError:
            print(
                "Collection Name already exists.  Pick a "
                "different name or specify an existing collection id"
            )

    tmp_dir = os.path.join(tempfile.gettempdir(), str(os.times()[-1]))
    os.makedirs(tmp_dir)

    def add_image_to_collection(api, collection, dat, tmp_dir, index_id=0, **kwargs):
        """Upload image to collection
        Args:
            api: pynv Client instance
            collection: collection information
            dat: BrainData instance to upload
            tmp_dir: temporary directory
            index_id: (int) index for file naming
        """
        if (len(dat.shape) > 1) & (dat.shape[0] > 1):
            raise ValueError('"dat" must be a single image.')
        if not dat.X.empty and isinstance(dat.X.name, str):
            img_name = dat.X.name
        else:
            img_name = collection["name"] + "_" + str(index_id) + ".nii.gz"
        f_path = os.path.join(tmp_dir, img_name)
        dat.write(f_path)
        if not dat.X.empty:
            kwargs.update(dict([(k, dat.X.loc[k]) for k in dat.X.keys()]))
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
