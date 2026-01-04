"""
NeuroLearn Brain Data
=====================

Classes to represent brain image data.

"""

from nilearn.signal import clean
from scipy.spatial.distance import cdist
from scipy.signal import detrend
from scipy.interpolate import pchip
import os
import shutil
import re
import nibabel as nib
import numpy as np
import pandas as pd
import tempfile
import warnings  # noqa: F401
from copy import deepcopy
from sklearn.preprocessing import scale
from pynv import Client
from nilearn.maskers import NiftiMasker
from nilearn.image import smooth_img, resample_to_img, resample_img
from nilearn.masking import intersect_masks
from nilearn.regions import connected_regions, connected_label_regions
from nltools.utils import (
    attempt_to_import,
    concatenate,
    set_decomposition_algorithm,
    check_brain_data,
    check_brain_data_is_single,
    to_h5,
)
from nltools.stats import (
    fisher_r_to_z,
    fisher_z_to_r,
    transform_pairwise,
    procrustes,
    find_spikes,
    compute_similarity,
    compute_multivariate_similarity,
)
from .adjacency import Adjacency
from nltools.prefs import MNI_Template
from pathlib import Path
from contextlib import redirect_stdout


warnings.filterwarnings("ignore", category=UserWarning, module="nilearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="nilearn")

# Optional dependencies
nx = attempt_to_import("networkx", "nx")
tables = attempt_to_import("tables")
MAX_INT = np.iinfo(np.int32).max


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


class BrainData(object):
    """
    BrainData is a class to represent neuroimaging data in python as a vector
    rather than a 3-dimensional matrix.This makes it easier to perform data
    manipulation and analyses.

    Args:
        data: nibabel data instance or list of files
        Y: Pandas DataFrame of training labels
        X: Pandas DataFrame Design Matrix for running univariate models
        mask: binary nifiti file to mask brain data
        **kwargs: Additional keyword arguments to pass to the prediction
                algorithm

    """

    def __init__(self, data=None, Y=None, X=None, mask=None, masker=None, **kwargs):
        """Initialize BrainData object.

        Args:
            data: Neuroimaging data. Can be:
                - None (empty BrainData)
                - BrainData object
                - List of BrainData objects or file paths
                - File path (str/Path) to .nii/.nii.gz/.h5/.hdf5
                - nibabel Nifti1Image object
                - URL to download data from
            mask: Brain mask. Can be:
                - None (uses MNI template with auto-detection)
                - nibabel Nifti1Image object
                - File path (str/Path) to mask file
                - Template name string: '{res}mm-MNI152-2009{version}'
                  (e.g., '2mm-MNI152-2009c', '3mm-MNI152-2009a', '2mm-MNI152-2009fsl')
                  where version: 'fsl' for default/, 'a' for nilearn/, 'c' for fmriprep/
            masker: nilearn masker object (e.g. ROI or searchlight extractor); Default will load data as voxels
            resample (bool, default=True): Whether to automatically resample data to mask space.
                - If True: Data is resampled to match mask spatial characteristics (affine, shape)
                - If False: Data must already be in mask space (faster, but may raise errors if spaces don't match)
                - Default True preserves backward compatibility with v0.5.1 behavior
            interpolation (str, default='auto'): Interpolation method for resampling.
                - 'auto': Automatically detect based on data type (recommended)
                    - Uses 'nearest' for discrete data (atlases, masks, labels)
                    - Uses 'continuous' for continuous data (stat maps, beta images)
                - 'nearest': Nearest-neighbor, preserves discrete values (use for atlases)
                - 'linear': Linear interpolation
                - 'continuous': Higher-order spline interpolation (use for stat maps)
            **kwargs: Additional arguments passed to NiftiMasker.
        """
        # Import validation functions
        from ._validation import validate_data_type

        # Initialize attributes
        self._h5_compression = kwargs.pop("h5_compression", "gzip")
        self.verbose = kwargs.pop("verbose", False)
        self._resample = kwargs.pop(
            "resample", True
        )  # Default True for backward compatibility
        self._interpolation = kwargs.pop(
            "interpolation", "auto"
        )  # 'auto', 'nearest', 'linear', or 'continuous'
        # Validate interpolation parameter
        valid_interpolations = ("auto", "nearest", "linear", "continuous")
        if self._interpolation not in valid_interpolations:
            raise ValueError(
                f"interpolation must be one of {valid_interpolations}, "
                f"got '{self._interpolation}'"
            )
        self.design_matrix = None
        self.masker = masker
        self._labels = None

        # Initialize mask
        self._initialize_mask(mask, **kwargs)

        # Initialize data based on type
        data_type = validate_data_type(data)

        if data_type == "none":
            self.data = np.array([])
        elif data_type == "brain_data":
            # Copy from another BrainData object
            # Note: mask was already initialized, but we may need to override it
            self._load_from_brain_data(data, mask)
        elif data_type == "h5":
            self._load_from_h5(data, mask)
            # H5 loading sets X and Y, so we're done
            return
        elif data_type == "list":
            self._load_from_list(data)
        elif data_type == "url":
            self._load_from_url(data)
        elif data_type in ["file", "nibabel"]:
            self._load_from_file(data)

        # Collapse extra trailing dimensions, but preserve samples dimension for list inputs
        # List inputs should always be 2D (n_samples, n_voxels) even with 1 sample
        # Direct file inputs should be 1D (n_voxels,) for single images
        if self.data is not None and self.data.ndim > 1 and data_type != "list":
            # Only squeeze for non-list inputs (direct file, nibabel, url)
            if 1 in self.data.shape:
                self.data = self.data.squeeze()

        # Set X and Y if provided (override copied values if explicitly provided)
        if X is not None:
            self.X = X
        elif data_type == "brain_data" and hasattr(data, "X") and data.X is not None:
            self.X = data.X.copy() if hasattr(data.X, "copy") else data.X
        else:
            self.X = None

        if Y is not None:
            self.Y = Y
        elif data_type == "brain_data" and hasattr(data, "Y") and data.Y is not None:
            self.Y = data.Y.copy() if hasattr(data.Y, "copy") else data.Y
        else:
            self.Y = None

    def _initialize_mask(self, mask, **kwargs):
        """Initialize the mask and NiftiMasker.

        Args:
            mask: Brain mask as nibabel object, file path, template name string, or None.
                Template name strings supported: '{res}mm-MNI152-2009{version}'
                (e.g., '2mm-MNI152-2009c', '3mm-MNI152-2009a', '2mm-MNI152-2009fsl')
            **kwargs: Additional arguments passed to NiftiMasker.
        """
        # Store whether mask was None (for auto-detection later)
        self._mask_was_none = mask is None

        if mask is None:
            # For empty BrainData or when data not yet loaded, use default template
            # Template will be auto-detected during data loading if data is provided
            self.mask = nib.load(MNI_Template.mask)
            self._detected_template = None  # Will be set during data loading if needed
        elif isinstance(mask, (str, Path)):
            mask_str = str(mask)
            # Check if it's a template name string (format: {res}mm-MNI152-2009{version})
            if re.match(r"^\d+mm-MNI152-2009[acfsl]+$", mask_str):
                # Resolve template name to file path
                from nltools.prefs import resolve_template_name

                mask_path = resolve_template_name(mask_str, file_type="mask")
                self.mask = nib.load(mask_path)
            else:
                # Regular file path
                self.mask = nib.load(mask_str)
            self._detected_template = None  # Explicit mask provided, no auto-detection
        elif isinstance(mask, nib.Nifti1Image):
            self.mask = mask
            self._detected_template = None  # Explicit mask provided, no auto-detection
        else:
            raise TypeError(
                f"mask must be a nibabel instance, file path, template name string, or None. "
                f"Received {type(mask).__name__}"
            )

        # Learn 3d/4d -> 1d/2d transform on template/mask
        self.nifti_masker = NiftiMasker(
            mask_img=self.mask, verbose=kwargs.get("verbose", 0), **kwargs
        )
        self.nifti_masker.fit()

        # Extract voxel resolution from mask affine matrix
        # The diagonal elements of the affine matrix (excluding translation) give voxel sizes
        affine = self.mask.affine
        self._voxel_resolution = np.abs(np.diag(affine[:3, :3]))

        # Determine space (MNI or native) based on mask
        self._space = self._detect_space(self.mask)

    def _get_interpolation(self, img):
        """Get the interpolation method to use for a given image.

        Resolves 'auto' to either 'nearest' or 'continuous' based on data type.

        Args:
            img: nibabel image to check (used when interpolation='auto')

        Returns:
            str: 'nearest', 'linear', or 'continuous'
        """
        if self._interpolation == "auto":
            return _detect_interpolation(img)
        return self._interpolation

    def _detect_and_update_mask(self, data_img):
        """Detect best matching template from data and update mask if mask was None.

        Also handles resampling if needed based on the resample kwarg.

        This method is called during data loading to auto-detect template when mask=None.
        After detecting or falling back to a template, it checks if resampling is needed
        and resamples the data_img accordingly.

        Args:
            data_img: nibabel Nifti1Image object from which to detect template

        Returns:
            nibabel.Nifti1Image: The data_img, possibly resampled to match the mask
        """
        if not self._mask_was_none:
            # Mask was explicitly provided, don't auto-detect
            # Still check if resampling is needed based on resample kwarg
            if not self._check_space_match(data_img, self.mask):
                if self._resample:
                    # Warn about resampling
                    self._warn_if_resampling()
                    # Resample data to mask space
                    from nilearn.image import resample_to_img
                    from contextlib import redirect_stdout
                    import os

                    if not self.verbose:
                        with open(os.devnull, "w") as devnull:
                            with redirect_stdout(devnull):
                                data_img = resample_to_img(
                                    data_img,
                                    self.mask,
                                    interpolation=self._get_interpolation(data_img),
                                    copy_header=True,
                                    force_resample=True,
                                )
                    else:
                        data_img = resample_to_img(
                            data_img,
                            self.mask,
                            interpolation=self._get_interpolation(data_img),
                            copy_header=True,
                        )
            return data_img

        try:
            from nltools.utils import detect_best_matching_template

            # Detect template from data
            template_info = detect_best_matching_template(
                data_img, prefer_exact_match=True, resample_enabled=self._resample
            )

            # Store detected template info
            self._detected_template = template_info

            # Load detected template mask
            detected_mask = nib.load(template_info["mask_path"])

            # Check if detected mask differs from current mask
            current_mask_path = self.mask.get_filename()
            detected_mask_path = template_info["mask_path"]

            if current_mask_path != detected_mask_path:
                # Update mask to detected template
                self.mask = detected_mask

                # Re-initialize masker with new mask
                self.nifti_masker = NiftiMasker(
                    mask_img=self.mask,
                    verbose=getattr(self, "verbose", False),
                )
                self.nifti_masker.fit()

                # Update voxel resolution
                affine = self.mask.affine
                self._voxel_resolution = np.abs(np.diag(affine[:3, :3]))

                # Update space detection
                self._space = self._detect_space(self.mask)

            # Always check if resampling is needed (regardless of whether mask changed)
            # Data might be in different space even if template matches
            if not self._check_space_match(data_img, self.mask):
                if self._resample:
                    # Warn about resampling
                    self._warn_if_resampling(
                        f"Detected template ({template_info['template']} {template_info['resolution']}mm) differs from data resolution."
                    )
                    # Resample data to detected mask space
                    from nilearn.image import resample_to_img
                    from contextlib import redirect_stdout
                    import os

                    if not self.verbose:
                        with open(os.devnull, "w") as devnull:
                            with redirect_stdout(devnull):
                                data_img = resample_to_img(
                                    data_img,
                                    self.mask,
                                    interpolation=self._get_interpolation(data_img),
                                    copy_header=True,
                                    force_resample=True,
                                )
                    else:
                        data_img = resample_to_img(
                            data_img,
                            self.mask,
                            interpolation=self._get_interpolation(data_img),
                            copy_header=True,
                        )
                else:
                    # resample=False but spaces don't match - error will be raised in caller
                    pass

        except Exception as e:
            # If detection fails, fall back to default template
            # This maintains backward compatibility
            import warnings

            warnings.warn(
                f"Failed to auto-detect template from data: {e}. "
                f"Using default template (MNI_Template.mask).",
                UserWarning,
                stacklevel=3,
            )

            # After falling back to default template, check if resampling is needed
            if not self._check_space_match(data_img, self.mask):
                if self._resample:
                    # Warn about resampling
                    self._warn_if_resampling(
                        "Template auto-detection failed; using default template."
                    )
                    # Resample data to default mask space
                    from nilearn.image import resample_to_img
                    from contextlib import redirect_stdout
                    import os

                    if not self.verbose:
                        with open(os.devnull, "w") as devnull:
                            with redirect_stdout(devnull):
                                data_img = resample_to_img(
                                    data_img,
                                    self.mask,
                                    interpolation=self._get_interpolation(data_img),
                                    copy_header=True,
                                    force_resample=True,
                                )
                    else:
                        data_img = resample_to_img(
                            data_img,
                            self.mask,
                            interpolation=self._get_interpolation(data_img),
                            copy_header=True,
                        )
                else:
                    # resample=False but spaces don't match - error will be raised in caller
                    pass

        return data_img

    def _detect_space(self, mask):
        """Detect if mask is in MNI space or native space.

        Args:
            mask: nibabel Nifti1Image object

        Returns:
            str: 'mni' if mask is MNI template, 'native' otherwise
        """
        from nltools.prefs import MNI_Template

        # Get mask filename if available
        mask_filename = mask.get_filename()

        # Check if mask is None (uses default MNI template)
        # This is handled in _initialize_mask, but check here for safety
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

    def _check_space_match(self, data_img, mask_img):
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

    def _warn_if_resampling(self, context=""):
        """Warn about resampling if verbose=True and resample=True.

        Args:
            context: Optional context string to include in warning message.
        """
        if self._resample and self.verbose:
            import warnings

            base_msg = "Resampling data to match mask space (resample=True)."
            if context:
                msg = f"{base_msg} {context}"
            else:
                msg = base_msg

            warnings.warn(msg, UserWarning, stacklevel=4)

    def _load_from_list(self, data_list):
        """Load data from a list of BrainData objects or file paths.

        Args:
            data_list: List of BrainData objects or file paths.
        """
        from ._validation import validate_list_data

        list_type = validate_list_data(data_list)

        if list_type == "brain_data":
            # Concatenate BrainData objects
            tmp = concatenate(data_list)
            for item in ["data", "mask", "nifti_masker"]:
                setattr(self, item, getattr(tmp, item))
        else:
            # Load files
            self.data = []

            # Auto-detect template from first item if mask was None
            if self._mask_was_none and len(data_list) > 0:
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
                    first_img = self._detect_and_update_mask(first_img)

            if not self.verbose:
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
                            if self._resample:
                                if not self._check_space_match(item_img, self.mask):
                                    # Warn about resampling (only once for first item)
                                    if item == data_list[0]:
                                        self._warn_if_resampling()
                                    item_img = resample_to_img(
                                        item_img,
                                        self.mask,
                                        interpolation=self._get_interpolation(item_img),
                                        copy_header=True,
                                        force_resample=True,
                                    )
                            else:
                                # Check if spaces match when resample=False
                                if not self._check_space_match(item_img, self.mask):
                                    raise ValueError(
                                        f"Data item and mask are in different spaces. "
                                        f"Set resample=True to automatically resample data to mask space, "
                                        f"or ensure all data items are already in the same space as the mask.\n"
                                        f"Item affine:\n{item_img.affine}\n"
                                        f"Mask affine:\n{self.mask.affine}\n"
                                        f"Item shape: {item_img.shape[:3]}\n"
                                        f"Mask shape: {self.mask.shape[:3]}"
                                    )

                            self.data.append(self.nifti_masker.transform(item_img))
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
                    if self._resample:
                        if not self._check_space_match(item_img, self.mask):
                            # Warn about resampling (only once for first item)
                            if item == data_list[0]:
                                self._warn_if_resampling()
                            item_img = resample_to_img(
                                item_img,
                                self.mask,
                                interpolation=self._get_interpolation(item_img),
                                copy_header=True,
                            )
                    else:
                        # Check if spaces match when resample=False
                        if not self._check_space_match(item_img, self.mask):
                            raise ValueError(
                                f"Data item and mask are in different spaces. "
                                f"Set resample=True to automatically resample data to mask space, "
                                f"or ensure all data items are already in the same space as the mask.\n"
                                f"Item affine:\n{item_img.affine}\n"
                                f"Mask affine:\n{self.mask.affine}\n"
                                f"Item shape: {item_img.shape[:3]}\n"
                                f"Mask shape: {self.mask.shape[:3]}"
                            )

                    self.data.append(self.nifti_masker.transform(item_img))
            # Use vstack for nilearn 0.12+ compatibility (transforms 3D → 1D instead of 3D → 2D)
            self.data = np.vstack(self.data)

    def _load_from_brain_data(self, brain_data, mask=None):
        """Load data from another BrainData object.

        Args:
            brain_data: BrainData object to copy from.
            mask: Optional mask to use. If None, uses mask from brain_data.
        """
        # Copy data array
        self.data = (
            brain_data.data.copy() if brain_data.data is not None else np.array([])
        )

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

                    new_mask = nib.load(
                        resolve_template_name(mask_str, file_type="mask")
                    )
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
            if not self._check_space_match(brain_data.mask, new_mask):
                # Need to resample data to new mask space
                if self._resample:
                    # Warn about resampling
                    self._warn_if_resampling(
                        "New mask differs from source BrainData mask."
                    )
                    # Convert data back to nifti, resample, then mask
                    source_nifti = brain_data.to_nifti()
                    resampled_nifti = resample_to_img(
                        source_nifti,
                        new_mask,
                        interpolation=self._get_interpolation(source_nifti),
                        copy_header=True,
                        force_resample=True,
                    )
                    # Update mask
                    self.mask = new_mask
                    self.nifti_masker = NiftiMasker(mask_img=self.mask).fit()
                    # Extract and transform data
                    self.data = self.nifti_masker.transform(resampled_nifti)
                    # Update voxel resolution and space
                    affine = self.mask.affine
                    self._voxel_resolution = np.abs(np.diag(affine[:3, :3]))
                    self._space = self._detect_space(self.mask)
                else:
                    raise ValueError(
                        "Source BrainData mask and provided mask are in different spaces. "
                        "Set resample=True to automatically resample data to new mask space."
                    )
            else:
                # Masks match - just update mask reference
                self.mask = new_mask
                self.nifti_masker = NiftiMasker(mask_img=self.mask).fit()
                affine = self.mask.affine
                self._voxel_resolution = np.abs(np.diag(affine[:3, :3]))
                self._space = self._detect_space(self.mask)
        else:
            # Use source mask - copy mask and masker
            self.mask = brain_data.mask
            self.nifti_masker = brain_data.nifti_masker
            self._voxel_resolution = brain_data._voxel_resolution
            self._space = brain_data._space

        # Copy detected template info if present
        if hasattr(brain_data, "_detected_template"):
            self._detected_template = brain_data._detected_template
        if hasattr(brain_data, "_mask_was_none"):
            self._mask_was_none = brain_data._mask_was_none

    def _load_from_h5(self, file_path, mask):
        """Load data from HDF5 file.

        Args:
            file_path: Path to HDF5 file.
            mask: User-specified mask (to determine if we should load mask from file).
        """
        from nltools.utils import load_brain_data_h5

        # Load data using utility function
        h5_data = load_brain_data_h5(file_path, mask)
        self.data = h5_data["data"]

        # Load X and Y if present (for backward compatibility)
        if "X" in h5_data:
            self.X = h5_data["X"]
        if "Y" in h5_data:
            self.Y = h5_data["Y"]

        # Handle mask if loaded from file
        if h5_data.get("load_mask", False):
            self.mask = h5_data["mask"]
            self.nifti_masker = NiftiMasker(self.mask).fit(self.mask)
            # Extract voxel resolution from mask affine matrix
            affine = self.mask.affine
            self._voxel_resolution = np.abs(np.diag(affine[:3, :3]))
            # Determine space (MNI or native) based on mask
            self._space = self._detect_space(self.mask)
        elif mask is not None and not h5_data.get("load_mask", True):
            warnings.warn(
                "Existing mask found in HDF5 file but is being ignored because "
                "you passed a value for mask. Set mask=None to use existing "
                "mask in the HDF5 file"
            )

        # Log if we used legacy format
        if h5_data.get("legacy_format", False) and self.verbose:
            warnings.warn("Loaded data using legacy HDF5 format")

    def _load_from_url(self, url):
        """Load data from URL.

        Args:
            url: URL to download data from.
        """
        from nltools.datasets import download_nifti

        tmp_dir = os.path.join(tempfile.gettempdir(), str(os.times()[-1]))
        os.makedirs(tmp_dir)
        downloaded_file = nib.load(download_nifti(url, data_dir=tmp_dir))
        self._load_from_file(downloaded_file)

    def _load_from_file(self, data):
        """Load data from file path or nibabel object.

        Args:
            data: File path or nibabel object.
        """
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
        data_img = self._detect_and_update_mask(data_img)

        # Check if resampling is needed (for resample=False case)
        if not self._resample:
            # Check if spaces match when resample=False
            if not self._check_space_match(data_img, self.mask):
                # Warn instead of raising error, but still resample to avoid data corruption
                if self.verbose:
                    warnings.warn(
                        f"Data and mask are in different spaces (affine or shape mismatch). "
                        f"Resampling data to match mask space despite resample=False. "
                        f"Set resample=True to explicitly enable resampling, or ensure data "
                        f"is already in the same space as the mask.\n"
                        f"Data affine:\n{data_img.affine}\n"
                        f"Mask affine:\n{self.mask.affine}\n"
                        f"Data shape: {data_img.shape[:3]}\n"
                        f"Mask shape: {self.mask.shape[:3]}",
                        UserWarning,
                        stacklevel=2,
                    )

                # Resample data to mask space (required for correct processing)
                if not self.verbose:
                    with open(os.devnull, "w") as devnull:
                        with redirect_stdout(devnull):
                            data_img = resample_to_img(
                                data_img,
                                self.mask,
                                interpolation=self._get_interpolation(data_img),
                                copy_header=True,
                                force_resample=True,
                            )
                else:
                    data_img = resample_to_img(
                        data_img,
                        self.mask,
                        interpolation=self._get_interpolation(data_img),
                        copy_header=True,
                    )

        # Transform data using masker
        if not self.verbose:
            with open(os.devnull, "w") as devnull:
                with redirect_stdout(devnull):
                    self.data = self.nifti_masker.transform(data_img)
        else:
            self.data = self.nifti_masker.transform(data_img)

    def _perform_arithmetic(self, other, operation, operation_name, inplace=False):
        """Perform arithmetic operation with validation.

        Args:
            other: The other operand.
            operation: The operation function (e.g., np.add, np.subtract).
            operation_name: Name of the operation for error messages.

        Returns:
            BrainData: Result of the operation.
        """
        from ._validation import validate_arithmetic_operand, validate_brain_data_shapes

        if not inplace:
            new = self._shallow_copy_with_data()
        else:
            new = self

        operand_type = validate_arithmetic_operand(other, operation_name)

        if operand_type == "scalar":
            new.data = operation(self.data, other)
        elif operand_type == "brain_data":
            validate_brain_data_shapes(self, other, operation_name)
            new.data = operation(self.data, other.data)
        elif operand_type == "array":
            # Only for multiplication
            if len(other) != len(self):
                raise ValueError(
                    f"Vector {operation_name} requires that the length of the vector "
                    f"({len(other)}) match the number of images ({len(self)})"
                )
            new.data = np.dot(self.data.T, other).T

        return new

    def _apply_func(self, stat_func, axis=0):
        """
        Apply a function to the `.data` attribute. If axis=0, returns a `BrainData` object with the statistic calculated over samples (e.g. within a voxel over time). If axis=1, returns a numpy array with the statistic calculated over features (e.g. across voxels within a specific time-point)

        Args:
            stat_func: Statistical function to apply (e.g., np.mean, np.std).
            axis: Axis along which to compute (0=across images, 1=within images).

        Returns:
            float/np.array/BrainData: Result of statistical operation.
        """

        # Single image case
        if check_brain_data_is_single(self):
            return stat_func(self.data)

        if axis == 1:
            # Return array with statistic within each image
            return stat_func(self.data, axis=1)
        elif axis == 0:
            # Return BrainData with statistic across images
            out = self._shallow_copy_with_data()
            out.data = stat_func(self.data, axis=0)
            out.X = pd.DataFrame()
            out.Y = pd.DataFrame()
            return out
        else:
            raise ValueError("axis must be 0 or 1")

    def __repr__(self):
        mask_filename = self.mask.get_filename()
        mask_display = os.path.basename(mask_filename) if mask_filename else "None"

        # Format voxel resolution
        if hasattr(self, "_voxel_resolution") and self._voxel_resolution is not None:
            # Check if resolution is isotropic (all values equal)
            if np.allclose(self._voxel_resolution, self._voxel_resolution[0]):
                resolution_str = f"{self._voxel_resolution[0]:.1f}mm"
            else:
                # Anisotropic: show all three dimensions
                resolution_str = f"{self._voxel_resolution[0]:.1f}x{self._voxel_resolution[1]:.1f}x{self._voxel_resolution[2]:.1f}mm"
        else:
            resolution_str = "unknown"

        # Format space
        space_str = getattr(self, "_space", "unknown")

        return "%s.%s(data=%s, resolution=%s, space=%s, mask=%s)" % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.shape,
            resolution_str,
            space_str,
            mask_display,
        )

    def __getitem__(self, index):
        new = self._shallow_copy_with_data()
        if isinstance(index, (int, np.integer)):
            new.data = np.array(self.data[index, :]).squeeze()
        else:
            if isinstance(index, slice):
                new.data = self.data[index, :]
            else:
                index = np.array(index).flatten()
                new.data = np.array(self.data[index, :]).squeeze()
        if hasattr(self, "Y") and self.Y is not None and not self.Y.empty:
            new.Y = self.Y.iloc[index]
            if isinstance(new.Y, pd.Series):
                new.Y.reset_index(inplace=True, drop=True)
        if hasattr(self, "X") and self.X is not None and not self.X.empty:
            new.X = self.X.iloc[index]
            if len(new.X) > 1:
                new.X.reset_index(inplace=True, drop=True)
        return new

    def __setitem__(self, index, value):
        if not isinstance(value, BrainData):
            raise ValueError(
                "Make sure the value you are trying to set is a BrainData() instance."
            )
        self.data[index, :] = value.data
        if hasattr(value, "Y") and value.Y is not None and not value.Y.empty:
            if not hasattr(self, "Y") or self.Y is None:
                raise ValueError("Cannot set Y values: self.Y does not exist.")
            self.Y.values[index] = value.Y
        if hasattr(value, "X") and value.X is not None and not value.X.empty:
            if not hasattr(self, "X") or self.X is None:
                raise ValueError("Cannot set X values: self.X does not exist.")
            if self.X.shape[1] != value.X.shape[1]:
                raise ValueError("Make sure self.X is the same size as value.X.")
            self.X.values[index] = value.X

    def __len__(self):
        return self.shape[0]

    def __add__(self, y):
        """Add to BrainData."""
        return self._perform_arithmetic(y, np.add, "add")

    def __radd__(self, y):
        """Right add to BrainData."""
        return self._perform_arithmetic(y, np.add, "add")

    def __sub__(self, y):
        """Subtract from BrainData."""
        return self._perform_arithmetic(y, np.subtract, "subtract")

    def __rsub__(self, y):
        """Right subtract from BrainData."""
        # For right subtraction, we need to reverse the operands
        new = self._shallow_copy_with_data()
        from ._validation import validate_arithmetic_operand, validate_brain_data_shapes

        operand_type = validate_arithmetic_operand(y, "subtract")
        if operand_type == "scalar":
            new.data = y - self.data
        elif operand_type == "brain_data":
            validate_brain_data_shapes(self, y, "subtract")
            new.data = y.data - self.data
        return new

    def __mul__(self, y):
        """Multiply BrainData."""
        return self._perform_arithmetic(y, np.multiply, "multiply")

    def __rmul__(self, y):
        """Right multiply BrainData."""
        return self._perform_arithmetic(y, np.multiply, "multiply")

    def __truediv__(self, y):
        """Divide BrainData."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return self._perform_arithmetic(y, np.divide, "divide")

    def __iadd__(self, y):
        """In-place addition (+=)."""
        return self._perform_arithmetic(y, np.add, "add", inplace=True)

    def __isub__(self, y):
        """In-place subtraction (-=)."""
        return self._perform_arithmetic(y, np.subtract, "subtract", inplace=True)

    def __imul__(self, y):
        """In-place multiplication (*=)."""
        return self._perform_arithmetic(y, np.multiply, "multiply", inplace=True)

    def __itruediv__(self, y):
        """In-place true division (/=)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return self._perform_arithmetic(y, np.divide, "divide", inplace=True)

    def __iter__(self):
        for x in range(len(self)):
            yield self[x]

    def __eq__(self, other):
        """Check equality between BrainData."""
        if not isinstance(other, BrainData):
            return False

        # Compare data arrays
        eq_data = np.all(self.data == other.data)

        # Compare X DataFrames - handle None cases properly
        if self.X is None and other.X is None:
            eq_X = True
        elif self.X is None or other.X is None:
            eq_X = False
        else:
            eq_X = self.X.equals(other.X)

        # Compare Y DataFrames - handle None cases properly
        if self.Y is None and other.Y is None:
            eq_Y = True
        elif self.Y is None or other.Y is None:
            eq_Y = False
        else:
            eq_Y = self.Y.equals(other.Y)

        # Compare masks - handle Nifti images by comparing file paths
        if self.mask is None and other.mask is None:
            eq_mask = True
        elif self.mask is None or other.mask is None:
            eq_mask = False
        elif hasattr(self.mask, "get_filename") and hasattr(other.mask, "get_filename"):
            # Both are Nifti images - compare file paths
            eq_mask = self.mask.get_filename() == other.mask.get_filename()
        else:
            # Fallback to direct comparison
            eq_mask = self.mask == other.mask

        # We don't check nifti masker
        return eq_data and eq_X and eq_Y and eq_mask

    def __deepcopy__(self, memo):
        """Custom deepcopy that handles model attributes.

        Model-related attributes (model_, X_, glm_*, ridge_*) are shared
        (not copied) to avoid pickle errors with unpicklable Backend objects.
        All other attributes are deep copied.

        Args:
            memo: Dictionary to track already copied objects

        Returns:
            BrainData: New instance with copied/shared attributes
        """
        from copy import deepcopy

        # Create new instance without calling __init__
        new = BrainData.__new__(BrainData)
        memo[id(self)] = new

        # Copy all attributes with appropriate strategy
        for key, value in self.__dict__.items():
            if key in ["mask", "nifti_masker", "masker"]:
                # Share immutable/expensive objects
                setattr(new, key, value)
            elif key in ["model_", "X_"]:
                # Share fitted model and training data (unpicklable)
                setattr(new, key, value)
            elif key.startswith("glm_") or key.startswith("ridge_"):
                # Share model results (often contain BrainData objects)
                setattr(new, key, value)
            else:
                # Deep copy everything else (data, X, Y, etc.)
                setattr(new, key, deepcopy(value, memo))

        return new

    def _shallow_copy_with_data(self):
        """Create a shallow copy for efficient method chaining.

        This method creates a new BrainData instance that shares immutable objects
        (mask, nifti_masker) but copies mutable attributes. The data array is NOT
        copied - methods should handle data copying as needed.

        Returns:
            BrainData: New instance with shared/copied attributes
        """
        # Create new instance without calling __init__
        new = BrainData.__new__(BrainData)

        # Copy all attributes with appropriate strategy
        for key, value in self.__dict__.items():
            if key == "data":
                # Don't copy data array - let methods handle this
                new.data = self.data  # Just reference for now
            elif key in ["mask", "nifti_masker", "masker"]:
                # Share immutable/expensive objects
                setattr(new, key, value)
            elif key in ["X", "Y", "design_matrix"]:
                # Deep copy DataFrames (they're small and mutable)
                if value is not None:
                    if hasattr(value, "copy"):  # DataFrame has .copy()
                        setattr(new, key, value.copy())
                    else:
                        setattr(new, key, deepcopy(value))
                else:
                    setattr(new, key, None)
            elif key.startswith("glm_") or key.startswith("ridge_"):
                # GLM/Ridge results - share for now (they're typically read-only)
                setattr(new, key, value)
            elif key in ["model_", "X_", "cv_results_"]:
                # Fitted model state - don't propagate to copies
                # A copy represents fresh data, not a fitted model
                pass
            else:
                # Small attributes - deep copy to be safe
                setattr(new, key, deepcopy(value))

        return new

    def copy(self):
        """Create a deep copy of a BrainData instance.

        All attributes including data, fitted models, and results are deep copied.
        Use this when you need a complete independent copy.

        Note: For methods like apply_mask(), threshold(), etc., fitted model state
        (model_, X_, cv_results_) is NOT propagated since the new data shape
        would invalidate the original fit.

        Returns:
            BrainData: Deep copied instance
        """
        return deepcopy(self)

    @property
    def shape(self):
        """Get images by voxels shape."""

        return self.data.shape

    @property
    def dtype(self):
        """Get data type of BrainData.data."""
        return self.data.dtype

    def mean(self, axis=0):
        """Get mean of each voxel or image.

        Args:
            axis: Axis along which to compute mean.
                0 = across images (default), returns BrainData
                1 = within images, returns array

        Returns:
            float/np.array/BrainData: Mean values.
        """
        return self._apply_func(np.mean, axis)

    def median(self, axis=0):
        """Get median of each voxel or image.

        Args:
            axis: Axis along which to compute median.
                0 = across images (default), returns BrainData
                1 = within images, returns array

        Returns:
            float/np.array/BrainData: Median values.
        """
        return self._apply_func(np.median, axis)

    def std(self, axis=0):
        """Get standard deviation of each voxel or image.

        Args:
            axis: Axis along which to compute standard deviation.
                0 = across images (default), returns BrainData
                1 = within images, returns array

        Returns:
            float/np.array/BrainData: Standard deviation values.
        """
        return self._apply_func(np.std, axis)

    def sum(self, axis=0):
        """Get sum of each voxel or image.

        Args:
            axis: Axis along which to compute sum.
                0 = across images (default), returns BrainData
                1 = within images, returns array

        Returns:
            float/np.array/BrainData: Sum values.
        """
        return self._apply_func(np.sum, axis)

    def astype(self, dtype):
        """Cast BrainData.data as type.

        Args:
            dtype: datatype to convert

        Returns:
            BrainData: BrainData instance with new datatype

        """
        # Optimized: Use shallow copy instead of deepcopy
        out = self._shallow_copy_with_data()
        out.data = self.data.astype(dtype)
        return out

    def to_nifti(self):
        """Convert BrainData Instance into Nifti Object"""

        return self.nifti_masker.inverse_transform(self.data)

    def resample_to(self, img=None, resolution=None, interpolation=None):
        """Resample BrainData to match target image or resolution.

        Args:
            img: Target image for resampling. Can be:
                - nibabel Nifti1Image object
                - str/Path to .nii/.nii.gz file
                - None (if using resolution parameter)
            resolution: Target voxel size in mm. Can be:
                - float/int: Isotropic resolution (e.g., 2.0 = 2mm³)
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
        if len(self) == 0:
            raise ValueError("Cannot resample empty BrainData object")

        # Convert current BrainData to nifti
        source_nifti = self.to_nifti()

        # Resolve interpolation: None uses instance setting with auto-detection
        if interpolation is None:
            interpolation = self._get_interpolation(source_nifti)

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
            mask_sform_code = self.mask.header.get_sform(coded=True)[1]
            if mask_sform_code == 0:
                self.mask.header.set_sform(self.mask.affine, code=2)

            resampled_mask = resample_img(
                self.mask,
                target_affine=target_affine,
                interpolation="nearest",  # Use nearest for binary masks
                copy=True,
                copy_header=True,
                force_resample=True,
            )

            # Preserve X and Y metadata if present
            kwargs = {"mask": resampled_mask, "resample": False}
            if hasattr(self, "X") and self.X is not None:
                kwargs["X"] = self.X
            if hasattr(self, "Y") and self.Y is not None:
                kwargs["Y"] = self.Y

            return BrainData(resampled_nifti, **kwargs)

    def write(self, file_name):
        """Write out BrainData object to Nifti or HDF5 File.

        Args:
            file_name: (str) name of nifti file including path

        """

        if isinstance(file_name, Path):
            file_name = str(file_name)

        if (".h5" in file_name) or (".hdf5" in file_name):
            to_h5(
                self,
                file_name,
                obj_type="brain_data",
                h5_compression=self._h5_compression,
            )
        else:
            self.to_nifti().to_filename(file_name)

    def scale(self, scale_val=100.0, axis=None):
        """Scale data via mean scaling.

        Two scaling modes are available:

        - **Grand-mean scaling** (axis=None, default): Divides all values by the
          global mean across all voxels and timepoints. This is consistent with
          FSL and SPM behavior. Use scale_val=10000 for FSL-style scaling.

        - **Voxel-wise scaling** (axis=0): Divides each voxel's time-series by
          its own temporal mean. This is AFNI-style scaling and can be useful
          when voxels have very different baseline intensities. Voxels with
          zero or near-zero mean are set to zero to avoid NaN/Inf.

        When scale_val=100 (default), the result can be interpreted as something
        akin to (but not exactly) "percent signal change."

        Args:
            scale_val: (int/float) Target value for the mean after scaling.
                    Default 100.
            axis: (int or None) Axis along which to compute the mean:
                    - None: Grand-mean scaling (default, FSL/SPM style)
                    - 0: Voxel-wise scaling (AFNI style) - each voxel scaled
                         by its own temporal mean

        Returns:
            BrainData: New BrainData instance with scaled data.

        Examples:
            >>> # Grand-mean scaling (default)
            >>> scaled = brain.scale(100.0)
            >>>
            >>> # Voxel-wise scaling (AFNI style)
            >>> scaled = brain.scale(100.0, axis=0)

        """
        out = self._shallow_copy_with_data()
        out.data = self.data.copy()

        if axis is None:
            # Grand-mean scaling: divide by global mean
            out.data = out.data / out.data.mean() * scale_val
        elif axis == 0:
            # Voxel-wise scaling: divide each voxel by its temporal mean
            # Compute mean along time axis (axis=0), keeping dims for broadcasting
            voxel_means = out.data.mean(axis=0, keepdims=True)

            # Handle zero-mean voxels to avoid NaN/Inf
            # Set zero-mean voxels to 1 temporarily, then zero out result
            zero_mask = np.abs(voxel_means) < np.finfo(float).eps
            voxel_means_safe = np.where(zero_mask, 1.0, voxel_means)

            # Scale
            out.data = out.data / voxel_means_safe * scale_val

            # Zero out voxels that had zero mean
            if np.any(zero_mask):
                out.data[:, zero_mask.squeeze()] = 0.0
        else:
            raise ValueError(f"axis must be None or 0, got {axis}")

        return out

    def cv(
        self,
        k: int = None,
        scheme: str = "kfold",
        split_by: str = None,
        groups: np.ndarray = None,
        random_state: int = None,
        **kwargs,
    ) -> "BrainDataPipeline":
        """Create a cross-validation pipeline for this BrainData.

        Returns a Pipeline object that enables fluent, chainable transforms
        with cross-validation. Terminal methods like .predict() execute the
        pipeline and return results.

        Args:
            k: Number of folds (for kfold scheme). Defaults to 5.
            scheme: CV scheme type. Options:
                - 'kfold': k-fold cross-validation (default)
                - 'loro': leave-one-run-out (requires split_by='runs' or groups)
                - 'bootstrap': bootstrap with out-of-bag test sets
            split_by: Attribute name for group splits (e.g., 'runs').
                If provided and groups is None, will try to get groups from
                self.X[split_by] if self.X is a DataFrame.
            groups: Explicit group labels for CV splits.
            random_state: Random seed for reproducibility.
            **kwargs: Additional arguments passed to CVScheme.

        Returns:
            BrainDataPipeline: A pipeline object for method chaining.

        Examples:
            >>> # Simple 5-fold CV with prediction
            >>> result = brain.cv(k=5).predict(y, algorithm='ridge')
            >>> print(f"Mean score: {result.mean_score:.3f}")

            >>> # With preprocessing
            >>> result = (brain
            ...     .cv(k=5)
            ...     .normalize()
            ...     .reduce(n_components=50)
            ...     .predict(y))

            >>> # Leave-one-run-out CV
            >>> result = brain.cv(scheme='loro', groups=run_labels).predict(y)

        See Also:
            BrainDataPipeline: For available transforms and terminal methods.
            CVScheme: For CV scheme configuration details.
        """
        from nltools.pipelines.cv import CVScheme

        # Handle split_by -> groups conversion
        if groups is None and split_by is not None:
            if hasattr(self, "X") and self.X is not None:
                if hasattr(self.X, "__getitem__") and split_by in self.X:
                    groups = np.array(self.X[split_by])

        # Create CV scheme
        cv_scheme = CVScheme(
            k=k,
            scheme=scheme,
            split_by=split_by,
            random_state=random_state,
            **kwargs,
        )

        # Create and return pipeline
        return BrainDataPipeline(self, cv=cv_scheme, groups=groups)

    def fit(
        self,
        model=None,
        X=None,
        cv=None,
        inplace=True,
        progress_bar=None,
        scale=True,
        scale_value=100.0,
        **kwargs,
    ):
        """Fit a model to brain imaging data.

        Creates and fits a model from string specification. The brain data
        (self.data) is always used as the target variable. Model and results
        are stored for later use with predict().

        Args:
            model (str): Model type: 'ridge', 'glm', or future model names
            X (array-like or DataFrame): Design matrix or feature matrix, shape (n_samples, n_features)
                - For GLM: Design matrix with regressors (n_samples must match self.data)
                - For Ridge: Feature matrix for prediction (n_samples must match self.data)
            cv (int, 'auto', or sklearn CV splitter, optional): Cross-validation specification (Ridge only):
                - int: Number of folds for k-fold CV (returns CV scores)
                - 'auto': Triggers alpha selection via CV (implies alpha='auto')
                - sklearn CV object: Custom CV splitter (e.g., KFold(3, shuffle=True))
                - None: No CV (default, backward compatible)
            inplace (bool, default=True): If True, mutate self and return self (backward compatible).
                If False, return Fit dataclass with results (self unchanged).
            progress_bar (bool, optional): Display progress bar during fitting.
                - If None: Uses self.verbose (default)
                - If True: Shows progress bar for long-running operations
                - If False: No progress bar
            scale (bool, default=True): Apply grand-mean scaling before fitting. Calls
                self.scale(scale_value) which divides all values by the global mean
                and multiplies by scale_value. This puts data in percent signal change
                units, which is standard for fMRI analysis.
            scale_value (float, default=100.0): Target value for mean after scaling.
                Only used if scale=True.
            **kwargs (dict): Additional arguments passed to model constructor
                - Ridge: alpha, alphas, backend, random_state
                - Glm: noise_model, minimize_memory, etc.

        Returns:
            BrainData or Fit: If ``inplace=True``, returns self (fitted BrainData).
                If ``inplace=False``, returns Fit dataclass with results.

        Attributes (when inplace=True):

            ``model_`` (BaseModel): Fitted model instance (Ridge, Glm, etc.)
            ``X_`` (ndarray): Training data X, stored for predict() default
            ``cv_results_`` (dict, optional): Cross-validation results dict with keys 'scores',
            'mean_score', 'predictions', 'folds', 'best_alpha', 'alpha_scores' (if cv is not None).
            glm_betas (BrainData, optional): Beta coefficients (for model='glm')
            glm_t (BrainData, optional): T-statistics (for model='glm')
            glm_p (BrainData, optional): P-values (for model='glm')
            glm_se (BrainData, optional): Standard errors (for model='glm')
            glm_residual (BrainData, optional): Residuals (for model='glm')
            glm_predicted (BrainData, optional): Fitted values (for model='glm')
            glm_r2 (BrainData, optional): R² values (for model='glm')
            ridge_weights (BrainData, optional): Model coefficients (for model='ridge')
            ridge_fitted_values (BrainData, optional): Fitted values (for model='ridge')
            ridge_scores (BrainData, optional): R² scores (for model='ridge')

        Examples:
            >>> # Old behavior (backward compatible): mutate self
            >>> brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
            >>> print(f"CV R²: {brain_data.cv_results_['mean_score'].mean():.3f}")
            >>> weights = brain_data.ridge_weights  # Access as attribute
            >>>
            >>> # New behavior: return Fit dataclass (self unchanged)
            >>> fit = brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features, inplace=False)
            >>> assert isinstance(fit, Fit)
            >>> assert 'weights' in fit.available()
            >>> assert not hasattr(brain_data, 'ridge_weights')  # brain_data unchanged
            >>> print(f"CV R²: {fit.cv_mean_score.mean():.3f}")
            >>>
            >>> # GLM with Fit dataclass
            >>> fit_glm = brain_data.fit(model='glm', X=design_matrix, inplace=False)
            >>> assert 'betas' in fit_glm.available()
            >>> assert 't_stats' in fit_glm.available()
        """
        from nltools.models import Ridge, Glm

        # Validate inputs
        if model is None:
            raise TypeError("model must be provided")
        if X is None:
            raise TypeError("X must be provided")

        # For GLM: preserve DataFrame/DesignMatrix (don't convert to numpy)
        # For Ridge: convert to numpy array for sklearn compatibility
        if model == "glm":
            X_model = X  # Keep as-is (DataFrame or DesignMatrix)
            # Validate shape using underlying array
            X_array = np.asarray(X)
            if X_array.shape[0] != self.shape[0]:
                raise ValueError(
                    f"X has {X_array.shape[0]} samples, but brain data has {self.shape[0]} samples. "
                    f"number of samples must match."
                )
        else:
            # Ridge: handle list (banded ridge) or array (regular ridge)
            if isinstance(X, list):
                # Banded ridge: keep as list, validate each element
                X_model = X
                for i, Xi in enumerate(X):
                    Xi_array = np.asarray(Xi)
                    if Xi_array.shape[0] != self.shape[0]:
                        raise ValueError(
                            f"X[{i}] has {Xi_array.shape[0]} samples, but brain data has {self.shape[0]} samples. "
                            f"number of samples must match."
                        )
            else:
                # Regular ridge: convert to numpy
                X_model = np.asarray(X)
                if X_model.shape[0] != self.shape[0]:
                    raise ValueError(
                        f"X has {X_model.shape[0]} samples, but brain data has {self.shape[0]} samples. "
                        f"number of samples must match."
                    )

        # Always store model_ and X_ for predict() to work (even if inplace=False)
        self.X_ = X_model

        # Determine progress_bar setting (default to self.verbose if not specified)
        if progress_bar is None:
            progress_bar = self.verbose

        # Create temporary copy if inplace=False to avoid mutating result attributes
        if inplace:
            target = self
        else:
            # Create temporary copy for fitting (to avoid mutating self's result attributes)
            target = self.copy()
            # Set X_ on copy (will be set below)
            target.X_ = X_model
            # Clean up any existing result attributes from the copy
            for attr in [
                "ridge_weights",
                "ridge_fitted_values",
                "ridge_scores",
                "glm_betas",
                "glm_t",
                "glm_p",
                "glm_se",
                "glm_residual",
                "glm_predicted",
                "glm_r2",
                "cv_results_",
            ]:
                if hasattr(target, attr):
                    delattr(target, attr)

        # Apply scaling before fitting (puts data in percent signal change units)
        if scale:
            scaled = target.scale(scale_value)
            target.data = scaled.data

        # Create model based on string
        if model == "ridge":
            # Pass progress_bar to Ridge model
            ridge_kwargs = kwargs.copy()
            if "progress_bar" not in ridge_kwargs:
                ridge_kwargs["progress_bar"] = progress_bar
            target.model_ = Ridge(**ridge_kwargs)
            target._fit_ridge(X_model, cv=cv, **kwargs)
        elif model == "glm":
            if cv is not None:
                raise NotImplementedError(
                    "Cross-validation not yet supported for GLM models"
                )
            # Pass mask from BrainData to Glm to prevent resampling during GLM estimation
            # The mask must match the one used to mask the data initially
            glm_kwargs = kwargs.copy()
            if "mask" not in glm_kwargs:
                glm_kwargs["mask"] = target.mask
            # Pass progress_bar to GLM (not verbose - we use tqdm progress bar)
            if "progress_bar" not in glm_kwargs:
                glm_kwargs["progress_bar"] = progress_bar
            target.model_ = Glm(**glm_kwargs)
            target._fit_glm(X_model)
        else:
            raise ValueError(f"Unknown model '{model}'. Must be one of: 'ridge', 'glm'")

        # If inplace=False, copy model_ to self (needed for predict()) and return Fit
        if not inplace:
            # Store model_ and X_ on self for predict() to work
            self.model_ = target.model_
            # Also store design_matrix for GLM compute_contrasts()
            if model == "glm" and hasattr(target, "design_matrix"):
                self.design_matrix = target.design_matrix
            # Return Fit dataclass with results
            return target._to_fit_dataclass(model=model)
        else:
            return self

    def _fit_ridge(self, X, cv=None, **kwargs):
        """Fit Ridge model and extract results.

        Args:
            X (ndarray): Training features
            cv (int, 'auto', or sklearn CV splitter, optional): Cross-validation specification
            **kwargs (dict): Additional arguments for CV (alpha, alphas, backend, etc.)
        """
        # Perform cross-validation if requested
        if cv is not None:
            self.cv_results_ = self._compute_ridge_cv(X, cv, **kwargs)

        # Always fit full model on all training data
        self.model_.fit(X, self.data)

        # Extract weights as BrainData
        # Ridge.coef_ is already (n_features, n_voxels) - no transpose needed
        self.ridge_weights = self._shallow_copy_with_data()
        self.ridge_weights.data = self.model_.coef_

        # Compute fitted values
        fitted = self.model_.predict(X)
        self.ridge_fitted_values = self._shallow_copy_with_data()
        self.ridge_fitted_values.data = fitted

        # Compute R² scores per voxel (using model's score method which now returns per-voxel)
        scores = self.model_.score(X, self.data)  # (n_voxels,)
        self.ridge_scores = self._shallow_copy_with_data()
        self.ridge_scores.data = scores.reshape(1, -1)  # (1, n_voxels)

    def _compute_ridge_cv(
        self, X, cv, alpha=None, alphas=None, backend="auto", **kwargs
    ):
        """Compute cross-validation results for Ridge regression.

        Args:
            X (ndarray or list): Training features. If ndarray, shape (n_samples, n_features).
                If list, list of feature spaces for banded ridge.
            cv (int, 'auto', or sklearn CV splitter): Cross-validation specification
            alpha (float or 'auto', optional): Regularization strength (extracted from model if not provided)
            alphas (array-like, optional): Alpha values to try for alpha selection
            backend (str): Computational backend ('numpy', 'torch', 'auto'). Default: 'auto'
            **kwargs (dict): Additional arguments (currently unused, for future extensibility)

        Returns:
            dict: Dictionary containing:
                - 'scores': (n_folds, n_voxels) array of R² per fold
                - 'mean_score': (n_voxels,) array of mean R² across folds
                - 'predictions': BrainData of out-of-fold predictions
                - 'folds': (n_samples,) array of fold indices
                - 'best_alpha': Selected alpha (if alpha selection performed)
                - 'alpha_scores': (n_folds, n_alphas, n_voxels) array (if alpha selection)
        """
        from sklearn.model_selection import check_cv
        from sklearn.metrics import r2_score
        from nltools.algorithms.ridge import ridge_cv, ridge_svd

        # Convert backend to parallel parameter for ridge functions
        # ridge_svd/ridge_cv expect: None, 'cpu', or 'gpu'
        if backend in ("auto", "numpy", None):
            parallel = "cpu"
        elif backend == "torch":
            parallel = "gpu"
        elif backend in ("cpu", "gpu"):
            parallel = backend
        else:
            parallel = "cpu"  # Safe default

        # Get alpha from model if not explicitly provided
        if alpha is None:
            alpha = self.model_.alpha if hasattr(self.model_, "alpha") else 1.0

        # Normalize cv parameter
        if cv == "auto":
            # cv='auto' implies alpha selection with default 5 folds
            cv_splitter = check_cv(5)
            perform_alpha_selection = True
        elif isinstance(cv, int):
            cv_splitter = check_cv(cv)
            perform_alpha_selection = alpha == "auto"
        else:
            # Assume sklearn CV object
            cv_splitter = cv
            perform_alpha_selection = alpha == "auto"

        # Handle list (banded ridge) vs array (regular ridge)
        if isinstance(X, list):
            # Banded ridge CV is handled by the model itself, not here
            # This shouldn't be called for banded ridge
            raise ValueError(
                "Cross-validation for banded ridge should be handled by the model. "
                "Use alpha='auto' with cv parameter in fit()."
            )

        n_samples, n_features = X.shape
        n_voxels = self.data.shape[1]

        # Initialize results
        cv_predictions = np.zeros_like(self.data)
        fold_indices = np.zeros(n_samples, dtype=int)

        # Perform alpha selection if requested
        if perform_alpha_selection:
            # Use efficient ridge_cv for alpha selection
            if alphas is None:
                alphas = np.logspace(-2, 4, 20)  # Default alpha grid

            result = ridge_cv(
                X, self.data, alphas=alphas, cv=cv_splitter.n_splits, parallel=parallel
            )

            best_alpha = result["alpha"]
            alpha_scores = result["cv_scores"]  # (n_folds, n_alphas, n_targets)

            # Update model's alpha to best_alpha
            self.model_.alpha = best_alpha
            alpha = best_alpha

            cv_results = {
                "best_alpha": best_alpha,
                "alpha_scores": alpha_scores,
            }
        else:
            cv_results = {}

        # Perform CV with fixed alpha to get predictions and per-fold scores
        fold_scores = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = self.data[train_idx], self.data[test_idx]

            # Fit Ridge on training fold
            coef = ridge_svd(X_train, y_train, alpha=alpha, parallel=parallel)

            # Predict on test fold
            y_pred = X_test @ coef

            # Store out-of-fold predictions
            cv_predictions[test_idx] = y_pred

            # Store fold indices
            fold_indices[test_idx] = fold_idx

            # Compute R² for this fold (per-voxel)
            fold_r2 = np.array(
                [r2_score(y_test[:, j], y_pred[:, j]) for j in range(n_voxels)]
            )
            fold_scores.append(fold_r2)

        # Aggregate results
        fold_scores = np.array(fold_scores)  # (n_folds, n_voxels)
        mean_scores = fold_scores.mean(axis=0)  # (n_voxels,)

        # Store predictions as BrainData
        cv_predictions_brain = self._shallow_copy_with_data()
        cv_predictions_brain.data = cv_predictions

        # Package results
        cv_results.update(
            {
                "scores": fold_scores,
                "mean_score": mean_scores,
                "predictions": cv_predictions_brain,
                "folds": fold_indices,
            }
        )

        return cv_results

    def _fit_glm(self, X):
        """Fit GLM model and extract results (same logic as current regress())."""
        from .design_matrix import DesignMatrix
        from nltools.data import BrainData

        # Ensure X is DesignMatrix
        if not isinstance(X, DesignMatrix):
            X = DesignMatrix(X)

        # Store design matrix for compute_contrasts()
        self.design_matrix = X

        # Convert data to 4D nifti for nilearn
        data_4d = self.to_nifti()

        # Fit Glm model
        self.model_.fit(data_4d, design_matrices=[X])

        # Extract results for each regressor (same as regress() lines 802-815)
        n_regressors = X.shape[1]
        beta_maps = []
        t_maps = []
        p_maps = []
        se_maps = []

        for i, col in enumerate(X.columns):
            # Create contrast vector (1 for this regressor, 0 for others)
            contrast = np.zeros(n_regressors)
            contrast[i] = 1

            # Compute contrast using Glm's compute_contrast method
            results = self.model_.compute_contrast(contrast, output_type="all")

            # Store maps
            beta_maps.append(results["effect_size"])
            t_maps.append(results["stat"])
            p_maps.append(results["p_value"])
            se_maps.append(results["effect_variance"])

        # Convert results to BrainData objects
        self.glm_betas = BrainData(beta_maps, mask=self.mask)
        self.glm_t = BrainData(t_maps, mask=self.mask)
        self.glm_p = BrainData(p_maps, mask=self.mask)

        # Convert effect variance to standard error (same as regress() lines 826-831)
        se_data = []
        for se_img in se_maps:
            se_brain = BrainData(se_img, mask=self.mask)
            se_brain.data = np.sqrt(np.abs(se_brain.data))
            se_data.append(se_brain)
        self.glm_se = BrainData(data=se_data, mask=self.mask)

        # Get residuals
        self.glm_residual = BrainData(self.model_.residuals, mask=self.mask)

        # Predicted = original - residuals
        self.glm_predicted = self.copy()
        self.glm_predicted.data = self.data - self.glm_residual.data

        # R-squared calculation
        ss_total = np.sum((self.data - self.data.mean(axis=0)) ** 2, axis=0)
        ss_residual = np.sum(self.glm_residual.data**2, axis=0)
        r2_values = 1 - (ss_residual / (ss_total + 1e-10))

        # Create single-image BrainData for R-squared
        self.glm_r2 = self[0].copy()
        self.glm_r2.data = r2_values.reshape(1, -1)

    def _to_fit_dataclass(self, model):
        """Convert BrainData fit results to Fit dataclass.

        Args:
            model (str): Model type ('ridge' or 'glm')

        Returns:
            Fit: Dataclass containing fit results
        """
        from nltools.data.fit_results import Fit

        if model == "ridge":
            # Extract Ridge results
            fitted_values = self.ridge_fitted_values.data  # (n_samples, n_voxels)
            weights = self.ridge_weights.data  # (n_features, n_voxels)
            scores = self.ridge_scores.data  # (1, n_voxels)
            # Squeeze first dimension to get (n_voxels,)
            if scores.ndim > 1 and scores.shape[0] == 1:
                scores = scores[0]  # (n_voxels,)
            else:
                scores = scores.squeeze()  # (n_voxels,)

            # Extract CV results if available
            cv_scores = None
            cv_mean_score = None
            cv_predictions = None
            cv_folds = None
            cv_best_alpha = None
            cv_alpha_scores = None

            if hasattr(self, "cv_results_") and self.cv_results_ is not None:
                cv_results = self.cv_results_
                cv_scores = cv_results.get("scores")  # (n_folds, n_voxels)
                cv_mean_score = cv_results.get("mean_score")  # (n_voxels,)

                # Extract predictions from BrainData
                if "predictions" in cv_results:
                    cv_predictions = cv_results[
                        "predictions"
                    ].data  # (n_samples, n_voxels)

                cv_folds = cv_results.get("folds")  # (n_samples,)
                cv_best_alpha = cv_results.get("best_alpha")  # float or None
                cv_alpha_scores = cv_results.get(
                    "alpha_scores"
                )  # (n_folds, n_alphas, n_voxels) or None

            return Fit(
                fitted_values=fitted_values,
                weights=weights,
                scores=scores,
                cv_scores=cv_scores,
                cv_mean_score=cv_mean_score,
                cv_predictions=cv_predictions,
                cv_folds=cv_folds,
                cv_best_alpha=cv_best_alpha,
                cv_alpha_scores=cv_alpha_scores,
            )

        elif model == "glm":
            # Extract GLM results
            fitted_values = self.glm_predicted.data  # (n_samples, n_voxels)
            betas = self.glm_betas.data  # (n_regressors, n_voxels)
            t_stats = self.glm_t.data  # (n_regressors, n_voxels)
            p_values = self.glm_p.data  # (n_regressors, n_voxels)
            se = self.glm_se.data  # (n_regressors, n_voxels)
            residuals = self.glm_residual.data  # (n_samples, n_voxels)
            r2 = self.glm_r2.data  # (1, n_voxels)
            # Squeeze first dimension to get (n_voxels,)
            if r2.ndim > 1 and r2.shape[0] == 1:
                r2 = r2[0]  # (n_voxels,)
            else:
                r2 = r2.squeeze()  # (n_voxels,)

            return Fit(
                fitted_values=fitted_values,
                betas=betas,
                t_stats=t_stats,
                p_values=p_values,
                se=se,
                residuals=residuals,
                r2=r2,
            )

        else:
            raise ValueError(f"Unknown model '{model}'. Must be 'ridge' or 'glm'")

    def regress(self, design_matrix=None, noise_model="ols", mode=None, **kwargs):
        """Deprecated: Use fit(model='glm', X=design_matrix) instead."""
        raise NotImplementedError(
            "The regress() method has been removed in v0.6.0. "
            "Use fit(model='glm', X=design_matrix) instead. "
            "See migration guide for examples."
        )

    def compute_contrasts(self, contrasts, contrast_type="t"):
        """Compute contrasts from fitted GLM results.

        This method computes contrasts as linear combinations of the GLM beta coefficients.
        Must be called after .fit(model='glm', X=design_matrix) has been run.

        Args:
            contrasts: Can be:

                - str: A string specifying the contrast using column names
                  e.g., "conditionA - conditionB" or "2*conditionA - conditionB - conditionC"
                - dict: Dictionary with contrast names as keys and contrast strings/vectors as values
                  e.g., {"main_effect": "conditionA - conditionB", "interaction": [1, -1, -1, 1]}
                - array: Numeric contrast vector matching the number of regressors
                  e.g., [1, -1, 0, 0] for a 4-regressor model
            contrast_type (str): Type of contrast statistic ('t' or 'F'). Default: 't'
                Note: Currently only 't' contrasts are supported.

        Returns:
            BrainData or dict: If single contrast, returns BrainData object with contrast map.
                               If multiple contrasts (dict input), returns dict of BrainData objects.

        Raises:
            RuntimeError: If .fit(model='glm') hasn't been called yet
            ValueError: If contrast vector length doesn't match number of regressors
            ValueError: If column name in string contrast not found in design matrix

        Examples:
            >>> # Fit GLM model
            >>> design_matrix = pd.DataFrame({
            ...     'intercept': np.ones(n_samples),
            ...     'conditionA': signal_a,
            ...     'conditionB': signal_b
            ... })
            >>> brain.fit(model='glm', X=design_matrix)
            >>>
            >>> # Simple numeric contrast: A - B
            >>> contrast1 = brain.compute_contrasts([0, 1, -1])
            >>>
            >>> # String-based contrast (more readable)
            >>> contrast2 = brain.compute_contrasts("conditionA - conditionB")
            >>>
            >>> # Multiple contrasts at once
            >>> contrasts = {
            ...     "A_vs_B": "conditionA - conditionB",
            ...     "avg_effect": [0, 0.5, 0.5],
            ...     "weighted": "2*conditionA - conditionB"
            ... }
            >>> results = brain.compute_contrasts(contrasts)
            >>> # results is a dict: {"A_vs_B": BrainData, "avg_effect": BrainData, ...}

        Notes:
            - String contrasts support coefficients: "2*A - B" or "0.5*A + 0.5*B"
            - Column names must match design matrix columns exactly (case-sensitive)
            - Contrast weights should sum to zero for proper inference in most cases
        """
        # Check that regression has been run
        if not hasattr(self, "glm_betas"):
            raise RuntimeError(
                "Must run .fit(model='glm', X=design_matrix) before computing contrasts"
            )

        # Parse contrasts
        if isinstance(contrasts, str):
            # Single string contrast
            contrast_dict = {"contrast": contrasts}
            single_contrast = True
        elif isinstance(contrasts, (list, np.ndarray)):
            # Single numeric contrast
            contrast_dict = {"contrast": contrasts}
            single_contrast = True
        elif isinstance(contrasts, dict):
            # Multiple contrasts
            contrast_dict = contrasts
            single_contrast = False
        else:
            raise TypeError("contrasts must be str, array, or dict")

        # Process each contrast
        results = {}
        for name, contrast_def in contrast_dict.items():
            # Parse string contrasts to create contrast vector
            if isinstance(contrast_def, str):
                # Parse string like "conditionA - conditionB"
                contrast_vector = self._parse_contrast_string(contrast_def)
            else:
                # Use numeric contrast directly
                contrast_vector = np.array(contrast_def)

            # Validate contrast vector length
            if len(contrast_vector) != self.glm_betas.shape[0]:
                raise ValueError(
                    f"Contrast vector length ({len(contrast_vector)}) must match "
                    f"number of regressors ({self.glm_betas.shape[0]})"
                )

            # Compute contrast by linear combination of betas
            contrast_data = np.zeros((1, self.glm_betas.shape[1]))
            for i, weight in enumerate(contrast_vector):
                if weight != 0:
                    contrast_data += weight * self.glm_betas[i].data

            # Create BrainData object for contrast
            contrast_brain = self[0].copy()
            contrast_brain.data = contrast_data

            # Store result
            results[name] = contrast_brain

        # Return single contrast or dict of contrasts
        if single_contrast:
            return results["contrast"]
        else:
            return results

    def _parse_contrast_string(self, contrast_str):
        """Parse a contrast string into a numeric contrast vector.

        Args:
            contrast_str (str): Contrast string like "A - B" or "2*A - B - C"

        Returns:
            np.array: Numeric contrast vector

        Raises:
            RuntimeError: If design_matrix not found (fit() not called)
            ValueError: If column name not found in design_matrix
        """
        if not hasattr(self, "design_matrix"):
            raise RuntimeError(
                "No design matrix found. Run .fit(model='glm', X=design_matrix) first."
            )

        # Get column names from design matrix
        if isinstance(self.design_matrix, pd.DataFrame):
            col_names = list(self.design_matrix.columns)
        else:
            col_names = list(self.design_matrix.columns)

        # Initialize contrast vector
        contrast_vector = np.zeros(len(col_names))

        # Parse the string
        # Split by + and - (keeping the operators)
        import re

        tokens = re.split(r"(\+|\-)", contrast_str)
        tokens = [t.strip() for t in tokens if t.strip()]

        # Process tokens
        sign = 1  # Start with positive
        for token in tokens:
            if token == "+":
                sign = 1
            elif token == "-":
                sign = -1
            else:
                # Parse coefficient and variable
                if "*" in token:
                    coef_str, var_name = token.split("*")
                    coef = float(coef_str.strip())
                    var_name = var_name.strip()
                else:
                    coef = 1
                    var_name = token

                # Find column index
                if var_name in col_names:
                    idx = col_names.index(var_name)
                    contrast_vector[idx] = sign * coef
                else:
                    raise ValueError(f"Column '{var_name}' not found in design matrix")

        return contrast_vector

    def append(self, data, ignore_attrs=False, **kwargs):
        """Append data to BrainData instance.

        Args:
            data: BrainData instance to append.
            ignore_attrs: (bool) If True, skip concatenation of X and Y
                    attributes. Useful when appending images where .X or .Y
                    have different column counts. Default False.
            kwargs: Optional arguments passed to pandas concat for X/Y.

        Returns:
            BrainData: New appended BrainData instance.
        """
        from ._validation import validate_append_shapes

        data = check_brain_data(data)

        if self.isempty:
            # If self is empty, return copy of the data to append
            out = data._shallow_copy_with_data()
            out.data = data.data.copy()
        else:
            # Validate shapes are compatible
            validate_append_shapes(self.shape, data.shape)

            out = self._shallow_copy_with_data()
            out.data = np.vstack([self.data, data.data])

            # Handle X and Y attributes
            if not ignore_attrs:
                # Concatenate X if both have it
                if (
                    hasattr(self, "X")
                    and self.X is not None
                    and hasattr(data, "X")
                    and data.X is not None
                ):
                    out.X = pd.concat([self.X, data.X], ignore_index=True, **kwargs)
                elif hasattr(data, "X") and data.X is not None:
                    # self.X is None but data.X exists - just use data.X
                    out.X = data.X.copy()
                # else: keep self.X (already copied in _shallow_copy_with_data)

                # Concatenate Y if both have it
                if (
                    hasattr(self, "Y")
                    and self.Y is not None
                    and hasattr(data, "Y")
                    and data.Y is not None
                ):
                    out.Y = pd.concat([self.Y, data.Y], ignore_index=True, **kwargs)
                elif hasattr(data, "Y") and data.Y is not None:
                    # self.Y is None but data.Y exists - just use data.Y
                    out.Y = data.Y.copy()
                # else: keep self.Y (already copied in _shallow_copy_with_data)
            else:
                # ignore_attrs=True: set X and Y to None to avoid confusion
                out.X = None
                out.Y = None

        return out

    def empty(self):
        """Create a copy of BrainData with empty data array"""
        from copy import deepcopy

        out = deepcopy(self)
        out.data = np.array([])
        return out

    @property
    def isempty(self):
        """Check if BrainData.data is empty"""

        if isinstance(self.data, np.ndarray):
            boolean = False if self.data.size else True
        if isinstance(self.data, list):
            boolean = True if not self.data else False
        return boolean

    # NOTE: uses nilearn.masking.intersect_masks
    def _check_masks(self, image):
        """Check to make sure masks are the same for each dataset and if not create a union mask

        Args:
            image (_type_): _description_

        Returns:
            _type_: _description_
        """

        if np.sum(self.nifti_masker.mask_img.get_fdata() == 1) != np.sum(
            image.nifti_masker.mask_img.get_fdata() == 1
        ):
            new_mask = intersect_masks(
                [self.nifti_masker.mask_img, image.nifti_masker.mask_img],
                threshold=1,
                connected=False,
            )
            new_nifti_masker = NiftiMasker(mask_img=new_mask)
            data2 = new_nifti_masker.fit_transform(self.to_nifti())
            image2 = new_nifti_masker.fit_transform(image.to_nifti())
        else:
            data2 = self.data
            image2 = image.data
        return data2, image2

    def similarity(self, image, method="correlation"):
        """Calculate similarity of BrainData() instance with single
        BrainData or Nibabel image

        Args:
            image: (BrainData, nifti)  image to evaluate similarity
            method: (str) Type of similarity
                    ['correlation','dot_product','cosine']
        Returns:
            pexp: (list) Outputs a vector of pattern expression values

        """
        supported_metrics = [
            "correlation",
            "pearson",
            "rank_correlation",
            "spearman",
            "dot_product",
            "cosine",
        ]
        if method not in supported_metrics:
            raise ValueError(f"method must be one of {supported_metrics}")

        image = check_brain_data(image)
        data2, image2 = self._check_masks(image)

        # Delegate to functional core (stats.py)
        return compute_similarity(data2, image2, method=method)

    def distance(self, metric="euclidean", **kwargs):
        """Calculate distance between images within a BrainData() instance.

        Args:
            metric: (str) type of distance metric (can use any scipy.spatial.distance
                    metric supported by cdist, e.g., 'euclidean', 'cityblock', 'cosine',
                    'correlation', 'hamming', 'jaccard', etc.)

        Returns:
            dist: (Adjacency) Outputs a 2D distance matrix.

        """
        # Use scipy.spatial.distance.cdist directly for efficiency
        # Computes pairwise distances between all images (rows)
        dist_matrix = cdist(self.data, self.data, metric=metric, **kwargs)
        return Adjacency(dist_matrix, matrix_type="Distance")

    def multivariate_similarity(self, images, method="ols"):
        """Predict spatial distribution of BrainData() instance from linear
        combination of other BrainData() instances or Nibabel images

        Args:
            self: BrainData instance of data to be applied
            images: BrainData instance of weight map

        Returns:
            out: dictionary of regression statistics in BrainData
                instances {'beta','t','p','df','residual'}

        """
        # Notes:  Should add ridge, and lasso, elastic net options options

        if len(self.shape) > 1:
            raise ValueError("This method can only decompose a single brain image.")

        images = check_brain_data(images)
        data2, image2 = self._check_masks(images)

        # Prepare data for functional core: y is single image, X is predictors
        # image2 shape: (n_images, n_voxels) -> transpose to (n_voxels, n_images)
        y = data2.squeeze()  # Single image: (n_voxels,)
        X = image2.T  # Predictors: (n_voxels, n_images)

        # Delegate to functional core (stats.py)
        return compute_multivariate_similarity(y, X, method=method)

    # NOTE: uses nilearn.masking.apply_mask
    def apply_mask(self, mask, resample_mask_to_brain=False):
        """Mask BrainData instance using nilearn functionality.

        Note target data will be resampled into the same space as the mask. If you would like the mask
        resampled into the BrainData space, then set resample_mask_to_brain=True.

        Args:
            mask: (BrainData or nifti object) mask to apply to BrainData object.
            resample_mask_to_brain: (bool) Will resample mask to brain space before applying mask (default=False).

        Returns:
            masked: (BrainData) masked BrainData object

        Note:
            Uses nilearn.masking.apply_mask for efficient, validated masking.
            Simplified from 47-line manual implementation to leverage nilearn's
            Cython-optimized code with better validation and memory management.

        """
        from nilearn.masking import apply_mask

        masked = self._shallow_copy_with_data()
        mask = check_brain_data(mask)
        if not check_brain_data_is_single(mask):
            raise ValueError("Mask must be a single image")

        # Handle resampling if requested (preserve existing feature)
        mask_img = mask.to_nifti()
        if resample_mask_to_brain:
            mask_img = resample_to_img(
                mask_img,
                masked.to_nifti(),
                interpolation="nearest",  # Masks are discrete, use nearest
                force_resample=True,
                copy_header=True,
            )

        # Use nilearn's apply_mask for efficient masking (C-optimized, single path, memory efficient)
        masked.data = apply_mask(masked.to_nifti(), mask_img)

        # Create masker for the masked space
        masked.nifti_masker = NiftiMasker(mask_img=mask_img).fit()

        # Update mask, voxel resolution, and space
        masked.mask = mask_img
        affine = mask_img.affine
        masked._voxel_resolution = np.abs(np.diag(affine[:3, :3]))
        masked._space = masked._detect_space(mask_img)

        # Preserve 1D output for single images (backward compatibility)
        if (len(masked.shape) > 1) & (masked.shape[0] == 1):
            masked.data = masked.data.flatten()

        return masked

    # NOTE: uses nilearn nilearn.maskers.NiftiLabelsMasker
    def extract_roi(self, mask, metric="mean", n_components=None):
        """Extract activity from mask or ROI atlas using NiftiLabelsMasker.

        This method now uses nilearn's NiftiLabelsMasker for efficient ROI extraction
        when dealing with labeled atlases (multiple ROIs).

        Args:
            mask: BrainData, nibabel image, or file path. Can be:

                  - Binary mask (extracts from single ROI)
                  - Labeled atlas (extracts from multiple ROIs)
            metric: Extraction method ('mean', 'median', 'pca'). Default: 'mean'
                    Note: 'median' and 'pca' require additional computation after extraction
            n_components: If metric='pca', number of components to return

        Returns:
            For binary mask:

                - Single image: scalar value
                - Multiple images: 1D array of values

            For labeled atlas:

                - Single image: 1D array (one value per ROI)
                - Multiple images: 2D array (images × ROIs)
                - If metric='pca': returns components array

        Examples:
            >>> # Extract mean from binary mask
            >>> roi_values = brain.extract_roi(binary_mask)
            >>> # Extract from atlas
            >>> atlas_values = brain.extract_roi(atlas_mask)
            >>> # PCA extraction
            >>> components = brain.extract_roi(mask, metric='pca', n_components=5)
        """
        from nilearn.maskers import NiftiLabelsMasker

        metrics = ["mean", "median", "pca"]
        if metric not in metrics:
            raise NotImplementedError(f"metric must be one of {metrics}, got {metric}")

        # Convert mask to BrainData if needed
        mask_brain = check_brain_data(mask)
        mask_img = mask_brain.to_nifti()

        # Check if binary or labeled mask
        unique_values = np.unique(mask_brain.data)
        n_unique = len(unique_values)

        if n_unique == 2:
            # Binary mask - use simple extraction
            masked = self.apply_mask(mask_brain)
            is_single = check_brain_data_is_single(masked)

            if metric == "mean":
                out = masked.mean() if is_single else masked.mean(axis=1)
            elif metric == "median":
                out = masked.median() if is_single else masked.median(axis=1)
            elif metric == "pca":
                if is_single:
                    raise ValueError("Cannot run PCA on a single image")
                # Check if masked has any data
                if masked.data.size == 0 or masked.data.shape[1] == 0:
                    raise ValueError(
                        "No voxels remain after masking - mask may not overlap with data"
                    )
                output = masked.decompose(
                    algorithm="pca", n_components=n_components, axis="images"
                )
                out = output["weights"].T

        elif n_unique > 2:
            # Labeled atlas - use NiftiLabelsMasker for efficiency
            # Round values to ensure integer labels (use int32 for nilearn/FSL/SPM compatibility)
            mask_brain.data = np.round(mask_brain.data).astype(np.int32)
            mask_img = mask_brain.to_nifti()

            # Create masker based on metric
            if metric in ["mean", "median"]:
                # For mean/median, use NiftiLabelsMasker
                strategy = "mean" if metric == "mean" else "median"
                labels_masker = NiftiLabelsMasker(
                    labels_img=mask_img,
                    strategy=strategy,
                    mask_img=self.mask,
                    standardize=False,
                    resampling_target="data" if hasattr(self, "mask") else None,
                )

                # Transform data
                data_4d = self.to_nifti()
                out = labels_masker.fit_transform(data_4d)

                # If single image, return 1D array
                if out.shape[0] == 1:
                    out = out[0]
                else:
                    # For multiple images, transpose to (n_labels, n_images)
                    out = out.T

            elif metric == "pca":
                # For PCA, we need to extract raw data and then apply PCA
                if check_brain_data_is_single(self):
                    raise ValueError("Cannot run PCA on a single image")

                # Use mean strategy to get the regions, then extract raw data
                labels_masker = NiftiLabelsMasker(
                    labels_img=mask_img,
                    strategy="mean",  # Just to get regions
                    mask_img=self.mask,
                    standardize=False,
                    resampling_target="data" if hasattr(self, "mask") else None,
                )

                # Fit to get regions
                labels_masker.fit(self.to_nifti())

                # Extract data for each region and apply PCA
                out = []
                unique_labels = np.unique(mask_brain.data[mask_brain.data != 0])

                for label in unique_labels:
                    # Create binary mask for this label
                    label_mask = mask_brain.copy()
                    label_mask.data = (mask_brain.data == label).astype(float)

                    # Extract data for this ROI
                    masked = self.apply_mask(label_mask)

                    # Apply PCA
                    output = masked.decompose(
                        algorithm="pca", n_components=n_components, axis="images"
                    )
                    out.append(output["weights"].T)

                # Stack results
                if len(out) > 0:
                    out = np.array(out) if n_components == 1 else out

        else:
            raise ValueError(
                "Mask must be binary (2 unique values) or labeled atlas (>2 unique values)"
            )

        return out

    def icc(
        self,
        n_subjects,
        n_sessions,
        icc_type="icc2",
        parallel=None,
        n_jobs=-1,
        max_gpu_memory_gb=4.0,
    ):
        """Calculate voxel-wise intraclass correlation coefficient for data within
            BrainData class.

        Computes ICC for each voxel independently, making it highly parallelizable.
        Supports GPU acceleration for large voxel counts.

        ICC Formulas are based on:
        Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
        assessing rater reliability. Psychological bulletin, 86(2), 420.

        icc1:  x_ij = mu + beta_j + w_ij
        icc2/3:  x_ij = mu + alpha_i + beta_j + (ab)_ij + epsilon_ij

        Args:
            n_subjects: Number of subjects in the data
            n_sessions: Number of sessions per subject
            icc_type: Type of ICC to calculate
                    - 'icc1': One-way random effects (subjects random, sessions treated as interchangeable)
                    - 'icc2': Two-way random effects (subjects and sessions random) (default)
                    - 'icc3': Two-way mixed effects (subjects random, sessions fixed)
            parallel: Parallelization method
                    - None: Single-threaded vectorized NumPy (default, memory efficient)
                    - 'cpu': CPU parallelization via joblib (for medium-sized problems, 1K-10K voxels)
                    - 'gpu': GPU acceleration via PyTorch (recommended for large voxel counts >10K, 10-50× speedup)
            n_jobs: Number of CPU cores (-1 = all cores). Only used when parallel='cpu'
            max_gpu_memory_gb: GPU memory budget in GB. Only used when parallel='gpu'

        Returns:
            BrainData: BrainData instance with ICC map (shape: (1, n_voxels))

        Examples:
            >>> # Typical test-retest reliability analysis
            >>> data = BrainData(...)  # Shape: (60, 238955) = 20 subjects × 3 sessions
            >>> icc_map = data.icc(n_subjects=20, n_sessions=3, icc_type='icc2')
            >>> icc_map.shape
            (1, 238955)
            >>> # Visualize ICC map
            >>> icc_map.plot()

        Notes:
            Data must be organized such that n_images = n_subjects * n_sessions.
            Images should be ordered as: [subject1_session1, subject1_session2, ...,
            subject2_session1, ...]
        """
        from nltools.algorithms.inference.icc import compute_icc_voxelwise

        # Validate data shape
        n_images = self.shape[0]

        if n_images != n_subjects * n_sessions:
            raise ValueError(
                f"Number of images ({n_images}) must equal n_subjects * n_sessions "
                f"({n_subjects} * {n_sessions} = {n_subjects * n_sessions}). "
                f"Make sure images are organized as: "
                f"[subject1_session1, subject1_session2, ..., subject2_session1, ...]"
            )

        # Compute voxel-wise ICC
        icc_map = compute_icc_voxelwise(
            self.data,
            n_subjects=n_subjects,
            n_sessions=n_sessions,
            icc_type=icc_type,
            parallel=parallel,
            n_jobs=n_jobs,
            max_gpu_memory_gb=max_gpu_memory_gb,
        )

        # Return as BrainData object (shape: (1, n_voxels))
        out = self._shallow_copy_with_data()
        out.data = icc_map[np.newaxis, :]  # (1, n_voxels)
        out.X = pd.DataFrame()
        out.Y = pd.DataFrame()
        return out

    # NOTE: uses scipy.signal.detrend
    def detrend(self, method="linear"):
        """Remove linear trend from each voxel

        Args:
            type: ('linear','constant', optional) type of detrending

        Returns:
            out: (BrainData) detrended BrainData instance

        """

        if len(self.shape) == 1:
            raise ValueError(
                "Make sure there is more than one image in order to detrend."
            )

        out = self._shallow_copy_with_data()
        # Copy data and detrend
        out.data = detrend(self.data.copy(), type=method, axis=0)
        return out

    # NOTE: uses pynv
    def upload_neurovault(
        self,
        access_token=None,
        collection_name=None,
        collection_id=None,
        img_type=None,
        img_modality=None,
        **kwargs,
    ):
        """Upload Data to Neurovault.  Will add any columns in self.X to image
            metadata. Index will be used as image name.

        Args:
            access_token: (str, Required) Neurovault api access token
            collection_name: (str, Optional) name of new collection to create
            collection_id: (int, Optional) neurovault collection_id if adding images
                            to existing collection
            img_type: (str, Required) Neurovault map_type
            img_modality: (str, Required) Neurovault image modality

        Returns:
            collection: (pd.DataFrame) neurovault collection information

        """

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

        def add_image_to_collection(
            api, collection, dat, tmp_dir, index_id=0, **kwargs
        ):
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

        if len(self.shape) == 1:
            add_image_to_collection(
                api, collection, self, tmp_dir, index_id=0, **kwargs
            )
        else:
            for i, x in enumerate(self):
                add_image_to_collection(
                    api, collection, x, tmp_dir, index_id=i, **kwargs
                )

        shutil.rmtree(tmp_dir, ignore_errors=True)
        return collection

    # NOTE: uses nltools.stats
    def r_to_z(self):
        """Apply Fisher's r to z transformation to each element of the data
        object."""

        out = self._shallow_copy_with_data()
        # fisher_r_to_z creates a new array
        out.data = fisher_r_to_z(self.data)
        return out

    # NOTE: uses ntlools.stats
    def z_to_r(self):
        """Convert z score back into r value for each element of data object"""

        out = self._shallow_copy_with_data()
        # fisher_z_to_r creates a new array
        out.data = fisher_z_to_r(self.data)
        return out

    # NOTE: uses nilearn.signal.clean
    def filter(self, sampling_freq=None, high_pass=None, low_pass=None, **kwargs):
        """Apply butterworth filter to data. Wraps nilearn.signal.clean.

        Does not default to detrending and standardizing like nilearn
        implementation, but this can be overridden using kwargs.

        Args:
            sampling_freq: Sampling freq in hertz (i.e. 1 / TR)
            high_pass: High pass cutoff frequency
            low_pass: Low pass cutoff frequency
            **kwargs: Additional arguments passed to nilearn.signal.clean
                      Common options:
                      - confounds: Confound timeseries to remove
                      - sample_mask: Volumes to exclude (scrubbing)
                      - detrend: Enable detrending (default False)
                      - standardize: Enable standardization (default False)
                      - ensure_finite: Replace NaN/inf (default False)

        Returns:
            BrainData: Filtered BrainData instance

        See Also:
            nilearn.signal.clean documentation for all available options
        """

        if sampling_freq is None:
            raise ValueError("Need to provide sampling rate (TR)!")
        if high_pass is None and low_pass is None:
            raise ValueError("high_pass and/or low_pass cutoff must beprovided!")

        standardize = kwargs.get("standardize", False)
        detrend = kwargs.get("detrend", False)

        # Optimized: Use shallow copy instead of deepcopy
        out = self._shallow_copy_with_data()
        out.data = clean(
            self.data,
            t_r=1.0 / sampling_freq,
            detrend=detrend,
            standardize=standardize,
            high_pass=high_pass,
            low_pass=low_pass,
            **kwargs,
        )
        return out

    def standardize(self, axis=0, method="center"):
        """Standardize BrainData() instance.

        Args:
            axis: 0 for observations 1 for voxels
            method: ['center','zscore']

        Returns:
            BrainData Instance

        """

        if axis == 1 and len(self.shape) == 1:
            raise IndexError(
                "BrainData is only 3d but standardization was requested over observations"
            )

        # Optimized: Use shallow copy instead of deepcopy
        out = self._shallow_copy_with_data()
        if method == "zscore":
            with_std = True
        elif method == "center":
            with_std = False
        else:
            raise ValueError('method must be ["center","zscore"')
        out.data = scale(self.data, axis=axis, with_std=with_std)
        return out

    def threshold(
        self,
        upper=None,
        lower=None,
        binarize=False,
        coerce_nan=True,
        cluster_threshold=0,
    ):
        """Threshold BrainData instance with optional cluster filtering.

        Provide upper and lower values or percentages to perform two-sided
        thresholding. Binarize will return a mask image respecting thresholds
        if provided, otherwise respecting every non-zero value.

        Args:
            upper: (float or str) Upper cutoff for thresholding. If string
                    will interpret as percentile; can be None for one-sided
                    thresholding.
            lower: (float or str) Lower cutoff for thresholding. If string
                    will interpret as percentile; can be None for one-sided
                    thresholding.
            binarize (bool): return binarized image respecting thresholds if
                    provided, otherwise binarize on every non-zero value;
                    default False
            coerce_nan (bool): coerce nan values to 0s; default True
            cluster_threshold (int): Minimum cluster size in voxels. If > 0, uses
                    nilearn.image.threshold_img with cluster filtering.
                    Band-pass filtering (both upper AND lower) not supported
                    with cluster thresholding. Default 0 (disabled).

        Returns:
            Thresholded BrainData object.

        Note:
            When cluster_threshold=0 (default), uses fast path for basic thresholding.
            When cluster_threshold>0, uses nilearn for cluster filtering.
            Band-pass filtering (unique nltools feature) preserved when cluster_threshold=0.

        """

        if cluster_threshold > 0:
            # Use nilearn for cluster thresholding
            from nilearn.image import threshold_img
            from nilearn.masking import apply_mask

            # Band-pass filtering not supported with cluster thresholding
            if upper is not None and lower is not None:
                raise ValueError(
                    "Band-pass filtering (both upper and lower) not supported "
                    "with cluster thresholding. Use one threshold only."
                )

            # Determine threshold value (from whichever is provided)
            threshold_val = upper if upper is not None else lower
            if threshold_val is None:
                raise ValueError("Must provide either upper or lower threshold")

            # Handle percentile strings
            b = self.copy()
            if coerce_nan:
                b.data = np.nan_to_num(b.data)

            if isinstance(threshold_val, str) and threshold_val[-1] == "%":
                threshold_val = np.percentile(b.data, float(threshold_val[:-1]))

            # Use nilearn's cluster thresholding
            out = self._shallow_copy_with_data()
            thresholded_img = threshold_img(
                b.to_nifti(),
                threshold=threshold_val,
                cluster_threshold=cluster_threshold,
                two_sided=(upper is not None),
                copy_header=True,
            )

            # Convert back to data array
            out.data = apply_mask(thresholded_img, self.nifti_masker.mask_img_)

            if binarize:
                out.data = (out.data != 0).astype(float)

            return out

        else:
            # Use current efficient implementation (fast path)
            b = self.copy()

            if coerce_nan:
                b.data = np.nan_to_num(b.data)

            if isinstance(upper, str) and upper[-1] == "%":
                upper = np.percentile(b.data, float(upper[:-1]))

            if isinstance(lower, str) and lower[-1] == "%":
                lower = np.percentile(b.data, float(lower[:-1]))

            if upper is not None and lower is not None:
                b.data[(b.data < upper) & (b.data > lower)] = 0
            elif upper is not None:
                b.data[b.data < upper] = 0
            elif lower is not None:
                b.data[b.data > lower] = 0

            if binarize:
                b.data[b.data != 0] = 1
            return b

    def regions(
        self,
        min_region_size=1350,
        extract_type="local_regions",
        smoothing_fwhm=6,
        is_mask=False,
    ):
        """Extract brain connected regions into separate regions.

        Args:
            min_region_size (int): Minimum volume in mm3 for a region to be
                                kept.
            extract_type (str): Type of extraction method
                                ['connected_components', 'local_regions'].
                                If 'connected_components', each component/region
                                in the image is extracted automatically by
                                labelling each region based upon the presence of
                                unique features in their respective regions.
                                If 'local_regions', each component/region is
                                extracted based on their maximum peak value to
                                define a seed marker and then using random
                                walker segementation algorithm on these
                                markers for region separation.
            smoothing_fwhm (scalar): Smooth an image to extract more sparser
                                regions. Only works for extract_type
                                'local_regions'.
            is_mask (bool): Whether the BrainData instance should be treated
                            as a boolean mask and if so, calls
                            connected_label_regions instead.

        Returns:
            BrainData: BrainData instance with extracted ROIs as data.
        """

        if is_mask:
            regions, _ = connected_label_regions(self.to_nifti())
        else:
            regions, _ = connected_regions(
                self.to_nifti(), min_region_size, extract_type, smoothing_fwhm
            )

        return BrainData(regions, mask=self.mask)

    # NOTE: nltools.stats
    def transform_pairwise(self):
        """Extract brain connected regions into separate regions.

        Args:

        Returns:
            BrainData: BrainData instance tranformed into pairwise comparisons
        """
        # Optimized: Use shallow copy instead of deepcopy
        out = self._shallow_copy_with_data()
        out.data, new_Y = transform_pairwise(self.data, self.Y)
        out.Y = pd.DataFrame(new_Y)
        out.Y.replace(-1, 0, inplace=True)
        return out

    def bootstrap(
        self,
        stat,
        n_samples=5000,
        save_boots=False,
        n_jobs=-1,
        random_state=None,
        percentiles=(2.5, 97.5),
        X_test=None,
        **kwargs,
    ):
        """Bootstrap statistics using efficient online algorithms.

        Uses memory-efficient bootstrap infrastructure with CPU parallelization or GPU acceleration.
        Supports simple aggregation statistics and fitted model statistics (Ridge).

        Args:
            stat: (str) Statistic to bootstrap. Options: Simple stats ('mean', 'median', 'std',
                'sum', 'min', 'max') or Model stats ('weights' requires fitted Ridge model,
                'predict' requires fitted Ridge model + X_test).
            n_samples: (int) Number of bootstrap iterations. Default: 5000
            save_boots: (bool) If True, store all bootstrap samples (memory intensive).
                       Default: False
            n_jobs: (int) Number of CPU cores for parallelization. -1 means all CPUs.
            random_state: (int, optional) Random seed for reproducibility
            percentiles: (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5)
            X_test: (np.ndarray, optional) Test features for 'predict' bootstrap.
                   Required if stat='predict'
            backend: (str, optional) Backend for computation ('numpy', 'torch', 'auto').
                    If 'torch' and GPU available, uses optimized GPU acceleration with
                    inline Ridge computation (no CPU round-trips). Default: None (CPU).
            max_gpu_memory_gb: (float) Maximum GPU memory to use in GB. Default: 4.0
            **kwargs: Additional parameters (currently unused, reserved for future extensions)

        Returns:
            BrainData or dict:
                - For simple stats: Returns BrainData with bootstrap mean
                - For model stats: Returns dict with keys: 'mean', 'std', 'Z', 'p',
                  'ci_lower', 'ci_upper' (all BrainData objects)
                - If ``save_boots=True``: Returns dict with 'samples' key containing all samples

        Examples:
            >>> # Simple aggregation
            >>> boot = brain.bootstrap(stat='mean', n_samples=1000)
            >>> assert isinstance(boot, BrainData)

            >>> # Ridge weights bootstrap (CPU)
            >>> brain.fit(X=dm, model='ridge', alpha=1.0)
            >>> boot = brain.bootstrap(stat='weights', n_samples=1000)
            >>> assert 'mean' in boot
            >>> assert isinstance(boot['mean'], BrainData)

            >>> # Ridge weights bootstrap (GPU accelerated)
            >>> brain.fit(X=dm, model='ridge', alpha=1.0)
            >>> boot = brain.bootstrap(stat='weights', n_samples=1000, backend='torch')
            >>> assert 'mean' in boot
            >>> assert isinstance(boot['mean'], BrainData)

            >>> # Ridge predict bootstrap
            >>> brain.fit(X=dm, model='ridge', alpha=1.0)
            >>> boot = brain.bootstrap(stat='predict', X_test=X_new, n_samples=1000)
            >>> assert 'mean' in boot
            >>> assert isinstance(boot['mean'], BrainData)

        Note:
            This method replaces the deprecated `summarize_bootstrap()` function from
            `nltools.stats`. To reproduce `summarize_bootstrap()` functionality:

            **Old API (deprecated):**
            >>> from nltools.stats import summarize_bootstrap
            >>> bootstrap_samples = BrainData(list_of_samples)  # Multiple samples
            >>> result = summarize_bootstrap(bootstrap_samples, save_weights=False)
            >>> # Returns: {'mean': BrainData, 'Z': BrainData, 'p': BrainData}

            **New API (recommended):**
            >>> # Option 1: Use BrainData.bootstrap() for generating bootstrap samples
            >>> boot = brain.bootstrap(stat='mean', n_samples=1000, save_boots=False)
            >>> # Returns BrainData with bootstrap mean
            >>> # To get Z and p, use stat='weights' or 'predict' which returns dict

            >>> # Option 2: For existing bootstrap samples (BrainData with multiple images),
            >>> # use OnlineBootstrapStats directly:
            >>> from nltools.algorithms.inference.bootstrap import OnlineBootstrapStats
            >>> stats = OnlineBootstrapStats(shape=(brain.shape[1],), save_samples=False)
            >>> for sample in bootstrap_samples:  # Iterate over samples
            ...     stats.update(sample.data)
            >>> result = stats.get_results()
            >>> # Returns: {'mean': array, 'std': array, 'Z': array, 'p': array,
            >>> #           'ci_lower': array, 'ci_upper': array}
            >>> # Convert to BrainData if needed:
            >>> mean_brain = brain._shallow_copy_with_data()
            >>> mean_brain.data = result['mean']
        """
        from nltools.algorithms.inference.bootstrap import (
            _bootstrap_simple_cpu_parallel,
            _bootstrap_ridge_weights_cpu_parallel,
            _bootstrap_ridge_predict_cpu_parallel,
            _bootstrap_ridge_weights_gpu_batched,
            _bootstrap_ridge_predict_gpu_batched,
        )
        from nltools.data import DesignMatrix
        from nltools.backends import Backend, check_gpu_available, auto_select_backend

        # Extract backend parameter from kwargs
        backend = kwargs.pop("backend", None)
        max_gpu_memory_gb = kwargs.pop("max_gpu_memory_gb", 4.0)

        # Determine if we should use GPU
        use_gpu = False
        if backend == "torch" or backend == "auto":
            if check_gpu_available()[0]:
                use_gpu = True
                if backend == "auto":
                    backend = auto_select_backend(
                        self.data.shape[0], self.data.shape[1]
                    )
                else:
                    backend = Backend("torch")
            elif backend == "torch":
                raise ValueError(
                    "GPU backend requested but GPU not available. "
                    "Use backend=None or backend='auto' for CPU fallback."
                )

        # Get data as numpy array
        data = self.data  # Shape: (n_samples, n_voxels)

        # Route to appropriate bootstrap function
        SIMPLE_STATS = ["mean", "median", "std", "sum", "min", "max"]
        FITTED_STATS = ["weights", "predict"]

        if stat in SIMPLE_STATS:
            # Simple aggregation bootstrap
            result = _bootstrap_simple_cpu_parallel(
                data,
                method=stat,
                n_samples=n_samples,
                save_boots=save_boots,
                n_jobs=n_jobs,
                random_state=random_state,
                percentiles=percentiles,
            )

            # Convert result to BrainData format
            return self._convert_bootstrap_results_to_brain_data(
                result, save_boots=save_boots, return_dict=False
            )

        elif stat in FITTED_STATS:
            # Check if model is fitted
            if not hasattr(self, "model_") or self.model_ is None:
                raise ValueError(
                    f"Must call .fit(model='ridge', X=design_matrix) before bootstrap(stat='{stat}')"
                )

            # Check if Ridge model
            if not hasattr(self.model_, "coef_") or not hasattr(self.model_, "alpha_"):
                raise ValueError(
                    f"Bootstrap stat='{stat}' only supports Ridge models. "
                    f"Got model type: {type(self.model_)}"
                )

            # Get design matrix from stored X_
            if not hasattr(self, "X_") or self.X_ is None:
                raise ValueError(
                    "Design matrix not found. Must call .fit(model='ridge', X=design_matrix) "
                    "with X parameter."
                )

            # Convert DesignMatrix to numpy if needed
            if isinstance(self.X_, DesignMatrix):
                X = self.X_.to_numpy()
            else:
                X = np.asarray(self.X_)

            # Get alpha from model
            alpha = (
                self.model_.alpha_
                if hasattr(self.model_, "alpha_")
                else self.model_.alpha
            )

            if stat == "weights":
                # Ridge weights bootstrap
                if use_gpu:
                    result = _bootstrap_ridge_weights_gpu_batched(
                        X,
                        data,
                        alpha=alpha,
                        n_samples=n_samples,
                        save_boots=save_boots,
                        backend=backend,
                        max_gpu_memory_gb=max_gpu_memory_gb,
                        random_state=random_state,
                        percentiles=percentiles,
                        **kwargs,
                    )
                else:
                    result = _bootstrap_ridge_weights_cpu_parallel(
                        X,
                        data,
                        alpha=alpha,
                        n_samples=n_samples,
                        save_boots=save_boots,
                        n_jobs=n_jobs,
                        random_state=random_state,
                        percentiles=percentiles,
                        **kwargs,
                    )

                # Convert results to BrainData format
                return self._convert_bootstrap_results_to_brain_data(
                    result, save_boots=save_boots, return_dict=True
                )

            elif stat == "predict":
                # Ridge predict bootstrap
                if X_test is None:
                    raise ValueError(
                        "X_test parameter required for bootstrap(stat='predict'). "
                        "Provide test features: bootstrap(stat='predict', X_test=...)"
                    )

                X_test = np.asarray(X_test)

                if use_gpu:
                    result = _bootstrap_ridge_predict_gpu_batched(
                        X,
                        data,
                        X_test,
                        alpha=alpha,
                        n_samples=n_samples,
                        save_boots=save_boots,
                        backend=backend,
                        max_gpu_memory_gb=max_gpu_memory_gb,
                        random_state=random_state,
                        percentiles=percentiles,
                        **kwargs,
                    )
                else:
                    result = _bootstrap_ridge_predict_cpu_parallel(
                        X,
                        data,
                        X_test,
                        alpha=alpha,
                        n_samples=n_samples,
                        save_boots=save_boots,
                        n_jobs=n_jobs,
                        random_state=random_state,
                        percentiles=percentiles,
                        **kwargs,
                    )

                # Convert results to BrainData format
                return self._convert_bootstrap_results_to_brain_data(
                    result, save_boots=save_boots, return_dict=True
                )

        else:
            # Invalid stat
            raise ValueError(
                f"Unsupported stat '{stat}'. "
                f"Supported simple stats: {SIMPLE_STATS}. "
                f"Supported fitted model stats: {FITTED_STATS}. "
                f"For fitted stats, you must call .fit() first."
            )

    def _convert_bootstrap_results_to_brain_data(
        self, result, save_boots=False, return_dict=False
    ):
        """Convert bootstrap results dictionary to BrainData format.

        Helper method to convert numpy arrays from bootstrap functions into
        BrainData objects or dicts of BrainData objects.

        Args:
            result: (dict) Result dictionary from bootstrap function with keys:
                    'mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper', and optionally 'samples'
            save_boots: (bool) If True, include 'samples' key in output
            return_dict: (bool) If True, always return dict even for simple stats.
                        If False, return BrainData for simple stats (when save_boots=False)

        Returns:
            BrainData or dict:
                - If return_dict=False and save_boots=False: Returns BrainData with mean
                - Otherwise: Returns dict with BrainData objects for each statistic
        """
        if save_boots:
            # Return dict with samples
            out = {}
            for key in ["mean", "std", "Z", "p", "ci_lower", "ci_upper"]:
                if key in result:
                    out[key] = self._shallow_copy_with_data()
                    # Reshape 1D arrays to 2D (1, n_voxels) for BrainData
                    data_2d = (
                        result[key]
                        if result[key].ndim == 2
                        else result[key].reshape(1, -1)
                    )
                    out[key].data = data_2d
            if "samples" in result:
                out["samples"] = result["samples"]
            return out
        elif return_dict:
            # Return dict format (for model stats)
            out = {}
            for key in ["mean", "std", "Z", "p", "ci_lower", "ci_upper"]:
                if key in result:
                    out[key] = self._shallow_copy_with_data()
                    out[key].data = result[key]
            return out
        else:
            # Return BrainData with mean (for simple stats)
            boot_mean = self._shallow_copy_with_data()
            # Reshape 1D arrays to 2D (1, n_voxels) for BrainData
            mean_2d = (
                result["mean"]
                if result["mean"].ndim == 2
                else result["mean"].reshape(1, -1)
            )
            boot_mean.data = mean_2d
            return boot_mean

    # NOTE: nltools.utils,
    def decompose(
        self, algorithm="pca", axis="voxels", n_components=None, *args, **kwargs
    ):
        """Decompose BrainData object

        Args:
            algorithm: (str) Algorithm to perform decomposition
                        types=['pca','ica','nnmf','fa','dictionary','kernelpca']
            axis: dimension to decompose ['voxels','images']
            n_components: (int) number of components. If None then retain
                        as many as possible.
        Returns:
            output: a dictionary of decomposition parameters
        """

        out = {
            "decomposition_object": set_decomposition_algorithm(
                *args, algorithm=algorithm, n_components=n_components, **kwargs
            )
        }

        if axis == "images":
            out["decomposition_object"].fit(self.data.T)
            out["components"] = self.empty()
            out["components"].data = (
                out["decomposition_object"].transform(self.data.T).T
            )
            out["weights"] = out["decomposition_object"].components_.T
        elif axis == "voxels":
            out["decomposition_object"].fit(self.data)
            out["weights"] = out["decomposition_object"].transform(self.data)
            out["components"] = self.empty()
            out["components"].data = out["decomposition_object"].components_
        return out

    # NOTE: nltools.stats
    def align(self, target, method="procrustes", axis=0, *args, **kwargs):
        """Align BrainData instance to target object using functional alignment

        Alignment type can be hyperalignment or Shared Response Model. When
        using hyperalignment, `target` image can be another subject or an
        already estimated common model. When using SRM, `target` must be a previously
        estimated common model stored as a numpy array. Transformed data can be back
        projected to original data using Tranformation matrix.

        See nltools.stats.align for aligning multiple BrainData instances

        Args:
            target: (BrainData) object to align to.
            method: (str) alignment method to use
                ['probabilistic_srm','deterministic_srm','procrustes']
            axis: (int) axis to align on

        Returns:
            out: (dict) a dictionary containing transformed object,
                transformation matrix, and the shared response matrix

        Examples:
            - Hyperalign using procrustes transform:
                >>> out = data.align(target, method='procrustes')
            - Align using shared response model:
                >>> out = data.align(target, method='probabilistic_srm', n_features=None)
            - Project aligned data into original data:
                >>> original_data = np.dot(out['transformed'].data,out['transformation_matrix'].T)
        """

        if method not in ["probabilistic_srm", "deterministic_srm", "procrustes"]:
            raise ValueError(
                "Method must be ['probabilistic_srm','deterministic_srm','procrustes']"
            )

        source = self.copy()
        data1 = self.data.copy()

        if method == "procrustes":
            target = check_brain_data(target)
            data2 = target.data.copy()

            # pad columns if different shapes
            sizes_1 = [x.shape[1] for x in [data1, data2]]
            C = max(sizes_1)
            y = data1[:, 0:C]
            missing = C - y.shape[1]
            add = np.zeros((y.shape[0], missing))
            data1 = np.append(y, add, axis=1)
        else:
            data2 = target.copy()

        if axis == 1:
            data1 = data1.T
            data2 = data2.T

        out = {}
        if method in ["deterministic_srm", "probabilistic_srm"]:
            if not isinstance(target, np.ndarray):
                raise ValueError(
                    "Common Model must be a numpy array for  ['deterministic_srm', 'probabilistic_srm']"
                )

            if data2.shape[0] != data1.shape[0]:
                raise ValueError(
                    "The number of timepoints(TRs) does not match the model."
                )

            A = data1.T.dot(data2)

            # # Solve the Procrustes problem
            U, _, V = np.linalg.svd(A, full_matrices=False)

            out["transformation_matrix"] = source
            out["transformation_matrix"].data = U.dot(V).T

            out["transformed"] = data1.dot(out["transformation_matrix"].data.T)
            out["common_model"] = target
        elif method == "procrustes":
            _, transformed, out["disparity"], tf_mtx, out["scale"] = procrustes(
                data2, data1
            )
            source.data = transformed
            out["transformed"] = source
            out["common_model"] = target
            out["transformation_matrix"] = source.copy()
            out["transformation_matrix"].data = tf_mtx
        if axis == 1:
            if method == "procrustes":
                out["transformed"].data = out["transformed"].data.T
            else:
                out["transformed"] = out["transformed"].T

        return out

    def smooth(self, fwhm):
        """Apply spatial smoothing using nilearn smooth_img()

        Args:
            fwhm: (float) full width half maximum of gaussian spatial filter
        Returns:
            BrainData instance (copy with smoothed data)
        """
        # Optimized: Use shallow copy instead of deepcopy, single conversion path
        out = self._shallow_copy_with_data()

        # Single conversion: data → nifti → smooth → data
        nifti = self.to_nifti()
        smoothed_nifti = smooth_img(nifti, fwhm)
        smoothed_data = self.nifti_masker.transform(smoothed_nifti)

        # Ensure single images remain 1D
        if check_brain_data_is_single(self):
            out.data = smoothed_data.flatten()
        else:
            out.data = smoothed_data

        return out

    def find_spikes(self, global_spike_cutoff=3, diff_spike_cutoff=3):
        """Function to identify spikes from Time Series Data

        Args:
            global_spike_cutoff: (int,None) cutoff to identify spikes in global signal
                                 in standard deviations, None indicates do not calculate.
            diff_spike_cutoff: (int,None) cutoff to identify spikes in average frame difference
                                 in standard deviations, None indicates do not calculate.
        Returns:
            pandas dataframe with spikes as indicator variables
        """
        return find_spikes(
            self,
            global_spike_cutoff=global_spike_cutoff,
            diff_spike_cutoff=diff_spike_cutoff,
        )

    def temporal_resample(self, sampling_freq=None, target=None, target_type="hz"):
        """
        Resample BrainData timeseries to a new target frequency or number of samples
        using Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) interpolation.
        This function can up- or down-sample data.

        Note: this function can use quite a bit of RAM.

        Args:
            sampling_freq:  (float) sampling frequency of data in hertz
            target: (float) upsampling target
            target_type: (str) type of target can be [samples,seconds,hz]

        Returns:
            upsampled BrainData instance
        """
        # Optimized: Use shallow copy instead of deepcopy
        out = self._shallow_copy_with_data()

        if target_type == "samples":
            n_samples = target
        elif target_type == "seconds":
            n_samples = target * sampling_freq
        elif target_type == "hz":
            n_samples = float(sampling_freq) / float(target)
        else:
            raise ValueError('Make sure target_type is "samples", "seconds", or "hz".')

        orig_spacing = np.arange(0, self.shape[0], 1)
        new_spacing = np.arange(0, self.shape[0], n_samples)

        out.data = np.zeros([len(new_spacing), self.shape[1]])
        for i in range(self.shape[1]):
            interpolate = pchip(orig_spacing, self.data[:, i])
            out.data[:, i] = interpolate(new_spacing)
        return out

    def predict(
        self,
        X: "np.ndarray | None" = None,
        y: "np.ndarray | None" = None,
        method: str = "whole_brain",
        estimator="svm",
        cv=5,
        groups: "np.ndarray | None" = None,
        roi_mask=None,
        radius: float = 10.0,
        scoring: str = "accuracy",
        standardize: bool = True,
        n_jobs: int = -1,
        show_progress: bool = True,
    ):
        """Generate predictions using fitted model OR classify patterns (MVPA).

        This method supports two prediction modes determined by which parameter
        is provided:

        1. **Timeseries prediction** (X provided): Use fitted ridge model to
           predict voxel responses for new feature data.

        2. **MVPA decoding** (y provided): Train a classifier to predict labels
           from brain patterns using cross-validation.

        Args:
            X: Features for timeseries prediction, shape (n_samples, n_features).
                If None and y is None, uses training data from fit().
            y: Labels for MVPA decoding, shape (n_samples,).
                If provided, performs pattern classification instead of
                timeseries prediction.

            # MVPA-specific parameters (only used when y is provided):
            method: Decoding method - 'whole_brain', 'searchlight', or 'roi'.
            estimator: Classifier to use. Can be:
                - 'svm': LinearSVC (default)
                - 'logistic': LogisticRegression
                - 'ridge': RidgeClassifier
                - 'lda': LinearDiscriminantAnalysis
                - Any sklearn-compatible estimator with fit/predict
            cv: Cross-validation specification. Int for k-fold or sklearn CV object.
            groups: Group labels for CV (e.g., run IDs for leave-one-run-out).
            roi_mask: Atlas/parcellation for ROI-based decoding.
            radius: Searchlight radius in mm (default 10.0).
            scoring: Metric for evaluation ('accuracy', 'balanced_accuracy', 'roc_auc').
            standardize: Z-score features before classification (default True).
            n_jobs: Number of parallel jobs for searchlight (-1 = all cores).
            show_progress: Show progress bar for searchlight.

        Returns:
            BrainData: For timeseries prediction, shape (n_samples, n_voxels).
                For MVPA, shape (1, n_voxels) with accuracy per voxel/ROI.

        Raises:
            ValueError: If both X and y are provided.
            ValueError: If fit() has not been called (for timeseries mode).

        Examples:
            >>> # Timeseries prediction (encoding model)
            >>> brain_data.fit(model='ridge', X=features)
            >>> predictions = brain_data.predict(X=new_features)

            >>> # MVPA decoding (pattern classification)
            >>> # brain_data.data has shape (n_trials, n_voxels)
            >>> accuracy = brain_data.predict(y=labels, method='searchlight')
            >>> print(accuracy.shape)  # (1, n_voxels)
        """
        # Validate mutually exclusive modes
        if X is not None and y is not None:
            raise ValueError(
                "Cannot specify both X and y. Use X for timeseries prediction "
                "or y for MVPA decoding."
            )

        # Dispatch to appropriate mode
        if y is not None:
            return self._predict_mvpa(
                y=y,
                method=method,
                estimator=estimator,
                cv=cv,
                groups=groups,
                roi_mask=roi_mask,
                radius=radius,
                scoring=scoring,
                standardize=standardize,
                n_jobs=n_jobs,
                show_progress=show_progress,
            )
        else:
            return self._predict_timeseries(X=X)

    def _predict_timeseries(self, X=None):
        """Generate timeseries predictions using fitted model.

        Internal method for encoding model prediction.

        Args:
            X: Features to predict on. If None, uses training data.

        Returns:
            BrainData with predicted timeseries.
        """
        from nltools.data import BrainData

        # Check model is fitted
        if not hasattr(self, "model_"):
            raise ValueError(
                "Must call fit() before predict(). "
                "Example: brain_data.fit(model='ridge', X=features)"
            )

        if not self.model_.is_fitted_:
            raise ValueError("Model is not fitted")

        # Use training data if X not provided
        using_training_data = X is None
        if using_training_data:
            if not hasattr(self, "X_"):
                raise ValueError(
                    "No training data stored. This should not happen - "
                    "please report this as a bug."
                )
            X = self.X_

        # Validate X
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D")

        # Validate number of features (handle Ridge and Glm differently)
        if hasattr(self.model_, "n_features_in_"):
            # Ridge model has n_features_in_
            if X.shape[1] != self.model_.n_features_in_:
                raise ValueError(
                    f"X has {X.shape[1]} features, but model was fitted with "
                    f"{self.model_.n_features_in_} features"
                )
        else:
            # Glm model - check against design matrices
            if (
                hasattr(self.model_, "design_matrices_")
                and self.model_.design_matrices_
            ):
                expected_features = self.model_.design_matrices_[0].shape[1]
                if X.shape[1] != expected_features:
                    raise ValueError(
                        f"X has {X.shape[1]} features, but model was fitted with "
                        f"{expected_features} features"
                    )

        # Generate predictions
        from nltools.models import Glm

        if isinstance(self.model_, Glm):
            # For GLM, check if using training data or new data
            if using_training_data:
                # Using training data - get fitted values
                y_pred_list = self.model_.predict()  # Returns list of nifti images
                # Convert to array
                y_pred = BrainData(y_pred_list, mask=self.mask).data
            else:
                # New design matrix - not yet implemented in Glm
                raise NotImplementedError(
                    "Prediction with new design matrix not yet implemented for GLM. "
                    "Use predict() without arguments to get fitted values."
                )
        else:
            # Ridge and other models
            y_pred = self.model_.predict(X)

        # Wrap in BrainData
        predictions = self._shallow_copy_with_data()
        predictions.data = y_pred

        return predictions

    def _predict_mvpa(
        self,
        y: np.ndarray,
        method: str = "whole_brain",
        estimator="svm",
        cv=5,
        groups: "np.ndarray | None" = None,
        roi_mask=None,
        radius: float = 10.0,
        scoring: str = "accuracy",
        standardize: bool = True,
        n_jobs: int = -1,
        show_progress: bool = True,
    ):
        """Perform MVPA decoding using cross-validation.

        Internal method for pattern classification.

        Args:
            y: Labels to predict, shape (n_samples,).
            method: 'whole_brain', 'searchlight', or 'roi'.
            estimator: Classifier (string shortcut or sklearn estimator).
            cv: Cross-validation specification.
            groups: Group labels for CV.
            roi_mask: Atlas for ROI-based decoding.
            radius: Searchlight radius in mm.
            scoring: Scoring metric.
            standardize: Whether to z-score features.
            n_jobs: Parallel jobs for searchlight.
            show_progress: Show progress bar.

        Returns:
            BrainData with accuracy values.
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.base import clone

        # Validate method
        valid_methods = {"whole_brain", "searchlight", "roi"}
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method: {method}. Must be one of {valid_methods}"
            )

        # Resolve estimator
        estimator = self._resolve_estimator(estimator)

        # Resolve CV
        if isinstance(cv, int):
            cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        # Validate y
        y = np.asarray(y)
        if y.shape[0] != self.shape[0]:
            raise ValueError(
                f"y has {y.shape[0]} samples but data has {self.shape[0]} samples"
            )

        # Get data as X for classification
        X_data = self.data  # (n_samples, n_voxels)

        # Build pipeline with optional standardization
        if standardize:
            pipe = make_pipeline(StandardScaler(), clone(estimator))
        else:
            pipe = clone(estimator)

        # Dispatch by method
        if method == "whole_brain":
            accuracy = self._mvpa_whole_brain(X_data, y, pipe, cv, groups, scoring)
        elif method == "searchlight":
            accuracy = self._mvpa_searchlight(
                X_data, y, pipe, cv, groups, scoring, radius, n_jobs, show_progress
            )
        elif method == "roi":
            if roi_mask is None:
                raise ValueError("roi_mask required for method='roi'")
            accuracy = self._mvpa_roi(
                X_data, y, pipe, cv, groups, scoring, roi_mask, n_jobs, show_progress
            )

        # Wrap in BrainData
        result = (
            self[0].copy() if len(self.shape) > 1 and self.shape[0] > 1 else self.copy()
        )
        result.data = accuracy.reshape(1, -1) if accuracy.ndim == 1 else accuracy

        return result

    def _resolve_estimator(self, estimator):
        """Resolve string shortcut to sklearn estimator."""
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import LogisticRegression, RidgeClassifier
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        shortcuts = {
            "svm": lambda: LinearSVC(dual="auto", max_iter=10000),
            "logistic": lambda: LogisticRegression(max_iter=1000),
            "ridge": lambda: RidgeClassifier(),
            "lda": lambda: LinearDiscriminantAnalysis(),
        }

        if isinstance(estimator, str):
            if estimator not in shortcuts:
                raise ValueError(
                    f"Unknown estimator: '{estimator}'. "
                    f"Valid options: {list(shortcuts.keys())}"
                )
            return shortcuts[estimator]()

        # Validate sklearn API
        if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
            raise TypeError(
                f"estimator must have fit() and predict() methods. "
                f"Got: {type(estimator).__name__}"
            )

        return estimator

    def _mvpa_whole_brain(self, X, y, pipe, cv, groups, scoring):
        """Whole-brain MVPA - single accuracy across all voxels."""
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(pipe, X, y, cv=cv, groups=groups, scoring=scoring)
        return np.array([np.mean(scores)])

    def _mvpa_searchlight(
        self, X, y, pipe, cv, groups, scoring, radius, n_jobs, show_progress
    ):
        """Searchlight MVPA - accuracy per voxel neighborhood."""
        from sklearn.model_selection import cross_val_score
        from sklearn.base import clone
        from joblib import Parallel, delayed
        from nltools.neighborhoods import compute_searchlight_neighborhoods

        # Get neighborhoods
        neighborhoods = compute_searchlight_neighborhoods(
            self.mask, radius_mm=radius, use_cache=True
        )

        def decode_sphere(center_idx, neighbor_indices):
            """Decode within a single sphere."""
            X_sphere = X[:, neighbor_indices]
            if X_sphere.shape[1] < 2:  # Skip tiny neighborhoods
                return np.nan
            try:
                scores = cross_val_score(
                    clone(pipe), X_sphere, y, cv=cv, groups=groups, scoring=scoring
                )
                return np.mean(scores)
            except Exception:
                return np.nan

        # Collect all neighborhoods
        neighborhood_list = list(neighborhoods.iter_neighborhoods())

        # Progress bar setup
        if show_progress:
            try:
                from tqdm import tqdm

                neighborhood_list = list(
                    tqdm(
                        neighborhood_list,
                        desc="Searchlight",
                        total=neighborhoods.n_voxels,
                    )
                )
            except ImportError:
                pass

        # Parallel execution
        if n_jobs == 1:
            accuracies = [decode_sphere(c, n) for c, n in neighborhood_list]
        else:
            accuracies = Parallel(n_jobs=n_jobs)(
                delayed(decode_sphere)(c, n) for c, n in neighborhood_list
            )

        return np.array(accuracies)

    def _mvpa_roi(
        self, X, y, pipe, cv, groups, scoring, roi_mask, n_jobs, show_progress
    ):
        """ROI-based MVPA - accuracy per ROI."""
        from sklearn.model_selection import cross_val_score
        from sklearn.base import clone
        from joblib import Parallel, delayed
        from nilearn.maskers import NiftiLabelsMasker

        # Load ROI mask if path
        if isinstance(roi_mask, (str, Path)):
            roi_mask = nib.load(roi_mask)

        # Get ROI labels
        roi_data = roi_mask.get_fdata()
        unique_labels = np.unique(roi_data)
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background

        def decode_roi(roi_label):
            """Decode within a single ROI."""
            try:
                masker = NiftiLabelsMasker(
                    labels_img=roi_mask,
                    labels=[roi_label],
                    standardize=False,
                )
                # Extract ROI mean for each sample
                X_roi = masker.fit_transform(self.to_nifti())
                scores = cross_val_score(
                    clone(pipe), X_roi, y, cv=cv, groups=groups, scoring=scoring
                )
                return np.mean(scores)
            except Exception:
                return np.nan

        # Progress bar
        iterator = unique_labels
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(unique_labels, desc="ROI decoding")
            except ImportError:
                pass

        # Parallel execution
        if n_jobs == 1:
            accuracies = [decode_roi(label) for label in iterator]
        else:
            accuracies = Parallel(n_jobs=n_jobs)(
                delayed(decode_roi)(label) for label in iterator
            )

        return np.array(accuracies)

    def plot(
        self,
        kind="glass",
        thr_upper=None,
        thr_lower=None,
        threshold=None,
        cut_coords=None,
        cmap=None,
        bg_img=None,
        ax=None,
        title=None,
        colorbar=True,
        save=None,
        stat="mean",
        **kwargs,
    ):
        """Plot BrainData instance using nilearn visualization or matplotlib.

        Creates brain visualizations using glass brain and/or multi-slice views,
        or matplotlib plots for timeseries/histograms. Respects MNI_Template settings
        for background image selection.

        Args:
            kind (str): Visualization type:
                - 'glass': Glass brain only
                - 'slices': Multi-slice views only
                - 'timeseries': Matplotlib line plot (mean/median/std across voxels)
                - 'histogram': Matplotlib histogram of voxel values
            thr_upper (str/float, optional): Upper threshold. Can be:
                - Float/int: Absolute threshold value
                - String ending in '%': Percentile threshold (e.g., '95%')
            thr_lower (str/float, optional): Lower threshold (same format as thr_upper)
            threshold (float, optional): Convenience parameter. If positive, sets thr_upper.
                If negative, sets thr_lower. Overrides thr_upper/thr_lower if provided.
            cut_coords (list, optional): Cut coordinates for multi-slice views.
                Format: [[x_coords], [y_coords], [z_coords]].
                If None, uses default coordinates.
            cmap (str, optional): Colormap name. If None, auto-selects based on data.
            bg_img (str/nibabel image, optional): Background image.
                If None, uses get_mni_from_img_resolution() which respects MNI_Template settings.
            ax (matplotlib.axes.Axes, optional): Matplotlib axis for timeseries/histogram plots.
                If None, creates new figure.
            title (str, optional): Plot title. If None, auto-generates based on data.
            colorbar (bool): Whether to show colorbar (nilearn plots only). Default: True.
            save (str, optional): Path to save figure(s). If provided, saves plots.
            stat (str): Statistic for timeseries plots ('mean', 'median', 'std').
                Default: 'mean'. Only used when kind='timeseries'.
            **kwargs: Additional arguments passed to nilearn plot functions.

        Returns:
            Display or matplotlib Figure: Nilearn Display object(s) or matplotlib Figure.

        Raises:
            ValueError: If BrainData is empty or invalid 'kind' parameter.
            RuntimeError: If to_nifti() conversion fails.

        Examples:
            >>> # Basic plotting with defaults
            >>> brain.plot()
            >>>
            >>> # Glass brain only
            >>> brain.plot(kind='glass')
            >>>
            >>> # Multi-slice views
            >>> brain.plot(kind='slices')
            >>>
            >>> # With thresholding
            >>> brain.plot(thr_upper='95%')
            >>> brain.plot(threshold=2.5)  # Convenience parameter
            >>>
            >>> # Timeseries plotting
            >>> brain.plot(kind='timeseries', stat='mean')
            >>> brain.plot(kind='timeseries', stat='std', ax=my_axis)
            >>>
            >>> # Histogram plotting
            >>> brain.plot(kind='histogram')
            >>>
            >>> # Custom title
            >>> brain.plot(title='My Brain Map')

        Note:
            This method respects MNI_Template settings (from nltools.prefs).
            Background images are automatically selected based on data resolution
            and current MNI_Template.template/resolution settings.
        """
        from nilearn.plotting import plot_glass_brain, plot_stat_map
        from nltools.utils import get_mni_from_img_resolution
        import matplotlib.pyplot as plt

        # Validate inputs
        if self.isempty:
            raise ValueError("Cannot plot empty BrainData object")

        # Handle convenience threshold parameter
        if threshold is not None:
            if threshold >= 0:
                thr_upper = threshold
            else:
                thr_lower = threshold

        # Validate 'kind' parameter
        valid_kinds = ["glass", "slices", "timeseries", "histogram"]
        if kind not in valid_kinds:
            raise ValueError(
                f"Invalid 'kind' parameter: '{kind}'. Must be one of: {valid_kinds}. "
            )

        # Handle matplotlib-based plots (timeseries, histogram)
        if kind in ["timeseries", "histogram"]:
            return self._plot_matplotlib(
                kind=kind, stat=stat, ax=ax, title=title, save=save
            )

        # Handle thresholding
        if thr_upper or thr_lower:
            obj = self.threshold(upper=thr_upper, lower=thr_lower)
        else:
            obj = self

        # Ensure single image for plotting
        if len(obj.shape) > 1 and obj.shape[0] > 1:
            obj = obj[0]

        # Default cut coordinates
        if cut_coords is None:
            cut_coords = [
                range(-50, 51, 8),  # x coordinates
                range(-80, 50, 10),  # y coordinates
                range(-40, 71, 9),  # z coordinates
            ]

        # Default colormap with auto-selection
        if cmap is None:
            cmap = self._auto_select_colormap(obj.data)

        # Views for multi-slice plotting
        views = ["x", "y", "z"]

        # Handle save paths
        save_paths = self._prepare_save_paths(save) if save else None

        # Convert to nifti
        try:
            nifti_img = obj.to_nifti()
        except Exception as e:
            raise RuntimeError(f"Failed to convert BrainData to NIfTI: {e}") from e

        # Plot based on 'kind' parameter
        display_objects = []

        # Prepare kwargs with title if provided
        # Remove 'how' from kwargs if present (backward compatibility)
        plot_kwargs = kwargs.copy()
        plot_kwargs.pop("how", None)  # Remove 'how' if accidentally passed
        if title:
            plot_kwargs["title"] = title

        if kind == "glass":
            display_glass = plot_glass_brain(
                nifti_img,
                display_mode="lzry",
                colorbar=colorbar,
                cmap=cmap,
                plot_abs=False,
                **plot_kwargs,
            )
            display_objects.append(display_glass)
            if save_paths:
                plt.savefig(save_paths["glass"], bbox_inches="tight")

        elif kind == "slices":
            # Background image selection (respects MNI_Template)
            if bg_img is None:
                try:
                    bg_img = get_mni_from_img_resolution(obj, img_type="brain")
                except ValueError as e:
                    # Handle non-isometric voxels gracefully
                    if "isometric" in str(e).lower():
                        # Use default MNI template as fallback
                        from nltools.prefs import MNI_Template

                        warnings.warn(
                            f"Non-isometric voxels detected: {str(e)}. "
                            f"Using default MNI152 template ({MNI_Template.template}, "
                            f"{MNI_Template.resolution}mm) as background image. "
                            f"To use a custom background, provide bg_img parameter.",
                            UserWarning,
                            stacklevel=2,
                        )
                        bg_img = MNI_Template.brain
                    else:
                        # Re-raise if it's a different ValueError
                        raise
            for v, c, savefile in zip(
                views, cut_coords, save_paths["slices"] if save_paths else [None] * 3
            ):
                display_slice = plot_stat_map(
                    nifti_img,
                    cut_coords=c,
                    display_mode=v,
                    cmap=cmap,
                    bg_img=bg_img,
                    colorbar=colorbar,
                    **plot_kwargs,
                )
                display_objects.append(display_slice)
                if savefile:
                    plt.savefig(savefile, bbox_inches="tight")

        # Return last display object or None
        return display_objects[-1] if display_objects else None

    def plot_flatmap(
        self,
        threshold=None,
        cmap="RdBu_r",
        vmax=None,
        vmin=None,
        template="fsaverage5",
        with_curvature=True,
        curvature_contrast=0.5,
        curvature_brightness=0.5,
        colorbar=True,
        colorbar_orientation="horizontal",
        figsize=(12, 6),
        title=None,
        radius=3.0,
        interpolation="linear",
        axes=None,
        save=None,
    ):
        """Plot brain data on cortical flatmap.

        Projects MNI152 volumetric data onto an fsaverage surface and renders
        as a 2D flattened cortical map. Uses nilearn's vol_to_surf for
        projection and matplotlib's tripcolor for rendering.

        This method provides publication-quality flatmap visualizations
        without requiring external dependencies like pycortex.

        Args:
            threshold (float or str, optional): Values below this absolute
                threshold are masked. Can be a float or percentile string
                like '95%'. Defaults to None (no threshold).
            cmap (str, optional): Matplotlib colormap for data. Defaults to
                'RdBu_r' (diverging red-blue).
            vmax (float, optional): Maximum value for colormap. If None,
                uses symmetric max of absolute values.
            vmin (float, optional): Minimum value for colormap. If None
                and vmax is set, uses -vmax for diverging maps.
            template (str, optional): fsaverage resolution. Options:
                'fsaverage3' (642 vertices), 'fsaverage4' (2562),
                'fsaverage5' (10242, default), 'fsaverage6' (40962),
                'fsaverage' (163842, full resolution).
            with_curvature (bool, optional): Show sulcal/gyral pattern as
                grayscale background. Defaults to True.
            curvature_contrast (float, optional): Contrast of curvature
                (0=flat gray, 1=full contrast). Defaults to 0.5.
            curvature_brightness (float, optional): Mean brightness of
                curvature (0=dark, 1=bright). Defaults to 0.5.
            colorbar (bool, optional): Show colorbar. Defaults to True.
            colorbar_orientation (str, optional): 'horizontal' or 'vertical'.
                Defaults to 'horizontal'.
            figsize (tuple, optional): Figure size (width, height).
                Defaults to (12, 6).
            title (str, optional): Figure title. Defaults to None.
            radius (float, optional): Sampling radius in mm for vol_to_surf
                projection. Larger values provide smoother projections.
                Defaults to 3.0.
            interpolation (str, optional): Interpolation for vol_to_surf.
                Options: 'linear', 'nearest'. Defaults to 'linear'.
            axes (matplotlib.axes.Axes, optional): Existing axes to plot on.
                If None, creates new figure. Defaults to None.
            save (str, optional): File path to save figure. Defaults to None.

        Returns:
            matplotlib.figure.Figure: The figure containing the flatmap.

        Examples:
            >>> # Basic flatmap
            >>> brain.plot_flatmap()
            >>>
            >>> # Thresholded with custom colormap
            >>> brain.plot_flatmap(threshold=2.5, cmap='hot')
            >>>
            >>> # Percentile threshold, no curvature background
            >>> brain.plot_flatmap(threshold='95%', with_curvature=False)
            >>>
            >>> # High resolution for publication
            >>> fig = brain.plot_flatmap(template='fsaverage6')
            >>> fig.savefig('flatmap.pdf', dpi=300)

        Notes:
            - Data is projected from MNI152 space to fsaverage surface space.
              Small alignment differences are expected at boundaries.
            - Higher resolution templates (fsaverage6, fsaverage) produce
              sharper images but take longer to render.
            - The flat surfaces are cached by nilearn after first download.
        """
        from nltools.plotting import plot_flatmap

        if self.isempty:
            raise ValueError("Cannot plot empty BrainData object")

        return plot_flatmap(
            brain=self,
            threshold=threshold,
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
            template=template,
            with_curvature=with_curvature,
            curvature_contrast=curvature_contrast,
            curvature_brightness=curvature_brightness,
            colorbar=colorbar,
            colorbar_orientation=colorbar_orientation,
            figsize=figsize,
            title=title,
            radius=radius,
            interpolation=interpolation,
            axes=axes,
            save=save,
        )

    def _plot_matplotlib(
        self,
        kind,
        stat="mean",
        ax=None,
        title=None,
        save=None,
    ):
        """Plot using matplotlib (timeseries or histogram).

        Args:
            kind (str): 'timeseries' or 'histogram'
            stat (str): Statistic for timeseries ('mean', 'median', 'std')
            ax (matplotlib.axes.Axes, optional): Axis to plot on
            title (str, optional): Plot title
            save (str, optional): Path to save figure

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        import matplotlib.pyplot as plt

        # Create axis if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        if kind == "timeseries":
            # For single image, raise informative error
            if len(self.shape) == 1 or (len(self.shape) > 1 and self.shape[0] == 1):
                raise ValueError(
                    "timeseries plotting requires multiple images. "
                    f"Got {self.shape[0] if len(self.shape) > 1 else 1} image(s). "
                    "Use histogram for single image visualization."
                )

            # Compute statistic across voxels for each image
            if stat == "mean":
                values = self.mean(axis=1)
            elif stat == "median":
                values = self.median(axis=1)
            elif stat == "std":
                values = self.std(axis=1)
            else:
                raise ValueError(
                    f"Invalid stat '{stat}'. Must be 'mean', 'median', or 'std'"
                )

            # Ensure values is 1D array
            if hasattr(values, "data"):
                values = values.data
            values = np.array(values).flatten()

            # Plot
            ax.plot(values, linewidth=2)
            ax.set_xlabel("Image Index", fontsize=12)
            ax.set_ylabel(f"{stat.capitalize()} Across Voxels", fontsize=12)
            if title is None:
                title = f"{stat.capitalize()} Across Voxels"
            ax.set_title(title, fontsize=14)
            ax.grid(True, alpha=0.3)

        elif kind == "histogram":
            # Flatten data for histogram
            if len(self.shape) == 1:
                data_flat = self.data
            else:
                data_flat = self.data.flatten()

            # Remove NaN/Inf
            data_flat = data_flat[np.isfinite(data_flat)]

            # Plot histogram
            ax.hist(data_flat, bins=50, edgecolor="black", alpha=0.7)
            ax.set_xlabel("Voxel Value", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            if title is None:
                title = "Voxel Value Distribution"
            ax.set_title(title, fontsize=14)
            ax.grid(True, alpha=0.3)

        # Save if requested
        if save:
            fig.savefig(save, bbox_inches="tight", dpi=150)

        return fig

    def _auto_select_colormap(self, data):
        """Auto-select colormap based on data characteristics.

        Args:
            data: numpy array of brain data

        Returns:
            str: Colormap name
        """
        # Flatten data for analysis
        if data.ndim > 1:
            data_flat = data.flatten()
        else:
            data_flat = data

        # Remove NaN/Inf
        data_flat = data_flat[np.isfinite(data_flat)]

        if len(data_flat) == 0:
            return "RdBu_r"  # Default fallback

        # Check data range for colormap selection
        # If mostly positive (> 90% positive), use hot/reds
        positive_ratio = np.sum(data_flat > 0) / len(data_flat)
        if positive_ratio > 0.9:
            return "hot"
        # If mostly negative (> 90% negative), use cool/blues
        elif (1 - positive_ratio) > 0.9:
            return "cool"
        # Otherwise use bipolar
        else:
            return "RdBu_r"

    def _prepare_save_paths(self, save):
        """Prepare save paths for multiple plot outputs.

        Args:
            save: Base save path (str or Path)

        Returns:
            dict: Dictionary with 'glass' and 'slices' keys containing save paths
        """
        save = str(save)  # Convert Path objects to strings
        path, filename = os.path.split(save)
        if "." in filename:
            filename, extension = filename.rsplit(".", 1)
        else:
            extension = "png"

        base_path = os.path.join(path, filename) if path else filename

        return {
            "glass": f"{base_path}_glass.{extension}",
            "slices": [
                f"{base_path}_x.{extension}",
                f"{base_path}_y.{extension}",
                f"{base_path}_z.{extension}",
            ],
        }

    # NOTE: Tracked in beads issue nltools-5dw for Model class refactoring
    def randomise(self, *args, **kwargs):
        """DEPRECATED: This method has been moved to the Model class."""
        raise NotImplementedError(
            "randomise() has been deprecated. Please use the new Model class for permutation-based inference."
        )

    # NOTE: Tracked in beads issue nltools-5dw for Model class refactoring
    def ttest(self, *args, **kwargs):
        """DEPRECATED: This method has been moved to the Model class."""
        raise NotImplementedError(
            "ttest() has been deprecated. Please use the new Model class for statistical testing."
        )


class BrainDataPipeline:
    """Pipeline specialized for BrainData with CV support.

    Wraps the base Pipeline to handle BrainData-specific operations
    like splitting by samples and accessing the underlying data array.
    """

    def __init__(self, brain_data: "BrainData", cv=None, groups=None):
        from nltools.pipelines.base import FittedStack  # noqa: F401

        self._brain_data = brain_data
        self._cv = cv
        self._groups = groups
        self._steps = []

    @property
    def data(self):
        """Get underlying data array."""
        return self._brain_data.data

    @property
    def cv(self):
        return self._cv

    @property
    def n_steps(self):
        return len(self._steps)

    def _add_step(self, step) -> "BrainDataPipeline":
        """Add step and return new pipeline (immutable)."""
        from copy import copy

        new = copy(self)
        new._steps = self._steps + [step]
        return new

    def normalize(self, method: str = "zscore", **kwargs) -> "BrainDataPipeline":
        """Add normalization step."""
        from nltools.pipelines.steps import NormalizeStep

        return self._add_step(NormalizeStep(method=method, **kwargs))

    def reduce(
        self, method: str = "pca", n_components: int = None, **kwargs
    ) -> "BrainDataPipeline":
        """Add dimensionality reduction step."""
        from nltools.pipelines.steps import ReduceStep

        return self._add_step(
            ReduceStep(method=method, n_components=n_components, **kwargs)
        )

    def pipe(self, transformer) -> "BrainDataPipeline":
        """Add custom sklearn transformer."""
        from nltools.pipelines.steps import PipeStep

        return self._add_step(PipeStep(transformer=transformer))

    def predict(self, y, algorithm: str = "ridge", **kwargs):
        """Execute pipeline with CV and return prediction results.

        This is a terminal method that executes the full pipeline.

        Args:
            y: Target variable (labels or continuous values).
            algorithm: Prediction algorithm ('ridge', 'svm', etc.)
            **kwargs: Additional arguments for the predictor.

        Returns:
            BrainDataCVResult with scores, predictions, and fold information.
        """
        from nltools.pipelines.base import FittedStack
        from sklearn.linear_model import Lasso, Ridge
        from sklearn.svm import SVC, SVR

        if self._cv is None:
            raise ValueError("predict() requires CV context")

        data = self.data
        y = np.asarray(y)

        results = []
        for train_idx, test_idx in self._cv.split(data, groups=self._groups):
            train_data = data[train_idx]
            test_data = data[test_idx]
            train_y = y[train_idx]
            test_y = y[test_idx]

            fitted_stack = FittedStack()

            # Apply transform steps
            for step in self._steps:
                fitted = step.fit(train_data)
                fitted_stack.append(fitted)
                train_data = fitted.transform(train_data)
                test_data = fitted.transform(test_data)

            # Fit predictor and evaluate
            if algorithm == "ridge":
                model = Ridge(**kwargs)
            elif algorithm == "lasso":
                model = Lasso(**kwargs)
            elif algorithm == "svr":
                model = SVR(**kwargs)
            elif algorithm == "svm":
                model = SVC(**kwargs)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            model.fit(train_data, train_y)
            predictions = model.predict(test_data)
            score = model.score(test_data, test_y)

            results.append(
                {
                    "score": score,
                    "predictions": predictions,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "fitted_stack": fitted_stack,
                }
            )

        return BrainDataCVResult(results, self)


class BrainDataCVResult:
    """Cross-validation results for BrainData pipelines."""

    def __init__(self, fold_results: list, pipeline):
        self.fold_results = fold_results
        self.pipeline = pipeline

    @property
    def scores(self) -> np.ndarray:
        """Per-fold scores."""
        return np.array([f["score"] for f in self.fold_results])

    @property
    def mean_score(self) -> float:
        """Mean score across folds."""
        return self.scores.mean()

    @property
    def std_score(self) -> float:
        """Standard deviation of scores."""
        return self.scores.std()

    @property
    def predictions(self) -> np.ndarray:
        """All predictions in original sample order."""
        # Reconstruct in original order
        n_samples = sum(len(f["test_idx"]) for f in self.fold_results)
        preds = np.zeros(n_samples)
        for f in self.fold_results:
            preds[f["test_idx"]] = f["predictions"]
        return preds

    def __repr__(self):
        return f"BrainDataCVResult(n_folds={len(self.fold_results)}, mean_score={self.mean_score:.4f})"
