"""
NeuroLearn Brain Data
=====================

Classes to represent brain image data.

"""

import os
import numpy as np
import pandas as pd
import warnings  # noqa: F401
from copy import deepcopy
from nltools.utils import (
    attempt_to_import,
    check_brain_data,
    check_brain_data_is_single,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import BrainDataPipeline

warnings.filterwarnings("ignore", category=UserWarning, module="nilearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="nilearn")

# Optional dependencies
nx = attempt_to_import("networkx", "nx")
tables = attempt_to_import("tables")
MAX_INT = np.iinfo(np.int32).max

__all__ = ["BrainData"]


class BrainData(object):
    """
    BrainData is a class to represent neuroimaging data in python as a vector
    rather than a 3-dimensional matrix. This makes it easier to perform data
    manipulation and analyses.

    Args:
        data: nibabel data instance or list of files
        Y: Pandas DataFrame of training labels
        X: Pandas DataFrame Design Matrix for running univariate models
        mask: binary nifti file to mask brain data
        masker: nilearn masker object (e.g., ROI or searchlight extractor).
            Default uses voxel-level masking.
        **kwargs: Additional keyword arguments passed to NiftiMasker

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
            mask: Brain mask. Can be None (uses MNI template), a nibabel
                Nifti1Image, a file path (str/Path) to a mask file, or a template
                name string like ``'2mm-MNI152-2009c'`` (version: 'fsl' for
                default/, 'a' for nilearn/, 'c' for fmriprep/).
            masker: nilearn masker object (e.g. ROI or searchlight extractor).
                Default will load data as voxels.
            resample (bool, default=True): Whether to automatically resample data
                to mask space. If True, data is resampled to match mask spatial
                characteristics. If False, data must already be in mask space.
                Default True preserves backward compatibility with v0.5.1.
            interpolation (str, default='auto'): Interpolation method for resampling.
                Options: 'auto' (detect based on data type; uses 'nearest' for
                discrete data like atlases/masks and 'continuous' for stat maps),
                'nearest' (nearest-neighbor, preserves discrete values),
                'linear' (linear interpolation),
                'continuous' (higher-order spline, use for stat maps).
            **kwargs: Additional arguments passed to NiftiMasker.
        """
        # Import validation functions
        from ..validation import validate_data_type

        # Import I/O functions
        from .io import (
            initialize_mask,
            load_from_list,
            load_from_brain_data,
            load_from_h5,
            load_from_url,
            load_from_file,
        )

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
        initialize_mask(self, mask, **kwargs)

        # Initialize data based on type
        data_type = validate_data_type(data)

        if data_type == "none":
            self.data = np.array([])
        elif data_type == "brain_data":
            # Copy from another BrainData object
            # Note: mask was already initialized, but we may need to override it
            load_from_brain_data(self, data, mask)
        elif data_type == "h5":
            load_from_h5(self, data, mask)
            # H5 loading sets X and Y, so we're done
            return
        elif data_type == "list":
            load_from_list(self, data)
        elif data_type == "url":
            load_from_url(self, data)
        elif data_type in ["file", "nibabel"]:
            load_from_file(self, data)

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

    # =========================================================================
    # I/O delegation methods (thin wrappers around _io functions)
    # =========================================================================

    def _initialize_mask(self, mask, **kwargs):
        """Initialize the mask and NiftiMasker.

        Args:
            mask: Brain mask as nibabel object, file path, template name string, or None.
                Template name strings supported: '{res}mm-MNI152-2009{version}'
                (e.g., '2mm-MNI152-2009c', '3mm-MNI152-2009a', '2mm-MNI152-2009fsl')
            **kwargs: Additional arguments passed to NiftiMasker.
        """
        from .io import initialize_mask

        initialize_mask(self, mask, **kwargs)

    def _get_interpolation(self, img):
        """Get the interpolation method to use for a given image.

        Resolves 'auto' to either 'nearest' or 'continuous' based on data type.

        Args:
            img: nibabel image to check (used when interpolation='auto')

        Returns:
            str: 'nearest', 'linear', or 'continuous'
        """
        from .io import get_interpolation

        return get_interpolation(self, img)

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
        from .io import detect_and_update_mask

        return detect_and_update_mask(self, data_img)

    def _detect_space(self, mask):
        """Detect if mask is in MNI space or native space.

        Args:
            mask: nibabel Nifti1Image object

        Returns:
            str: 'mni' if mask is MNI template, 'native' otherwise
        """
        from .io import detect_space

        return detect_space(self, mask)

    def _check_space_match(self, data_img, mask_img):
        """Check if data and mask are in same space.

        Args:
            data_img: nibabel Nifti1Image object
            mask_img: nibabel Nifti1Image object (mask)

        Returns:
            bool: True if spaces match (no resampling needed), False otherwise
        """
        from .io import check_space_match

        return check_space_match(data_img, mask_img)

    def _warn_if_resampling(self, context=""):
        """Warn about resampling if verbose=True and resample=True.

        Args:
            context: Optional context string to include in warning message.
        """
        from .io import warn_if_resampling

        warn_if_resampling(self, context)

    def _load_from_list(self, data_list):
        """Load data from a list of BrainData objects or file paths.

        Args:
            data_list: List of BrainData objects or file paths.
        """
        from .io import load_from_list

        load_from_list(self, data_list)

    def _load_from_brain_data(self, brain_data, mask=None):
        """Load data from another BrainData object.

        Args:
            brain_data: BrainData object to copy from.
            mask: Optional mask to use. If None, uses mask from brain_data.
        """
        from .io import load_from_brain_data

        load_from_brain_data(self, brain_data, mask)

    def _load_from_h5(self, file_path, mask):
        """Load data from HDF5 file.

        Args:
            file_path: Path to HDF5 file.
            mask: User-specified mask (to determine if we should load mask from file).
        """
        from .io import load_from_h5

        load_from_h5(self, file_path, mask)

    def _load_from_url(self, url):
        """Load data from URL.

        Args:
            url: URL to download data from.
        """
        from .io import load_from_url

        load_from_url(self, url)

    def _load_from_file(self, data):
        """Load data from file path or nibabel object.

        Args:
            data: File path or nibabel object.
        """
        from .io import load_from_file

        load_from_file(self, data)

    def to_nifti(self):
        """Convert BrainData Instance into Nifti Object.

        Returns:
            nibabel.Nifti1Image: Brain data as a NIfTI image.
        """
        from .io import to_nifti

        return to_nifti(self)

    def resample_to(self, img=None, resolution=None, interpolation=None):
        """Resample BrainData to match target image or resolution.

        Args:
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
        from .io import resample_to

        return resample_to(self, img, resolution, interpolation)

    def write(self, file_name):
        """Write out BrainData object to Nifti or HDF5 File.

        Args:
            file_name (str or Path): Output file path (.nii/.nii.gz for NIfTI,
                .h5/.hdf5 for HDF5).

        """
        from .io import write_brain_data

        write_brain_data(self, file_name)

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
        from .io import upload_neurovault

        return upload_neurovault(
            self,
            access_token,
            collection_name,
            collection_id,
            img_type,
            img_modality,
            **kwargs,
        )

    # =========================================================================
    # Arithmetic and functional helpers (inline, to be extracted later)
    # =========================================================================

    def _perform_arithmetic(self, other, operation, operation_name, inplace=False):
        """Perform arithmetic operation with validation.

        Args:
            other: The other operand.
            operation: The operation function (e.g., np.add, np.subtract).
            operation_name: Name of the operation for error messages.
            inplace (bool): If True, modify in-place. Default: False.

        Returns:
            BrainData: Result of the operation.
        """
        from ..validation import (
            validate_arithmetic_operand,
            validate_brain_data_shapes,
        )

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

    # =========================================================================
    # Dunder methods
    # =========================================================================

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
        from ..validation import (
            validate_arithmetic_operand,
            validate_brain_data_shapes,
        )

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

    # =========================================================================
    # Copy methods
    # =========================================================================

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
        (``model_``, ``X_``, ``cv_results_``) is NOT propagated since the new
        data shape would invalidate the original fit.

        Returns:
            BrainData: Deep copied instance
        """
        return deepcopy(self)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def shape(self):
        """Get images by voxels shape."""

        return self.data.shape

    @property
    def dtype(self):
        """Get data type of BrainData.data."""
        return self.data.dtype

    # =========================================================================
    # Aggregation methods
    # =========================================================================

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

    # =========================================================================
    # Modeling methods (delegated to _modeling.py)
    # =========================================================================

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
            axis: (int or None) Axis along which to compute the mean.
                None for grand-mean scaling (default, FSL/SPM style).
                0 for voxel-wise scaling (AFNI style, each voxel scaled
                by its own temporal mean).

        Returns:
            BrainData: New BrainData instance with scaled data.

        Examples:
            >>> # Grand-mean scaling (default)
            >>> scaled = brain.scale(100.0)
            >>>
            >>> # Voxel-wise scaling (AFNI style)
            >>> scaled = brain.scale(100.0, axis=0)

        """
        from .analysis import scale_data

        return scale_data(self, scale_val, axis)

    def cv(
        self,
        k: int | None = None,
        scheme: str = "kfold",
        split_by: str | None = None,
        groups: np.ndarray | None = None,
        random_state: int | None = None,
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
        from .modeling import cv

        return cv(
            self,
            k=k,
            scheme=scheme,
            split_by=split_by,
            groups=groups,
            random_state=random_state,
            **kwargs,
        )

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
            glm_r2 (BrainData, optional): R-squared values (for model='glm')
            ridge_weights (BrainData, optional): Model coefficients (for model='ridge')
            ridge_fitted_values (BrainData, optional): Fitted values (for model='ridge')
            ridge_scores (BrainData, optional): R-squared scores (for model='ridge')

        Examples:
            >>> # Old behavior (backward compatible): mutate self
            >>> brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
            >>> print(f"CV R2: {brain_data.cv_results_['mean_score'].mean():.3f}")
            >>> weights = brain_data.ridge_weights  # Access as attribute
            >>>
            >>> # New behavior: return Fit dataclass (self unchanged)
            >>> fit = brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features, inplace=False)
            >>> assert isinstance(fit, Fit)
            >>> assert 'weights' in fit.available()
            >>> assert not hasattr(brain_data, 'ridge_weights')  # brain_data unchanged
            >>> print(f"CV R2: {fit.cv_mean_score.mean():.3f}")
            >>>
            >>> # GLM with Fit dataclass
            >>> fit_glm = brain_data.fit(model='glm', X=design_matrix, inplace=False)
            >>> assert 'betas' in fit_glm.available()
            >>> assert 't_stats' in fit_glm.available()
        """
        from .modeling import fit

        return fit(
            self,
            model=model,
            X=X,
            cv=cv,
            inplace=inplace,
            progress_bar=progress_bar,
            scale=scale,
            scale_value=scale_value,
            **kwargs,
        )

    def _fit_ridge(self, X, cv=None, **kwargs):
        """Fit Ridge model and extract results.

        Args:
            X (ndarray): Training features
            cv (int, 'auto', or sklearn CV splitter, optional): Cross-validation specification
            **kwargs (dict): Additional arguments for CV (alpha, alphas, backend, etc.)
        """
        from .modeling import fit_ridge

        return fit_ridge(self, X, cv=cv, **kwargs)

    def _compute_ridge_cv(
        self, X, cv, alpha=None, alphas=None, backend="auto", **kwargs
    ):
        """Compute cross-validation results for Ridge regression.

        Args:
            X (ndarray or list): Training features.
            cv (int, 'auto', or sklearn CV splitter): Cross-validation specification
            alpha (float or 'auto', optional): Regularization strength
            alphas (array-like, optional): Alpha values to try for alpha selection
            backend (str): Computational backend. Default: 'auto'
            **kwargs (dict): Additional arguments

        Returns:
            dict: CV results dictionary.
        """
        from .modeling import compute_ridge_cv

        return compute_ridge_cv(
            self, X, cv, alpha=alpha, alphas=alphas, backend=backend, **kwargs
        )

    def _fit_glm(self, X):
        """Fit GLM model and extract results.

        Args:
            X: Design matrix.
        """
        from .modeling import fit_glm

        return fit_glm(self, X)

    def _to_fit_dataclass(self, model):
        """Convert BrainData fit results to Fit dataclass.

        Args:
            model (str): Model type ('ridge' or 'glm')

        Returns:
            Fit: Dataclass containing fit results
        """
        from .modeling import to_fit_dataclass

        return to_fit_dataclass(self, model)

    def regress(self, design_matrix=None, noise_model="ols", mode=None, **kwargs):
        """Deprecated: Use fit(model='glm', X=design_matrix) instead.

        .. deprecated:: 0.6.0
            Use :meth:`fit` with ``model='glm'`` instead.
        """
        from .modeling import regress

        return regress(
            self,
            design_matrix=design_matrix,
            noise_model=noise_model,
            mode=mode,
            **kwargs,
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
        from .modeling import compute_contrasts

        return compute_contrasts(self, contrasts, contrast_type=contrast_type)

    def _parse_contrast_string(self, contrast_str):
        """Parse a contrast string into a numeric contrast vector.

        Args:
            contrast_str (str): Contrast string like "A - B" or "2*A - B - C"

        Returns:
            np.array: Numeric contrast vector
        """
        from .modeling import parse_contrast_string

        return parse_contrast_string(self, contrast_str)

    # =========================================================================
    # Data manipulation methods
    # =========================================================================

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
        from ..validation import validate_append_shapes

        data = check_brain_data(data)

        if self.is_empty:
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

    def create_empty(self):
        """Create a copy of BrainData with empty data array.

        Returns:
            BrainData: A copy of this object with an empty data array.
        """
        from copy import deepcopy

        out = deepcopy(self)
        out.data = np.array([])
        return out

    def empty(self):
        """Deprecated: Create a copy of BrainData with empty data array.

        .. deprecated:: 0.6.0
            Use :meth:`create_empty` instead.
        """
        import warnings

        warnings.warn(
            "empty() is deprecated and will be removed in a future version. "
            "Use create_empty() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.create_empty()

    @property
    def is_empty(self) -> bool:
        """Check if BrainData.data is empty.

        Returns:
            bool: True if the data array is empty, False otherwise.
        """
        if isinstance(self.data, np.ndarray):
            return self.data.size == 0
        if isinstance(self.data, list):
            return len(self.data) == 0
        return True

    @property
    def isempty(self) -> bool:
        """Check if BrainData.data is empty.

        .. deprecated:: 0.6.0
            Use :attr:`is_empty` instead.
        """
        import warnings

        warnings.warn(
            "isempty is deprecated and will be removed in a future version. "
            "Use is_empty instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.is_empty

    def _check_masks(self, image):
        """Check to make sure masks are the same for each dataset and if not create a union mask.

        Args:
            image: BrainData instance to compare masks with.

        Returns:
            tuple: (data2, image2) numpy arrays with compatible masks.
        """
        from .analysis import check_masks

        return check_masks(self, image)

    # =========================================================================
    # Analysis methods (delegated to _analysis.py)
    # =========================================================================

    def similarity(self, image, method="correlation"):
        """Calculate similarity of BrainData() instance with single
        BrainData or Nibabel image.

        Args:
            image: (BrainData, nifti) image to evaluate similarity
            method: (str) Type of similarity
                    ['correlation','dot_product','cosine']

        Returns:
            float or np.ndarray: Similarity value(s).
        """
        from .analysis import similarity

        return similarity(self, image, method=method)

    def distance(self, metric="euclidean", **kwargs):
        """Calculate distance between images within a BrainData() instance.

        Args:
            metric: (str) type of distance metric (can use any scipy.spatial.distance
                    metric supported by cdist)

        Returns:
            Adjacency: Pairwise distance matrix.
        """
        from .analysis import distance

        return distance(self, metric=metric, **kwargs)

    def multivariate_similarity(self, images, method="ols"):
        """Predict spatial distribution of BrainData() instance from linear
        combination of other BrainData() instances or Nibabel images.

        Args:
            images: BrainData instance of weight map
            method (str): Regression method. Default: 'ols'.

        Returns:
            out: dictionary of regression statistics in BrainData
                instances {'beta','t','p','df','residual'}
        """
        from .analysis import multivariate_similarity

        return multivariate_similarity(self, images, method=method)

    def apply_mask(self, mask, resample_mask_to_brain=False):
        """Mask BrainData instance using nilearn functionality.

        Note target data will be resampled into the same space as the mask. If you would like the mask
        resampled into the BrainData space, then set resample_mask_to_brain=True.

        Args:
            mask: (BrainData or nifti object) mask to apply to BrainData object.
            resample_mask_to_brain: (bool) Will resample mask to brain space before applying mask (default=False).

        Returns:
            masked: (BrainData) masked BrainData object
        """
        from .analysis import apply_mask

        return apply_mask(self, mask, resample_mask_to_brain=resample_mask_to_brain)

    def extract_roi(self, mask, metric="mean", n_components=None):
        """Extract activity from mask or ROI atlas using NiftiLabelsMasker.

        Args:
            mask: BrainData, nibabel image, or file path. Can be:

                  - Binary mask (extracts from single ROI)
                  - Labeled atlas (extracts from multiple ROIs)
            metric: Extraction method ('mean', 'median', 'pca'). Default: 'mean'
            n_components: If metric='pca', number of components to return

        Returns:
            For binary mask: scalar or 1D array.
            For labeled atlas: 1D or 2D array, or PCA components.

        Examples:
            >>> roi_values = brain.extract_roi(binary_mask)
            >>> atlas_values = brain.extract_roi(atlas_mask)
            >>> components = brain.extract_roi(mask, metric='pca', n_components=5)
        """
        from .analysis import extract_roi

        return extract_roi(self, mask, metric=metric, n_components=n_components)

    def icc(
        self,
        n_subjects,
        n_sessions,
        icc_type="icc2",
        parallel=None,
        n_jobs=-1,
        max_gpu_memory_gb=4.0,
    ):
        """Calculate voxel-wise intraclass correlation coefficient.

        ICC Formulas based on Shrout & Fleiss (1979).

        Args:
            n_subjects: Number of subjects in the data
            n_sessions: Number of sessions per subject
            icc_type: Type of ICC ('icc1', 'icc2', 'icc3'). Default: 'icc2'
            parallel: Parallelization method (None, 'cpu', 'gpu')
            n_jobs: Number of CPU cores (-1 = all cores)
            max_gpu_memory_gb: GPU memory budget in GB

        Returns:
            BrainData: BrainData instance with ICC map (shape: (1, n_voxels))

        Examples:
            >>> icc_map = data.icc(n_subjects=20, n_sessions=3, icc_type='icc2')
        """
        from .analysis import icc

        return icc(
            self,
            n_subjects,
            n_sessions,
            icc_type=icc_type,
            parallel=parallel,
            n_jobs=n_jobs,
            max_gpu_memory_gb=max_gpu_memory_gb,
        )

    def detrend(self, method="linear"):
        """Remove linear trend from each voxel.

        Args:
            method: ('linear','constant', optional) type of detrending

        Returns:
            out: (BrainData) detrended BrainData instance
        """
        from .analysis import detrend_data

        return detrend_data(self, method=method)

    def r_to_z(self):
        """Apply Fisher's r to z transformation to each element of the data
        object."""
        from .analysis import r_to_z

        return r_to_z(self)

    def z_to_r(self):
        """Convert z score back into r value for each element of data object."""
        from .analysis import z_to_r

        return z_to_r(self)

    def filter(self, sampling_freq=None, high_pass=None, low_pass=None, **kwargs):
        """Apply butterworth filter to data. Wraps nilearn.signal.clean.

        Note:
            Unlike nilearn's default, does not detrend or standardize. Pass
            detrend=True or standardize=True via kwargs to enable.

        Args:
            sampling_freq: Sampling freq in hertz (i.e. 1 / TR)
            high_pass: High pass cutoff frequency
            low_pass: Low pass cutoff frequency
            **kwargs: Additional arguments passed to nilearn.signal.clean

        Returns:
            BrainData: Filtered BrainData instance
        """
        from .analysis import filter_data

        return filter_data(
            self,
            sampling_freq=sampling_freq,
            high_pass=high_pass,
            low_pass=low_pass,
            **kwargs,
        )

    def standardize(self, axis=0, method="center"):
        """Standardize BrainData() instance.

        Args:
            axis (int): Axis along which to standardize. 0 standardizes each
                voxel across observations (default). 1 standardizes each
                observation across voxels.
            method (str): Standardization method. 'center' subtracts the mean
                (default). 'zscore' subtracts the mean and divides by standard
                deviation.

        Returns:
            BrainData: Standardized BrainData instance.
        """
        from .analysis import standardize

        return standardize(self, axis=axis, method=method)

    def threshold(
        self,
        upper=None,
        lower=None,
        binarize=False,
        coerce_nan=True,
        cluster_threshold=0,
    ):
        """Threshold BrainData instance with optional cluster filtering.

        Args:
            upper: (float or str) Upper cutoff for thresholding.
            lower: (float or str) Lower cutoff for thresholding.
            binarize (bool): return binarized image. Default False.
            coerce_nan (bool): coerce nan values to 0s. Default True.
            cluster_threshold (int): Minimum cluster size in voxels. Default 0.

        Returns:
            Thresholded BrainData object.
        """
        from .analysis import threshold_data

        return threshold_data(
            self,
            upper=upper,
            lower=lower,
            binarize=binarize,
            coerce_nan=coerce_nan,
            cluster_threshold=cluster_threshold,
        )

    def regions(
        self,
        min_region_size=1350,
        extract_type="local_regions",
        smoothing_fwhm=6,
        is_mask=False,
    ):
        """Extract brain connected regions into separate regions.

        Args:
            min_region_size (int): Minimum volume in mm3 for a region to be kept.
            extract_type (str): Type of extraction method
                                ['connected_components', 'local_regions'].
            smoothing_fwhm (scalar): Smooth an image to extract more sparser regions.
            is_mask (bool): Whether to treat as boolean mask.

        Returns:
            BrainData: BrainData instance with extracted ROIs as data.
        """
        from .analysis import regions

        return regions(
            self,
            min_region_size=min_region_size,
            extract_type=extract_type,
            smoothing_fwhm=smoothing_fwhm,
            is_mask=is_mask,
        )

    def transform_pairwise(self):
        """Transform data into pairwise comparisons.

        Returns:
            BrainData: BrainData instance transformed into pairwise comparisons
        """
        from .analysis import transform_pairwise_data

        return transform_pairwise_data(self)

    # =========================================================================
    # Bootstrap methods (delegated to _bootstrap.py)
    # =========================================================================

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
            save_boots: (bool) If True, store all bootstrap samples. Default: False
            n_jobs: (int) Number of CPU cores for parallelization. -1 means all CPUs.
            random_state: (int, optional) Random seed for reproducibility
            percentiles: (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5)
            X_test: (np.ndarray, optional) Test features for 'predict' bootstrap.
            **kwargs: Additional parameters (backend, max_gpu_memory_gb, etc.)

        Returns:
            BrainData or dict:
                - For simple stats: Returns BrainData with bootstrap mean
                - For model stats: Returns dict with keys: 'mean', 'std', 'Z', 'p',
                  'ci_lower', 'ci_upper' (all BrainData objects)
                - If ``save_boots=True``: Returns dict with 'samples' key containing all samples

        Examples:
            >>> boot = brain.bootstrap(stat='mean', n_samples=1000)
            >>> brain.fit(X=dm, model='ridge', alpha=1.0)
            >>> boot = brain.bootstrap(stat='weights', n_samples=1000)
        """
        from .bootstrap import bootstrap

        return bootstrap(
            self,
            stat,
            n_samples=n_samples,
            save_boots=save_boots,
            n_jobs=n_jobs,
            random_state=random_state,
            percentiles=percentiles,
            X_test=X_test,
            **kwargs,
        )

    def _convert_bootstrap_results_to_brain_data(
        self, result, save_boots=False, return_dict=False
    ):
        """Convert bootstrap results dictionary to BrainData format.

        Args:
            result: (dict) Result dictionary from bootstrap function.
            save_boots: (bool) If True, include 'samples' key in output.
            return_dict: (bool) If True, always return dict.

        Returns:
            BrainData or dict
        """
        from .bootstrap import convert_bootstrap_results_to_brain_data

        return convert_bootstrap_results_to_brain_data(
            self, result, save_boots=save_boots, return_dict=return_dict
        )

    # =========================================================================
    # Decomposition and alignment methods (delegated to _analysis.py)
    # =========================================================================

    def decompose(
        self, algorithm="pca", axis="voxels", n_components=None, *args, **kwargs
    ):
        """Decompose BrainData object.

        Args:
            algorithm: (str) Algorithm to perform decomposition
                        types=['pca','ica','nnmf','fa','dictionary','kernelpca']
            axis: dimension to decompose ['voxels','images']
            n_components: (int) number of components. If None then retain
                        as many as possible.

        Returns:
            output: a dictionary of decomposition parameters
        """
        from .analysis import decompose

        return decompose(
            self,
            algorithm=algorithm,
            axis=axis,
            n_components=n_components,
            *args,
            **kwargs,
        )

    def align(self, target, method="procrustes", axis=0, *args, **kwargs):
        """Align BrainData instance to target object using functional alignment.

        Args:
            target: (BrainData) object to align to.
            method: (str) alignment method to use
                ['probabilistic_srm','deterministic_srm','procrustes']
            axis: (int) axis to align on

        Returns:
            out: (dict) a dictionary containing transformed object,
                transformation matrix, and the shared response matrix

        Examples:
            >>> out = data.align(target, method='procrustes')
            >>> out = data.align(target, method='probabilistic_srm', n_features=None)
        """
        from .analysis import align

        return align(self, target, method=method, axis=axis, *args, **kwargs)

    def smooth(self, fwhm):
        """Apply spatial smoothing using nilearn smooth_img().

        Args:
            fwhm: (float) full width half maximum of gaussian spatial filter

        Returns:
            BrainData instance (copy with smoothed data)
        """
        from .analysis import smooth

        return smooth(self, fwhm)

    def find_spikes(self, global_spike_cutoff=3, diff_spike_cutoff=3):
        """Identify spikes from Time Series Data.

        Args:
            global_spike_cutoff (int or None): cutoff to identify spikes in global signal
                in standard deviations, or None to skip.
            diff_spike_cutoff (int or None): cutoff to identify spikes in average frame
                difference in standard deviations, or None to skip.

        Returns:
            pandas dataframe with spikes as indicator variables
        """
        from .analysis import find_spikes_data

        return find_spikes_data(
            self,
            global_spike_cutoff=global_spike_cutoff,
            diff_spike_cutoff=diff_spike_cutoff,
        )

    def temporal_resample(self, sampling_freq=None, target=None, target_type="hz"):
        """Resample BrainData timeseries to a new target frequency or number of samples.

        Args:
            sampling_freq: (float) sampling frequency of data in hertz
            target: (float) upsampling target
            target_type: (str) type of target can be [samples,seconds,hz]

        Returns:
            upsampled BrainData instance
        """
        from .analysis import temporal_resample

        return temporal_resample(
            self, sampling_freq=sampling_freq, target=target, target_type=target_type
        )

    # =========================================================================
    # Prediction methods (delegated to _prediction.py)
    # =========================================================================

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

        Two modes:
        1. **Timeseries prediction** (X provided): Use fitted ridge model to predict voxel responses.
        2. **MVPA decoding** (y provided): Train a classifier to predict labels from brain patterns.

        Args:
            X: Features for timeseries prediction, shape (n_samples, n_features).
            y: Labels for MVPA decoding, shape (n_samples,).
            method: Decoding method - 'whole_brain', 'searchlight', or 'roi'.
            estimator: Classifier to use ('svm', 'logistic', 'ridge', 'lda', or sklearn estimator).
            cv: Cross-validation specification.
            groups: Group labels for CV.
            roi_mask: Atlas/parcellation for ROI-based decoding.
            radius: Searchlight radius in mm (default 10.0).
            scoring: Metric for evaluation.
            standardize: Z-score features before classification (default True).
            n_jobs: Number of parallel jobs (-1 = all cores).
            show_progress: Show progress bar for searchlight.

        Returns:
            BrainData: Predicted timeseries or accuracy map.

        Examples:
            >>> brain_data.fit(model='ridge', X=features)
            >>> predictions = brain_data.predict(X=new_features)
            >>> accuracy = brain_data.predict(y=labels, method='searchlight')
        """
        from .prediction import predict

        return predict(
            self,
            X=X,
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

    def _predict_timeseries(self, X=None):
        """Generate timeseries predictions using fitted model.

        Args:
            X: Features to predict on. If None, uses training data.

        Returns:
            BrainData with predicted timeseries.
        """
        from .prediction import predict_timeseries

        return predict_timeseries(self, X=X)

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
        from .prediction import predict_mvpa

        return predict_mvpa(
            self,
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

    def _resolve_estimator(self, estimator):
        """Resolve string shortcut to sklearn estimator."""
        from .prediction import resolve_estimator

        return resolve_estimator(self, estimator)

    def _mvpa_whole_brain(self, X, y, pipe, cv, groups, scoring):
        """Whole-brain MVPA - single accuracy across all voxels."""
        from .prediction import mvpa_whole_brain

        return mvpa_whole_brain(self, X, y, pipe, cv, groups, scoring)

    def _mvpa_whole_brain_pipeline(self, y, estimator, cv, groups, standardize):
        """Whole-brain MVPA using Pipeline infrastructure."""
        from .prediction import mvpa_whole_brain_pipeline

        return mvpa_whole_brain_pipeline(self, y, estimator, cv, groups, standardize)

    def _mvpa_searchlight(
        self, X, y, pipe, cv, groups, scoring, radius, n_jobs, show_progress
    ):
        """Searchlight MVPA - accuracy per voxel neighborhood."""
        from .prediction import mvpa_searchlight

        return mvpa_searchlight(
            self, X, y, pipe, cv, groups, scoring, radius, n_jobs, show_progress
        )

    def _mvpa_roi(
        self, X, y, pipe, cv, groups, scoring, roi_mask, n_jobs, show_progress
    ):
        """ROI-based MVPA - accuracy per ROI."""
        from .prediction import mvpa_roi

        return mvpa_roi(
            self, X, y, pipe, cv, groups, scoring, roi_mask, n_jobs, show_progress
        )

    # =========================================================================
    # Plotting methods (delegated to _plotting.py)
    # =========================================================================

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
            thr_upper (str/float, optional): Upper threshold.
            thr_lower (str/float, optional): Lower threshold.
            threshold (float, optional): Convenience parameter. If positive, sets thr_upper.
                If negative, sets thr_lower.
            cut_coords (list, optional): Cut coordinates for multi-slice views.
            cmap (str, optional): Colormap name. If None, auto-selects based on data.
            bg_img (str/nibabel image, optional): Background image.
            ax (matplotlib.axes.Axes, optional): Matplotlib axis.
            title (str, optional): Plot title.
            colorbar (bool): Whether to show colorbar. Default: True.
            save (str, optional): Path to save figure(s).
            stat (str): Statistic for timeseries plots. Default: 'mean'.
            **kwargs: Additional arguments passed to nilearn plot functions.

        Returns:
            Display or matplotlib Figure.
        """
        from .plotting import plot_brain

        return plot_brain(
            self,
            kind=kind,
            thr_upper=thr_upper,
            thr_lower=thr_lower,
            threshold=threshold,
            cut_coords=cut_coords,
            cmap=cmap,
            bg_img=bg_img,
            ax=ax,
            title=title,
            colorbar=colorbar,
            save=save,
            stat=stat,
            **kwargs,
        )

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

        Args:
            threshold (float, optional): Values below this absolute threshold
                are masked.
            cmap (str): Matplotlib colormap for data. Default: 'RdBu_r'.
            vmax (float, optional): Maximum value for colormap.
            vmin (float, optional): Minimum value for colormap.
            template (str): Freesurfer surface resolution. Default: 'fsaverage5'.
            with_curvature (bool): Show sulcal/gyral pattern. Default: True.
            curvature_contrast (float): Contrast of curvature overlay.
                Default: 0.5.
            curvature_brightness (float): Mean brightness of curvature overlay.
                Default: 0.5.
            colorbar (bool): Show colorbar. Default: True.
            colorbar_orientation (str): 'horizontal' or 'vertical'.
                Default: 'horizontal'.
            figsize (tuple): Figure size as (width, height). Default: (12, 6).
            title (str, optional): Figure title.
            radius (float): Sampling radius in mm for vol_to_surf projection.
                Default: 3.0.
            interpolation (str): Interpolation method for vol_to_surf.
                Default: 'linear'.
            axes (matplotlib.axes.Axes, optional): Existing axes to plot on.
            save (str, optional): File path to save figure.

        Returns:
            matplotlib.figure.Figure
        """
        from .plotting import plot_flatmap_brain

        return plot_flatmap_brain(
            self,
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

    def _plot_matplotlib(self, kind, stat="mean", ax=None, title=None, save=None):
        """Plot using matplotlib (timeseries or histogram)."""
        from .plotting import plot_matplotlib

        return plot_matplotlib(
            self, kind=kind, stat=stat, ax=ax, title=title, save=save
        )

    def _auto_select_colormap(self, data):
        """Auto-select colormap based on data characteristics."""
        from .plotting import auto_select_colormap

        return auto_select_colormap(data)

    def _prepare_save_paths(self, save):
        """Prepare save paths for multiple plot outputs."""
        from .plotting import prepare_save_paths

        return prepare_save_paths(save)

    # =========================================================================
    # Deprecated stubs
    # =========================================================================

    # NOTE: Historical follow-up for future Model-class refactoring
    def randomise(self, *args, **kwargs):
        """DEPRECATED: This method has been moved to the Model class.

        .. deprecated:: 0.6.0
            Use the Model class for permutation-based inference.
        """
        raise NotImplementedError(
            "randomise() has been deprecated. Please use the new Model class for permutation-based inference."
        )

    # NOTE: Historical follow-up for future Model-class refactoring
    def ttest(self, *args, **kwargs):
        """DEPRECATED: This method has been moved to the Model class.

        .. deprecated:: 0.6.0
            Use the Model class for statistical testing.
        """
        raise NotImplementedError(
            "ttest() has been deprecated. Please use the new Model class for statistical testing."
        )
