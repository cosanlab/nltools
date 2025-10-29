"""
NeuroLearn Brain Data
=====================

Classes to represent brain image data.

"""

from nilearn.signal import clean
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist
from scipy.stats import t as t_dist
from scipy.signal import detrend
from scipy.interpolate import pchip
import os
import shutil
import nibabel as nib
import numpy as np
import pandas as pd
import tempfile
import warnings  # noqa: F401
from copy import deepcopy
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import check_random_state
from sklearn.preprocessing import scale
from pynv import Client
from joblib import Parallel, delayed
from nltools.mask import expand_mask
from nilearn.maskers import NiftiMasker
from nilearn.image import smooth_img, resample_to_img
from nilearn.masking import intersect_masks
from nilearn.regions import connected_regions, connected_label_regions
from nltools.utils import (
    attempt_to_import,
    concatenate,
    _bootstrap_apply_func,
    set_decomposition_algorithm,
    check_brain_data,
    check_brain_data_is_single,
    to_h5,
)
from nltools.stats import (
    fisher_r_to_z,
    fisher_z_to_r,
    transform_pairwise,
    summarize_bootstrap,
    procrustes,
    find_spikes,
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


class Brain_Data(object):
    """
    Brain_Data is a class to represent neuroimaging data in python as a vector
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
        """Initialize Brain_Data object.

        Args:
            data: Neuroimaging data. Can be:
                - None (empty Brain_Data)
                - Brain_Data object
                - List of Brain_Data objects or file paths
                - File path (str/Path) to .nii/.nii.gz/.h5/.hdf5
                - nibabel Nifti1Image object
                - URL to download data from
            mask: Brain mask as nibabel object, file path, or None (uses MNI template).
            masker: nilearn masker object (e.g. ROI or searchlight extractor); Default will load data as voxels
            **kwargs: Additional arguments passed to NiftiMasker.
        """
        # Import validation functions
        from ._validation import validate_data_type, validate_frame

        # Initialize attributes
        self._h5_compression = kwargs.pop("h5_compression", "gzip")
        self.verbose = kwargs.pop("verbose", False)
        self.design_matrix = None
        self.masker = masker
        self._labels = None

        # Initialize mask
        self._initialize_mask(mask, **kwargs)

        # Initialize data based on type
        data_type = validate_data_type(data)

        if data_type == "none":
            self.data = np.array([])
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

        # Collapse any extra data dimension
        if self.data is not None and 1 in self.data.shape:
            self.data = self.data.squeeze()

    # TODO: update to respect new masker kwarg Check if complete and delete or update comment accordingly
    def _initialize_mask(self, mask, **kwargs):
        """Initialize the mask and NiftiMasker.

        Args:
            mask: Brain mask as nibabel object, file path, or None.
            **kwargs: Additional arguments passed to NiftiMasker.
        """
        if mask is None:
            self.mask = nib.load(MNI_Template.mask)
        elif isinstance(mask, (str, Path)):
            self.mask = nib.load(str(mask))
        elif isinstance(mask, nib.Nifti1Image):
            self.mask = mask
        else:
            raise TypeError(
                f"mask must be a nibabel instance or a valid file name. "
                f"Received {type(mask).__name__}"
            )

        # Learn 3d/4d -> 1d/2d transform on template/mask
        self.nifti_masker = NiftiMasker(
            mask_img=self.mask, verbose=kwargs.get("verbose", 0), **kwargs
        )
        self.nifti_masker.fit()

    def _load_from_list(self, data_list):
        """Load data from a list of Brain_Data objects or file paths.

        Args:
            data_list: List of Brain_Data objects or file paths.
        """
        from ._validation import validate_list_data

        list_type = validate_list_data(data_list)

        if list_type == "brain_data":
            # Concatenate Brain_Data objects
            tmp = concatenate(data_list)
            for item in ["data", "mask", "nifti_masker"]:
                setattr(self, item, getattr(tmp, item))
        else:
            # Load files
            self.data = []
            if not self.verbose:
                with open(os.devnull, "w") as devnull:
                    with redirect_stdout(devnull):
                        for item in data_list:
                            self.data.append(self.nifti_masker.transform(item))
            else:
                for item in data_list:
                    self.data.append(self.nifti_masker.transform(item))
            self.data = np.concatenate(self.data)

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
        # Transform data using masker
        if not self.verbose:
            with open(os.devnull, "w") as devnull:
                with redirect_stdout(devnull):
                    self.data = self.nifti_masker.transform(data)
        else:
            self.data = self.nifti_masker.transform(data)

    def _perform_arithmetic(self, other, operation, operation_name, inplace=False):
        """Perform arithmetic operation with validation.

        Args:
            other: The other operand.
            operation: The operation function (e.g., np.add, np.subtract).
            operation_name: Name of the operation for error messages.

        Returns:
            Brain_Data: Result of the operation.
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
        Apply a function to the `.data` attribute. If axis=0, returns a `Brain_Data` object with the statistic calculated over samples (e.g. within a voxel over time). If axis=1, returns a numpy array with the statistic calculated over features (e.g. across voxels within a specific time-point)

        Args:
            stat_func: Statistical function to apply (e.g., np.mean, np.std).
            axis: Axis along which to compute (0=across images, 1=within images).

        Returns:
            float/np.array/Brain_Data: Result of statistical operation.
        """

        # Single image case
        if check_brain_data_is_single(self):
            return stat_func(self.data)

        if axis == 1:
            # Return array with statistic within each image
            return stat_func(self.data, axis=1)
        elif axis == 0:
            # Return Brain_Data with statistic across images
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
        return "%s.%s(data=%s, Y=%s, X=%s, mask=%s)" % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.shape,
            self.Y.shape,
            self.X.shape,
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
        if not isinstance(value, Brain_Data):
            raise ValueError(
                "Make sure the value you are trying to set is a Brain_Data() instance."
            )
        self.data[index, :] = value.data
        if not value.Y.empty:
            self.Y.values[index] = value.Y
        if not value.X.empty:
            if self.X.shape[1] != value.X.shape[1]:
                raise ValueError("Make sure self.X is the same size as value.X.")
            self.X.values[index] = value.X

    def __len__(self):
        return self.shape[0]

    def __add__(self, y):
        """Add to Brain_Data."""
        return self._perform_arithmetic(y, np.add, "add")

    def __radd__(self, y):
        """Right add to Brain_Data."""
        return self._perform_arithmetic(y, np.add, "add")

    def __sub__(self, y):
        """Subtract from Brain_Data."""
        return self._perform_arithmetic(y, np.subtract, "subtract")

    def __rsub__(self, y):
        """Right subtract from Brain_Data."""
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
        """Multiply Brain_Data."""
        return self._perform_arithmetic(y, np.multiply, "multiply")

    def __rmul__(self, y):
        """Right multiply Brain_Data."""
        return self._perform_arithmetic(y, np.multiply, "multiply")

    def __truediv__(self, y):
        """Divide Brain_Data."""
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
        """Check equality between Brain_Data."""
        if not isinstance(other, Brain_Data):
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

    @property
    def shape(self):
        """Get images by voxels shape."""

        return self.data.shape

    def mean(self, axis=0):
        """Get mean of each voxel or image.

        Args:
            axis: Axis along which to compute mean.
                0 = across images (default), returns Brain_Data
                1 = within images, returns array

        Returns:
            float/np.array/Brain_Data: Mean values.
        """
        return self._apply_func(np.mean, axis)

    def median(self, axis=0):
        """Get median of each voxel or image.

        Args:
            axis: Axis along which to compute median.
                0 = across images (default), returns Brain_Data
                1 = within images, returns array

        Returns:
            float/np.array/Brain_Data: Median values.
        """
        return self._apply_func(np.median, axis)

    def std(self, axis=0):
        """Get standard deviation of each voxel or image.

        Args:
            axis: Axis along which to compute standard deviation.
                0 = across images (default), returns Brain_Data
                1 = within images, returns array

        Returns:
            float/np.array/Brain_Data: Standard deviation values.
        """
        return self._apply_func(np.std, axis)

    def sum(self, axis=0):
        """Get sum of each voxel or image.

        Args:
            axis: Axis along which to compute sum.
                0 = across images (default), returns Brain_Data
                1 = within images, returns array

        Returns:
            float/np.array/Brain_Data: Sum values.
        """
        return self._apply_func(np.sum, axis)

    def to_nifti(self):
        """Convert Brain_Data Instance into Nifti Object"""

        return self.nifti_masker.inverse_transform(self.data)

    def write(self, file_name):
        """Write out Brain_Data object to Nifti or HDF5 File.

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

    def scale(self, scale_val=100.0):
        """
        Scale all values such that they are on the range [0, scale_val], via grand-mean scaling. This is NOT global-scaling/intensity normalization. It rescales each voxel to be a proportion of the global average * `scale_val`. This is useful for ensuring that data is on a common scale (e.g. good for multiple runs, participants, etc) and if the default value of 100 is used, can be interpreted as something akin to (but not exactly) "percent signal change." This is consistent with default behavior in AFNI and SPM.Change this value to 10000 to make consistent with FSL.

        Args:
            scale_val: (int/float) what value to send the grand-mean to;
                        default 100

        """

        out = self._shallow_copy_with_data()
        # Copy data array and modify in-place
        out.data = self.data.copy()
        out.data = out.data / out.data.mean() * scale_val

        return out

    # TODO: update Check if complete and delete or update comment accordingly
    def regress(self, design_matrix=None, noise_model="ols", mode=None, **kwargs):
        """Runs a mass-univariate GLM analysis using the provided Design_Matrix.

        This is a wrapper around nltools.models.Glm, which wraps nilearn.glm.first_level.FirstLevelModel.
        Results are stored as attributes on the Brain_Data object rather than returned.

        We override some nilearn defaults:
        - no smoothing (use `.smooth()` beforehand if needed)
        - no scaling (use `.scale()` beforehand if needed)
        - no drift model (should already be in the Design_Matrix)
        - OLS noise model by default (use noise_model='ar1' for autocorrelation)

        Args:
            design_matrix: Design_Matrix object or pandas DataFrame with regressors
                          If None, will use self.X (deprecated, for backward compatibility)
            noise_model (str): temporal variance model ('ols' or 'ar1'). Default: 'ols'
            mode (str): deprecated parameter for backward compatibility ('robust' is ignored)
            **kwargs: additional arguments for nltools.models.Glm

        Returns:
            dict: For backward compatibility, returns dict with 'beta', 't', 'p', 'residual' keys
                  (deprecated - in future versions will return None)

        Sets attributes:
            self.design_matrix: The design matrix used
            self.glm_model: The fitted Glm model instance
            self.glm_betas: Beta coefficients (Brain_Data)
            self.glm_t: T-statistics (Brain_Data)
            self.glm_p: P-values (Brain_Data)
            self.glm_se: Standard errors (Brain_Data)
            self.glm_residual: Residuals (Brain_Data)
            self.glm_predicted: Predicted values (Brain_Data)
            self.glm_r2: R-squared values (Brain_Data)
        """
        from .design_matrix import Design_Matrix
        from ..models import Glm
        import warnings

        # Handle backward compatibility - use self.X if no design_matrix provided
        if design_matrix is None:
            if hasattr(self, "X") and self.X is not None:
                warnings.warn(
                    "Using self.X as design matrix is deprecated. "
                    "Pass design_matrix explicitly to regress().",
                    DeprecationWarning,
                    stacklevel=2,
                )
                design_matrix = self.X
            else:
                raise TypeError("design_matrix must be provided or set as self.X")

        # Handle deprecated 'mode' parameter
        if mode == "robust":
            warnings.warn(
                "mode='robust' is deprecated and ignored. "
                "Robust regression will be implemented in future Model class.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Validate inputs
        if not isinstance(design_matrix, (Design_Matrix, pd.DataFrame)):
            raise TypeError("design_matrix must be a Design_Matrix or DataFrame")

        # Store design matrix
        self.design_matrix = design_matrix

        # Convert to DataFrame if needed
        if isinstance(design_matrix, Design_Matrix):
            dm_df = design_matrix
        else:
            dm_df = pd.DataFrame(design_matrix)

        # Set up Glm model with same defaults as before
        # Note: Glm handles smoothing_fwhm, standardize, drift_model, mask internally
        glm_params = {
            "smoothing_fwhm": kwargs.pop("smoothing_fwhm", None),  # No smoothing by default
            "noise_model": noise_model,
            "mask": self.mask,  # Use our mask
            "t_r": kwargs.pop("t_r", None),  # Will be set from data if needed
        }

        # Pass any remaining kwargs to Glm (they'll be passed to FirstLevelModel)
        glm_params.update(kwargs)

        # Create Glm model instance
        self.glm_model = Glm(**glm_params)

        # Convert data to 4D nifti for nilearn
        data_4d = self.to_nifti()

        # Fit the model
        self.glm_model.fit(data_4d, design_matrices=dm_df)

        # Extract results for each regressor
        n_regressors = dm_df.shape[1]

        # Initialize storage for results
        beta_maps = []
        t_maps = []
        p_maps = []
        se_maps = []

        # Get results for each regressor
        for i, col in enumerate(dm_df.columns):
            # Create contrast vector (1 for this regressor, 0 for others)
            contrast = np.zeros(n_regressors)
            contrast[i] = 1

            # Compute contrast using Glm's compute_contrast method
            results = self.glm_model.compute_contrast(contrast, output_type="all")

            # Store maps
            beta_maps.append(results["effect_size"])
            t_maps.append(results["stat"])
            p_maps.append(results["p_value"])
            se_maps.append(results["effect_variance"])

        # Convert results to Brain_Data objects and store as attributes
        self.glm_betas = Brain_Data(beta_maps, mask=self.mask)
        self.glm_t = Brain_Data(t_maps, mask=self.mask)
        self.glm_p = Brain_Data(p_maps, mask=self.mask)

        # Convert effect variance to standard error: SE(β) = √Var(β)
        # nilearn provides Var(beta) = sigma^2 * (X'X)^-1
        # We need SE(beta) = sqrt(Var(beta)) for interpretability
        # np.abs() handles numerical precision issues (variance should be >= 0)
        se_data = []
        for se_img in se_maps:
            se_brain = Brain_Data(se_img, mask=self.mask)
            se_brain.data = np.sqrt(np.abs(se_brain.data))
            se_data.append(se_brain)
        self.glm_se = Brain_Data(data=se_data, mask=self.mask)

        # Get residuals using Glm's residuals property
        self.glm_residual = Brain_Data(self.glm_model.residuals, mask=self.mask)

        # Predicted = original - residuals
        self.glm_predicted = self.copy()
        self.glm_predicted.data = self.data - self.glm_residual.data

        # R-squared calculation (nilearn doesn't provide this for whole-brain maps)
        # Standard formula: R² = 1 - (SS_residual / SS_total)
        # Current numpy implementation is optimal for voxel-wise operations
        # nilearn.glm focuses on inference (t-stats, contrasts) not model quality metrics
        ss_total = np.sum((self.data - self.data.mean(axis=0)) ** 2, axis=0)
        ss_residual = np.sum(self.glm_residual.data**2, axis=0)
        r2_values = 1 - (
            ss_residual / (ss_total + 1e-10)
        )  # Add small value to avoid division by zero

        # Create single-image Brain_Data for R-squared
        self.glm_r2 = self[0].copy()
        self.glm_r2.data = r2_values.reshape(1, -1)

        # Return dictionary for backward compatibility (deprecated)
        warnings.warn(
            "Returning a dictionary from regress() is deprecated. "
            "In future versions, results will only be stored as attributes.",
            DeprecationWarning,
            stacklevel=2,
        )

        return {
            "beta": self.glm_betas,
            "t": self.glm_t,
            "p": self.glm_p,
            "residual": self.glm_residual,
        }

    def compute_contrasts(self, contrasts, contrast_type="t"):
        """Compute contrasts from fitted GLM results.

        This method computes contrasts as linear combinations of the GLM beta coefficients.
        Must be called after .regress() has been run.

        Args:
            contrasts: Can be:
                - str: A string specifying the contrast using column names
                  e.g., "conditionA - conditionB" or "2*conditionA - conditionB - conditionC"
                - dict: Dictionary with contrast names as keys and contrast strings/vectors as values
                  e.g., {"main_effect": "conditionA - conditionB", "interaction": [1, -1, -1, 1]}
                - array: Numeric contrast vector matching the number of regressors
                  e.g., [1, -1, 0, 0] for a 4-regressor model
            contrast_type (str): Type of contrast statistic ('t' or 'F'). Default: 't'

        Returns:
            Brain_Data or dict: If single contrast, returns Brain_Data object with contrast map.
                               If multiple contrasts (dict input), returns dict of Brain_Data objects.

        Examples:
            >>> # After running regression
            >>> brain.regress(design_matrix)
            >>> # Simple contrast
            >>> contrast1 = brain.compute_contrasts("conditionA - conditionB")
            >>> # Multiple contrasts
            >>> contrasts = {
            ...     "A_vs_B": "conditionA - conditionB",
            ...     "main_effect": [1, -1, 0, 0]
            ... }
            >>> results = brain.compute_contrasts(contrasts)
        """
        # Check that regression has been run
        if not hasattr(self, "glm_betas"):
            raise RuntimeError("Must run .regress() before computing contrasts")

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

            # Create Brain_Data object for contrast
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
        """
        if not hasattr(self, "design_matrix"):
            raise RuntimeError("No design matrix found. Run .regress() first.")

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

    def append(self, data, **kwargs):
        """Append data to Brain_Data instance.

        Args:
            data: Brain_Data instance to append.
            kwargs: Optional arguments passed to pandas concat.

        Returns:
            Brain_Data: New appended Brain_Data instance.
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

        return out

    def empty(self):
        """Create a copy of Brain_Data with empty data array"""
        from copy import deepcopy

        out = deepcopy(self)
        out.data = np.array([])
        return out

    @property
    def isempty(self):
        """Check if Brain_Data.data is empty"""

        if isinstance(self.data, np.ndarray):
            boolean = False if self.data.size else True
        if isinstance(self.data, list):
            boolean = True if not self.data else False
        return boolean

    def similarity(self, image, method="correlation"):
        """Calculate similarity of Brain_Data() instance with single
        Brain_Data or Nibabel image

        Args:
            image: (Brain_Data, nifti)  image to evaluate similarity
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

        # Check to make sure masks are the same for each dataset and if not
        # create a union mask
        # This might be handy code for a new Brain_Data method
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

        if method == "dot_product":
            func = lambda x, y: np.dot(x, y)
        elif method in ["pearson", "correlation"]:
            func = lambda x, y: pearsonr(x, y)[0]
        elif method in ["spearman", "rank_correlation"]:
            func = lambda x, y: spearmanr(x, y)[0]
        elif method == "cosine":
            func = method

        out = cdist(np.atleast_2d(data2), np.atleast_2d(image2), func).squeeze()
        # cdist metric argument returns distances by default (unless we specific a
        # custom function like above) so flip it to similarity
        out = 1 - out if method == "cosine" else out
        return out

    def distance(self, metric="euclidean", **kwargs):
        """Calculate distance between images within a Brain_Data() instance.

        Args:
            metric: (str) type of distance metric (can use any scikit learn or
                    sciypy metric)

        Returns:
            dist: (Adjacency) Outputs a 2D distance matrix.

        """

        return Adjacency(
            pairwise_distances(self.data, metric=metric, **kwargs),
            matrix_type="Distance",
        )

    def multivariate_similarity(self, images, method="ols"):
        """Predict spatial distribution of Brain_Data() instance from linear
        combination of other Brain_Data() instances or Nibabel images

        Args:
            self: Brain_Data instance of data to be applied
            images: Brain_Data instance of weight map

        Returns:
            out: dictionary of regression statistics in Brain_Data
                instances {'beta','t','p','df','residual'}

        """
        # Notes:  Should add ridge, and lasso, elastic net options options

        if len(self.shape) > 1:
            raise ValueError("This method can only decompose a single brain image.")

        images = check_brain_data(images)

        # Check to make sure masks are the same for each dataset and if not create a union mask
        # TODO: This might be handy code for a new Brain_Data method
        if np.sum(self.nifti_masker.mask_img.get_fdata() == 1) != np.sum(
            images.nifti_masker.mask_img.get_fdata() == 1
        ):
            new_mask = intersect_masks(
                [self.nifti_masker.mask_img, images.nifti_masker.mask_img],
                threshold=1,
                connected=False,
            )
            new_nifti_masker = NiftiMasker(mask_img=new_mask)
            data2 = new_nifti_masker.fit_transform(self.to_nifti())
            image2 = new_nifti_masker.fit_transform(images.to_nifti())
        else:
            data2 = self.data
            image2 = images.data

        # Add intercept and transpose
        image2 = np.vstack((np.ones(image2.shape[1]), image2)).T

        # Calculate pattern expression
        if method == "ols":
            b = np.dot(np.linalg.pinv(image2), data2)
            res = data2 - np.dot(image2, b)
            sigma = np.std(res, axis=0)
            stderr = np.dot(
                np.matrix(
                    np.diagonal(np.linalg.inv(np.dot(image2.T, image2))) ** 0.5
                ).T,
                np.matrix(sigma),
            )
            t_out = b / stderr
            df = image2.shape[0] - image2.shape[1]
            p = 2 * (1 - t_dist.cdf(np.abs(t_out), df))
        else:
            raise NotImplementedError

        return {
            "beta": b,
            "t": t_out,
            "p": p,
            "df": df,
            "sigma": sigma,
            "residual": res,
        }

    def apply_mask(self, mask, resample_mask_to_brain=False):
        """Mask Brain_Data instance using nilearn functionality.

        Note target data will be resampled into the same space as the mask. If you would like the mask
        resampled into the Brain_Data space, then set resample_mask_to_brain=True.

        Args:
            mask: (Brain_Data or nifti object) mask to apply to Brain_Data object.
            resample_mask_to_brain: (bool) Will resample mask to brain space before applying mask (default=False).

        Returns:
            masked: (Brain_Data) masked Brain_Data object

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
                force_resample=True,
                copy_header=True,
            )

        # Use nilearn's apply_mask for efficient masking (C-optimized, single path, memory efficient)
        masked.data = apply_mask(masked.to_nifti(), mask_img)

        # Create masker for the masked space
        masked.nifti_masker = NiftiMasker(mask_img=mask_img).fit()

        # Preserve 1D output for single images (backward compatibility)
        if (len(masked.shape) > 1) & (masked.shape[0] == 1):
            masked.data = masked.data.flatten()

        return masked

    # TODO: replace with nilearn or speed-up? Check if complete and delete or update comment accordingly
    def extract_roi(self, mask, metric="mean", n_components=None):
        """Extract activity from mask or ROI atlas using NiftiLabelsMasker.

        This method now uses nilearn's NiftiLabelsMasker for efficient ROI extraction
        when dealing with labeled atlases (multiple ROIs).

        Args:
            mask: Brain_Data, nibabel image, or file path. Can be:
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

        # Convert mask to Brain_Data if needed
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
            # Round values to ensure integer labels
            mask_brain.data = np.round(mask_brain.data).astype(int)
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

    # TODO: replace with nilearn or speed-up? Check if complete and delete or update comment accordingly
    def icc(self, icc_type="icc2"):
        """Calculate intraclass correlation coefficient for data within
            Brain_Data class

        ICC Formulas are based on:
        Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
        assessing rater reliability. Psychological bulletin, 86(2), 420.

        icc1:  x_ij = mu + beta_j + w_ij
        icc2/3:  x_ij = mu + alpha_i + beta_j + (ab)_ij + epsilon_ij

        Code modifed from nipype algorithms.icc
        https://github.com/nipy/nipype/blob/master/nipype/algorithms/icc.py

        Args:
            icc_type: type of icc to calculate (icc: voxel random effect,
                    icc2: voxel and column random effect, icc3: voxel and
                    column fixed effect)

        Returns:
            ICC: (np.array) intraclass correlation coefficient

        """

        Y = self.data.T
        [n, k] = Y.shape

        # Degrees of Freedom
        dfc = k - 1
        dfe = (n - 1) * (k - 1)
        dfr = n - 1

        # Sum Square Total
        mean_Y = np.mean(Y)
        SST = ((Y - mean_Y) ** 2).sum()

        # create the design matrix for the different levels
        x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
        x0 = np.tile(np.eye(n), (k, 1))  # subjects
        X = np.hstack([x, x0])

        # Sum Square Error
        predicted_Y = np.dot(
            np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T), Y.flatten("F")
        )
        residuals = Y.flatten("F") - predicted_Y
        SSE = (residuals**2).sum()

        MSE = SSE / dfe

        # Sum square column effect - between colums
        SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
        MSC = SSC / dfc / n

        # Sum Square subject effect - between rows/subjects
        SSR = SST - SSC - SSE
        MSR = SSR / dfr

        if icc_type == "icc1":
            # ICC(2,1) = (mean square subject - mean square error) /
            # (mean square subject + (k-1)*mean square error +
            # k*(mean square columns - mean square error)/n)
            # ICC = (MSR - MSRW) / (MSR + (k-1) * MSRW)
            NotImplementedError("This method isn't implemented yet.")

        elif icc_type == "icc2":
            # ICC(2,1) = (mean square subject - mean square error) /
            # (mean square subject + (k-1)*mean square error +
            # k*(mean square columns - mean square error)/n)
            ICC = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)

        elif icc_type == "icc3":
            # ICC(3,1) = (mean square subject - mean square error) /
            # (mean square subject + (k-1)*mean square error)
            ICC = (MSR - MSE) / (MSR + (k - 1) * MSE)

        return ICC

    # TODO: replace with nilearn or speed-up; currently scipy. Check if complete and delete or update comment accordingly
    def detrend(self, method="linear"):
        """Remove linear trend from each voxel

        Args:
            type: ('linear','constant', optional) type of detrending

        Returns:
            out: (Brain_Data) detrended Brain_Data instance

        """

        if len(self.shape) == 1:
            raise ValueError(
                "Make sure there is more than one image in order to detrend."
            )

        out = self._shallow_copy_with_data()
        # Copy data and detrend
        out.data = detrend(self.data.copy(), type=method, axis=0)
        return out

    def _shallow_copy_with_data(self):
        """Create a shallow copy for efficient method chaining.

        This method creates a new Brain_Data instance that shares immutable objects
        (mask, nifti_masker) but copies mutable attributes. The data array is NOT
        copied - methods should handle data copying as needed.

        Returns:
            Brain_Data: New instance with shared/copied attributes
        """
        # Create new instance without calling __init__
        new = Brain_Data.__new__(Brain_Data)

        # Copy all attributes with appropriate strategy
        for key, value in self.__dict__.items():
            if key == 'data':
                # Don't copy data array - let methods handle this
                new.data = self.data  # Just reference for now
            elif key in ['mask', 'nifti_masker', 'masker']:
                # Share immutable/expensive objects
                setattr(new, key, value)
            elif key in ['X', 'Y', 'design_matrix']:
                # Deep copy DataFrames (they're small and mutable)
                if value is not None:
                    if hasattr(value, 'copy'):  # DataFrame has .copy()
                        setattr(new, key, value.copy())
                    else:
                        setattr(new, key, deepcopy(value))
                else:
                    setattr(new, key, None)
            elif key.startswith('glm_'):
                # GLM results - share for now (they're typically read-only)
                setattr(new, key, value)
            else:
                # Small attributes - deep copy to be safe
                setattr(new, key, deepcopy(value))

        return new

    def copy(self):
        """Create a deep copy of a Brain_Data instance."""
        return deepcopy(self)

    # NOTE: utils
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
                dat: Brain_Data instance to upload
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

    # NOTE: stats
    def r_to_z(self):
        """Apply Fisher's r to z transformation to each element of the data
        object."""

        out = self._shallow_copy_with_data()
        # fisher_r_to_z creates a new array
        out.data = fisher_r_to_z(self.data)
        return out

    # NOTE: stats
    def z_to_r(self):
        """Convert z score back into r value for each element of data object"""

        out = self._shallow_copy_with_data()
        # fisher_z_to_r creates a new array
        out.data = fisher_z_to_r(self.data)
        return out

    # TODO: generalize to support other nilearn ops? Check if complete and delete or update comment accordingly
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
            Brain_Data: Filtered Brain_Data instance

        See Also:
            nilearn.signal.clean documentation for all available options
        """

        if sampling_freq is None:
            raise ValueError("Need to provide sampling rate (TR)!")
        if high_pass is None and low_pass is None:
            raise ValueError("high_pass and/or low_pass cutoff must beprovided!")
        standardize = kwargs.get("standardize", False)
        detrend = kwargs.get("detrend", False)
        out = self.copy()
        out.data = clean(
            out.data,
            t_r=1.0 / sampling_freq,
            detrend=detrend,
            standardize=standardize,
            high_pass=high_pass,
            low_pass=low_pass,
            **kwargs,
        )
        return out

    @property
    def dtype(self):
        """Get data type of Brain_Data.data."""
        return self.data.dtype

    def astype(self, dtype):
        """Cast Brain_Data.data as type.

        Args:
            dtype: datatype to convert

        Returns:
            Brain_Data: Brain_Data instance with new datatype

        """

        out = self.copy()
        out.data = out.data.astype(dtype)
        return out

    # TODO: switch to nilearn? Check if complete and delete or update comment accordingly
    def standardize(self, axis=0, method="center"):
        """Standardize Brain_Data() instance.

        Args:
            axis: 0 for observations 1 for voxels
            method: ['center','zscore']

        Returns:
            Brain_Data Instance

        """

        if axis == 1 and len(self.shape) == 1:
            raise IndexError(
                "Brain_Data is only 3d but standardization was requested over observations"
            )
        out = self.copy()
        if method == "zscore":
            with_std = True
        elif method == "center":
            with_std = False
        else:
            raise ValueError('method must be ["center","zscore"')
        out.data = scale(out.data, axis=axis, with_std=with_std)
        return out

    # TODO: switch to nilearn? Check if complete and delete or update comment accordingly
    def threshold(self, upper=None, lower=None, binarize=False, coerce_nan=True, cluster_threshold=0):
        """Threshold Brain_Data instance with optional cluster filtering.

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
            Thresholded Brain_Data object.

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
                copy_header=True
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

            if upper and lower:
                b.data[(b.data < upper) & (b.data > lower)] = 0
            elif upper:
                b.data[b.data < upper] = 0
            elif lower:
                b.data[b.data > lower] = 0

            if binarize:
                b.data[b.data != 0] = 1
            return b

    # TODO: refactor with updated nilearn Check if complete and delete or update comment accordingly
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
            is_mask (bool): Whether the Brain_Data instance should be treated
                            as a boolean mask and if so, calls
                            connected_label_regions instead.

        Returns:
            Brain_Data: Brain_Data instance with extracted ROIs as data.
        """

        if is_mask:
            regions, _ = connected_label_regions(self.to_nifti())
        else:
            regions, _ = connected_regions(
                self.to_nifti(), min_region_size, extract_type, smoothing_fwhm
            )

        return Brain_Data(regions, mask=self.mask)

    # NOTE: stats
    def transform_pairwise(self):
        """Extract brain connected regions into separate regions.

        Args:

        Returns:
            Brain_Data: Brain_Data instance tranformed into pairwise comparisons
        """
        out = self.copy()
        out.data, new_Y = transform_pairwise(self.data, self.Y)
        out.Y = pd.DataFrame(new_Y)
        out.Y.replace(-1, 0, inplace=True)
        return out

    # NOTE: stats
    def bootstrap(
        self,
        function,
        n_samples=5000,
        save_weights=False,
        n_jobs=-1,
        random_state=None,
        *args,
        **kwargs,
    ):
        """Bootstrap a `Brain_Data` method.

        Args:
            function: (str) method to apply to data for each bootstrap
            n_samples: (int) number of samples to bootstrap with replacement
            save_weights: (bool) Save each bootstrap iteration (useful for aggregating
            many bootstraps on a cluster)
            n_jobs: (int) The number of CPUs to use to do the computation. -1 means all
            CPUs.Returns:

        Returns:
            output: summarized studentized bootstrap output

        Examples:
            >>>  b = dat.bootstrap('mean', n_samples=5000)
            >>>  b = dat.bootstrap('predict', n_samples=5000, algorithm='ridge')
            >>>  b = dat.bootstrap('predict', n_samples=5000, save_weights=True)

        """

        random_state = check_random_state(random_state)
        seeds = random_state.randint(MAX_INT, size=n_samples)

        bootstrapped = Parallel(n_jobs=n_jobs)(
            delayed(_bootstrap_apply_func)(
                self, function, random_state=seeds[i], *args, **kwargs
            )
            for i in range(n_samples)
        )

        if function == "predict":
            bootstrapped = [x["weight_map"] for x in bootstrapped]
        bootstrapped = Brain_Data(bootstrapped, mask=self.mask)
        return summarize_bootstrap(bootstrapped, save_weights=save_weights)

    # NOTE: utils,
    def decompose(
        self, algorithm="pca", axis="voxels", n_components=None, *args, **kwargs
    ):
        """Decompose Brain_Data object

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

    # NOTE: stats
    def align(self, target, method="procrustes", axis=0, *args, **kwargs):
        """Align Brain_Data instance to target object using functional alignment

        Alignment type can be hyperalignment or Shared Response Model. When
        using hyperalignment, `target` image can be another subject or an
        already estimated common model. When using SRM, `target` must be a previously
        estimated common model stored as a numpy array. Transformed data can be back
        projected to original data using Tranformation matrix.

        See nltools.stats.align for aligning multiple Brain_Data instances

        Args:
            target: (Brain_Data) object to align to.
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

    # NOTE: nilearn
    # TODO: generalize with nilearn? Check if complete and delete or update comment accordingly
    def smooth(self, fwhm):
        """Apply spatial smoothing using nilearn smooth_img()

        Args:
            fwhm: (float) full width half maximum of gaussian spatial filter
        Returns:
            Brain_Data instance (copy with smoothed data)
        """
        from copy import deepcopy

        out = deepcopy(self)
        smoothed_data = out.nifti_masker.transform(smooth_img(out.to_nifti(), fwhm))

        # Ensure single images remain 1D
        if check_brain_data_is_single(out):
            out.data = smoothed_data.flatten()
        else:
            out.data = smoothed_data

        return out

    # NOTE: stats
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
        Resample Brain_Data timeseries to a new target frequency or number of samples
        using Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) interpolation.
        This function can up- or down-sample data.

        Note: this function can use quite a bit of RAM.

        Args:
            sampling_freq:  (float) sampling frequency of data in hertz
            target: (float) upsampling target
            target_type: (str) type of target can be [samples,seconds,hz]

        Returns:
            upsampled Brain_Data instance
        """

        out = self.copy()

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

    # Deprecated methods - will be moved to Model class in future release
    def predict(self, *args, **kwargs):
        """DEPRECATED: This method has been moved to the Model class."""
        raise NotImplementedError(
            "predict() has been deprecated. Please use the new Model class for prediction functionality."
        )

    def predict_multi(self, *args, **kwargs):
        """DEPRECATED: This method has been moved to the Model class."""
        raise NotImplementedError(
            "predict_multi() has been deprecated. Please use the new Model class for searchlight/multi-ROI prediction."
        )

    def randomise(self, *args, **kwargs):
        """DEPRECATED: This method has been moved to the Model class."""
        raise NotImplementedError(
            "randomise() has been deprecated. Please use the new Model class for permutation-based inference."
        )

    def ttest(self, *args, **kwargs):
        """DEPRECATED: This method has been moved to the Model class."""
        raise NotImplementedError(
            "ttest() has been deprecated. Please use the new Model class for statistical testing."
        )
