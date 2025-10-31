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
            mask: Brain mask as nibabel object, file path, or None (uses MNI template).
            masker: nilearn masker object (e.g. ROI or searchlight extractor); Default will load data as voxels
            **kwargs: Additional arguments passed to NiftiMasker.
        """
        # Import validation functions
        from ._validation import validate_data_type

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
            if not self.verbose:
                with open(os.devnull, "w") as devnull:
                    with redirect_stdout(devnull):
                        for item in data_list:
                            self.data.append(self.nifti_masker.transform(item))
            else:
                for item in data_list:
                    self.data.append(self.nifti_masker.transform(item))
            # Use vstack for nilearn 0.12+ compatibility (transforms 3D → 1D instead of 3D → 2D)
            self.data = np.vstack(self.data)

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
        if not isinstance(value, BrainData):
            raise ValueError(
                "Make sure the value you are trying to set is a BrainData() instance."
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

    @property
    def shape(self):
        """Get images by voxels shape."""

        return self.data.shape

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

    def to_nifti(self):
        """Convert BrainData Instance into Nifti Object"""

        return self.nifti_masker.inverse_transform(self.data)

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

    def fit(self, model=None, X=None, cv=None, **kwargs):
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
            **kwargs (dict): Additional arguments passed to model constructor
                - Ridge: alpha, alphas, backend, random_state
                - Glm: noise_model, minimize_memory, etc.

        Returns:
            BrainData: Fitted BrainData instance (self)

        Attributes:

            model_ (BaseModel): Fitted model instance (Ridge, Glm, etc.)
            X_ (ndarray): Training data X, stored for predict() default
            cv_results_ (dict, optional): Cross-validation results (if cv is not None). Contains:
                - 'scores': (n_folds, n_voxels) array of R² per fold
                - 'mean_score': (n_voxels,) array of mean R² across folds
                - 'predictions': BrainData of out-of-fold predictions
                - 'folds': (n_samples,) array of fold indices
                - 'best_alpha': Selected alpha (if alpha selection performed)
                - 'alpha_scores': (n_folds, n_alphas, n_voxels) array (if alpha selection)

            For model='glm':
                glm_betas (BrainData): Beta coefficients
                glm_t (BrainData): T-statistics
                glm_p (BrainData): P-values
                glm_se (BrainData): Standard errors
                glm_residual (BrainData): Residuals
                glm_predicted (BrainData): Fitted values
                glm_r2 (BrainData): R² values

            For model='ridge':
                ridge_weights (BrainData): Model coefficients
                ridge_fitted_values (BrainData): Fitted values
                ridge_scores (BrainData): R² scores

        Examples:
            >>> # Ridge prediction with CV for reporting performance
            >>> brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
            >>> print(f"CV R²: {brain_data.cv_results_['mean_score'].mean():.3f}")
            >>> predictions = brain_data.predict(X=new_features)
            >>>
            >>> # Ridge with automatic alpha selection
            >>> brain_data.fit(model='ridge', cv='auto', X=features)
            >>> print(f"Selected alpha: {brain_data.cv_results_['best_alpha']}")
            >>>
            >>> # OLS (unregularized) regression with CV
            >>> # Set alpha to small value (more stable than alpha=0)
            >>> brain_data.fit(model='ridge', alpha=1e-6, cv=5, X=features)
            >>> weights = brain_data.ridge_weights  # OLS coefficients
            >>> cv_r2 = brain_data.cv_results_['mean_score']  # Per-voxel CV R²
            >>>
            >>> # GLM regression (CV not supported yet)
            >>> brain_data.fit(model='glm', noise_model='ar1', X=design_matrix)
            >>> new_predictions = brain_data.predict(X=new_design_matrix)
        """
        from nltools.models import Ridge, Glm

        # Validate inputs
        if model is None:
            raise TypeError("model must be provided")
        if X is None:
            raise TypeError("X must be provided")

        X = np.asarray(X)
        if X.shape[0] != self.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples, but brain data has {self.shape[0]} samples. "
                f"number of samples must match."
            )

        # Store training data for predict() default
        self.X_ = X

        # Create model based on string
        if model == "ridge":
            self.model_ = Ridge(**kwargs)
            self._fit_ridge(X, cv=cv, **kwargs)
        elif model == "glm":
            if cv is not None:
                raise NotImplementedError(
                    "Cross-validation not yet supported for GLM models"
                )
            self.model_ = Glm(**kwargs)
            self._fit_glm(X)
        else:
            raise ValueError(f"Unknown model '{model}'. Must be one of: 'ridge', 'glm'")

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

        # Compute R² scores
        scores = self.model_.score(X, self.data)
        self.ridge_scores = self._shallow_copy_with_data()
        self.ridge_scores.data = scores.reshape(1, -1)  # (1, n_voxels)

    def _compute_ridge_cv(
        self, X, cv, alpha=None, alphas=None, backend="auto", **kwargs
    ):
        """Compute cross-validation results for Ridge regression.

        Args:
            X (ndarray): Training features, shape (n_samples, n_features)
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
                X, self.data, alphas=alphas, cv=cv_splitter.n_splits, backend=backend
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
            coef = ridge_svd(X_train, y_train, alpha=alpha, backend=backend)

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

    def regress(self, design_matrix=None, noise_model="ols", mode=None, **kwargs):
        """
        DEPRECATED: Use fit(model='glm', X=design_matrix) instead.

        This method is deprecated and will raise an error in v0.7.0.
        Please update your code to use the new fit/predict API.

        Args:
            design_matrix: DesignMatrix object or pandas DataFrame with regressors
                          If None, will use self.X (deprecated)
            noise_model (str): temporal variance model ('ols' or 'ar1'). Default: 'ols'
            mode (str): deprecated parameter (ignored)
            **kwargs: additional arguments for nltools.models.Glm

        Returns:
            dict: For backward compatibility, returns dict with 'beta', 't', 'p', 'residual' keys

        Sets attributes:
            self.glm_betas: Beta coefficients (BrainData)
            self.glm_t: T-statistics (BrainData)
            self.glm_p: P-values (BrainData)
            self.glm_se: Standard errors (BrainData)
            self.glm_residual: Residuals (BrainData)
            self.glm_predicted: Predicted values (BrainData)
            self.glm_r2: R-squared values (BrainData)
        """
        import warnings

        # Single strong deprecation warning
        warnings.warn(
            "regress() is deprecated and will raise an error in v0.7.0. "
            "Please use fit(model='glm', X=design_matrix) instead. "
            "Example: brain_data.fit(model='glm', noise_model='ols', X=design_matrix)",
            FutureWarning,
            stacklevel=2,
        )

        # Handle self.X backward compatibility
        if design_matrix is None:
            if hasattr(self, "X") and self.X is not None:
                design_matrix = self.X
            else:
                raise TypeError("design_matrix must be provided")

        # Ignore deprecated mode parameter silently
        # Call new fit() API
        self.fit(model="glm", noise_model=noise_model, X=design_matrix, **kwargs)

        # Set backward compatibility attributes
        self.glm_model = self.model_  # Alias for old code expecting glm_model
        self.design_matrix = design_matrix  # Store design matrix for old code

        # Return dict for backward compatibility
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
            BrainData or dict: If single contrast, returns BrainData object with contrast map.
                               If multiple contrasts (dict input), returns dict of BrainData objects.

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
        """Append data to BrainData instance.

        Args:
            data: BrainData instance to append.
            kwargs: Optional arguments passed to pandas concat.

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

        # Check to make sure masks are the same for each dataset and if not
        # create a union mask
        # This might be handy code for a new BrainData method
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
        """Calculate distance between images within a BrainData() instance.

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

        # Check to make sure masks are the same for each dataset and if not create a union mask
        # TODO: This might be handy code for a new BrainData method
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
            BrainData class

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
            elif key in ["model_", "X_"]:
                # Fitted model and training data - share (shouldn't be copied)
                setattr(new, key, value)
            else:
                # Small attributes - deep copy to be safe
                setattr(new, key, deepcopy(value))

        return new

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

    def copy(self):
        """Create a copy of a BrainData instance.

        Note: Fitted models (model\\_, X\\_) and model results (glm_*, ridge_*)
        are shared (not copied) to avoid pickle errors with unpicklable objects.
        All other attributes including the data array are deep copied.

        Returns:
            BrainData: Copied instance with shared model attributes
        """
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
        """Get data type of BrainData.data."""
        return self.data.dtype

    def astype(self, dtype):
        """Cast BrainData.data as type.

        Args:
            dtype: datatype to convert

        Returns:
            BrainData: BrainData instance with new datatype

        """

        out = self.copy()
        out.data = out.data.astype(dtype)
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
        out = self.copy()
        if method == "zscore":
            with_std = True
        elif method == "center":
            with_std = False
        else:
            raise ValueError('method must be ["center","zscore"')
        out.data = scale(out.data, axis=axis, with_std=with_std)
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

    # NOTE: stats
    def transform_pairwise(self):
        """Extract brain connected regions into separate regions.

        Args:

        Returns:
            BrainData: BrainData instance tranformed into pairwise comparisons
        """
        out = self.copy()
        out.data, new_Y = transform_pairwise(self.data, self.Y)
        out.Y = pd.DataFrame(new_Y)
        out.Y.replace(-1, 0, inplace=True)
        return out

    # TODO: update this to only support mean, median, std, sum, min, max, weights (if fit already only, predict (if fit already only))
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
        """Bootstrap a `BrainData` method.

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
        bootstrapped = BrainData(bootstrapped, mask=self.mask)
        return summarize_bootstrap(bootstrapped, save_weights=save_weights)

    # NOTE: utils,
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

    # NOTE: stats
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
    def predict(self, X=None):
        """Generate predictions using fitted model.

        Uses the model fitted during fit() to generate predictions for new data.
        Works with both Ridge and GLM models. If X is not provided, returns
        predictions on the training data used in fit().

        Args:
            X (array-like or DataFrame, optional): Data to predict on, shape (n_samples, n_features).
                Must have same n_features as training data.
                If None, uses training data from fit() (stored in ``self.X_``).

        Returns:
            BrainData: Predicted brain data with shape (n_samples, n_voxels)

        Raises:
            ValueError: If fit() has not been called yet
            ValueError: If X has wrong number of features

        Examples:
            >>> brain_data.fit(model='ridge', alpha=1.0, X=features)
            >>> predictions = brain_data.predict(X=new_features)
            >>> print(predictions.shape)
            >>>
            >>> # Predict on training data
            >>> train_predictions = brain_data.predict()
            >>> print(train_predictions.shape)
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
        if X is None:
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
            if X is self.X_:
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
