"""
NeuroLearn Brain Data
=====================

Classes to represent brain image data.

"""

import os
import warnings  # noqa: F401

import numpy as np
import pandas as pd
from copy import deepcopy
from nltools.utils import attempt_to_import
from .utils import check_brain_data
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
    """BrainData is a class to represent neuroimaging data in python as a vector
    rather than a 3-dimensional matrix. This makes it easier to perform data
    manipulation and analyses.

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

    def __init__(self, data=None, Y=None, X=None, mask=None, masker=None, **kwargs):
        from .validation import validate_data_type
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
        self._resample = kwargs.pop("resample", True)
        self._interpolation = kwargs.pop("interpolation", "auto")
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
            load_from_brain_data(self, data, mask)
        elif data_type == "h5":
            load_from_h5(self, data, mask)
            return
        elif data_type == "list":
            load_from_list(self, data)
        elif data_type == "url":
            load_from_url(self, data)
        elif data_type in ["file", "nibabel"]:
            load_from_file(self, data)

        # Collapse extra trailing dimensions, but preserve samples dimension for list inputs
        if self.data is not None and self.data.ndim > 1 and data_type != "list":
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
    # Dunders (alphabetical)
    # =========================================================================

    def __add__(self, y):
        """Add to BrainData."""
        from .utils import perform_arithmetic

        return perform_arithmetic(self, y, np.add, "add")

    def __deepcopy__(self, memo):
        """Custom deepcopy that handles model attributes.

        Model-related attributes (model_, X_, glm_*, ridge_*) are shared
        (not copied) to avoid pickle errors with unpicklable Backend objects.
        All other attributes are deep copied.
        """
        new = BrainData.__new__(BrainData)
        memo[id(self)] = new

        for key, value in self.__dict__.items():
            if key in ("mask", "nifti_masker", "masker"):
                setattr(new, key, value)
            elif key in ("model_", "X_"):
                setattr(new, key, value)
            elif key.startswith("glm_") or key.startswith("ridge_"):
                setattr(new, key, value)
            else:
                setattr(new, key, deepcopy(value, memo))

        return new

    def __eq__(self, other):
        """Check equality between BrainData."""
        if not isinstance(other, BrainData):
            return False

        eq_data = np.all(self.data == other.data)

        if self.X is None and other.X is None:
            eq_X = True
        elif self.X is None or other.X is None:
            eq_X = False
        else:
            eq_X = self.X.equals(other.X)

        if self.Y is None and other.Y is None:
            eq_Y = True
        elif self.Y is None or other.Y is None:
            eq_Y = False
        else:
            eq_Y = self.Y.equals(other.Y)

        if self.mask is None and other.mask is None:
            eq_mask = True
        elif self.mask is None or other.mask is None:
            eq_mask = False
        elif hasattr(self.mask, "get_filename") and hasattr(other.mask, "get_filename"):
            eq_mask = self.mask.get_filename() == other.mask.get_filename()
        else:
            eq_mask = self.mask == other.mask

        return eq_data and eq_X and eq_Y and eq_mask

    def __getitem__(self, index):
        from .utils import shallow_copy

        new = shallow_copy(self)
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

    def __iadd__(self, y):
        """In-place addition (+=)."""
        from .utils import perform_arithmetic

        return perform_arithmetic(self, y, np.add, "add", inplace=True)

    def __imul__(self, y):
        """In-place multiplication (*=)."""
        from .utils import perform_arithmetic

        return perform_arithmetic(self, y, np.multiply, "multiply", inplace=True)

    def __isub__(self, y):
        """In-place subtraction (-=)."""
        from .utils import perform_arithmetic

        return perform_arithmetic(self, y, np.subtract, "subtract", inplace=True)

    def __iter__(self):
        for x in range(len(self)):
            yield self[x]

    def __itruediv__(self, y):
        """In-place true division (/=)."""
        from .utils import perform_arithmetic

        with np.errstate(invalid="ignore", divide="ignore"):
            return perform_arithmetic(self, y, np.divide, "divide", inplace=True)

    def __len__(self):
        return self.shape[0]

    def __mul__(self, y):
        """Multiply BrainData."""
        from .utils import perform_arithmetic

        return perform_arithmetic(self, y, np.multiply, "multiply")

    def __radd__(self, y):
        """Right add to BrainData."""
        from .utils import perform_arithmetic

        return perform_arithmetic(self, y, np.add, "add")

    def __repr__(self):
        mask_filename = self.mask.get_filename()
        mask_display = os.path.basename(mask_filename) if mask_filename else "None"

        if hasattr(self, "_voxel_resolution") and self._voxel_resolution is not None:
            if np.allclose(self._voxel_resolution, self._voxel_resolution[0]):
                resolution_str = f"{self._voxel_resolution[0]:.1f}mm"
            else:
                resolution_str = (
                    f"{self._voxel_resolution[0]:.1f}x"
                    f"{self._voxel_resolution[1]:.1f}x"
                    f"{self._voxel_resolution[2]:.1f}mm"
                )
        else:
            resolution_str = "unknown"

        space_str = getattr(self, "_space", "unknown")

        return "%s.%s(data=%s, resolution=%s, space=%s, mask=%s)" % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.shape,
            resolution_str,
            space_str,
            mask_display,
        )

    def __rmul__(self, y):
        """Right multiply BrainData."""
        from .utils import perform_arithmetic

        return perform_arithmetic(self, y, np.multiply, "multiply")

    def __rsub__(self, y):
        """Right subtract from BrainData."""
        from .utils import perform_arithmetic

        return perform_arithmetic(self, y, np.subtract, "subtract", reverse=True)

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

    def __sub__(self, y):
        """Subtract from BrainData."""
        from .utils import perform_arithmetic

        return perform_arithmetic(self, y, np.subtract, "subtract")

    def __truediv__(self, y):
        """Divide BrainData."""
        from .utils import perform_arithmetic

        with np.errstate(invalid="ignore", divide="ignore"):
            return perform_arithmetic(self, y, np.divide, "divide")

    # =========================================================================
    # Properties (alphabetical)
    # =========================================================================

    @property
    def dtype(self):
        """Get data type of BrainData.data."""
        return self.data.dtype

    @property
    def is_empty(self) -> bool:
        """Check if BrainData.data is empty."""
        if isinstance(self.data, np.ndarray):
            return self.data.size == 0
        if isinstance(self.data, list):
            return len(self.data) == 0
        return True

    @property
    def shape(self):
        """Get images by voxels shape."""
        return self.data.shape

    # =========================================================================
    # Public methods (alphabetical)
    # =========================================================================

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
        from .validation import validate_append_shapes
        from .utils import shallow_copy

        data = check_brain_data(data)

        if self.is_empty:
            out = shallow_copy(data)
            out.data = data.data.copy()
        else:
            validate_append_shapes(self.shape, data.shape)

            out = shallow_copy(self)
            out.data = np.vstack([self.data, data.data])

            if not ignore_attrs:
                if (
                    hasattr(self, "X")
                    and self.X is not None
                    and hasattr(data, "X")
                    and data.X is not None
                ):
                    out.X = pd.concat([self.X, data.X], ignore_index=True, **kwargs)
                elif hasattr(data, "X") and data.X is not None:
                    out.X = data.X.copy()

                if (
                    hasattr(self, "Y")
                    and self.Y is not None
                    and hasattr(data, "Y")
                    and data.Y is not None
                ):
                    out.Y = pd.concat([self.Y, data.Y], ignore_index=True, **kwargs)
                elif hasattr(data, "Y") and data.Y is not None:
                    out.Y = data.Y.copy()
            else:
                out.X = None
                out.Y = None

        return out

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

    def astype(self, dtype):
        """Cast BrainData.data as type.

        Args:
            dtype: datatype to convert

        Returns:
            BrainData: BrainData instance with new datatype
        """
        from .utils import shallow_copy

        out = shallow_copy(self)
        out.data = self.data.astype(dtype)
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
            >>> brain.fit(model='glm', X=design_matrix)
            >>> contrast1 = brain.compute_contrasts([0, 1, -1])
            >>> contrast2 = brain.compute_contrasts("conditionA - conditionB")
            >>> results = brain.compute_contrasts({
            ...     "A_vs_B": "conditionA - conditionB",
            ...     "avg_effect": [0, 0.5, 0.5],
            ... })

        Notes:
            - String contrasts support coefficients: "2*A - B" or "0.5*A + 0.5*B"
            - Column names must match design matrix columns exactly (case-sensitive)
            - Contrast weights should sum to zero for proper inference in most cases
        """
        from .modeling import compute_contrasts

        return compute_contrasts(self, contrasts, contrast_type=contrast_type)

    def copy(self):
        """Create a deep copy of a BrainData instance.

        All attributes including data, fitted models, and results are deep copied.
        Use this when you need a complete independent copy.

        Returns:
            BrainData: Deep copied instance
        """
        return deepcopy(self)

    def create_empty(self):
        """Create a copy of BrainData with empty data array.

        Returns:
            BrainData: A copy of this object with an empty data array.
        """
        out = deepcopy(self)
        out.data = np.array([])
        return out

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
            groups: Explicit group labels for CV splits.
            random_state: Random seed for reproducibility.
            **kwargs: Additional arguments passed to CVScheme.

        Returns:
            BrainDataPipeline: A pipeline object for method chaining.

        Examples:
            >>> result = brain.cv(k=5).predict(y, algorithm='ridge')
            >>> result = brain.cv(scheme='loro', groups=run_labels).predict(y)
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

    def detrend(self, method="linear"):
        """Remove linear trend from each voxel.

        Args:
            method: ('linear','constant', optional) type of detrending

        Returns:
            out: (BrainData) detrended BrainData instance
        """
        from .analysis import detrend_data

        return detrend_data(self, method=method)

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
            X (array-like or DataFrame): Design matrix or feature matrix
            cv (int, 'auto', or sklearn CV splitter, optional): Cross-validation specification (Ridge only)
            inplace (bool, default=True): If True, mutate self and return self.
                If False, return Fit dataclass with results (self unchanged).
            progress_bar (bool, optional): Display progress bar during fitting.
            scale (bool, default=True): Apply grand-mean scaling before fitting.
            scale_value (float, default=100.0): Target value for mean after scaling.
            **kwargs (dict): Additional arguments passed to model constructor

        Returns:
            BrainData or Fit: If ``inplace=True``, returns self (fitted BrainData).
                If ``inplace=False``, returns Fit dataclass with results.

        Examples:
            >>> brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
            >>> fit = brain_data.fit(model='ridge', alpha=1.0, X=features, inplace=False)
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

    def mean(self, axis=0):
        """Get mean of each voxel or image.

        Args:
            axis: 0 = across images (default, returns BrainData),
                1 = within images (returns array)

        Returns:
            float/np.array/BrainData: Mean values.
        """
        from .utils import apply_func

        return apply_func(self, np.mean, axis)

    def median(self, axis=0):
        """Get median of each voxel or image.

        Args:
            axis: 0 = across images (default, returns BrainData),
                1 = within images (returns array)

        Returns:
            float/np.array/BrainData: Median values.
        """
        from .utils import apply_func

        return apply_func(self, np.median, axis)

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

        Args:
            kind (str): Visualization type: 'glass', 'slices', 'timeseries', 'histogram'
            thr_upper (str/float, optional): Upper threshold.
            thr_lower (str/float, optional): Lower threshold.
            threshold (float, optional): Convenience parameter for thresholding.
            cut_coords (list, optional): Cut coordinates for multi-slice views.
            cmap (str, optional): Colormap name.
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
            threshold (float, optional): Values below this absolute threshold are masked.
            cmap (str): Matplotlib colormap. Default: 'RdBu_r'.
            vmax (float, optional): Maximum value for colormap.
            vmin (float, optional): Minimum value for colormap.
            template (str): Freesurfer surface resolution. Default: 'fsaverage5'.
            with_curvature (bool): Show sulcal/gyral pattern. Default: True.
            curvature_contrast (float): Contrast of curvature overlay. Default: 0.5.
            curvature_brightness (float): Mean brightness of curvature overlay. Default: 0.5.
            colorbar (bool): Show colorbar. Default: True.
            colorbar_orientation (str): 'horizontal' or 'vertical'. Default: 'horizontal'.
            figsize (tuple): Figure size as (width, height). Default: (12, 6).
            title (str, optional): Figure title.
            radius (float): Sampling radius in mm. Default: 3.0.
            interpolation (str): Interpolation method. Default: 'linear'.
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
            estimator: Classifier ('svm', 'logistic', 'ridge', 'lda', or sklearn estimator).
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

    def r_to_z(self):
        """Apply Fisher's r to z transformation to each element of the data
        object."""
        from .analysis import r_to_z

        return r_to_z(self)

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

    def resample_to(self, img=None, resolution=None, interpolation=None):
        """Resample BrainData to match target image or resolution.

        Args:
            img: Target image for resampling (nibabel Nifti1Image, str/Path, or None).
            resolution: Target voxel size in mm (float/int for isotropic, or None).
            interpolation: Interpolation method ('nearest', 'linear', 'continuous', or None).

        Returns:
            BrainData: New BrainData instance with resampled data

        Raises:
            ValueError: If both img and resolution are None, or both are provided
        """
        from .io import resample_to

        return resample_to(self, img, resolution, interpolation)

    def scale(self, scale_val=100.0, axis=None):
        """Scale data via mean scaling.

        Two scaling modes are available:

        - **Grand-mean scaling** (axis=None, default): Divides all values by the
          global mean across all voxels and timepoints.

        - **Voxel-wise scaling** (axis=0): Divides each voxel's time-series by
          its own temporal mean.

        Args:
            scale_val: (int/float) Target value for the mean after scaling. Default 100.
            axis: (int or None) None for grand-mean scaling (default),
                0 for voxel-wise scaling.

        Returns:
            BrainData: New BrainData instance with scaled data.
        """
        from .analysis import scale_data

        return scale_data(self, scale_val, axis)

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

    def smooth(self, fwhm):
        """Apply spatial smoothing using nilearn smooth_img().

        Args:
            fwhm: (float) full width half maximum of gaussian spatial filter

        Returns:
            BrainData instance (copy with smoothed data)
        """
        from .analysis import smooth

        return smooth(self, fwhm)

    def standardize(self, axis=0, method="center"):
        """Standardize BrainData() instance.

        Args:
            axis (int): 0 standardizes each voxel across observations (default).
                1 standardizes each observation across voxels.
            method (str): 'center' subtracts the mean (default).
                'zscore' subtracts the mean and divides by standard deviation.

        Returns:
            BrainData: Standardized BrainData instance.
        """
        from .analysis import standardize

        return standardize(self, axis=axis, method=method)

    def std(self, axis=0):
        """Get standard deviation of each voxel or image.

        Args:
            axis: 0 = across images (default, returns BrainData),
                1 = within images (returns array)

        Returns:
            float/np.array/BrainData: Standard deviation values.
        """
        from .utils import apply_func

        return apply_func(self, np.std, axis)

    def sum(self, axis=0):
        """Get sum of each voxel or image.

        Args:
            axis: 0 = across images (default, returns BrainData),
                1 = within images (returns array)

        Returns:
            float/np.array/BrainData: Sum values.
        """
        from .utils import apply_func

        return apply_func(self, np.sum, axis)

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

    def to_nifti(self):
        """Convert BrainData Instance into Nifti Object.

        Returns:
            nibabel.Nifti1Image: Brain data as a NIfTI image.
        """
        from .io import to_nifti

        return to_nifti(self)

    def transform_pairwise(self):
        """Transform data into pairwise comparisons.

        Returns:
            BrainData: BrainData instance transformed into pairwise comparisons
        """
        from .analysis import transform_pairwise_data

        return transform_pairwise_data(self)

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

    def write(self, file_name):
        """Write out BrainData object to Nifti or HDF5 File.

        Args:
            file_name (str or Path): Output file path (.nii/.nii.gz for NIfTI,
                .h5/.hdf5 for HDF5).
        """
        from .io import write_brain_data

        write_brain_data(self, file_name)

    def z_to_r(self):
        """Convert z score back into r value for each element of data object."""
        from .analysis import z_to_r

        return z_to_r(self)
