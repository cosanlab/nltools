"""Represent brain image data with the BrainData class.

# NeuroLearn Brain Data

Classes to represent brain image data.

"""

import os
import warnings  # noqa: F401
from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nltools.data.atlases import Atlas, ClusterReport

from nltools.utils import attempt_to_import, coalesced_gc

from .utils import check_brain_data

warnings.filterwarnings("ignore", category=UserWarning, module="nilearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="nilearn")

# Optional dependencies
nx = attempt_to_import("networkx", "nx")
MAX_INT = np.iinfo(np.int32).max

__all__ = ["BrainData"]


class BrainData:
    """Represent neuroimaging data as vectors instead of three-dimensional matrices.

    This representation makes it easier to perform data manipulation and analyses.

    Args:
        data: Neuroimaging data. Can be:
            - None (empty BrainData)
            - BrainData object
            - List of BrainData objects or file paths
            - File path (str/Path) to .nii/.nii.gz/.h5/.hdf5
            - nibabel Nifti1Image object
            - URL to download data from
            - numpy array (1D ``(n_voxels,)`` for a single image or 2D
              ``(n_images, n_voxels)`` for a stack). The ``mask`` argument
              is required and must define the same number of in-mask voxels.
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
    """

    def __init__(
        self,
        data=None,
        *,
        Y=None,
        X=None,
        mask=None,
        masker=None,
        h5_compression="gzip",
        verbose=False,
        resample=True,
        interpolation="auto",
    ):
        from .io import (
            initialize_mask,
            load_from_brain_data,
            load_from_file,
            load_from_h5,
            load_from_list,
            load_from_url,
        )
        from .validation import validate_data_type

        # Initialize attributes
        self._h5_compression = h5_compression
        self.verbose = verbose
        self._resample = resample
        self._interpolation = interpolation
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
        initialize_mask(self, mask)

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
        elif data_type == "array":
            # Raw numpy array path. Requires an explicit mask because without
            # one we can't map the flat voxel axis to 3D space. Accepts 1D
            # (n_voxels,) for a single image or 2D (n_images, n_voxels) for a
            # stack. Values are stored as-is; users are expected to have
            # already applied any scaling they want.
            if mask is None:
                raise ValueError(
                    "Constructing BrainData from a numpy array requires an "
                    "explicit mask — pass mask=<path|Nifti1Image> that matches "
                    "the array's voxel axis."
                )
            arr = np.asarray(data)
            if arr.ndim not in (1, 2):
                raise ValueError(
                    f"numpy array input must be 1D (n_voxels,) or 2D "
                    f"(n_images, n_voxels); got shape {arr.shape}"
                )
            n_voxels_mask = int((self.mask.get_fdata() > 0).sum())
            if arr.shape[-1] != n_voxels_mask:
                raise ValueError(
                    f"numpy array last axis ({arr.shape[-1]}) must match the "
                    f"number of in-mask voxels ({n_voxels_mask})."
                )
            self.data = arr

        # Collapse extra trailing dimensions, but preserve samples dimension for list inputs
        if self.data is not None and self.data.ndim > 1 and data_type != "list":
            if 1 in self.data.shape:
                self.data = self.data.squeeze()

        # Set X and Y. Invariant: .X and .Y are always polars DataFrames
        # (possibly empty). Assignment goes through the property setter,
        # which pipes through validate_frame for pandas/numpy/csv ingress.
        if X is not None:
            self.X = X
        elif data_type == "brain_data" and hasattr(data, "X"):
            self.X = data.X
        else:
            self.X = None

        if Y is not None:
            self.Y = Y
        elif data_type == "brain_data" and hasattr(data, "Y"):
            self.Y = data.Y
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
            if key in ("mask", "masker"):
                setattr(new, key, value)
            elif key in ("model_", "X_"):
                setattr(new, key, value)
            elif key.startswith(("glm_", "ridge_")):
                setattr(new, key, value)
            else:
                setattr(new, key, deepcopy(value, memo))

        return new

    def __eq__(self, other):
        """Check equality between BrainData."""
        if not isinstance(other, BrainData):
            return False

        eq_data = np.all(self.data == other.data)
        eq_X = self.X.equals(other.X)
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
        from .utils import _polars_row_select, shallow_copy

        new = shallow_copy(self)
        if isinstance(index, (int, np.integer)):
            new.data = np.array(self.data[index, :]).squeeze()
        elif isinstance(index, slice):
            new.data = self.data[index, :]
        else:
            index = np.array(index).flatten()
            new.data = np.array(self.data[index, :]).squeeze()

        if not self.Y.is_empty():
            new.Y = _polars_row_select(self.Y, index)
        if not self.X.is_empty():
            new.X = _polars_row_select(self.X, index)
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

        return f"{self.__class__.__module__}.{self.__class__.__name__}(data={self.shape}, resolution={resolution_str}, space={space_str}, mask={mask_display})"

    def __rmul__(self, y):
        """Right multiply BrainData."""
        from .utils import perform_arithmetic

        return perform_arithmetic(self, y, np.multiply, "multiply")

    def __rsub__(self, y):
        """Right subtract from BrainData."""
        from .utils import perform_arithmetic

        return perform_arithmetic(self, y, np.subtract, "subtract", reverse=True)

    def __setitem__(self, index, value):
        import polars as pl

        if not isinstance(value, BrainData):
            raise ValueError(
                "Make sure the value you are trying to set is a BrainData() instance."
            )
        self.data[index, :] = value.data
        if not value.Y.is_empty():
            if self.Y.is_empty():
                raise ValueError("Cannot set Y values: self.Y is empty.")
            arr = self.Y.to_numpy()
            arr[index] = value.Y.to_numpy()
            self.Y = pl.DataFrame(arr, schema=self.Y.columns)
        if not value.X.is_empty():
            if self.X.is_empty():
                raise ValueError("Cannot set X values: self.X is empty.")
            if self.X.shape[1] != value.X.shape[1]:
                raise ValueError("Make sure self.X is the same size as value.X.")
            arr = self.X.to_numpy()
            arr[index] = value.X.to_numpy()
            self.X = pl.DataFrame(arr, schema=self.X.columns)

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

    @property
    def size(self):
        """Total number of elements in BrainData.data (numpy convention)."""
        return self.data.size

    @property
    def X(self):
        """Design matrix / per-image covariates as a polars DataFrame."""
        return self._X

    @X.setter
    def X(self, value):
        from .validation import validate_frame

        self._X = validate_frame(value, frame_type="X")

    @property
    def Y(self):
        """Per-image targets as a polars DataFrame."""
        return self._Y

    @Y.setter
    def Y(self, value):
        from .validation import validate_frame

        self._Y = validate_frame(value, frame_type="Y")

    # =========================================================================
    # Public methods (alphabetical)
    # =========================================================================

    @coalesced_gc()
    def align(
        self,
        target,
        method="procrustes",
        axis=0,
        *,
        spatial_scale: str = "whole_brain",
        roi_mask=None,
        radius_mm: float = 10.0,
    ):
        """Align BrainData instance to target object using functional alignment.

        Args:
            target: (BrainData) object to align to.
            method: (str) alignment method to use
                ['probabilistic_srm','deterministic_srm','procrustes']
            axis: (int) axis to align on
            spatial_scale: ``'whole_brain'`` (default), ``'roi'``, or
                ``'searchlight'``. ``'roi'`` / ``'searchlight'`` are not
                yet implemented (per-parcel transforms + reassembly is a
                follow-up slice).
            roi_mask: Reserved for ``spatial_scale='roi'``.
            radius_mm: Reserved for ``spatial_scale='searchlight'``.

        Returns:
            out: (dict) a dictionary containing transformed object,
                transformation matrix, and the shared response matrix

        Examples:
            >>> out = data.align(target, method='procrustes')
            >>> out = data.align(target, method='probabilistic_srm')
        """
        if spatial_scale == "searchlight":
            raise NotImplementedError(
                "align(spatial_scale='searchlight') is not implemented: "
                "searchlight neighborhoods overlap, so a single voxel "
                "belongs to many spheres and there is no canonical value "
                "to put back at that voxel for the 'transformed' field. "
                "Use spatial_scale='roi' (disjoint parcels) or "
                "spatial_scale='whole_brain'; if you need per-sphere "
                "transforms, iterate compute_searchlight_neighborhoods() "
                "and call .align() yourself."
            )
        if spatial_scale == "roi":
            from .analysis import align_per_roi

            return align_per_roi(
                self, target, method=method, axis=axis, roi_mask=roi_mask
            )
        if spatial_scale != "whole_brain":
            raise ValueError(
                f"spatial_scale must be one of "
                f"{{'whole_brain', 'roi', 'searchlight'}}, got {spatial_scale!r}"
            )
        from .analysis import align

        return align(self, target, method=method, axis=axis)

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
        from .utils import shallow_copy
        from .validation import validate_append_shapes

        data = check_brain_data(data)

        if self.is_empty:
            out = shallow_copy(data)
            out.data = data.data.copy()
        else:
            validate_append_shapes(self.shape, data.shape)

            out = shallow_copy(self)
            out.data = np.vstack([self.data, data.data])

            if not ignore_attrs:
                import polars as pl

                if not self.X.is_empty() and not data.X.is_empty():
                    out.X = pl.concat([self.X, data.X], how="vertical_relaxed")
                elif not data.X.is_empty():
                    out.X = data.X

                if not self.Y.is_empty() and not data.Y.is_empty():
                    out.Y = pl.concat([self.Y, data.Y], how="vertical_relaxed")
                elif not data.Y.is_empty():
                    out.Y = data.Y
            else:
                out.X = None
                out.Y = None

        return out

    @coalesced_gc()
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
        percentiles=(2.5, 97.5),
        X_test=None,
        backend=None,
        max_gpu_memory_gb=4.0,
        n_jobs=-1,
        random_state=None,
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
            percentiles: (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5)
            X_test: (np.ndarray, optional) Test features for 'predict' bootstrap.
            backend: (str, optional) Backend for Ridge bootstrap: None (CPU), 'torch'
                (GPU if available), or 'auto' (auto-select). Ignored for simple stats.
            max_gpu_memory_gb: (float) Maximum GPU memory to use when backend is 'torch'
                or 'auto'. Default: 4.0
            n_jobs: (int) Number of CPU cores for parallelization. -1 means all CPUs.
            random_state: (int, optional) Random seed for reproducibility

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
            percentiles=percentiles,
            X_test=X_test,
            backend=backend,
            max_gpu_memory_gb=max_gpu_memory_gb,
            n_jobs=n_jobs,
            random_state=random_state,
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
            contrast_type (str): What to return per contrast. One of `"t"` (default,
                t-statistic map), `"z"` (z-score), `"p"` (p-value), `"beta"` /
                `"effect_size"` (effect-size β map — use this when feeding a
                second-level group analysis), or `"all"` (a bundle dict
                `{"beta", "t", "z", "p", "se"}` of maps for one contrast). Default: `"t"`.

        Returns:
            BrainData or dict: A single contrast with a scalar `contrast_type` returns a
                `BrainData` map; with `contrast_type="all"` it returns a flat dict keyed by
                `"beta"`/`"t"`/`"z"`/`"p"`/`"se"`. A dict of contrasts returns a dict keyed
                by contrast name (nested under the five keys when `contrast_type="all"`).

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

    @coalesced_gc()
    def decompose(self, *, method="pca", axis="voxels", n_components=None, **kwargs):
        """Decompose BrainData object.

        Args:
            method: (str) Algorithm to perform decomposition
                        types=['pca','ica','nnmf','fa','dictionary','kernelpca']
            axis: dimension to decompose ['voxels','images']
            n_components: (int) number of components. If None then retain
                        as many as possible.
            **kwargs: forwarded to the underlying sklearn decomposition estimator.

        Returns:
            output: a dictionary of decomposition parameters
        """
        from .analysis import decompose

        return decompose(
            self,
            method=method,
            axis=axis,
            n_components=n_components,
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

    @coalesced_gc()
    def distance(
        self,
        metric="euclidean",
        *,
        spatial_scale: str = "whole_brain",
        roi_mask=None,
        radius_mm: float = 10.0,
        **kwargs,
    ):
        """Calculate distance between images within a BrainData() instance.

        Args:
            metric: (str) type of distance metric (can use any scipy.spatial.distance
                    metric supported by cdist)
            spatial_scale: One of ``'whole_brain'`` (default), ``'roi'``, or
                ``'searchlight'``. ``'whole_brain'`` returns a single
                pairwise distance ``Adjacency`` between images. ``'roi'``
                requires ``roi_mask`` and returns a stacked ``Adjacency``
                with one RDM per parcel and ``spatial_scale`` provenance
                attached for back-projection via ``Adjacency.to_brain()``.
                ``'searchlight'`` requires ``radius_mm`` (and is not yet
                implemented in this slice).
            roi_mask: Atlas image (BrainData / Nifti1Image / path) for
                ``spatial_scale='roi'``.
            radius_mm: Searchlight radius in mm. Default 10.0.

        Returns:
            Adjacency: Single pairwise distance matrix for ``'whole_brain'``;
                stacked Adjacency (one matrix per parcel/searchlight) with
                ``spatial_scale`` set for ``'roi'`` / ``'searchlight'``.
        """
        from .analysis import distance

        return distance(
            self,
            metric=metric,
            spatial_scale=spatial_scale,
            roi_mask=roi_mask,
            radius_mm=radius_mm,
            **kwargs,
        )

    @coalesced_gc()
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

    def find_spikes(
        self,
        global_spike_cutoff=3,
        diff_spike_cutoff=3,
        *,
        TR: float | None = None,
        sampling_freq: float | None = None,
    ):
        """Identify spikes from Time Series Data.

        Args:
            global_spike_cutoff (int or None): cutoff to identify spikes in global signal
                in standard deviations, or None to skip.
            diff_spike_cutoff (int or None): cutoff to identify spikes in average frame
                difference in standard deviations, or None to skip.
            TR: Repetition time in seconds. Sets the returned DesignMatrix's
                sampling_freq for downstream `.append(...)` / `.convolve()`.
                Pass exactly one of `TR` or `sampling_freq`.
            sampling_freq: Sampling frequency in Hz (= 1/TR). See `TR`.

        Returns:
            DesignMatrix with one indicator column per detected spike, with
            all spike columns pre-marked as confounds.
        """
        from .analysis import find_spikes_data

        return find_spikes_data(
            self,
            global_spike_cutoff=global_spike_cutoff,
            diff_spike_cutoff=diff_spike_cutoff,
            TR=TR,
            sampling_freq=sampling_freq,
        )

    @coalesced_gc()
    def fit(
        self,
        model="glm",
        *,
        X=None,
        cv=None,
        local_alpha=True,
        fit_intercept=False,
        inplace=True,
        scale=True,
        scale_value=100.0,
        progress_bar=None,
        design_clean=True,
        design_clean_thresh=0.95,
        design_clean_exclude_confounds=False,
        design_clean_fill_na=0,
        **kwargs,
    ):
        """Fit a model to brain imaging data.

        Creates and fits a model from string specification. The brain data
        (self.data) is always used as the target variable. Model and results
        are stored for later use with predict().

        Args:
            model (str): Model type: 'ridge', 'glm', or future model names
            X (array-like or DataFrame): Design matrix or feature matrix
            cv (int or sklearn CV splitter, optional): Cross-validation
                specification (Ridge only). int → ``KFold(cv)``; pass a
                splitter object (e.g. ``KFold(5, shuffle=True)``,
                ``GroupKFold(8)``) for non-contiguous folds. Generators
                (``splitter.split(X)``) are rejected.
            local_alpha (bool, default=True): Ridge only. If True, select
                α independently per voxel via ``solve_ridge_cv``. If False,
                pick a single α shared across all voxels.
            fit_intercept (bool, default=False): Ridge only. Forwarded to
                the Ridge model — center X and y on the training fold mean
                per fold and recover the intercept after.
            inplace (bool, default=True): If True, mutate self and return self.
                If False, return a Fit dataclass with the results. ``self.data``
                and the result attributes (``ridge_*`` / ``glm_*`` /
                ``cv_results_``) are left unchanged, but ``self.model_`` and
                ``self.X_`` (plus ``self.design_matrix`` for GLM) ARE updated on
                self so ``predict()`` / ``compute_contrasts()`` still work.
            scale (bool, default=True): Apply grand-mean scaling before fitting.
            scale_value (float, default=100.0): Target value for mean after scaling.
            progress_bar (bool, optional): Display progress bar during fitting.
            design_clean (bool, default=True): GLM only. Run
                ``DesignMatrix.clean()`` on ``X`` before fitting to drop
                highly correlated regressors. Coerces ``X`` to
                ``DesignMatrix`` if needed. Ignored when ``model='ridge'``.
            design_clean_thresh (float, default=0.95): GLM only. Correlation
                threshold passed to ``DesignMatrix.clean()`` (drops if
                ``abs(r) >= thresh``). Ignored when ``model='ridge'``.
            design_clean_exclude_confounds (bool, default=False): GLM only.
                If True, ``DesignMatrix.clean()`` skips confound columns when
                checking correlations. Ignored when ``model='ridge'``.
            design_clean_fill_na (int, float, or None, default=0): GLM only.
                Fill value for NaNs before correlation check in
                ``DesignMatrix.clean()``. Ignored when ``model='ridge'``.
            **kwargs (dict): Additional arguments passed to model constructor

        Returns:
            BrainData or Fit: If ``inplace=True``, returns self (fitted BrainData).
                If ``inplace=False``, returns Fit dataclass with results.

        Notes:
            After ``model="glm"``, the following per-regressor BrainData
            attributes are populated — one map per design-matrix column:

            - ``glm_betas``: effect-size (β) maps.
            - ``glm_t``: marginal t-statistic for each regressor.
            - ``glm_p``: marginal p-value.
            - ``glm_se``: standard error of β.
            - ``glm_r2``: voxel-wise R².

            ``glm_t[i]`` is a valid t-map for the trivial one-hot contrast on
            regressor ``i`` only. For contrasts across regressors
            (``"A - B"``, ``[1, -1, 0, ...]``) use `compute_contrasts` —
            you cannot correctly combine these per-regressor maps by hand
            because t-statistic arithmetic requires the off-diagonal elements
            of the parameter covariance matrix, which are not stored. Pass
            ``contrast_type="all"`` to get ``β``/``t``/``z``/``p``/``se`` for
            one contrast in a single call.

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
            local_alpha=local_alpha,
            fit_intercept=fit_intercept,
            inplace=inplace,
            scale=scale,
            scale_value=scale_value,
            progress_bar=progress_bar,
            design_clean=design_clean,
            design_clean_thresh=design_clean_thresh,
            design_clean_exclude_confounds=design_clean_exclude_confounds,
            design_clean_fill_na=design_clean_fill_na,
            **kwargs,
        )

    def icc(
        self,
        n_subjects,
        n_sessions,
        method="icc2",
        parallel=None,
        n_jobs=-1,
        max_gpu_memory_gb=4.0,
    ):
        """Calculate voxel-wise intraclass correlation coefficient.

        ICC Formulas based on Shrout & Fleiss (1979).

        Args:
            n_subjects: Number of subjects in the data
            n_sessions: Number of sessions per subject
            method: Type of ICC ('icc1', 'icc2', 'icc3'). Default: 'icc2'
            parallel: Parallelization method (None, 'cpu', 'gpu')
            n_jobs: Number of CPU cores (-1 = all cores)
            max_gpu_memory_gb: GPU memory budget in GB

        Returns:
            BrainData: BrainData instance with ICC map (shape: (1, n_voxels))

        Examples:
            >>> icc_map = data.icc(n_subjects=20, n_sessions=3, method='icc2')
        """
        from .analysis import icc

        return icc(
            self,
            n_subjects,
            n_sessions,
            method=method,
            parallel=parallel,
            n_jobs=n_jobs,
            max_gpu_memory_gb=max_gpu_memory_gb,
        )

    def mean(self, axis=0, *, spatial_scale: str = "whole_brain", roi_mask=None):
        """Get mean of each voxel or image.

        Args:
            axis: 0 = across images (default, returns BrainData),
                1 = within images (returns array). Ignored when
                ``spatial_scale='roi'``.
            spatial_scale: ``'whole_brain'`` (default) preserves existing
                behavior. ``'roi'`` requires ``roi_mask`` and returns a
                BrainData of the same shape with each voxel painted with
                its parcel's mean per image (parcellation smoothing).
            roi_mask: Atlas image for ``spatial_scale='roi'``.

        Returns:
            float/np.array/BrainData: Mean values.
        """
        if spatial_scale == "roi":
            from .analysis import reduce_per_roi

            return reduce_per_roi(self, np.mean, roi_mask=roi_mask)
        from .utils import apply_func

        return apply_func(self, np.mean, axis)

    def median(self, axis=0, *, spatial_scale: str = "whole_brain", roi_mask=None):
        """Get median of each voxel or image.

        Args:
            axis: 0 = across images (default, returns BrainData),
                1 = within images (returns array). Ignored when
                ``spatial_scale='roi'``.
            spatial_scale: ``'whole_brain'`` (default) or ``'roi'`` (paints
                each voxel with its parcel's median per image).
            roi_mask: Atlas image for ``spatial_scale='roi'``.

        Returns:
            float/np.array/BrainData: Median values.
        """
        if spatial_scale == "roi":
            from .analysis import reduce_per_roi

            return reduce_per_roi(self, np.median, roi_mask=roi_mask)
        from .utils import apply_func

        return apply_func(self, np.median, axis)

    def multivariate_similarity(self, images, method="ols"):
        """Predict a BrainData spatial distribution from a linear combination.

        The predictors may be other BrainData instances or nibabel images.

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
        method="glass",
        upper=None,
        lower=None,
        threshold=None,
        view="z",
        cut_coords=None,
        cmap=None,
        bg_img=None,
        ax=None,
        figsize=(8, 6),
        title=None,
        colorbar=True,
        save=None,
        stat="mean",
        limit=3,
        **kwargs,
    ):
        """Plot BrainData instance using nilearn visualization or matplotlib.

        Args:
            method (str): Visualization type: 'glass', 'slices', 'timeseries', 'histogram'
            upper (str/float, optional): Upper threshold.
            lower (str/float, optional): Lower threshold.
            threshold (float, optional): Convenience parameter for thresholding.
            view (str): For ``method="slices"``, any non-empty combination of
                ``"x"``, ``"y"``, ``"z"`` (e.g. ``"xyz"``, ``"xz"``, ``"y"``).
                Default: ``"z"``.
            cut_coords (list or dict, optional): Cut coordinates for
                multi-slice views. Takes precedence over ``view``-based
                defaults. Either a list matching ``len(view)`` or a dict
                keyed by axis letter.
            cmap (str, optional): Colormap name.
            bg_img (str/nibabel image, optional): Background image.
            ax (matplotlib.axes.Axes, optional): Matplotlib axis.
            figsize (tuple, optional): default figure size if no axis (8, 6)
            title (str, optional): Plot title.
            colorbar (bool): Whether to show colorbar. Default: True.
            save (str, optional): Path to save figure(s).
            stat (str): Statistic for timeseries plots. Default: 'mean'.
            limit (int): Maximum number of images to render when this
                BrainData contains multiple maps and ``method`` is
                ``"glass"`` or ``"slices"``. Default: 3. Warns when more
                images exist than ``limit``.
            **kwargs: Additional arguments passed to nilearn plot functions.

        Returns:
            matplotlib.figure.Figure or list[matplotlib.figure.Figure]: A
            single figure for single-image data; a list of figures for
            multi-image data with ``method`` in ``{"glass", "slices"}``
            (one per image for glass; one per image-and-view pair for
            slices).
        """
        from .plotting import plot_brain

        return plot_brain(
            self,
            method=method,
            upper=upper,
            lower=lower,
            threshold=threshold,
            view=view,
            cut_coords=cut_coords,
            cmap=cmap,
            bg_img=bg_img,
            ax=ax,
            figsize=figsize,
            title=title,
            colorbar=colorbar,
            save=save,
            stat=stat,
            limit=limit,
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
        transparency="auto",
        colorbar=True,
        colorbar_orientation="horizontal",
        figsize=(12, 6),
        title=None,
        radius_mm=3.0,
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
            transparency (BrainData, Nifti1Image, str, or "auto"): Binary mask
                used to render vertices outside the mask as transparent.
                ``"auto"`` (default) uses the instance's ``.mask``; pass
                ``None`` to disable masking.
            colorbar (bool): Show colorbar. Default: True.
            colorbar_orientation (str): 'horizontal' or 'vertical'. Default: 'horizontal'.
            figsize (tuple): Figure size as (width, height). Default: (12, 6).
            title (str, optional): Figure title.
            radius_mm (float): Sampling radius in mm. Default: 3.0.
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
            transparency=transparency,
            colorbar=colorbar,
            colorbar_orientation=colorbar_orientation,
            figsize=figsize,
            title=title,
            radius_mm=radius_mm,
            interpolation=interpolation,
            axes=axes,
            save=save,
        )

    def plot_surf(
        self,
        *,
        hemi="both",
        view="montage",
        surface="pial",
        template="fsaverage5",
        threshold=None,
        cmap="RdBu_r",
        vmin=None,
        vmax=None,
        transparency="auto",
        bg_on_data=False,
        colorbar=True,
        colorbar_orientation="horizontal",
        figsize=(10, 8),
        title=None,
        radius_mm=3.0,
        interpolation="linear",
        zoom=1.2,
        axes=None,
        save=None,
    ):
        """Render this BrainData on fsaverage surfaces as a tight 2×2 montage.

        Facade over `plot_surf`. See that function's
        docstring for the full argument reference. Notable defaults:
        ``surface="pial"``, ``zoom=1.2``, ``transparency="auto"`` (uses
        this instance's ``.mask``).

        Returns:
            matplotlib.figure.Figure
        """
        from nltools.plotting import plot_surf
        from .plotting import _require_standard_space

        _require_standard_space(
            self,
            "plot_surf",
            remedy=(
                "Surface projection samples vol_to_surf at fsaverage "
                "(MNI-aligned) coordinates and produces garbage on "
                "native-space data. Use bd.plot(method='slices', "
                "bg_img=<your subject anatomical>) instead, or call "
                "bd.resample() to bring data into standard space first."
            ),
        )

        return plot_surf(
            self,
            hemi=hemi,
            view=view,
            surface=surface,
            template=template,
            threshold=threshold,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transparency=transparency,
            bg_on_data=bg_on_data,
            colorbar=colorbar,
            colorbar_orientation=colorbar_orientation,
            figsize=figsize,
            title=title,
            radius_mm=radius_mm,
            interpolation=interpolation,
            zoom=zoom,
            axes=axes,
            save=save,
        )

    def iplot(
        self,
        *,
        view: str = "ortho",
        threshold: float | None = None,
        lower: float | None = None,
        upper: float | None = None,
        cmap: str = "warm",
        bg_img: "str | bool | None" = None,
        atlas: "str | Atlas | None" = None,
        opacity: float = 1.0,
        outline: float = 0.0,
        colorbar: bool = True,
        controls: bool = True,
        **kwargs,
    ):
        """Interactive WebGL brain viewer powered by niivue (`ipyniivue`).

        Renders inline in a live kernel (Jupyter, marimo) with live windowing
        (right-drag to set the threshold/contrast), slice scrolling, native 4D
        frame scrubbing, true 3D rendering, a stat-map colorbar, and optional
        nltools-atlas overlays. Static-built docs are not supported; use
        `plot` there.

        By default (``controls=True``) the return value is an
        `ipywidgets.VBox` stacking a threshold slider above the viewer; access
        the underlying `NiiVue` via its ``.viewer`` attribute and the slider
        via ``.threshold_slider``. Pass ``controls=False`` to get the bare
        `NiiVue` widget instead.

        Thresholding is a divergent magnitude window: ``cal_min`` is the
        display floor (sub-floor voxels render transparent), ``cal_max`` the
        saturation point, with the positive limb using ``cmap`` and the
        negative limb its mirrored partner. Precedence: ``lower``/``upper``
        win; otherwise ``threshold`` sets the floor (ceiling auto);
        otherwise the window is fully auto.

        Args:
            view: ``"ortho"`` (default), ``"axial"``, ``"coronal"``,
                ``"sagittal"``, or ``"render"`` (3D volume render).
                ``"surface"`` is no longer supported — use ``"render"`` or
                `plot_flatmap` / `plot_surf`.
            threshold: Convenience symmetric magnitude floor (→ ``cal_min``).
            lower: Window floor (→ ``cal_min``). Overrides ``threshold``.
            upper: Window ceiling (→ ``cal_max``). Overrides ``threshold``.
            cmap: niivue colormap for the positive limb (default ``"warm"``).
                Common matplotlib names are auto-mapped with a warning.
            bg_img: ``None``/``True`` auto-loads the matching MNI template
                when the data is in standard space (else none); ``False``
                disables the background; a path string uses that image.
            atlas: Atlas overlay — a registry name (e.g. ``"aal"``), a
                loaded `Atlas`, or ``None``. Deterministic atlases
                only; probabilistic atlases raise.
            opacity: Stat-map (and filled-atlas) opacity in ``0..1``.
            outline: ``> 0`` draws atlas region boundaries of that width
                (stat map stays visible); ``0`` draws filled regions.
            colorbar: Show the stat-map colorbar (default ``True``). An
                explicit ``is_colorbar`` kwarg overrides this.
            controls: Wrap the viewer in a `VBox` with an interactive
                threshold slider (default ``True``). ``False`` returns the
                bare `NiiVue`. Requires the ``ipywidgets`` optional
                dependency when ``True``.
            **kwargs: Forwarded verbatim to ``ipyniivue.NiiVue(**kwargs)``
                (e.g. ``height``, ConfigOptions like ``is_colorbar``).

        Returns:
            ipywidgets.VBox with ``.viewer`` (the `NiiVue`) and
            ``.threshold_slider`` when ``controls=True``; otherwise the bare
            ``ipyniivue.NiiVue`` widget.
        """
        from .viewer import build_controls, build_viewer

        if lower is not None or upper is not None:
            cal_min, cal_max = lower, upper
        elif threshold is not None:
            cal_min, cal_max = threshold, None
        else:
            cal_min, cal_max = None, None

        nv = build_viewer(
            self,
            view=view,
            cal_min=cal_min,
            cal_max=cal_max,
            cmap=cmap,
            atlas=atlas,
            bg_img=bg_img,
            opacity=opacity,
            outline=outline,
            colorbar=colorbar,
            niivue_opts=kwargs,
        )
        if controls:
            try:
                import ipywidgets  # noqa: F401
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "iplot(controls=True) requires the optional dependency "
                    "`ipywidgets`. Install it with `uv add ipywidgets` (or "
                    "`pip install 'nltools[interactive_plots]'`), or call "
                    "iplot(controls=False) for the bare viewer."
                ) from exc
            return build_controls(nv, self, cal_min=cal_min, cal_max=cal_max)
        return nv

    @coalesced_gc()
    def predict(
        self,
        *,
        y: "np.ndarray | None" = None,
        X: "np.ndarray | None" = None,
        spatial_scale: str = "whole_brain",
        model="svm",
        cv: int = 5,
        standardize: bool = True,
        reduce: "str | None" = None,
        n_components: "int | None" = None,
        scoring: str = "auto",
        groups: "np.ndarray | None" = None,
        roi_mask=None,
        radius_mm: float = 10.0,
        inplace: bool = False,
        n_jobs: int = 1,
        random_state: "int | None" = None,
        progress_bar: bool = False,
    ):
        """Predict voxel timeseries (encoding) or decode labels (MVPA).

        Dispatched by which of ``X`` or ``y`` is provided:

        1. **Timeseries prediction** (``X`` provided): use a fitted ridge /
           GLM encoding model on ``self`` to predict voxel responses.
           Returns a fresh ``BrainData`` whose ``.data`` holds the predicted
           timeseries (composes directly with ``.plot()``, ``.standardize()``
           etc.). ``inplace`` has no effect in this mode.
        2. **MVPA decoding** (``y`` provided): train a classifier or
           regressor with cross-validation. Returns a `Predict`
           dataclass. Spatial fields (``weight_map``, ``fold_weight_maps``,
           ``final_weight_map``, ``accuracy_map``) are `BrainData`
           objects so ``result.weight_map.plot()`` works directly. Drop down
           to numpy via ``result.weight_map.data``.

        Field shapes by ``spatial_scale=``:

        - **whole_brain**: ``predictions`` (n_samples,) OOF predictions,
          ``scores`` (n_folds,), ``mean_score`` float, ``std_score`` float,
          ``weight_map`` BrainData (``coef_`` from one fit on the **full**
          ``(X, y)`` — the publishable map), ``fold_weight_maps`` BrainData
          (n_folds, n_voxels) for stability analysis, ``estimator`` the
          fitted all-data sklearn estimator (use for ``.predict()`` on new
          data).
        - **roi**: ``scores`` (n_folds, n_rois), ``mean_score`` (n_rois,),
          ``std_score`` (n_rois,), ``roi_labels`` (n_rois,) atlas IDs in
          matching order, ``accuracy_map`` / ``weight_map`` /
          ``fold_weight_maps`` BrainData (per-parcel coefs reassembled to
          voxel space; voxels outside the atlas = NaN), ``estimator`` dict
          keyed by atlas label.
        - **searchlight**: ``accuracy_map`` BrainData.

        With ``inplace=True``, fields are attached to ``self`` with a
        ``predict_`` prefix (e.g. ``self.predict_weight_map``,
        ``self.predict_accuracy_map``), mirroring ``bd.fit()``'s
        ``glm_*`` / ``ridge_*`` naming.

        Why ``weight_map`` is the all-data refit, not the CV mean:
        the mean of K per-fold ``coef_`` vectors doesn't correspond to
        any actual fitted estimator (each fold saw a different subset).
        The all-data refit is a single legitimate model with all the
        information used. CV gives the honest *score*; the refit gives
        the publishable *map*. The CV-mean is one line away if you want
        it: ``result.fold_weight_maps.data.mean(axis=0)``.

        Args:
            y (array-like, optional): Labels (classification) or continuous
                targets (regression), shape ``(n_samples,)``. Triggers MVPA mode.
            X (array-like, optional): Features for timeseries prediction,
                shape ``(n_samples, n_features)``. Triggers encoding mode.
            spatial_scale (str): MVPA dispatch — ``'whole_brain'``,
                ``'searchlight'``, or ``'roi'``.
            model (str or sklearn estimator): Algorithm. String shortcuts:

                - Classification: ``'svm'`` (LinearSVC), ``'logistic'``,
                  ``'lda'``, ``'ridge_classifier'``.
                - Regression: ``'ridge'``, ``'lasso'``, ``'svr'``.

                Or pass any sklearn estimator / Pipeline (e.g.,
                ``make_pipeline(StandardScaler(), SelectKBest(k=500), LinearSVC())``).
                When ``model`` is a sklearn ``Pipeline``, ``standardize`` is
                auto-defaulted to ``False`` (with a warning) so we don't wrap
                another StandardScaler around your pipeline. Pass
                ``standardize=True`` explicitly to override.
            cv (int or sklearn CV splitter): ``int`` → KFold (regression) or
                StratifiedKFold (classification); pass a splitter for custom
                schemes (e.g., ``GroupKFold``).
            standardize (bool): Z-score features per fold before fitting.
                Default ``True``. Auto-flipped to ``False`` when ``model`` is
                a sklearn ``Pipeline`` (see ``model`` above).
            reduce (str, optional): Per-fold dimensionality reduction.
                Currently only ``'pca'`` supported. Default ``None``. Weight
                maps are back-projected through PCA to voxel space.
            n_components (int, optional): PCA components when ``reduce='pca'``.
            scoring (str): Sklearn scoring string. Default ``'auto'`` →
                ``'accuracy'`` if classifier, ``'r2'`` if regressor.
            groups (array-like, optional): Group labels for CV splitters
                that need them (e.g., leave-one-run-out).
            roi_mask (Nifti1Image or path-like, optional): Atlas image for
                ``spatial_scale='roi'``.
            radius_mm (float): Searchlight radius in mm. Default ``10.0``.
            inplace (bool): If ``True``, populate result fields as
                ``predict_*`` attributes on ``self`` and return ``self``.
                Default ``False`` returns a fresh `Predict`.
            n_jobs (int): Parallel jobs for searchlight / ROI. Default ``1``;
                searchlight on a real brain at higher ``n_jobs`` can be
                memory-heavy.
            random_state (int, optional): Seed for the shuffled fold splitter
                when ``cv`` is an int (MVPA mode). Default ``None`` (unseeded
                shuffle each call). Ignored when ``cv`` is a splitter object —
                set its own ``random_state`` instead.
            progress_bar (bool): Show progress bar for searchlight / ROI.

        Returns:
            Predict | BrainData: ``Predict`` dataclass when ``inplace=False``;
                ``self`` (mutated, with ``predict_*`` attrs) when ``inplace=True``.

        Examples:
            >>> result = brain.predict(y=labels, spatial_scale='whole_brain', cv=5)
            >>> result.weight_map.plot()       # publishable map (all-data fit)
            >>> result.mean_score              # honest CV-derived accuracy
            >>> new_pred = result.estimator.predict(new_X)  # apply to new data

            >>> result = brain.predict(y=labels, spatial_scale='searchlight',
            ...                        radius_mm=8.0, n_jobs=4)
            >>> result.accuracy_map.plot()

            >>> result = brain.predict(y=labels, spatial_scale='roi', roi_mask=atlas)
            >>> top = result.roi_labels[result.mean_score.argsort()[::-1][:10]]
            >>> result.accuracy_map.plot()  # brain-space view of the same map

            Custom sklearn pipeline as model — standardize auto-defaults to
            False because we detect the Pipeline:

            ```python
            from sklearn.feature_selection import SelectKBest
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import LinearSVC
            pipe = make_pipeline(StandardScaler(), SelectKBest(k=500),
                                 LinearSVC())
            result = brain.predict(y=labels, model=pipe)
            ```
        """
        from .prediction import predict

        return predict(
            self,
            y=y,
            X=X,
            spatial_scale=spatial_scale,
            model=model,
            cv=cv,
            standardize=standardize,
            reduce=reduce,
            n_components=n_components,
            scoring=scoring,
            groups=groups,
            roi_mask=roi_mask,
            radius_mm=radius_mm,
            inplace=inplace,
            n_jobs=n_jobs,
            random_state=random_state,
            progress_bar=progress_bar,
        )

    def r_to_z(self):
        """Apply Fisher's r-to-z transformation to each data element."""
        from .analysis import r_to_z

        return r_to_z(self)

    @coalesced_gc()
    def regions(
        self,
        min_region_size=1350,
        method="local_regions",
        smoothing_fwhm=6,
        is_mask=False,
    ):
        """Extract brain connected regions into separate regions.

        Args:
            min_region_size (int): Minimum volume in mm3 for a region to be kept.
            method (str): Type of extraction method
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
            method=method,
            smoothing_fwhm=smoothing_fwhm,
            is_mask=is_mask,
        )

    def regress(self, design_matrix=None, method="ols", mode=None):
        """Deprecated: Use fit(model='glm', X=design_matrix) instead.

        **Deprecated:** Since version 0.6.0. Use `fit` with ``model='glm'``
        instead.
        """
        from .modeling import regress

        return regress(
            self,
            design_matrix=design_matrix,
            method=method,
            mode=mode,
        )

    def predict_multi(self, *args, **kwargs):
        """Deprecated: removed in v0.6.0; will return in a future Model class.

        Per the v0.6 migration guide, the multi-method MVPA wrapper has
        been removed. Use `predict` for whole-brain MVPA, or compose
        sklearn estimators directly via the new Model API.
        """
        raise NotImplementedError(
            "BrainData.predict_multi() is deprecated and was removed in "
            "v0.6.0. It will return in a future Model class. For now, use "
            ".predict(y=...) for whole-brain MVPA, or compose sklearn "
            "estimators directly. See the migration guide."
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

    @coalesced_gc()
    def similarity(self, image, method="correlation"):
        """Calculate similarity to a single BrainData or nibabel image.

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

    def standardize(self, axis=0, method="center", verbose=True):
        """Standardize BrainData() instance.

        Args:
            axis (int): 0 standardizes each voxel across observations (default).
                1 standardizes each observation across voxels.
            method (str): 'center' subtracts the mean (default).
                'zscore' subtracts the mean and divides by standard deviation.
            verbose (bool): If False, suppress sklearn numerical warnings that
                occur when voxels have near-zero variance. Default: True.

        Returns:
            BrainData: Standardized BrainData instance.
        """
        from .analysis import standardize

        return standardize(self, axis=axis, method=method, verbose=verbose)

    def std(self, axis=0, *, spatial_scale: str = "whole_brain", roi_mask=None):
        """Get standard deviation of each voxel or image.

        Args:
            axis: 0 = across images (default, returns BrainData),
                1 = within images (returns array). Ignored when
                ``spatial_scale='roi'``.
            spatial_scale: ``'whole_brain'`` (default) or ``'roi'`` (paints
                each voxel with its parcel's std per image).
            roi_mask: Atlas image for ``spatial_scale='roi'``.

        Returns:
            float/np.array/BrainData: Standard deviation values.
        """
        if spatial_scale == "roi":
            from .analysis import reduce_per_roi

            return reduce_per_roi(self, np.std, roi_mask=roi_mask)
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

    @coalesced_gc()
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

    def cluster_report(
        self,
        *,
        stat_threshold: float | None = 3.0,
        cluster_threshold: int = 10,
        two_sided: bool = True,
        min_distance: float = 8.0,
        atlas: str | Sequence[str] | None = None,
        prob_threshold: float = 5.0,
    ) -> "ClusterReport":
        """Generate a cluster report with anatomical labels.

        Identifies surviving clusters in the stat map (after voxel + extent
        thresholding), reports peak coordinates and sub-peaks, and labels
        each peak/cluster against one or more atlases.

        Args:
            stat_threshold: Voxel-level threshold (e.g. z- or t-cutoff).
                ``None`` treats ``self`` as already thresholded.
            cluster_threshold: Minimum cluster size in voxels.
            two_sided: Report negative clusters separately.
            min_distance: Minimum mm between sub-peaks within a cluster.
            atlas: Atlas name or list of names (see `list_atlases`).
                Defaults to ``("harvard_oxford", "aal", "schaefer_200")``.
            prob_threshold: Drop probabilistic-atlas regions below this %.

        Returns:
            `ClusterReport` with ``peaks``,
            ``clusters`` (polars DataFrames), and ``stat_img`` (BrainData).
        """
        from nltools.data.atlases import (
            DEFAULT_ATLASES,
            ClusterReport,
            cluster_report_data,
        )

        peaks, clusters, thr = cluster_report_data(
            self,
            stat_threshold=stat_threshold,
            cluster_threshold=cluster_threshold,
            two_sided=two_sided,
            min_distance=min_distance,
            atlas=DEFAULT_ATLASES if atlas is None else atlas,
            prob_threshold=prob_threshold,
        )
        return ClusterReport(peaks=peaks, clusters=clusters, stat_img=thr)

    def transform_pairwise(self):
        """Transform data into pairwise comparisons.

        Returns:
            BrainData: BrainData instance transformed into pairwise comparisons
        """
        from .analysis import transform_pairwise_data

        return transform_pairwise_data(self)

    def ttest(
        self,
        popmean=0.0,
        permutation=False,
        n_permute=5000,
        tail=2,
        return_null=False,
        n_jobs=-1,
        random_state=None,
    ):
        """One-sample voxelwise t-test across images (axis 0).

        Tests whether the per-voxel mean across images differs from
        ``popmean``. Operates on a stack of images (e.g. subject-level
        contrast maps) with shape ``(n_samples, n_voxels)``.

        Args:
            popmean: Population mean to test against. Default 0.0.
            permutation: If True, use sign-flip permutation test via
                `one_sample_permutation_test`.
            n_permute: Number of permutations (used only when
                ``permutation=True``). Default 5000.
            tail: Tail of the test (1 or 2). Default 2.
            return_null: If True, also return the null distribution.
                Default False.
            n_jobs: Number of parallel jobs. Default -1 (all cores).
            random_state: Random seed for reproducibility.

        Returns:
            dict with four BrainData keys:

                - ``"mean"``: voxelwise mean across images (effect size).
                - ``"t"``: parametric one-sample t-statistic.
                - ``"z"``: signed z-score, ``sign(t) * norm.isf(p/2)`` —
                  matches nilearn's ``output_type='z_score'``.
                - ``"p"``: parametric p-value, or empirical p when
                  ``permutation=True``.

            The effect size is always returned alongside the inferential maps
            so group-level code never has to recompute the mean.

        Raises:
            ValueError: If this BrainData contains fewer than 2 images.

        Examples:
            >>> # Stack of subject-level contrast maps
            >>> result = contrast_maps.ttest()
            >>> sig = result["p"].data < 0.05
            >>> effect = result["mean"]       # for reporting magnitude
            >>> z_map = result["z"]           # for nilearn-style thresholding

            >>> # Permutation-based p-values; still reports t/z/mean
            >>> result = contrast_maps.ttest(permutation=True, n_permute=5000)
        """
        from .modeling import ttest

        return ttest(
            self,
            popmean=popmean,
            permutation=permutation,
            n_permute=n_permute,
            tail=tail,
            return_null=return_null,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def ttest2(self, other, equal_var=True):
        """Two-sample voxelwise t-test between two BrainData stacks.

        Args:
            other: BrainData to compare against. Must have the same
                number of voxels.
            equal_var: If True (default), standard two-sample t-test.
                If False, Welch's t-test.

        Returns:
            dict: ``{"t": BrainData, "p": BrainData}``.

        Raises:
            ValueError: If the two BrainData objects have different
                ``n_voxels``.
        """
        from .modeling import ttest2

        return ttest2(self, other, equal_var=equal_var)

    def upload_neurovault(
        self,
        access_token=None,
        collection_name=None,
        collection_id=None,
        img_type=None,
        img_modality=None,
        **kwargs,
    ):
        """Upload BrainData images and metadata to NeuroVault.

        Adds any columns in ``self.X`` to image metadata. The index is used as
        the image name.

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
