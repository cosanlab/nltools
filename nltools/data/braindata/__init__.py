"""Represent brain image data with the BrainData class."""

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from nltools.data.atlases import Atlas, ClusterReport

from nltools.utils import coalesced_gc

from .utils import check_brain_data

__all__ = ["BrainData"]


class BrainData:
    """Represent neuroimaging data as vectors instead of three-dimensional matrices.

    Each image is flattened to its in-mask voxels, so a stack of images is a 2D
    ``(n_images, n_voxels)`` array. This representation makes it easier to perform
    data manipulation and analyses.

    Args:
        data (None | BrainData | list | str | Path | Nifti1Image | np.ndarray):
            Neuroimaging data. Accepts ``None`` (an empty BrainData), another
            BrainData, a list of BrainData objects or file paths, a file path to
            ``.nii``/``.nii.gz``/``.h5``/``.hdf5``, a nibabel ``Nifti1Image``, a URL
            to download from, or a numpy array (1D ``(n_voxels,)`` for a single image
            or 2D ``(n_images, n_voxels)`` for a stack). Array input requires
            ``mask``, whose in-mask voxel count must match the array's last axis.
        mask (None | Nifti1Image | str | Path): Brain mask. ``None`` uses the MNI
            template; otherwise a nibabel ``Nifti1Image``, a file path to a mask
            file, or a template name string like ``'2mm-MNI152-2009c'`` (version:
            ``'fsl'`` for default/, ``'a'`` for nilearn/, ``'c'`` for fmriprep/).
        masker (nilearn masker | None): nilearn masker object (e.g. ROI or
            searchlight extractor). Default ``None`` loads data as voxels.
        Y (DataFrame | np.ndarray | str | None): Optional per-image target/label
            values, stored as a polars DataFrame (``.Y``). Default ``None``. If
            ``data`` is a BrainData with a ``.Y``, that value is inherited when this
            is ``None``.
        X (DataFrame | np.ndarray | str | None): Optional per-image design/feature
            values, stored as a polars DataFrame (``.X``). Default ``None``. If
            ``data`` is a BrainData with an ``.X``, that value is inherited when this
            is ``None``.
        h5_compression (str): Compression filter used when writing HDF5
            (``.h5``/``.hdf5``) output. Default ``'gzip'``.
        verbose (bool): Emit informational messages during loading and other
            operations. Default ``False``.
        resample (bool): Whether to automatically resample data to mask space.
            If ``True`` (default), data is resampled to match the mask's spatial
            characteristics. If ``False``, data must already be in mask space.
        interpolation (str): Interpolation method for resampling. ``'auto'``
            (default) detects based on data type — ``'nearest'`` for discrete data
            like atlases/masks and ``'continuous'`` for stat maps; ``'nearest'``
            (nearest-neighbor, preserves discrete values), ``'linear'`` (linear
            interpolation), or ``'continuous'`` (higher-order spline, use for stat
            maps).

    Attributes:
        data (np.ndarray): In-mask voxel values, shape ``(n_voxels,)`` for a single
            image or ``(n_images, n_voxels)`` for a stack.
        mask (Nifti1Image): The brain mask every image is flattened against.
        masker (nilearn masker | None): Masker used to extract data, or ``None``
            when data are plain voxels.
        design_matrix (DesignMatrix | None): Design matrix attached by
            ``fit(model='glm', ...)``; ``None`` until a GLM is fit.
        verbose (bool): Whether informational messages are emitted.
        X (pl.DataFrame): Design matrix / per-image covariates (possibly empty).
        Y (pl.DataFrame): Per-image targets (possibly empty).
        dtype (np.dtype): Data type of ``data``.
        is_empty (bool): Whether ``data`` holds no elements.
        shape (tuple[int, ...]): Images-by-voxels shape of ``data``.
        size (int): Total number of elements in ``data`` (numpy convention).
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

    def __copy__(self):
        """Create an independent snapshot of all data and fitted state."""
        from .utils import _copy_complete

        return _copy_complete(self)

    def __deepcopy__(self, memo):
        """Create an independent snapshot of all data and fitted state."""
        from .utils import _copy_complete

        return _copy_complete(self, memo)

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
        elif hasattr(self.mask, "dataobj") and hasattr(other.mask, "dataobj"):
            eq_mask = (
                self.mask.shape == other.mask.shape
                and np.array_equal(self.mask.affine, other.mask.affine)
                and np.array_equal(
                    np.asanyarray(self.mask.dataobj),
                    np.asanyarray(other.mask.dataobj),
                )
            )
        else:
            eq_mask = self.mask == other.mask

        return eq_data and eq_X and eq_Y and eq_mask

    def __getitem__(self, index):
        from .utils import _result_from_selection

        return _result_from_selection(self, index)

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
        from .utils import _clear_fit_state

        if not isinstance(value, BrainData):
            raise ValueError(
                "Make sure the value you are trying to set is a BrainData() instance."
            )
        new_data = self.data.copy()
        new_data[index, :] = value.data
        new_y = None
        if not value.Y.is_empty():
            if self.Y.is_empty():
                raise ValueError("Cannot set Y values: self.Y is empty.")
            arr = self.Y.to_numpy()
            arr[index] = value.Y.to_numpy()
            new_y = pl.DataFrame(arr, schema=self.Y.columns)
        new_X = None
        if not value.X.is_empty():
            if self.X.is_empty():
                raise ValueError("Cannot set X values: self.X is empty.")
            if self.X.shape[1] != value.X.shape[1]:
                raise ValueError("Make sure self.X is the same size as value.X.")
            arr = self.X.to_numpy()
            arr[index] = value.X.to_numpy()
            new_X = pl.DataFrame(arr, schema=self.X.columns)

        _clear_fit_state(self)
        self.data = new_data
        if new_y is not None:
            self.Y = new_y
        if new_X is not None:
            self.X = new_X

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
            target (BrainData): Object to align to.
            method (str): Alignment method: ``'probabilistic_srm'``,
                ``'deterministic_srm'``, or ``'procrustes'``. Default ``'procrustes'``.
            axis (int): Axis to align on. Default 0.
            spatial_scale (str): ``'whole_brain'`` (default), ``'roi'``, or
                ``'searchlight'``. ``'roi'`` is supported (per-parcel
                transforms + reassembly, requires `roi_mask`). ``'searchlight'``
                is not yet implemented (overlapping spheres have no canonical
                per-voxel transform).
            roi_mask (BrainData | Nifti1Image | str | Path | None): Atlas image
                used when ``spatial_scale='roi'``.
            radius_mm (float): Reserved for ``spatial_scale='searchlight'``.

        Returns:
            dict: A dictionary containing the transformed object, transformation
                matrix, and the shared response matrix.

        Examples:
            ```python
            # Hyperalign using procrustes transform
            out = data.align(target, method='procrustes')

            # Align using shared response model
            out = data.align(target, method='probabilistic_srm')
            ```
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

    def append(  # nosemgrep: kwargs-internal-forwarding  # forwards to polars.concat
        self, data, ignore_attrs=False, **kwargs
    ):
        """Append data to BrainData instance.

        Args:
            data (BrainData): BrainData instance to append.
            ignore_attrs (bool): Clear both X and Y on the result when True.
                Otherwise, each metadata frame must be empty on both inputs or
                have compatible columns on both inputs. Default False.
            **kwargs (dict): Currently ignored. X/Y are concatenated with polars'
                ``pl.concat(..., how="vertical_relaxed")``, which takes no
                caller-supplied options.

        Returns:
            BrainData: Independently owned data with concatenated row metadata.

        Raises:
            ValueError: Metadata is present on only one input or has incompatible columns.
        """
        from .utils import _result_from_rows
        from .validation import validate_append_shapes
        import polars as pl

        data = check_brain_data(data)
        if self.is_empty:
            return _result_from_rows(
                data,
                data.data,
                X=None if ignore_attrs else data.X,
                Y=None if ignore_attrs else data.Y,
            )
        validate_append_shapes(self.shape, data.shape)
        frames = []
        for name in ("X", "Y"):
            left, right = getattr(self, name), getattr(data, name)
            if ignore_attrs or (left.is_empty() and right.is_empty()):
                frames.append(None)
            elif left.is_empty() or right.is_empty() or left.columns != right.columns:
                raise ValueError(
                    f"append requires compatible {name} metadata on both operands"
                )
            else:
                try:
                    frames.append(pl.concat([left, right], how="vertical_relaxed"))
                except pl.exceptions.SchemaError as error:
                    raise ValueError(
                        f"append requires compatible {name} metadata schemas"
                    ) from error
        return _result_from_rows(
            self, np.vstack([self.data, data.data]), X=frames[0], Y=frames[1]
        )

    @coalesced_gc()
    def apply_mask(self, mask, resample_mask_to_brain=False):
        """Mask BrainData instance using nilearn functionality.

        Note target data will be resampled into the same space as the mask. If you would like the mask
        resampled into the BrainData space, then set resample_mask_to_brain=True.

        Args:
            mask (BrainData | Nifti1Image): Mask to apply to BrainData object.
            resample_mask_to_brain (bool): Resample the mask to brain space before
                applying it. Default False.

        Returns:
            BrainData: Masked BrainData object.
        """
        from .analysis import apply_mask

        return apply_mask(self, mask, resample_mask_to_brain=resample_mask_to_brain)

    def astype(self, dtype):
        """Cast BrainData.data as type.

        Args:
            dtype (np.dtype | type | str): Datatype to convert to.

        Returns:
            BrainData: BrainData instance with new datatype.
        """
        from .utils import _result_from_array

        out = _result_from_array(self, self.data.astype(dtype), rows="preserve")
        return out

    def bootstrap(
        self,
        stat,
        *,
        n_samples=5000,
        save_boots=False,
        percentiles=(2.5, 97.5),
        X_test=None,
        device="cpu",
        max_gpu_memory_gb=None,
        tail=2,
        n_jobs=-1,
        random_state=None,
        progress_bar: bool = False,
    ):
        """Bootstrap statistics using efficient online algorithms.

        Uses memory-efficient bootstrap infrastructure with CPU parallelization or GPU acceleration.
        Supports simple aggregation statistics and fitted model statistics (Ridge).

        Args:
            stat (str): Statistic to bootstrap. Simple stats: ``'mean'``,
                ``'median'``, ``'std'``, ``'sum'``, ``'min'``, ``'max'``. Model
                stats: ``'weights'`` (requires a fitted Ridge model) or
                ``'predict'`` (requires a fitted Ridge model plus ``X_test``).
            n_samples (int): Number of bootstrap iterations. Default 5000.
            save_boots (bool): If True, store all bootstrap samples. Default False.
            percentiles (tuple[float, float]): Percentiles for confidence
                intervals. Default ``(2.5, 97.5)``.
            X_test (np.ndarray | None): Test features for the ``'predict'`` bootstrap.
            device (str): Compute device for the Ridge bootstrap: ``'cpu'``
                (default), ``'gpu'`` (PyTorch on CUDA/MPS if available), or
                ``'auto'`` (GPU if present, else CPU). Ignored for simple stats.
            max_gpu_memory_gb (float | None): Explicit GPU memory budget in GB
                when device is ``'gpu'`` or ``'auto'``. ``None`` (default) measures
                the device.
            tail (int | str): ``2``/``'two'`` for two-tailed p-values (default),
                ``1``/``'one'`` for one-tailed.
            n_jobs (int): Number of CPU cores for parallelization. -1 (default)
                means all CPUs.
            random_state (int | None): Random seed for reproducibility.
            progress_bar (bool): If True, show a progress bar. Default False.

        Returns:
            BrainData | dict: For simple stats, a BrainData holding the bootstrap
                mean. For model stats, a dict of BrainData objects keyed ``'mean'``,
                ``'std'``, ``'Z'``, ``'p'``, ``'ci_lower'``, ``'ci_upper'``. With
                ``save_boots=True`` the dict also carries a ``'samples'`` key
                holding every bootstrap sample.

        Examples:
            ```python
            boot = brain.bootstrap(stat='mean', n_samples=1000)
            brain.fit(X=dm, model='ridge', alpha=1.0)
            boot = brain.bootstrap(stat='weights', n_samples=1000)
            ```
        """
        from .bootstrap import bootstrap

        return bootstrap(
            self,
            stat,
            n_samples=n_samples,
            save_boots=save_boots,
            percentiles=percentiles,
            X_test=X_test,
            device=device,
            max_gpu_memory_gb=max_gpu_memory_gb,
            tail=tail,
            n_jobs=n_jobs,
            random_state=random_state,
            progress_bar=progress_bar,
        )

    def compute_contrasts(self, contrasts, statistic="t"):
        """Compute contrasts from fitted GLM results.

        This method computes contrasts as linear combinations of the GLM beta coefficients.
        Must be called after ``fit(model='glm', X=design_matrix)`` has been run.

        A contrast can be given three ways. A **string** names design-matrix columns
        with optional coefficients, e.g. ``"conditionA - conditionB"`` or
        ``"2*conditionA - conditionB - conditionC"``. A **numeric vector** lists one
        weight per regressor, e.g. ``[1, -1, 0, 0]`` for a 4-regressor model. A
        **dict** maps contrast names to either form, e.g.
        ``{"main_effect": "conditionA - conditionB", "interaction": [1, -1, -1, 1]}``.

        Args:
            contrasts (str | array-like | dict): The contrast(s) to compute — a
                string, a numeric vector, or a dict of named contrasts (see above).
            statistic (str): Which statistic to return per contrast. One of ``"t"``
                (default, t-statistic map), ``"z"`` (z-score), ``"p"`` (p-value),
                ``"beta"`` / ``"effect_size"`` (effect-size β map — use this when
                feeding a second-level group analysis), or ``"all"`` (a bundle dict
                ``{"beta", "t", "z", "p", "se"}`` of maps for one contrast).

        Returns:
            BrainData | dict: A single contrast with a scalar ``statistic`` returns a
                ``BrainData`` map; with ``statistic="all"`` it returns a flat dict keyed
                by ``"beta"``/``"t"``/``"z"``/``"p"``/``"se"``. A dict of contrasts
                returns a dict keyed by contrast name (nested under the five keys when
                ``statistic="all"``).

        Raises:
            RuntimeError: If ``fit(model='glm')`` hasn't been called yet.
            ValueError: If a contrast vector's length doesn't match the number of
                regressors, or a column named in a string contrast is not in the
                design matrix.

        Examples:
            ```python
            brain.fit(model='glm', X=design_matrix)
            contrast1 = brain.compute_contrasts([0, 1, -1])
            contrast2 = brain.compute_contrasts("conditionA - conditionB")
            results = brain.compute_contrasts({
                "A_vs_B": "conditionA - conditionB",
                "avg_effect": [0, 0.5, 0.5],
            })
            ```

        Note:
            String contrasts support coefficients (``"2*A - B"``, ``"0.5*A + 0.5*B"``).
            Column names must match design-matrix columns exactly (case-sensitive).
            Contrast weights should sum to zero for proper inference in most cases.
        """
        from .modeling import compute_contrasts

        return compute_contrasts(self, contrasts, statistic=statistic)

    def report(  # nosemgrep: kwargs-internal-forwarding  # forwards to nilearn generate_report
        self, contrasts=None, **kwargs
    ):
        """Generate a nilearn HTML report for a fitted GLM.

        Must be called after ``fit(model='glm', ...)``. Renders the design
        matrix, requested contrast maps, and model parameters as a
        self-contained HTML report.

        Args:
            contrasts (str | list | dict | None): Contrast(s) to render,
                same forms as `compute_contrasts`.
            **kwargs (dict): Forwarded to nilearn's ``generate_report`` (e.g.
                ``title``, ``threshold``, ``alpha``).

        Returns:
            HTMLReport: nilearn report; call ``.save_as_html(path)`` or display
                it in a notebook.

        Raises:
            RuntimeError: If a GLM has not been fit yet.

        Examples:
            ```python
            brain.fit(model='glm', X=design_matrix)
            brain.report(contrasts='conditionA - conditionB').save_as_html('report.html')
            ```
        """
        from nltools.models import Glm

        if not isinstance(getattr(self, "model_", None), Glm):
            raise RuntimeError(
                "report() requires a fitted GLM; call fit(model='glm', ...) first."
            )
        return self.model_.report(contrasts=contrasts, **kwargs)

    def copy(self):
        """Create an independent snapshot of a BrainData instance.

        Data, metadata, mask state, and any fitted model/results are copied.
        Mutating either object after copying does not affect the other.
        Python's `copy.copy()` and `copy.deepcopy()` have the same semantics.

        Returns:
            BrainData: An independent copy, including fitted state.
        """
        from .utils import _copy_complete

        return _copy_complete(self)

    def create_empty(self):
        """Create a copy of BrainData with empty data array.

        Returns:
            BrainData: A copy of this object with an empty data array.
        """
        from .utils import _result_from_array

        out = _result_from_array(self, np.array([]), rows="clear")
        return out

    @coalesced_gc()  # nosemgrep: kwargs-internal-forwarding  # forwards to the sklearn decomposition estimator
    def decompose(self, *, method="pca", axis="voxels", n_components=None, **kwargs):
        """Decompose BrainData object.

        Args:
            method (str): Decomposition algorithm: ``'pca'``, ``'ica'``, ``'nnmf'``,
                ``'fa'``, ``'dictionary'``, or ``'kernelpca'``. Default ``'pca'``.
            axis (str): Dimension to decompose: ``'voxels'`` (default) or ``'images'``.
            n_components (int | None): Number of components. If ``None`` then retain
                as many as possible.
            **kwargs (dict): Forwarded to the underlying sklearn decomposition
                estimator.

        Returns:
            dict: A dictionary of decomposition parameters.
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
            method (str): Type of detrending: ``'linear'`` (default) or ``'constant'``.

        Returns:
            BrainData: Detrended BrainData instance.
        """
        from .analysis import detrend_data

        return detrend_data(self, method=method)

    @coalesced_gc()  # nosemgrep: kwargs-internal-forwarding  # forwards to scipy.spatial.distance.cdist via analysis.distance
    def distance(  # nosemgrep: kwargs-internal-forwarding  # forwards to scipy.spatial.distance.cdist
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
            metric (str): Distance metric — any ``scipy.spatial.distance`` metric
                supported by ``cdist``. Default ``'euclidean'``.
            spatial_scale (str): One of ``'whole_brain'`` (default), ``'roi'``, or
                ``'searchlight'``. ``'whole_brain'`` returns a single
                pairwise distance ``Adjacency`` between images. ``'roi'``
                requires ``roi_mask`` and returns a stacked ``Adjacency``
                with one RDM per sorted nonzero atlas label present inside the
                source mask after nearest-neighbor resampling. `'searchlight'`
                returns one RDM per source-mask voxel in mask order.
            roi_mask (BrainData | Nifti1Image | str | Path | None): Atlas image
                for ``spatial_scale='roi'``.
            radius_mm (float): Searchlight radius in mm. Default 10.0.
            **kwargs (dict): Additional metric options forwarded to
                ``scipy.spatial.distance.cdist`` (e.g. ``p`` for minkowski).

        Returns:
            Adjacency: Single pairwise distance matrix for ``'whole_brain'``;
                ordinary stack for `'roi'` / `'searchlight'`. Map per-matrix values
                externally using `roi_to_brain_from_atlas` with the aligned atlas
                and sorted surviving ROI labels, or `nilearn.masking.unmask`
                with the source mask for searchlights. Subset the mapping whenever
                selecting matrices from the returned stack.
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
    def extract_roi(self, mask, method="mean", n_components=None):
        """Extract activity from mask or ROI atlas using NiftiLabelsMasker.

        The mask may be binary (a single ROI) or a labeled atlas (one value per
        region, extracting from every ROI at once).

        Args:
            mask (BrainData | Nifti1Image | str | Path): Binary mask or labeled
                atlas to extract from.
            method (str): Extraction method: ``'mean'`` (default), ``'median'``, or
                ``'pca'``.
            n_components (int | None): Number of components to return when
                ``method='pca'``.

        Returns:
            float | np.ndarray: For a binary mask, a scalar (single image) or 1D
                array (multiple images). For a labeled atlas, a 1D array (single
                image), a 2D array of images x ROIs (multiple images), or the PCA
                components array when ``method='pca'``.

        Examples:
            ```python
            roi_values = brain.extract_roi(binary_mask)
            atlas_values = brain.extract_roi(atlas_mask)
            components = brain.extract_roi(mask, method='pca', n_components=5)
            ```
        """
        from .analysis import extract_roi

        return extract_roi(self, mask, method=method, n_components=n_components)

    def filter(  # nosemgrep: kwargs-internal-forwarding  # forwards to nilearn.signal.clean
        self, *, sampling_freq=None, high_pass=None, low_pass=None, **kwargs
    ):
        """Apply a Butterworth filter to data (wraps `nilearn.signal.clean`).

        Note:
            Unlike nilearn's default, does not detrend or standardize. Pass
            detrend=True or standardize=True via kwargs to enable.

        Args:
            sampling_freq (float | None): Sampling frequency in hertz (i.e. 1 / TR).
            high_pass (float | None): High-pass cutoff frequency in hertz.
            low_pass (float | None): Low-pass cutoff frequency in hertz.
            **kwargs (dict): Additional arguments passed to ``nilearn.signal.clean``.

        Returns:
            BrainData: Filtered BrainData instance.
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
            DesignMatrix: One indicator column per detected spike TR, with all
                spike columns pre-marked as confounds. A TR flagged by both
                detectors yields a single column (named `global_spike*`); the
                colliding detections are bitwise identical, so only the retained
                name differs.
        """
        from .analysis import find_spikes_data

        return find_spikes_data(
            self,
            global_spike_cutoff=global_spike_cutoff,
            diff_spike_cutoff=diff_spike_cutoff,
            TR=TR,
            sampling_freq=sampling_freq,
        )

    @coalesced_gc()  # nosemgrep: kwargs-internal-forwarding  # forwards model params to the nilearn/sklearn estimator via modeling.fit
    def fit(
        self,
        model="glm",
        *,
        X=None,
        cv=None,
        device="cpu",
        local_alpha=True,
        fit_intercept=False,
        inplace=True,
        scale="auto",
        standardize="auto",
        progress_bar=False,
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
            device (str, default='cpu'): Ridge only. Compute device for the
                ridge solve/CV: ``'cpu'`` (NumPy), ``'gpu'`` (PyTorch on
                CUDA/MPS when available), or ``'auto'`` (GPU if present, else
                CPU). Ignored when ``model='glm'``.
            local_alpha (bool, default=True): Ridge only. If True, select
                α independently per voxel via ``solve_ridge_cv``. If False,
                pick a single α shared across all voxels.
            fit_intercept (bool, default=False): Ridge only. Forwarded to
                the Ridge model — center X and y on the training fold mean
                per fold and recover the intercept after.
            inplace (bool, default=True): If True, mutate self and return self.
                If False, fit and return an independent `BrainData` copy while
                leaving every part of self untouched.
            scale (bool or 'auto', default='auto'): Apply percent-signal-change
                scaling before fitting via nilearn's per-voxel ``mean_scaling``.
                ``'auto'`` → False for both models (PSC is opt-in). Redundant
                with ``standardize='zscore'`` (warns). Applied before
                ``standardize``.
            standardize (str or None or 'auto', default='auto'): Standardize
                each voxel across observations after scaling. ``'center'``,
                ``'zscore'``, or ``None``. ``'auto'`` → ``'zscore'`` for ridge,
                ``None`` for glm.
            progress_bar (bool): Display a progress bar during fitting. Default: False.
            **kwargs (dict): Additional arguments passed to the model constructor
                (e.g. ``alpha`` for ridge).

        Returns:
            BrainData: Self when ``inplace=True``; otherwise an independently
                owned fitted copy.

        Note:
            After ``model="glm"``, the following per-regressor BrainData
            attributes are populated — one map per design-matrix column:
            ``glm_betas`` (effect-size β maps), ``glm_t`` (marginal t-statistic for
            each regressor), ``glm_p`` (marginal p-value), ``glm_se`` (standard
            error of β), and ``glm_r2`` (voxel-wise R²).

            ``glm_t[i]`` is a valid t-map for the trivial one-hot contrast on
            regressor ``i`` only. For contrasts across regressors
            (``"A - B"``, ``[1, -1, 0, ...]``) use `compute_contrasts` —
            you cannot correctly combine these per-regressor maps by hand
            because t-statistic arithmetic requires the off-diagonal elements
            of the parameter covariance matrix, which are not stored. Pass
            ``statistic="all"`` to get ``β``/``t``/``z``/``p``/``se`` for
            one contrast in a single call.

        Examples:
            ```python
            brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
            fit = brain_data.fit(model='ridge', alpha=1.0, X=features, inplace=False)
            ```
        """
        from .modeling import fit

        return fit(
            self,
            model=model,
            X=X,
            cv=cv,
            device=device,
            local_alpha=local_alpha,
            fit_intercept=fit_intercept,
            inplace=inplace,
            scale=scale,
            standardize=standardize,
            progress_bar=progress_bar,
            **kwargs,
        )

    def mean(self, axis=0, *, spatial_scale: str = "whole_brain", roi_mask=None):
        """Get mean of each voxel or image.

        Args:
            axis (int): 0 = across images (default, returns BrainData),
                1 = within images (returns array). Ignored when
                ``spatial_scale='roi'``.
            spatial_scale (str): ``'whole_brain'`` (default) reduces along
                ``axis``. ``'roi'`` requires ``roi_mask`` and returns a
                BrainData of the same shape with each voxel painted with
                its parcel's mean per image (parcellation smoothing).
            roi_mask (BrainData | Nifti1Image | str | Path | None): Atlas image
                for ``spatial_scale='roi'``.

        Returns:
            float | np.ndarray | BrainData: Mean values.
        """
        if spatial_scale == "roi":
            from .analysis import reduce_per_roi

            return reduce_per_roi(self, np.mean, roi_mask=roi_mask)
        from .utils import apply_func

        return apply_func(self, np.mean, axis)

    def median(self, axis=0, *, spatial_scale: str = "whole_brain", roi_mask=None):
        """Get median of each voxel or image.

        Args:
            axis (int): 0 = across images (default, returns BrainData),
                1 = within images (returns array). Ignored when
                ``spatial_scale='roi'``.
            spatial_scale (str): ``'whole_brain'`` (default) or ``'roi'`` (paints
                each voxel with its parcel's median per image).
            roi_mask (BrainData | Nifti1Image | str | Path | None): Atlas image
                for ``spatial_scale='roi'``.

        Returns:
            float | np.ndarray | BrainData: Median values.
        """
        if spatial_scale == "roi":
            from .analysis import reduce_per_roi

            return reduce_per_roi(self, np.median, roi_mask=roi_mask)
        from .utils import apply_func

        return apply_func(self, np.median, axis)

    def multivariate_similarity(self, images, method="ols", tail=2):
        """Predict a BrainData spatial distribution from a linear combination.

        The predictors may be other BrainData instances or nibabel images.

        Args:
            images (BrainData | Nifti1Image | list): Predictor image(s) — a
                BrainData stack of weight maps or nibabel images.
            method (str): Regression method. Default: 'ols'.
            tail (int | str): ``2`` or ``'two'`` for two-tailed (default); ``1`` or
                ``'one'`` for one-tailed (positive direction) regression p-values.

        Returns:
            dict: Regression statistics as BrainData instances, keyed
                `'beta'`, `'t'`, `'p'`, `'df'`, `'residual'`.
        """
        from .analysis import multivariate_similarity

        return multivariate_similarity(self, images, method=method, tail=tail)

    def plot(  # nosemgrep: kwargs-internal-forwarding  # forwards to nilearn plotting functions
        self,
        *,
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
            threshold (float | str, optional): Absolute transparency cutoff.
                Percentile strings resolve over finite, nonzero magnitudes.
            view (str): For ``method="slices"``, any non-empty combination of
                ``"x"``, ``"y"``, ``"z"`` (e.g. ``"xyz"``, ``"xz"``, ``"y"``).
                Default: ``"z"``.
            cut_coords (list or dict, optional): Cut coordinates for
                multi-slice views. Takes precedence over ``view``-based
                defaults. Either a list matching ``len(view)`` or a dict
                keyed by axis letter.
            cmap (str, optional): Colormap name. Defaults are sign-aware.
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
            **kwargs (dict): Additional arguments passed to nilearn plot functions.

        Returns:
            matplotlib.figure.Figure | list[matplotlib.figure.Figure]: A single
                figure for single-image data; a list of figures for multi-image
                data with `method` in `{"glass", "slices"}` (one per image for
                glass; one per image-and-view pair for slices).
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
        *,
        threshold=None,
        cmap=None,
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
            threshold (float | str, optional): Absolute cutoff or percentile string.
            cmap (str, optional): Matplotlib colormap. Defaults are sign-aware.
            vmax (float, optional): Maximum value; inferred from displayed data.
            vmin (float, optional): Minimum value; inferred from displayed data.
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
            matplotlib.figure.Figure: The rendered figure.
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
        cmap=None,
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
            matplotlib.figure.Figure: The rendered figure.
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

    def iplot(  # nosemgrep: kwargs-internal-forwarding  # forwards to new Niivue(opts)
        self,
        *,
        view: str = "ortho",
        threshold: "float | str | None" = None,
        lower: "float | str | None" = None,
        upper: "float | str | None" = None,
        autoscale: bool = True,
        symmetric: bool | Literal["auto"] = "auto",
        cmap: "str | None" = None,
        bg_img: "str | bool | None" = None,
        atlas: "str | Atlas | None" = None,
        opacity: float = 1.0,
        outline: float = 0.0,
        colorbar: bool = True,
        controls: bool = True,
        **kwargs,
    ):
        """Interactive WebGL brain viewer powered by niivue.

        Renders inline in a live kernel (Jupyter, marimo) with
        live windowing (right-drag to set the threshold/contrast), slice
        scrolling, native 4D frame scrubbing, true 3D rendering, a stat-map
        colorbar, and optional nltools-atlas overlays. Static-built docs (plain
        Markdown) are not interactive; use `plot` there.

        Returns a `NiivueViewer` widget. By default (``controls=True``) it
        renders an in-widget threshold slider above the viewer; the window is
        reactive through the ``cal_min`` / ``cal_max`` traits. Pass
        ``controls=False`` to hide the slider (right-drag windowing still
        works).

        Thresholding uses positive and negative display limbs. ``cal_min`` is
        the magnitude floor and ``cal_max`` the positive saturation point;
        niivue receives the negative endpoints explicitly. By default, mixed
        maps use symmetric limbs while each sign in a one-sided map determines
        its own ceiling. The window is computed in Python, and the two controls
        show the shared floor and positive-limb ceiling.

        Args:
            view: ``"ortho"`` (default), ``"axial"``, ``"coronal"``,
                ``"sagittal"``, or ``"render"`` (3D volume render).
                ``"surface"`` is no longer supported — use ``"render"`` or
                `plot_flatmap` / `plot_surf`.
            threshold: Convenience symmetric magnitude floor (→ ``cal_min``).
                Accepts a percentile string (``"95%"``) resolved over the
                finite nonzero magnitudes, consistent with `threshold`.
            lower: Window floor (→ ``cal_min``). Overrides ``threshold``.
                Accepts a percentile string.
            upper: Window ceiling (→ ``cal_max``). Overrides ``threshold``.
                Accepts a percentile string.
            autoscale: Robust default window for the edges not set above.
                ``True`` (default): ceiling at the 98th percentile of the
                finite nonzero magnitudes — a couple of outlier voxels no
                longer wash out the whole map — and an epsilon floor, never
                above the smallest nonzero magnitude, so zeros render
                transparent and every real voxel stays visible (threshold up
                from there). ``False``: the raw magnitude range from zero to
                the largest absolute value. For a custom percentile window pass
                ``lower``/``upper`` (e.g. ``lower="60%", upper="98%"``).
            symmetric: ``"auto"`` (default) mirrors mixed-signed maps but lets
                each sign in a one-sided map determine its own ceiling. ``True``
                always mirrors; ``False`` scales positive and negative limbs
                independently.
            cmap: niivue colormap for the positive limb. The default uses
                niivue's red positive and blue negative palettes. Common
                matplotlib names are auto-mapped with a warning.
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
            controls: Render an in-widget threshold slider above the viewer
                (default ``True``). ``False`` hides it; the viewer still
                supports niivue's right-drag windowing. No extra dependency
                either way — the slider is native to the widget frontend.
            **kwargs (dict): Passed as niivue options. ``height`` configures
                the canvas and ``is_colorbar`` overrides ``colorbar``.

        Returns:
            NiivueViewer: An `anywidget.AnyWidget` whose threshold window is
                reactive via the `cal_min` and `cal_max` traits.

        Raises:
            TypeError: If ``autoscale`` is not a bool or ``symmetric`` is not
                ``True``, ``False``, or ``"auto"``.
        """
        from .viewer import build_viewer, compute_display_windows

        cal_min, cal_max, cal_min_neg, cal_max_neg, mirror_negative = (
            compute_display_windows(
                self.data,
                autoscale=autoscale,
                threshold=threshold,
                lower=lower,
                upper=upper,
                symmetric=symmetric,
            )
        )

        return build_viewer(
            self,
            view=view,
            cal_min=cal_min,
            cal_max=cal_max,
            cal_min_neg=cal_min_neg,
            cal_max_neg=cal_max_neg,
            mirror_negative=mirror_negative,
            cmap=cmap,
            atlas=atlas,
            bg_img=bg_img,
            opacity=opacity,
            outline=outline,
            colorbar=colorbar,
            controls=controls,
            niivue_opts=kwargs,
        )

    @coalesced_gc()
    def predict(
        self,
        *,
        y: "np.ndarray | str | None" = None,
        X: "np.ndarray | None" = None,
        spatial_scale: str = "whole_brain",
        model="svm",
        cv: int | str = 5,
        standardize: bool = True,
        reduce: "str | None" = None,
        n_components: "int | None" = None,
        scoring: str = "auto",
        groups: "np.ndarray | str | None" = None,
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
        2. **MVPA decoding** (``y`` provided, or resolvable from ``.Y``):
           train a classifier or regressor with cross-validation. Returns a
           `Predict` dataclass. Spatial fields (``weight_map``,
           ``fold_weight_maps``, ``final_weight_map``, ``accuracy_map``) are
           `BrainData` objects so ``result.weight_map.plot()`` works
           directly. Drop down to numpy via ``result.weight_map.data``.

        Labels travel with the data: when ``y`` is omitted and this object
        carries a single-column ``.Y`` frame, that column is decoded
        (``y='name'`` picks a column of a multi-column ``.Y``; ``groups``
        accepts a ``.Y`` column name the same way). An object with both a
        fitted encoding model and a stored ``.Y`` refuses the no-argument
        call as ambiguous — pass ``y=`` or ``X=`` explicitly.

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

        **Choosing a model.** String shortcuts for classification are ``'svm'``
        (LinearSVC), ``'logistic'``, ``'lda'``, and ``'ridge_classifier'``; for
        regression, ``'ridge'``, ``'lasso'``, and ``'svr'``. Any sklearn estimator
        or ``Pipeline`` is also accepted (e.g.
        ``make_pipeline(StandardScaler(), SelectKBest(k=500), LinearSVC())``).
        When ``model`` is a sklearn ``Pipeline``, ``standardize`` is auto-defaulted
        to ``False`` (with a warning) so we don't wrap another StandardScaler
        around your pipeline; pass ``standardize=True`` explicitly to override.

        Args:
            y (array-like, str, optional): Labels (classification) or
                continuous targets (regression), shape ``(n_samples,)``, or
                the name of a ``.Y`` column. Triggers MVPA mode; omitted, it
                falls back to a single-column ``.Y``.
            X (array-like, optional): Features for timeseries prediction,
                shape ``(n_samples, n_features)``. Triggers encoding mode.
            spatial_scale (str): MVPA dispatch — ``'whole_brain'``,
                ``'searchlight'``, or ``'roi'``.
            model (str | sklearn estimator): Algorithm — a string shortcut
                (``'svm'``, ``'logistic'``, ``'lda'``, ``'ridge_classifier'``,
                ``'ridge'``, ``'lasso'``, ``'svr'``) or any sklearn estimator /
                Pipeline. Default ``'svm'``; see "Choosing a model" above.
            cv (int, str, or sklearn CV splitter): ``int`` → shuffled KFold
                (regression) or StratifiedKFold (classification), honoring
                ``groups`` via the Group variants; ``'loo'`` (leave-one-out);
                ``'logo'`` (leave-one-group-out — pass the grouping variable
                via ``groups``, e.g. runs for leave-one-run-out); or any
                sklearn splitter.
            standardize (bool): Z-score features per fold before fitting.
                Default ``True``. Auto-flipped to ``False`` when ``model`` is
                a sklearn ``Pipeline`` (see ``model`` above).
            reduce (str, optional): Per-fold dimensionality reduction.
                Currently only ``'pca'`` supported. Default ``None``. Weight
                maps are back-projected through PCA to voxel space.
            n_components (int, optional): PCA components when ``reduce='pca'``.
            scoring (str): Sklearn scoring string. Default ``'auto'`` →
                ``'accuracy'`` if classifier, ``'r2'`` if regressor.
            groups (array-like, str, optional): Group labels for CV splitters
                that need them (e.g., leave-one-run-out), or the name of a
                ``.Y`` column holding them.
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
            Whole-brain decoding:

            ```python
            result = brain.predict(y=labels, spatial_scale='whole_brain', cv=5)
            result.weight_map.plot()       # publishable map (all-data fit)
            result.mean_score              # honest CV-derived accuracy
            new_pred = result.estimator.predict(new_X)  # apply to new data
            ```

            Searchlight and ROI decoding:

            ```python
            result = brain.predict(y=labels, spatial_scale='searchlight',
                                   radius_mm=8.0, n_jobs=4)
            result.accuracy_map.plot()

            result = brain.predict(y=labels, spatial_scale='roi', roi_mask=atlas)
            top = result.roi_labels[result.mean_score.argsort()[::-1][:10]]
            result.accuracy_map.plot()  # brain-space view of the same map
            ```

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
        *,
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

    def resample_to(self, *, img=None, resolution=None, interpolation=None):
        """Resample BrainData to match target image or resolution.

        Args:
            img (Nifti1Image | str | Path | None): Target image for resampling.
            resolution (float | int | None): Target isotropic voxel size in mm.
            interpolation (str | None): Interpolation method: ``'nearest'``,
                ``'linear'``, ``'continuous'``, or ``None`` to use the instance's
                setting.

        Returns:
            BrainData: New BrainData instance with resampled data.

        Raises:
            ValueError: If both ``img`` and ``resolution`` are None, or both are
                provided.
        """
        from .io import resample_to

        return resample_to(
            self, img=img, resolution=resolution, interpolation=interpolation
        )

    def scale(self, scale_val=100.0, axis=None):
        """Scale data via mean scaling.

        Two scaling modes are available. **Grand-mean scaling** (``axis=None``,
        default) divides all values by the global mean across all voxels and
        timepoints. **Voxel-wise scaling** (``axis=0``) divides each voxel's
        time-series by its own temporal mean.

        Args:
            scale_val (int | float): Target value for the mean after scaling.
                Default 100.
            axis (int | None): ``None`` for grand-mean scaling (default), ``0``
                for voxel-wise scaling.

        Returns:
            BrainData: New BrainData instance with scaled data.
        """
        from .analysis import scale_data

        return scale_data(self, scale_val, axis)

    @coalesced_gc()
    def similarity(self, image, metric="correlation"):
        """Calculate similarity to a single BrainData or nibabel image.

        Args:
            image (BrainData | Nifti1Image): Image to evaluate similarity against.
            metric (str): Type of similarity: ``'correlation'`` (default),
                ``'pearson'``, ``'rank_correlation'``, ``'spearman'``,
                ``'dot_product'``, or ``'cosine'``.

        Returns:
            float or np.ndarray: Similarity value(s).
        """
        from .analysis import similarity

        return similarity(self, image, metric=metric)

    def smooth(self, fwhm):
        """Apply spatial smoothing using nilearn smooth_img().

        Args:
            fwhm (float): Full width at half maximum of the Gaussian spatial
                filter, in mm.

        Returns:
            BrainData: Copy with smoothed data.
        """
        from .analysis import smooth

        return smooth(self, fwhm)

    def standardize(self, *, axis=0, method="center"):
        """Standardize BrainData() instance.

        Constant voxels (or observations) z-score to 0 rather than NaN.

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

    def std(self, axis=0, *, spatial_scale: str = "whole_brain", roi_mask=None):
        """Get standard deviation of each voxel or image.

        Args:
            axis (int): 0 = across images (default, returns BrainData),
                1 = within images (returns array). Ignored when
                ``spatial_scale='roi'``.
            spatial_scale (str): ``'whole_brain'`` (default) or ``'roi'`` (paints
                each voxel with its parcel's std per image).
            roi_mask (BrainData | Nifti1Image | str | Path | None): Atlas image
                for ``spatial_scale='roi'``.

        Returns:
            float | np.ndarray | BrainData: Standard deviation values.
        """
        if spatial_scale == "roi":
            from .analysis import reduce_per_roi

            return reduce_per_roi(self, np.std, roi_mask=roi_mask)
        from .utils import apply_func

        return apply_func(self, np.std, axis)

    def sum(self, axis=0):
        """Get sum of each voxel or image.

        Args:
            axis (int): 0 = across images (default, returns BrainData),
                1 = within images (returns array).

        Returns:
            float | np.ndarray | BrainData: Sum values.
        """
        from .utils import apply_func

        return apply_func(self, np.sum, axis)

    def temporal_resample(self, *, sampling_freq=None, target=None, target_type="hz"):
        """Resample BrainData timeseries to a new target frequency or number of samples.

        Args:
            sampling_freq (float | None): Sampling frequency of the data in hertz.
            target (float | None): Resampling target, interpreted per ``target_type``.
            target_type (str): How to read ``target``: ``'hz'`` (default),
                ``'samples'``, or ``'seconds'``.

        Returns:
            BrainData: Resampled BrainData instance.
        """
        from .analysis import temporal_resample

        return temporal_resample(
            self, sampling_freq=sampling_freq, target=target, target_type=target_type
        )

    @coalesced_gc()
    def threshold(
        self,
        *,
        upper=None,
        lower=None,
        binarize=False,
        coerce_nan=True,
        cluster_threshold=0,
    ):
        """Threshold BrainData instance with optional cluster filtering.

        Args:
            upper (float | str | None): Upper cutoff for thresholding; a
                percentile string like ``'95%'`` is accepted.
            lower (float | str | None): Lower cutoff for thresholding; a
                percentile string is accepted.
            binarize (bool): Return a binarized image. Default False.
            coerce_nan (bool): Coerce NaN values to 0s. Default True.
            cluster_threshold (int): Minimum cluster size in voxels. Default 0.

        Returns:
            BrainData: Thresholded BrainData object.
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
            ClusterReport: Report with `peaks` and `clusters` (polars DataFrames)
                and `stat_img` (BrainData).
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
        *,
        popmean=0.0,
        permutation=False,
        n_permute=5000,
        tail=2,
        return_null=False,
        n_jobs=-1,
        random_state=None,
    ):
        """Run a one-sample voxelwise t-test across images (axis 0).

        Tests whether the per-voxel mean across a stack of images (e.g.
        subject-level contrast maps, shape `(n_images, n_voxels)`) differs from
        `popmean`.

        Args:
            popmean (float): Population mean to test against. Default 0.0.
            permutation (bool): If True, take p from a sign-flip permutation
                test on `images - popmean`. The reported `t` stays the observed
                parametric statistic. Default False.
            n_permute (int): Number of permutations, used only when
                `permutation=True`. Default 5000.
            tail (int | str): `2` or `'two'` for two-tailed (default); `1` or
                `'one'` for one-tailed (mean > `popmean`).
            return_null (bool): If True, also return the permutation null. Has
                no effect on the parametric path, which computes no null.
                Default False.
            n_jobs (int): Number of parallel jobs. Default -1 (all cores).
            random_state (int | None): Random seed for reproducibility.

        Returns:
            dict: `"mean"`, `"t"`, `"z"` and `"p"` as independent `BrainData`
                images with observation metadata cleared. `"mean"` is the
                voxelwise mean minus `popmean` — the effect relative to the
                tested null, equal to the raw mean only when `popmean=0`.
                `"t"` is the observed one-sample t-statistic on both paths.
                `"p"` is parametric, or the empirical sign-flip p-value when
                `permutation=True`. `"z"` is the tail-aware normal score of `p`
                (`sign(t) * norm.isf(p/2)` two-tailed), matching nilearn's
                `output_type='z_score'`. With `permutation=True` and
                `return_null=True` the dict also holds `"null_dist"`, an owned
                `(n_permute, n_voxels)` array of centered means in the units of
                `"mean"`. Maps are unthresholded. Apply a cutoff or a
                multiple-comparison correction afterwards.

        Raises:
            ValueError: If this BrainData contains fewer than 2 images.

        Examples:
            ```python
            # Stack of subject-level contrast maps
            result = contrast_maps.ttest()
            effect = result["mean"]  # magnitude, for reporting
            z_map = result["z"]  # for nilearn-style thresholding

            # Threshold after testing, never inside it
            from nltools.algorithms import threshold

            z_thresh = threshold(result["z"], result["p"], thr=0.001)

            # Permutation p-values, keeping the null for a custom correction
            perm = contrast_maps.ttest(
                permutation=True, n_permute=5000, return_null=True, random_state=0
            )
            perm["null_dist"].shape  # → (5000, n_voxels)
            ```
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

    def ttest2(self, other, equal_var=True, tail=2):
        """Two-sample voxelwise t-test between two BrainData stacks.

        Args:
            other (BrainData): BrainData to compare against. Must have the same
                number of voxels.
            equal_var (bool): If True (default), standard two-sample t-test.
                If False, Welch's t-test.
            tail (int | str): ``2`` or ``'two'`` for two-tailed (default); ``1`` or
                ``'one'`` for one-tailed (self > other; swap the operands for the
                other direction).

        Returns:
            dict: ``{"t": BrainData, "p": BrainData}``.

        Raises:
            ValueError: If the two BrainData objects have different
                ``n_voxels``.
        """
        from .modeling import ttest2

        return ttest2(self, other, equal_var=equal_var, tail=tail)

    def upload_neurovault(  # nosemgrep: kwargs-internal-forwarding  # forwards to the NeuroVault API via io.upload_neurovault
        self,
        *,
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
            access_token (str): NeuroVault API access token. Required.
            collection_name (str | None): Name of a new collection to create.
            collection_id (int | None): NeuroVault ``collection_id`` when adding
                images to an existing collection.
            img_type (str): NeuroVault ``map_type``. Required.
            img_modality (str): NeuroVault image modality. Required.
            **kwargs (dict): Additional image metadata forwarded to the NeuroVault
                API.

        Returns:
            dict: NeuroVault collection information.
        """
        from .io import upload_neurovault

        return upload_neurovault(
            self,
            access_token=access_token,
            collection_name=collection_name,
            collection_id=collection_id,
            img_type=img_type,
            img_modality=img_modality,
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
