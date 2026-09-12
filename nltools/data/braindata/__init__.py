"""Represent brain image data with the BrainData class."""

import os
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from nibabel import Nifti1Image
    from sklearn.base import BaseEstimator
    from sklearn.model_selection import BaseCrossValidator

    from nltools.data.atlases import Atlas, ClusterReport
    from nltools.data.designmatrix import DesignMatrix
    from nltools.data.results import Predict

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
            (``.h5``/``.hdf5``) output, ``'gzip'`` (default) or ``'lzf'``.
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
    ):
        """Align BrainData instance to target object using functional alignment.

        Args:
            target (BrainData): Object to align to.
            method (str): Alignment method: ``'probabilistic_srm'``,
                ``'deterministic_srm'``, or ``'procrustes'``. Default ``'procrustes'``.
            axis (int): Axis to align on. Default 0.
            spatial_scale (str): ``'whole_brain'`` (default) or ``'roi'``
                (per-parcel transforms + reassembly, requires `roi_mask`).
            roi_mask (BrainData | Nifti1Image | str | Path | None): Atlas image
                used when ``spatial_scale='roi'``.

        Returns:
            dict: ``'transformed'``, ``'transformation_matrix'`` and
                ``'common_model'``, plus ``'disparity'`` and ``'scale'`` for
                ``method='procrustes'``. A value is a `BrainData` when its
                columns are a voxel axis matching the mask it carries, and a
                raw `np.ndarray` otherwise. ``'procrustes'`` therefore returns
                all three as independently owned `BrainData`, with float
                ``'disparity'`` and ``'scale'``. The SRM methods return
                ``'transformed'`` ``(n_images, n_features)`` and
                ``'common_model'`` ``(n_model_rows, n_features)`` as raw
                `np.ndarray`, since both span the common model's feature axis
                rather than voxels, and ``'transformation_matrix'`` as a
                `BrainData` of ``n_features`` voxel maps. With ``axis=1`` the
                transformation matrix spans images on its column axis for
                either method, so it is a raw `np.ndarray` too. With
                ``spatial_scale='roi'`` the result also carries
                ``'roi_labels'``, ``'transformed'`` is one stitched
                `BrainData`, ``'transformation_matrix'`` and ``'common_model'``
                are dicts keyed by atlas label whose values follow the same
                rule on that parcel's mask, and ``'disparity'`` and
                ``'scale'`` are one-per-parcel arrays.

        Raises:
            ValueError: If a value that must be returned as a `BrainData` has a
                column count other than the mask support — for example a
                ``'procrustes'`` target with more voxels than the source, which
                zero-pads the source data to the target's width.

        Examples:
            ```python
            # Hyperalign using procrustes transform
            out = data.align(target, method='procrustes')

            # Align using shared response model
            out = data.align(target, method='probabilistic_srm')

            # Project procrustes-aligned data back into original voxel space
            original = np.dot(
                out['transformed'].data, out['transformation_matrix'].data.T
            )
            ```
        """
        if spatial_scale == "roi":
            from .analysis import align_per_roi

            return align_per_roi(
                self, target, method=method, axis=axis, roi_mask=roi_mask
            )
        if spatial_scale != "whole_brain":
            raise ValueError(
                f"spatial_scale must be one of {{'whole_brain', 'roi'}}, "
                f"got {spatial_scale!r}"
            )
        from .analysis import align

        return align(self, target, method=method, axis=axis)

    def append(self, data, *, ignore_attrs=False):
        """Append data to BrainData instance.

        Args:
            data (BrainData): BrainData instance to append.
            ignore_attrs (bool): Clear both X and Y on the result when True.
                Otherwise, each metadata frame must be empty on both inputs or
                have compatible columns on both inputs. Default False.

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
    def apply_mask(self, mask):
        """Restrict the data to a mask's support, leaving the grid unchanged.

        The mask must be a single three-dimensional image on the same grid and
        with the same affine as this object. A mismatch raises: resample the
        mask or the data with `resample()` first, rather than relying on an
        implicit resample here.

        Support is every voxel of `mask` greater than zero, and the mask defines
        the result's voxel axis on its own. Where it reaches past this object's
        current support the result gains those voxels with zero values, so a
        mask larger than `self.mask` widens the array rather than intersecting
        with it.

        Args:
            mask (BrainData | Nifti1Image | str | Path): Mask to apply.

        Returns:
            BrainData: Masked BrainData object.

        Raises:
            ValueError: If the mask is not a single 3-D image, or its shape or
                affine differs from this object's.
            TypeError: If `mask` is not a BrainData, nibabel image, or file path.
        """
        from .analysis import apply_mask

        return apply_mask(self, mask)

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
        statistic,
        *,
        X=None,
        X_test=None,
        n_samples=5000,
        confidence_level=0.95,
        device="cpu",
        memory_budget_gb=None,
        return_samples=False,
        n_jobs=-1,
        random_state=None,
        progress_bar: bool = False,
    ):
        """Bootstrap a statistic and its uncertainty, on CPU workers or a GPU.

        Resamples rows with replacement and aggregates the replicates as they
        complete, into a running Welford variance plus just enough retained
        order statistics per output element to reproduce the exact percentile
        interval. What the run holds is that retained tail — about
        ``(1 - confidence_level)`` of the replicates per element — plus one
        dispatch window, rather than all ``n_samples`` maps. This is
        memory-efficient, not constant-memory: the tail still grows with
        ``n_samples``, and ``return_samples=True`` keeps the whole
        distribution.

        A Ridge bootstrap resamples the training features you pass as ``X``
        together with ``self.data``, using the same row indices for every
        feature space, and refits with the fitted model's selected ``alpha_``
        — and, for a banded model, its ``feature_space_weights_`` — held fixed.
        It never reruns cross-validation or the banded random search. Fitting
        keeps no hidden copy of the training features, so ``X`` is required
        even when the same features were passed to `fit`.

        Args:
            statistic (str): Statistic to bootstrap. Basic aggregates:
                ``'mean'``, ``'median'``, ``'std'``, ``'sum'``, ``'min'``,
                ``'max'`` — each the corresponding NumPy reduction over rows,
                with ``'std'`` at ``ddof=0``. Model statistics (require a
                fitted `Ridge`): ``'weights'`` or ``'predict'``.
            X (np.ndarray | Mapping[str, np.ndarray] | None): Training features
                in their original row order — a matrix for ordinary Ridge, a
                mapping with exactly the fitted feature-space names for banded
                Ridge. Required by both model statistics; rejected by the basic
                ones.
            X_test (np.ndarray | Mapping[str, np.ndarray] | None): Evaluation
                features for ``statistic='predict'``, in the same structure as
                ``X``. Any row count is allowed.
            n_samples (int): Number of bootstrap replicates, at least two.
                Default 5000.
            confidence_level (float): Confidence level of the reported
                interval, strictly between zero and one. Default 0.95. The
                bounds are the central percentile interval by linear
                interpolation, and they are elementwise marginal: the nominal
                level applies separately to each voxel, feature, or test row,
                with no simultaneous-coverage claim. A different level needs a
                new run unless ``return_samples=True`` kept the distribution.
            device (str): Compute device for the Ridge refits: ``'cpu'``
                (default) or ``'gpu'`` (PyTorch on CUDA/MPS, or an error when
                neither is available). Basic statistics reject ``'gpu'``.
            memory_budget_gb (float | None): Working-memory budget in GB. It
                governs the output preflight and CPU-worker planning for every
                statistic, and GPU batch sizing for the Ridge ones. ``None``
                (default) measures the device.
            return_samples (bool): Retain and return every replicate. Default
                False. It changes retention only, never interval semantics.
            n_jobs (int): CPU worker ceiling. -1 (default) means all cores; the
                planner may use fewer.
            random_state (int | None): Random seed for reproducibility.
            progress_bar (bool): If True, show a progress bar. Default False.

        Returns:
            BootstrapResult: ``estimate`` (the statistic on the unresampled
                full sample — for ``'weights'`` the fitted coefficients, for
                ``'predict'`` the full-data model at ``X_test``),
                ``standard_error`` (the ``ddof=1`` deviation across
                replicates), ``ci_lower`` and ``ci_upper``, all `BrainData` of
                identical shape, plus ``samples`` as a NumPy array with the
                bootstrap axis first when ``return_samples=True``.

        Raises:
            ValueError: If `statistic` is unknown, a basic statistic is given
                ``X``, ``X_test`` or ``device='gpu'``, a Ridge statistic is
                missing its features, the fitted model is not a `Ridge`, an
                argument is out of range, or the retained output cannot fit the
                memory budget.

        Examples:
            ```python
            boot = brain.bootstrap('mean', n_samples=1000)
            boot.estimate.plot()

            brain.fit(model='ridge', X=features, ridge_alpha=1.0)
            boot = brain.bootstrap('weights', X=features, n_samples=1000)
            ```

        Note:
            This is an IID row bootstrap. Rows must be exchangeable for the
            interval to be meaningful; it implements no grouped, clustered,
            stratified, or block resampling, so an autocorrelated fMRI time
            series must not be treated as IID rows.
        """
        from .bootstrap import bootstrap

        return bootstrap(
            self,
            statistic,
            X=X,
            X_test=X_test,
            n_samples=n_samples,
            confidence_level=confidence_level,
            device=device,
            memory_budget_gb=memory_budget_gb,
            return_samples=return_samples,
            n_jobs=n_jobs,
            random_state=random_state,
            progress_bar=progress_bar,
        )

    def compute_contrasts(self, contrasts, *, inference=False):
        """Compute contrasts on a fitted GLM.

        Call after ``fit(model='glm', X=design)``. The fitted `Glm` owns
        contrast parsing and inference; this method forwards each definition
        unchanged and wraps the results as `BrainData` maps.

        A contrast is a **string** naming design columns with optional
        coefficients (``"conditionA - conditionB"``, ``"2*A - B - C"``) or a
        **numeric vector** with one weight per column (``[1, -1, 0, 0]``). A
        **mapping** of names to those forms computes several at once and is the
        only batch form.

        Args:
            contrasts (str | array-like | Mapping): One contrast definition, or
                a mapping of names to definitions.
            inference (bool): If True, return `ContrastResult` records carrying
                effect, variance, standard error, t-statistic, z-score,
                one-sided p-value, and degrees of freedom. Default False.

        Returns:
            BrainData | ContrastResult | dict: An effect map for one contrast,
                or a `ContrastResult` of maps when ``inference=True``; a
                dictionary with the same keys for a mapping.

        Raises:
            RuntimeError: If no model has been fitted.
            ValueError: If the fitted model is not a `Glm`, or a contrast is
                invalid (see `Glm.compute_contrasts`).

        Examples:
            ```python
            brain.fit(model='glm', X=design)

            # Effect maps — what a second-level model consumes
            effect = brain.compute_contrasts("conditionA - conditionB")
            effects = brain.compute_contrasts({
                "A_vs_B": "conditionA - conditionB",
                "avg": [0, 0.5, 0.5],
            })

            # First-level inference
            result = brain.compute_contrasts("conditionA - conditionB", inference=True)
            result.statistic.plot(threshold=3.09)
            ```

        Note:
            Contrast p-values are one-sided, following the nilearn/SPM
            directional-contrast convention; negate the contrast to test the
            other direction.
        """
        from .modeling import compute_contrasts

        return compute_contrasts(self, contrasts, inference=inference)

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
        radius: float = 10.0,
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
            radius (float): Searchlight radius in mm. Default 10.0.
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
            radius=radius,
            **kwargs,
        )

    @coalesced_gc()
    def extract_roi(self, mask, method="mean", n_components=None):
        """Extract activity from mask or ROI atlas using NiftiLabelsMasker.

        The mask may be binary (a single ROI) or a labeled atlas (one value per
        region, extracting from every ROI at once). Unlike `apply_mask`, this
        is an extraction convenience: `mask` is resampled onto this object's
        own grid with nearest-neighbor interpolation before extracting, so it
        need not already share this object's grid.

        Args:
            mask (BrainData | Nifti1Image | str | Path): Binary mask or labeled
                atlas to extract from, on any grid.
            method (str): Extraction method: ``'mean'`` (default), ``'median'``, or
                ``'pca'``.
            n_components (int | None): Number of components to return when
                ``method='pca'``.

        Returns:
            float | np.ndarray: For a binary mask, a scalar (single image) or 1D
                array (multiple images). For a labeled atlas, a 1D array (single
                image), a 2D array of images x ROIs (multiple images), or the PCA
                components array when ``method='pca'``.

        Raises:
            ValueError: If, after resampling onto this object's grid, `mask`
                has no overlap with it.

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
        ridge_alpha=1.0,
        ridge_cv=None,
        ridge_search_iterations=100,
        ridge_dirichlet_concentration=(0.1, 1.0),
        ridge_device="cpu",
        ridge_memory_budget_gb=None,
        ridge_per_target_alpha=True,
        ridge_prefer_conservative_alpha=False,
        ridge_progress_bar=False,
        glm_noise_model="ols",
        glm_bins=100,
        glm_n_jobs=1,
        inplace=True,
        random_state=None,
    ):
        """Fit a model to brain imaging data.

        ``self.data`` is always the response. The fitted estimator and its
        results are stored for later use with `predict` and, for a GLM,
        `compute_contrasts`.

        Every model-specific option carries a ``glm_`` or ``ridge_`` prefix
        naming the estimator it configures; ``random_state`` keeps its bare
        name because both estimators accept it. Supplying a non-default option
        belonging to the estimator ``model`` did not select raises
        `ValueError`.

        `fit` does not preprocess the response. Compose `scale` and
        `standardize` before calling it when you want them, so the fitted
        object stays in the response space you supplied.

        Args:
            model (str): ``'glm'`` (default) or ``'ridge'``.
            X (DesignMatrix | array-like | Mapping): A precomputed
                `DesignMatrix` for a GLM; a feature matrix for ridge, or a
                mapping of feature-space names to matrices for banded ridge.
                Required.
            ridge_alpha (float | Sequence[float]): Ridge only. A positive
                scalar fits a fixed α and requires ``ridge_cv=None``; a
                sequence selects α by cross-validation and requires
                ``ridge_cv``. Default 1.0.
            ridge_cv (int | sklearn splitter | None): Ridge only.
                Cross-validation specification; ``int`` → unshuffled
                ``KFold(cv)``. Generators are rejected. Default None.
            ridge_search_iterations (int): Ridge only, banded. Sampled
                feature-space weight vectors. Default 100.
            ridge_dirichlet_concentration (float | Sequence[float]): Ridge
                only, banded. Dirichlet concentration for those candidate
                weights. Default ``(0.1, 1.0)``.
            ridge_device (str): Ridge only. ``'cpu'`` (default) or ``'gpu'``.
            ridge_memory_budget_gb (float | None): Ridge only. Working-memory
                budget in GB for the solver's internal batching. Default None
                (measure the device).
            ridge_per_target_alpha (bool): Ridge only. Select α per voxel
                (default True) or one shared α.
            ridge_prefer_conservative_alpha (bool): Ridge only. Select the
                largest α within one standard deviation of the best score.
                Default False.
            ridge_progress_bar (bool): Ridge only. Show a progress bar over the
                banded search. Default False.
            glm_noise_model (str): GLM only. ``'ols'`` (default) or ``'arN'``
                for Nilearn's autoregressive model of order N.
            glm_bins (int): GLM only. Nilearn's discretization of the estimated
                AR coefficients. Default 100.
            glm_n_jobs (int): GLM only. CPUs Nilearn uses for autoregressive
                groups; the default OLS fit does not use this path. Default 1.
            inplace (bool): If True (default), mutate self and return self. If
                False, fit and return an independent `BrainData` copy while
                leaving every part of self untouched.
            random_state (int | None): Seed shared by both estimators.

        Returns:
            BrainData: Self when ``inplace=True``; otherwise an independently
                owned fitted copy.

        Note:
            A GLM fit attaches ``model_``, ``glm_betas`` (one map per design
            column), ``glm_residual``, ``glm_predicted``, and ``glm_r2``.
            ``glm_r2`` is Nilearn's whitened variance ratio: conventional
            R-squared for an OLS fit whose design has an intercept, and a
            pseudo-R-squared in the whitened space for an autoregressive one.
            A GLM fit does not compute eager per-regressor t, p, or
            standard-error maps: ask for them one contrast at a time with
            ``compute_contrasts(..., inference=True)``, which uses the full
            per-voxel parameter covariance and is therefore correct for
            contrasts spanning several regressors.

        Examples:
            ```python
            brain_data.fit(model='glm', X=design)
            effect = brain_data.compute_contrasts('conditionA - conditionB')

            fitted = brain_data.fit(
                model='ridge', ridge_alpha=1.0, X=features, inplace=False
            )
            ```
        """
        from .modeling import fit

        return fit(
            self,
            model=model,
            X=X,
            ridge_alpha=ridge_alpha,
            ridge_cv=ridge_cv,
            ridge_search_iterations=ridge_search_iterations,
            ridge_dirichlet_concentration=ridge_dirichlet_concentration,
            ridge_device=ridge_device,
            ridge_memory_budget_gb=ridge_memory_budget_gb,
            ridge_per_target_alpha=ridge_per_target_alpha,
            ridge_prefer_conservative_alpha=ridge_prefer_conservative_alpha,
            ridge_progress_bar=ridge_progress_bar,
            glm_noise_model=glm_noise_model,
            glm_bins=glm_bins,
            glm_n_jobs=glm_n_jobs,
            inplace=inplace,
            random_state=random_state,
        )

    def mean(self, axis=0):
        """Get mean of each voxel or image.

        Args:
            axis (int): 0 = across images (default, returns BrainData),
                1 = within images (returns array).

        Returns:
            float | np.ndarray | BrainData: Mean values.
        """
        from .utils import apply_func

        return apply_func(self, np.mean, axis)

    def median(self, axis=0):
        """Get median of each voxel or image.

        Args:
            axis (int): 0 = across images (default, returns BrainData),
                1 = within images (returns array).

        Returns:
            float | np.ndarray | BrainData: Median values.
        """
        from .utils import apply_func

        return apply_func(self, np.median, axis)

    def multivariate_similarity(self, images, tail=2):
        """Predict a BrainData spatial distribution from a linear combination.

        The predictors may be other BrainData instances or nibabel images.

        Args:
            images (BrainData | Nifti1Image | list): Predictor image(s) — a
                BrainData stack of weight maps or nibabel images.
            tail (int | str): ``2`` or ``'two'`` for two-tailed (default); ``1`` or
                ``'one'`` for one-tailed (positive direction) regression p-values.

        Returns:
            dict: Regression statistics as BrainData instances, keyed
                `'beta'`, `'t'`, `'p'`, `'df'`, `'residual'`.
        """
        from .analysis import multivariate_similarity

        return multivariate_similarity(self, images, tail=tail)

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
        transparency="auto",
        colorbar=True,
        figsize=(12, 6),
        title=None,
        save=None,
    ):
        """Plot brain data on cortical flatmap.

        Args:
            threshold (float | str, optional): Absolute cutoff or percentile string.
            cmap (str, optional): Matplotlib colormap. Defaults are sign-aware.
            vmax (float, optional): Maximum value; inferred from displayed data.
            vmin (float, optional): Minimum value; inferred from displayed data.
            template (str): Freesurfer surface resolution. Default: 'fsaverage5'.
            transparency (BrainData, Nifti1Image, str, or "auto"): Binary mask
                used to render vertices outside the mask as transparent.
                ``"auto"`` (default) uses the instance's ``.mask``; pass
                ``None`` to disable masking.
            colorbar (bool): Show colorbar. Default: True.
            figsize (tuple): Figure size as (width, height). Default: (12, 6).
            title (str, optional): Figure title.
            save (str, optional): File path to save figure.

        Returns:
            matplotlib.figure.Figure: The rendered figure.
        """
        from nltools.plotting import plot_flatmap

        return plot_flatmap(
            self,
            threshold=threshold,
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
            template=template,
            transparency=transparency,
            colorbar=colorbar,
            figsize=figsize,
            title=title,
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
        colorbar=True,
        figsize=(10, 8),
        title=None,
        save=None,
    ):
        """Render this BrainData on fsaverage surfaces as a tight 2×2 montage.

        Facade over `plot_surf`. See that function's docstring for the full
        argument reference. Notable defaults: ``surface="pial"``,
        ``transparency="auto"`` (uses this instance's ``.mask``).

        Returns:
            matplotlib.figure.Figure: The rendered figure.
        """
        from nltools.plotting import plot_surf

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
            colorbar=colorbar,
            figsize=figsize,
            title=title,
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
        from .viewer import build_viewer, compute_display_window

        window = compute_display_window(
            self.data,
            autoscale=autoscale,
            threshold=threshold,
            lower=lower,
            upper=upper,
            symmetric=symmetric,
        )

        return build_viewer(
            self,
            view=view,
            window=window,
            cmap=cmap,
            atlas=atlas,
            bg_img=bg_img,
            opacity=opacity,
            outline=outline,
            colorbar=colorbar,
            controls=controls,
            niivue_opts=kwargs,
        )

    @overload
    def predict(
        self,
        *,
        X: "DesignMatrix | np.ndarray | Mapping[str, np.ndarray]",
        y: None = None,
        estimator: "str | BaseEstimator" = "linear_svc",
        cv: "int | BaseCrossValidator | None" = None,
        groups: "np.ndarray | str | None" = None,
        scoring: "str | Callable | None" = None,
        spatial_scale: Literal["whole_brain", "roi", "searchlight"] = "whole_brain",
        roi_mask: "Nifti1Image | str | Path | None" = None,
        radius: float = 10.0,
        n_jobs: int = 1,
        progress_bar: bool = False,
    ) -> "BrainData": ...

    @overload
    def predict(
        self,
        *,
        X: None = None,
        y: "np.ndarray | str | None" = None,
        estimator: "str | BaseEstimator" = "linear_svc",
        cv: "int | BaseCrossValidator | None" = None,
        groups: "np.ndarray | str | None" = None,
        scoring: "str | Callable | None" = None,
        spatial_scale: Literal["whole_brain", "roi", "searchlight"] = "whole_brain",
        roi_mask: "Nifti1Image | str | Path | None" = None,
        radius: float = 10.0,
        n_jobs: int = 1,
        progress_bar: bool = False,
    ) -> "Predict": ...

    @coalesced_gc()
    def predict(
        self,
        *,
        X: "DesignMatrix | np.ndarray | Mapping[str, np.ndarray] | None" = None,
        y: "np.ndarray | str | None" = None,
        estimator: "str | BaseEstimator" = "linear_svc",
        cv: "int | BaseCrossValidator | None" = None,
        groups: "np.ndarray | str | None" = None,
        scoring: "str | Callable | None" = None,
        spatial_scale: Literal["whole_brain", "roi", "searchlight"] = "whole_brain",
        roi_mask: "Nifti1Image | str | Path | None" = None,
        radius: float = 10.0,
        n_jobs: int = 1,
        progress_bar: bool = False,
    ):
        """Predict voxel responses from a fitted model, or decode labels with MVPA.

        Exactly one mode is resolved before any work happens:

        - an explicit ``y=`` runs MVPA decoding and returns a `Predict`;
        - an explicit ``X=`` predicts from the fitted `Glm` or `Ridge` and
          returns a new, independently owned `BrainData`;
        - with neither argument and a fitted model, an independent copy of the
          stored training predictions;
        - with neither argument, no fitted model, and exactly one ``.Y`` column,
          MVPA on that column.

        Supplying both ``X`` and ``y``, or a decoding argument on a
        fitted-model call, raises before prediction begins. A fitted model wins
        over an attached ``.Y`` on the no-argument call — pass ``y=``
        explicitly to decode instead. `predict` never mutates the source and
        attaches nothing to it.

        Labels travel with the data: ``y='name'`` picks a column of ``.Y``, and
        ``groups`` accepts a ``.Y`` column name the same way. With an explicit
        ``X=``, the estimator validates and aligns it: a `DesignMatrix` whose
        column names `Glm.predict` matches to the fitted order, or, for a
        banded `Ridge`, a mapping with exactly the fitted feature-space names
        in any order.

        Args:
            X (DesignMatrix | array-like | Mapping, optional): Features for
                fitted-model prediction, shape ``(n_samples, n_features)``, or a
                mapping of feature-space names to matrices for a banded `Ridge`.
            y (array-like | str, optional): Labels (classification) or
                continuous targets (regression), shape ``(n_samples,)``, or the
                name of a ``.Y`` column. Must be one-dimensional with one value
                per row; multioutput and multilabel targets are not accepted.
            estimator (str | sklearn estimator): A built-in shortcut —
                ``'linear_svc'``, ``'logistic_regression'``,
                ``'linear_discriminant_analysis'``, ``'ridge_classifier'``,
                ``'ridge'``, ``'lasso'``, ``'linear_svr'`` — or any sklearn
                estimator or `Pipeline`, which is used exactly as supplied.
                Default ``'linear_svc'``. Every shortcut standardizes voxels
                inside each fold and then fits a linear estimator; a
                classification shortcut on a multiclass target is wrapped in
                `OneVsRestClassifier`, so every class gets its own signed map.
                A caller-supplied estimator is never wrapped and never has its
                multiclass strategy overridden — pass a `OneVsRestClassifier`
                to get one. Every preprocessing step, in every spatial
                scale, must be one of `StandardScaler`, `PCA`,
                `VarianceThreshold`,
                `GenericUnivariateSelect`, `SelectPercentile`, `SelectKBest`,
                `SelectFpr`, `SelectFdr`, `SelectFwe`, `SelectFromModel`,
                `RFE`, `RFECV`, `SequentialFeatureSelector`, ``None``, or
                ``'passthrough'``. Whole-brain and ROI pipelines must also end
                in an estimator exposing ``coef_``, since those two scales
                extract a weight map; searchlight builds none and does not
                require it.
            cv (int | sklearn splitter, optional): ``None`` (the default) is a
                deterministic five-fold ``KFold`` (regression) or
                ``StratifiedKFold`` (classification); an int selects that many
                folds; an sklearn splitter is used as supplied. Test folds must
                partition the rows, so shuffle-split and repeated splitters
                raise. Rows ordered by condition make unshuffled contiguous
                folds degenerate — pass a shuffled splitter to control that,
                e.g. ``cv=KFold(n_splits=5, shuffle=True, random_state=0)``.
            groups (array-like | str, optional): Group labels passed to the
                splitter (e.g. ``LeaveOneGroupOut`` for leave-one-run-out), one
                value per row, or the name of a ``.Y`` column holding them.
            scoring (str | callable, optional): Follows scikit-learn's
                single-metric scoring contract. ``None`` (the default) uses the
                estimator's own ``score`` method; a scoring name or callable
                overrides it. Multimetric mappings are not accepted.
            spatial_scale (str): MVPA dispatch — ``'whole_brain'``, ``'roi'``,
                or ``'searchlight'``.
            roi_mask (Nifti1Image | path-like, optional): Atlas image; required
                by, and only valid for, ``spatial_scale='roi'``.
            radius (float): Searchlight sphere radius in millimeters; only
                valid for ``spatial_scale='searchlight'``. Default ``10.0``.
            n_jobs (int): Parallel workers for the outer independent work of
                the selected spatial scale — cross-validation folds for
                whole-brain, parcels for ROI, spheres for searchlight. Default
                ``1``; every worker holds a copy of the data, so a real brain
                at higher ``n_jobs`` can be memory-heavy.
            progress_bar (bool): Show a progress bar for searchlight and ROI.

        Returns:
            Predict | BrainData: A `Predict` record for MVPA; a new `BrainData`
                holding the predicted timeseries for fitted-model prediction.
                The record's ``spatial_scale`` says which of its fields carry
                values: whole-brain fills ``predictions``, ``cv_folds``,
                ``scores``, ``estimator`` and ``weight_map``; ROI fills
                ``scores``, ``roi_labels``, ``score_map`` and ``weight_map``;
                searchlight fills ``score_map`` alone. ``classes`` accompanies
                any classifier and ``scoring`` records the scoring
                specification in every mode. ``mean_score`` and ``std_score``
                are computed from ``scores`` on demand and do not exist for a
                searchlight result. ``weight_map`` holds one coefficient map
                for regression and binary classification (the signed map for
                ``classes[1]`` versus ``classes[0]``) and one map per class, in
                ``classes`` order, for multiclass — never an average across
                classes. It is projected back to voxel units through the
                pipeline's fitted preprocessing, but centering is not undone,
                so ``raw_data @ weight_map`` does not reproduce the decision
                function; use ``result.estimator`` to predict.

        Raises:
            ValueError: On both ``X`` and ``y``, a decoding argument on a
                fitted-model call, an unknown estimator shortcut or spatial
                scale, a target or group vector that is not one value per row,
                cross-validation folds that do not partition the rows, a
                preprocessing step outside the supported set, or — for
                whole-brain and ROI decoding — a pipeline whose coefficients
                cannot be projected back onto the voxel axis.
            TypeError: On a removed keyword, an `estimator` that is neither a
                shortcut name nor an object with `fit`/`predict`, or a `cv`
                that is neither `None`, an int, nor a splitter.

        Examples:
            Whole-brain decoding:

            ```python
            result = brain.predict(y=labels, cv=5)
            result.weight_map.plot()   # the all-data refit — the publishable map
            result.mean_score          # the cross-validated score
            new_pred = result.estimator.predict(new_X)
            ```

            Searchlight and ROI decoding:

            ```python
            result = brain.predict(
                y=labels, spatial_scale='searchlight', radius=8.0, n_jobs=4
            )
            result.score_map.plot()    # one score per sphere center

            result = brain.predict(y=labels, spatial_scale='roi', roi_mask=atlas)
            result.mean_score          # one score per parcel
            result.score_map.plot()    # those scores painted into voxel space
            ```

            Prediction from a fitted encoding model:

            ```python
            brain.fit(model='ridge', X=features)
            predicted = brain.predict(X=new_features)
            ```
        """
        from .prediction import predict

        return predict(
            self,
            X=X,
            y=y,
            estimator=estimator,
            cv=cv,
            groups=groups,
            scoring=scoring,
            spatial_scale=spatial_scale,
            roi_mask=roi_mask,
            radius=radius,
            n_jobs=n_jobs,
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

    def resample(self, *, img=None, resolution=None, interpolation=None):
        """Resample onto a new voxel grid, carrying the mask along.

        Exactly one of `img` or `resolution` is required. An `img` supplies
        only the target grid: its intensity values never define the output
        mask. The current mask is resampled onto the target grid with
        nearest-neighbor interpolation, so the result's voxel support is the
        source support expressed on the new grid. Row-aligned `X` and `Y`
        survive; fitted state does not.

        Args:
            img (Nifti1Image | str | Path | None): Target image supplying the
                grid to match.
            resolution (float | int | None): Target isotropic voxel size in mm.
            interpolation (str | None): Interpolation method for the data:
                ``'nearest'``, ``'linear'``, ``'continuous'``, or ``None`` to
                use the instance's setting.

        Returns:
            BrainData: New BrainData instance with resampled data and mask.

        Raises:
            ValueError: If both ``img`` and ``resolution`` are None, both are
                provided, or ``resolution`` is not positive.
            TypeError: If ``img`` is not a valid image type.

        Examples:
            ```python
            coarse = brain.resample(resolution=3.0)
            on_atlas_grid = brain.resample(img=atlas_img)
            ```
        """
        from .io import resample

        return resample(
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
    def similarity(self, data, *, metric="correlation"):
        """Calculate similarity to a single BrainData or nibabel image.

        Args:
            data (BrainData | Nifti1Image): Image to evaluate similarity against.
            metric (str): Type of similarity: ``'correlation'`` (default),
                ``'pearson'``, ``'rank_correlation'``, ``'spearman'``,
                ``'dot_product'``, or ``'cosine'``.

        Returns:
            float or np.ndarray: Similarity value(s).
        """
        from .analysis import similarity

        return similarity(self, data, metric=metric)

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

    def standardize(self, *, method="center", axis=0):
        """Standardize data by centering it, optionally scaling to unit variance.

        Constant voxels (or observations) z-score to 0 rather than NaN.

        Args:
            method (str): ``'center'`` subtracts the mean (default);
                ``'zscore'`` subtracts the mean and divides by the standard
                deviation.
            axis (int): 0 standardizes each voxel across observations (default).
                1 standardizes each observation across voxels.

        Returns:
            BrainData: Standardized BrainData instance.

        Raises:
            ValueError: If `method` is neither ``'center'`` nor ``'zscore'``.
        """
        from .analysis import standardize

        return standardize(self, method=method, axis=axis)

    def std(self, axis=0):
        """Get standard deviation of each voxel or image.

        Args:
            axis (int): 0 = across images (default, returns BrainData),
                1 = within images (returns array).

        Returns:
            float | np.ndarray | BrainData: Standard deviation values.
        """
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
        from nltools.data.atlases import ClusterReport, cluster_report_data
        from nltools.data.atlases.registry import DEFAULT_ATLASES

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
        progress_bar: bool = False,
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
            progress_bar (bool): If True, show a progress bar. Default False.

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
            progress_bar=progress_bar,
        )

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
