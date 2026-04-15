"""
BrainCollection: Multi-subject brain data container.

Provides tensor-like semantics for efficient group analyses with lazy loading
and memory-efficient operations.

Shape semantics: (n_images, n_observations, n_voxels)
    - axis 0: images (subjects, runs, etc.)
    - axis 1: observations (timepoints, TRs)
    - axis 2: voxels (spatial)
"""

from __future__ import annotations

import numpy as np
import nibabel as nib
import polars as pl
from pathlib import Path
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Iterator,
    TypeVar,
    Generator,
    Any,
)

from nltools.utils import attempt_to_import

# Re-exports from submodules (used by external code / tests)
from .io import _resolve_save_path  # noqa: F401
from .modeling import _build_subject_design_matrix, _fit_glm_by_run  # noqa: F401
from .pipeline import (  # noqa: F401
    BrainCollectionPipeline,
    BrainCollectionCVResult,
    FittedBrainCollection,
)

if TYPE_CHECKING:
    import pandas as pd

    from ..braindata import BrainData


def _coerce_metadata(
    metadata: "pl.DataFrame | pd.DataFrame | dict | None",
    n_items: int,
) -> pl.DataFrame:
    """Coerce metadata input to a polars DataFrame.

    Accepts polars/pandas DataFrame, dict-of-columns, or None (→ empty frame
    of length ``n_items``). Pandas is taken at the boundary as a convenience
    affordance; internal state is always polars.
    """
    if metadata is None:
        return pl.DataFrame()

    if isinstance(metadata, pl.DataFrame):
        out = metadata
    elif isinstance(metadata, dict):
        out = pl.DataFrame(metadata)
    else:
        try:
            import pandas as pd
        except ImportError:
            pd = None
        if pd is not None and isinstance(metadata, pd.DataFrame):
            out = pl.DataFrame(
                {str(c): metadata[c].to_numpy() for c in metadata.columns}
            )
        else:
            raise TypeError(
                "metadata must be a polars/pandas DataFrame, dict, or None. "
                f"Received {type(metadata).__name__}"
            )

    if not out.is_empty() and out.height != n_items:
        raise ValueError(
            f"metadata length ({out.height}) must match items length ({n_items})"
        )
    return out


# Lazy imports for optional dependencies
tqdm = attempt_to_import("tqdm", "tqdm")

T = TypeVar("T")

# Axis name mapping for intuitive access
_AXIS_NAMES = {
    "images": 0,
    "subjects": 0,
    "image": 0,
    "subject": 0,
    "observations": 1,
    "time": 1,
    "timepoints": 1,
    "obs": 1,
    "voxels": 2,
    "space": 2,
    "spatial": 2,
}


class BrainCollection:
    """
    Collection of brain images with tensor-like operations.

    BrainCollection provides a container for multiple brain images (e.g., multiple
    subjects or runs) with numpy-style indexing and axis operations. It supports
    lazy loading for memory efficiency and integrates with pybids for BIDS datasets.

    Shape semantics: (n_images, n_observations, n_voxels)
        - axis 0: images (subjects, runs, etc.)
        - axis 1: observations (timepoints, TRs)
        - axis 2: voxels (spatial locations)

    Args:
        items: List of file paths, BrainData objects, or mix of both.
            Paths are loaded lazily by default.
        mask: Brain mask. Required. Can be:
            - nibabel Nifti1Image
            - Path to mask file
            - Template name (e.g., '2mm-MNI152-2009c')
        metadata: Optional DataFrame with per-image metadata (subject, session, etc.).
            Index should match items order.
        lazy: If True (default), paths are not loaded until accessed.

    Examples:
        >>> # Create from paths (lazy loading)
        >>> bc = BrainCollection(
        ...     ['/data/sub-01.nii.gz', '/data/sub-02.nii.gz'],
        ...     mask='2mm-MNI152-2009c'
        ... )
        >>> bc.shape
        (2, 100, 228453)

        >>> # NumPy-style indexing
        >>> bc[0]  # First subject -> BrainData
        >>> bc[:, 0]  # First timepoint across all subjects -> BrainCollection
        >>> bc[0:5, 10:20]  # 5 subjects, timepoints 10-20 -> BrainCollection

        >>> # Axis operations
        >>> bc.mean(axis=0)  # Mean across subjects -> BrainData
        >>> bc.mean(axis=1)  # Mean across time per subject -> BrainCollection

        >>> # From BIDS dataset
        >>> bc = BrainCollection.from_bids('/data/bids', task='rest', mask=mask)

    Notes:
        - All images must share the same mask/space. Heterogeneous masks are not
          supported; data is resampled to mask space on load.
        - Some operations (e.g., to_tensor) require uniform observation counts
          across all images.
    """

    def __init__(
        self,
        items: list[Path | str | "BrainData"],
        mask: nib.Nifti1Image | Path | str,
        metadata: "pl.DataFrame | pd.DataFrame | dict | None" = None,
        lazy: bool = True,
    ):
        """Initialize BrainCollection.

        Args:
            items: List of paths or BrainData objects.
            mask: Shared mask (required). Path, nibabel image, or template name.
            metadata: Optional per-image metadata. Accepts polars/pandas
                DataFrame or dict-of-columns; stored as polars.
            lazy: If True, paths are loaded on demand.
        """
        from ..braindata import BrainData

        if not items:
            raise ValueError("items cannot be empty")

        # Resolve mask
        self._mask = self._resolve_mask(mask)
        self._n_voxels = int(self._mask.get_fdata().sum())

        # Store items (paths or loaded BrainData)
        self._items: list[Path | BrainData] = []
        self._is_loaded: list[bool] = []
        self._sample_counts: list[int | None] = []  # None if not yet known

        for item in items:
            if isinstance(item, (str, Path)):
                path = Path(item)
                if not path.exists():
                    raise FileNotFoundError(f"File not found: {path}")
                if lazy:
                    self._items.append(path)
                    self._is_loaded.append(False)
                    self._sample_counts.append(None)
                else:
                    bd = BrainData(path, mask=self._mask)
                    self._items.append(bd)
                    self._is_loaded.append(True)
                    self._sample_counts.append(self._get_n_obs(bd))
            elif isinstance(item, BrainData):
                # Validate mask compatibility
                self._validate_mask_compatibility(item)
                self._items.append(item)
                self._is_loaded.append(True)
                self._sample_counts.append(self._get_n_obs(item))
            else:
                raise TypeError(
                    f"Expected path or BrainData, got {type(item).__name__}"
                )

        # Metadata — always polars internally; accepts pandas/dict at ingress
        self._metadata = _coerce_metadata(metadata, len(self._items))

    def _resolve_mask(self, mask: nib.Nifti1Image | Path | str) -> nib.Nifti1Image:
        """Resolve mask to nibabel image."""
        if isinstance(mask, nib.Nifti1Image):
            return mask
        elif isinstance(mask, (str, Path)):
            path = Path(mask)
            if path.exists():
                return nib.Nifti1Image.from_filename(str(path))
            # Try as template name
            try:
                from nltools.templates import resolve_template_name

                mask_path = resolve_template_name(str(mask), file_type="mask")
                return nib.Nifti1Image.from_filename(str(mask_path))
            except (ValueError, FileNotFoundError):
                raise ValueError(
                    f"mask must be a path to a nifti file or a valid template name. "
                    f"Got: {mask}"
                )
        else:
            raise TypeError(
                f"mask must be nibabel image, path, or template name. "
                f"Got: {type(mask).__name__}"
            )

    def _validate_mask_compatibility(self, bd: "BrainData") -> None:
        """Validate that BrainData is compatible with collection mask."""
        if bd.mask is None:
            return  # Will be set on access

        # Check shape compatibility
        bd_shape = bd.mask.shape
        mask_shape = self._mask.shape
        if bd_shape != mask_shape:
            raise ValueError(
                f"BrainData mask shape {bd_shape} does not match "
                f"collection mask shape {mask_shape}. "
                "All images must share the same mask/space."
            )

    def _get_n_obs(self, bd: "BrainData") -> int:
        """Get number of observations from BrainData."""
        if bd.data.ndim == 1:
            return 1
        return bd.data.shape[0]

    def _load_item(self, idx: int) -> BrainData:
        """Load a single item if it's a path, return BrainData."""
        from ..braindata import BrainData

        item = self._items[idx]
        if isinstance(item, Path):
            bd = BrainData(item, mask=self._mask)
            self._items[idx] = bd
            self._is_loaded[idx] = True
            self._sample_counts[idx] = self._get_n_obs(bd)
            return bd
        return item  # type: ignore[return-value]

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def n_images(self) -> int:
        """Number of images in collection."""
        return len(self._items)

    @property
    def n_voxels(self) -> int:
        """Number of voxels (from mask)."""
        return self._n_voxels

    @property
    def mask(self) -> nib.Nifti1Image:
        """Shared NIfTI brain mask image used to define the voxel space for the collection."""
        return self._mask

    @property
    def metadata(self) -> pl.DataFrame:
        """Per-image metadata as a polars DataFrame."""
        return self._metadata

    @property
    def is_loaded(self) -> list[bool]:
        """List indicating which images are currently in memory."""
        return self._is_loaded.copy()

    @property
    def shape(self) -> tuple[int, int | None, int]:
        """
        Shape as (n_images, n_observations, n_voxels).

        n_observations is None if images have variable counts or not all are loaded.
        """
        # Check if we know all sample counts
        known_counts = [c for c in self._sample_counts if c is not None]
        if len(known_counts) == len(self._items):
            # All known - check if uniform
            if len(set(known_counts)) == 1:
                return (self.n_images, known_counts[0], self.n_voxels)
        return (self.n_images, None, self.n_voxels)

    # =========================================================================
    # Memory Management (Glass-box)
    # =========================================================================

    def memory_estimate(self) -> str:
        """
        Estimate memory usage for loading all images.

        Returns:
            Human-readable string like "12.4 GB total (1.2 GB per image avg)"
        """
        # If we know sample counts, use them; otherwise estimate from loaded
        known_counts = [c for c in self._sample_counts if c is not None]
        if known_counts:
            avg_obs = np.mean(known_counts)
        else:
            # Load first item to estimate
            self._load_item(0)
            avg_obs = self._sample_counts[0]

        bytes_per_element = 8  # float64
        bytes_per_image = avg_obs * self.n_voxels * bytes_per_element
        total_bytes = bytes_per_image * self.n_images

        def format_bytes(b: float) -> str:
            """Format a byte count as a human-readable string (KB, MB, or GB).

            Args:
                b: Byte count to format.

            Returns:
                Formatted string with one decimal place and appropriate unit.
            """
            if b >= 1e9:
                return f"{b / 1e9:.1f} GB"
            elif b >= 1e6:
                return f"{b / 1e6:.1f} MB"
            else:
                return f"{b / 1e3:.1f} KB"

        return (
            f"{format_bytes(total_bytes)} total "
            f"({format_bytes(bytes_per_image)} per image avg)"
        )

    def load(self, indices: list[int] | None = None) -> "BrainCollection":
        """
        Load specified images into memory.

        Args:
            indices: List of indices to load. If None, loads all.

        Returns:
            self (for chaining)
        """
        idx_iter = range(len(self._items)) if indices is None else indices

        for idx in idx_iter:
            self._load_item(idx)

        return self

    def unload(self, indices: list[int] | None = None) -> "BrainCollection":
        """
        Free memory for specified images (keep paths for reloading).

        Only works for items that were originally loaded from paths.

        Args:
            indices: List of indices to unload. If None, unloads all possible.

        Returns:
            self (for chaining)
        """
        from ..braindata import BrainData

        idx_iter = range(len(self._items)) if indices is None else indices

        for idx in idx_iter:
            item = self._items[idx]
            if isinstance(item, BrainData) and hasattr(item, "_source_path"):
                # Can only unload if we know the original path
                self._items[idx] = item._source_path
                self._is_loaded[idx] = False

        return self

    # =========================================================================
    # Dunder Methods
    # =========================================================================

    def __len__(self) -> int:
        """Number of images."""
        return self.n_images

    def __iter__(self) -> Iterator["BrainData"]:
        """Iterate over images with progress bar."""
        if tqdm is not None:
            iterator = tqdm.tqdm(range(len(self)), desc="Iterating images")
        else:
            iterator = range(len(self))

        for idx in iterator:
            yield self._load_item(idx)

    def __repr__(self) -> str:
        """String representation."""
        n_loaded = sum(self._is_loaded)
        shape_str = f"({self.n_images}, {self.shape[1] or '?'}, {self.n_voxels})"
        return (
            f"BrainCollection(shape={shape_str}, "
            f"loaded={n_loaded}/{self.n_images}, "
            f"mask_shape={self._mask.shape})"
        )

    def __getitem__(self, key) -> "BrainData | BrainCollection":
        """
        NumPy-style 3D indexing.

        Supports:
            bc[i]           -> BrainData (obs, voxels) - single image
            bc[i, j]        -> BrainData (voxels,) - single image, single obs
            bc[i, j, k]     -> scalar or array - single image, single obs, voxel slice
            bc[slice]       -> BrainCollection - subset of images
            bc[:, slice]    -> BrainCollection - all images, sliced observations
            bc['sub-01']    -> BrainData - by metadata lookup (if 'subject' column)

        Args:
            key: Index, slice, list, tuple, or string.

        Returns:
            BrainData for single-image access, BrainCollection for multi-image.
        """
        # String key: lookup by metadata
        if isinstance(key, str):
            return self._getitem_by_metadata(key)

        # Tuple: multi-dimensional indexing
        if isinstance(key, tuple):
            return self._getitem_multidim(key)

        # Single index
        if isinstance(key, int):
            return self._load_item(key)

        # Slice or list
        if isinstance(key, slice):
            indices = range(*key.indices(len(self)))
            return self._subset(list(indices))

        if isinstance(key, (list, np.ndarray)):
            return self._subset(list(key))

        raise TypeError(f"Invalid index type: {type(key).__name__}")

    def _getitem_by_metadata(self, key: str) -> "BrainData":
        """Get item by metadata value (e.g., subject ID)."""
        for col in ["subject", "subject_id", "sub", "id"]:
            if col in self._metadata.columns:
                mask = (self._metadata[col] == key).to_numpy()
                match_idx = np.flatnonzero(mask)
                if len(match_idx) == 1:
                    return self._load_item(int(match_idx[0]))
                elif len(match_idx) > 1:
                    raise KeyError(
                        f"Multiple images match '{key}' in column '{col}'. "
                        "Use integer indexing or more specific key."
                    )
        raise KeyError(
            f"No image found for key '{key}'. "
            "Ensure metadata has 'subject' column or use integer indexing."
        )

    def _getitem_multidim(self, key: tuple) -> "BrainData | BrainCollection":
        """Handle multi-dimensional indexing: bc[i, j] or bc[i, j, k]."""
        if len(key) == 0:
            raise IndexError("Empty index")

        # First dimension: images
        img_key = key[0]

        # Single image case
        if isinstance(img_key, int):
            bd = self._load_item(img_key)

            if len(key) == 1:
                return bd

            # Observation indexing
            obs_key = key[1]
            if isinstance(obs_key, int):
                # Single observation
                if bd.data.ndim == 1:
                    if obs_key != 0:
                        raise IndexError(
                            f"Observation index {obs_key} out of range for single image"
                        )
                    sliced_data = bd.data
                else:
                    sliced_data = bd.data[obs_key]

                # Return as BrainData with single observation
                from ..braindata import BrainData

                result = BrainData(mask=bd.mask)
                result.data = sliced_data
                return result

            elif isinstance(obs_key, slice):
                # Slice observations
                from ..braindata import BrainData

                if bd.data.ndim == 1:
                    sliced_data = bd.data[np.newaxis, :][obs_key]
                else:
                    sliced_data = bd.data[obs_key]

                result = BrainData(mask=bd.mask)
                result.data = sliced_data
                return result

            else:
                raise TypeError(f"Invalid observation index type: {type(obs_key)}")

        # Multiple images case
        if isinstance(img_key, slice):
            indices = list(range(*img_key.indices(len(self))))
        elif isinstance(img_key, (list, np.ndarray)):
            indices = list(img_key)
        else:
            raise TypeError(f"Invalid image index type: {type(img_key)}")

        # Create subset collection
        subset = self._subset(indices)

        if len(key) == 1:
            return subset

        # Apply observation slicing to each image
        obs_key = key[1]
        if isinstance(obs_key, (int, slice)):
            # Apply obs indexing via apply
            def slice_obs(bd: "BrainData") -> "BrainData":
                """Slice observations from a BrainData object using the captured obs_key.

                Handles 1-D data by temporarily expanding to 2-D before indexing.

                Args:
                    bd: BrainData object to slice.

                Returns:
                    New BrainData containing only the selected observation(s).
                """
                from ..braindata import BrainData

                if bd.data.ndim == 1:
                    data = bd.data[np.newaxis, :]
                else:
                    data = bd.data

                if isinstance(obs_key, int):
                    sliced = data[obs_key]
                else:
                    sliced = data[obs_key]

                result = BrainData(mask=bd.mask)
                result.data = sliced
                return result

            # Apply without creating new collection
            new_items = [slice_obs(subset._load_item(i)) for i in range(len(subset))]
            return BrainCollection(
                new_items, mask=self._mask, metadata=subset._metadata
            )

        raise TypeError(f"Invalid observation index type: {type(obs_key)}")

    def _subset(self, indices: list[int]) -> "BrainCollection":
        """Create a new BrainCollection with subset of items."""
        new_items = [self._items[i] for i in indices]
        if self._metadata.is_empty():
            new_metadata = self._metadata
        else:
            new_metadata = self._metadata[list(indices)]

        # Create new collection without re-validating
        new_bc = object.__new__(BrainCollection)
        new_bc._mask = self._mask
        new_bc._n_voxels = self._n_voxels
        new_bc._items = new_items
        new_bc._is_loaded = [self._is_loaded[i] for i in indices]
        new_bc._sample_counts = [self._sample_counts[i] for i in indices]
        new_bc._metadata = new_metadata

        return new_bc

    # =========================================================================
    # Construction Class Methods
    # =========================================================================

    @classmethod
    def from_bids(
        cls,
        layout: Any,  # BIDSLayout or path
        mask: nib.Nifti1Image | Path | str,
        *,
        task: str | None = None,
        subject: str | list[str] | None = None,
        session: str | list[str] | None = None,
        run: int | list[int] | None = None,
        space: str | None = None,
        suffix: str = "bold",
        extension: str = "nii.gz",
        **bids_filters,
    ) -> "BrainCollection":
        """
        Create BrainCollection from a BIDS dataset.

        Requires pybids to be installed: `pip install pybids`

        Args:
            layout: pybids BIDSLayout object or path to BIDS dataset.
            mask: Shared mask (required).
            task: BIDS task filter.
            subject: Subject ID(s) to include.
            session: Session ID(s) to include.
            run: Run number(s) to include.
            space: BIDS space filter (e.g., 'MNI152NLin2009cAsym').
            suffix: BIDS suffix (default 'bold').
            extension: File extension (default 'nii.gz').
            **bids_filters: Additional BIDS entity filters.

        Returns:
            BrainCollection with metadata extracted from BIDS entities.

        Examples:
            >>> bc = BrainCollection.from_bids(
            ...     '/data/bids_dataset',
            ...     mask='2mm-MNI152-2009c',
            ...     task='rest',
            ...     space='MNI152NLin2009cAsym'
            ... )
        """
        from .constructors import from_bids

        return from_bids(
            layout,
            mask,
            task=task,
            subject=subject,
            session=session,
            run=run,
            space=space,
            suffix=suffix,
            extension=extension,
            **bids_filters,
        )

    @classmethod
    def from_glob(
        cls,
        pattern: str,
        mask: nib.Nifti1Image | Path | str,
        *,
        pattern_groups: dict[str, int] | str | None = None,
        sort: bool = True,
    ) -> "BrainCollection":
        """
        Create BrainCollection from glob pattern.

        Args:
            pattern: Glob pattern (e.g., ``'/data/*/func/*_bold.nii.gz'``).
            mask: Shared mask (required).
            pattern_groups: Regex pattern with named groups for metadata extraction.
                Example: ``r'sub-(?P<subject>\\w+)/.*run-(?P<run>\\d+)'``
            sort: Sort files alphabetically (default True).

        Returns:
            BrainCollection with optional metadata from pattern groups.

        Examples:
            >>> bc = BrainCollection.from_glob(
            ...     '/data/sub-*/func/*_bold.nii.gz',
            ...     mask=mask,
            ...     pattern_groups=r'sub-(?P<subject>\\w+)'
            ... )
        """
        from .constructors import from_glob

        return from_glob(pattern, mask, pattern_groups=pattern_groups, sort=sort)

    @classmethod
    def from_stacked(
        cls,
        brain_data: "BrainData",
        splits: list[int] | None = None,
        n_images: int | None = None,
    ) -> "BrainCollection":
        """
        Create BrainCollection by splitting a stacked BrainData.

        Args:
            brain_data: BrainData with shape (n_total_obs, n_voxels).
            splits: List of observation counts per image. Must sum to n_total_obs.
            n_images: Number of images (splits evenly). Mutually exclusive with splits.

        Returns:
            BrainCollection with data split according to specification.

        Examples:
            >>> # Split evenly into 3 images
            >>> bc = BrainCollection.from_stacked(bd, n_images=3)

            >>> # Split with explicit counts
            >>> bc = BrainCollection.from_stacked(bd, splits=[100, 100, 150])
        """
        from .constructors import from_stacked

        return from_stacked(brain_data, splits=splits, n_images=n_images)

    # =========================================================================
    # Axis Operations (to be implemented in nltools-cyb)
    # =========================================================================

    def _normalize_axis(
        self, axis: int | str | tuple[int | str, ...]
    ) -> int | tuple[int, ...]:
        """Convert axis name to integer."""
        if isinstance(axis, str):
            if axis.lower() not in _AXIS_NAMES:
                raise ValueError(
                    f"Unknown axis name: {axis}. "
                    f"Valid names: {list(_AXIS_NAMES.keys())}"
                )
            return _AXIS_NAMES[axis.lower()]
        if isinstance(axis, tuple):
            normalized: list[int] = []
            for a in axis:
                result = self._normalize_axis(a)
                if isinstance(result, tuple):
                    raise ValueError("Nested tuple axes are not supported")
                normalized.append(result)
            return tuple(normalized)
        return axis

    def _aggregate_axis0(
        self,
        func: str,
        batch_size: int | None = None,
    ) -> "BrainData":
        """Aggregate across images (axis=0) using streaming algorithm."""
        from ..braindata import BrainData

        # Ensure all sample counts are known
        for i in range(len(self)):
            if self._sample_counts[i] is None:
                self._load_item(i)

        # Check uniform observation counts
        unique_counts = set(self._sample_counts)
        if len(unique_counts) > 1:
            raise ValueError(
                f"Cannot aggregate axis=0: images have variable observation counts "
                f"{sorted(unique_counts)}. Use apply() for per-image operations."
            )

        n_obs = self._sample_counts[0]

        if func == "mean":
            # Welford's online mean algorithm
            running_mean = np.zeros((n_obs, self.n_voxels))
            count = 0

            iterator = range(self.n_images)
            if tqdm is not None:
                iterator = tqdm.tqdm(iterator, desc="Computing mean (axis=0)")

            for i in iterator:
                bd = self._load_item(i)
                data = bd.data if bd.data.ndim == 2 else bd.data[np.newaxis, :]
                count += 1
                delta = data - running_mean
                running_mean += delta / count

            result = BrainData(mask=self._mask)
            result.data = running_mean
            return result

        elif func == "sum":
            running_sum = np.zeros((n_obs, self.n_voxels))
            for i in range(self.n_images):
                bd = self._load_item(i)
                data = bd.data if bd.data.ndim == 2 else bd.data[np.newaxis, :]
                running_sum += data
            result = BrainData(mask=self._mask)
            result.data = running_sum
            return result

        elif func in ("std", "var"):
            # Welford's online variance algorithm
            running_mean = np.zeros((n_obs, self.n_voxels))
            running_m2 = np.zeros((n_obs, self.n_voxels))
            count = 0

            for i in range(self.n_images):
                bd = self._load_item(i)
                data = bd.data if bd.data.ndim == 2 else bd.data[np.newaxis, :]
                count += 1
                delta = data - running_mean
                running_mean += delta / count
                delta2 = data - running_mean
                running_m2 += delta * delta2

            variance = running_m2 / max(count - 1, 1)  # Sample variance
            result = BrainData(mask=self._mask)
            result.data = np.sqrt(variance) if func == "std" else variance
            return result

        elif func in ("min", "max"):
            agg_func = np.minimum if func == "min" else np.maximum
            running = None
            for i in range(self.n_images):
                bd = self._load_item(i)
                data = bd.data if bd.data.ndim == 2 else bd.data[np.newaxis, :]
                if running is None:
                    running = data.copy()
                else:
                    running = agg_func(running, data)
            result = BrainData(mask=self._mask)
            result.data = running
            return result

        elif func == "median":
            # Median requires all data in memory
            tensor = self.to_tensor()
            median_data = np.median(tensor, axis=0)
            result = BrainData(mask=self._mask)
            result.data = median_data
            return result

        else:
            raise ValueError(f"Unknown aggregation function: {func}")

    def _aggregate_axis1(self, func: str) -> "BrainCollection":
        """Aggregate across observations (axis=1) per image."""
        from ..braindata import BrainData

        agg_func = getattr(np, func)
        new_items = []

        iterator = range(self.n_images)
        if tqdm is not None:
            iterator = tqdm.tqdm(iterator, desc=f"Computing {func} (axis=1)")

        for i in iterator:
            bd = self._load_item(i)
            data = bd.data if bd.data.ndim == 2 else bd.data[np.newaxis, :]
            agg_data = agg_func(data, axis=0)
            new_bd = BrainData(mask=self._mask)
            new_bd.data = agg_data
            new_items.append(new_bd)

        return BrainCollection(new_items, mask=self._mask, metadata=self._metadata)

    def _aggregate_axis2(self, func: str) -> np.ndarray:
        """Aggregate across voxels (axis=2) -> numpy array."""
        agg_func = getattr(np, func)
        results = []

        for i in range(self.n_images):
            bd = self._load_item(i)
            data = bd.data if bd.data.ndim == 2 else bd.data[np.newaxis, :]
            agg_data = agg_func(data, axis=1)  # Aggregate over voxels
            results.append(agg_data)

        return np.array(results)

    def mean(
        self,
        axis: int | str | tuple[int, ...] = 0,
        batch_size: int | None = None,
    ) -> "BrainData | BrainCollection | np.ndarray":
        """
        Compute mean along axis.

        Args:
            axis: Axis or axes to aggregate:
                - 0 or 'images': Mean across images -> BrainData (n_obs, n_voxels)
                - 1 or 'time': Mean across time -> BrainCollection (n_images, n_voxels)
                - 2 or 'voxels': Mean across voxels -> np.ndarray (n_images, n_obs)
                - (0, 1): Mean across images and time -> BrainData (n_voxels,)
            batch_size: Number of images to process at once (for memory efficiency).
                If None, uses streaming algorithm.

        Returns:
            BrainData, BrainCollection, or np.ndarray depending on axis.

        Examples:
            >>> bc.mean(axis=0)  # Mean across subjects
            >>> bc.mean(axis='images')  # Same as above
            >>> bc.mean(axis=1)  # Mean across time per subject
            >>> bc.mean(axis=(0, 1))  # Grand mean
        """
        return self._aggregate("mean", axis, batch_size)

    def std(
        self,
        axis: int | str | tuple[int, ...] = 0,
        batch_size: int | None = None,
    ) -> "BrainData | BrainCollection | np.ndarray":
        """Compute standard deviation along axis. See mean() for details."""
        return self._aggregate("std", axis, batch_size)

    def var(
        self,
        axis: int | str | tuple[int, ...] = 0,
        batch_size: int | None = None,
    ) -> "BrainData | BrainCollection | np.ndarray":
        """Compute variance along axis. See mean() for details."""
        return self._aggregate("var", axis, batch_size)

    def sum(
        self,
        axis: int | str | tuple[int, ...] = 0,
        batch_size: int | None = None,
    ) -> "BrainData | BrainCollection | np.ndarray":
        """Compute sum along axis. See mean() for details."""
        return self._aggregate("sum", axis, batch_size)

    def min(
        self,
        axis: int | str | tuple[int, ...] = 0,
        batch_size: int | None = None,
    ) -> "BrainData | BrainCollection | np.ndarray":
        """Compute minimum along axis. See mean() for details."""
        return self._aggregate("min", axis, batch_size)

    def max(
        self,
        axis: int | str | tuple[int, ...] = 0,
        batch_size: int | None = None,
    ) -> "BrainData | BrainCollection | np.ndarray":
        """Compute maximum along axis. See mean() for details."""
        return self._aggregate("max", axis, batch_size)

    def median(
        self,
        axis: int | str | tuple[int, ...] = 0,
        batch_size: int | None = None,
    ) -> "BrainData | BrainCollection | np.ndarray":
        """Compute median along axis. See mean() for details."""
        return self._aggregate("median", axis, batch_size)

    def _aggregate(
        self,
        func: str,
        axis: int | str | tuple[int, ...],
        batch_size: int | None = None,
    ) -> "BrainData | BrainCollection | np.ndarray":
        """Dispatch aggregation to appropriate axis handler."""

        axis = self._normalize_axis(axis)

        # Handle tuple of axes
        if isinstance(axis, tuple):
            # Sort axes to process in order
            axes = sorted(axis)
            result = self
            for ax in reversed(axes):
                # After each reduction, axis indices shift
                result = result._aggregate(func, ax, batch_size)
            return result

        if axis == 0:
            return self._aggregate_axis0(func, batch_size)
        elif axis == 1:
            return self._aggregate_axis1(func)
        elif axis == 2:
            return self._aggregate_axis2(func)
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, 2, or tuple.")

    # =========================================================================
    # Conversion Methods
    # =========================================================================

    def to_tensor(
        self,
        batch_size: int | None = None,
    ) -> np.ndarray | Generator[np.ndarray, None, None]:
        """
        Convert to numpy array (n_images, n_obs, n_voxels).

        Args:
            batch_size: If specified, returns generator yielding batches.

        Returns:
            Full tensor if batch_size is None, otherwise generator.

        Raises:
            ValueError: If images have variable observation counts.

        Examples:
            >>> tensor = bc.to_tensor()  # Full array
            >>> tensor.shape
            (3, 100, 50000)

            >>> # Batched iteration
            >>> for batch in bc.to_tensor(batch_size=10):
            ...     process(batch)  # batch.shape = (10, 100, 50000)
        """
        # First, ensure all sample counts are known
        for i in range(len(self)):
            if self._sample_counts[i] is None:
                self._load_item(i)

        # Check for uniform observation counts
        unique_counts = set(self._sample_counts)
        if len(unique_counts) > 1:
            raise ValueError(
                f"Cannot convert to tensor: images have variable observation counts "
                f"{sorted(unique_counts)}. Use to_list() instead."
            )

        n_obs = self._sample_counts[0]

        if batch_size is not None:
            return self._to_tensor_batched(batch_size, n_obs)

        # Full tensor
        tensor = np.zeros((self.n_images, n_obs, self.n_voxels))
        for i in range(self.n_images):
            bd = self._load_item(i)
            data = bd.data
            if data.ndim == 1:
                data = data[np.newaxis, :]
            tensor[i] = data

        return tensor

    def _to_tensor_batched(
        self, batch_size: int, n_obs: int
    ) -> Generator[np.ndarray, None, None]:
        """Generator yielding batches of the tensor."""
        for start in range(0, self.n_images, batch_size):
            end = min(start + batch_size, self.n_images)
            batch_tensor = np.zeros((end - start, n_obs, self.n_voxels))
            for i, idx in enumerate(range(start, end)):
                bd = self._load_item(idx)
                data = bd.data
                if data.ndim == 1:
                    data = data[np.newaxis, :]
                batch_tensor[i] = data
            yield batch_tensor

    def to_list(self) -> list["BrainData"]:
        """
        Return list of BrainData objects.

        Loads all items if not already loaded.

        Returns:
            List of BrainData objects.
        """
        return [self._load_item(i) for i in range(len(self))]

    def to_stacked(self) -> "BrainData":
        """
        Stack all into single BrainData (n_total_obs, n_voxels).

        Returns:
            Single BrainData with all observations concatenated.

        Examples:
            >>> bc = BrainCollection([bd1, bd2, bd3], mask=mask)
            >>> stacked = bc.to_stacked()
            >>> stacked.shape
            (300, 50000)  # 3 images * 100 obs each
        """
        from ..braindata import BrainData

        # Collect all data
        all_data = []
        for i in range(len(self)):
            bd = self._load_item(i)
            data = bd.data
            if data.ndim == 1:
                data = data[np.newaxis, :]
            all_data.append(data)

        stacked_data = np.vstack(all_data)

        result = BrainData(mask=self._mask)
        result.data = stacked_data
        return result

    def write(
        self,
        directory: str | Path,
        pattern: str = "image_{i:04d}.nii.gz",
        metadata_file: str | None = "metadata.csv",
    ) -> list[Path]:
        """Write all images in collection to files.

        Args:
            directory: Output directory path. Will be created if it doesn't exist.
            pattern: Filename pattern with {i} placeholder for image index.
                Default: "image_{i:04d}.nii.gz" produces image_0000.nii.gz, etc.
            metadata_file: Optional filename for metadata CSV. Set to None to skip.
                Default: "metadata.csv"

        Returns:
            List of paths to written files.

        Examples:
            >>> bc = BrainCollection([bd1, bd2, bd3], mask=mask)
            >>> paths = bc.write("output/")
            >>> # Creates: output/image_0000.nii.gz, image_0001.nii.gz, etc.

            >>> # Custom pattern
            >>> bc.write("output/", pattern="sub-{i:02d}_bold.nii.gz")
            >>> # Creates: output/sub-00_bold.nii.gz, sub-01_bold.nii.gz, etc.

            >>> # With BIDS-style naming using metadata
            >>> bc.metadata["filename"] = [f"sub-{s}_bold.nii.gz" for s in subjects]
            >>> for i, bd in enumerate(bc):
            ...     bd.write(f"output/{bc.metadata.loc[i, 'filename']}")
        """
        from .io import write

        return write(self, directory, pattern, metadata_file)

    def iter_batches(
        self,
        batch_size: int,
        axis: int = 0,
        progress_bar: bool = False,
    ) -> Generator["BrainCollection", None, None]:
        """
        Iterate in batches along axis.

        Args:
            batch_size: Number of items per batch.
            axis: Axis to batch along:
                - 0: Batches of images (default)
                - 1: Batches of timepoints (within each image)
            progress_bar: Show tqdm progress bar.

        Yields:
            BrainCollection for each batch.

        Examples:
            >>> # Batch over images
            >>> for batch in bc.iter_batches(batch_size=5):
            ...     process(batch)  # batch is BrainCollection with 5 images

            >>> # Batch over time
            >>> for batch in bc.iter_batches(batch_size=10, axis=1):
            ...     process(batch)  # batch has 10 timepoints per image
        """
        axis = self._normalize_axis(axis)

        if axis == 0:
            # Batch over images
            n_batches = int(np.ceil(self.n_images / batch_size))
            iterator = range(n_batches)

            if progress_bar and tqdm is not None:
                iterator = tqdm.tqdm(iterator, desc="Batching images", total=n_batches)

            for batch_idx in iterator:
                start = batch_idx * batch_size
                end = min(start + batch_size, self.n_images)
                yield self._subset(list(range(start, end)))

        elif axis == 1:
            # Batch over observations - requires uniform obs counts
            for i in range(len(self)):
                if self._sample_counts[i] is None:
                    self._load_item(i)

            unique_counts = set(self._sample_counts)
            if len(unique_counts) > 1:
                raise ValueError(
                    "Cannot batch over observations with variable counts. "
                    f"Found counts: {sorted(unique_counts)}"
                )

            n_obs = self._sample_counts[0]
            n_batches = int(np.ceil(n_obs / batch_size))
            iterator = range(n_batches)

            if progress_bar and tqdm is not None:
                iterator = tqdm.tqdm(
                    iterator, desc="Batching observations", total=n_batches
                )

            for batch_idx in iterator:
                start = batch_idx * batch_size
                end = min(start + batch_size, n_obs)
                # Slice observations for each image
                batch = self[:, start:end]
                assert isinstance(batch, BrainCollection)
                yield batch

        else:
            raise ValueError(f"Cannot batch over axis {axis}. Use axis=0 or axis=1.")

    # =========================================================================
    # Group Inference Methods
    # =========================================================================

    def ttest(
        self,
        popmean: float = 0.0,
        axis: int | str = 0,
    ) -> tuple["BrainData", "BrainData"]:
        """
        One-sample t-test across images.

        Tests whether the mean across images is significantly different from
        a population mean (default: 0). This is the voxel-wise equivalent of
        scipy.stats.ttest_1samp.

        Args:
            popmean: Population mean to test against (default: 0).
            axis: Axis to test across. Only axis=0 (images) supported.

        Returns:
            Tuple of (t_stat, p_value) as BrainData objects.
            Both have shape (n_obs, n_voxels) if uniform obs counts.

        Raises:
            ValueError: If images have variable observation counts.

        Examples:
            >>> t_stat, p_val = bc.ttest()  # Test mean != 0
            >>> t_stat, p_val = bc.ttest(popmean=0.5)  # Test mean != 0.5

            >>> # Threshold significant voxels
            >>> sig_mask = p_val.data < 0.05
        """
        from .inference import ttest

        return ttest(self, popmean, axis)

    def ttest2(
        self,
        other: "BrainCollection",
        equal_var: bool = True,
    ) -> tuple["BrainData", "BrainData"]:
        """
        Two-sample t-test between collections.

        Tests whether two collections have different means. This is the
        voxel-wise equivalent of scipy.stats.ttest_ind.

        Args:
            other: Another BrainCollection to compare against.
            equal_var: If True (default), perform standard t-test assuming
                equal variances. If False, use Welch's t-test.

        Returns:
            Tuple of (t_stat, p_value) as BrainData objects.

        Raises:
            ValueError: If collections have different masks or variable obs counts.

        Examples:
            >>> t_stat, p_val = patients.ttest2(controls)
            >>> t_stat, p_val = group1.ttest2(group2, equal_var=False)  # Welch's
        """
        from .inference import ttest2

        return ttest2(self, other, equal_var)

    def permutation_test(
        self,
        n_permute: int = 5000,
        tail: int = 2,
        device: str = "cpu",
        max_gpu_memory_gb: float = 4.0,
        return_null: bool = False,
        n_jobs: int = -1,
        random_state: int | None = None,
    ) -> dict:
        """
        One-sample permutation test across images (sign-flipping).

        Tests whether the mean across images is significantly different from
        zero using sign-flipping permutation. More robust than parametric
        t-test for non-normal distributions.

        This is a collection-level interface to
        nltools.algorithms.inference.one_sample_permutation_test.

        Args:
            n_permute: Number of permutations (default: 5000).
            tail: Test type - 1 for one-tailed, 2 for two-tailed (default).
            device: Compute device: 'cpu' (default) or 'gpu' (via PyTorch).
            max_gpu_memory_gb: GPU memory budget (default: 4.0 GB).
            return_null: If True, include null distribution in result.
            n_jobs: Number of CPU jobs (-1 = all cores, 1 = single-threaded).
            random_state: Random seed for reproducibility.

        Returns:
            dict with keys:
                - 'mean': BrainData with observed mean across images
                - 'p': BrainData with p-values
                - 'null_dist': np.ndarray (if return_null=True)
                - 'device': compute device used

        Raises:
            ValueError: If images have variable observation counts.

        Examples:
            >>> result = bc.permutation_test(n_permute=5000)
            >>> mean_bd, p_bd = result['mean'], result['p']

            >>> # With GPU acceleration
            >>> result = bc.permutation_test(device='gpu')
        """
        from .inference import permutation_test

        return permutation_test(
            self,
            n_permute,
            tail,
            device,
            max_gpu_memory_gb,
            return_null,
            n_jobs,
            random_state,
        )

    def permutation_test2(
        self,
        other: "BrainCollection",
        n_permute: int = 5000,
        tail: int = 2,
        device: str = "cpu",
        max_gpu_memory_gb: float = 4.0,
        return_null: bool = False,
        n_jobs: int = -1,
        random_state: int | None = None,
    ) -> dict:
        """
        Two-sample permutation test between collections.

        Tests whether two collections have different means using group
        label permutation. More robust than parametric t-test.

        Args:
            other: Another BrainCollection to compare against.
            n_permute: Number of permutations (default: 5000).
            tail: Test type - 1 for one-tailed, 2 for two-tailed (default).
            device: Compute device: 'cpu' (default) or 'gpu' (via PyTorch).
            max_gpu_memory_gb: GPU memory budget (default: 4.0 GB).
            return_null: If True, include null distribution in result.
            n_jobs: Number of CPU jobs (-1 = all cores, 1 = single-threaded).
            random_state: Random seed for reproducibility.

        Returns:
            dict with keys:
                - 'mean_diff': BrainData with observed mean difference
                - 'p': BrainData with p-values
                - 'null_dist': np.ndarray (if return_null=True)
                - 'device': compute device used

        Examples:
            >>> result = patients.permutation_test2(controls)
            >>> diff_bd, p_bd = result['mean_diff'], result['p']
        """
        from .inference import permutation_test2

        return permutation_test2(
            self,
            other,
            n_permute,
            tail,
            device,
            max_gpu_memory_gb,
            return_null,
            n_jobs,
            random_state,
        )

    def anova(
        self,
        groups: str | list | np.ndarray,
    ) -> tuple["BrainData", "BrainData"]:
        """
        One-way ANOVA across groups defined by metadata.

        Tests whether group means differ significantly. This is the
        voxel-wise equivalent of scipy.stats.f_oneway.

        Args:
            groups: Group assignment for each image. Can be:
                - str: Column name in metadata
                - list/array: Group labels of length n_images

        Returns:
            Tuple of (F_stat, p_value) as BrainData objects.

        Raises:
            ValueError: If groups length doesn't match n_images.
            KeyError: If group column not found in metadata.

        Examples:
            >>> # Groups from metadata column
            >>> f_stat, p_val = bc.anova('condition')

            >>> # Explicit group labels
            >>> groups = ['control'] * 10 + ['patient'] * 15
            >>> f_stat, p_val = bc.anova(groups)
        """
        from .inference import anova

        return anova(self, groups)

    # =========================================================================
    # Transformation Methods
    # =========================================================================

    def map(
        self,
        fn: Callable,
        axis: int | str = 0,
        n_jobs: int = 1,
        progress_bar: bool = False,
    ) -> "BrainCollection":
        """
        Apply function across specified axis.

        This is the general-purpose transformation method. For common operations,
        use convenience methods like standardize(), smooth(), etc.

        Args:
            fn: Function to apply. Signature depends on axis:
                - axis=0: fn(BrainData) → BrainData (per image)
                - axis=1: fn(BrainData) → BrainData (per timepoint slice)
                - axis=2: fn(ndarray[n_obs]) → ndarray (per voxel timeseries)
            axis: Axis to iterate over:
                - 0 or 'images': Apply fn to each image independently
                - 1 or 'time': Apply fn to each timepoint across images
                - 2 or 'voxels': Apply fn to each voxel timeseries per image
            n_jobs: Number of parallel jobs. -1 for all cores. Default 1.
            progress_bar: Show tqdm progress bar. Default True.

        Returns:
            BrainCollection with transformed data.

        Examples:
            >>> # Per-image operation
            >>> bc.map(lambda bd: bd.standardize())

            >>> # Per-voxel timeseries (e.g., detrend each voxel)
            >>> from scipy.signal import detrend
            >>> bc.map(detrend, axis=2)

            >>> # Parallel processing
            >>> bc.map(expensive_fn, n_jobs=-1)
        """
        from .transforms import map_collection

        return map_collection(self, fn, axis, n_jobs, progress_bar)

    def _map_axis0(
        self,
        fn: Callable,
        n_jobs: int,
        progress_bar: bool,
    ) -> "BrainCollection":
        """Map function over images (axis=0)."""
        from .transforms import map_axis0

        return map_axis0(self, fn, n_jobs, progress_bar)

    def _map_axis1(
        self,
        fn: Callable,
        n_jobs: int,
        progress_bar: bool,
    ) -> "BrainCollection":
        """Map function over timepoints (axis=1)."""
        from .transforms import map_axis1

        return map_axis1(self, fn, n_jobs, progress_bar)

    def _map_axis2(
        self,
        fn: Callable,
        n_jobs: int,
        progress_bar: bool,
    ) -> "BrainCollection":
        """Map function over voxels (axis=2) per image."""
        from .transforms import map_axis2

        return map_axis2(self, fn, n_jobs, progress_bar)

    def filter(
        self,
        predicate: "Callable | list | np.ndarray | pl.Series | pd.Series",
    ) -> "BrainCollection":
        """
        Filter collection by predicate.

        Args:
            predicate: Filter condition. Can be:
                - callable: fn(BrainData) → bool
                - list/ndarray: Boolean mask of length n_images
                - pl.Series / pd.Series: Boolean series

        Returns:
            BrainCollection with subset of images matching predicate.

        Examples:
            >>> # Filter by callable
            >>> bc.filter(lambda bd: bd.data.mean() > 0)

            >>> # Filter by boolean mask
            >>> mask = [True, False, True]
            >>> bc.filter(mask)

            >>> # Filter by metadata condition
            >>> bc.filter(bc.metadata['group'] == 'control')
        """
        from .transforms import filter_collection

        return filter_collection(self, predicate)

    def align(
        self,
        method: str = "procrustes",
        scheme: str = "searchlight",
        radius_mm: float = 10.0,
        parcellation: "nib.Nifti1Image | None" = None,
        n_features: int | None = None,
        n_iter: int = 3,
        device: str = "cpu",
        return_model: bool = False,
        n_jobs: int = -1,
        progress_bar: bool = False,
    ) -> "BrainCollection | tuple[BrainCollection, object]":
        """
        Align subjects using local functional alignment.

        Performs neighborhood-based functional alignment across subjects using
        LocalAlignment. Each subject's data is aligned to a common template space
        using local transforms learned within searchlight spheres or parcels.

        Args:
            method: Alignment method. Options:
                - 'procrustes': Orthogonal Procrustes (default, preserves dimensions)
                - 'srm': Shared Response Model (dimensionality reduction)
                - 'hyperalignment': Hyperalignment (iterative Procrustes)
            scheme: Spatial scheme. Options:
                - 'searchlight': Overlapping spheres with center-only aggregation
                - 'piecewise': Non-overlapping parcels (requires parcellation)
            radius_mm: Sphere radius in millimeters for searchlight scheme.
            parcellation: Parcellation image for piecewise scheme (required if
                scheme='piecewise').
            n_features: Number of features for SRM. None uses full dimensions.
            n_iter: Number of iterations for alignment refinement.
            device: Compute device: 'cpu' (default) or 'gpu' (via PyTorch).
            return_model: If True, return (aligned_collection, model) tuple for
                fit/transform workflow with new data.
            n_jobs: Number of CPU jobs (-1 = all cores, 1 = single-threaded).
            progress_bar: Show progress bar during fitting.

        Returns:
            BrainCollection with aligned data. If return_model=True, returns
            tuple of (aligned_collection, LocalAlignment_model).

        Examples:
            >>> # Basic searchlight alignment
            >>> aligned_bc = bc.align(method='procrustes', radius_mm=10.0)

            >>> # Piecewise alignment with parcellation
            >>> aligned_bc = bc.align(
            ...     scheme='piecewise',
            ...     parcellation=parcellation_img,
            ...     method='srm',
            ...     n_features=50
            ... )

            >>> # Fit/transform workflow for train/test split
            >>> aligned_train, model = train_bc.align(return_model=True)
            >>> aligned_test = model.transform(test_data_list)

            >>> # GPU-accelerated alignment
            >>> aligned_bc = bc.align(device='gpu')

        Notes:
            Based on Bazeille et al. 2021 "An empirical evaluation of functional
            alignment using inter-subject decoding". Center-only aggregation is
            used for searchlight to preserve local orthogonality of transforms.

        See Also:
            nltools.algorithms.alignment.LocalAlignment: Underlying alignment class.
        """
        from .transforms import align

        return align(
            self,
            method,
            scheme,
            radius_mm,
            parcellation,
            n_features,
            n_iter,
            device,
            return_model,
            n_jobs,
            progress_bar,
        )

    # =========================================================================
    # Convenience Methods (Delegators to BrainData methods)
    # =========================================================================

    def standardize(
        self,
        axis: int = 0,
        method: str = "center",
        n_jobs: int = 1,
        progress_bar: bool = False,
        verbose: bool = True,
    ) -> "BrainCollection":
        """
        Standardize each image.

        Delegates to BrainData.standardize() for each image.

        Args:
            axis: Axis for standardization within each image:
                - 0: Standardize across observations (time) per voxel
                - 1: Standardize across voxels per observation
            method: 'center' (subtract mean) or 'zscore' (subtract mean, divide std)
            n_jobs: Number of parallel jobs.
            progress_bar: Show progress bar.
            verbose: If False, suppress sklearn numerical warnings that occur
                when voxels have near-zero variance. Default: True.

        Returns:
            BrainCollection with standardized images.

        Examples:
            >>> bc.standardize()  # Center each image across time
            >>> bc.standardize(method='zscore')  # Z-score each image
            >>> bc.standardize(axis=1)  # Standardize across voxels
        """
        from .transforms import standardize

        return standardize(self, axis, method, n_jobs, progress_bar, verbose)

    def smooth(
        self,
        fwhm: float,
        n_jobs: int = 1,
        progress_bar: bool = False,
    ) -> "BrainCollection":
        """
        Spatially smooth each image.

        Delegates to BrainData.smooth() for each image.

        Args:
            fwhm: Full width at half maximum of Gaussian kernel in mm.
            n_jobs: Number of parallel jobs.
            progress_bar: Show progress bar.

        Returns:
            BrainCollection with smoothed images.

        Examples:
            >>> bc.smooth(fwhm=6)  # 6mm FWHM smoothing
        """
        from .transforms import smooth

        return smooth(self, fwhm, n_jobs, progress_bar)

    def threshold(
        self,
        upper: float | str | None = None,
        lower: float | str | None = None,
        binarize: bool = False,
        coerce_nan: bool = True,
        n_jobs: int = 1,
        progress_bar: bool = False,
    ) -> "BrainCollection":
        """
        Threshold each image.

        Delegates to BrainData.threshold() for each image.

        Args:
            upper: Upper cutoff. String interpreted as percentile.
            lower: Lower cutoff. String interpreted as percentile.
            binarize: Return binary mask.
            coerce_nan: Replace NaN with 0.
            n_jobs: Number of parallel jobs.
            progress_bar: Show progress bar.

        Returns:
            BrainCollection with thresholded images.

        Examples:
            >>> bc.threshold(lower=0)  # Zero out negative values
            >>> bc.threshold(upper='95%')  # Keep top 5%
            >>> bc.threshold(lower=2, binarize=True)  # Binary mask
        """
        from .transforms import threshold

        return threshold(self, upper, lower, binarize, coerce_nan, n_jobs, progress_bar)

    def detrend(
        self,
        method: str = "linear",
        n_jobs: int = 1,
        progress_bar: bool = False,
    ) -> "BrainCollection":
        """
        Remove trend from each image.

        Delegates to BrainData.detrend() for each image.

        Args:
            method: 'linear' or 'constant'.
            n_jobs: Number of parallel jobs.
            progress_bar: Show progress bar.

        Returns:
            BrainCollection with detrended images.

        Examples:
            >>> bc.detrend()  # Remove linear trend
            >>> bc.detrend(method='constant')  # Remove mean only
        """
        from .transforms import detrend

        return detrend(self, method, n_jobs, progress_bar)

    # =========================================================================
    # ISC (Intersubject Correlation) Methods
    # =========================================================================

    def _extract_for_isc(
        self,
        roi_mask: "nib.Nifti1Image | Path | str | None" = None,
        radius: float | None = 6.0,
        progress_bar: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """
        Extract data for ISC computation.

        Memory-efficient extraction that processes one subject at a time.
        Returns data in ISC-compatible format: (n_obs, n_subjects, n_features).

        Args:
            roi_mask: If provided, extract mean per ROI. Can be:
                - NIfTI image with integer labels (atlas/parcellation)
                - Path to parcellation file
            radius: Searchlight radius in mm. If None, use voxelwise mode.
                Ignored if roi_mask is provided.
            progress_bar: Show progress bar during extraction.

        Returns:
            Tuple of:
                - extracted_data: Array of shape (n_obs, n_subjects, n_features)
                - extraction_info: Dict with metadata for projection back:
                    - 'mode': 'roi', 'searchlight', or 'voxelwise'
                    - 'n_features': Number of features
                    - 'roi_mask': ROI mask if mode='roi'
                    - 'neighborhoods': SphereNeighborhoods if mode='searchlight'
        """
        from .inference import extract_for_isc

        return extract_for_isc(self, roi_mask, radius, progress_bar)

    def _extract_roi(
        self,
        roi_mask: "nib.Nifti1Image | Path | str",
        progress_bar: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """Extract mean signal per ROI."""
        from .inference import extract_roi

        return extract_roi(self, roi_mask, progress_bar)

    def _extract_searchlight(
        self,
        radius: float,
        progress_bar: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """Extract mean signal per searchlight sphere."""
        from .inference import extract_searchlight

        return extract_searchlight(self, radius, progress_bar)

    def _extract_voxelwise(
        self,
        progress_bar: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """Extract raw voxel data."""
        from .inference import extract_voxelwise

        return extract_voxelwise(self, progress_bar)

    def _project_to_brain(
        self,
        values: np.ndarray,
        extraction_info: dict,
    ) -> "BrainData":
        """
        Project ISC values back to brain space.

        Args:
            values: ISC values, shape depends on extraction mode:
                - ROI mode: (n_rois,)
                - Searchlight/voxelwise: (n_voxels,)
            extraction_info: Dict from _extract_for_isc with mode info.

        Returns:
            BrainData with ISC values in brain space.
        """
        from .inference import project_to_brain

        return project_to_brain(self, values, extraction_info)

    def isc(
        self,
        method: str = "loo",
        roi_mask: "nib.Nifti1Image | Path | str | None" = None,
        radius: float | None = 6.0,
        metric: str = "median",
        device: str = "cpu",
        n_jobs: int = -1,
        progress_bar: bool = False,
    ) -> dict:
        """
        Compute intersubject correlation (ISC) across the collection.

        ISC measures the similarity of brain responses across subjects,
        computed by correlating each subject's timeseries with others.

        Args:
            method: ISC computation method:
                - 'loo': Leave-one-out (correlate each subject with mean of others)
                - 'pairwise': All pairwise correlations between subjects
            roi_mask: If provided, compute ISC per ROI. Can be:
                - NIfTI image with integer labels (atlas/parcellation)
                - Path to parcellation file
            radius: Searchlight radius in mm. If None, use voxelwise mode.
                Ignored if roi_mask is provided.
            metric: Summary statistic for aggregating ISC values:
                - 'median': Robust to outliers (default)
                - 'mean': Fisher z-transformed mean
            device: Compute device: 'cpu' (default) or 'gpu' (via PyTorch).
            n_jobs: Number of CPU jobs (-1 = all cores, 1 = single-threaded).
            progress_bar: Show progress bar during extraction.

        Returns:
            Dictionary with:
                - 'isc': BrainData with ISC values
                - 'method': ISC method used ('loo' or 'pairwise')
                - 'extraction': Extraction mode ('roi', 'searchlight', 'voxelwise')
                - 'n_subjects': Number of subjects
                - 'extraction_info': Dict with extraction metadata

        Examples:
            >>> # ROI-based ISC using atlas
            >>> result = bc.isc(roi_mask="atlas.nii.gz")
            >>> result['isc'].plot()

            >>> # Searchlight ISC
            >>> result = bc.isc(radius=10.0)

            >>> # Voxelwise ISC
            >>> result = bc.isc(radius=None)

        Notes:
            For permutation testing, see BrainCollection.isc_test() (requires
            discussion of statistical methodology first).
        """
        from .inference import isc

        return isc(self, method, roi_mask, radius, metric, device, n_jobs, progress_bar)

    def isc_test(
        self,
        method: str = "loo",
        roi_mask: "nib.Nifti1Image | Path | str | None" = None,
        radius: float | None = 6.0,
        n_permute: int = 5000,
        permutation_method: str = "bootstrap",
        metric: str = "median",
        tail: int = 2,
        ci_percentile: float = 95,
        device: str = "cpu",
        return_null: bool = False,
        n_jobs: int = -1,
        random_state: int | None = None,
        progress_bar: bool = False,
    ) -> dict:
        """
        Compute ISC with permutation testing for statistical inference.

        This method combines ISC computation with permutation testing to
        determine statistical significance. It uses the same extraction
        pipeline as isc() and wraps isc_permutation_test().

        Args:
            method: ISC computation method:
                - 'loo': Leave-one-out (correlate each subject with mean of others)
                - 'pairwise': All pairwise correlations between subjects
            roi_mask: If provided, compute ISC per ROI. Can be:
                - NIfTI image with integer labels (atlas/parcellation)
                - Path to parcellation file
            radius: Searchlight radius in mm. If None, use voxelwise mode.
                Ignored if roi_mask is provided.
            n_permute: Number of permutations/bootstrap iterations. Default 5000.
            permutation_method: Method for null distribution:

                - 'bootstrap': Subject-wise bootstrap (default, Chen et al. 2016).
                  Tests whether observed ISC differs from random groupings.
                - 'circle_shift': Circular time-shift (preserves autocorrelation).
                  Tests for temporally-locked shared signal.
                - 'phase_randomize': FFT phase randomization (preserves power spectrum).
                  Tests for nonlinear temporal coupling.
            metric: Summary statistic for aggregating ISC values:
                - 'median': Robust to outliers (default)
                - 'mean': Fisher z-transformed mean
            tail: One-tailed (1) or two-tailed (2) test. Default 2.
            ci_percentile: Confidence interval percentile (e.g., 95). Default 95.
            device: Compute device: 'cpu' (default) or 'gpu' (via PyTorch).
            n_jobs: Number of CPU jobs (-1 = all cores, 1 = single-threaded).
            random_state: Random seed for reproducibility.
            return_null: If True, include null distribution in results.
            progress_bar: Show progress bar during extraction and permutation.

        Returns:
            Dictionary with:
                - 'isc': BrainData with ISC values
                - 'p': BrainData with p-values (Phipson-Smyth corrected)
                - 'ci': Tuple of (lower, upper) BrainData confidence intervals
                - 'method': ISC method used ('loo' or 'pairwise')
                - 'permutation_method': Permutation method used
                - 'extraction': Extraction mode ('roi', 'searchlight', 'voxelwise')
                - 'n_subjects': Number of subjects
                - 'n_permute': Number of permutations
                - 'null_dist': (optional) Null distribution array if return_null=True

        Examples:
            >>> # ROI-based ISC with permutation testing
            >>> result = bc.isc_test(roi_mask="atlas.nii.gz", n_permute=5000)
            >>> sig_mask = result['p'].data < 0.05
            >>> print(f"Significant ROIs: {sig_mask.sum()}")

            >>> # Searchlight ISC testing
            >>> result = bc.isc_test(radius=10.0)
            >>> result['isc'].plot()  # Show ISC values
            >>> result['p'].plot()    # Show p-values

            >>> # Voxelwise with phase randomization (tests temporal coupling)
            >>> result = bc.isc_test(
            ...     radius=None,
            ...     permutation_method='phase_randomize',
            ...     device='gpu'
            ... )

        Notes:
            - Bootstrap (default) is recommended for standard ISC inference
              (Chen et al. 2016). It tests whether ISC is significant at
              the group level.
            - Circle_shift and phase_randomize are more specialized - they
              test for temporally-structured shared signal beyond what's
              explained by autocorrelation or spectral structure alone.
            - For large voxelwise analyses, bootstrap is much faster as it
              resamples pre-computed values rather than recomputing ISC.

        References:
            Chen, G., et al. (2016). Untangling the relatedness among
            correlations, part I: nonparametric approaches to inter-subject
            correlation analysis at the group level. NeuroImage, 142, 248-259.
        """
        from .inference import isc_test

        return isc_test(
            self,
            method,
            roi_mask,
            radius,
            n_permute,
            permutation_method,
            metric,
            tail,
            ci_percentile,
            device,
            return_null,
            n_jobs,
            random_state,
            progress_bar,
        )

    def cv(
        self,
        k: int | None = None,
        method: str = "kfold",
        split_by: str | None = None,
        groups: np.ndarray | None = None,
        random_state: int | None = None,
        **kwargs,
    ) -> "BrainCollectionPipeline":
        """Create a cross-validation pipeline for multi-subject analysis.

        Returns a pipeline object that enables fluent, chainable transforms
        with cross-validation across subjects or runs.

        Args:
            k: Number of folds (for kfold method). Defaults to 5.
            method: CV scheme type. Options:
                - 'kfold': k-fold cross-validation on pooled data
                - 'loso': leave-one-subject-out (one image held out per fold)
                - 'loro': leave-one-run-out (requires groups)
            split_by: Metadata column for group splits.
                If provided and groups is None, gets groups from self.metadata[split_by].
            groups: Explicit group labels for CV splits.
            random_state: Random seed for reproducibility.
            **kwargs: Additional arguments passed to CVScheme.

        Returns:
            BrainCollectionPipeline: Pipeline for method chaining.

        Examples:
            >>> # Leave-one-subject-out classification
            >>> result = bc.cv(method='loso').normalize().predict(subject_labels, algorithm='svm')
            >>> print(f"Mean accuracy: {result.mean_score:.2%}")

            >>> # With preprocessing
            >>> result = (bc
            ...     .cv(method='loso')
            ...     .normalize()
            ...     .reduce(n_components=50)
            ...     .predict(labels))

            >>> # Run-based CV with metadata
            >>> result = bc.cv(method='loro', split_by='run').predict(y)

        See Also:
            BrainCollectionPipeline: For available transforms and terminals.
            CVScheme: For CV scheme configuration details.
        """
        from .modeling import cv

        return cv(self, k, method, split_by, groups, random_state, **kwargs)

    def fit(
        self,
        model: str,
        X: "pd.DataFrame | np.ndarray | str | list",
        cv: int | None = None,
        scale: bool = True,
        scale_value: float = 100.0,
        progress_bar: bool = False,
        **kwargs,
    ) -> "FittedBrainCollection":
        """
        Fit a model to each subject in the collection.

        Unified fitting method that shadows BrainData.fit() API for multi-subject
        analysis. Dispatches to model-specific implementations based on the
        model parameter.

        Args:
            model: Model type - 'glm' or 'ridge'
            X: Design/feature matrix. Can be:
                - pd.DataFrame/DesignMatrix: Shared (used for all subjects)
                - np.ndarray: Shared array (used for all subjects)
                - str: Column name in metadata pointing to file paths
                - list: Per-subject list of DataFrames/arrays/paths
            cv: Cross-validation folds (Ridge only). Default is None for GLM,
                5 for Ridge when output='scores'.
            scale: If True, apply percent signal change scaling before fitting.
            scale_value: Scaling value (default 100.0 for percent signal change).
            progress_bar: Show progress bar during fitting.
            **kwargs: Model-specific arguments passed to _fit_glm or _fit_ridge:
                - GLM: return_stats, save
                - Ridge: alpha, output, save, backend, random_state

        Returns:
            FittedBrainCollection wrapping the fitted results. Supports:

            - ``.results``: Access underlying BrainCollection(s) directly
            - ``.betas``: Convenience accessor for beta coefficients (GLM)
            - ``.pool()``: Aggregate across subjects for group analysis

            The underlying results contain:

            - GLM: Beta coefficients (n_regressors, n_voxels) per subject
            - Ridge: Scores or weights depending on 'output' kwarg

            If return_stats (GLM) or output='both' (Ridge), results is a dict.

        Examples:
            >>> # GLM with shared design matrix
            >>> fitted = bc.fit(model='glm', X=dm)
            >>> betas = fitted.results  # Access BrainCollection directly
            >>>
            >>> # Two-stage analysis with pool()
            >>> pool = bc.fit(model='glm', X=dm).pool(param='beta')
            >>> t_map = pool.fit(model='ttest', contrast='A-B')
            >>>
            >>> # GLM with per-subject design matrices
            >>> fitted = bc.fit(model='glm', X=[dm1, dm2, dm3])
            >>>
            >>> # Ridge encoding model with CV scores
            >>> fitted = bc.fit(model='ridge', X=features, cv=5)
            >>> scores = fitted.results

        See Also:
            fit_from_events: Convenience method for event-based GLM workflows
            fit_glm: Legacy GLM fitting (use fit_from_events instead)
            fit_ridge: Legacy Ridge fitting (use fit(..., model='ridge') instead)
        """
        from .modeling import fit

        return fit(self, model, X, cv, scale, scale_value, progress_bar, **kwargs)

    def fit_glm(
        self,
        events: pd.DataFrame,
        t_r: float,
        confounds: str | list[pd.DataFrame | Path | str] | None = None,
        confound_columns: list[str] | None = None,
        hrf_model: str = "spm",
        drift_model: str = "cosine",
        high_pass: float = 0.01,
        scale: bool = True,
        scale_value: float = 100.0,
        return_stats: list[str] | None = None,
        return_residuals: bool = False,
        save: dict[str, str] | None = None,
        progress_bar: bool = False,
        by_run: bool = False,
        run_column: str = "run",
        run_lengths: int | list[int] | None = None,
    ) -> "BrainCollection | dict[str, BrainCollection]":
        """
        Fit GLM to each subject in collection.

        Memory-efficient first-level GLM analysis that processes subjects
        one at a time. Returns a BrainCollection of beta coefficients for
        task regressors (confounds and drift terms are fit but not returned).

        Args:
            events: Task events DataFrame with onset, duration, trial_type columns.
                This is shared across all subjects (same experimental paradigm).
                If by_run=True, must also have a run column.
            t_r: Repetition time (TR) in seconds.
            confounds: Subject-specific confounds. Can be:
                - str: Column name in metadata pointing to confound file paths
                - list: List of DataFrames or paths, one per subject
                - None: No confounds (only task + drift terms)
            confound_columns: Columns to extract from confound files.
                If None and confounds provided, uses all columns.
            hrf_model: HRF model for convolution ('spm', 'glover', 'fir', etc.)
            drift_model: Drift model ('cosine', 'polynomial', None)
            high_pass: High-pass filter cutoff in Hz (default 0.01)
            scale: If True, apply percent signal change scaling before fitting.
            scale_value: Scaling value (default 100.0 for percent signal change).
            return_stats: Optional list of statistics to return as separate
                BrainCollections. Options: 't', 'r2', 'p', 'se', 'residual'.
            return_residuals: If True, return residuals (same as return_stats=['residual']).
            save: Dict mapping output type to path template, e.g.
                ``{'betas': 'output/{subject}_betas.nii.gz',
                't': 'output/{subject}_tstat.nii.gz'}``.
                Supports {subject}, {session}, {idx}, and other metadata columns.
            progress_bar: Show progress bar during fitting.
            by_run: If True, fit GLM separately per run and return run-level betas.
                This enables MVPA decoding with leave-one-run-out CV.
                Each subject will have (n_runs * n_conditions, n_voxels) betas.
            run_column: Column name in events identifying runs (default 'run').
            run_lengths: Number of TRs per run. Required when by_run=True.

                - int: All runs have same length
                - list of int: Different length per run
                - None: Will attempt to infer equal-length runs from total scans

        Returns:
            BrainCollection where each BrainData has shape:

            - (n_task_regressors, n_voxels) if by_run=False (default)
            - (n_runs * n_task_regressors, n_voxels) if by_run=True

            The ``._design_columns`` attribute stores task regressor names.
            If by_run=True, also stores ``._condition_labels`` and ``._run_labels``.
            If return_stats specified, returns dict with keys 'betas', 't', etc.

        Examples:
            >>> # Basic GLM fit
            >>> betas = bc.fit_glm(events=events_df, t_r=2.0)
            >>> # Group t-test on first regressor
            >>> group_t = betas[:, 0].ttest()

            >>> # Run-level betas for MVPA decoding
            >>> betas = bc.fit_glm(events=events_df, t_r=2.0, by_run=True)
            >>> # betas._condition_labels = ['face', 'house', 'face', 'house', ...]
            >>> # betas._run_labels = [1, 1, 2, 2, 3, 3, ...]
            >>> accuracy = betas.predict(y=None, method='searchlight')

            >>> # With confounds from metadata column
            >>> betas = bc.fit_glm(
            ...     events=events_df,
            ...     t_r=2.0,
            ...     confounds='confound_file',  # column name in metadata
            ...     confound_columns=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
            ... )
        """
        from .modeling import fit_glm

        return fit_glm(
            self,
            events,
            t_r,
            confounds,
            confound_columns,
            hrf_model,
            drift_model,
            high_pass,
            scale,
            scale_value,
            return_stats,
            return_residuals,
            save,
            progress_bar,
            by_run,
            run_column,
            run_lengths,
        )

    def fit_from_events(
        self,
        events: pd.DataFrame,
        t_r: float,
        confounds: str | list[pd.DataFrame | Path | str] | None = None,
        confound_columns: list[str] | None = None,
        hrf_model: str = "spm",
        drift_model: str = "cosine",
        high_pass: float = 0.01,
        scale: bool = True,
        scale_value: float = 100.0,
        return_stats: list[str] | None = None,
        return_residuals: bool = False,
        save: dict[str, str] | None = None,
        progress_bar: bool = False,
        by_run: bool = False,
        run_column: str = "run",
        run_lengths: int | list[int] | None = None,
    ) -> "BrainCollection | dict[str, BrainCollection]":
        """
        Build design matrices from events and fit GLM to each subject.

        Convenience method for event-based experimental designs. Builds
        nilearn-compatible design matrices from the events DataFrame and
        fits a GLM to each subject in the collection.

        This is the recommended method for typical task-based fMRI analysis
        where you have event timing information. For more control, use
        fit(model='glm', X=design_matrices) with pre-built design matrices.

        Args:
            events: Task events DataFrame with onset, duration, trial_type columns.
                This is shared across all subjects (same experimental paradigm).
                If by_run=True, must also have a run column.
            t_r: Repetition time (TR) in seconds.
            confounds: Subject-specific confounds. Can be:
                - str: Column name in metadata pointing to confound file paths
                - list: List of DataFrames or paths, one per subject
                - None: No confounds (only task + drift terms)
            confound_columns: Columns to extract from confound files.
                If None and confounds provided, uses all columns.
            hrf_model: HRF model for convolution ('spm', 'glover', 'fir', etc.)
            drift_model: Drift model ('cosine', 'polynomial', None)
            high_pass: High-pass filter cutoff in Hz (default 0.01)
            scale: If True, apply percent signal change scaling before fitting.
            scale_value: Scaling value (default 100.0 for percent signal change).
            return_stats: Optional list of statistics to return as separate
                BrainCollections. Options: 't', 'r2', 'p', 'se', 'residual'.
            return_residuals: If True, return residuals (same as return_stats=['residual']).
            save: Dict mapping output type to path template.
            progress_bar: Show progress bar during fitting.
            by_run: If True, fit GLM separately per run and return run-level betas.
                This enables MVPA decoding with leave-one-run-out CV.
            run_column: Column name in events identifying runs (default 'run').
            run_lengths: Number of TRs per run. Required when by_run=True.

        Returns:
            BrainCollection of beta coefficients for task regressors.
            If return_stats specified, returns dict with keys 'betas', 't', etc.

        Examples:
            >>> # Basic GLM fit from events
            >>> betas = bc.fit_from_events(events=events_df, t_r=2.0)
            >>> group_t = betas.ttest()
            >>>
            >>> # With confounds from metadata column
            >>> betas = bc.fit_from_events(
            ...     events=events_df,
            ...     t_r=2.0,
            ...     confounds='confound_file',
            ...     confound_columns=['trans_x', 'trans_y', 'trans_z']
            ... )
            >>>
            >>> # Run-level betas for MVPA
            >>> betas = bc.fit_from_events(events=events_df, t_r=2.0, by_run=True)

        See Also:
            fit: Unified fit method that accepts pre-built design matrices
            _fit_glm: Internal method for design matrix-based fitting
        """
        from .modeling import fit_from_events

        return fit_from_events(
            self,
            events,
            t_r,
            confounds,
            confound_columns,
            hrf_model,
            drift_model,
            high_pass,
            scale,
            scale_value,
            return_stats,
            return_residuals,
            save,
            progress_bar,
            by_run,
            run_column,
            run_lengths,
        )

    def _resolve_confounds(
        self,
        confounds: str | list[pd.DataFrame | Path | str] | None,
    ) -> list[pd.DataFrame | Path | str] | None:
        """Resolve confounds argument to per-subject list.

        Args:
            confounds: Either:
                - str: Column name in metadata containing confound paths
                - list: Already per-subject list of DataFrames or paths
                - None: No confounds

        Returns:
            List of confounds (one per subject) or None
        """
        from .modeling import resolve_confounds

        return resolve_confounds(self, confounds)

    def _fit_glm(
        self,
        X: "pd.DataFrame | np.ndarray | str | list",
        scale: bool = True,
        scale_value: float = 100.0,
        return_stats: list[str] | None = None,
        save: dict[str, str] | None = None,
        progress_bar: bool = False,
    ) -> "BrainCollection | dict[str, BrainCollection]":
        """Internal GLM fitting with design matrix input.

        Core implementation that accepts DesignMatrix/DataFrame directly.
        Called by fit(model='glm') and fit_from_events().

        Args:
            X: Design matrix. Can be:
                - pd.DataFrame/DesignMatrix: Shared (used for all subjects)
                - np.ndarray: Shared array (converted to DataFrame internally)
                - str: Column name in metadata pointing to file paths
                - list: Per-subject list of DataFrames/arrays/paths
            scale: If True, apply percent signal change scaling before fitting.
            scale_value: Scaling value (default 100.0 for percent signal change).
            return_stats: Optional list of statistics to return as separate
                BrainCollections. Options: 't', 'r2', 'p', 'se', 'residual'.
            save: Dict mapping output type to path template.
            progress_bar: Show progress bar during fitting.

        Returns:
            BrainCollection of betas, or dict with betas + requested stats.
        """
        from .modeling import fit_glm_internal

        return fit_glm_internal(
            self, X, scale, scale_value, return_stats, save, progress_bar
        )

    def _load_design_matrix(self, path: str | Path) -> pd.DataFrame:
        """Load design matrix from a file path.

        Supports common formats: .csv, .tsv, .txt
        """
        from .modeling import load_design_matrix

        return load_design_matrix(self, path)

    def fit_ridge(
        self,
        X: "np.ndarray | str | list",
        alpha: float | str = 1.0,
        cv: int | None = 5,
        scale: bool = True,
        scale_value: float = 100.0,
        output: str = "scores",
        save: dict[str, str] | None = None,
        progress_bar: bool = False,
        **ridge_kwargs,
    ) -> "BrainCollection | dict[str, BrainCollection]":
        """
        Fit ridge regression to each subject in collection.

        Memory-efficient encoding model fitting that processes subjects one at a
        time. Default behavior returns cross-validated R² scores per voxel,
        suitable for group-level inference on encoding model performance.

        Args:
            X: Feature matrix. Can be:
                - np.ndarray: Shared features (n_samples, n_features) used for all subjects
                - str: Column name in metadata pointing to feature file paths
                - list: List of arrays/DataFrames, one per subject
            alpha: Ridge regularization parameter. Can be:
                - float: Fixed regularization strength
                - 'auto': Use cross-validation to select optimal alpha
            cv: Cross-validation folds for computing scores. Default is 5.
                Required when output='scores' or 'both'. Set to None only when
                output='weights'.
            scale: If True, apply percent signal change scaling before fitting.
            scale_value: Scaling value (default 100.0 for percent signal change).
            output: What to return. Options:
                - 'scores': CV R² scores per voxel (default, for encoding workflow)
                - 'weights': Model weights (n_features, n_voxels)
                - 'both': Dict with both 'scores' and 'weights'
            save: Dict mapping output type to path template, e.g.
                ``{'weights': 'output/{subject}_weights.nii.gz',
                'scores': 'output/{subject}_scores.nii.gz'}``.
                Supports {subject}, {session}, {idx}, and other metadata columns.
            progress_bar: Show progress bar during fitting.
            **ridge_kwargs: Additional arguments passed to Ridge model
                (e.g., backend='torch', random_state=42).

        Returns:
            BrainCollection of scores or weights, or dict with both if output='both'.
            Each BrainData will have ``cv_results_`` attribute when cv is used.

        Examples:
            >>> # Encoding model workflow: get CV scores for group analysis
            >>> scores = bc.fit_ridge(X=features, alpha=1.0)
            >>> group_ttest = scores.ttest()  # Test encoding accuracy vs chance

            >>> # Get both scores and weights
            >>> results = bc.fit_ridge(X=features, alpha=1.0, output='both')
            >>> scores = results['scores']
            >>> weights = results['weights']

            >>> # Auto-select alpha with CV
            >>> scores = bc.fit_ridge(X=features, alpha='auto', cv=5)

            >>> # Get weights only (no CV needed)
            >>> weights = bc.fit_ridge(X=features, alpha=1.0, output='weights', cv=None)
        """
        from .modeling import fit_ridge

        return fit_ridge(
            self,
            X,
            alpha,
            cv,
            scale,
            scale_value,
            output,
            save,
            progress_bar,
            **ridge_kwargs,
        )

    def _resolve_X(
        self,
        X: "np.ndarray | pd.DataFrame | str | list | None",
    ) -> list | None:
        """Resolve design/feature matrix X to per-subject list.

        Unified helper for resolving X parameter across fit methods. Supports
        three input patterns:
        1. Shared matrix (array/DataFrame/DesignMatrix): Same X for all subjects
        2. Per-subject list: List of matrices, one per subject
        3. Metadata column: String column name pointing to file paths

        Args:
            X: Design/feature matrix. Can be:
                - np.ndarray: Shared array (used for all subjects)
                - pd.DataFrame: Shared DataFrame/DesignMatrix (used for all subjects)
                - str: Column name in metadata containing file paths
                - list: Per-subject list of arrays/DataFrames/paths
                - None: Error

        Returns:
            list | None: Per-subject list if X varies by subject, None if shared.
                Caller should use: `X_subj = X_list[i] if X_list else X`
        """
        from .modeling import resolve_X

        return resolve_X(self, X)

    def _load_features(self, path: str | Path) -> np.ndarray:
        """Load features from a file path.

        Supports common formats: .npy, .csv, .tsv, .txt
        """
        from .modeling import load_features

        return load_features(self, path)

    def predict(
        self,
        X: "np.ndarray | str | list | None" = None,
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
        progress_bar: bool = False,
    ) -> "BrainCollection":
        """
        Generate predictions for each subject in collection.

        This method supports two prediction modes determined by which parameter
        is provided:

        1. **Timeseries prediction** (X provided): Use fitted ridge model to
           predict voxel responses for new feature data.

        2. **MVPA decoding** (y provided): Train a classifier to predict labels
           from brain patterns using cross-validation.

        For MVPA, if this collection was created with by_run=True, you can
        use y=None to infer labels from _condition_labels and groups from
        _run_labels (leave-one-run-out CV).

        Args:
            X: Features for timeseries prediction. Can be:
                - np.ndarray: Shared features (same for all subjects)
                - str: Metadata column with per-subject feature paths
                - list: Per-subject feature arrays
            y: Labels for MVPA decoding. If None and _condition_labels exists,
                will use stored condition labels (from fit_glm with by_run=True).
            method: MVPA method - 'whole_brain', 'searchlight', or 'roi'.
            estimator: Classifier - 'svm', 'logistic', 'ridge', 'lda' or
                sklearn estimator instance.
            cv: Cross-validation strategy. If None and _run_labels exists,
                uses leave-one-group-out with run labels.
            groups: Group labels for GroupKFold/LeaveOneGroupOut. If None
                and _run_labels exists, uses stored run labels.
            roi_mask: Mask for ROI-based MVPA. Required if method='roi'.
            radius: Searchlight radius in mm (default 10.0).
            scoring: Scoring metric (default 'accuracy').
            standardize: If True, standardize features before classification.
            n_jobs: Parallel jobs for searchlight (-1 = all cores).
            progress_bar: Show progress bar during fitting.

        Returns:
            BrainCollection with prediction results:
            - For timeseries: (n_timepoints, n_voxels) predicted responses
            - For MVPA: (1, n_voxels) accuracy values

        Examples:
            >>> # MVPA workflow with run-level betas
            >>> betas = bc.fit_glm(events=events, t_r=2.0, by_run=True)
            >>> accuracy = betas.predict(y=None, method='whole_brain')
            >>> # y and groups inferred from _condition_labels, _run_labels

            >>> # Explicit labels
            >>> accuracy = betas.predict(y=labels, method='searchlight')

            >>> # Timeseries prediction with ridge weights
            >>> weights = bc.fit_ridge(X=features, output='weights')
            >>> predictions = weights.predict(X=new_features)
        """
        from .prediction import predict

        return predict(
            self,
            X,
            y,
            method,
            estimator,
            cv,
            groups,
            roi_mask,
            radius,
            scoring,
            standardize,
            n_jobs,
            progress_bar,
        )

    def compute_contrasts(
        self,
        contrasts: "str | dict | np.ndarray | list",
    ) -> "BrainCollection | dict[str, BrainCollection]":
        """
        Compute contrasts from fitted GLM beta coefficients.

        Applies contrast weights to each subject's betas and returns a
        BrainCollection of contrast values suitable for group-level analysis.

        Must be called on a BrainCollection created by fit_glm() which has
        the _design_columns attribute set.

        Args:
            contrasts: Can be:
                - str: Contrast string using column names, e.g., "face - house"
                - dict: Multiple contrasts, e.g., {"main": "face - house", "avg": [0.5, 0.5]}
                - array/list: Numeric contrast vector, e.g., [1, -1]

        Returns:
            BrainCollection where each BrainData has shape (n_voxels,) containing
            the contrast values. If dict input, returns dict of BrainCollections.

        Raises:
            RuntimeError: If _design_columns not set (not from fit_glm)
            ValueError: If contrast vector length doesn't match number of regressors
            ValueError: If column name in string contrast not found

        Examples:
            >>> # Fit GLM and compute contrast
            >>> betas = bc.fit_glm(events=events_df, t_r=2.0)
            >>> contrast = betas.compute_contrasts("face - house")
            >>> # Group t-test on contrast
            >>> group_result = contrast.ttest()

            >>> # Multiple contrasts
            >>> contrasts = betas.compute_contrasts({
            ...     "face_vs_house": "face - house",
            ...     "face_vs_baseline": "face",
            ... })
            >>> face_vs_house_ttest = contrasts["face_vs_house"].ttest()
        """
        from .prediction import compute_contrasts

        return compute_contrasts(self, contrasts)

    def _compute_single_contrast(
        self,
        contrast: "str | np.ndarray | list",
        design_columns: list[str],
    ) -> "BrainCollection":
        """Compute a single contrast across all subjects.

        Args:
            contrast: Contrast specification (string, array, or list)
            design_columns: List of regressor names from fit_glm

        Returns:
            BrainCollection with contrast values for each subject
        """
        from .prediction import compute_single_contrast

        return compute_single_contrast(self, contrast, design_columns)

    def _parse_contrast_string(
        self,
        contrast_str: str,
        design_columns: list[str],
    ) -> np.ndarray:
        """Parse a contrast string into a numeric contrast vector.

        Args:
            contrast_str: Contrast string like "A - B" or "2*A - B"
            design_columns: List of regressor column names

        Returns:
            Numeric contrast vector

        Raises:
            ValueError: If column name not found in design_columns
        """
        from .prediction import parse_contrast_string

        return parse_contrast_string(self, contrast_str, design_columns)

    def select_feature(
        self,
        feature: "int | str",
    ) -> "BrainCollection":
        """
        Select a single feature's weights across all subjects.

        Used after fit_ridge() to extract weights for a specific feature
        for group-level analysis (e.g., t-test on feature weights).

        Must be called on a BrainCollection created by fit_ridge() where
        each subject has shape (n_features, n_voxels).

        Args:
            feature: Feature to select. Can be:
                - int: Feature index (0-based)
                - str: Feature name (requires _feature_names attribute)

        Returns:
            BrainCollection where each BrainData has shape (n_voxels,)
            containing the weights for the specified feature.

        Raises:
            IndexError: If feature index out of range
            KeyError: If feature name not found in _feature_names
            RuntimeError: If string feature given but _feature_names not set

        Examples:
            >>> # Fit ridge and select feature
            >>> weights = bc.fit_ridge(X=features, alpha=1.0)
            >>> feature_0 = weights.select_feature(0)
            >>> # Group t-test on first feature's weights
            >>> group_result = feature_0.ttest()

            >>> # By name (if features had column names)
            >>> face_weights = weights.select_feature("face_response")
        """
        from .prediction import select_feature

        return select_feature(self, feature)
