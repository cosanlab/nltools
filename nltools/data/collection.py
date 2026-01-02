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
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Iterator,
    TypeVar,
    Generator,
    Any,
)

from nltools.utils import attempt_to_import
from nltools.prefs import MNI_Template

if TYPE_CHECKING:
    from .brain_data import BrainData

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

    Attributes:
        shape: Tuple of (n_images, n_observations, n_voxels). n_observations is None
            if images have variable observation counts.
        n_images: Number of images in collection.
        n_voxels: Number of voxels (from mask).
        mask: The shared brain mask as nibabel image.
        metadata: DataFrame with per-image metadata.
        is_loaded: List of booleans indicating which images are in memory.

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
        metadata: pd.DataFrame | None = None,
        lazy: bool = True,
    ):
        """Initialize BrainCollection.

        Args:
            items: List of paths or BrainData objects.
            mask: Shared mask (required). Path, nibabel image, or template name.
            metadata: Optional per-image metadata DataFrame.
            lazy: If True, paths are loaded on demand.
        """
        from .brain_data import BrainData

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

        # Metadata
        if metadata is not None:
            if len(metadata) != len(self._items):
                raise ValueError(
                    f"metadata length ({len(metadata)}) must match "
                    f"items length ({len(self._items)})"
                )
            self._metadata = metadata.reset_index(drop=True)
        else:
            self._metadata = pd.DataFrame(index=range(len(self._items)))

    def _resolve_mask(self, mask: nib.Nifti1Image | Path | str) -> nib.Nifti1Image:
        """Resolve mask to nibabel image."""
        if isinstance(mask, nib.Nifti1Image):
            return mask
        elif isinstance(mask, (str, Path)):
            path = Path(mask)
            if path.exists():
                return nib.load(path)
            # Try as template name
            try:
                template = MNI_Template(mask)
                return nib.load(template.mask_path)
            except Exception:
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

    def _load_item(self, idx: int) -> "BrainData":
        """Load a single item if it's a path, return BrainData."""
        from .brain_data import BrainData

        if isinstance(self._items[idx], Path):
            bd = BrainData(self._items[idx], mask=self._mask)
            self._items[idx] = bd
            self._is_loaded[idx] = True
            self._sample_counts[idx] = self._get_n_obs(bd)
        return self._items[idx]

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
        """Shared brain mask."""
        return self._mask

    @property
    def metadata(self) -> pd.DataFrame:
        """Per-image metadata DataFrame."""
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
        if indices is None:
            indices = range(len(self._items))

        for idx in indices:
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
        from .brain_data import BrainData

        if indices is None:
            indices = range(len(self._items))

        for idx in indices:
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
        # Try common columns
        for col in ["subject", "subject_id", "sub", "id"]:
            if col in self._metadata.columns:
                matches = self._metadata[self._metadata[col] == key].index
                if len(matches) == 1:
                    return self._load_item(matches[0])
                elif len(matches) > 1:
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
                from .brain_data import BrainData

                result = BrainData(mask=bd.mask)
                result.data = sliced_data
                return result

            elif isinstance(obs_key, slice):
                # Slice observations
                from .brain_data import BrainData

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
                from .brain_data import BrainData

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
        new_metadata = self._metadata.iloc[indices].reset_index(drop=True)

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
        # Import pybids
        BIDSLayout = attempt_to_import(
            "bids",
            "BIDSLayout",
            "pybids required for BIDS loading. Install with: pip install pybids",
        )

        # Create layout if path provided
        if isinstance(layout, (str, Path)):
            layout = BIDSLayout(layout, derivatives=True)

        # Build filter dict
        filters = {"extension": extension, "suffix": suffix}
        if task is not None:
            filters["task"] = task
        if subject is not None:
            filters["subject"] = subject
        if session is not None:
            filters["session"] = session
        if run is not None:
            filters["run"] = run
        if space is not None:
            filters["space"] = space
        filters.update(bids_filters)

        # Get files
        files = layout.get(return_type="file", **filters)
        if not files:
            raise ValueError(f"No files found matching filters: {filters}")

        # Extract metadata
        metadata_rows = []
        for f in files:
            bf = layout.get_file(f)
            entities = bf.get_entities() if bf else {}
            metadata_rows.append(
                {
                    "subject": entities.get("subject"),
                    "session": entities.get("session"),
                    "run": entities.get("run"),
                    "task": entities.get("task"),
                    "space": entities.get("space"),
                }
            )

        metadata = pd.DataFrame(metadata_rows)
        return cls(files, mask=mask, metadata=metadata)

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
            pattern: Glob pattern (e.g., '/data/*/func/*_bold.nii.gz').
            mask: Shared mask (required).
            pattern_groups: Regex pattern with named groups for metadata extraction.
                Example: r'sub-(?P<subject>\\w+)/.*run-(?P<run>\\d+)'
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
        import glob
        import re

        files = glob.glob(pattern, recursive=True)
        if not files:
            raise ValueError(f"No files found matching pattern: {pattern}")

        if sort:
            files = sorted(files)

        # Extract metadata from paths
        metadata = None
        if pattern_groups is not None:
            if isinstance(pattern_groups, str):
                regex = re.compile(pattern_groups)
                metadata_rows = []
                for f in files:
                    match = regex.search(f)
                    if match:
                        metadata_rows.append(match.groupdict())
                    else:
                        metadata_rows.append({})
                metadata = pd.DataFrame(metadata_rows)

        return cls(files, mask=mask, metadata=metadata)

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
        from .brain_data import BrainData

        if splits is None and n_images is None:
            raise ValueError("Must provide either splits or n_images")
        if splits is not None and n_images is not None:
            raise ValueError("Cannot provide both splits and n_images")

        data = brain_data.data
        if data.ndim == 1:
            data = data[np.newaxis, :]

        n_total = data.shape[0]

        if n_images is not None:
            if n_total % n_images != 0:
                raise ValueError(
                    f"Cannot evenly split {n_total} observations into {n_images} images"
                )
            splits = [n_total // n_images] * n_images

        if sum(splits) != n_total:
            raise ValueError(
                f"splits sum ({sum(splits)}) must equal total observations ({n_total})"
            )

        # Split data
        items = []
        idx = 0
        for count in splits:
            bd = BrainData(mask=brain_data.mask)
            bd.data = data[idx : idx + count]
            items.append(bd)
            idx += count

        return cls(items, mask=brain_data.mask)

    # =========================================================================
    # Axis Operations (to be implemented in nltools-cyb)
    # =========================================================================

    def _normalize_axis(
        self, axis: int | str | tuple[int, ...]
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
            return tuple(self._normalize_axis(a) for a in axis)
        return axis

    def _aggregate_axis0(
        self,
        func: str,
        batch_size: int | None = None,
    ) -> "BrainData":
        """Aggregate across images (axis=0) using streaming algorithm."""
        from .brain_data import BrainData

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
        from .brain_data import BrainData

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
        from .brain_data import BrainData

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

    def iter_batches(
        self,
        batch_size: int,
        axis: int = 0,
        show_progress: bool = True,
    ) -> Generator["BrainCollection", None, None]:
        """
        Iterate in batches along axis.

        Args:
            batch_size: Number of items per batch.
            axis: Axis to batch along:
                - 0: Batches of images (default)
                - 1: Batches of timepoints (within each image)
            show_progress: Show tqdm progress bar.

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

            if show_progress and tqdm is not None:
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

            if show_progress and tqdm is not None:
                iterator = tqdm.tqdm(
                    iterator, desc="Batching observations", total=n_batches
                )

            for batch_idx in iterator:
                start = batch_idx * batch_size
                end = min(start + batch_size, n_obs)
                # Slice observations for each image
                yield self[:, start:end]

        else:
            raise ValueError(f"Cannot batch over axis {axis}. Use axis=0 or axis=1.")
