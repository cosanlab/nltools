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


# =============================================================================
# Helper Functions for fit_glm / fit_ridge
# =============================================================================


def _resolve_save_path(
    template: str,
    metadata_row: pd.Series,
    idx: int,
) -> Path:
    """Resolve a template path using metadata values.

    Replaces {column_name} placeholders with values from metadata.
    Falls back to {idx} for the subject index.

    Args:
        template: Path template with {placeholders}, e.g., 'output/{subject}_betas.nii.gz'
        metadata_row: Series with metadata for this subject
        idx: Subject index (used for {idx} placeholder)

    Returns:
        Resolved Path object

    Raises:
        KeyError: If placeholder not found in metadata and not 'idx'

    Example:
        >>> row = pd.Series({'subject': 'sub-01', 'session': 'ses-01'})
        >>> _resolve_save_path('out/{subject}_{session}.nii.gz', row, 0)
        PosixPath('out/sub-01_ses-01.nii.gz')
    """
    import re

    result = template

    # Find all {placeholder} patterns
    placeholders = re.findall(r"\{(\w+)\}", template)

    for placeholder in placeholders:
        if placeholder == "idx":
            value = str(idx)
        elif placeholder in metadata_row.index:
            value = str(metadata_row[placeholder])
        else:
            available = list(metadata_row.index) + ["idx"]
            raise KeyError(
                f"Placeholder '{{{placeholder}}}' not found in metadata. "
                f"Available: {available}"
            )
        result = result.replace(f"{{{placeholder}}}", value)

    path = Path(result)

    # Create parent directories if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    return path


def _build_subject_design_matrix(
    events: pd.DataFrame,
    n_scans: int,
    t_r: float,
    confounds: pd.DataFrame | Path | str | None = None,
    confound_columns: list[str] | None = None,
    hrf_model: str = "spm",
    drift_model: str = "cosine",
    high_pass: float = 0.01,
) -> tuple[pd.DataFrame, list[str]]:
    """Build complete design matrix for a subject.

    Combines task design (from events) with subject-specific confounds.

    Args:
        events: Task events DataFrame with 'onset', 'duration', 'trial_type' columns
        n_scans: Number of scans/timepoints
        t_r: Repetition time in seconds
        confounds: Subject confounds - DataFrame, path to TSV, or None
        confound_columns: Which confound columns to include (None = all)
        hrf_model: HRF model for task regressors ('spm', 'glover', etc.)
        drift_model: Drift model ('cosine', 'polynomial', None)
        high_pass: High-pass filter cutoff in Hz

    Returns:
        Tuple of:
            - design_matrix: Complete design matrix (task + confounds + drift)
            - task_columns: List of task regressor column names

    Example:
        >>> events = pd.DataFrame({
        ...     'onset': [0, 10, 20],
        ...     'duration': [2, 2, 2],
        ...     'trial_type': ['face', 'house', 'face']
        ... })
        >>> dm, task_cols = _build_subject_design_matrix(events, 100, 2.0)
        >>> print(task_cols)
        ['face', 'house']
    """
    from nilearn.glm.first_level import make_first_level_design_matrix

    # Create frame times
    frame_times = np.arange(n_scans) * t_r

    # Build task design matrix
    task_dm = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events,
        hrf_model=hrf_model,
        drift_model=drift_model,
        high_pass=high_pass,
    )

    # Identify task columns (everything except drift and constant)
    drift_cols = [
        c for c in task_dm.columns if c.startswith("drift_") or c == "constant"
    ]
    task_columns = [c for c in task_dm.columns if c not in drift_cols]

    # Load confounds if provided
    if confounds is not None:
        if isinstance(confounds, (str, Path)):
            confounds_path = Path(confounds)
            if not confounds_path.exists():
                raise FileNotFoundError(f"Confounds file not found: {confounds_path}")
            # Detect separator (TSV vs CSV)
            sep = "\t" if confounds_path.suffix in [".tsv", ".txt"] else ","
            confounds_df = pd.read_csv(confounds_path, sep=sep)
        else:
            confounds_df = confounds

        # Select columns if specified
        if confound_columns is not None:
            missing = set(confound_columns) - set(confounds_df.columns)
            if missing:
                raise ValueError(
                    f"Confound columns not found: {missing}. "
                    f"Available: {list(confounds_df.columns)}"
                )
            confounds_df = confounds_df[confound_columns]

        # Validate length
        if len(confounds_df) != n_scans:
            raise ValueError(
                f"Confounds have {len(confounds_df)} rows but data has {n_scans} scans. "
                "Lengths must match."
            )

        # Handle NaN values in confounds (common for first few rows of derivatives)
        confounds_df = confounds_df.fillna(0)

        # Align confounds index with design matrix frame_times index
        confounds_df.index = task_dm.index

        # Concatenate: task_dm already has drift and constant, add confounds before them
        # Reorder: task | confounds | drift | constant
        full_dm = pd.concat(
            [task_dm[task_columns], confounds_df, task_dm[drift_cols]],
            axis=1,
        )
    else:
        full_dm = task_dm

    return full_dm, task_columns


def _fit_glm_by_run(
    bd: "BrainData",
    events: pd.DataFrame,
    runs: list,
    run_column: str,
    run_lengths: int | list[int] | None,
    t_r: float,
    confounds: pd.DataFrame | None,
    confound_columns: list[str] | None,
    hrf_model: str,
    drift_model: str,
    high_pass: float,
    scale: bool,
    scale_value: float,
) -> tuple["BrainData", list[str], list[str], list]:
    """Fit GLM separately for each run and stack betas.

    Args:
        bd: Subject's BrainData (concatenated timeseries)
        events: Events DataFrame with run column
        runs: Unique run identifiers (sorted)
        run_column: Column name for run identifier
        run_lengths: TRs per run (int for uniform, list for variable, None to infer)
        t_r: Repetition time
        confounds: Subject confounds (concatenated) or None
        confound_columns: Columns to extract from confounds
        hrf_model: HRF model for design matrix
        drift_model: Drift model
        high_pass: High-pass filter cutoff
        scale: Whether to apply percent signal change scaling
        scale_value: Scaling value

    Returns:
        Tuple of:
            - BrainData with stacked run-level betas (n_runs * n_conditions, n_voxels)
            - task_columns: List of condition names
            - condition_labels: Condition label for each beta row
            - run_labels: Run label for each beta row
    """
    n_scans = bd.shape[0]
    n_runs = len(runs)

    # Resolve run lengths
    if run_lengths is None:
        # Try to infer equal-length runs
        if n_scans % n_runs != 0:
            raise ValueError(
                f"Cannot infer run lengths: {n_scans} scans not evenly divisible by "
                f"{n_runs} runs. Please provide run_lengths parameter."
            )
        run_length_list = [n_scans // n_runs] * n_runs
    elif isinstance(run_lengths, int):
        run_length_list = [run_lengths] * n_runs
    else:
        run_length_list = list(run_lengths)

    # Validate total matches
    if sum(run_length_list) != n_scans:
        raise ValueError(
            f"run_lengths sum ({sum(run_length_list)}) does not match "
            f"total scans ({n_scans})"
        )

    # Calculate run start indices
    run_starts = [0]
    for length in run_length_list[:-1]:
        run_starts.append(run_starts[-1] + length)

    # Storage for run-level results
    all_betas = []
    condition_labels = []
    run_labels = []
    task_columns = None

    for run_idx, run_id in enumerate(runs):
        # Get run boundaries
        start_tr = run_starts[run_idx]
        end_tr = start_tr + run_length_list[run_idx]
        n_run_scans = run_length_list[run_idx]

        # Slice data for this run
        run_bd = bd[start_tr:end_tr]

        # Filter events to this run and adjust onsets (assuming run-relative onsets)
        run_events = events[events[run_column] == run_id].copy()
        # Remove run column for design matrix building
        run_events = run_events.drop(columns=[run_column])

        # Slice confounds if provided
        run_confounds = None
        if confounds is not None:
            run_confounds = confounds.iloc[start_tr:end_tr].reset_index(drop=True)

        # Build design matrix for this run
        dm, task_cols = _build_subject_design_matrix(
            events=run_events,
            n_scans=n_run_scans,
            t_r=t_r,
            confounds=run_confounds,
            confound_columns=confound_columns,
            hrf_model=hrf_model,
            drift_model=drift_model,
            high_pass=high_pass,
        )

        # Store task columns from first run
        if task_columns is None:
            task_columns = task_cols

        # Apply scaling if requested
        if scale:
            run_bd = run_bd.scale(scale_value)

        # Fit GLM
        run_bd.fit(model="glm", X=dm)

        # Extract task betas only
        task_indices = [dm.columns.get_loc(col) for col in task_cols]
        run_betas = run_bd.glm_betas.data[task_indices, :]

        # Store betas and labels
        all_betas.append(run_betas)
        condition_labels.extend(task_cols)
        run_labels.extend([run_id] * len(task_cols))

    # Stack all run betas
    stacked_betas = np.vstack(all_betas)

    # Create output BrainData
    result = bd[0].copy()
    result.data = stacked_betas
    result._design_columns = task_columns

    return result, task_columns, condition_labels, run_labels


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
        bids_module = attempt_to_import("bids", "bids")
        if bids_module is None:
            raise ImportError(
                "pybids required for BIDS loading. Install with: pip install pybids"
            )
        BIDSLayout = bids_module.BIDSLayout

        # Create layout if path provided
        if isinstance(layout, (str, Path)):
            layout = BIDSLayout(layout, validate=False)

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
        from scipy import stats
        from .brain_data import BrainData

        axis = self._normalize_axis(axis)
        if axis != 0:
            raise ValueError(
                "ttest only supports axis=0 (across images). "
                "For per-image tests, use apply()."
            )

        # Get tensor - requires uniform observation counts
        tensor = self.to_tensor()  # (n_images, n_obs, n_voxels)

        # Compute t-test across axis 0
        t_stat_arr, p_val_arr = stats.ttest_1samp(tensor, popmean, axis=0)

        # Package as BrainData
        t_stat = BrainData(mask=self._mask)
        t_stat.data = t_stat_arr

        p_val = BrainData(mask=self._mask)
        p_val.data = p_val_arr

        return t_stat, p_val

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
        from scipy import stats
        from .brain_data import BrainData

        # Validate mask compatibility
        if self._mask.shape != other._mask.shape:
            raise ValueError(
                f"Collections must have same mask shape. "
                f"Got {self._mask.shape} and {other._mask.shape}."
            )

        # Get tensors
        tensor1 = self.to_tensor()  # (n1, n_obs, n_voxels)
        tensor2 = other.to_tensor()  # (n2, n_obs, n_voxels)

        # Check obs counts match
        if tensor1.shape[1] != tensor2.shape[1]:
            raise ValueError(
                f"Collections must have same observation count per image. "
                f"Got {tensor1.shape[1]} and {tensor2.shape[1]}."
            )

        # Compute two-sample t-test across axis 0
        t_stat_arr, p_val_arr = stats.ttest_ind(
            tensor1, tensor2, axis=0, equal_var=equal_var
        )

        # Package as BrainData
        t_stat = BrainData(mask=self._mask)
        t_stat.data = t_stat_arr

        p_val = BrainData(mask=self._mask)
        p_val.data = p_val_arr

        return t_stat, p_val

    def permutation_test(
        self,
        n_permute: int = 5000,
        tail: int = 2,
        parallel: str | None = "cpu",
        n_jobs: int = -1,
        max_gpu_memory_gb: float = 4.0,
        random_state: int | None = None,
        return_null: bool = False,
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
            parallel: Parallelization method:
                - 'cpu': CPU parallelization via joblib (default)
                - 'gpu': GPU acceleration via PyTorch
                - None: Single-threaded (for debugging)
            n_jobs: Number of CPU cores (default: -1 = all cores).
            max_gpu_memory_gb: GPU memory budget (default: 4.0 GB).
            random_state: Random seed for reproducibility.
            return_null: If True, include null distribution in result.

        Returns:
            dict with keys:
                - 'mean': BrainData with observed mean across images
                - 'p': BrainData with p-values
                - 'null_dist': np.ndarray (if return_null=True)
                - 'parallel': parallelization method used

        Raises:
            ValueError: If images have variable observation counts.

        Examples:
            >>> result = bc.permutation_test(n_permute=5000)
            >>> mean_bd, p_bd = result['mean'], result['p']

            >>> # With GPU acceleration
            >>> result = bc.permutation_test(parallel='gpu')
        """
        from nltools.algorithms.inference import one_sample_permutation_test
        from .brain_data import BrainData

        # Get tensor - requires uniform observation counts
        tensor = self.to_tensor()  # (n_images, n_obs, n_voxels)
        n_images, n_obs, n_voxels = tensor.shape

        # For each observation/timepoint, run permutation test across images
        mean_results = []
        p_results = []
        null_dists = [] if return_null else None

        iterator = range(n_obs)
        if tqdm is not None:
            iterator = tqdm.tqdm(iterator, desc="Permutation tests")

        for obs_idx in iterator:
            # Data for this observation: (n_images, n_voxels)
            data = tensor[:, obs_idx, :]

            result = one_sample_permutation_test(
                data,
                n_permute=n_permute,
                tail=tail,
                return_null=return_null,
                parallel=parallel,
                n_jobs=n_jobs,
                max_gpu_memory_gb=max_gpu_memory_gb,
                random_state=random_state,
            )

            mean_results.append(result["mean"])
            p_results.append(result["p"])
            if return_null:
                null_dists.append(result["null_dist"])

        # Stack results: (n_obs, n_voxels)
        mean_arr = np.vstack(mean_results)
        p_arr = np.vstack(p_results)

        # Package as BrainData
        mean_bd = BrainData(mask=self._mask)
        mean_bd.data = mean_arr if n_obs > 1 else mean_arr.squeeze()

        p_bd = BrainData(mask=self._mask)
        p_bd.data = p_arr if n_obs > 1 else p_arr.squeeze()

        result_dict = {
            "mean": mean_bd,
            "p": p_bd,
            "parallel": parallel,
        }

        if return_null:
            result_dict["null_dist"] = np.array(null_dists)

        return result_dict

    def permutation_test2(
        self,
        other: "BrainCollection",
        n_permute: int = 5000,
        tail: int = 2,
        parallel: str | None = "cpu",
        n_jobs: int = -1,
        max_gpu_memory_gb: float = 4.0,
        random_state: int | None = None,
        return_null: bool = False,
    ) -> dict:
        """
        Two-sample permutation test between collections.

        Tests whether two collections have different means using group
        label permutation. More robust than parametric t-test.

        Args:
            other: Another BrainCollection to compare against.
            n_permute: Number of permutations (default: 5000).
            tail: Test type - 1 for one-tailed, 2 for two-tailed (default).
            parallel: Parallelization method ('cpu', 'gpu', or None).
            n_jobs: Number of CPU cores (default: -1 = all cores).
            max_gpu_memory_gb: GPU memory budget (default: 4.0 GB).
            random_state: Random seed for reproducibility.
            return_null: If True, include null distribution in result.

        Returns:
            dict with keys:
                - 'mean_diff': BrainData with observed mean difference
                - 'p': BrainData with p-values
                - 'null_dist': np.ndarray (if return_null=True)
                - 'parallel': parallelization method used

        Examples:
            >>> result = patients.permutation_test2(controls)
            >>> diff_bd, p_bd = result['mean_diff'], result['p']
        """
        from nltools.algorithms.inference import two_sample_permutation_test
        from .brain_data import BrainData

        # Validate mask compatibility
        if self._mask.shape != other._mask.shape:
            raise ValueError(
                f"Collections must have same mask shape. "
                f"Got {self._mask.shape} and {other._mask.shape}."
            )

        # Get tensors
        tensor1 = self.to_tensor()  # (n1, n_obs, n_voxels)
        tensor2 = other.to_tensor()  # (n2, n_obs, n_voxels)

        if tensor1.shape[1] != tensor2.shape[1]:
            raise ValueError(
                f"Collections must have same observation count. "
                f"Got {tensor1.shape[1]} and {tensor2.shape[1]}."
            )

        n_obs = tensor1.shape[1]

        diff_results = []
        p_results = []
        null_dists = [] if return_null else None

        iterator = range(n_obs)
        if tqdm is not None:
            iterator = tqdm.tqdm(iterator, desc="Two-sample permutation tests")

        for obs_idx in iterator:
            data1 = tensor1[:, obs_idx, :]  # (n1, n_voxels)
            data2 = tensor2[:, obs_idx, :]  # (n2, n_voxels)

            result = two_sample_permutation_test(
                data1,
                data2,
                n_permute=n_permute,
                tail=tail,
                return_null=return_null,
                parallel=parallel,
                n_jobs=n_jobs,
                max_gpu_memory_gb=max_gpu_memory_gb,
                random_state=random_state,
            )

            diff_results.append(result["mean_diff"])
            p_results.append(result["p"])
            if return_null:
                null_dists.append(result["null_dist"])

        # Stack results
        diff_arr = np.vstack(diff_results)
        p_arr = np.vstack(p_results)

        diff_bd = BrainData(mask=self._mask)
        diff_bd.data = diff_arr if n_obs > 1 else diff_arr.squeeze()

        p_bd = BrainData(mask=self._mask)
        p_bd.data = p_arr if n_obs > 1 else p_arr.squeeze()

        result_dict = {
            "mean_diff": diff_bd,
            "p": p_bd,
            "parallel": parallel,
        }

        if return_null:
            result_dict["null_dist"] = np.array(null_dists)

        return result_dict

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
        from scipy import stats
        from .brain_data import BrainData

        # Resolve groups
        if isinstance(groups, str):
            if groups not in self._metadata.columns:
                raise KeyError(
                    f"Column '{groups}' not found in metadata. "
                    f"Available: {list(self._metadata.columns)}"
                )
            group_labels = self._metadata[groups].values
        else:
            group_labels = np.asarray(groups)
            if len(group_labels) != self.n_images:
                raise ValueError(
                    f"groups length ({len(group_labels)}) must match "
                    f"n_images ({self.n_images})"
                )

        # Get tensor
        tensor = self.to_tensor()  # (n_images, n_obs, n_voxels)
        n_images, n_obs, n_voxels = tensor.shape

        # Get unique groups
        unique_groups = np.unique(group_labels)
        if len(unique_groups) < 2:
            raise ValueError("ANOVA requires at least 2 groups")

        # Compute F-test for each observation
        f_results = []
        p_results = []

        for obs_idx in range(n_obs):
            data = tensor[:, obs_idx, :]  # (n_images, n_voxels)

            # Split by groups
            group_data = [data[group_labels == g] for g in unique_groups]

            # F-test across groups
            f_stat_arr, p_val_arr = stats.f_oneway(*group_data)

            f_results.append(f_stat_arr)
            p_results.append(p_val_arr)

        # Stack results
        f_arr = np.vstack(f_results) if n_obs > 1 else np.array(f_results[0])
        p_arr = np.vstack(p_results) if n_obs > 1 else np.array(p_results[0])

        # Package as BrainData
        f_stat = BrainData(mask=self._mask)
        f_stat.data = f_arr

        p_val = BrainData(mask=self._mask)
        p_val.data = p_arr

        return f_stat, p_val

    # =========================================================================
    # Transformation Methods
    # =========================================================================

    def map(
        self,
        fn: callable,
        axis: int | str = 0,
        n_jobs: int = 1,
        show_progress: bool = True,
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
            show_progress: Show tqdm progress bar. Default True.

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
        axis = self._normalize_axis(axis)

        if axis == 0:
            return self._map_axis0(fn, n_jobs, show_progress)
        elif axis == 1:
            return self._map_axis1(fn, n_jobs, show_progress)
        elif axis == 2:
            return self._map_axis2(fn, n_jobs, show_progress)
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, or 2.")

    def _map_axis0(
        self,
        fn: callable,
        n_jobs: int,
        show_progress: bool,
    ) -> "BrainCollection":
        """Map function over images (axis=0)."""
        from joblib import Parallel, delayed

        indices = range(self.n_images)

        if n_jobs == 1:
            # Sequential processing
            if show_progress and tqdm is not None:
                indices = tqdm.tqdm(indices, desc="Mapping over images")

            results = []
            for i in indices:
                bd = self._load_item(i)
                results.append(fn(bd))
        else:
            # Parallel processing
            def _process(i):
                bd = self._load_item(i)
                return fn(bd)

            if show_progress and tqdm is not None:
                indices = tqdm.tqdm(indices, desc="Mapping over images")

            results = Parallel(n_jobs=n_jobs)(delayed(_process)(i) for i in indices)

        return BrainCollection(results, mask=self._mask, metadata=self._metadata)

    def _map_axis1(
        self,
        fn: callable,
        n_jobs: int,
        show_progress: bool,
    ) -> "BrainCollection":
        """Map function over timepoints (axis=1)."""
        from .brain_data import BrainData

        # Ensure all sample counts known and uniform
        for i in range(len(self)):
            if self._sample_counts[i] is None:
                self._load_item(i)

        unique_counts = set(self._sample_counts)
        if len(unique_counts) > 1:
            raise ValueError(
                f"map(axis=1) requires uniform observation counts. "
                f"Found: {sorted(unique_counts)}"
            )

        n_obs = self._sample_counts[0]
        indices = range(n_obs)

        if show_progress and tqdm is not None:
            indices = tqdm.tqdm(indices, desc="Mapping over timepoints")

        # For each timepoint, create a BrainCollection slice and apply fn
        results_per_t = []
        for t in indices:
            # Get timepoint slice: bc[:, t] returns BrainCollection with 1 obs each
            t_slice = self[:, t]
            result = fn(t_slice)
            results_per_t.append(result)

        # results_per_t is list of BrainData (one per timepoint)
        # Need to reassemble into images
        # Each result should be a BrainData with shape (n_voxels,) or similar
        # Stack them back into (n_obs, n_voxels) per image

        # If fn returns BrainData, stack across timepoints
        if isinstance(results_per_t[0], BrainData):
            # Reassemble: each image gets data from all timepoints
            new_items = []
            for img_idx in range(self.n_images):
                img_data = []
                for t in range(n_obs):
                    # results_per_t[t] is result for timepoint t
                    # If it's a single BrainData (reduced), extract scalar per image
                    if hasattr(results_per_t[t], "data"):
                        img_data.append(results_per_t[t].data)
                stacked = np.vstack(img_data) if len(img_data) > 1 else img_data[0]
                new_bd = BrainData(mask=self._mask)
                new_bd.data = stacked
                new_items.append(new_bd)
            return BrainCollection(new_items, mask=self._mask, metadata=self._metadata)
        else:
            raise TypeError(
                f"map(axis=1) function must return BrainData, got {type(results_per_t[0])}"
            )

    def _map_axis2(
        self,
        fn: callable,
        n_jobs: int,
        show_progress: bool,
    ) -> "BrainCollection":
        """Map function over voxels (axis=2) per image."""
        from .brain_data import BrainData

        indices = range(self.n_images)
        if show_progress and tqdm is not None:
            indices = tqdm.tqdm(indices, desc="Mapping over voxels")

        results = []
        for i in indices:
            bd = self._load_item(i)
            data = bd.data
            if data.ndim == 1:
                data = data[np.newaxis, :]

            # Apply fn to each voxel's timeseries (each column)
            # fn receives (n_obs,) array, returns (n_obs,) or scalar
            transformed_cols = []
            for v in range(data.shape[1]):
                transformed_cols.append(fn(data[:, v]))

            transformed = np.column_stack(transformed_cols)

            new_bd = BrainData(mask=self._mask)
            new_bd.data = (
                transformed.squeeze() if transformed.shape[0] == 1 else transformed
            )
            results.append(new_bd)

        return BrainCollection(results, mask=self._mask, metadata=self._metadata)

    def filter(
        self,
        predicate: callable | list | np.ndarray | "pd.Series",
    ) -> "BrainCollection":
        """
        Filter collection by predicate.

        Args:
            predicate: Filter condition. Can be:
                - callable: fn(BrainData) → bool
                - list/ndarray: Boolean mask of length n_images
                - pd.Series: Boolean series (index ignored)

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
        if callable(predicate):
            # Apply predicate to each image
            mask = []
            for i in range(self.n_images):
                bd = self._load_item(i)
                mask.append(bool(predicate(bd)))
            mask = np.array(mask)
        elif isinstance(predicate, pd.Series):
            mask = predicate.values.astype(bool)
        else:
            mask = np.asarray(predicate, dtype=bool)

        if len(mask) != self.n_images:
            raise ValueError(
                f"Predicate length ({len(mask)}) must match n_images ({self.n_images})"
            )

        indices = np.where(mask)[0].tolist()
        return self._subset(indices)

    # =========================================================================
    # Convenience Methods (Delegators to BrainData methods)
    # =========================================================================

    def standardize(
        self,
        axis: int = 0,
        method: str = "center",
        n_jobs: int = 1,
        show_progress: bool = True,
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
            show_progress: Show progress bar.

        Returns:
            BrainCollection with standardized images.

        Examples:
            >>> bc.standardize()  # Center each image across time
            >>> bc.standardize(method='zscore')  # Z-score each image
            >>> bc.standardize(axis=1)  # Standardize across voxels
        """
        return self.map(
            lambda bd: bd.standardize(axis=axis, method=method),
            axis=0,
            n_jobs=n_jobs,
            show_progress=show_progress,
        )

    def smooth(
        self,
        fwhm: float,
        n_jobs: int = 1,
        show_progress: bool = True,
    ) -> "BrainCollection":
        """
        Spatially smooth each image.

        Delegates to BrainData.smooth() for each image.

        Args:
            fwhm: Full width at half maximum of Gaussian kernel in mm.
            n_jobs: Number of parallel jobs.
            show_progress: Show progress bar.

        Returns:
            BrainCollection with smoothed images.

        Examples:
            >>> bc.smooth(fwhm=6)  # 6mm FWHM smoothing
        """
        return self.map(
            lambda bd: bd.smooth(fwhm),
            axis=0,
            n_jobs=n_jobs,
            show_progress=show_progress,
        )

    def threshold(
        self,
        upper: float | str | None = None,
        lower: float | str | None = None,
        binarize: bool = False,
        coerce_nan: bool = True,
        n_jobs: int = 1,
        show_progress: bool = True,
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
            show_progress: Show progress bar.

        Returns:
            BrainCollection with thresholded images.

        Examples:
            >>> bc.threshold(lower=0)  # Zero out negative values
            >>> bc.threshold(upper='95%')  # Keep top 5%
            >>> bc.threshold(lower=2, binarize=True)  # Binary mask
        """
        return self.map(
            lambda bd: bd.threshold(
                upper=upper, lower=lower, binarize=binarize, coerce_nan=coerce_nan
            ),
            axis=0,
            n_jobs=n_jobs,
            show_progress=show_progress,
        )

    def detrend(
        self,
        method: str = "linear",
        n_jobs: int = 1,
        show_progress: bool = True,
    ) -> "BrainCollection":
        """
        Remove trend from each image.

        Delegates to BrainData.detrend() for each image.

        Args:
            method: 'linear' or 'constant'.
            n_jobs: Number of parallel jobs.
            show_progress: Show progress bar.

        Returns:
            BrainCollection with detrended images.

        Examples:
            >>> bc.detrend()  # Remove linear trend
            >>> bc.detrend(method='constant')  # Remove mean only
        """
        return self.map(
            lambda bd: bd.detrend(method=method),
            axis=0,
            n_jobs=n_jobs,
            show_progress=show_progress,
        )

    # =========================================================================
    # ISC (Intersubject Correlation) Methods
    # =========================================================================

    def _extract_for_isc(
        self,
        roi_mask: "nib.Nifti1Image | Path | str | None" = None,
        radius: float | None = 6.0,
        show_progress: bool = True,
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
            show_progress: Show progress bar during extraction.

        Returns:
            Tuple of:
                - extracted_data: Array of shape (n_obs, n_subjects, n_features)
                - extraction_info: Dict with metadata for projection back:
                    - 'mode': 'roi', 'searchlight', or 'voxelwise'
                    - 'n_features': Number of features
                    - 'roi_mask': ROI mask if mode='roi'
                    - 'neighborhoods': SphereNeighborhoods if mode='searchlight'
        """
        n_obs = self.shape[1]
        if n_obs is None:
            raise ValueError(
                "ISC requires uniform observation counts across subjects. "
                f"Got variable counts: {[bd.shape[0] for bd in self]}"
            )

        # Determine extraction mode
        if roi_mask is not None:
            return self._extract_roi(roi_mask, show_progress)
        elif radius is not None:
            return self._extract_searchlight(radius, show_progress)
        else:
            return self._extract_voxelwise(show_progress)

    def _extract_roi(
        self,
        roi_mask: "nib.Nifti1Image | Path | str",
        show_progress: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """Extract mean signal per ROI."""
        from nilearn.maskers import NiftiLabelsMasker

        # Load ROI mask if path
        if isinstance(roi_mask, (str, Path)):
            roi_mask = nib.load(roi_mask)

        # Create masker
        masker = NiftiLabelsMasker(
            labels_img=roi_mask,
            standardize=False,
            resampling_target="data",
        )

        n_subjects = len(self)
        n_obs = self.shape[1]

        # Get number of ROIs from first subject
        first_img = self[0].to_nifti()
        first_signals = masker.fit_transform(first_img)
        n_rois = first_signals.shape[1]

        # Preallocate output: (n_obs, n_subjects, n_rois)
        extracted = np.zeros((n_obs, n_subjects, n_rois), dtype=np.float32)
        extracted[:, 0, :] = first_signals

        # Extract remaining subjects
        iterator = range(1, n_subjects)
        if show_progress:
            iterator = tqdm.tqdm(iterator, desc="Extracting ROIs", unit="subjects")

        for i in iterator:
            img = self[i].to_nifti()
            signals = masker.transform(img)
            extracted[:, i, :] = signals

        extraction_info = {
            "mode": "roi",
            "n_features": n_rois,
            "roi_mask": roi_mask,
            "masker": masker,
        }

        return extracted, extraction_info

    def _extract_searchlight(
        self,
        radius: float,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """Extract mean signal per searchlight sphere."""
        from nltools.neighborhoods import compute_searchlight_neighborhoods

        n_subjects = len(self)
        n_obs = self.shape[1]
        n_voxels = self.n_voxels

        # Get cached neighborhoods
        neighborhoods = compute_searchlight_neighborhoods(
            self._mask, radius_mm=radius, use_cache=True
        )

        # Preallocate output: (n_obs, n_subjects, n_voxels)
        extracted = np.zeros((n_obs, n_subjects, n_voxels), dtype=np.float32)

        # Extract each subject
        iterator = range(n_subjects)
        if show_progress:
            iterator = tqdm.tqdm(
                iterator, desc="Extracting searchlight", unit="subjects"
            )

        for subj_idx in iterator:
            bd = self[subj_idx]
            data = bd.data  # (n_obs, n_voxels)

            # Compute mean per sphere neighborhood
            for voxel_idx, neighbor_indices in neighborhoods.iter_neighborhoods():
                extracted[:, subj_idx, voxel_idx] = data[:, neighbor_indices].mean(
                    axis=1
                )

        extraction_info = {
            "mode": "searchlight",
            "n_features": n_voxels,
            "radius": radius,
            "neighborhoods": neighborhoods,
        }

        return extracted, extraction_info

    def _extract_voxelwise(
        self,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """Extract raw voxel data."""
        n_subjects = len(self)
        n_obs = self.shape[1]
        n_voxels = self.n_voxels

        # Preallocate output: (n_obs, n_subjects, n_voxels)
        extracted = np.zeros((n_obs, n_subjects, n_voxels), dtype=np.float32)

        # Extract each subject
        iterator = range(n_subjects)
        if show_progress:
            iterator = tqdm.tqdm(iterator, desc="Extracting voxels", unit="subjects")

        for subj_idx in iterator:
            bd = self[subj_idx]
            extracted[:, subj_idx, :] = bd.data

        extraction_info = {
            "mode": "voxelwise",
            "n_features": n_voxels,
        }

        return extracted, extraction_info

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
        from .brain_data import BrainData

        mode = extraction_info["mode"]

        if mode == "roi":
            # For ROI mode, values are per-ROI, not per-voxel
            # Return a BrainData with ROI values directly (not expanded to voxels)
            result = BrainData(mask=self._mask)
            result.data = values
            return result

        elif mode in ("searchlight", "voxelwise"):
            # Direct assignment
            result = BrainData(mask=self._mask)
            result.data = values
            return result

        else:
            raise ValueError(f"Unknown extraction mode: {mode}")

    def isc(
        self,
        method: str = "loo",
        roi_mask: "nib.Nifti1Image | Path | str | None" = None,
        radius: float | None = 6.0,
        metric: str = "median",
        parallel: str = "cpu",
        n_jobs: int = -1,
        show_progress: bool = True,
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
            parallel: Parallelization method ('cpu', 'gpu', or None).
            n_jobs: Number of parallel jobs (-1 = all cores).
            show_progress: Show progress bar during extraction.

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
        from nltools.algorithms.inference.isc import (
            _compute_loo_isc,
            _compute_pairwise_isc,
        )

        # Extract data
        extracted, extraction_info = self._extract_for_isc(
            roi_mask=roi_mask,
            radius=radius,
            show_progress=show_progress,
        )

        # Data is (n_obs, n_subjects, n_features)
        # ISC functions expect this shape

        # Compute ISC
        if method == "loo":
            # LOO ISC: (n_subjects, n_features)
            backend = "torch" if parallel == "gpu" else "numpy"
            loo_values = _compute_loo_isc(extracted, backend=backend)

            # Aggregate across subjects
            if metric == "median":
                isc_values = np.median(loo_values, axis=0)
            elif metric == "mean":
                z = np.arctanh(np.clip(loo_values, -0.9999, 0.9999))
                isc_values = np.tanh(np.mean(z, axis=0))
            else:
                raise ValueError(f"metric must be 'median' or 'mean', got {metric}")

        elif method == "pairwise":
            # Pairwise ISC: (n_pairs, n_features)
            pairwise = _compute_pairwise_isc(extracted, backend="numpy")

            # Aggregate across pairs
            if metric == "median":
                isc_values = np.nanmedian(pairwise, axis=0)
            elif metric == "mean":
                z = np.arctanh(np.clip(pairwise, -0.9999, 0.9999))
                isc_values = np.tanh(np.nanmean(z, axis=0))
            else:
                raise ValueError(f"metric must be 'median' or 'mean', got {metric}")

        else:
            raise ValueError(f"method must be 'loo' or 'pairwise', got {method}")

        # Project back to brain space
        isc_brain = self._project_to_brain(isc_values, extraction_info)

        return {
            "isc": isc_brain,
            "method": method,
            "extraction": extraction_info["mode"],
            "n_subjects": len(self),
            "extraction_info": extraction_info,
        }

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
        parallel: str = "cpu",
        n_jobs: int = -1,
        random_state: int | None = None,
        return_null: bool = False,
        show_progress: bool = True,
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
            parallel: Parallelization method ('cpu', 'gpu', or None).
            n_jobs: Number of parallel jobs (-1 = all cores).
            random_state: Random seed for reproducibility.
            return_null: If True, include null distribution in results.
            show_progress: Show progress bar during extraction and permutation.

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
            ...     parallel='gpu'
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
        from nltools.algorithms.inference.isc import isc_permutation_test

        # Map method names
        summary_statistic = "leave-one-out" if method == "loo" else method

        # Extract data
        extracted, extraction_info = self._extract_for_isc(
            roi_mask=roi_mask,
            radius=radius,
            show_progress=show_progress,
        )

        # Run permutation test
        result = isc_permutation_test(
            data=extracted,
            n_permute=n_permute,
            metric=metric,
            summary_statistic=summary_statistic,
            method=permutation_method,
            ci_percentile=ci_percentile,
            tail=tail,
            return_null=return_null,
            progress_bar=show_progress,
            parallel=parallel,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        # Project results to brain space
        isc_brain = self._project_to_brain(result["isc"], extraction_info)
        p_brain = self._project_to_brain(result["p"], extraction_info)
        ci_lower = self._project_to_brain(result["ci"][0], extraction_info)
        ci_upper = self._project_to_brain(result["ci"][1], extraction_info)

        output = {
            "isc": isc_brain,
            "p": p_brain,
            "ci": (ci_lower, ci_upper),
            "method": method,
            "permutation_method": permutation_method,
            "extraction": extraction_info["mode"],
            "n_subjects": len(self),
            "n_permute": n_permute,
        }

        if return_null:
            output["null_dist"] = result.get("null_dist")

        return output

    def cv(
        self,
        k: int = None,
        scheme: str = "kfold",
        split_by: str = None,
        groups: np.ndarray = None,
        random_state: int = None,
        **kwargs,
    ) -> "BrainCollectionPipeline":
        """Create a cross-validation pipeline for multi-subject analysis.

        Returns a pipeline object that enables fluent, chainable transforms
        with cross-validation across subjects or runs.

        Args:
            k: Number of folds (for kfold scheme). Defaults to 5.
            scheme: CV scheme type. Options:
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
            >>> result = bc.cv(scheme='loso').normalize().predict(subject_labels, algorithm='svm')
            >>> print(f"Mean accuracy: {result.mean_score:.2%}")

            >>> # With preprocessing
            >>> result = (bc
            ...     .cv(scheme='loso')
            ...     .normalize()
            ...     .reduce(n_components=50)
            ...     .predict(labels))

            >>> # Run-based CV with metadata
            >>> result = bc.cv(scheme='loro', split_by='run').predict(y)

        See Also:
            BrainCollectionPipeline: For available transforms and terminals.
            CVScheme: For CV scheme configuration details.
        """
        from nltools.pipelines.cv import CVScheme

        # Handle split_by -> groups conversion from metadata
        if groups is None and split_by is not None:
            if self.metadata is not None and split_by in self.metadata.columns:
                groups = np.array(self.metadata[split_by])

        # Create CV scheme
        cv_scheme = CVScheme(
            k=k,
            scheme=scheme,
            split_by=split_by,
            random_state=random_state,
            **kwargs,
        )

        return BrainCollectionPipeline(self, cv=cv_scheme, groups=groups)

    def fit(
        self,
        model: str,
        X: "pd.DataFrame | np.ndarray | str | list",
        cv: int | None = None,
        scale: bool = True,
        scale_value: float = 100.0,
        show_progress: bool = True,
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
            show_progress: Show progress bar during fitting.
            **kwargs: Model-specific arguments passed to _fit_glm or _fit_ridge:
                - GLM: return_stats, save
                - Ridge: alpha, output, save, backend, random_state

        Returns:
            FittedBrainCollection wrapping the fitted results. Supports:
                - .results: Access underlying BrainCollection(s) directly
                - .betas: Convenience accessor for beta coefficients (GLM)
                - .pool(): Aggregate across subjects for group analysis
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
        if model == "glm":
            results = self._fit_glm(
                X=X,
                scale=scale,
                scale_value=scale_value,
                show_progress=show_progress,
                **kwargs,
            )
            # Extract condition names from results
            condition_names = None
            if isinstance(results, dict):
                betas = results.get("betas")
                if betas is not None and hasattr(betas, "_design_columns"):
                    condition_names = betas._design_columns
            elif hasattr(results, "_design_columns"):
                condition_names = results._design_columns

            return FittedBrainCollection(
                brain_collection=self,
                fitted_results=results,
                model=model,
                condition_names=condition_names,
            )
        elif model == "ridge":
            # Handle cv default for Ridge
            if cv is None:
                output = kwargs.get("output", "scores")
                if output in ("scores", "both"):
                    cv = 5  # Default for scores
            results = self.fit_ridge(
                X=X,
                cv=cv,
                scale=scale,
                scale_value=scale_value,
                show_progress=show_progress,
                **kwargs,
            )
            return FittedBrainCollection(
                brain_collection=self,
                fitted_results=results,
                model=model,
                condition_names=None,  # Ridge doesn't have condition names
            )
        else:
            raise ValueError(f"Unknown model: '{model}'. Supported: 'glm', 'ridge'")

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
        show_progress: bool = True,
        by_run: bool = False,
        run_column: str = "run",
        run_lengths: int | list[int] | None = None,
    ) -> "BrainCollection":
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
            save: Dict mapping output type to path template, e.g.:
                {'betas': 'output/{subject}_betas.nii.gz',
                 't': 'output/{subject}_tstat.nii.gz'}
                Supports {subject}, {session}, {idx}, and other metadata columns.
            show_progress: Show progress bar during fitting.
            by_run: If True, fit GLM separately per run and return run-level betas.
                This enables MVPA decoding with leave-one-run-out CV.
                Each subject will have (n_runs * n_conditions, n_voxels) betas.
            run_column: Column name in events identifying runs (default 'run').
            run_lengths: Number of TRs per run. Required when by_run=True.
                - int: All runs have same length
                - list[int]: Different length per run
                - None: Will attempt to infer equal-length runs from total scans

        Returns:
            BrainCollection where each BrainData has shape:
                - (n_task_regressors, n_voxels) if by_run=False (default)
                - (n_runs * n_task_regressors, n_voxels) if by_run=True
            The ._design_columns attribute stores task regressor names.
            If by_run=True, also stores ._condition_labels and ._run_labels.
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
        # Handle return_residuals shorthand
        if return_residuals and return_stats is None:
            return_stats = ["residual"]
        elif return_residuals and "residual" not in return_stats:
            return_stats = list(return_stats) + ["residual"]

        # Validate return_stats
        valid_stats = {"t", "r2", "p", "se", "residual"}
        if return_stats is not None:
            invalid = set(return_stats) - valid_stats
            if invalid:
                raise ValueError(
                    f"Invalid return_stats: {invalid}. Valid options: {valid_stats}"
                )

        # Validate by_run parameters
        if by_run:
            if run_column not in events.columns:
                raise ValueError(
                    f"by_run=True requires '{run_column}' column in events. "
                    f"Available columns: {list(events.columns)}"
                )
            runs = sorted(events[run_column].unique())
            if return_stats is not None:
                raise NotImplementedError(
                    "return_stats is not yet supported with by_run=True. "
                    "Only beta coefficients are returned."
                )
        else:
            runs = None

        # Resolve confounds to per-subject list
        confounds_list = self._resolve_confounds(confounds)

        # Progress bar setup
        iterator = range(len(self))
        if show_progress and tqdm is not None:
            iterator = tqdm.tqdm(iterator, desc="Fitting GLM", unit="subject")

        # Storage for results
        beta_data_list = []
        beta_metadata = []
        stat_data = {stat: [] for stat in (return_stats or [])}
        task_columns = None  # Will be set on first subject
        # For by_run mode: store labels (same for all subjects)
        all_condition_labels = None
        all_run_labels = None

        for i in iterator:
            # Load subject data
            bd = self._load_item(i)
            metadata_row = self._metadata.iloc[i]
            n_scans = bd.shape[0]

            # Get subject-specific confounds
            subj_confounds = confounds_list[i] if confounds_list else None

            if by_run:
                # ===== BY_RUN MODE: Fit GLM separately per run =====
                # Load confounds as DataFrame for slicing
                if subj_confounds is not None:
                    if isinstance(subj_confounds, (str, Path)):
                        conf_path = Path(subj_confounds)
                        sep = "\t" if conf_path.suffix in [".tsv", ".txt"] else ","
                        subj_confounds = pd.read_csv(conf_path, sep=sep)
                        if confound_columns:
                            subj_confounds = subj_confounds[confound_columns]
                        subj_confounds = subj_confounds.fillna(0)

                task_betas, task_cols, cond_labels, run_lbls = _fit_glm_by_run(
                    bd=bd,
                    events=events,
                    runs=runs,
                    run_column=run_column,
                    run_lengths=run_lengths,
                    t_r=t_r,
                    confounds=subj_confounds,
                    confound_columns=confound_columns,
                    hrf_model=hrf_model,
                    drift_model=drift_model,
                    high_pass=high_pass,
                    scale=scale,
                    scale_value=scale_value,
                )

                # Store task columns and labels from first subject
                if task_columns is None:
                    task_columns = task_cols
                    all_condition_labels = cond_labels
                    all_run_labels = run_lbls

            else:
                # ===== STANDARD MODE: Single GLM per subject =====
                # Build design matrix
                dm, task_cols = _build_subject_design_matrix(
                    events=events,
                    n_scans=n_scans,
                    t_r=t_r,
                    confounds=subj_confounds,
                    confound_columns=confound_columns,
                    hrf_model=hrf_model,
                    drift_model=drift_model,
                    high_pass=high_pass,
                )

                # Store task columns for later
                if task_columns is None:
                    task_columns = task_cols

                # Apply scaling if requested
                if scale:
                    bd = bd.scale(scale_value)

                # Fit GLM
                bd.fit(model="glm", X=dm)

                # Extract task betas only (not confounds/drift)
                task_indices = [dm.columns.get_loc(col) for col in task_cols]
                task_betas_data = bd.glm_betas.data[task_indices, :]

                # Create BrainData for task betas by copying structure
                task_betas = bd[0].copy()
                task_betas.data = task_betas_data
                task_betas._design_columns = task_cols  # Store for contrast parsing

            # Save if requested
            if save and "betas" in save:
                save_path = _resolve_save_path(save["betas"], metadata_row, i)
                task_betas.write(str(save_path))

            beta_data_list.append(task_betas)
            beta_metadata.append(metadata_row.to_dict())

            # Extract optional stats (standard mode only, validated earlier)
            if return_stats:
                for stat in return_stats:
                    if stat == "t":
                        stat_data_arr = bd.glm_t.data[task_indices, :]
                    elif stat == "p":
                        stat_data_arr = bd.glm_p.data[task_indices, :]
                    elif stat == "se":
                        stat_data_arr = bd.glm_se.data[task_indices, :]
                    elif stat == "r2":
                        stat_data_arr = bd.glm_r2.data  # Shape (1, n_voxels)
                    elif stat == "residual":
                        stat_data_arr = (
                            bd.glm_residual.data
                        )  # Shape (n_scans, n_voxels)

                    # Create BrainData by copying structure
                    stat_bd = bd[0].copy()
                    stat_bd.data = stat_data_arr

                    if save and stat in save:
                        save_path = _resolve_save_path(save[stat], metadata_row, i)
                        stat_bd.write(str(save_path))

                    stat_data[stat].append(stat_bd)

            # Unload to free memory (only works for path-based collections)
            self.unload([i])

        # Build result collection
        beta_collection = BrainCollection(
            beta_data_list,
            mask=self.mask,
            metadata=pd.DataFrame(beta_metadata),
        )
        beta_collection._design_columns = task_columns

        # Store run-level labels for MVPA workflows
        if by_run:
            beta_collection._condition_labels = all_condition_labels
            beta_collection._run_labels = all_run_labels

        # Return based on what was requested
        if return_stats:
            result = {"betas": beta_collection}
            for stat in return_stats:
                stat_collection = BrainCollection(
                    stat_data[stat],
                    mask=self.mask,
                    metadata=pd.DataFrame(beta_metadata),
                )
                result[stat] = stat_collection
            return result

        return beta_collection

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
        show_progress: bool = True,
        by_run: bool = False,
        run_column: str = "run",
        run_lengths: int | list[int] | None = None,
    ) -> "BrainCollection":
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
            show_progress: Show progress bar during fitting.
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
        return self.fit_glm(
            events=events,
            t_r=t_r,
            confounds=confounds,
            confound_columns=confound_columns,
            hrf_model=hrf_model,
            drift_model=drift_model,
            high_pass=high_pass,
            scale=scale,
            scale_value=scale_value,
            return_stats=return_stats,
            return_residuals=return_residuals,
            save=save,
            show_progress=show_progress,
            by_run=by_run,
            run_column=run_column,
            run_lengths=run_lengths,
        )

    def _resolve_confounds(
        self,
        confounds: str | list[pd.DataFrame | Path | str] | None,
    ) -> list[pd.DataFrame | Path | str | None] | None:
        """Resolve confounds argument to per-subject list.

        Args:
            confounds: Either:
                - str: Column name in metadata containing confound paths
                - list: Already per-subject list of DataFrames or paths
                - None: No confounds

        Returns:
            List of confounds (one per subject) or None
        """
        if confounds is None:
            return None

        if isinstance(confounds, str):
            # It's a metadata column name
            if confounds not in self._metadata.columns:
                raise KeyError(
                    f"Confounds column '{confounds}' not found in metadata. "
                    f"Available: {list(self._metadata.columns)}"
                )
            return list(self._metadata[confounds])

        if isinstance(confounds, list):
            if len(confounds) != len(self):
                raise ValueError(
                    f"confounds list length ({len(confounds)}) must match "
                    f"collection length ({len(self)})"
                )
            return confounds

        raise TypeError(
            f"confounds must be str, list, or None, got {type(confounds).__name__}"
        )

    def _fit_glm(
        self,
        X: "pd.DataFrame | np.ndarray | str | list",
        scale: bool = True,
        scale_value: float = 100.0,
        return_stats: list[str] | None = None,
        save: dict[str, str] | None = None,
        show_progress: bool = True,
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
            show_progress: Show progress bar during fitting.

        Returns:
            BrainCollection of betas, or dict with betas + requested stats.
        """
        # Validate return_stats
        valid_stats = {"t", "r2", "p", "se", "residual"}
        if return_stats is not None:
            invalid = set(return_stats) - valid_stats
            if invalid:
                raise ValueError(
                    f"Invalid return_stats: {invalid}. Valid options: {valid_stats}"
                )

        # Resolve X to per-subject list (or None if shared)
        X_list = self._resolve_X(X)

        # Progress bar setup
        iterator = range(len(self))
        if show_progress and tqdm is not None:
            iterator = tqdm.tqdm(iterator, desc="Fitting GLM", unit="subject")

        # Storage for results
        beta_data_list = []
        beta_metadata = []
        stat_data = {stat: [] for stat in (return_stats or [])}
        design_columns = None  # Will be set on first subject

        for i in iterator:
            # Load subject data
            bd = self._load_item(i)
            metadata_row = self._metadata.iloc[i]

            # Get subject-specific design matrix
            X_subj = X_list[i] if X_list else X

            # Load from file if needed
            if isinstance(X_subj, (str, Path)):
                X_subj = self._load_design_matrix(X_subj)

            # Convert array to DataFrame if needed
            if isinstance(X_subj, np.ndarray):
                X_subj = pd.DataFrame(
                    X_subj, columns=[f"col_{j}" for j in range(X_subj.shape[1])]
                )

            # Store design columns for result metadata
            if design_columns is None:
                design_columns = list(X_subj.columns)

            # Validate shapes match
            if X_subj.shape[0] != bd.shape[0]:
                raise ValueError(
                    f"Subject {i}: X has {X_subj.shape[0]} samples but data has "
                    f"{bd.shape[0]} samples. Shapes must match."
                )

            # Apply scaling if requested (scale=False since we scale here)
            if scale:
                bd = bd.scale(scale_value)

            # Fit GLM using BrainData.fit()
            bd.fit(model="glm", X=X_subj, scale=False)

            # Extract betas
            betas = bd[0].copy()
            betas.data = bd.glm_betas.data
            betas._design_columns = design_columns

            # Save if requested
            if save and "betas" in save:
                save_path = _resolve_save_path(save["betas"], metadata_row, i)
                betas.write(str(save_path))

            beta_data_list.append(betas)
            beta_metadata.append(metadata_row.to_dict())

            # Extract optional stats
            if return_stats:
                for stat in return_stats:
                    if stat == "t":
                        stat_bd = bd[0].copy()
                        stat_bd.data = bd.glm_t.data
                    elif stat == "p":
                        stat_bd = bd[0].copy()
                        stat_bd.data = bd.glm_p.data
                    elif stat == "se":
                        stat_bd = bd[0].copy()
                        stat_bd.data = bd.glm_se.data
                    elif stat == "r2":
                        stat_bd = bd[0].copy()
                        stat_bd.data = bd.glm_r2.data
                    elif stat == "residual":
                        stat_bd = bd[0].copy()
                        stat_bd.data = bd.glm_residual.data

                    stat_bd._design_columns = design_columns

                    if save and stat in save:
                        save_path = _resolve_save_path(save[stat], metadata_row, i)
                        stat_bd.write(str(save_path))

                    stat_data[stat].append(stat_bd)

        # Build result collection
        result_metadata = pd.DataFrame(beta_metadata)
        beta_collection = BrainCollection(
            beta_data_list, mask=self._mask, metadata=result_metadata, lazy=False
        )
        beta_collection._design_columns = design_columns

        # Return stats if requested
        if return_stats:
            result = {"betas": beta_collection}
            for stat, data_list in stat_data.items():
                stat_collection = BrainCollection(
                    data_list, mask=self._mask, metadata=result_metadata, lazy=False
                )
                stat_collection._design_columns = design_columns
                result[stat] = stat_collection
            return result

        return beta_collection

    def _load_design_matrix(self, path: str | Path) -> pd.DataFrame:
        """Load design matrix from a file path.

        Supports common formats: .csv, .tsv, .txt
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Design matrix file not found: {path}")

        sep = "\t" if path.suffix in [".tsv", ".txt"] else ","
        return pd.read_csv(path, sep=sep)

    def fit_ridge(
        self,
        X: "np.ndarray | str | list",
        alpha: float | str = 1.0,
        cv: int | None = 5,
        scale: bool = True,
        scale_value: float = 100.0,
        output: str = "scores",
        save: dict[str, str] | None = None,
        show_progress: bool = True,
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
            save: Dict mapping output type to path template, e.g.:
                {'weights': 'output/{subject}_weights.nii.gz',
                 'scores': 'output/{subject}_scores.nii.gz'}
                Supports {subject}, {session}, {idx}, and other metadata columns.
            show_progress: Show progress bar during fitting.
            **ridge_kwargs: Additional arguments passed to Ridge model
                (e.g., backend='torch', random_state=42).

        Returns:
            BrainCollection of scores or weights, or dict with both if output='both'.
            Each BrainData will have cv_results_ attribute when cv is used.

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
        # Validate output parameter
        valid_outputs = {"scores", "weights", "both"}
        if output not in valid_outputs:
            raise ValueError(
                f"Invalid output: '{output}'. Valid options: {valid_outputs}"
            )

        # CV is required for scores
        if output in ("scores", "both") and cv is None:
            raise ValueError(
                f"cv must be specified when output='{output}'. "
                "Set cv=5 (or another int) to compute cross-validated scores, "
                "or use output='weights' if you only need model weights."
            )

        # Resolve X to per-subject list (returns None if shared array)
        X_list = self._resolve_X(X)

        # Progress bar setup
        iterator = range(len(self))
        if show_progress and tqdm is not None:
            iterator = tqdm.tqdm(iterator, desc="Fitting Ridge", unit="subject")

        # Storage for results based on output type
        need_weights = output in ("weights", "both")
        need_scores = output in ("scores", "both")

        weight_data_list = [] if need_weights else None
        score_data_list = [] if need_scores else None
        result_metadata = []
        cv_results_list = []
        feature_names = None

        for i in iterator:
            # Load subject data
            bd = self._load_item(i)
            metadata_row = self._metadata.iloc[i]

            # Get subject features
            X_subj = X_list[i] if X_list else X

            # Load from file if needed
            if isinstance(X_subj, (str, Path)):
                X_subj = self._load_features(X_subj)

            # Extract feature names if available
            if feature_names is None and hasattr(X_subj, "columns"):
                feature_names = list(X_subj.columns)

            # Apply scaling if requested
            if scale:
                bd = bd.scale(scale_value)

            # Fit ridge
            bd.fit(model="ridge", X=X_subj, alpha=alpha, cv=cv, **ridge_kwargs)

            result_metadata.append(metadata_row.to_dict())

            # Store CV results if available
            cv_result = None
            if hasattr(bd, "cv_results_") and bd.cv_results_ is not None:
                cv_result = bd.cv_results_
                cv_results_list.append(cv_result)

            # Extract weights if needed
            if need_weights:
                weights_data = bd.ridge_weights.data
                weights = bd[0].copy()
                weights.data = weights_data
                if feature_names:
                    weights._feature_names = feature_names
                if cv_result:
                    weights.cv_results_ = cv_result

                if save and "weights" in save:
                    save_path = _resolve_save_path(save["weights"], metadata_row, i)
                    weights.write(str(save_path))

                weight_data_list.append(weights)

            # Extract scores if needed
            if need_scores:
                scores_data = bd.ridge_scores.data
                scores = bd[0].copy()
                scores.data = scores_data
                if cv_result:
                    scores.cv_results_ = cv_result

                if save and "scores" in save:
                    save_path = _resolve_save_path(save["scores"], metadata_row, i)
                    scores.write(str(save_path))

                score_data_list.append(scores)

            # Unload to free memory (only works for path-based collections)
            self.unload([i])

        # Build result collection(s)
        result_meta_df = pd.DataFrame(result_metadata)

        if output == "weights":
            weight_collection = BrainCollection(
                weight_data_list,
                mask=self.mask,
                metadata=result_meta_df,
            )
            if feature_names:
                weight_collection._feature_names = feature_names
            if cv_results_list:
                weight_collection.cv_results_ = cv_results_list
            return weight_collection

        elif output == "scores":
            score_collection = BrainCollection(
                score_data_list,
                mask=self.mask,
                metadata=result_meta_df,
            )
            if cv_results_list:
                score_collection.cv_results_ = cv_results_list
            return score_collection

        else:  # output == "both"
            weight_collection = BrainCollection(
                weight_data_list,
                mask=self.mask,
                metadata=result_meta_df,
            )
            if feature_names:
                weight_collection._feature_names = feature_names
            if cv_results_list:
                weight_collection.cv_results_ = cv_results_list

            score_collection = BrainCollection(
                score_data_list,
                mask=self.mask,
                metadata=result_meta_df,
            )
            if cv_results_list:
                score_collection.cv_results_ = cv_results_list

            return {"weights": weight_collection, "scores": score_collection}

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
        if X is None:
            raise ValueError("X must be provided")

        # Shared array - return None to signal no per-subject list
        if isinstance(X, np.ndarray):
            return None

        # Shared DataFrame - return None to signal shared
        if isinstance(X, pd.DataFrame):
            return None

        # Shared DesignMatrix (Polars-based, doesn't inherit from pd.DataFrame)
        from .design_matrix import DesignMatrix

        if isinstance(X, DesignMatrix):
            return None

        # Metadata column name - return list of file paths
        if isinstance(X, str):
            if X not in self._metadata.columns:
                raise KeyError(
                    f"Column '{X}' not found in metadata. "
                    f"Available: {list(self._metadata.columns)}"
                )
            return list(self._metadata[X])

        # Per-subject list - validate length
        if isinstance(X, list):
            if len(X) != len(self):
                raise ValueError(
                    f"X list length ({len(X)}) must match "
                    f"collection length ({len(self)})"
                )
            return X

        raise TypeError(
            f"X must be np.ndarray, DataFrame, DesignMatrix, str, or list, "
            f"got {type(X).__name__}"
        )

    def _load_features(self, path: str | Path) -> np.ndarray:
        """Load features from a file path.

        Supports common formats: .npy, .csv, .tsv, .txt
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.load(path)
        elif suffix in [".csv", ".tsv", ".txt"]:
            sep = "\t" if suffix in [".tsv", ".txt"] else ","
            return pd.read_csv(path, sep=sep).values
        else:
            raise ValueError(f"Unsupported feature file format: {suffix}")

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
        show_progress: bool = True,
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
            show_progress: Show progress bar during fitting.

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
        # Validate mutually exclusive modes
        if X is not None and y is not None:
            raise ValueError(
                "Cannot specify both X and y. Use X for timeseries prediction "
                "or y for MVPA decoding."
            )

        # Infer y from _condition_labels if available
        if y is None and X is None:
            if hasattr(self, "_condition_labels") and self._condition_labels:
                y = np.array(self._condition_labels)
            else:
                raise ValueError(
                    "Must provide X for timeseries prediction or y for MVPA. "
                    "If using fit_glm(by_run=True), y can be inferred from "
                    "_condition_labels."
                )

        # Infer groups from _run_labels if available
        if y is not None and groups is None:
            if hasattr(self, "_run_labels") and self._run_labels:
                groups = np.array(self._run_labels)

        # Progress bar setup
        iterator = range(len(self))
        if show_progress and tqdm is not None:
            desc = "Predicting (MVPA)" if y is not None else "Predicting (timeseries)"
            iterator = tqdm.tqdm(iterator, desc=desc, unit="subject")

        # Resolve per-subject features if X is provided
        X_list = None
        shared_X = None
        if X is not None:
            X_resolved = self._resolve_X(X)
            if X_resolved is None:
                # Shared features
                shared_X = X
            else:
                X_list = X_resolved

        # Storage for results
        result_data_list = []
        result_metadata = []

        for i in iterator:
            # Load subject data
            bd = self._load_item(i)
            metadata_row = self._metadata.iloc[i]

            if X is not None:
                # Timeseries prediction mode
                if X_list is not None:
                    subj_X = X_list[i]
                    if isinstance(subj_X, (str, Path)):
                        subj_X = self._load_features(subj_X)
                else:
                    subj_X = shared_X

                result = bd.predict(X=subj_X)
            else:
                # MVPA mode
                result = bd.predict(
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
                    show_progress=False,  # Disable per-subject progress
                )

            result_data_list.append(result)
            result_metadata.append(metadata_row.to_dict())

            # Unload to free memory
            self.unload([i])

        # Build result collection
        result_collection = BrainCollection(
            result_data_list,
            mask=self.mask,
            metadata=pd.DataFrame(result_metadata),
        )

        return result_collection

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
        # Validate that this collection has design columns
        if not hasattr(self, "_design_columns") or self._design_columns is None:
            raise RuntimeError(
                "No design columns found. This method requires a BrainCollection "
                "created by fit_glm() which stores the task regressor names."
            )

        design_columns = self._design_columns

        # Handle dict of contrasts
        if isinstance(contrasts, dict):
            results = {}
            for name, contrast_spec in contrasts.items():
                results[name] = self._compute_single_contrast(
                    contrast_spec, design_columns
                )
            return results

        # Single contrast
        return self._compute_single_contrast(contrasts, design_columns)

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
        # Parse contrast to numeric vector
        if isinstance(contrast, str):
            contrast_vector = self._parse_contrast_string(contrast, design_columns)
        else:
            contrast_vector = np.asarray(contrast)

        # Validate contrast vector length
        n_regressors = len(design_columns)
        if len(contrast_vector) != n_regressors:
            raise ValueError(
                f"Contrast vector length ({len(contrast_vector)}) must match "
                f"number of regressors ({n_regressors}). "
                f"Regressors: {design_columns}"
            )

        # Compute contrast for each subject
        contrast_data_list = []
        for i in range(len(self)):
            bd = self._load_item(i)

            # Compute weighted sum of betas
            # bd.data has shape (n_regressors, n_voxels)
            contrast_values = np.zeros(bd.shape[1])
            for j, weight in enumerate(contrast_vector):
                if weight != 0:
                    contrast_values += weight * bd.data[j, :]

            # Create BrainData with contrast values
            contrast_bd = bd[0].copy()
            contrast_bd.data = contrast_values

            contrast_data_list.append(contrast_bd)

        # Build result collection
        return BrainCollection(
            contrast_data_list,
            mask=self.mask,
            metadata=self._metadata.copy(),
        )

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
        import re

        # Initialize contrast vector
        contrast_vector = np.zeros(len(design_columns))

        # Split by + and - (keeping the operators)
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
                    coef = 1.0
                    var_name = token

                # Find column index
                if var_name in design_columns:
                    idx = design_columns.index(var_name)
                    contrast_vector[idx] = sign * coef
                else:
                    raise ValueError(
                        f"Column '{var_name}' not found in design columns. "
                        f"Available: {design_columns}"
                    )

        return contrast_vector

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
        # Resolve feature name to index
        if isinstance(feature, str):
            if not hasattr(self, "_feature_names") or self._feature_names is None:
                raise RuntimeError(
                    "Cannot select feature by name: _feature_names not set. "
                    "Use integer index or pass features with column names to fit_ridge()."
                )
            if feature not in self._feature_names:
                raise KeyError(
                    f"Feature '{feature}' not found. Available: {self._feature_names}"
                )
            feature_idx = self._feature_names.index(feature)
        else:
            feature_idx = feature

        # Extract feature weights for each subject
        feature_data_list = []
        for i in range(len(self)):
            bd = self._load_item(i)

            # Validate index
            if feature_idx < 0 or feature_idx >= bd.shape[0]:
                raise IndexError(
                    f"Feature index {feature_idx} out of range for subject {i} "
                    f"with {bd.shape[0]} features."
                )

            # Extract single feature's weights
            feature_values = bd.data[feature_idx, :]

            # Create BrainData with feature weights
            feature_bd = bd[0].copy()
            feature_bd.data = feature_values

            feature_data_list.append(feature_bd)

        # Build result collection
        return BrainCollection(
            feature_data_list,
            mask=self.mask,
            metadata=self._metadata.copy(),
        )


# =============================================================================
# Pipeline Classes for BrainCollection
# =============================================================================


class BrainCollectionPipeline:
    """Pipeline for BrainCollection with multi-subject CV support.

    Wraps BrainCollection to provide fluent pipeline API with LOSO
    and run-based cross-validation.

    This class enables method chaining for preprocessing and prediction
    with proper cross-validation semantics for multi-subject neuroimaging
    analyses.

    Attributes:
        n_subjects: Number of subjects/images in the collection.
        cv: The cross-validation scheme configuration.
        n_steps: Number of transform steps in the pipeline.

    Examples:
        >>> # Leave-one-subject-out with preprocessing
        >>> result = (bc
        ...     .cv(scheme='loso')
        ...     .normalize()
        ...     .reduce(n_components=50)
        ...     .predict(labels, algorithm='svm'))
        >>> print(f"Mean accuracy: {result.mean_score:.2%}")
    """

    def __init__(
        self, brain_collection: "BrainCollection", cv=None, groups: np.ndarray = None
    ):
        """Initialize pipeline wrapper.

        Args:
            brain_collection: BrainCollection to wrap.
            cv: CVScheme configuration.
            groups: Group labels for CV splits.
        """
        self._bc = brain_collection
        self._cv = cv
        self._groups = groups
        self._steps = []

    @property
    def n_subjects(self) -> int:
        """Number of subjects/images."""
        return self._bc.n_images

    @property
    def cv(self):
        """Cross-validation scheme."""
        return self._cv

    @property
    def n_steps(self) -> int:
        """Number of transform steps."""
        return len(self._steps)

    def _add_step(self, step) -> "BrainCollectionPipeline":
        """Add step and return new pipeline (immutable).

        Args:
            step: Transform step to add.

        Returns:
            New pipeline with step added.
        """
        from copy import copy

        new = copy(self)
        new._steps = self._steps + [step]
        return new

    def normalize(self, method: str = "zscore", **kwargs) -> "BrainCollectionPipeline":
        """Add normalization step.

        Args:
            method: Normalization method ('zscore', 'minmax').
            **kwargs: Additional arguments for NormalizeStep.

        Returns:
            New pipeline with normalization step added.
        """
        from nltools.pipelines.steps import NormalizeStep

        return self._add_step(NormalizeStep(method=method, **kwargs))

    def reduce(
        self, method: str = "pca", n_components: int = None, **kwargs
    ) -> "BrainCollectionPipeline":
        """Add dimensionality reduction step.

        Args:
            method: Reduction method ('pca', 'ica').
            n_components: Number of components to keep.
            **kwargs: Additional arguments for ReduceStep.

        Returns:
            New pipeline with reduction step added.
        """
        from nltools.pipelines.steps import ReduceStep

        return self._add_step(
            ReduceStep(method=method, n_components=n_components, **kwargs)
        )

    def pipe(self, transformer) -> "BrainCollectionPipeline":
        """Add custom sklearn transformer.

        Args:
            transformer: sklearn-compatible transformer with fit/transform interface.

        Returns:
            New pipeline with custom step added.
        """
        from nltools.pipelines.steps import PipeStep

        return self._add_step(PipeStep(transformer=transformer))

    def predict(
        self, y, algorithm: str = "ridge", **kwargs
    ) -> "BrainCollectionCVResult":
        """Execute pipeline with CV and return prediction results.

        Parameters
        ----------
        y : array-like
            Target variable. For LOSO, shape should be (n_subjects,).
        algorithm : str
            Prediction algorithm ('ridge', 'svm', 'logistic', etc.)
        **kwargs
            Passed to model constructor.

        Returns
        -------
        BrainCollectionCVResult
            Cross-validation results with scores and predictions.

        Raises
        ------
        ValueError
            If no CV context is set or if non-LOSO CV is used without groups.
        """

        if self._cv is None:
            raise ValueError("predict() requires CV context")

        y = np.asarray(y)

        # Get data as list of numpy arrays
        brain_data_list = self._bc.to_list()
        subject_data = [bd.data for bd in brain_data_list]

        if self._cv.is_loso:
            return self._execute_loso(subject_data, y, algorithm, kwargs)
        else:
            return self._execute_pooled_cv(subject_data, y, algorithm, kwargs)

    def _execute_loso(
        self, subject_data: list, y: np.ndarray, algorithm: str, model_kwargs: dict
    ) -> "BrainCollectionCVResult":
        """Execute leave-one-subject-out CV.

        Args:
            subject_data: List of arrays, one per subject.
            y: Target labels.
            algorithm: Prediction algorithm name.
            model_kwargs: Kwargs passed to model constructor.

        Returns:
            Cross-validation results.
        """
        from nltools.pipelines.base import FittedStack

        results = []
        n_subjects = len(subject_data)

        for held_out_idx in range(n_subjects):
            # Split subjects
            train_subjects = [
                subject_data[i] for i in range(n_subjects) if i != held_out_idx
            ]
            test_subject = subject_data[held_out_idx]

            fitted_stack = FittedStack()

            # Pool training data
            train_pooled = np.vstack(train_subjects)
            test_data = (
                test_subject if test_subject.ndim == 2 else test_subject[np.newaxis, :]
            )

            # Apply transforms
            for step in self._steps:
                fitted = step.fit(train_pooled)
                fitted_stack.append(fitted)
                train_pooled = fitted.transform(train_pooled)
                test_data = fitted.transform(test_data)

            # Handle y based on shape
            if y.shape[0] == n_subjects:
                # One label per subject
                train_y = np.concatenate(
                    [
                        np.full(subject_data[i].shape[0], y[i])
                        for i in range(n_subjects)
                        if i != held_out_idx
                    ]
                )
                test_y = np.full(test_data.shape[0], y[held_out_idx])
            else:
                # Labels match observations - need proper indexing
                obs_per_subj = [s.shape[0] for s in subject_data]
                cumsum = np.cumsum([0] + obs_per_subj)
                train_mask = np.ones(sum(obs_per_subj), dtype=bool)
                train_mask[cumsum[held_out_idx] : cumsum[held_out_idx + 1]] = False
                train_y = y[train_mask]
                test_y = y[cumsum[held_out_idx] : cumsum[held_out_idx + 1]]

            # Get model
            model = self._get_model(algorithm, model_kwargs)
            model.fit(train_pooled, train_y)

            predictions = model.predict(test_data)
            score = model.score(test_data, test_y)

            train_idx = np.arange(len(train_y))
            test_idx = np.arange(len(train_y), len(train_y) + len(test_y))

            results.append(
                {
                    "score": score,
                    "predictions": predictions,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "fitted_stack": fitted_stack,
                    "held_out_subject": held_out_idx,
                }
            )

        return BrainCollectionCVResult(results, self)

    def _execute_pooled_cv(
        self, subject_data: list, y: np.ndarray, algorithm: str, model_kwargs: dict
    ) -> "BrainCollectionCVResult":
        """Execute CV on pooled data.

        Args:
            subject_data: List of arrays, one per subject.
            y: Target labels.
            algorithm: Prediction algorithm name.
            model_kwargs: Kwargs passed to model constructor.

        Returns:
            Cross-validation results.

        Raises:
            ValueError: If groups parameter is not set.
        """
        from nltools.pipelines.base import FittedStack

        # Pool all data
        pooled_data = np.vstack(subject_data)

        if self._groups is None:
            raise ValueError("Non-LOSO CV requires groups parameter")

        results = []

        for train_idx, test_idx in self._cv.split(pooled_data, groups=self._groups):
            fitted_stack = FittedStack()

            train_data = pooled_data[train_idx]
            test_data = pooled_data[test_idx]
            train_y = y[train_idx]
            test_y = y[test_idx]

            # Apply transforms
            for step in self._steps:
                fitted = step.fit(train_data)
                fitted_stack.append(fitted)
                train_data = fitted.transform(train_data)
                test_data = fitted.transform(test_data)

            # Fit and evaluate
            model = self._get_model(algorithm, model_kwargs)
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

        return BrainCollectionCVResult(results, self)

    def _get_model(self, algorithm: str, kwargs: dict):
        """Get sklearn model instance.

        Args:
            algorithm: Algorithm name.
            kwargs: Model constructor arguments.

        Returns:
            Sklearn estimator instance.

        Raises:
            ValueError: If algorithm is unknown.
        """
        if algorithm == "ridge":
            from sklearn.linear_model import Ridge

            return Ridge(**kwargs)
        elif algorithm == "lasso":
            from sklearn.linear_model import Lasso

            return Lasso(**kwargs)
        elif algorithm == "svm":
            from sklearn.svm import SVC

            return SVC(**kwargs)
        elif algorithm == "svr":
            from sklearn.svm import SVR

            return SVR(**kwargs)
        elif algorithm == "logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(**kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def __repr__(self):
        """Return string representation."""
        return f"BrainCollectionPipeline(n_subjects={self.n_subjects}, n_steps={self.n_steps})"


class BrainCollectionCVResult:
    """Cross-validation results for BrainCollection pipelines.

    Contains fold-by-fold results from cross-validated prediction,
    with convenience properties for accessing scores and predictions.

    Attributes:
        fold_results: List of dictionaries with per-fold results.
        pipeline: The pipeline that generated these results.
        scores: Per-fold prediction scores.
        mean_score: Mean score across all folds.
        std_score: Standard deviation of scores.
        n_folds: Number of CV folds.
    """

    def __init__(self, fold_results: list, pipeline: BrainCollectionPipeline):
        """Initialize CV results.

        Args:
            fold_results: List of fold result dictionaries.
            pipeline: The pipeline that generated these results.
        """
        self.fold_results = fold_results
        self.pipeline = pipeline

    @property
    def scores(self) -> np.ndarray:
        """Per-fold scores."""
        return np.array([f["score"] for f in self.fold_results])

    @property
    def mean_score(self) -> float:
        """Mean score across folds."""
        return float(self.scores.mean())

    @property
    def std_score(self) -> float:
        """Standard deviation of scores."""
        return float(self.scores.std())

    @property
    def n_folds(self) -> int:
        """Number of CV folds."""
        return len(self.fold_results)

    def __repr__(self):
        """Return string representation."""
        return f"BrainCollectionCVResult(n_folds={self.n_folds}, mean_score={self.mean_score:.4f})"


# =============================================================================
# FittedBrainCollection: Wrapper for chaining pool() after fit()
# =============================================================================


class FittedBrainCollection:
    """Wrapper for fitted BrainCollection enabling pool() chaining.

    This class wraps the results of bc.fit() and provides the .pool()
    method for aggregating across subjects.

    The execution model:
    - fit() executes immediately (eager)
    - pool() aggregates the fitted parameters
    - pool() returns PooledData for second-level analysis

    Parameters
    ----------
    brain_collection : BrainCollection
        The original collection that was fitted.
    fitted_results : BrainCollection | dict
        The fitted results. Can be:
        - BrainCollection: Betas or scores
        - dict: {'betas': BrainCollection, 't': BrainCollection, ...}
    model : str
        The model type that was fitted ('glm' or 'ridge').
    condition_names : list of str, optional
        Names of conditions/regressors from the design matrix.

    Examples
    --------
    >>> fitted = bc.fit(model='glm', X=dm)
    >>> pool = fitted.pool(param='beta')
    >>> result = pool.fit(model='ttest', contrast='A-B')
    """

    def __init__(
        self,
        brain_collection: "BrainCollection",
        fitted_results: "BrainCollection | dict[str, BrainCollection]",
        model: str,
        condition_names: list[str] | None = None,
    ):
        self._bc = brain_collection
        self._fitted = fitted_results
        self._model = model
        self._condition_names = condition_names

    @property
    def n_subjects(self) -> int:
        """Number of subjects in the fitted collection."""
        if isinstance(self._fitted, dict):
            # Get from first value in dict
            first = next(iter(self._fitted.values()))
            return len(first)
        return len(self._fitted)

    @property
    def results(self) -> "BrainCollection | dict[str, BrainCollection]":
        """Access the fitted results directly.

        Returns the underlying BrainCollection or dict of BrainCollections.
        Use this for backward compatibility or when pool() is not needed.
        """
        return self._fitted

    @property
    def betas(self) -> "BrainCollection":
        """Convenience accessor for beta coefficients.

        Returns
        -------
        BrainCollection
            Beta coefficients from GLM fit.

        Raises
        ------
        ValueError
            If model is not GLM or betas not available.
        """
        if isinstance(self._fitted, dict):
            if "betas" in self._fitted:
                return self._fitted["betas"]
            raise ValueError("No 'betas' key in fitted results dict")
        if self._model == "glm":
            return self._fitted
        raise ValueError(f"'betas' not available for model='{self._model}'")

    def pool(
        self,
        param: str = "beta",
        contrast: str | None = None,
        save: str | None = None,
        save_fitted: bool = False,
    ):
        """Pool fitted parameters across subjects.

        Aggregates per-subject fitted results for group-level analysis.
        Returns a PooledData object that can be passed to second-level
        statistical tests.

        Parameters
        ----------
        param : str, default='beta'
            Parameter to pool. Options depend on model:
            - GLM: 'beta', 't', 'r2', 'p', 'se', 'residual'
            - Ridge: 'scores', 'weights'
        contrast : str, optional
            Apply contrast before pooling. Format: 'A-B' or 'A+B'.
            Requires condition_names to be available.
        save : str, optional
            Path template to save per-subject results before pooling.
            Supports {subject}, {idx} placeholders.
        save_fitted : bool, default=False
            If True, save full fitted state for later repool().

        Returns
        -------
        PooledData
            Pooled data ready for second-level analysis.

        Examples
        --------
        >>> pool = bc.fit(model='glm', X=designs).pool(param='beta')
        >>> result = pool.fit(model='ttest', contrast='face-house')

        >>> # Pool t-statistics instead of betas
        >>> pool = bc.fit(model='glm', X=dm, return_stats=['t']).pool(param='t')
        """
        from nltools.pipelines.pool import PooledData

        # Determine what data to pool
        if isinstance(self._fitted, dict):
            # Results include multiple stats
            param_key = param if param != "beta" else "betas"
            if param_key not in self._fitted:
                available = list(self._fitted.keys())
                raise ValueError(
                    f"Parameter '{param}' not found. Available: {available}"
                )
            data_to_pool = self._fitted[param_key]
        else:
            # Single BrainCollection result
            if param not in ("beta", "betas", "scores", "weights"):
                raise ValueError(
                    f"Parameter '{param}' not available. For GLM stats, "
                    "use return_stats=['t', 'p', ...] in fit()."
                )
            data_to_pool = self._fitted

        # Extract data as array: (n_subjects, n_conditions, n_voxels)
        # Each item in data_to_pool is a BrainData with shape (n_conditions, n_voxels)
        pooled_list = []
        for i in range(len(data_to_pool)):
            bd = data_to_pool[i]
            pooled_list.append(bd.data)

        pooled_array = np.stack(pooled_list)

        # Apply contrast if specified
        if contrast is not None:
            pooled_array = self._apply_contrast(pooled_array, contrast)

        # Save per-subject if requested
        if save:
            self._save_per_subject(save)

        # Get subject IDs from metadata if available
        subject_ids = None
        if self._bc.metadata is not None and "subject" in self._bc.metadata.columns:
            subject_ids = list(self._bc.metadata["subject"])

        # Get condition names from fitted results if available
        condition_names = self._condition_names
        if condition_names is None and hasattr(data_to_pool, "_design_columns"):
            condition_names = data_to_pool._design_columns

        return PooledData(
            data=pooled_array,
            param=param,
            condition_names=condition_names,
            subject_ids=subject_ids,
            mask=self._bc.mask,
            fitted_state=self._fitted if save_fitted else None,
            save_path=save,
        )

    def _apply_contrast(self, data: np.ndarray, contrast: str) -> np.ndarray:
        """Apply contrast weights to pooled data.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_subjects, n_conditions, n_voxels).
        contrast : str
            Contrast specification like 'A-B' or 'A+B'.

        Returns
        -------
        np.ndarray
            Shape (n_subjects, n_voxels) after contrast.
        """
        if self._condition_names is None:
            raise ValueError(
                "Cannot apply contrast: condition_names not available. "
                "Ensure fit() received a design matrix with column names."
            )

        # Parse contrast string
        contrast_weights = self._parse_contrast(contrast)

        # Apply contrast: weighted sum across conditions
        result = np.zeros((data.shape[0], data.shape[2]))  # (n_subjects, n_voxels)
        for i, (cond, weight) in enumerate(contrast_weights.items()):
            if cond not in self._condition_names:
                raise ValueError(
                    f"Condition '{cond}' not in conditions: {self._condition_names}"
                )
            idx = self._condition_names.index(cond)
            result += weight * data[:, idx, :]

        return result

    def _parse_contrast(self, contrast: str) -> dict[str, float]:
        """Parse contrast string into weights dict.

        Examples:
        - 'A-B' -> {'A': 1.0, 'B': -1.0}
        - 'A+B' -> {'A': 1.0, 'B': 1.0}
        - '2*A-B' -> {'A': 2.0, 'B': -1.0}
        """
        import re

        weights = {}
        # Split by + or - keeping the delimiter
        parts = re.split(r"(\+|-)", contrast)

        sign = 1.0
        for part in parts:
            part = part.strip()
            if part == "+":
                sign = 1.0
            elif part == "-":
                sign = -1.0
            elif part:
                # Check for coefficient
                match = re.match(r"(\d*\.?\d*)\*?(.+)", part)
                if match:
                    coef_str, name = match.groups()
                    coef = float(coef_str) if coef_str else 1.0
                    weights[name.strip()] = sign * coef
                else:
                    weights[part] = sign

        return weights

    def _save_per_subject(self, save_dir: str) -> None:
        """Save each subject's fitted results."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        data_to_save = (
            self._fitted
            if not isinstance(self._fitted, dict)
            else self._fitted.get("betas", next(iter(self._fitted.values())))
        )

        for i in range(len(data_to_save)):
            bd = data_to_save[i]
            subj_path = save_path / f"subj{i + 1:02d}.nii.gz"
            bd.write(str(subj_path))

    def __repr__(self) -> str:
        if isinstance(self._fitted, dict):
            keys = list(self._fitted.keys())
            return (
                f"FittedBrainCollection(n_subjects={self.n_subjects}, "
                f"model='{self._model}', outputs={keys})"
            )
        return (
            f"FittedBrainCollection(n_subjects={self.n_subjects}, "
            f"model='{self._model}')"
        )

    # Delegate common operations to underlying results for backward compatibility
    def __len__(self) -> int:
        return self.n_subjects

    def __getitem__(self, key):
        """Allow indexing into fitted results."""
        if isinstance(self._fitted, dict):
            if isinstance(key, str):
                return self._fitted[key]
            # Numeric index - apply to all values
            return {k: v[key] for k, v in self._fitted.items()}
        return self._fitted[key]

    def __iter__(self):
        """Iterate over fitted results."""
        if isinstance(self._fitted, dict):
            return iter(self._fitted)
        return iter(self._fitted)
