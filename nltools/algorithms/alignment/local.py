"""LocalAlignment: Neighborhood-based functional alignment.

Implements searchlight and piecewise schemes from Bazeille et al. 2021.
Uses center-only aggregation to preserve orthogonality of local transforms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from collections.abc import Iterator

import numpy as np
from scipy.linalg import orthogonal_procrustes

from nltools.algorithms.backends import Backend

if TYPE_CHECKING:
    import nibabel as nib

    from nltools.data.braindata.neighborhoods import SphereNeighborhoods

logger = logging.getLogger(__name__)


def _orthogonal_procrustes_backend(
    A: np.ndarray, B: np.ndarray, backend: Backend
) -> tuple[np.ndarray, float]:
    """GPU-compatible orthogonal Procrustes using Backend.svd().

    Finds the orthogonal matrix R that minimizes ||A - B @ R||_F.

    This is equivalent to scipy.linalg.orthogonal_procrustes but uses
    the Backend abstraction for GPU acceleration.

    Args:
        A: Target matrix, shape (n, m)
        B: Matrix to transform, shape (n, m)
        backend: Backend instance for computations

    Returns:
        R: Orthogonal matrix, shape (m, m)
        scale: Sum of singular values (not used, kept for compatibility)
    """
    if backend.name == "numpy":
        # Use scipy for numpy backend (more efficient)
        return orthogonal_procrustes(A, B)

    # GPU path: Compute SVD of B.T @ A
    # The solution is R = V @ U.T where U, s, V.T = svd(B.T @ A)
    A_device = backend.to_device(A.astype(np.float32))
    B_device = backend.to_device(B.astype(np.float32))

    # M = B.T @ A
    M = backend.matmul(B_device.T if hasattr(B_device, "T") else B_device.t(), A_device)

    # SVD: M = U @ diag(s) @ Vt
    U, s, Vt = backend.svd(M, full_matrices=False)

    # R = V @ U.T = Vt.T @ U.T
    if backend.name.startswith("torch"):
        import torch

        R = torch.matmul(
            Vt.T if hasattr(Vt, "T") else Vt.t(), U.T if hasattr(U, "T") else U.t()
        )
        scale = s.sum().item()
        return backend.to_numpy(R).astype(np.float64), scale
    R = Vt.T @ U.T
    scale = s.sum()
    return R, scale


@dataclass
class PiecewiseNeighborhoods:
    """Neighborhoods from a brain parcellation for piecewise alignment.

    Unlike searchlight (overlapping spheres), piecewise uses non-overlapping
    parcels where each voxel belongs to exactly one region.

    Attributes:
        parcel_to_voxels: Dict mapping parcel_id → array of voxel indices
        voxel_to_parcel: Array mapping voxel_idx → parcel_id
        n_voxels: Total number of voxels
        n_parcels: Number of parcels (excluding background)
    """

    parcel_to_voxels: dict[int, np.ndarray]
    voxel_to_parcel: np.ndarray
    n_voxels: int
    n_parcels: int

    def iter_neighborhoods(
        self, progress_bar: bool = False
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over all parcels.

        Yields:
            Tuple of (parcel_id, voxel_indices) for each parcel
        """
        iterator = self.parcel_to_voxels.items()

        if progress_bar:
            from tqdm import tqdm

            iterator = tqdm(list(iterator), desc="Piecewise", unit="parcels")

        yield from iterator

    def __repr__(self) -> str:
        return (
            f"PiecewiseNeighborhoods(n_voxels={self.n_voxels}, "
            f"n_parcels={self.n_parcels})"
        )


def _compute_piecewise_neighborhoods(
    parcellation: nib.Nifti1Image,
    mask: nib.Nifti1Image,
) -> PiecewiseNeighborhoods:
    """Compute piecewise neighborhoods from a parcellation image.

    Args:
        parcellation: NIfTI image with integer labels for each parcel.
            Background/unlabeled voxels should be 0.
        mask: Brain mask defining the voxel space.

    Returns:
        PiecewiseNeighborhoods with parcel-to-voxel mappings.

    Raises:
        ValueError: If parcellation and mask have incompatible shapes.
    """
    parc_data = parcellation.get_fdata().astype(int)
    mask_data = mask.get_fdata().astype(bool)

    if parc_data.shape != mask_data.shape:
        raise ValueError(
            f"Parcellation shape {parc_data.shape} != mask shape {mask_data.shape}"
        )

    # Get masked voxel coordinates
    mask_indices = np.where(mask_data.ravel())[0]
    n_voxels = len(mask_indices)

    # Map 3D coordinates to flat masked indices
    # For each masked voxel, get its parcel label
    parc_flat = parc_data.ravel()
    parcel_labels = parc_flat[mask_indices]

    # Build parcel → voxel mapping (excluding background label 0)
    unique_parcels = np.unique(parcel_labels)
    unique_parcels = unique_parcels[unique_parcels > 0]  # Exclude background

    parcel_to_voxels: dict[int, np.ndarray] = {}
    for parcel_id in unique_parcels:
        # Find which masked voxel indices belong to this parcel
        voxel_indices = np.where(parcel_labels == parcel_id)[0]
        if len(voxel_indices) > 0:
            parcel_to_voxels[int(parcel_id)] = voxel_indices

    # Build voxel → parcel mapping
    voxel_to_parcel = parcel_labels.copy()

    return PiecewiseNeighborhoods(
        parcel_to_voxels=parcel_to_voxels,
        voxel_to_parcel=voxel_to_parcel,
        n_voxels=n_voxels,
        n_parcels=len(parcel_to_voxels),
    )


def _fit_one_neighborhood(
    region_id: int,
    voxel_indices: np.ndarray,
    data: list[np.ndarray],
    method: str,
    n_iter: int,
    n_features: int | None,
    backend: Backend | None = None,
) -> tuple[int, list[np.ndarray], np.ndarray]:
    """Fit alignment for a single neighborhood.

    Helper function for parallel processing.

    Args:
        region_id: ID of the neighborhood/parcel
        voxel_indices: Voxel indices in this neighborhood
        data: Full subject data arrays
        method: Alignment method ('procrustes', 'srm', 'hyperalignment')
        n_iter: Number of iterations
        n_features: Number of features for SRM (None for auto)
        backend: Backend instance for GPU acceleration (None for numpy)

    Returns:
        Tuple of (region_id, transforms, template)
    """
    n_subjects = len(data)

    # Extract local data for all subjects
    local_data = [subj[voxel_indices, :] for subj in data]
    n_local_voxels = len(voxel_indices)

    # Handle unequal sample counts by zero-padding to max length
    sample_counts = [ld.shape[1] for ld in local_data]
    max_samples = max(sample_counts)
    if not all(s == max_samples for s in sample_counts):
        local_data = [
            np.hstack([ld, np.zeros((ld.shape[0], max_samples - ld.shape[1]))])
            if ld.shape[1] < max_samples
            else ld
            for ld in local_data
        ]
    n_samples = max_samples

    # Handle degenerate cases
    if n_local_voxels < 2:
        transforms = [np.eye(n_local_voxels) for _ in range(n_subjects)]
        template = np.mean(np.array(local_data), axis=0)
        return region_id, transforms, template

    # Fit based on method
    if method == "procrustes":
        transforms, template = _fit_local_procrustes(
            local_data, n_iter=n_iter, backend=backend
        )
    elif method == "srm":
        from .srm import SRM

        feat = n_features if n_features is not None else min(n_local_voxels, n_samples)
        srm = SRM(n_iter=n_iter, features=feat)
        srm.fit(local_data, parallel=None)
        transforms, template = srm.w_, srm.s_
    elif method == "hyperalignment":
        from .hyperalignment import HyperAlignment

        ha = HyperAlignment(n_iter=n_iter)
        ha.fit(local_data, parallel=None)
        transforms, template = ha.w_, ha.s_
    else:
        raise ValueError(f"Unknown method: {method}")

    return region_id, transforms, template


def _fit_local_procrustes(
    data: list[np.ndarray],
    n_iter: int = 3,
    backend: Backend | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Fit multi-subject Procrustes alignment on local data.

    Iteratively refines a template by aligning all subjects and averaging.
    This is a simplified version of HyperAlignment for local neighborhoods.

    Args:
        data: List of arrays, each shape (n_local_voxels, n_samples).
        n_iter: Number of refinement iterations.
        backend: Backend instance for GPU acceleration (None for numpy/scipy).

    Returns:
        transforms: List of orthogonal transforms, each (n_local_voxels, n_local_voxels).
        template: Mean template in aligned space, shape (n_local_voxels, n_samples).
    """
    n_subjects = len(data)
    n_voxels, n_samples = data[0].shape

    # Use numpy backend if none provided
    if backend is None:
        backend = Backend("numpy")

    # Center and normalize each subject
    centered = []
    for x in data:
        c = x - x.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(c)
        if norm > 0:
            c = c / norm
        centered.append(c)

    # Initialize template as mean of centered data
    template = np.mean(centered, axis=0)

    # Iteratively refine
    transforms = [np.eye(n_voxels) for _ in range(n_subjects)]
    for _ in range(n_iter):
        aligned = []
        for i, x in enumerate(centered):
            # Solve Procrustes: min ||template - x @ R.T||_F s.t. R orthogonal
            # scipy's orthogonal_procrustes: R, scale = argmin ||A - B @ R||_F
            R, _ = _orthogonal_procrustes_backend(template.T, x.T, backend)
            # R transforms x.T to align with template.T, so aligned = x.T @ R
            # But we want row-wise (voxels), so: aligned = R.T @ x
            transforms[i] = R.T
            aligned.append(R.T @ x)
        # Update template
        template = np.mean(aligned, axis=0)

    return transforms, template


@dataclass
class LocalAlignment:
    """Local (neighborhood-based) functional alignment across subjects.

    Learns alignment transforms within local neighborhoods (searchlight spheres
    or parcels) and applies center-only aggregation to preserve orthogonality.

    Args:
    scheme : str, default='searchlight'
        Spatial scheme: 'searchlight' (overlapping spheres) or 'piecewise'
        (non-overlapping parcels).
    method : str, default='procrustes'
        Alignment method: 'procrustes', 'srm', or 'hyperalignment'.
    radius_mm : float, default=10.0
        Sphere radius in millimeters for searchlight scheme.
    parcellation : Nifti1Image, optional
        Parcellation image for piecewise scheme. Required if scheme='piecewise'.
    n_features : int, optional
        Number of features for SRM. None uses full Procrustes (preserves dims).
    n_iter : int, default=3
        Number of iterations for alignment refinement.
    aggregation : str, default='center'
        Aggregation method: 'center' (center-only, preserves orthogonality).
    parallel : str, optional
        Parallelization: 'cpu', 'gpu', or None.
        - None: Single-threaded numpy
        - 'cpu': CPU parallelization with joblib
        - 'gpu': GPU acceleration via PyTorch (falls back to CPU if unavailable)
    n_jobs : int, default=-1
        Number of jobs for CPU parallelization.

    Attributes:
    transforms_ : Dict[int, List[np.ndarray]]
        Per-neighborhood transforms. Keys are center voxel indices,
        values are lists of transform matrices (one per subject).
    template_ : Dict[int, np.ndarray]
        Per-neighborhood templates used for alignment.
    neighborhoods_ : SphereNeighborhoods or Dict
        Computed neighborhoods (searchlight or piecewise).
    n_voxels_ : int
        Total number of voxels in the mask.
    mask_ : Nifti1Image
        Brain mask used for fitting.

    Examples:
    >>> import numpy as np
    >>> from nltools.algorithms.alignment import LocalAlignment
    >>> # Create synthetic multi-subject data (voxels, samples)
    >>> data = [np.random.randn(1000, 100) for _ in range(5)]
    >>> la = LocalAlignment(scheme='searchlight', method='procrustes', radius_mm=10.0)
    >>> la.fit(data, mask)
    >>> aligned = la.transform(data)

    Notes:
    Based on Bazeille et al. 2021 "An empirical evaluation of functional
    alignment using inter-subject decoding". Center-only aggregation is
    used to preserve local orthogonality of transforms.
    """

    # Configuration
    scheme: str = "searchlight"
    method: str = "procrustes"
    radius_mm: float = 10.0
    parcellation: Any | None = None  # Nifti1Image
    n_features: int | None = None
    n_iter: int = 3
    aggregation: str = "center"
    parallel: str | None = "cpu"
    n_jobs: int = -1

    # Batching parameters (Phase 2)
    n_neighborhoods_batch: int | None = None  # None = auto-calculate
    max_memory_gb: float = 4.0  # Memory budget for auto batch sizing

    # Fitted state (set by fit())
    transforms_: dict[int, list[np.ndarray]] | None = field(default=None, repr=False)
    template_: dict[int, np.ndarray] | None = field(default=None, repr=False)
    neighborhoods_: SphereNeighborhoods | dict[int, np.ndarray] | None = field(
        default=None, repr=False
    )
    n_voxels_: int | None = field(default=None, repr=False)
    mask_: Any | None = field(default=None, repr=False)  # Nifti1Image
    backend_: Backend | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        if self.scheme not in ("searchlight", "piecewise"):
            raise ValueError(f"Unknown scheme: {self.scheme}")
        if self.method not in ("procrustes", "srm", "hyperalignment"):
            raise ValueError(f"Unknown method: {self.method}")
        if self.aggregation not in ("center", "all"):
            raise ValueError(
                f"Unknown aggregation: {self.aggregation}. "
                "Supported: 'center' (searchlight), 'all' (piecewise)."
            )
        # Validate scheme/aggregation compatibility
        if self.scheme == "piecewise" and self.aggregation == "center":
            # Auto-switch to 'all' for piecewise (center-only doesn't make sense)
            object.__setattr__(self, "aggregation", "all")
        if self.scheme == "piecewise" and self.parcellation is None:
            raise ValueError("parcellation is required for piecewise scheme")

    def _init_backend(self) -> Backend:
        """Initialize backend based on parallel setting.

        Returns:
            Backend instance configured for the requested execution mode.
        """
        if self.parallel is None or self.parallel == "cpu":
            return Backend("numpy")
        if self.parallel == "gpu":
            # Try GPU, gracefully fall back to CPU if unavailable
            try:
                backend = Backend("torch")
                logger.info(f"Using backend: {backend.name}")
                return backend
            except ImportError:
                logger.warning("PyTorch not available, falling back to numpy backend")
                return Backend("numpy")
        else:
            # Unknown parallel value, use numpy
            return Backend("numpy")

    def _auto_batch_size(
        self, n_subjects: int, avg_region_size: int, n_samples: int
    ) -> int:
        """Calculate batch size based on memory budget.

        Estimates memory per neighborhood and divides into budget.

        Args:
            n_subjects: Number of subjects
            avg_region_size: Average voxels per neighborhood/parcel
            n_samples: Number of time samples

        Returns:
            Number of neighborhoods per batch
        """
        # Memory per neighborhood during fitting:
        # - Local data: n_subjects × avg_region_size × n_samples × 8 bytes (float64)
        # - Transforms: n_subjects × avg_region_size × avg_region_size × 8 bytes
        # - Template: avg_region_size × n_samples × 8 bytes
        bytes_per_neighborhood = (
            n_subjects * avg_region_size * n_samples * 8  # local data
            + n_subjects * avg_region_size * avg_region_size * 8  # transforms
            + avg_region_size * n_samples * 8  # template
        )

        max_bytes = self.max_memory_gb * 1e9
        batch_size = max(1, int(max_bytes / bytes_per_neighborhood))

        logger.debug(
            f"Auto batch size: {batch_size} neighborhoods "
            f"({bytes_per_neighborhood / 1e6:.1f} MB each, "
            f"{self.max_memory_gb} GB budget)"
        )
        return batch_size

    def _batch_neighborhoods(
        self,
        neighborhoods: SphereNeighborhoods | PiecewiseNeighborhoods,
        n_subjects: int,
        n_samples: int,
    ) -> Iterator[list[tuple[int, np.ndarray]]]:
        """Generator yielding batches of neighborhoods.

        Critical: Uses yield + del pattern for memory efficiency.
        Only one batch is in memory at a time.

        Args:
            neighborhoods: Computed neighborhoods (searchlight or piecewise)
            n_subjects: Number of subjects
            n_samples: Number of time samples

        Yields:
            Lists of (region_id, voxel_indices) tuples, one batch at a time
        """
        # Collect all neighborhoods
        all_neighborhoods = list(neighborhoods.iter_neighborhoods(progress_bar=False))

        # Calculate batch size
        if self.n_neighborhoods_batch is not None:
            batch_size = self.n_neighborhoods_batch
        else:
            # Estimate average region size
            if len(all_neighborhoods) > 0:
                avg_region_size = int(
                    np.mean([len(indices) for _, indices in all_neighborhoods])
                )
            else:
                avg_region_size = 1
            batch_size = self._auto_batch_size(n_subjects, avg_region_size, n_samples)

        n_total = len(all_neighborhoods)
        n_batches = (n_total + batch_size - 1) // batch_size

        logger.info(
            f"Processing {n_total} neighborhoods in {n_batches} batches "
            f"(batch_size={batch_size})"
        )

        for i in range(0, n_total, batch_size):
            batch = all_neighborhoods[i : i + batch_size]
            yield batch
            # Caller should process and then del batch for memory efficiency

    def fit(self, data: list[np.ndarray], mask: nib.Nifti1Image) -> LocalAlignment:
        """Fit local alignment on multi-subject data.

        Args:
        data : List[np.ndarray]
            List of subject data arrays, each shape (n_voxels, n_samples).
            Subjects can have different numbers of samples - the underlying
            alignment methods (SRM, HyperAlignment) handle this via zero-padding.
        mask : Nifti1Image
            Brain mask defining the voxel space.

        Returns:
        self : LocalAlignment
            Fitted alignment model.
        """
        from nltools.data.braindata.neighborhoods import (
            compute_searchlight_neighborhoods,
        )

        # Validate inputs
        if not isinstance(data, list) or len(data) < 2:
            raise ValueError("data must be a list of at least 2 subject arrays")

        n_subjects = len(data)
        n_voxels = data[0].shape[0]
        # Allow different sample counts - underlying methods (SRM, HyperAlignment) handle padding
        sample_counts = [subj.shape[1] for subj in data]
        n_samples = max(sample_counts)  # Use max for memory estimation

        for i, subj in enumerate(data):
            if subj.shape[0] != n_voxels:
                raise ValueError(
                    f"All subjects must have same number of voxels. "
                    f"Subject 0 has {n_voxels}, subject {i} has {subj.shape[0]}"
                )

        # Store mask and voxel count
        self.mask_ = mask
        self.n_voxels_ = n_voxels

        # Initialize backend
        self.backend_ = self._init_backend()

        # Compute neighborhoods based on scheme
        if self.scheme == "searchlight":
            self.neighborhoods_ = compute_searchlight_neighborhoods(
                mask, radius_mm=self.radius_mm
            )
        elif self.scheme == "piecewise":
            self.neighborhoods_ = _compute_piecewise_neighborhoods(
                self.parcellation, mask
            )

        # Initialize storage for transforms and templates
        self.transforms_: dict[int, list[np.ndarray]] = {}
        self.template_: dict[int, np.ndarray] = {}

        # Fit local alignment for each neighborhood using batched processing
        if self.scheme == "searchlight":
            n_regions = self.neighborhoods_.n_voxels
            region_type = "searchlight spheres"
        else:
            n_regions = self.neighborhoods_.n_parcels
            region_type = "parcels"
        logger.info(f"Fitting LocalAlignment with {n_regions} {region_type}")

        # Use batched iteration for memory efficiency
        from tqdm import tqdm

        batch_gen = self._batch_neighborhoods(
            self.neighborhoods_, n_subjects, n_samples
        )

        # Progress bar for total neighborhoods
        pbar = tqdm(total=n_regions, desc=region_type.capitalize(), unit="regions")

        # Determine parallelization strategy
        # CPU parallel: use joblib with numpy backend (each worker gets own numpy)
        # GPU: sequential processing with torch backend (GPU parallelizes internally)
        use_cpu_parallel = self.parallel == "cpu" and self.n_jobs != 1
        use_gpu = self.parallel == "gpu" and self.backend_.name.startswith("torch")

        for batch in batch_gen:
            if use_cpu_parallel and len(batch) > 1:
                # CPU parallel processing within batch (uses numpy, no backend passed)
                from joblib import Parallel, delayed

                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(_fit_one_neighborhood)(
                        region_id,
                        voxel_indices,
                        data,
                        self.method,
                        self.n_iter,
                        self.n_features,
                        None,  # Each worker uses numpy backend
                    )
                    for region_id, voxel_indices in batch
                )

                # Store results
                for region_id, transforms, template in results:
                    self.transforms_[region_id] = transforms
                    self.template_[region_id] = template

                pbar.update(len(batch))
            else:
                # Sequential processing (GPU mode or single-threaded)
                # Pass backend for GPU acceleration
                backend_to_use = self.backend_ if use_gpu else None
                for region_id, voxel_indices in batch:
                    _, transforms, template = _fit_one_neighborhood(
                        region_id,
                        voxel_indices,
                        data,
                        self.method,
                        self.n_iter,
                        self.n_features,
                        backend_to_use,
                    )
                    self.transforms_[region_id] = transforms
                    self.template_[region_id] = template
                    pbar.update(1)

            # Explicit cleanup after each batch for memory efficiency
            del batch

        pbar.close()
        logger.info("LocalAlignment fitting complete")
        return self

    def transform(self, data: list[np.ndarray]) -> list[np.ndarray]:
        """Apply local transforms to data.

        For searchlight scheme with center-only aggregation: each voxel uses
        the transform from the neighborhood where it was the center.

        For piecewise scheme: all voxels in each parcel use the same transform.

        Args:
        data : List[np.ndarray]
            List of subject data arrays, each shape (n_voxels, n_samples).

        Returns:
        List[np.ndarray]
            Aligned data for each subject, shape (n_voxels, n_samples).
        """
        if self.transforms_ is None:
            raise ValueError("Model must be fit before transform")

        n_subjects = len(data)
        n_voxels, n_samples = data[0].shape

        if n_voxels != self.n_voxels_:
            raise ValueError(
                f"Data has {n_voxels} voxels but model was fit with {self.n_voxels_}"
            )

        # Determine parallelization strategy
        use_parallel = self.parallel == "cpu" and self.n_jobs != 1 and n_subjects > 1

        if use_parallel:
            # Parallel transform across subjects
            from joblib import Parallel, delayed

            def _transform_one_subject(subj_idx: int) -> np.ndarray:
                """Transform one subject's data."""
                subj_data = data[subj_idx]
                result = np.zeros((n_voxels, n_samples))

                for region_id, voxel_indices in self.neighborhoods_.iter_neighborhoods(
                    progress_bar=False
                ):
                    transforms = self.transforms_[region_id]
                    local_data = subj_data[voxel_indices, :]
                    transform = transforms[subj_idx]

                    if self.method == "procrustes":
                        aligned_local = transform @ local_data
                    else:
                        shared = transform.T @ local_data
                        aligned_local = transform @ shared

                    if self.aggregation == "center":
                        center_pos = np.where(voxel_indices == region_id)[0]
                        if len(center_pos) == 0:
                            center_pos = 0
                        else:
                            center_pos = center_pos[0]
                        result[region_id, :] = aligned_local[center_pos, :]
                    else:
                        result[voxel_indices, :] = aligned_local

                return result

            aligned = Parallel(n_jobs=self.n_jobs)(
                delayed(_transform_one_subject)(i) for i in range(n_subjects)
            )
        else:
            # Sequential transform
            aligned = [np.zeros((n_voxels, n_samples)) for _ in range(n_subjects)]

            for region_id, voxel_indices in self.neighborhoods_.iter_neighborhoods(
                progress_bar=True
            ):
                transforms = self.transforms_[region_id]

                for subj_idx, subj_data in enumerate(data):
                    local_data = subj_data[voxel_indices, :]
                    transform = transforms[subj_idx]

                    if self.method == "procrustes":
                        aligned_local = transform @ local_data
                    else:
                        shared = transform.T @ local_data
                        aligned_local = transform @ shared

                    if self.aggregation == "center":
                        center_pos = np.where(voxel_indices == region_id)[0]
                        if len(center_pos) == 0:
                            center_pos = 0
                        else:
                            center_pos = center_pos[0]
                        aligned[subj_idx][region_id, :] = aligned_local[center_pos, :]
                    else:
                        aligned[subj_idx][voxel_indices, :] = aligned_local

        return aligned

    def fit_transform(
        self, data: list[np.ndarray], mask: nib.Nifti1Image
    ) -> list[np.ndarray]:
        """Fit alignment and transform data in one step.

        Args:
        data : List[np.ndarray]
            List of subject data arrays, each shape (n_voxels, n_samples).
        mask : Nifti1Image
            Brain mask defining the voxel space.

        Returns:
        List[np.ndarray]
            Aligned data for each subject.
        """
        return self.fit(data, mask).transform(data)


__all__ = ["LocalAlignment"]
