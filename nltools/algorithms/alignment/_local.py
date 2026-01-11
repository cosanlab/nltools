"""LocalAlignment: Neighborhood-based functional alignment.

Implements searchlight and piecewise schemes from Bazeille et al. 2021.
Uses center-only aggregation to preserve orthogonality of local transforms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

import numpy as np
from scipy.linalg import orthogonal_procrustes

if TYPE_CHECKING:
    import nibabel as nib

    from nltools.neighborhoods import SphereNeighborhoods

logger = logging.getLogger(__name__)


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

    parcel_to_voxels: Dict[int, np.ndarray]
    voxel_to_parcel: np.ndarray
    n_voxels: int
    n_parcels: int

    def iter_neighborhoods(
        self, show_progress: bool = False
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over all parcels.

        Yields:
            Tuple of (parcel_id, voxel_indices) for each parcel
        """
        iterator = self.parcel_to_voxels.items()

        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(list(iterator), desc="Piecewise", unit="parcels")

        for parcel_id, voxel_indices in iterator:
            yield parcel_id, voxel_indices

    def __repr__(self) -> str:
        return (
            f"PiecewiseNeighborhoods(n_voxels={self.n_voxels}, "
            f"n_parcels={self.n_parcels})"
        )


def _compute_piecewise_neighborhoods(
    parcellation: "nib.Nifti1Image",
    mask: "nib.Nifti1Image",
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

    parcel_to_voxels: Dict[int, np.ndarray] = {}
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


def _fit_local_procrustes(
    data: List[np.ndarray], n_iter: int = 3
) -> tuple[List[np.ndarray], np.ndarray]:
    """Fit multi-subject Procrustes alignment on local data.

    Iteratively refines a template by aligning all subjects and averaging.
    This is a simplified version of HyperAlignment for local neighborhoods.

    Args:
        data: List of arrays, each shape (n_local_voxels, n_samples).
        n_iter: Number of refinement iterations.

    Returns:
        transforms: List of orthogonal transforms, each (n_local_voxels, n_local_voxels).
        template: Mean template in aligned space, shape (n_local_voxels, n_samples).
    """
    n_subjects = len(data)
    n_voxels, n_samples = data[0].shape

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
            R, _ = orthogonal_procrustes(template.T, x.T)
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

    Parameters
    ----------
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
        Parallelization: 'cpu' or None.
    n_jobs : int, default=-1
        Number of jobs for CPU parallelization.

    Attributes
    ----------
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

    Examples
    --------
    >>> import numpy as np
    >>> from nltools.algorithms.alignment import LocalAlignment
    >>> # Create synthetic multi-subject data (voxels, samples)
    >>> data = [np.random.randn(1000, 100) for _ in range(5)]
    >>> la = LocalAlignment(scheme='searchlight', method='procrustes', radius_mm=10.0)
    >>> la.fit(data, mask)
    >>> aligned = la.transform(data)

    Notes
    -----
    Based on Bazeille et al. 2021 "An empirical evaluation of functional
    alignment using inter-subject decoding". Center-only aggregation is
    used to preserve local orthogonality of transforms.
    """

    # Configuration
    scheme: str = "searchlight"
    method: str = "procrustes"
    radius_mm: float = 10.0
    parcellation: Optional[Any] = None  # Nifti1Image
    n_features: Optional[int] = None
    n_iter: int = 3
    aggregation: str = "center"
    parallel: Optional[str] = "cpu"
    n_jobs: int = -1

    # Batching parameters (Phase 2)
    n_neighborhoods_batch: Optional[int] = None  # None = auto-calculate
    max_memory_gb: float = 4.0  # Memory budget for auto batch sizing

    # Fitted state (set by fit())
    transforms_: Optional[Dict[int, List[np.ndarray]]] = field(default=None, repr=False)
    template_: Optional[Dict[int, np.ndarray]] = field(default=None, repr=False)
    neighborhoods_: Optional[Union["SphereNeighborhoods", Dict[int, np.ndarray]]] = (
        field(default=None, repr=False)
    )
    n_voxels_: Optional[int] = field(default=None, repr=False)
    mask_: Optional[Any] = field(default=None, repr=False)  # Nifti1Image

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
        neighborhoods: Union["SphereNeighborhoods", "PiecewiseNeighborhoods"],
        n_subjects: int,
        n_samples: int,
    ) -> Iterator[List[tuple[int, np.ndarray]]]:
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
        all_neighborhoods = list(neighborhoods.iter_neighborhoods(show_progress=False))

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

    def fit(self, data: List[np.ndarray], mask: "nib.Nifti1Image") -> "LocalAlignment":
        """Fit local alignment on multi-subject data.

        Parameters
        ----------
        data : List[np.ndarray]
            List of subject data arrays, each shape (n_voxels, n_samples).
            All subjects must have the same number of samples.
        mask : Nifti1Image
            Brain mask defining the voxel space.

        Returns
        -------
        self : LocalAlignment
            Fitted alignment model.
        """
        from nltools.neighborhoods import compute_searchlight_neighborhoods

        # Validate inputs
        if not isinstance(data, list) or len(data) < 2:
            raise ValueError("data must be a list of at least 2 subject arrays")

        n_subjects = len(data)
        n_voxels, n_samples = data[0].shape

        for i, subj in enumerate(data):
            if subj.shape[0] != n_voxels:
                raise ValueError(
                    f"All subjects must have same number of voxels. "
                    f"Subject 0 has {n_voxels}, subject {i} has {subj.shape[0]}"
                )
            if subj.shape[1] != n_samples:
                raise ValueError(
                    f"All subjects must have same number of samples. "
                    f"Subject 0 has {n_samples}, subject {i} has {subj.shape[1]}"
                )

        # Store mask and voxel count
        self.mask_ = mask
        self.n_voxels_ = n_voxels

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
        self.transforms_: Dict[int, List[np.ndarray]] = {}
        self.template_: Dict[int, np.ndarray] = {}

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

        for batch in batch_gen:
            for region_id, voxel_indices in batch:
                # Extract local data for all subjects
                # Each subject's local data: (n_local_voxels, n_samples)
                local_data = [subj[voxel_indices, :] for subj in data]

                # Skip if region is too small
                n_local_voxels = len(voxel_indices)
                if n_local_voxels < 2:
                    # Store identity transforms for degenerate cases
                    self.transforms_[region_id] = [
                        np.eye(n_local_voxels) for _ in range(n_subjects)
                    ]
                    self.template_[region_id] = np.mean(local_data, axis=0)
                    pbar.update(1)
                    continue

                # Fit local alignment based on method
                if self.method == "procrustes":
                    transforms, template = _fit_local_procrustes(
                        local_data, n_iter=self.n_iter
                    )
                    self.transforms_[region_id] = transforms
                    self.template_[region_id] = template

                elif self.method == "srm":
                    from nltools.algorithms.srm import SRM

                    # SRM expects (n_voxels, n_samples) - already in this format
                    # n_features defaults to min(n_local_voxels, n_samples) if None
                    n_features = self.n_features
                    if n_features is None:
                        n_features = min(n_local_voxels, n_samples)

                    srm = SRM(n_iter=self.n_iter, features=n_features)
                    srm.fit(local_data, parallel=None)  # No nested parallelism

                    # Store transforms (w_) and template (s_)
                    self.transforms_[region_id] = srm.w_
                    self.template_[region_id] = srm.s_

                elif self.method == "hyperalignment":
                    from nltools.algorithms.hyperalignment import HyperAlignment

                    # HyperAlignment expects (n_features, n_samples) - same format
                    ha = HyperAlignment(n_iter=self.n_iter)
                    ha.fit(local_data, parallel=None)  # No nested parallelism

                    # Store transforms (w_) and template (s_)
                    self.transforms_[region_id] = ha.w_
                    self.template_[region_id] = ha.s_

                pbar.update(1)

            # Explicit cleanup after each batch for memory efficiency
            del batch

        pbar.close()
        logger.info("LocalAlignment fitting complete")
        return self

    def transform(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """Apply local transforms to data.

        For searchlight scheme with center-only aggregation: each voxel uses
        the transform from the neighborhood where it was the center.

        For piecewise scheme: all voxels in each parcel use the same transform.

        Parameters
        ----------
        data : List[np.ndarray]
            List of subject data arrays, each shape (n_voxels, n_samples).

        Returns
        -------
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

        # Initialize output arrays
        aligned = [np.zeros((n_voxels, n_samples)) for _ in range(n_subjects)]

        # Apply transforms based on aggregation mode
        for region_id, voxel_indices in self.neighborhoods_.iter_neighborhoods(
            show_progress=True
        ):
            transforms = self.transforms_[region_id]

            for subj_idx, subj_data in enumerate(data):
                # Extract local data for this subject
                local_data = subj_data[voxel_indices, :]

                # Apply transform
                # For procrustes: transform is (n_local, n_local)
                # For srm/ha: transform is (n_local, n_features)
                transform = transforms[subj_idx]

                if self.method == "procrustes":
                    # Full Procrustes rotation: aligned = R @ local_data
                    aligned_local = transform @ local_data
                else:
                    # SRM/HA: project to shared space and back
                    # transform shape: (n_local, n_features)
                    shared = transform.T @ local_data  # (n_features, n_samples)
                    aligned_local = transform @ shared  # (n_local, n_samples)

                # Apply aggregation
                if self.aggregation == "center":
                    # Center-only: extract just the center voxel
                    # For searchlight, region_id is the center voxel index
                    center_pos = np.where(voxel_indices == region_id)[0]
                    if len(center_pos) == 0:
                        center_pos = 0
                    else:
                        center_pos = center_pos[0]
                    aligned[subj_idx][region_id, :] = aligned_local[center_pos, :]
                else:
                    # 'all': Apply to all voxels in region (piecewise)
                    aligned[subj_idx][voxel_indices, :] = aligned_local

        return aligned

    def fit_transform(
        self, data: List[np.ndarray], mask: "nib.Nifti1Image"
    ) -> List[np.ndarray]:
        """Fit alignment and transform data in one step.

        Parameters
        ----------
        data : List[np.ndarray]
            List of subject data arrays, each shape (n_voxels, n_samples).
        mask : Nifti1Image
            Brain mask defining the voxel space.

        Returns
        -------
        List[np.ndarray]
            Aligned data for each subject.
        """
        return self.fit(data, mask).transform(data)


__all__ = ["LocalAlignment"]
