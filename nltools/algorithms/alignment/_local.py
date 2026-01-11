"""LocalAlignment: Neighborhood-based functional alignment.

Implements searchlight and piecewise schemes from Bazeille et al. 2021.
Uses center-only aggregation to preserve orthogonality of local transforms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
from scipy.linalg import orthogonal_procrustes

if TYPE_CHECKING:
    import nibabel as nib

    from nltools.neighborhoods import SphereNeighborhoods

logger = logging.getLogger(__name__)


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
        if self.aggregation not in ("center",):
            raise ValueError(
                f"Unknown aggregation: {self.aggregation}. "
                "Only 'center' is supported in Phase 1."
            )

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
            raise NotImplementedError(
                "Piecewise scheme not yet implemented - see nltools-oqil.4"
            )

        # Initialize storage for transforms and templates
        self.transforms_: Dict[int, List[np.ndarray]] = {}
        self.template_: Dict[int, np.ndarray] = {}

        # Fit local alignment for each neighborhood
        logger.info(
            f"Fitting LocalAlignment with {self.neighborhoods_.n_voxels} neighborhoods"
        )

        for center_idx, neighbor_indices in self.neighborhoods_.iter_neighborhoods(
            show_progress=True
        ):
            # Extract local data for all subjects
            # Each subject's local data: (n_neighbors, n_samples)
            local_data = [subj[neighbor_indices, :] for subj in data]

            # Skip if neighborhood is too small
            n_neighbors = len(neighbor_indices)
            if n_neighbors < 2:
                # Store identity transforms for degenerate cases
                self.transforms_[center_idx] = [
                    np.eye(n_neighbors) for _ in range(n_subjects)
                ]
                self.template_[center_idx] = np.mean(local_data, axis=0)
                continue

            # Fit local alignment based on method
            if self.method == "procrustes":
                transforms, template = _fit_local_procrustes(
                    local_data, n_iter=self.n_iter
                )
                self.transforms_[center_idx] = transforms
                self.template_[center_idx] = template

            elif self.method == "srm":
                from nltools.algorithms.srm import SRM

                # SRM expects (n_voxels, n_samples) - already in this format
                # n_features defaults to min(n_neighbors, n_samples) if None
                n_features = self.n_features
                if n_features is None:
                    n_features = min(n_neighbors, n_samples)

                srm = SRM(n_iter=self.n_iter, features=n_features)
                srm.fit(local_data, parallel=None)  # No nested parallelism

                # Store transforms (w_) and template (s_)
                self.transforms_[center_idx] = srm.w_
                self.template_[center_idx] = srm.s_

            elif self.method == "hyperalignment":
                from nltools.algorithms.hyperalignment import HyperAlignment

                # HyperAlignment expects (n_features, n_samples) - same format
                ha = HyperAlignment(n_iter=self.n_iter)
                ha.fit(local_data, parallel=None)  # No nested parallelism

                # Store transforms (w_) and template (s_)
                self.transforms_[center_idx] = ha.w_
                self.template_[center_idx] = ha.s_

        logger.info("LocalAlignment fitting complete")
        return self

    def transform(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """Apply local transforms to data.

        Uses center-only aggregation: each voxel uses the transform from
        the neighborhood where it was the center.

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

        # Apply center-only aggregation
        for center_idx, neighbor_indices in self.neighborhoods_.iter_neighborhoods(
            show_progress=True
        ):
            transforms = self.transforms_[center_idx]

            # Find position of center within the neighborhood
            center_pos = np.where(neighbor_indices == center_idx)[0]
            if len(center_pos) == 0:
                # Center not in its own neighborhood (shouldn't happen normally)
                # Use first position as fallback
                center_pos = 0
            else:
                center_pos = center_pos[0]

            for subj_idx, subj_data in enumerate(data):
                # Extract local data for this subject
                local_data = subj_data[neighbor_indices, :]

                # Apply transform
                # For procrustes: transform is (n_neighbors, n_neighbors)
                # For srm/ha: transform is (n_neighbors, n_features)
                transform = transforms[subj_idx]

                if self.method == "procrustes":
                    # Full Procrustes rotation: aligned = R @ local_data
                    aligned_local = transform @ local_data
                    # Extract center voxel value
                    aligned[subj_idx][center_idx, :] = aligned_local[center_pos, :]
                else:
                    # SRM/HA: project to shared space and back
                    # transform shape: (n_neighbors, n_features)
                    # To get aligned center voxel, we need to:
                    # 1. Project to shared space: W.T @ local_data → (n_features, n_samples)
                    # 2. Project back using template structure
                    # For center-only, we just use the center row of W
                    # to get the features, then use template to reconstruct
                    shared = transform.T @ local_data  # (n_features, n_samples)
                    # Use template row for center to weight reconstruction
                    # Actually, simpler: aligned center = W[center_pos,:] @ shared
                    # But W @ shared would give us back local_data-ish
                    # For center-only: store the shared representation weighted by center
                    # Since we want aligned output in original space dimensions,
                    # we project back: aligned_local = W @ shared (n_neighbors, n_samples)
                    # Then extract center
                    aligned_local = transform @ shared
                    aligned[subj_idx][center_idx, :] = aligned_local[center_pos, :]

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
