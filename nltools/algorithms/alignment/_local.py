"""LocalAlignment: Neighborhood-based functional alignment.

Implements searchlight and piecewise schemes from Bazeille et al. 2021.
Uses center-only aggregation to preserve orthogonality of local transforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import nibabel as nib

    from nltools.neighborhoods import SphereNeighborhoods


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
        raise NotImplementedError("fit() not yet implemented - see nltools-oqil.2")

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
        raise NotImplementedError(
            "transform() not yet implemented - see nltools-oqil.3"
        )

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
