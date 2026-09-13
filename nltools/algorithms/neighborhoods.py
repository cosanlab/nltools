"""Spatial neighborhood computation for neuroimaging analyses.

This module computes spatial neighborhoods (spheres) around brain voxels. It is
designed to support searchlight analyses, ISC, and other operations that require
iterating over local brain regions.

Examples:
    ```python
    import nibabel as nib
    from nltools.algorithms.neighborhoods import compute_searchlight_neighborhoods

    mask = nib.load("mask.nii.gz")
    neighborhoods = compute_searchlight_neighborhoods(mask, radius=10.0)

    # Iterate over all voxels and their neighborhoods
    for center_idx, neighbor_indices in neighborhoods.iter_neighborhoods():
        local_data = data[:, neighbor_indices]  # data for the voxels in this sphere
        result[center_idx] = analyze(local_data)
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from collections.abc import Iterator

import numpy as np
from scipy import sparse
from sklearn import neighbors

from nltools.utils import _maybe_tqdm

if TYPE_CHECKING:
    from nibabel import Nifti1Image


@dataclass(frozen=True)
class _SphereNeighborhoods:
    """Precomputed sphere neighborhoods for a brain mask.

    This dataclass stores a sparse adjacency matrix where row i contains True
    for all voxels within the specified radius of voxel i. It provides efficient
    iteration over neighborhoods for searchlight-style analyses.

    Attributes:
        adjacency (sparse.csr_matrix): ``(n_voxels, n_voxels)`` matrix where
            ``adjacency[i, j]`` is nonzero if voxel ``j`` is within the radius
            of voxel ``i``.
        radius (float): Radius in millimeters.
        n_voxels (int): Number of voxels in the mask.
        mean_size (float): Mean neighborhood size in voxels.
        min_size (int): Smallest neighborhood size in voxels.
        max_size (int): Largest neighborhood size in voxels.

    Examples:
        ```python
        neighborhoods = compute_searchlight_neighborhoods(mask, radius=10.0)
        print(f"Mean neighborhood size: {neighborhoods.mean_size:.1f} voxels")

        # Get neighbors of a specific voxel
        neighbor_idx = neighborhoods.get_neighbors(100)
        print(f"Voxel 100 has {len(neighbor_idx)} neighbors")
        ```
    """

    adjacency: sparse.csr_matrix
    radius: float
    n_voxels: int

    def get_neighbors(self, voxel_idx: int) -> np.ndarray:
        """Get indices of all voxels in the neighborhood of a given voxel.

        Args:
            voxel_idx: Index of the center voxel (0 to n_voxels-1)

        Returns:
            Array of voxel indices within radius of the center voxel
        """
        return self.adjacency[voxel_idx].indices

    def get_neighborhood_size(self, voxel_idx: int) -> int:
        """Get the number of voxels in a neighborhood.

        Args:
            voxel_idx: Index of the center voxel

        Returns:
            Number of voxels in the neighborhood
        """
        return self.adjacency[voxel_idx].nnz

    def iter_neighborhoods(
        self, *, progress_bar: bool = False
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over all neighborhoods.

        Args:
            progress_bar: If True, wrap the iterator with a tqdm progress bar.

        Yields:
            tuple[int, np.ndarray]: ``(center_voxel_idx, neighbor_indices)`` for
                each voxel.
        """
        iterator = _maybe_tqdm(
            range(self.n_voxels),
            progress_bar=progress_bar,
            desc="Searchlight",
            unit="voxels",
        )

        for i in iterator:
            yield i, self.get_neighbors(i)

    @property
    def mean_size(self) -> float:
        """Mean neighborhood size in voxels."""
        return float(self.adjacency.sum() / self.n_voxels)

    @property
    def min_size(self) -> int:
        """Minimum neighborhood size."""
        sizes = np.diff(self.adjacency.indptr)
        return int(sizes.min())

    @property
    def max_size(self) -> int:
        """Maximum neighborhood size."""
        sizes = np.diff(self.adjacency.indptr)
        return int(sizes.max())

    def __repr__(self) -> str:
        return (
            f"_SphereNeighborhoods(n_voxels={self.n_voxels}, "
            f"radius={self.radius}mm, "
            f"mean_size={self.mean_size:.1f})"
        )


def compute_searchlight_neighborhoods(
    mask_img: Nifti1Image,
    radius: float = 10.0,
) -> _SphereNeighborhoods:
    """Compute sphere neighborhoods for all voxels in a brain mask.

    For each voxel in the mask, this function identifies all other voxels
    within the specified radius (in millimeters).

    The algorithm uses sklearn's BallTree for efficient radius queries in
    world coordinates (mm), ensuring accurate neighborhoods regardless of
    voxel resolution.

    Args:
        mask_img: NIfTI mask image defining the brain region
        radius: Radius of spheres in millimeters (default: 10.0)

    Returns:
        _SphereNeighborhoods with precomputed adjacency matrix

    Raises:
        ValueError: If mask has no non-zero voxels

    Examples:
        ```python
        import nibabel as nib

        mask = nib.load("brain_mask.nii.gz")
        neighborhoods = compute_searchlight_neighborhoods(mask, radius=8.0)

        print(neighborhoods)
        # _SphereNeighborhoods(n_voxels=50000, radius=8.0mm, mean_size=33.2)
        ```
    """
    from nilearn.image.resampling import coord_transform

    mask_data = mask_img.get_fdata().astype(bool)
    affine = mask_img.affine

    # Get voxel coordinates in world space (mm)
    mask_coords_voxel = np.array(np.nonzero(mask_data)).T  # (n_voxels, 3)
    n_voxels = mask_coords_voxel.shape[0]

    if n_voxels == 0:
        raise ValueError("Mask contains no non-zero voxels")

    # Transform to world coordinates using affine
    mask_coords_world = np.array(
        coord_transform(
            mask_coords_voxel[:, 0],
            mask_coords_voxel[:, 1],
            mask_coords_voxel[:, 2],
            affine,
        )
    ).T  # (n_voxels, 3)

    # Use BallTree for efficient radius queries
    # This is the same approach used by nilearn's searchlight
    clf = neighbors.NearestNeighbors(radius=radius, algorithm="ball_tree")
    clf.fit(mask_coords_world)
    adjacency = clf.radius_neighbors_graph(mask_coords_world, mode="connectivity")
    adjacency = adjacency.tocsr()

    return _SphereNeighborhoods(
        adjacency=adjacency,
        radius=radius,
        n_voxels=n_voxels,
    )
