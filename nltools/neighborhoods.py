"""Spatial neighborhood computation for neuroimaging analyses.

This module provides efficient computation and caching of spatial neighborhoods
(spheres) around brain voxels. It is designed to support searchlight analyses,
ISC, and other operations that require iterating over local brain regions.

The key insight is that for a given mask and radius, the neighborhood structure
is deterministic and can be cached for reuse across analyses.

Example:
    >>> import nibabel as nib
    >>> from nltools.neighborhoods import compute_searchlight_neighborhoods
    >>>
    >>> mask = nib.load("mask.nii.gz")
    >>> neighborhoods = compute_searchlight_neighborhoods(mask, radius_mm=10.0)
    >>>
    >>> # Iterate over all voxels and their neighborhoods
    >>> for center_idx, neighbor_indices in neighborhoods.iter_neighborhoods():
    ...     # Extract data for these voxels
    ...     local_data = data[:, neighbor_indices]
    ...     result[center_idx] = analyze(local_data)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import numpy as np
from scipy import sparse
from sklearn import neighbors

if TYPE_CHECKING:
    from nibabel import Nifti1Image


__all__ = ["SphereNeighborhoods", "compute_searchlight_neighborhoods"]


@dataclass
class SphereNeighborhoods:
    """Precomputed sphere neighborhoods for a brain mask.

    This dataclass stores a sparse adjacency matrix where row i contains True
    for all voxels within the specified radius of voxel i. It provides efficient
    iteration over neighborhoods for searchlight-style analyses.

    Attributes:
        adjacency: Sparse CSR matrix (n_voxels, n_voxels) where adjacency[i, j]
            is True if voxel j is within radius of voxel i
        mask_hash: Hash of the source mask for validation
        radius_mm: Radius in millimeters
        n_voxels: Number of voxels in the mask

    Example:
        >>> neighborhoods = compute_searchlight_neighborhoods(mask, radius_mm=10.0)
        >>> print(f"Mean neighborhood size: {neighborhoods.mean_size:.1f} voxels")
        >>>
        >>> # Get neighbors of a specific voxel
        >>> neighbor_idx = neighborhoods.get_neighbors(100)
        >>> print(f"Voxel 100 has {len(neighbor_idx)} neighbors")
    """

    adjacency: sparse.csr_matrix
    mask_hash: str
    radius_mm: float
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
        self, show_progress: bool = False
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over all neighborhoods.

        Yields:
            Tuple of (center_voxel_idx, neighbor_indices) for each voxel

        Args:
            show_progress: If True, wrap iterator with tqdm progress bar
        """
        iterator = range(self.n_voxels)

        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Searchlight", unit="voxels")

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
            f"SphereNeighborhoods(n_voxels={self.n_voxels}, "
            f"radius={self.radius_mm}mm, "
            f"mean_size={self.mean_size:.1f})"
        )


def compute_searchlight_neighborhoods(
    mask_img: "Nifti1Image",
    radius_mm: float = 10.0,
    use_cache: bool = True,
) -> SphereNeighborhoods:
    """Compute sphere neighborhoods for all voxels in a brain mask.

    For each voxel in the mask, this function identifies all other voxels
    within the specified radius (in millimeters). The result is cached to
    disk for fast reloading in subsequent analyses.

    The algorithm uses sklearn's BallTree for efficient radius queries in
    world coordinates (mm), ensuring accurate neighborhoods regardless of
    voxel resolution.

    Args:
        mask_img: NIfTI mask image defining the brain region
        radius_mm: Radius of spheres in millimeters (default: 10.0)
        use_cache: If True, cache results to ~/.nltools/cache/searchlight/
            for fast reloading (default: True)

    Returns:
        SphereNeighborhoods with precomputed adjacency matrix

    Raises:
        ValueError: If mask has no non-zero voxels

    Example:
        >>> import nibabel as nib
        >>> mask = nib.load("brain_mask.nii.gz")
        >>>
        >>> # First call computes and caches (may take a few seconds)
        >>> neighborhoods = compute_searchlight_neighborhoods(mask, radius_mm=8.0)
        >>>
        >>> # Subsequent calls load from cache (~50ms)
        >>> neighborhoods = compute_searchlight_neighborhoods(mask, radius_mm=8.0)
        >>>
        >>> print(neighborhoods)
        SphereNeighborhoods(n_voxels=50000, radius=8.0mm, mean_size=33.2)

    Notes:
        Cache location: ~/.nltools/cache/searchlight/{mask_hash}_{radius}mm.npz

        For a typical 2mm MNI mask (~50k voxels) with 10mm radius:
        - First run: ~1-2 seconds
        - Cached load: ~50ms
    """
    from nilearn.image.resampling import coord_transform

    from .cache import CacheManager, hash_mask

    # Compute mask hash for cache key
    mask_hash = hash_mask(mask_img)
    cache_key = f"{mask_hash}_{radius_mm}mm"

    # Try to load from cache
    if use_cache:
        cache = CacheManager("searchlight")
        cached = cache.load(cache_key)
        if cached is not None:
            # Reconstruct sparse matrix from cached components
            # We only store indices and indptr (not data, since all values are 1)
            n_entries = len(cached["indices"])
            adjacency = sparse.csr_matrix(
                (
                    np.ones(n_entries, dtype=np.float32),
                    cached["indices"],
                    cached["indptr"],
                ),
                shape=tuple(cached["shape"]),
            )
            return SphereNeighborhoods(
                adjacency=adjacency,
                mask_hash=str(cached["mask_hash"]),
                radius_mm=float(cached["radius_mm"]),
                n_voxels=int(cached["n_voxels"]),
            )

    # Compute neighborhoods
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
    clf = neighbors.NearestNeighbors(radius=radius_mm, algorithm="ball_tree")
    clf.fit(mask_coords_world)
    adjacency = clf.radius_neighbors_graph(mask_coords_world, mode="connectivity")
    adjacency = adjacency.tocsr()

    # Cache the result (omit data array since all values are 1)
    # Use uncompressed npz for faster load times (more important than file size)
    if use_cache:
        cache.save(
            cache_key,
            compressed=False,
            indices=adjacency.indices,
            indptr=adjacency.indptr,
            shape=np.array(adjacency.shape),
            mask_hash=np.array(mask_hash),
            radius_mm=np.array(radius_mm),
            n_voxels=np.array(n_voxels),
        )

    return SphereNeighborhoods(
        adjacency=adjacency,
        mask_hash=mask_hash,
        radius_mm=radius_mm,
        n_voxels=n_voxels,
    )
