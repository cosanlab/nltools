"""Disk-based caching infrastructure for expensive computations.

This module provides a general-purpose caching system for nltools, designed to
be reused across various computationally expensive operations like searchlight
neighborhoods, ISC, and SRM.

Example:
    >>> from nltools.data.braindata.cache import CacheManager, hash_mask
    >>> import nibabel as nib
    >>>
    >>> # Hash a mask for cache key generation
    >>> mask = nib.load("mask.nii.gz")
    >>> mask_hash = hash_mask(mask)
    >>>
    >>> # Use cache manager for searchlight neighborhoods
    >>> cache = CacheManager("searchlight")
    >>> if not cache.exists(f"{mask_hash}_10mm"):
    ...     # Compute expensive operation
    ...     result = compute_something()
    ...     cache.save(f"{mask_hash}_10mm", data=result)
    >>> else:
    ...     result = cache.load(f"{mask_hash}_10mm")["data"]
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nibabel import Nifti1Image


__all__ = ["CacheManager", "clear_cache", "get_cache_dir", "hash_mask"]


def get_cache_dir() -> Path:
    """Get the nltools cache directory.

    Returns ~/.nltools/cache, creating it if necessary.

    Returns:
        Path to cache directory
    """
    cache_dir = Path.home() / ".nltools" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def hash_mask(mask_img: Nifti1Image) -> str:
    """Compute a stable hash for a NIfTI mask image.

    The hash is based on the mask's shape, affine transformation, and the
    actual voxel positions. This ensures that masks with the same shape but
    different voxel locations (or different affines) produce different hashes.

    Args:
        mask_img: NIfTI image to hash (typically a binary mask)

    Returns:
        16-character hexadecimal hash string

    Example:
        >>> import nibabel as nib
        >>> mask = nib.load("mask.nii.gz")
        >>> hash_mask(mask)
        'a1b2c3d4e5f6g7h8'
    """
    mask_data = mask_img.get_fdata().astype(bool)
    affine = mask_img.affine

    # Create reproducible hash from shape, affine, and mask positions
    shape_bytes = np.array(mask_data.shape, dtype=np.int64).tobytes()
    affine_bytes = affine.astype(np.float64).tobytes()
    # Include actual mask positions to detect different masks with same shape
    mask_indices = np.array(np.nonzero(mask_data), dtype=np.int64).tobytes()

    combined = shape_bytes + affine_bytes + mask_indices
    return hashlib.sha256(combined).hexdigest()[:16]


class CacheManager:
    """Manages disk-based caching for expensive computations.

    CacheManager provides a simple key-value interface for caching numpy arrays
    to disk. It organizes cached files by category (e.g., "searchlight", "isc")
    in separate subdirectories.

    Args:
        category: Category name for organizing cached files (e.g., "searchlight")

    Example:
        >>> cache = CacheManager("searchlight")
        >>>
        >>> # Check if something is cached
        >>> if cache.exists("mykey"):
        ...     data = cache.load("mykey")
        ... else:
        ...     result = expensive_computation()
        ...     cache.save("mykey", adjacency=result, metadata=metadata)
        ...     data = {"adjacency": result, "metadata": metadata}
    """

    def __init__(self, category: str = "general"):
        self.category = category
        self.cache_dir = get_cache_dir() / category
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, key: str, ext: str = ".npz") -> Path:
        """Get the file path for a cache key.

        Args:
            key: Cache key
            ext: File extension (default: ".npz")

        Returns:
            Path to the cache file
        """
        return self.cache_dir / f"{key}{ext}"

    def exists(self, key: str, ext: str = ".npz") -> bool:
        """Check if a cache key exists.

        Args:
            key: Cache key
            ext: File extension (default: ".npz")

        Returns:
            True if cached file exists
        """
        return self.get_path(key, ext).exists()

    def load(self, key: str) -> dict | None:
        """Load cached data.

        Args:
            key: Cache key

        Returns:
            Dictionary of cached arrays, or None if not cached
        """
        path = self.get_path(key)
        if not path.exists():
            return None
        return dict(np.load(path, allow_pickle=True))

    def save(self, key: str, compressed: bool = True, **arrays) -> Path:
        """Save arrays to cache.

        Args:
            key: Cache key
            compressed: If True, use compressed npz format (smaller but slower)
            **arrays: Named arrays to cache

        Returns:
            Path to saved cache file
        """
        path = self.get_path(key)
        if compressed:
            np.savez_compressed(path, **arrays)
        else:
            np.savez(path, **arrays)
        return path

    def delete(self, key: str, ext: str = ".npz") -> bool:
        """Delete a cached file.

        Args:
            key: Cache key
            ext: File extension

        Returns:
            True if file was deleted, False if it didn't exist
        """
        path = self.get_path(key, ext)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        """Clear all cached files in this category.

        Returns:
            Number of files deleted
        """
        count = 0
        for path in self.cache_dir.glob("*"):
            if path.is_file():
                path.unlink()
                count += 1
        return count

    def list_keys(self, ext: str = ".npz") -> list[str]:
        """List all cached keys in this category.

        Args:
            ext: File extension to match

        Returns:
            List of cache keys (without extension)
        """
        return [p.stem for p in self.cache_dir.glob(f"*{ext}")]


def clear_cache(category: str | None = None) -> int:
    """Clear the nltools cache.

    Args:
        category: If provided, only clear this category. Otherwise clear all.

    Returns:
        Number of files deleted
    """
    cache_dir = get_cache_dir()
    count = 0

    if category:
        cat_dir = cache_dir / category
        if cat_dir.exists():
            for path in cat_dir.glob("*"):
                if path.is_file():
                    path.unlink()
                    count += 1
    else:
        for cat_dir in cache_dir.iterdir():
            if cat_dir.is_dir():
                for path in cat_dir.glob("*"):
                    if path.is_file():
                        path.unlink()
                        count += 1

    return count
