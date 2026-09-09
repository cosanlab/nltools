"""Synthetic data factories for nltools benchmarks.

Factories for two kinds of benchmark input:

- **arrays** (`make_regression_arrays`, `make_group_maps`) — raw numpy for the
  algorithm-layer calls (inference primitives) and the `Ridge` estimator.
- **in-memory BrainData** (`make_braindata`, `make_mask`, `make_labels`) — for
  the `BrainData.fit`/`.predict` facades.

Voxel counts follow the neuroimaging conventions used across nltools: ~50k for
a 3mm whole-brain mask, ~230k for 2mm.
"""

from __future__ import annotations


import nibabel as nib
import numpy as np


def make_mask(n_voxels: int, seed: int = 0) -> nib.Nifti1Image:
    """A 3D Nifti mask with exactly ``n_voxels`` scattered in-brain voxels."""
    side = int(np.ceil(n_voxels ** (1 / 3))) + 1
    vol = np.zeros(side**3, dtype=np.int16)
    vol[:n_voxels] = 1
    np.random.default_rng(seed).shuffle(vol)  # scatter, not a contiguous block
    return nib.Nifti1Image(vol.reshape(side, side, side), affine=np.eye(4))


def make_regression_arrays(
    n_samples: int, n_voxels: int, n_features: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Design ``X (n_samples, n_features)`` + brain ``y (n_samples, n_voxels)``."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = rng.standard_normal((n_samples, n_voxels)).astype(np.float32)
    return x, y


def make_group_maps(n_subjects: int, n_voxels: int, seed: int = 0) -> np.ndarray:
    """Group-level ``(n_subjects, n_voxels)`` maps for permutation/bootstrap."""
    return (
        np.random.default_rng(seed)
        .standard_normal((n_subjects, n_voxels))
        .astype(np.float32)
    )


def make_braindata(n_images: int, n_voxels: int, seed: int = 0):
    """In-memory `BrainData` of shape ``(n_images, n_voxels)`` with a fresh mask."""
    from nltools.data import BrainData

    mask = make_mask(n_voxels, seed)
    data = (
        np.random.default_rng(seed)
        .standard_normal((n_images, n_voxels))
        .astype(np.float32)
    )
    return BrainData(data, mask=mask)


def make_labels(n_samples: int, n_classes: int = 2, seed: int = 0) -> np.ndarray:
    """Balanced-ish integer class labels for MVPA decoding benchmarks."""
    return np.random.default_rng(seed).integers(0, n_classes, size=n_samples)
