"""Synthetic data factories for nltools benchmarks.

Three tiers, matching the three benchmark altitudes:

- **arrays** (`make_regression_arrays`, `make_group_maps`) — raw numpy for the
  algorithm-layer calls (`ridge_cv`, inference primitives).
- **in-memory BrainData** (`make_braindata`, `make_mask`, `make_labels`) — for
  the `BrainData.fit`/`.predict` facades.
- **on-disk subjects** (`make_ondisk_subjects`) — 4D niftis + a shared mask for
  the memory-critical `BrainCollection` lazy/path-backed path.

Voxel counts follow the neuroimaging conventions used across nltools: ~50k for
a 3mm whole-brain mask, ~230k for 2mm.
"""

from __future__ import annotations

from pathlib import Path

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


def make_ondisk_subjects(
    out_dir: Path | str,
    n_subjects: int,
    n_images: int,
    n_voxels: int,
    seed: int = 0,
) -> tuple[list[Path], Path]:
    """Write ``n_subjects`` 4D niftis + a shared mask; return ``(paths, mask_path)``.

    Each subject is a ``(x, y, z, n_images)`` volume; the mask selects
    ``n_voxels`` voxels, so a masked load yields ``(n_images, n_voxels)`` per
    subject. Keep ``n_voxels`` modest for `BrainCollection` scaling runs — the
    point is N-subject memory behavior under lazy loading, not voxel count.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_img = make_mask(n_voxels, seed)
    mask_path = out_dir / "mask.nii.gz"
    nib.save(mask_img, mask_path)

    spatial = mask_img.shape
    rng = np.random.default_rng(seed)
    paths: list[Path] = []
    for i in range(n_subjects):
        data = rng.standard_normal((*spatial, n_images)).astype(np.float32)
        path = out_dir / f"sub-{i:03d}.nii.gz"
        nib.save(nib.Nifti1Image(data, affine=np.eye(4)), path)
        paths.append(path)
    return paths, mask_path
