"""MVPA decoding benchmarks: BrainData.predict across spatial scales.

Covers the three `spatial_scale` paths (Jolly & Chang, 2021): whole-brain (one
model over all voxels), ROI (one model per parcel), and searchlight (one model
per voxel-sphere). Searchlight cost is O(n_voxels) model fits, so it is swept at
a deliberately small voxel count even in full mode.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np

from benchmarks.harness import BenchResult, benchmark
from benchmarks.workloads import make_braindata, make_labels


def _make_atlas(
    mask: nib.Nifti1Image, n_parcels: int, seed: int = 0
) -> nib.Nifti1Image:
    """Label image assigning each in-mask voxel to one of ``n_parcels`` parcels."""
    vol = np.asarray(mask.dataobj)
    labels = np.zeros(vol.shape, dtype=np.int16)
    in_mask = vol > 0
    n_vox = int(in_mask.sum())
    parcel_ids = np.random.default_rng(seed).integers(1, n_parcels + 1, size=n_vox)
    labels[in_mask] = parcel_ids
    return nib.Nifti1Image(labels, affine=mask.affine)


def run(reps: int = 3, quick: bool = False) -> list[BenchResult]:
    results: list[BenchResult] = []

    # Whole-brain + ROI at realistic-ish sizes; MVPA decoding uses many samples.
    n_images, n_voxels = (30, 2_000) if quick else (200, 20_000)
    bd = make_braindata(n_images, n_voxels)
    y = make_labels(n_images, n_classes=2)

    results.append(
        benchmark(
            lambda: bd.predict(y=y, spatial_scale="whole_brain", cv=5, n_jobs=1),
            domain="predict",
            name=f"whole_brain[{n_images}x{n_voxels}]",
            device="cpu",
            reps=reps,
            params={
                "spatial_scale": "whole_brain",
                "n_samples": n_images,
                "n_voxels": n_voxels,
            },
        )
    )

    n_parcels = 10 if quick else 50
    atlas = _make_atlas(bd.mask, n_parcels)
    results.append(
        benchmark(
            lambda: bd.predict(
                y=y, spatial_scale="roi", roi_mask=atlas, cv=5, n_jobs=1
            ),
            domain="predict",
            name=f"roi[{n_parcels}parcels]",
            device="cpu",
            reps=reps,
            params={
                "spatial_scale": "roi",
                "n_samples": n_images,
                "n_voxels": n_voxels,
                "n_parcels": n_parcels,
            },
        )
    )

    # Searchlight: one model per voxel-sphere -> keep voxel count small.
    sl_images, sl_voxels = (20, 300) if quick else (60, 400)
    bd_sl = make_braindata(sl_images, sl_voxels)
    y_sl = make_labels(sl_images, n_classes=2)
    results.append(
        benchmark(
            lambda: bd_sl.predict(
                y=y_sl, spatial_scale="searchlight", radius_mm=8.0, cv=3, n_jobs=-1
            ),
            domain="predict",
            name=f"searchlight[{sl_images}x{sl_voxels}]",
            device="cpu",
            reps=max(1, reps - 1),
            warmup=0,
            params={
                "spatial_scale": "searchlight",
                "n_samples": sl_images,
                "n_voxels": sl_voxels,
                "radius_mm": 8.0,
            },
        )
    )
    return results
