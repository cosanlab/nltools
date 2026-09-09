"""Ridge-regression benchmarks: the estimator + BrainData facade, CPU vs GPU.

Neuroimaging convention: ``X`` is the design matrix ``(n_samples, n_features)``
and ``y`` is brain data ``(n_samples, n_voxels)`` — ridge predicts every voxel
from the shared design. GPU (MPS/CUDA) wins when ``n_voxels`` is large because
the SVD solve is shared across voxels and the per-alpha work vectorizes.
"""

from __future__ import annotations

import numpy as np

from benchmarks.harness import BenchResult, benchmark, gpu_device
from benchmarks.workloads import make_braindata, make_regression_arrays

# (n_samples, n_voxels, n_features): 500=task fMRI, 1000=naturalistic; 50k=3mm, 230k=2mm.
SIZES_FULL = [
    (500, 20_000, 50),
    (1000, 20_000, 100),
]
SIZES_QUICK = [(200, 2_000, 20)]


# 2mm (~230k voxels) is GPU territory — CPU there runs minutes per condition.
# The GPU leg passes Ridge's device="gpu"; the harness device string
# ("cuda"/"mps"/None) is resolved per host by gpu_device().
def _backends() -> list[tuple[str, str]]:
    """(Ridge device=, harness device=) pairs — GPU leg included iff present."""
    backends = [("cpu", "cpu")]
    gpu = gpu_device()
    if gpu is not None:
        backends.append(("gpu", gpu))
    return backends


def run(reps: int = 3, quick: bool = False) -> list[BenchResult]:
    from nltools.models import Ridge

    sizes = SIZES_QUICK if quick else SIZES_FULL
    results: list[BenchResult] = []

    for n_samples, n_voxels, n_features in sizes:
        x, y = make_regression_arrays(n_samples, n_voxels, n_features)
        for ridge_device, device in _backends():
            params = {
                "n_samples": n_samples,
                "n_voxels": n_voxels,
                "n_features": n_features,
                "cv": 5,
                "backend": ridge_device,
            }
            results.append(
                benchmark(
                    lambda d=ridge_device: Ridge(
                        alpha=[0.1, 1.0, 10.0], cv=5, device=d
                    ).fit(x, y),
                    domain="ridge",
                    name=f"Ridge.fit[{n_samples}x{n_voxels}f{n_features}]",
                    device=device,
                    reps=reps,
                    params=params,
                )
            )

    # Facade path: BrainData.fit(model='ridge') — same math through the class glue.
    n_images, n_vox_facade, n_feat = (6, 2_000, 20) if quick else (200, 20_000, 50)
    bd = make_braindata(n_images, n_vox_facade)
    design = (
        np.random.default_rng(0).standard_normal((n_images, n_feat)).astype(np.float32)
    )
    results.append(
        benchmark(
            lambda: bd.fit(
                model="ridge", X=design, alpha=[0.1, 1.0, 10.0], cv=5, inplace=False
            ),
            domain="ridge",
            name=f"BrainData.fit[ridge,{n_images}x{n_vox_facade}]",
            device="cpu",
            reps=reps,
            params={
                "n_samples": n_images,
                "n_voxels": n_vox_facade,
                "n_features": n_feat,
                "cv": 5,
                "backend": "facade",
            },
        )
    )
    return results
