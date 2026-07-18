"""Ridge-regression benchmarks: algorithm layer + BrainData facade, CPU vs GPU.

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


# CPU leg uses joblib ('cpu'), not single-threaded 'numpy': multi-core is the
# realistic baseline and keeps whole-brain conditions tractable. 2mm (~230k
# voxels) is GPU-territory — single-threaded CPU there runs minutes/condition.
# The GPU leg passes ridge's device-agnostic parallel="gpu" alias; the harness
# device string ("cuda"/"mps"/None) is resolved per host by gpu_device().
def _backends() -> list[tuple[str, str]]:
    """(ridge parallel=, harness device=) pairs — GPU leg included iff present."""
    backends = [("cpu", "cpu")]
    gpu = gpu_device()
    if gpu is not None:
        backends.append(("gpu", gpu))
    return backends


def run(reps: int = 3, quick: bool = False) -> list[BenchResult]:
    from nltools.algorithms.ridge import ridge_cv

    sizes = SIZES_QUICK if quick else SIZES_FULL
    results: list[BenchResult] = []

    for n_samples, n_voxels, n_features in sizes:
        x, y = make_regression_arrays(n_samples, n_voxels, n_features)
        for parallel, device in _backends():
            params = {
                "n_samples": n_samples,
                "n_voxels": n_voxels,
                "n_features": n_features,
                "cv": 5,
                "backend": parallel,
            }
            results.append(
                benchmark(
                    lambda p=parallel: ridge_cv(x, y, cv=5, parallel=p, random_state=0),
                    domain="ridge",
                    name=f"ridge_cv[{n_samples}x{n_voxels}f{n_features}]",
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
            lambda: bd.fit(model="ridge", X=design, cv=5, inplace=False),
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
