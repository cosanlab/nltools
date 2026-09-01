"""Permutation-test benchmarks: one-sample / two-sample / correlation, CPU vs GPU.

Realistic shape is ``(n_subjects, n_voxels)`` — each voxel is an independent
test and the permutations vectorize across voxels, which is exactly where the
GPU backend pays off. Cost scales with ``n_permute`` (1k–10k), so that is the
primary swept dimension.
"""

from __future__ import annotations

from benchmarks.harness import BenchResult, benchmark, gpu_device
from benchmarks.workloads import make_group_maps

N_SUBJECTS = 30
N_VOXELS_FULL = 5_000
N_VOXELS_QUICK = 300
N_PERMUTE_FULL = [1_000, 3_000]
N_PERMUTE_QUICK = [200]


def _backends() -> list[tuple[str, str]]:
    """(inference device=, harness device=) pairs — GPU leg included iff present."""
    backends = [("cpu", "cpu")]
    gpu = gpu_device()
    if gpu is not None:
        backends.append(("gpu", gpu))
    return backends


def run(reps: int = 3, quick: bool = False) -> list[BenchResult]:
    from nltools.algorithms.inference import (
        correlation_permutation_test,
        one_sample_permutation_test,
        two_sample_permutation_test,
    )

    n_voxels = N_VOXELS_QUICK if quick else N_VOXELS_FULL
    n_permutes = N_PERMUTE_QUICK if quick else N_PERMUTE_FULL

    data = make_group_maps(N_SUBJECTS, n_voxels, seed=0)
    data2 = make_group_maps(N_SUBJECTS, n_voxels, seed=1)
    # Correlation is a per-vector test; use one column pair scaled by n_permute.
    vec1, vec2 = data[:, 0], data2[:, 0]

    cases = [
        (
            "one_sample",
            lambda p, n: one_sample_permutation_test(
                data, n_permute=n, device=p, random_state=0
            ),
        ),
        (
            "two_sample",
            lambda p, n: two_sample_permutation_test(
                data, data2, n_permute=n, device=p, random_state=0
            ),
        ),
        (
            "correlation",
            lambda p, n: correlation_permutation_test(
                vec1, vec2, n_permute=n, device=p, random_state=0
            ),
        ),
    ]

    results: list[BenchResult] = []
    for name, fn in cases:
        for n_permute in n_permutes:
            for api_device, harness_device in _backends():

                def _call(fn=fn, p=api_device, n=n_permute):
                    fn(p, n)

                results.append(
                    benchmark(
                        _call,
                        domain="inference",
                        name=f"{name}[perm={n_permute}]",
                        device=harness_device,
                        reps=reps,
                        params={
                            "test": name,
                            "n_subjects": N_SUBJECTS,
                            "n_voxels": n_voxels,
                            "n_permute": n_permute,
                            "backend": api_device,
                        },
                    )
                )
    return results
