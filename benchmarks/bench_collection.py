"""BrainCollection benchmarks: the memory-critical path across N subjects.

This is the benchmark the others can't express: BrainCollection's whole reason
for existing is running whole-brain ops over many subjects without holding them
all in RAM. The headline comparison is a reduction (`.mean()`) over N on-disk
subjects in two modes:

- **lazy** (path-backed) — subjects stream through; peak RSS should stay ~flat
  as N grows.
- **in-memory** (`lazy=False`) — every subject resident at once; peak RSS scales
  with N.

Plus a parallel-execution leg (`.apply` at `n_jobs=1` vs `-1`) for the speed
side of the story.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from benchmarks.harness import BenchResult, benchmark
from benchmarks.workloads import make_ondisk_subjects

N_SUBJECTS_FULL = [20, 50]
N_SUBJECTS_QUICK = [3, 6]
# ~6 MB/subject (20k voxels x 80 images x 4B) so the in-memory path visibly holds
# all N at once (~300 MB at N=50) while lazy streaming stays flat. Quick mode
# stays tiny.
N_IMAGES = 80
N_VOXELS = 20_000


def run(reps: int = 2, quick: bool = False) -> list[BenchResult]:
    from nltools.data import BrainCollection

    subject_counts = N_SUBJECTS_QUICK if quick else N_SUBJECTS_FULL
    n_images, n_voxels = (20, 2_000) if quick else (N_IMAGES, N_VOXELS)
    results: list[BenchResult] = []
    tmp = Path(tempfile.mkdtemp(prefix="nltools-bench-collection-"))

    try:
        for n_subjects in subject_counts:
            data_dir = tmp / f"n{n_subjects}"
            paths, mask_path = make_ondisk_subjects(
                data_dir, n_subjects, n_images, n_voxels
            )
            paths = [str(p) for p in paths]
            mask_path = str(mask_path)

            # --- Memory headline: lazy vs in-memory reduction over N subjects ---
            def _lazy_mean(n=n_subjects, p=paths, m=mask_path, d=data_dir):
                bc = BrainCollection(p, mask=m, lazy=True, cache_dir=str(d / ".cache"))
                bc.mean()
                bc.cleanup()

            results.append(
                benchmark(
                    _lazy_mean,
                    domain="collection",
                    name=f"mean[lazy,N={n_subjects}]",
                    device="cpu",
                    reps=reps,
                    params={
                        "mode": "lazy",
                        "op": "mean",
                        "n_subjects": n_subjects,
                        "n_images": n_images,
                        "n_voxels": n_voxels,
                    },
                )
            )

            def _inmem_mean(p=paths, m=mask_path):
                bc = BrainCollection(p, mask=m, lazy=False, cache_dir=None)
                bc.mean()

            results.append(
                benchmark(
                    _inmem_mean,
                    domain="collection",
                    name=f"mean[in_memory,N={n_subjects}]",
                    device="cpu",
                    reps=reps,
                    params={
                        "mode": "in_memory",
                        "op": "mean",
                        "n_subjects": n_subjects,
                        "n_images": n_images,
                        "n_voxels": n_voxels,
                    },
                )
            )

            # --- Parallel execution: n_jobs scaling on a per-subject op ---
            for n_jobs in (1, -1):

                def _apply(n_jobs=n_jobs, p=paths, m=mask_path, d=data_dir):
                    bc = BrainCollection(
                        p, mask=m, lazy=True, cache_dir=str(d / f".cache_j{n_jobs}")
                    )
                    bc.apply("standardize", n_jobs=n_jobs, cache=False)
                    bc.cleanup()

                results.append(
                    benchmark(
                        _apply,
                        domain="collection",
                        name=f"apply[standardize,N={n_subjects},n_jobs={n_jobs}]",
                        device="cpu",
                        reps=reps,
                        params={
                            "mode": "lazy",
                            "op": "apply:standardize",
                            "n_subjects": n_subjects,
                            "n_images": n_images,
                            "n_voxels": n_voxels,
                            "n_jobs": n_jobs,
                        },
                    )
                )
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    return results
