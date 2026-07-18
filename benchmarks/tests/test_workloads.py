"""Smoke tests for the benchmark workload factories."""

from __future__ import annotations

import numpy as np

from benchmarks.workloads import (
    make_braindata,
    make_labels,
    make_mask,
    make_ondisk_subjects,
    make_regression_arrays,
)


def test_make_mask_exact_voxel_count():
    mask = make_mask(2000)
    assert int(np.asarray(mask.dataobj).sum()) == 2000


def test_make_regression_arrays_shapes():
    x, y = make_regression_arrays(n_samples=100, n_voxels=5000, n_features=20)
    assert x.shape == (100, 20)
    assert y.shape == (100, 5000)


def test_make_braindata_shape_matches_mask():
    bd = make_braindata(n_images=6, n_voxels=1500)
    assert bd.data.shape == (6, 1500)


def test_make_labels_range():
    labels = make_labels(50, n_classes=2)
    assert labels.shape == (50,)
    assert set(np.unique(labels)).issubset({0, 1})


def test_make_ondisk_subjects_loads_into_collection(tmp_path):
    from nltools.data import BrainCollection

    paths, mask_path = make_ondisk_subjects(
        tmp_path, n_subjects=3, n_images=8, n_voxels=500
    )
    assert len(paths) == 3
    assert all(p.exists() for p in paths)

    bc = BrainCollection(
        [str(p) for p in paths], mask=str(mask_path), lazy=True, cache_dir=None
    )
    assert bc.n_subjects == 3
    # A masked load yields (n_images, n_voxels) per subject.
    assert bc[0].data.shape == (8, 500)
