"""Smoke tests for the benchmark workload factories."""

from __future__ import annotations

import numpy as np

from benchmarks.workloads import (
    make_braindata,
    make_labels,
    make_mask,
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
