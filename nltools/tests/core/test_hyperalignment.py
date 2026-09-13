"""Tests for `_hyperalign`, the Procrustes template loop behind `align`.

Hyperalignment (Haxby et al., 2011) builds a common template, refines it over
`n_iter` rounds of align-and-average, then aligns every subject to the refined
template. `align(method='procrustes')` is the only caller.
"""

import numpy as np
import pytest

from nltools.algorithms.alignment.procrustes import _hyperalign, align


@pytest.fixture(scope="module")
def equal_sized_subjects():
    """Three subjects of 50 features x 20 samples."""
    rng = np.random.default_rng(42)
    return [rng.standard_normal((50, 20)) for _ in range(3)]


@pytest.fixture(scope="module")
def different_sized_subjects():
    """Three subjects whose feature counts differ: 50, 45, 52."""
    rng = np.random.default_rng(42)
    return [
        rng.standard_normal((50, 20)),
        rng.standard_normal((45, 20)),
        rng.standard_normal((52, 20)),
    ]


def test_fitted_output_properties(equal_sized_subjects):
    """Shapes, per-subject counts, and orthogonality of the fitted transforms."""
    n_features, n_samples = equal_sized_subjects[0].shape

    aligned, matrices, template, disparity, scale = _hyperalign(
        equal_sized_subjects, n_iter=2
    )

    assert len(aligned) == len(equal_sized_subjects)
    assert len(matrices) == len(equal_sized_subjects)
    assert len(disparity) == len(equal_sized_subjects)
    assert len(scale) == len(equal_sized_subjects)
    assert template.shape == (n_samples, n_features)
    for subject, (values, matrix) in enumerate(zip(aligned, matrices)):
        assert values.shape == (n_features, n_samples)
        assert matrix.shape == (n_features, n_features)
        np.testing.assert_almost_equal(
            matrix @ matrix.T,
            np.eye(n_features),
            decimal=5,
            err_msg=f"subject {subject} transform is not orthogonal",
        )


def test_identical_subjects_align_almost_perfectly():
    """Subjects that are already identical have near-zero disparity."""
    base = np.random.default_rng(42).standard_normal((50, 20))

    _, _, _, disparity, _ = _hyperalign([base.copy() for _ in range(3)], n_iter=2)

    assert all(value < 0.1 for value in disparity)


def test_mismatched_samples_raises():
    """Subjects must share a sample count; the feature axis is the padded one."""
    rng = np.random.default_rng(0)
    data = [rng.standard_normal((50, 20)), rng.standard_normal((50, 25))]

    with pytest.raises(ValueError, match="same number of samples"):
        _hyperalign(data, n_iter=1)


def test_zero_pads_features_to_max_not_truncates(different_sized_subjects):
    """F001: the feature axis is zero-padded up to the LARGEST subject.

    v0.5.1 truncated every subject to the smallest feature count, silently
    dropping data. The common space must be 52-dimensional, not 45.
    """
    max_features = max(x.shape[0] for x in different_sized_subjects)
    n_samples = different_sized_subjects[0].shape[1]
    assert max_features == 52  # guard the fixture assumption

    _, matrices, template, _, _ = _hyperalign(different_sized_subjects, n_iter=1)

    assert template.shape == (n_samples, max_features), (
        f"expected zero-padding to {max_features} features, got {template.shape}"
    )
    for subject, matrix in enumerate(matrices):
        assert matrix.shape == (max_features, max_features), (
            f"subject {subject} transform {matrix.shape} was truncated below "
            f"{max_features} features"
        )


def test_matches_align_procrustes():
    """`align(method='procrustes')` is wired to `_hyperalign(..., n_iter=1)`.

    A wiring test, not a numerical one: `align`'s procrustes branch is a
    two-line unpack of this function, and the numbers themselves are pinned by
    `test_seeded_regression`. `align` takes (observations, features) and
    transposes internally, so the function is called here on the transposed
    data it would build.
    """
    rng = np.random.default_rng(42)
    subjects = [rng.standard_normal((50, 20)) for _ in range(5)]

    aligned, matrices, template, disparity, scale = _hyperalign(
        [x.T for x in subjects], n_iter=1
    )
    out = align(subjects, method="procrustes")

    for subject in range(len(subjects)):
        np.testing.assert_allclose(aligned[subject], out["transformed"][subject])
        np.testing.assert_allclose(
            matrices[subject], out["transformation_matrix"][subject]
        )
        np.testing.assert_allclose(disparity[subject], out["disparity"][subject])
        np.testing.assert_allclose(scale[subject], out["scale"][subject])
    np.testing.assert_allclose(template, out["common_model"])
