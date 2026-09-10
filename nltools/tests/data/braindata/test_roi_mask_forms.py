"""Cross-cutting contract: which ``roi_mask=`` forms every ROI entry point accepts.

``spatial_scale='roi'`` is offered by `BrainData.predict`, `BrainData.distance`,
`BrainData.mean` / `.std` / `.median`, and `BrainData.align`. Each takes an
``roi_mask=`` atlas, and users build those atlases with nltools' own helpers —
`expand_mask` returns a stacked binary `BrainData`, and loading an atlas with
`BrainData(path, mask=data.mask)` returns a label vector. Every entry point must
accept all four idiomatic forms and resolve them to the same parcellation.

Regression for the four-way duplication of the atlas-resolution block, where
`predict` alone dropped the `BrainData` branch and no entry point handled a
stacked/4-D expanded mask.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from nltools.data import BrainData
from nltools.mask import expand_mask

N_ROIS = 2


@pytest.fixture
def atlas_nifti(minimal_brain_data):
    """3-D integer label image aligned with the fixture's mask (the baseline form)."""
    mask_data = minimal_brain_data.mask.get_fdata().astype(bool)
    n_voxels = int(mask_data.sum())
    flat = np.zeros(n_voxels, dtype=np.int16)
    chunk = max(1, n_voxels // N_ROIS)
    for k in range(N_ROIS):
        flat[k * chunk : (k + 1) * chunk] = k + 1
    flat[N_ROIS * chunk :] = N_ROIS
    out = np.zeros(mask_data.shape, dtype=np.int16)
    out[mask_data] = flat
    return nib.Nifti1Image(out, minimal_brain_data.mask.affine)


@pytest.fixture
def roi_mask_forms(minimal_brain_data, atlas_nifti, tmp_path):
    """The same parcellation expressed five ways.

    ``label_braindata`` is what ``BrainData(atlas_path, mask=dat.mask)`` gives you;
    ``expanded_braindata`` is what `expand_mask` gives you; ``expanded_nifti`` is
    that stack written to disk as a 4-D image.
    """
    label_bd = BrainData(atlas_nifti, mask=minimal_brain_data.mask)
    expanded_bd = expand_mask(label_bd)
    path = tmp_path / "atlas.nii.gz"
    atlas_nifti.to_filename(path)
    return {
        "label_nifti": atlas_nifti,
        "label_path": str(path),
        "label_braindata": label_bd,
        "expanded_braindata": expanded_bd,
        "expanded_nifti": expanded_bd.to_nifti(),
    }


FORMS = [
    "label_nifti",
    "label_path",
    "label_braindata",
    "expanded_braindata",
    "expanded_nifti",
]


@pytest.fixture
def binary_y(minimal_brain_data):
    n = minimal_brain_data.shape[0]
    return np.array([0] * (n // 2) + [1] * (n - n // 2))


@pytest.mark.parametrize("form", FORMS)
def test_predict_accepts_every_roi_mask_form(
    minimal_brain_data, roi_mask_forms, binary_y, form
):
    """`predict` resolves all five forms to the same parcellation and scores."""
    baseline = minimal_brain_data.predict(
        y=binary_y,
        spatial_scale="roi",
        roi_mask=roi_mask_forms["label_nifti"],
        cv=3,
        n_jobs=1,
        random_state=0,
    )
    result = minimal_brain_data.predict(
        y=binary_y,
        spatial_scale="roi",
        roi_mask=roi_mask_forms[form],
        cv=3,
        n_jobs=1,
        random_state=0,
    )
    assert list(result.roi_labels) == list(baseline.roi_labels) == [1, 2]
    np.testing.assert_allclose(result.mean_score, baseline.mean_score)


@pytest.mark.parametrize("form", FORMS)
def test_distance_accepts_every_roi_mask_form(minimal_brain_data, roi_mask_forms, form):
    """`distance` yields one RDM per parcel regardless of atlas form."""
    baseline = minimal_brain_data.distance(
        metric="correlation",
        spatial_scale="roi",
        roi_mask=roi_mask_forms["label_nifti"],
    )
    result = minimal_brain_data.distance(
        metric="correlation", spatial_scale="roi", roi_mask=roi_mask_forms[form]
    )
    assert len(result) == N_ROIS
    assert list(result.spatial_scale.roi_labels) == [1, 2]
    np.testing.assert_allclose(result[0].squareform(), baseline[0].squareform())


@pytest.mark.parametrize("form", FORMS)
def test_mean_accepts_every_roi_mask_form(minimal_brain_data, roi_mask_forms, form):
    """`mean(spatial_scale='roi')` paints identical parcel means for every form."""
    baseline = minimal_brain_data.mean(
        spatial_scale="roi", roi_mask=roi_mask_forms["label_nifti"]
    )
    result = minimal_brain_data.mean(spatial_scale="roi", roi_mask=roi_mask_forms[form])
    np.testing.assert_allclose(np.nan_to_num(result.data), np.nan_to_num(baseline.data))


def test_expanded_mask_labels_are_sequential_in_stack_order(
    minimal_brain_data, atlas_nifti
):
    """A stacked binary mask carries no label values, so parcels are numbered
    1..n in stack order. Round-tripping a 1..n atlas through `expand_mask`
    therefore preserves the original labels."""
    label_bd = BrainData(atlas_nifti, mask=minimal_brain_data.mask)
    expanded = expand_mask(label_bd)
    result = minimal_brain_data.distance(
        metric="correlation", spatial_scale="roi", roi_mask=expanded
    )
    assert list(result.spatial_scale.roi_labels) == [1, 2]


def test_predict_rejects_missing_roi_mask(minimal_brain_data, binary_y):
    with pytest.raises(ValueError, match="roi_mask"):
        minimal_brain_data.predict(y=binary_y, spatial_scale="roi", cv=3, n_jobs=1)


def test_roi_mask_with_no_labels_in_mask_space_raises(minimal_brain_data, binary_y):
    """An all-zero atlas is a user error worth naming, not an empty result."""
    empty = nib.Nifti1Image(
        np.zeros(minimal_brain_data.mask.shape, dtype=np.int16),
        minimal_brain_data.mask.affine,
    )
    with pytest.raises(ValueError, match="no nonzero labels"):
        minimal_brain_data.predict(
            y=binary_y, spatial_scale="roi", roi_mask=empty, cv=3, n_jobs=1
        )
