"""Spatial RSA computes ordinary stacks with explicit external mapping."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from nltools.data import Adjacency, BrainData


def _atlas_for(bd, n_rois=2):
    """Return a labeled atlas Nifti aligned with bd.mask, partitioning the
    masked voxels into n_rois contiguous chunks."""
    mask_data = bd.mask.get_fdata().astype(bool)
    n_voxels = int(mask_data.sum())
    flat = np.zeros(n_voxels, dtype=np.int16)
    chunk = max(1, n_voxels // n_rois)
    for k in range(n_rois):
        flat[k * chunk : (k + 1) * chunk] = k + 1
    flat[n_rois * chunk :] = n_rois  # any remainder → last parcel
    out = np.zeros(mask_data.shape, dtype=np.int16)
    out[mask_data] = flat
    return nib.Nifti1Image(out, bd.mask.affine, bd.mask.header)


class TestDistanceWholeBrain:
    def test_default_unchanged(self, minimal_brain_data):
        # spatial_scale defaults to 'whole_brain' → existing behavior:
        # a single Adjacency over images.
        result = minimal_brain_data.distance(metric="euclidean")
        assert isinstance(result, Adjacency)
        assert result.is_single_matrix
        assert not hasattr(result, "spatial_scale")


class TestDistanceROI:
    def test_returns_plain_stack(self, minimal_brain_data):
        atlas = _atlas_for(minimal_brain_data, n_rois=2)
        result = minimal_brain_data.distance(
            metric="correlation", spatial_scale="roi", roi_mask=atlas
        )
        assert isinstance(result, Adjacency)
        assert not result.is_single_matrix
        assert len(result) == 2  # one RDM per parcel
        assert not hasattr(result, "spatial_scale")

    def test_per_parcel_rdm_matches_manual(self, minimal_brain_data):
        from nilearn.masking import apply_mask
        from sklearn.metrics import pairwise_distances

        atlas = _atlas_for(minimal_brain_data, n_rois=2)
        result = minimal_brain_data.distance(
            metric="correlation", spatial_scale="roi", roi_mask=atlas
        )
        # Compute the expected RDM for parcel 1 manually.
        label_vec = apply_mask(atlas, minimal_brain_data.mask).astype(int)
        cols = label_vec == 1
        expected = pairwise_distances(
            minimal_brain_data.data[:, cols], metric="correlation"
        )
        # The first stacked matrix corresponds to parcel 1 (sorted labels).
        np.testing.assert_allclose(result[0].squareform(), expected, atol=1e-10)

    def test_requires_roi_mask(self, minimal_brain_data):
        with pytest.raises(ValueError, match="roi_mask"):
            minimal_brain_data.distance(metric="correlation", spatial_scale="roi")


class TestDistanceSearchlight:
    """Searchlight stacks follow the source mask voxel order."""

    def test_returns_stack_per_voxel_center(self, minimal_brain_data):
        result = minimal_brain_data.distance(
            metric="correlation",
            spatial_scale="searchlight",
            radius=10.0,
        )
        assert isinstance(result, Adjacency)
        assert not result.is_single_matrix
        # Minimal mask has 5 voxels — expect 5 searchlight RDMs.
        n_voxels = minimal_brain_data.shape[-1]
        assert len(result) == n_voxels
        assert not hasattr(result, "spatial_scale")

    def test_per_center_rdm_matches_manual(self, minimal_brain_data):
        from nltools.algorithms.neighborhoods import (
            compute_searchlight_neighborhoods,
        )
        from sklearn.metrics import pairwise_distances

        radius = 10.0
        result = minimal_brain_data.distance(
            metric="correlation",
            spatial_scale="searchlight",
            radius=radius,
        )
        nbrs = compute_searchlight_neighborhoods(minimal_brain_data.mask, radius=radius)
        # Spot-check the first center.
        center0_neighbors = nbrs.get_neighbors(0)
        expected = pairwise_distances(
            minimal_brain_data.data[:, center0_neighbors], metric="correlation"
        )
        np.testing.assert_allclose(result[0].squareform(), expected, atol=1e-10)


class TestDistanceInvalidScale:
    def test_unknown_scale_errors(self, minimal_brain_data):
        with pytest.raises(ValueError, match="spatial_scale"):
            minimal_brain_data.distance(spatial_scale="bogus")


class TestAlignROI:
    """Per-parcel functional alignment: each parcel aligned independently,
    transformed data stitched back to voxel space, transforms kept as a
    dict keyed by atlas label (per-parcel matrices don't reassemble)."""

    def test_per_parcel_transformed_matches_manual(self, minimal_brain_data):
        from nilearn.masking import apply_mask

        atlas = _atlas_for(minimal_brain_data, n_rois=2)
        # Manually align parcel 1 and check it matches the stitched result.
        label_vec = apply_mask(atlas, minimal_brain_data.mask).astype(int)
        cols = label_vec == 1
        parcel_mask = nib.Nifti1Image(
            (atlas.get_fdata() == 1).astype(np.float32), atlas.affine
        )
        sub = minimal_brain_data.apply_mask(parcel_mask)
        manual = sub.align(sub, method="procrustes")

        out = minimal_brain_data.align(
            minimal_brain_data,
            method="procrustes",
            spatial_scale="roi",
            roi_mask=atlas,
        )
        np.testing.assert_allclose(
            out["transformed"].data[:, cols],
            manual["transformed"].data,
            atol=1e-10,
        )


def test_roi_order_and_selected_mapping(minimal_brain_data):
    from nilearn.masking import apply_mask
    from scipy.spatial.distance import cdist
    from nltools.mask import roi_to_brain_from_atlas

    atlas = _atlas_for(minimal_brain_data, n_rois=2)
    values = atlas.get_fdata().astype(np.int16)
    values[values == 1] = 9
    values[values == 2] = 3
    atlas = nib.Nifti1Image(values, atlas.affine)
    label_vec = apply_mask(atlas, minimal_brain_data.mask).astype(int)
    roi_labels = np.unique(label_vec[label_vec != 0])
    rdms = minimal_brain_data.distance(
        metric="euclidean", spatial_scale="roi", roi_mask=atlas
    )
    for index, label in enumerate(roi_labels):
        reference = cdist(
            minimal_brain_data.data[:, label_vec == label],
            minimal_brain_data.data[:, label_vec == label],
        )
        np.testing.assert_allclose(rdms[index].squareform(), reference)
    selection = [1]
    selected = rdms[selection]
    per_roi = selected.mean(axis=1)
    painted = roi_to_brain_from_atlas(
        per_roi,
        atlas=atlas,
        source_mask=minimal_brain_data.mask,
        roi_labels=roi_labels[selection],
    )
    np.testing.assert_allclose(painted.data[label_vec == 9], per_roi[0])
    assert np.isnan(painted.data[label_vec == 3]).all()


class TestRadiusKeyword:
    """Searchlight facades take nilearn's `radius` (millimeters), not `radius_mm`."""

    def test_radius_mm_is_not_a_parameter(self):
        import inspect

        parameters = inspect.signature(BrainData.distance).parameters
        assert "radius" in parameters
        assert "radius_mm" not in parameters
