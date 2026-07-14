"""Tests for BrainData spatial_scale dispatch (RSA workflow).

Covers BrainData.distance(spatial_scale='roi'|'searchlight'|'whole_brain')
producing Adjacency results carrying SpatialScale provenance suitable for
``Adjacency.to_brain()`` / ``Adjacency.similarity(project=True)``.
"""

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
        assert result.spatial_scale is None

    def test_explicit_whole_brain(self, minimal_brain_data):
        result = minimal_brain_data.distance(
            metric="euclidean", spatial_scale="whole_brain"
        )
        assert isinstance(result, Adjacency)
        assert result.is_single_matrix
        assert result.spatial_scale is None


class TestDistanceROI:
    def test_returns_stack_with_spatial_scale(self, minimal_brain_data):
        atlas = _atlas_for(minimal_brain_data, n_rois=2)
        result = minimal_brain_data.distance(
            metric="correlation", spatial_scale="roi", roi_mask=atlas
        )
        assert isinstance(result, Adjacency)
        assert not result.is_single_matrix
        assert len(result) == 2  # one RDM per parcel
        ss = result.spatial_scale
        assert ss is not None
        assert ss.kind == "roi"
        assert list(ss.roi_labels) == [1, 2]

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

    def test_chain_to_brain_round_trip(self, minimal_brain_data):
        """End-to-end: per-ROI distance → user reduction → voxel map."""
        atlas = _atlas_for(minimal_brain_data, n_rois=2)
        rdms = minimal_brain_data.distance(
            metric="correlation", spatial_scale="roi", roi_mask=atlas
        )
        # Stand in for a model-similarity reduction: take the per-RDM mean.
        per_roi = np.array([float(rdm.data.mean()) for rdm in rdms])
        brain_map = rdms.to_brain(per_roi)
        assert isinstance(brain_map, BrainData)
        assert brain_map.shape[-1] == minimal_brain_data.shape[-1]


class TestDistanceSearchlight:
    """Per-voxel-center searchlight distance — one RDM per center, with
    SpatialScale(kind='searchlight') and a synthetic 1-voxel-per-label
    atlas so that to_brain paints the scalar to that center voxel."""

    def test_returns_stack_per_voxel_center(self, minimal_brain_data):
        result = minimal_brain_data.distance(
            metric="correlation",
            spatial_scale="searchlight",
            radius_mm=10.0,
        )
        assert isinstance(result, Adjacency)
        assert not result.is_single_matrix
        # Minimal mask has 5 voxels — expect 5 searchlight RDMs.
        n_voxels = minimal_brain_data.shape[-1]
        assert len(result) == n_voxels
        ss = result.spatial_scale
        assert ss is not None
        assert ss.kind == "searchlight"
        assert len(ss.roi_labels) == n_voxels

    def test_per_center_rdm_matches_manual(self, minimal_brain_data):
        from nltools.data.braindata.neighborhoods import (
            compute_searchlight_neighborhoods,
        )
        from sklearn.metrics import pairwise_distances

        radius_mm = 10.0
        result = minimal_brain_data.distance(
            metric="correlation",
            spatial_scale="searchlight",
            radius_mm=radius_mm,
        )
        nbrs = compute_searchlight_neighborhoods(
            minimal_brain_data.mask, radius_mm=radius_mm, use_cache=False
        )
        # Spot-check the first center.
        center0_neighbors = nbrs.get_neighbors(0)
        expected = pairwise_distances(
            minimal_brain_data.data[:, center0_neighbors], metric="correlation"
        )
        np.testing.assert_allclose(result[0].squareform(), expected, atol=1e-10)

    def test_chain_to_brain_round_trip(self, minimal_brain_data):
        """Per-center RSA → similarity to model → voxel map — same RSA chain
        as the ROI case but each searchlight paints to its center voxel."""
        rdms = minimal_brain_data.distance(
            metric="correlation",
            spatial_scale="searchlight",
            radius_mm=10.0,
        )
        n = minimal_brain_data.shape[0]
        rng = np.random.default_rng(0)
        model = Adjacency(np.abs(rng.standard_normal((n, n))), matrix_type="distance")
        brain_map = rdms.similarity(
            model, project=True, permutation_method=None, metric="spearman"
        )
        assert isinstance(brain_map, BrainData)
        assert brain_map.shape[-1] == minimal_brain_data.shape[-1]


class TestDistanceInvalidScale:
    def test_unknown_scale_errors(self, minimal_brain_data):
        with pytest.raises(ValueError, match="spatial_scale"):
            minimal_brain_data.distance(spatial_scale="bogus")


class TestReductionsROI:
    """``.mean(spatial_scale='roi') / .std / .median`` replace each voxel's
    value with its parcel's reduction (parcellation smoothing per image)."""

    def test_mean_roi_paints_per_parcel_means(self, minimal_brain_data):
        from nilearn.masking import apply_mask

        atlas = _atlas_for(minimal_brain_data, n_rois=2)
        out = minimal_brain_data.mean(spatial_scale="roi", roi_mask=atlas)
        assert isinstance(out, BrainData)
        # Output preserves image-by-voxel shape: each voxel painted with
        # its parcel's mean for that image.
        assert out.shape == minimal_brain_data.shape

        label_vec = apply_mask(atlas, minimal_brain_data.mask).astype(int)
        for label in (1, 2):
            cols = label_vec == label
            expected = minimal_brain_data.data[:, cols].mean(axis=1)
            np.testing.assert_allclose(
                out.data[:, cols], expected[:, None].repeat(cols.sum(), axis=1)
            )

    def test_std_roi(self, minimal_brain_data):
        from nilearn.masking import apply_mask

        atlas = _atlas_for(minimal_brain_data, n_rois=2)
        out = minimal_brain_data.std(spatial_scale="roi", roi_mask=atlas)
        label_vec = apply_mask(atlas, minimal_brain_data.mask).astype(int)
        cols = label_vec == 1
        expected = minimal_brain_data.data[:, cols].std(axis=1)
        np.testing.assert_allclose(
            out.data[:, cols], expected[:, None].repeat(cols.sum(), axis=1)
        )

    def test_median_roi(self, minimal_brain_data):
        from nilearn.masking import apply_mask

        atlas = _atlas_for(minimal_brain_data, n_rois=2)
        out = minimal_brain_data.median(spatial_scale="roi", roi_mask=atlas)
        label_vec = apply_mask(atlas, minimal_brain_data.mask).astype(int)
        cols = label_vec == 1
        expected = np.median(minimal_brain_data.data[:, cols], axis=1)
        np.testing.assert_allclose(
            out.data[:, cols], expected[:, None].repeat(cols.sum(), axis=1)
        )

    def test_mean_whole_brain_unchanged(self, minimal_brain_data):
        # Default spatial_scale='whole_brain' must preserve existing behavior.
        out = minimal_brain_data.mean()
        # Existing behavior returns a BrainData of voxel-axis means across images.
        assert isinstance(out, BrainData)

    def test_align_searchlight_not_implemented(self, minimal_brain_data):
        with pytest.raises(NotImplementedError, match="overlap"):
            minimal_brain_data.align(
                minimal_brain_data,
                spatial_scale="searchlight",
                radius_mm=10.0,
            )


class TestAlignROI:
    """Per-parcel functional alignment: each parcel aligned independently,
    transformed data stitched back to voxel space, transforms kept as a
    dict keyed by atlas label (per-parcel matrices don't reassemble)."""

    def test_returns_dict_with_per_parcel_fields(self, minimal_brain_data):
        atlas = _atlas_for(minimal_brain_data, n_rois=2)
        out = minimal_brain_data.align(
            minimal_brain_data,
            method="procrustes",
            spatial_scale="roi",
            roi_mask=atlas,
        )
        assert isinstance(out, dict)
        # Stitched transformed data → voxel-space BrainData of original shape.
        assert isinstance(out["transformed"], BrainData)
        assert out["transformed"].shape == minimal_brain_data.shape
        # Per-parcel transforms: dict keyed by atlas label.
        assert isinstance(out["transformation_matrix"], dict)
        assert set(out["transformation_matrix"].keys()) == {1, 2}
        # roi_labels ndarray in stack order.
        assert list(out["roi_labels"]) == [1, 2]
        # disparity / scale: per-parcel arrays.
        assert out["disparity"].shape == (2,)
        assert out["scale"].shape == (2,)

    def test_per_parcel_transformed_matches_manual(self, minimal_brain_data):
        from nilearn.masking import apply_mask

        atlas = _atlas_for(minimal_brain_data, n_rois=2)
        # Manually align parcel 1 and check it matches the stitched result.
        label_vec = apply_mask(atlas, minimal_brain_data.mask).astype(int)
        cols = label_vec == 1
        sub = minimal_brain_data.copy()
        sub.data = minimal_brain_data.data[:, cols]
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


class TestSimilarityProject:
    """Adjacency.similarity(project=True) returns a voxel-space BrainData
    by paying out per-matrix similarity scores via to_brain()."""

    def test_project_true_returns_braindata(self, minimal_brain_data):
        atlas = _atlas_for(minimal_brain_data, n_rois=2)
        rdms = minimal_brain_data.distance(
            metric="correlation", spatial_scale="roi", roi_mask=atlas
        )
        # Build a model RDM matching the per-parcel n_nodes (n_images here).
        n = minimal_brain_data.shape[0]
        rng = np.random.default_rng(0)
        model = Adjacency(
            np.abs(rng.standard_normal((n, n))) + np.eye(n) * 0,
            matrix_type="distance",
        )
        out = rdms.similarity(
            model, project=True, permutation_method=None, metric="spearman"
        )
        assert isinstance(out, BrainData)
        assert out.shape[-1] == minimal_brain_data.shape[-1]

    def test_project_true_matches_manual_to_brain(self, minimal_brain_data):
        atlas = _atlas_for(minimal_brain_data, n_rois=2)
        rdms = minimal_brain_data.distance(
            metric="correlation", spatial_scale="roi", roi_mask=atlas
        )
        n = minimal_brain_data.shape[0]
        rng = np.random.default_rng(0)
        model = Adjacency(np.abs(rng.standard_normal((n, n))), matrix_type="distance")
        results = rdms.similarity(
            model, project=False, permutation_method=None, metric="spearman"
        )
        per_roi = np.array([r["correlation"] for r in results])
        manual = rdms.to_brain(per_roi)
        sugared = rdms.similarity(
            model, project=True, permutation_method=None, metric="spearman"
        )
        np.testing.assert_array_equal(np.asarray(manual.data), np.asarray(sugared.data))

    def test_project_true_errors_without_spatial_scale(self):
        # A stacked Adjacency without provenance can't project to voxel space.
        from sklearn.metrics import pairwise_distances

        rng = np.random.default_rng(0)
        mats = [
            pairwise_distances(rng.standard_normal((5, 4)), metric="correlation")
            for _ in range(3)
        ]
        rdms = Adjacency(mats, matrix_type="distance")
        model = Adjacency(mats[0], matrix_type="distance")
        with pytest.raises(ValueError, match="spatial_scale"):
            rdms.similarity(model, project=True, permutation_method=None)
