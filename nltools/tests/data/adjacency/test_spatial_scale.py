"""Tests for the spatial-scale interop layer between Adjacency and BrainData.

Covers:
- ``SpatialScale`` dataclass: shape, immutability, validation.
- ``Adjacency.spatial_scale`` optional field: storage, preservation rules.
- ``Adjacency.to_brain()`` back-projection (later slice).
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from nltools.data import Adjacency
from nltools.data.adjacency.spatial import SpatialScale


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_atlas(n_voxels=12, n_rois=3, affine=None):
    """Build a tiny labeled atlas as a (n_voxels-shape) nibabel image plus
    integer roi_labels in stack order."""
    affine = np.eye(4) if affine is None else affine
    # Lay out n_voxels along a line, partition into n_rois contiguous chunks.
    flat = np.zeros(n_voxels, dtype=np.int16)
    chunk = n_voxels // n_rois
    for k in range(n_rois):
        flat[k * chunk : (k + 1) * chunk] = k + 1
    flat[n_rois * chunk :] = n_rois  # any remainder → last parcel
    spatial = (n_voxels, 1, 1)
    arr = flat.reshape(spatial)
    return nib.Nifti1Image(arr, affine), np.arange(1, n_rois + 1)


def _stack_matrices(n_matrices=3, n_nodes=4):
    """Build a list of symmetric distance matrices for Adjacency stack init."""
    from sklearn.metrics import pairwise_distances

    rng = np.random.default_rng(0)
    return [
        pairwise_distances(rng.standard_normal((n_nodes, 5)), metric="correlation")
        for _ in range(n_matrices)
    ]


# ---------------------------------------------------------------------------
# SpatialScale dataclass
# ---------------------------------------------------------------------------


class TestSpatialScaleDataclass:
    def test_constructs_with_valid_fields(self):
        atlas_img, roi_labels = _stub_atlas(n_voxels=12, n_rois=3)
        mask_img = nib.Nifti1Image(
            (atlas_img.get_fdata() > 0).astype(np.int8), atlas_img.affine
        )
        ss = SpatialScale(
            atlas=atlas_img,
            roi_labels=roi_labels,
            source_mask=mask_img,
            kind="roi",
        )
        assert ss.kind == "roi"
        assert ss.roi_labels.tolist() == [1, 2, 3]
        assert ss.atlas is atlas_img
        assert ss.source_mask is mask_img

    def test_is_frozen(self):
        atlas_img, roi_labels = _stub_atlas()
        mask_img = nib.Nifti1Image(np.ones((12, 1, 1), dtype=np.int8), np.eye(4))
        ss = SpatialScale(
            atlas=atlas_img,
            roi_labels=roi_labels,
            source_mask=mask_img,
            kind="roi",
        )
        with pytest.raises(Exception):
            ss.kind = "searchlight"  # frozen → mutation raises

    def test_kind_validates(self):
        atlas_img, roi_labels = _stub_atlas()
        mask_img = nib.Nifti1Image(np.ones((12, 1, 1), dtype=np.int8), np.eye(4))
        with pytest.raises(ValueError, match="kind"):
            SpatialScale(
                atlas=atlas_img,
                roi_labels=roi_labels,
                source_mask=mask_img,
                kind="bogus",
            )

    def test_roi_labels_coerced_to_ndarray(self):
        atlas_img, _ = _stub_atlas(n_voxels=12, n_rois=3)
        mask_img = nib.Nifti1Image(np.ones((12, 1, 1), dtype=np.int8), np.eye(4))
        # Pass a list — should be normalized to a numpy array.
        ss = SpatialScale(
            atlas=atlas_img,
            roi_labels=[1, 2, 3],
            source_mask=mask_img,
            kind="roi",
        )
        assert isinstance(ss.roi_labels, np.ndarray)
        assert ss.roi_labels.dtype.kind in ("i", "u")


# ---------------------------------------------------------------------------
# Adjacency.spatial_scale storage and preservation
# ---------------------------------------------------------------------------


class TestAdjacencySpatialScaleField:
    def _ss(self, n_matrices=3):
        atlas_img, roi_labels = _stub_atlas(n_voxels=12, n_rois=n_matrices)
        mask_img = nib.Nifti1Image(
            (atlas_img.get_fdata() > 0).astype(np.int8), atlas_img.affine
        )
        return SpatialScale(
            atlas=atlas_img,
            roi_labels=roi_labels,
            source_mask=mask_img,
            kind="roi",
        )

    def test_default_is_none(self):
        adj = Adjacency(_stack_matrices(n_matrices=3, n_nodes=4))
        assert adj.spatial_scale is None

    def test_init_accepts_spatial_scale(self):
        ss = self._ss(n_matrices=3)
        adj = Adjacency(_stack_matrices(n_matrices=3, n_nodes=4), spatial_scale=ss)
        assert adj.spatial_scale is ss

    def test_rejects_single_matrix(self):
        # SpatialScale only makes sense on a stack of matrices (one per parcel).
        single_mat = _stack_matrices(n_matrices=1, n_nodes=4)[0]
        ss = self._ss(n_matrices=1)
        with pytest.raises(ValueError, match="stack"):
            Adjacency(single_mat, matrix_type="distance", spatial_scale=ss)

    def test_rejects_length_mismatch(self):
        # roi_labels length must match the number of stacked matrices.
        ss_wrong = self._ss(n_matrices=2)  # 2 ROI labels for a 3-matrix stack
        with pytest.raises(ValueError, match="roi_labels"):
            Adjacency(_stack_matrices(n_matrices=3, n_nodes=4), spatial_scale=ss_wrong)

    def test_preserved_through_copy(self):
        ss = self._ss(n_matrices=3)
        adj = Adjacency(_stack_matrices(n_matrices=3, n_nodes=4), spatial_scale=ss)
        adj2 = adj.copy()
        assert adj2.spatial_scale is not None
        np.testing.assert_array_equal(adj2.spatial_scale.roi_labels, ss.roi_labels)
        assert adj2.spatial_scale.kind == "roi"

    def test_dropped_when_getitem_collapses_to_single(self):
        ss = self._ss(n_matrices=3)
        adj = Adjacency(_stack_matrices(n_matrices=3, n_nodes=4), spatial_scale=ss)
        single = adj[0]
        assert single.is_single_matrix
        assert single.spatial_scale is None


# ---------------------------------------------------------------------------
# Adjacency.to_brain() back-projection
# ---------------------------------------------------------------------------


class TestAdjacencyToBrain:
    def _setup(self, n_rois=3, n_voxels_per_roi=4):
        """Build matched (atlas, mask, SpatialScale, Adjacency stack)."""
        from nltools.data import BrainData

        n_voxels = n_rois * n_voxels_per_roi
        # Atlas: contiguous chunks of n_voxels_per_roi labeled 1..n_rois.
        affine = np.eye(4)
        flat = np.zeros(n_voxels, dtype=np.int16)
        for k in range(n_rois):
            flat[k * n_voxels_per_roi : (k + 1) * n_voxels_per_roi] = k + 1
        spatial = (n_voxels, 1, 1)
        atlas_img = nib.Nifti1Image(flat.reshape(spatial), affine)
        # Mask covers every atlas voxel (so apply_mask gives length n_voxels).
        mask_img = nib.Nifti1Image(np.ones(spatial, dtype=np.int8), affine)
        atlas_bd = BrainData(atlas_img, mask=mask_img)
        ss = SpatialScale(
            atlas=atlas_bd,
            roi_labels=np.arange(1, n_rois + 1),
            source_mask=mask_img,
            kind="roi",
        )
        adj = Adjacency(_stack_matrices(n_matrices=n_rois, n_nodes=4), spatial_scale=ss)
        return adj, mask_img, n_voxels, n_voxels_per_roi

    def test_paints_per_parcel_scalars_into_voxel_space(self):
        from nltools.data import BrainData

        adj, mask_img, n_voxels, n_per_roi = self._setup(n_rois=3)
        values = np.array([10.0, 20.0, 30.0])
        out = adj.to_brain(values)
        assert isinstance(out, BrainData)
        # Voxel data: first chunk = 10, second = 20, third = 30.
        flat = np.asarray(out.data).ravel()
        assert flat.shape == (n_voxels,)
        assert np.all(flat[:n_per_roi] == 10.0)
        assert np.all(flat[n_per_roi : 2 * n_per_roi] == 20.0)
        assert np.all(flat[2 * n_per_roi :] == 30.0)

    def test_voxels_outside_atlas_get_fill(self):
        from nltools.data import BrainData

        # Use a mask that covers more voxels than the atlas labels —
        # off-atlas voxels should receive `fill` (default NaN).
        n_rois = 2
        n_voxels_per_roi = 3
        n_atlas_voxels = n_rois * n_voxels_per_roi
        # Mask is bigger than atlas: 2 extra voxels with label 0.
        n_voxels = n_atlas_voxels + 2
        affine = np.eye(4)
        atlas_arr = np.zeros((n_voxels, 1, 1), dtype=np.int16)
        for k in range(n_rois):
            start = k * n_voxels_per_roi
            atlas_arr[start : start + n_voxels_per_roi, 0, 0] = k + 1
        atlas_img = nib.Nifti1Image(atlas_arr, affine)
        mask_img = nib.Nifti1Image(np.ones((n_voxels, 1, 1), dtype=np.int8), affine)
        atlas_bd = BrainData(atlas_img, mask=mask_img)
        ss = SpatialScale(
            atlas=atlas_bd,
            roi_labels=np.arange(1, n_rois + 1),
            source_mask=mask_img,
            kind="roi",
        )
        adj = Adjacency(_stack_matrices(n_matrices=n_rois, n_nodes=4), spatial_scale=ss)
        out = adj.to_brain(np.array([1.0, 2.0]), fill=-99.0)
        flat = np.asarray(out.data).ravel()
        # Last 2 voxels are off-atlas → fill value.
        assert flat[-1] == -99.0
        assert flat[-2] == -99.0

    def test_errors_when_no_spatial_scale(self):
        adj = Adjacency(_stack_matrices(n_matrices=3, n_nodes=4))
        assert adj.spatial_scale is None
        with pytest.raises(ValueError, match="spatial_scale"):
            adj.to_brain(np.array([1.0, 2.0, 3.0]))

    def test_errors_on_length_mismatch(self):
        adj, *_ = self._setup(n_rois=3)
        with pytest.raises(ValueError, match="values"):
            adj.to_brain(np.array([1.0, 2.0]))  # wrong length
