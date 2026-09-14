"""Tests for nltools.algorithms.corrections — multiple comparison corrections."""

import numpy as np

from nltools.algorithms.corrections import fdr, holm_bonf, threshold, multi_threshold


class TestFDR:
    """Test false discovery rate correction."""

    def test_fdr_basic(self):
        """FDR with mix of significant and non-significant p-values."""
        p_values = np.array([0.001, 0.01, 0.03, 0.04, 0.05, 0.5, 0.9])
        thr = fdr(p_values, q=0.05)
        assert isinstance(thr, float)
        assert 0 < thr < 1


class TestHolmBonf:
    """Test Holm-Bonferroni correction."""

    def test_holm_bonf_returns_threshold(self):
        """Holm-Bonferroni should return a threshold value."""
        p_values = np.array([0.001, 0.01, 0.03, 0.5])
        thr = holm_bonf(p_values)
        assert isinstance(thr, (float, np.floating))
        assert 0 < thr <= 0.05  # default alpha=0.05

    def test_holm_bonf_stops_at_the_first_failure(self):
        """The step-down walk stops at the first p above its boundary (G-01)."""
        # Boundaries are [0.025, 0.05]; 0.03 fails the first one, so the whole
        # family fails and no threshold survives.
        assert holm_bonf(np.array([0.03, 0.04])) == -1


class TestThreshold:
    """Test statistical thresholding on BrainData."""

    def test_threshold_basic(self, minimal_brain_data):
        """Threshold should zero out non-significant voxels."""
        from nltools.data import BrainData

        # Use a single image from the collection
        stat = minimal_brain_data[0]
        p = stat.copy()
        p.data = np.random.rand(*stat.data.shape)
        result = threshold(stat, p, thr=0.05)
        assert isinstance(result, BrainData)

    def test_threshold_drops_source_fit_state(self, minimal_brain_data):
        X = np.random.default_rng(0).standard_normal((len(minimal_brain_data), 3))
        minimal_brain_data.fit(model="ridge", X=X, ridge_alpha=1.0)
        p = minimal_brain_data.copy()
        p.data = np.full_like(p.data, 0.01)

        result = threshold(minimal_brain_data, p)

        assert result.model is None
        assert not hasattr(result, "X_")


class TestMultiThreshold:
    """Test multi-level thresholding on BrainData."""

    def test_multi_threshold_basic(self, minimal_brain_data):
        """Multi-threshold should return a cumulative BrainData map."""
        from nltools.data import BrainData

        stat = minimal_brain_data[0]
        p = stat.copy()
        p.data = np.random.rand(*stat.data.shape)
        result = multi_threshold(stat, p, [0.05, 0.01, 0.001])
        assert isinstance(result, BrainData)


def _brain_on_grid(values, mask_flat):
    """BrainData on a 4-voxel grid with the given boolean mask support."""
    import nibabel as nib
    from nltools.data import BrainData

    spatial_shape = (4, 1, 1)
    mask_data = np.array(mask_flat, dtype=bool).reshape(spatial_shape)
    values = np.atleast_2d(np.asarray(values, dtype=float))
    volume = np.zeros(spatial_shape + (values.shape[0],))
    for i, row in enumerate(values):
        volume[..., i][mask_data] = row
    affine = np.eye(4)
    return BrainData(
        nib.Nifti1Image(volume, affine),
        mask=nib.Nifti1Image(mask_data.astype(np.float32), affine),
    )


class TestThresholdVoxelCorrespondence:
    """G-03: the two images must describe the same voxels."""

    def test_threshold_rejects_disjoint_masks(self):
        """Equal voxel counts on disjoint mask support are not the same voxels."""
        import pytest

        stat = _brain_on_grid([5.0, 6.0], [True, True, False, False])
        p = _brain_on_grid([0.01, 0.9], [False, False, True, True])

        with pytest.raises(ValueError, match="same voxels"):
            threshold(stat, p)


class TestThresholdShapePreservation:
    """G-02: thresholding preserves (n_images, n_voxels) (GH #304, #372)."""

    def test_threshold_keeps_shape_when_nothing_survives(self):
        stat = _brain_on_grid([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], [1, 1, 1, 0])
        p = _brain_on_grid([[0.9, 0.9, 0.9], [0.9, 0.9, 0.9]], [1, 1, 1, 0])

        out = threshold(stat, p)

        assert out.data.shape == (2, 3)
        assert np.all(out.data == 0)

    def test_multi_threshold_runs_on_a_multi_image_map(self):
        stat = _brain_on_grid([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], [1, 1, 1, 0])
        p = _brain_on_grid([[0.01, 0.01, 0.01], [0.01, 0.01, 0.01]], [1, 1, 1, 0])

        out = multi_threshold(stat, p, [0.05, 0.02])

        np.testing.assert_array_equal(out.data, 2 * np.sign(stat.data))
