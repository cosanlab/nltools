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

        assert not hasattr(result, "model_")
        assert not hasattr(result, "X_")
        assert not hasattr(result, "ridge_weights")


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
