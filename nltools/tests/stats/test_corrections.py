"""Tests for nltools.stats.corrections — multiple comparison corrections."""

import numpy as np

from nltools.stats.corrections import fdr, holm_bonf, threshold, multi_threshold


class TestFDR:
    """Test false discovery rate correction."""

    def test_fdr_basic(self):
        """FDR with mix of significant and non-significant p-values."""
        p_values = np.array([0.001, 0.01, 0.03, 0.04, 0.05, 0.5, 0.9])
        thr = fdr(p_values, q=0.05)
        assert isinstance(thr, float)
        assert 0 < thr < 1

    def test_fdr_all_significant(self):
        """FDR with all very small p-values."""
        p_values = np.array([0.001, 0.002, 0.003, 0.004])
        thr = fdr(p_values, q=0.05)
        assert thr > 0

    def test_fdr_none_significant(self):
        """FDR with all large p-values should return -1 (no threshold)."""
        p_values = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        thr = fdr(p_values, q=0.05)
        assert thr == -1


class TestHolmBonf:
    """Test Holm-Bonferroni correction."""

    def test_holm_bonf_returns_threshold(self):
        """Holm-Bonferroni should return a threshold value."""
        p_values = np.array([0.001, 0.01, 0.03, 0.5])
        thr = holm_bonf(p_values)
        assert isinstance(thr, (float, np.floating))
        assert 0 < thr <= 0.05  # default alpha=0.05

    def test_holm_bonf_none_significant(self):
        """All large p-values should produce threshold of -1 (no threshold)."""
        p_values = np.array([0.3, 0.5, 0.8, 0.9])
        thr = holm_bonf(p_values)
        assert thr == -1


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
