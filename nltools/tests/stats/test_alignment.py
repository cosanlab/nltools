"""Tests for nltools.stats.alignment — data alignment and Procrustes."""

import numpy as np
import pytest

from nltools.stats.alignment import align, procrustes, procrustes_distance, align_states
from nltools.data.simulator import Simulator
from nltools.mask import create_sphere


class TestAlign:
    """Test hyperalignment algorithms (SRM, Procrustes)."""

    @pytest.fixture
    def simulated_brains(self):
        """Create simulated BrainData for alignment tests."""
        sim = Simulator()
        s1 = create_sphere([0, 0, 0], radius=3)
        d1 = sim.create_data([0, 1], 1, reps=10, output_dir=None).apply_mask(s1)
        d2 = sim.create_data([0, 1], 2, reps=10, output_dir=None).apply_mask(s1)
        d3 = sim.create_data([0, 1], 3, reps=10, output_dir=None).apply_mask(s1)
        return d1, d2, d3

    @pytest.mark.slow
    def test_deterministic_srm_numpy(self, simulated_brains):
        """Deterministic SRM on numpy arrays."""
        d1, d2, d3 = simulated_brains
        data = [d1.data, d2.data, d3.data]
        out = align(data, method="deterministic_srm")
        assert len(data) == len(out["transformed"])
        assert len(data) == len(out["transformation_matrix"])
        assert data[0].shape == out["common_model"].shape
        transformed = np.dot(data[0], out["transformation_matrix"][0])
        np.testing.assert_almost_equal(
            np.sum(out["transformed"][0] - transformed.T), 0, decimal=3
        )

    @pytest.mark.slow
    def test_probabilistic_srm_numpy(self, simulated_brains):
        """Probabilistic SRM on numpy arrays."""
        d1, d2, d3 = simulated_brains
        data = [d1.data, d2.data, d3.data]
        out = align(data, method="probabilistic_srm")
        assert len(data) == len(out["transformed"])
        assert len(data) == len(out["transformation_matrix"])
        assert data[0].shape == out["common_model"].shape
        transformed = np.dot(data[0], out["transformation_matrix"][0])
        np.testing.assert_almost_equal(
            np.sum(out["transformed"][0] - transformed.T), 0, decimal=3
        )

    @pytest.mark.slow
    def test_procrustes_numpy(self, simulated_brains):
        """Procrustes alignment on numpy arrays."""
        d1, d2, d3 = simulated_brains
        data = [d1.data, d2.data, d3.data]
        out = align(data, method="procrustes")
        assert len(data) == len(out["transformed"])
        assert data[0].shape == out["common_model"].shape
        assert len(data) == len(out["transformation_matrix"])
        assert len(data) == len(out["disparity"])
        centered = data[0] - np.mean(data[0], 0)
        transformed = (
            np.dot(centered / np.linalg.norm(centered), out["transformation_matrix"][0])
            * out["scale"][0]
        )
        np.testing.assert_almost_equal(
            np.sum(out["transformed"][0] - transformed.T), 0, decimal=3
        )


class TestProcrustes:
    """Test Procrustes transformation directly."""

    def test_basic(self):
        """Procrustes on two similar matrices should yield small disparity."""
        np.random.seed(42)
        n = 20
        mat1 = np.random.randn(n, 5)
        mat2 = mat1 + np.random.randn(n, 5) * 0.1
        mtx1, mtx2, disparity, R, s = procrustes(mat1, mat2)
        assert disparity < 0.5  # Should be small since matrices are similar
        assert R.shape == (5, 5)  # Rotation matrix


class TestProcrustesDistance:
    """Test Procrustes distance with permutation testing."""

    def test_basic(self):
        """Procrustes distance with permutation test."""
        np.random.seed(42)
        mat1 = np.random.randn(20, 5)
        mat2 = mat1 + np.random.randn(20, 5) * 0.1
        result = procrustes_distance(mat1, mat2, n_permute=100, random_state=42)
        assert "similarity" in result
        assert "p" in result
        assert 0 <= result["p"] <= 1
        assert isinstance(result["similarity"], (float, np.floating))


class TestAlignStates:
    """Test state alignment using Hungarian algorithm."""

    def test_reorder_scrambled_states(self):
        """Align scrambled state columns back to reference."""
        import pandas as pd

        n = 20
        states = pd.DataFrame(
            {
                "State1": np.random.randint(1, 100, n),
                "State2": np.random.randint(1, 100, n),
                "State3": np.random.randint(1, 100, n),
            }
        )
        scramble_index = np.array([2, 0, 1])
        scrambled = states.iloc[:, scramble_index]

        assert np.array_equal(
            align_states(scrambled, states, return_index=True), scramble_index
        )
        assert np.array_equal(
            states.shape, align_states(scrambled, states, return_index=False).shape
        )
