"""Tests for nltools.algorithms.inference.intersubject — ISC, ISFC, ISPS."""

import numpy as np
import polars as pl
from numpy import sin, pi, arange
import pytest

from nltools.algorithms.inference.intersubject import isc, isc_group, isfc, isps


class TestISC:
    """Test intersubject correlation calculation."""

    def test_isc_accepts_polars_dataframe(self, multisubject_correlated_data):
        """ISC should accept a polars DataFrame and produce the same result as numpy."""
        np_result = isc(multisubject_correlated_data, n_samples=50, random_state=1)
        pl_df = pl.DataFrame(
            multisubject_correlated_data,
            schema=[f"s{i}" for i in range(multisubject_correlated_data.shape[1])],
        )
        pl_result = isc(pl_df, n_samples=50, random_state=1)
        np.testing.assert_allclose(pl_result["isc"], np_result["isc"])

    def test_summary_statistic_selects_leave_one_out(self):
        """`summary_statistic='leave-one-out'` reaches the engine's LOO path."""
        data = np.random.default_rng(3).standard_normal((60, 3))

        loo = isc(data, summary_statistic="leave-one-out", n_samples=20, random_state=0)

        expected = np.median(
            [
                np.corrcoef(
                    data[:, i],
                    data[:, [j for j in range(3) if j != i]].mean(axis=1),
                )[0, 1]
                for i in range(3)
            ]
        )
        np.testing.assert_allclose(loo["isc"], expected, rtol=1e-12)

        pairwise = isc(data, n_samples=20, random_state=0)
        assert not np.isclose(loo["isc"], pairwise["isc"])

    def test_three_dimensional_input_returns_one_value_per_voxel(self):
        """A `(n_obs, n_subjects, n_voxels)` array gives one result per voxel."""
        rng = np.random.default_rng(7)
        shared = np.repeat(rng.standard_normal((40, 1)), 4, axis=1)
        noise = rng.standard_normal((40, 4))
        data = np.stack([shared, noise], axis=-1)

        result = isc(data, n_samples=20, random_state=0)

        assert result["isc"].shape == (2,)
        assert result["p"].shape == (2,)
        assert result["ci"][0].shape == (2,)
        assert np.isclose(result["isc"][0], 1.0)


class TestISCGroup:
    """Test group-level ISC comparison."""

    @pytest.mark.parametrize("method", ["permute"])
    def test_isc_group_comparison(self, method):
        """Group ISC difference should reflect underlying correlation difference."""
        n_samples = 100
        diff = 0.2
        data = np.random.RandomState(42).multivariate_normal(
            [0] * 10,
            [
                [1, 0.2, 0.5, 0.7, 0.3, 0, 0, 0, 0, 0],
                [0.2, 1, 0.6, 0.1, 0.2, 0, 0, 0, 0, 0],
                [0.5, 0.6, 1, 0.3, 0.1, 0, 0, 0, 0, 0],
                [0.7, 0.1, 0.3, 1, 0.4, 0, 0, 0, 0, 0],
                [0.3, 0.2, 0.1, 0.4, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0.2 + diff, 0.5 + diff, 0.7 + diff, 0.3 + diff],
                [0, 0, 0, 0, 0, 0.2 + diff, 1, 0.6 + diff, 0.1 + diff, 0.2 + diff],
                [0, 0, 0, 0, 0, 0.5 + diff, 0.6 + diff, 1, 0.3 + diff, 0.1 + diff],
                [0, 0, 0, 0, 0, 0.7 + diff, 0.1 + diff, 0.3 + diff, 1, 0.4 + diff],
                [0, 0, 0, 0, 0, 0.3 + diff, 0.2 + diff, 0.1 + diff, 0.4 + diff, 1],
            ],
            500,
        )
        group1 = data[:, :5]
        group2 = data[:, 5:]

        stats = isc_group(
            group1,
            group2,
            method=method,
            return_null=True,
            n_samples=n_samples,
        )
        np.testing.assert_almost_equal(stats["isc_group_difference"], diff, decimal=0)
        assert 0 < stats["p"] < 1
        assert len(stats["null_dist"]) <= n_samples
        assert len(stats["null_dist"]) >= n_samples * 0.95


class TestISFC:
    """Test intersubject functional connectivity."""

    def test_isfc_basic(self, sub_roi_data):
        """ISFC should return per-subject connectivity matrices."""
        isfc_out = isfc(sub_roi_data)
        isfc_mean = np.array(isfc_out).mean(axis=0)
        assert len(isfc_out) == 10
        assert isfc_mean.shape == (5, 5)
        np.testing.assert_almost_equal(
            np.array(isfc_out).mean(axis=0).mean(), 0, decimal=1
        )


class TestISPS:
    """Test intersubject phase synchrony."""

    def test_isps_basic(self):
        """ISPS should detect synchronized vs desynchronized periods."""
        sampling_freq = 0.5
        time = arange(0, 200, 1)
        amplitude = 5
        freq = 0.1
        n_sub = 15
        simulation = amplitude * sin(2 * pi * freq * time)
        simulation = np.array([simulation] * n_sub).T
        simulation += np.random.randn(simulation.shape[0], simulation.shape[1]) * 2
        # Desynchronize middle portion
        simulation[50:150, :] = np.random.randn(100, simulation.shape[1]) * 5

        stats = isps(
            simulation, low_cut=0.05, high_cut=0.2, sampling_freq=sampling_freq
        )

        assert stats["average_angle"].shape == time.shape
        assert stats["vector_length"].shape == time.shape
        assert stats["p"].shape == time.shape
        # Desynchronized period should have higher p-values
        assert stats["p"][50:150].mean() > np.mean(
            [stats["p"][:50].mean(), stats["p"][150:].mean()]
        )
        # Desynchronized period should have lower vector length
        assert stats["vector_length"][50:150].mean() < np.mean(
            [stats["vector_length"][:50].mean(), stats["vector_length"][150:].mean()]
        )
