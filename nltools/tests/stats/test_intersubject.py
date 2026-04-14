"""Tests for nltools.stats.intersubject — ISC, ISFC, ISPS."""

import numpy as np
import polars as pl
from numpy import sin, pi, arange
import pytest

from nltools.stats.intersubject import isc, isc_group, isfc, isps


class TestISC:
    """Test intersubject correlation calculation."""

    @pytest.mark.parametrize("method", ["bootstrap", "circle_shift", "phase_randomize"])
    @pytest.mark.parametrize("metric", ["median", "mean"])
    def test_isc_methods_and_metrics(
        self, multisubject_correlated_data, method, metric
    ):
        """ISC with various methods and aggregation metrics."""
        stats = isc(
            multisubject_correlated_data,
            method=method,
            metric=metric,
            n_samples=100,
            return_null=True,
        )
        assert stats["isc"] > 0.1
        assert -1 < stats["isc"] < 1
        assert 0 < stats["p"] < 1
        assert len(stats["null_distribution"]) == 100

    def test_isc_accepts_polars_dataframe(self, multisubject_correlated_data):
        """ISC should accept a polars DataFrame and produce the same result as numpy."""
        np_result = isc(multisubject_correlated_data, n_samples=50, random_state=1)
        pl_df = pl.DataFrame(
            multisubject_correlated_data,
            schema=[f"s{i}" for i in range(multisubject_correlated_data.shape[1])],
        )
        pl_result = isc(pl_df, n_samples=50, random_state=1)
        np.testing.assert_allclose(pl_result["isc"], np_result["isc"])


class TestISCGroup:
    """Test group-level ISC comparison."""

    @pytest.mark.parametrize("method", ["permute", "bootstrap"])
    @pytest.mark.parametrize("metric", ["median", "mean"])
    def test_isc_group_comparison(self, method, metric):
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
            metric=metric,
            method=method,
            return_null=True,
            n_samples=n_samples,
        )
        np.testing.assert_almost_equal(stats["isc_group_difference"], diff, decimal=0)
        assert 0 < stats["p"] < 1
        assert len(stats["null_distribution"]) <= n_samples
        assert len(stats["null_distribution"]) >= n_samples * 0.95

    def test_isc_group_accepts_polars_dataframe(self):
        """isc_group should accept polars DataFrames for group1/group2."""
        rng = np.random.RandomState(0)
        g1 = rng.randn(100, 5)
        g2 = rng.randn(100, 5)
        g1_pl = pl.DataFrame(g1, schema=[f"s{i}" for i in range(5)])
        g2_pl = pl.DataFrame(g2, schema=[f"s{i}" for i in range(5)])
        out_np = isc_group(g1, g2, n_samples=50, random_state=1)
        out_pl = isc_group(g1_pl, g2_pl, n_samples=50, random_state=1)
        np.testing.assert_allclose(
            out_pl["isc_group_difference"], out_np["isc_group_difference"]
        )


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

    def test_isfc_parallelization(self, sub_roi_data):
        """Serial and parallel ISFC should give identical results."""
        result_serial = isfc(sub_roi_data, n_jobs=1)
        result_parallel = isfc(sub_roi_data, n_jobs=-1)
        assert len(result_serial) == len(result_parallel) == 10
        for i in range(10):
            np.testing.assert_allclose(
                result_serial[i], result_parallel[i], rtol=1e-10, atol=1e-10
            )

    def test_isfc_deterministic(self, sub_roi_data):
        """Parallel ISFC should be deterministic across runs."""
        r1 = isfc(sub_roi_data, n_jobs=-1)
        r2 = isfc(sub_roi_data, n_jobs=-1)
        for i in range(10):
            np.testing.assert_allclose(r1[i], r2[i], rtol=1e-10, atol=1e-10)

    @pytest.mark.slow
    def test_isfc_different_njobs(self, sub_roi_data):
        """Different n_jobs values should produce identical results."""
        r1 = isfc(sub_roi_data, n_jobs=1)
        r2 = isfc(sub_roi_data, n_jobs=2)
        r_all = isfc(sub_roi_data, n_jobs=-1)
        for i in range(10):
            np.testing.assert_allclose(r1[i], r2[i], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(r1[i], r_all[i], rtol=1e-10, atol=1e-10)

    def test_isfc_default_parallel(self, sub_roi_data):
        """Default call should use parallel execution."""
        result_default = isfc(sub_roi_data)
        result_explicit = isfc(sub_roi_data, n_jobs=-1)
        for i in range(10):
            np.testing.assert_allclose(
                result_default[i], result_explicit[i], rtol=1e-10, atol=1e-10
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

    def test_isps_accepts_polars_dataframe(self):
        """isps should accept a polars DataFrame."""
        sampling_freq = 0.5
        time = arange(0, 50, 1)
        sim = np.array([sin(2 * pi * 0.1 * time)] * 5).T
        sim_pl = pl.DataFrame(sim, schema=[f"s{i}" for i in range(5)])
        out = isps(sim_pl, low_cut=0.05, high_cut=0.2, sampling_freq=sampling_freq)
        assert out["average_angle"].shape == time.shape
