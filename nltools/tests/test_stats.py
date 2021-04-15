import numpy as np
from numpy import sin, pi, arange
import pandas as pd
from nltools.stats import (
    one_sample_permutation,
    two_sample_permutation,
    correlation_permutation,
    matrix_permutation,
    downsample,
    upsample,
    winsorize,
    align,
    transform_pairwise,
    _calc_pvalue,
    find_spikes,
    isc,
    isfc,
    isps,
    fisher_r_to_z,
    fisher_z_to_r,
    align_states,
)
from nltools.simulator import Simulator
from nltools.mask import create_sphere
from scipy.spatial.distance import squareform

# import pytest


def test_permutation():
    dat = np.random.multivariate_normal([2, 6], [[0.5, 2], [0.5, 3]], 1000)
    x = dat[:, 0]
    y = dat[:, 1]
    stats = two_sample_permutation(x, y, tail=1, n_permute=1000)
    assert (stats["mean"] < -2) & (stats["mean"] > -6) & (stats["p"] < 0.001)
    stats = one_sample_permutation(x - y, tail=1, n_permute=1000)
    assert (stats["mean"] < -2) & (stats["mean"] > -6) & (stats["p"] < 0.001)
    for method in ["permute", "circle_shift", "phase_randomize"]:
        for metric in ["spearman", "kendall", "pearson"]:
            stats = correlation_permutation(
                x, y, metric=metric, method=method, n_permute=500, tail=1
            )
            assert (
                (stats["correlation"] > 0.4)
                & (stats["correlation"] < 0.85)
                & (stats["p"] < 0.05)
            )

    # with pytest.raises(ValueError):
    # 	correlation_permutation(x, y, metric='kendall',tail=3)
    # with pytest.raises(ValueError):
    # 	correlation_permutation(x, y, metric='doesntwork',tail=3)
    s = np.random.normal(0, 1, 10000)
    two_sided = _calc_pvalue(all_p=s, stat=1.96, tail=2)
    upper_p = _calc_pvalue(all_p=s, stat=1.96, tail=1)
    lower_p = _calc_pvalue(all_p=s, stat=-1.96, tail=1)
    sum_p = upper_p + lower_p
    np.testing.assert_almost_equal(two_sided, sum_p, decimal=3)

    # Test matrix_permutation
    dat = np.random.multivariate_normal([2, 6], [[0.5, 2], [0.5, 3]], 190)
    x = squareform(dat[:, 0])
    y = squareform(dat[:, 1])
    stats = matrix_permutation(x, y, n_permute=1000)
    assert (
        (stats["correlation"] > 0.4)
        & (stats["correlation"] < 0.85)
        & (stats["p"] < 0.001)
    )


def test_downsample():
    dat = pd.DataFrame()
    dat["x"] = range(0, 100)
    dat["y"] = np.repeat(range(1, 11), 10)
    assert (
        dat.groupby("y").mean().values.ravel()
        == downsample(
            data=dat["x"], sampling_freq=10, target=1, target_type="hz", method="mean"
        ).values
    ).all
    assert (
        dat.groupby("y").median().values.ravel()
        == downsample(
            data=dat["x"], sampling_freq=10, target=1, target_type="hz", method="median"
        ).values
    ).all
    # with pytest.raises(ValueError):
    # 	downsample(data=list(dat['x']),sampling_freq=10,target=1,target_type='hz',method='median')
    # with pytest.raises(ValueError):
    # 	downsample(data=dat['x'],sampling_freq=10,target=1,target_type='hz',method='doesnotwork')
    # with pytest.raises(ValueError):
    # 	downsample(data=dat['x'],sampling_freq=10,target=1,target_type='doesnotwork',method='median')


def test_upsample():
    dat = pd.DataFrame()
    dat["x"] = range(0, 100)
    dat["y"] = np.repeat(range(1, 11), 10)
    fs = 2
    us = upsample(dat, sampling_freq=1, target=fs, target_type="hz")
    assert dat.shape[0] * fs - fs == us.shape[0]
    fs = 3
    us = upsample(dat, sampling_freq=1, target=fs, target_type="hz")
    assert dat.shape[0] * fs - fs == us.shape[0]
    # with pytest.raises(ValueError):
    # 	upsample(dat,sampling_freq=1,target=fs,target_type='hz',method='doesnotwork')
    # with pytest.raises(ValueError):
    # 	upsample(dat,sampling_freq=1,target=fs,target_type='doesnotwork',method='linear')


def test_winsorize():
    outlier_test = pd.DataFrame(
        [
            92,
            19,
            101,
            58,
            1053,
            91,
            26,
            78,
            10,
            13,
            -40,
            101,
            86,
            85,
            15,
            89,
            89,
            28,
            -5,
            41,
        ]
    )

    out = winsorize(
        outlier_test, cutoff={"quantile": [0.05, 0.95]}, replace_with_cutoff=False
    ).values.squeeze()
    correct_result = np.array(
        [
            92,
            19,
            101,
            58,
            101,
            91,
            26,
            78,
            10,
            13,
            -5,
            101,
            86,
            85,
            15,
            89,
            89,
            28,
            -5,
            41,
        ]
    )
    assert np.sum(out == correct_result) == 20

    out = winsorize(
        outlier_test, cutoff={"std": [2, 2]}, replace_with_cutoff=False
    ).values.squeeze()
    correct_result = np.array(
        [
            92,
            19,
            101,
            58,
            101,
            91,
            26,
            78,
            10,
            13,
            -40,
            101,
            86,
            85,
            15,
            89,
            89,
            28,
            -5,
            41,
        ]
    )
    assert np.sum(out == correct_result) == 20

    out = winsorize(
        outlier_test, cutoff={"std": [2, 2]}, replace_with_cutoff=True
    ).values.squeeze()
    correct_result = np.array(
        [
            92.0,
            19.0,
            101.0,
            58.0,
            556.97961997,
            91.0,
            26.0,
            78.0,
            10.0,
            13.0,
            -40.0,
            101.0,
            86.0,
            85.0,
            15.0,
            89.0,
            89.0,
            28.0,
            -5.0,
            41.0,
        ]
    )
    assert np.round(np.mean(out)) == np.round(np.mean(correct_result))


def test_align():
    # Test hyperalignment matrix
    sim = Simulator()
    y = [0, 1]
    n_reps = 10
    s1 = create_sphere([0, 0, 0], radius=3)
    d1 = sim.create_data(y, 1, reps=n_reps, output_dir=None).apply_mask(s1)
    d2 = sim.create_data(y, 2, reps=n_reps, output_dir=None).apply_mask(s1)
    d3 = sim.create_data(y, 3, reps=n_reps, output_dir=None).apply_mask(s1)

    data = [d1.data, d2.data, d3.data]
    out = align(data, method="deterministic_srm")
    assert len(data) == len(out["transformed"])
    assert len(data) == len(out["transformation_matrix"])
    assert data[0].shape == out["common_model"].shape
    transformed = np.dot(data[0], out["transformation_matrix"][0])
    np.testing.assert_almost_equal(
        np.sum(out["transformed"][0] - transformed.T), 0, decimal=3
    )
    assert len(out["isc"]) == out["transformed"][0].shape[0]

    out = align(data, method="probabilistic_srm")
    assert len(data) == len(out["transformed"])
    assert len(data) == len(out["transformation_matrix"])
    assert data[0].shape == out["common_model"].shape
    transformed = np.dot(data[0], out["transformation_matrix"][0])
    np.testing.assert_almost_equal(
        np.sum(out["transformed"][0] - transformed.T), 0, decimal=3
    )
    assert len(out["isc"]) == out["transformed"][0].shape[0]

    out2 = align(data, method="procrustes")
    assert len(data) == len(out2["transformed"])
    assert data[0].shape == out2["common_model"].shape
    assert len(data) == len(out2["transformation_matrix"])
    assert len(data) == len(out2["disparity"])
    centered = data[0] - np.mean(data[0], 0)
    transformed = (
        np.dot(centered / np.linalg.norm(centered), out2["transformation_matrix"][0])
        * out2["scale"][0]
    )
    np.testing.assert_almost_equal(
        np.sum(out2["transformed"][0] - transformed.T), 0, decimal=3
    )
    assert out2["transformed"][0].shape == out2["transformed"][0].shape
    assert (
        out2["transformation_matrix"][0].shape == out2["transformation_matrix"][0].shape
    )
    assert len(out2["isc"]) == out["transformed"][0].shape[0]

    # Test hyperalignment on Brain_Data
    data = [d1, d2, d3]
    out = align(data, method="deterministic_srm")
    assert len(data) == len(out["transformed"])
    assert len(data) == len(out["transformation_matrix"])
    assert data[0].shape() == out["common_model"].shape
    transformed = np.dot(d1.data, out["transformation_matrix"][0].data.T)
    np.testing.assert_almost_equal(
        np.sum(out["transformed"][0].data - transformed), 0, decimal=3
    )
    assert len(out["isc"]) == out["transformed"][0].shape[1]

    out = align(data, method="probabilistic_srm")
    assert len(data) == len(out["transformed"])
    assert len(data) == len(out["transformation_matrix"])
    assert data[0].shape() == out["common_model"].shape
    transformed = np.dot(d1.data, out["transformation_matrix"][0].data.T)
    np.testing.assert_almost_equal(
        np.sum(out["transformed"][0].data - transformed), 0, decimal=3
    )
    assert len(out["isc"]) == out["transformed"][0].shape[1]

    out2 = align(data, method="procrustes")
    assert len(data) == len(out2["transformed"])
    assert data[0].shape() == out2["common_model"].shape()
    assert len(data) == len(out2["transformation_matrix"])
    assert len(data) == len(out2["disparity"])
    centered = data[0].data - np.mean(data[0].data, 0)
    transformed = (
        np.dot(
            centered / np.linalg.norm(centered), out2["transformation_matrix"][0].data
        )
        * out2["scale"][0]
    )
    np.testing.assert_almost_equal(
        np.sum(out2["transformed"][0].data - transformed), 0, decimal=3
    )
    assert out2["transformed"][0].shape() == out2["transformed"][0].shape()
    assert (
        out2["transformation_matrix"][0].shape == out2["transformation_matrix"][0].shape
    )
    assert len(out2["isc"]) == out2["transformed"][0].shape()[1]

    # Test hyperalignment on matrix over time (axis=1)
    sim = Simulator()
    y = [0, 1]
    n_reps = 10
    s1 = create_sphere([0, 0, 0], radius=5)
    d1 = sim.create_data(y, 1, reps=n_reps, output_dir=None).apply_mask(s1)
    d2 = sim.create_data(y, 2, reps=n_reps, output_dir=None).apply_mask(s1)
    d3 = sim.create_data(y, 3, reps=n_reps, output_dir=None).apply_mask(s1)
    data = [d1.data, d2.data, d3.data]

    out = align(data, method="deterministic_srm", axis=1)
    assert len(data) == len(out["transformed"])
    assert len(data) == len(out["transformation_matrix"])
    assert data[0].shape == out["common_model"].shape
    transformed = np.dot(data[0].T, out["transformation_matrix"][0].data)
    np.testing.assert_almost_equal(
        np.sum(out["transformed"][0] - transformed), 0, decimal=3
    )
    assert len(out["isc"]) == out["transformed"][0].shape[1]

    out = align(data, method="probabilistic_srm", axis=1)
    assert len(data) == len(out["transformed"])
    assert len(data) == len(out["transformation_matrix"])
    assert data[0].shape == out["common_model"].shape
    transformed = np.dot(data[0].T, out["transformation_matrix"][0])
    np.testing.assert_almost_equal(
        np.sum(out["transformed"][0] - transformed), 0, decimal=3
    )
    assert len(out["isc"]) == out["transformed"][0].shape[1]

    out2 = align(data, method="procrustes", axis=1)
    assert len(data) == len(out2["transformed"])
    assert data[0].shape == out2["common_model"].shape
    assert len(data) == len(out2["transformation_matrix"])
    assert len(data) == len(out2["disparity"])
    centered = data[0] - np.mean(data[0], 0)
    transformed = (
        np.dot(
            (centered / np.linalg.norm(centered)).T,
            out2["transformation_matrix"][0].data,
        )
        * out2["scale"][0]
    )
    np.testing.assert_almost_equal(
        np.sum(out2["transformed"][0] - transformed), 0, decimal=3
    )
    assert out2["transformed"][0].shape == out2["transformed"][0].shape
    assert (
        out2["transformation_matrix"][0].shape == out2["transformation_matrix"][0].shape
    )
    assert len(out2["isc"]) == out2["transformed"][0].shape[0]

    # Test hyperalignment on Brain_Data over time (axis=1)
    data = [d1, d2, d3]
    out = align(data, method="deterministic_srm", axis=1)
    assert len(data) == len(out["transformed"])
    assert len(data) == len(out["transformation_matrix"])
    assert data[0].shape() == out["common_model"].shape
    transformed = np.dot(d1.data.T, out["transformation_matrix"][0].data).T
    np.testing.assert_almost_equal(
        np.sum(out["transformed"][0].data - transformed), 0, decimal=5
    )
    assert len(out["isc"]) == out["transformed"][0].shape[0]

    out = align(data, method="probabilistic_srm", axis=1)
    assert len(data) == len(out["transformed"])
    assert len(data) == len(out["transformation_matrix"])
    assert data[0].shape() == out["common_model"].shape
    transformed = np.dot(d1.data.T, out["transformation_matrix"][0].data).T
    np.testing.assert_almost_equal(
        np.sum(out["transformed"][0].data - transformed), 0, decimal=5
    )
    assert len(out["isc"]) == out["transformed"][0].shape[0]

    out2 = align(data, method="procrustes", axis=1)
    assert len(data) == len(out2["transformed"])
    assert data[0].shape() == out2["common_model"].shape()
    assert len(data) == len(out2["transformation_matrix"])
    assert len(data) == len(out2["disparity"])
    centered = data[0].data.T - np.mean(data[0].data.T, 0)
    transformed = (
        np.dot(
            centered / np.linalg.norm(centered), out2["transformation_matrix"][0].data
        )
        * out2["scale"][0]
    ).T
    np.testing.assert_almost_equal(
        np.sum(out2["transformed"][0].data - transformed), 0, decimal=5
    )
    assert out2["transformed"][0].shape() == out2["transformed"][0].shape()
    assert (
        out2["transformation_matrix"][0].shape == out2["transformation_matrix"][0].shape
    )
    assert len(out2["isc"]) == out2["transformed"][0].shape()[1]


def test_transform_pairwise():
    n_features = 50
    n_samples = 100
    # Test without groups
    new_n_samples = int(n_samples * (n_samples - 1) / 2)
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(
        n_samples,
    )
    x_new, y_new = transform_pairwise(X, y)
    assert x_new.shape == (new_n_samples, n_features)
    assert y_new.shape == (new_n_samples,)
    assert y_new.ndim == 1
    # Test with groups
    n_subs = 4
    new_n_samples = int(n_subs * ((n_samples / n_subs) * (n_samples / n_subs - 1)) / 2)
    groups = np.repeat(np.arange(1, 1 + n_subs), n_samples / n_subs)
    y = np.vstack((y, groups)).T
    x_new, y_new = transform_pairwise(X, y)
    assert x_new.shape == (new_n_samples, n_features)
    assert y_new.shape == (new_n_samples, 2)
    assert y_new.ndim == 2
    a = y_new[:, 1] == np.repeat(
        np.arange(1, 1 + n_subs), ((n_samples / n_subs) * (n_samples / n_subs - 1)) / 2
    )
    assert a.all()


def test_find_spikes():
    sim = Simulator()
    y = [0, 1]
    n_reps = 50
    s1 = create_sphere([0, 0, 0], radius=3)
    d1 = sim.create_data(y, 1, reps=n_reps, output_dir=None).apply_mask(s1)

    spikes = find_spikes(d1)
    assert isinstance(spikes, pd.DataFrame)
    assert spikes.shape[0] == len(d1)

    spikes = find_spikes(d1.to_nifti())
    assert isinstance(spikes, pd.DataFrame)
    assert spikes.shape[0] == len(d1)


def test_isc():
    n_boot = 100
    dat = np.random.multivariate_normal(
        [0, 0, 0, 0, 0],
        [
            [1, 0.2, 0.5, 0.7, 0.3],
            [0.2, 1, 0.6, 0.1, 0.2],
            [0.5, 0.6, 1, 0.3, 0.1],
            [0.7, 0.1, 0.3, 1, 0.4],
            [0.3, 0.2, 0.1, 0.4, 1],
        ],
        500,
    )
    for method in ["bootstrap", "circle_shift", "phase_randomize"]:
        for metric in ["median", "mean"]:
            stats = isc(
                dat,
                method=method,
                metric=metric,
                n_bootstraps=n_boot,
                return_bootstraps=True,
            )
            assert stats["isc"] > 0.1
            assert (stats["isc"] > -1) & (stats["isc"] < 1)
            assert (stats["p"] > 0) & (stats["p"] < 1)
            assert len(stats["null_distribution"]) == n_boot


def test_isfc():
    def simulate_sub_roi_data(n_sub, n_tr):
        sub_dat = []
        for _ in range(n_sub):
            sub_dat.append(
                np.random.multivariate_normal(
                    [0, 0, 0, 0, 0],
                    [
                        [1, 0.2, 0.5, 0.7, 0.3],
                        [0.2, 1, 0.6, 0.1, 0.2],
                        [0.5, 0.6, 1, 0.3, 0.1],
                        [0.7, 0.1, 0.3, 1, 0.4],
                        [0.3, 0.2, 0.1, 0.4, 1],
                    ],
                    n_tr,
                )
            )
        return sub_dat

    n_sub = 10
    sub_dat = simulate_sub_roi_data(n_sub, 500)
    isfc_out = isfc(sub_dat)
    isfc_mean = np.array(isfc_out).mean(axis=0)
    assert len(isfc_out) == n_sub
    assert isfc_mean.shape == (5, 5)
    np.testing.assert_almost_equal(np.array(isfc_out).mean(axis=0).mean(), 0, decimal=1)


def test_isps():
    sampling_freq = 0.5
    time = arange(0, 200, 1)
    amplitude = 5
    freq = 0.1
    theta = 0
    n_sub = 15
    simulation = amplitude * sin(2 * pi * freq * time + theta)
    simulation = np.array([simulation] * n_sub).T
    simulation += np.random.randn(simulation.shape[0], simulation.shape[1]) * 2
    simulation[50:150, :] = np.random.randn(100, simulation.shape[1]) * 5
    stats = isps(simulation, low_cut=0.05, high_cut=0.2, sampling_freq=sampling_freq)

    assert stats["average_angle"].shape == time.shape
    assert stats["vector_length"].shape == time.shape
    assert stats["p"].shape == time.shape
    assert stats["p"][50:150].mean() > (
        np.mean([stats["p"][:50].mean(), stats["p"][150:].mean()])
    )
    assert stats["vector_length"][50:150].mean() < (
        np.mean(
            [stats["vector_length"][:50].mean(), stats["vector_length"][150:].mean()]
        )
    )


def test_fisher_r_to_z():
    for r in np.arange(0, 1, 0.05):
        np.testing.assert_almost_equal(r, fisher_z_to_r(fisher_r_to_z(r)), decimal=3)


def test_align_states():
    n = 20
    states = pd.DataFrame(
        {
            "State1": np.random.randint(1, 100, n),
            "State2": np.random.randint(1, 100, n),
            "State3": np.random.randint(1, 100, n),
        }
    )
    scramble_index = np.array([2, 0, 1])
    scrambled_states = states.iloc[:, scramble_index]

    assert np.array_equal(
        align_states(scrambled_states, states, return_index=True), scramble_index
    )
    assert np.array_equal(
        states.shape, align_states(scrambled_states, states, return_index=False).shape
    )
