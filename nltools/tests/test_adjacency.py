import os
import numpy as np
import pandas as pd
from nltools.data import Adjacency, Design_Matrix
import networkx as nx
from scipy.stats import pearsonr
from scipy.linalg import block_diag
from pathlib import Path


def test_type_single(sim_adjacency_single):
    assert sim_adjacency_single.matrix_type == "distance"
    dat_single2 = Adjacency(1 - sim_adjacency_single.squareform())
    assert dat_single2.matrix_type == "similarity"
    assert sim_adjacency_single.issymmetric


def test_type_directed(sim_adjacency_directed):
    assert not sim_adjacency_directed.issymmetric


def test_length(sim_adjacency_multiple):
    assert len(sim_adjacency_multiple) == sim_adjacency_multiple.data.shape[0]
    assert len(sim_adjacency_multiple[0]) == 1


def test_indexing(sim_adjacency_multiple):
    assert len(sim_adjacency_multiple[0]) == 1
    assert len(sim_adjacency_multiple[0:4]) == 4
    assert len(sim_adjacency_multiple[0, 2, 3]) == 3


def test_arithmetic(sim_adjacency_directed):
    assert (sim_adjacency_directed + 5).data[0] == sim_adjacency_directed.data[0] + 5
    assert (sim_adjacency_directed - 0.5).data[0] == sim_adjacency_directed.data[
        0
    ] - 0.5
    assert (sim_adjacency_directed * 5).data[0] == sim_adjacency_directed.data[0] * 5
    assert np.all(
        np.isclose(
            (sim_adjacency_directed + sim_adjacency_directed).data,
            (sim_adjacency_directed * 2).data,
        )
    )
    assert np.all(
        np.isclose(
            (sim_adjacency_directed * 2 - sim_adjacency_directed).data,
            sim_adjacency_directed.data,
        )
    )
    np.testing.assert_almost_equal(
        ((2 * sim_adjacency_directed / 2) / sim_adjacency_directed).mean(), 1, decimal=4
    )


def test_copy(sim_adjacency_multiple):
    assert np.all(sim_adjacency_multiple.data == sim_adjacency_multiple.copy().data)


def test_squareform(sim_adjacency_multiple):
    assert len(sim_adjacency_multiple.squareform()) == len(sim_adjacency_multiple)
    assert (
        sim_adjacency_multiple[0].squareform().shape
        == sim_adjacency_multiple[0].square_shape()
    )


def test_write_multiple(sim_adjacency_multiple, tmpdir):
    sim_adjacency_multiple.write(
        os.path.join(str(tmpdir.join("Test.csv"))), method="long"
    )
    dat_multiple2 = Adjacency(
        os.path.join(str(tmpdir.join("Test.csv"))), matrix_type="distance_flat"
    )
    assert np.all(np.isclose(sim_adjacency_multiple.data, dat_multiple2.data))

    # Test i/o for hdf5
    sim_adjacency_multiple.write(os.path.join(str(tmpdir.join("test_write.h5"))))
    b = Adjacency(os.path.join(tmpdir.join("test_write.h5")))
    for k in ["Y", "matrix_type", "is_single_matrix", "issymmetric", "data"]:
        if k == "data":
            assert np.allclose(b.__dict__[k], sim_adjacency_multiple.__dict__[k])
        elif k == "Y":
            assert all(b.__dict__[k].eq(sim_adjacency_multiple.__dict__[k]).values)
        else:
            assert b.__dict__[k] == sim_adjacency_multiple.__dict__[k]


def test_read_and_write_directed(sim_adjacency_directed, tmpdir):
    sim_adjacency_directed.write(
        os.path.join(str(tmpdir.join("Test.csv"))), method="long"
    )
    dat_directed2 = Adjacency(
        os.path.join(str(tmpdir.join("Test.csv"))), matrix_type="directed_flat"
    )
    assert np.all(np.isclose(sim_adjacency_directed.data, dat_directed2.data))

    # Load Path
    dat_directed2 = Adjacency(
        Path(tmpdir.join("Test.csv")), matrix_type="directed_flat"
    )
    assert np.all(np.isclose(sim_adjacency_directed.data, dat_directed2.data))


def test_mean(sim_adjacency_multiple):
    assert isinstance(sim_adjacency_multiple.mean(axis=0), Adjacency)
    assert len(sim_adjacency_multiple.mean(axis=0)) == 1
    assert len(sim_adjacency_multiple.mean(axis=1)) == len(
        np.mean(sim_adjacency_multiple.data, axis=1)
    )


def test_std(sim_adjacency_multiple):
    assert isinstance(sim_adjacency_multiple.std(axis=0), Adjacency)
    assert len(sim_adjacency_multiple.std(axis=0)) == 1
    assert len(sim_adjacency_multiple.std(axis=1)) == len(
        np.std(sim_adjacency_multiple.data, axis=1)
    )


def test_sum():
    n = 10
    a = Adjacency(np.ones((n, n)), matrix_type="directed")
    assert a.sum() == n**2
    a = Adjacency([a, a])
    assert a.sum().data.sum() == (n**2) * 2

    a = Adjacency(np.ones((n, n)), matrix_type="similarity")
    assert a.sum() == n * (n - 1) / 2
    a = Adjacency([a, a])
    assert a.sum().data.sum() == n * (n - 1)

    a = Adjacency(np.ones((n, n)), matrix_type="distance")
    assert a.sum() == n * (n - 1) / 2
    a = Adjacency([a, a])
    assert a.sum().data.sum() == n * (n - 1)


def test_similarity(sim_adjacency_multiple):
    n_permute = 1000
    assert len(
        sim_adjacency_multiple.similarity(
            sim_adjacency_multiple[0].squareform(), perm_type="1d", n_permute=n_permute
        )
    ) == len(sim_adjacency_multiple)
    assert len(
        sim_adjacency_multiple.similarity(
            sim_adjacency_multiple[0].squareform(),
            perm_type="1d",
            metric="pearson",
            n_permute=n_permute,
        )
    ) == len(sim_adjacency_multiple)
    assert len(
        sim_adjacency_multiple.similarity(
            sim_adjacency_multiple[0].squareform(),
            perm_type="1d",
            metric="kendall",
            n_permute=n_permute,
        )
    ) == len(sim_adjacency_multiple)

    data2 = sim_adjacency_multiple[0].copy()
    data2.data = data2.data + np.random.randn(len(data2.data)) * 0.1
    assert (
        sim_adjacency_multiple[0].similarity(
            data2.squareform(), perm_type=None, n_permute=n_permute
        )["correlation"]
        > 0.5
    )
    assert (
        sim_adjacency_multiple[0].similarity(
            data2.squareform(), perm_type="1d", n_permute=n_permute
        )["correlation"]
        > 0.5
    )
    assert (
        sim_adjacency_multiple[0].similarity(
            data2.squareform(), perm_type="2d", n_permute=n_permute
        )["correlation"]
        > 0.5
    )


def test_similarity_matrix_permutation():
    dat = np.random.multivariate_normal([2, 6], [[0.5, 2], [0.5, 3]], 190)
    x = Adjacency(dat[:, 0])
    y = Adjacency(dat[:, 1])
    stats = x.similarity(y, perm_type="2d", n_permute=1000)
    assert (
        (stats["correlation"] > 0.4)
        & (stats["correlation"] < 0.85)
        & (stats["p"] < 0.001)
    )
    stats = x.similarity(y, perm_type=None)
    assert (stats["correlation"] > 0.4) & (stats["correlation"] < 0.85)


def test_directed_similarity():
    dat = np.random.multivariate_normal([2, 6], [[0.5, 2], [0.5, 3]], 400)
    x = Adjacency(dat[:, 0].reshape(20, 20), matrix_type="directed")
    y = Adjacency(dat[:, 1].reshape(20, 20), matrix_type="directed")
    # Ignore diagonal
    stats = x.similarity(y, perm_type="1d", ignore_diagonal=True, n_permute=1000)
    assert (
        (stats["correlation"] > 0.4)
        & (stats["correlation"] < 0.85)
        & (stats["p"] < 0.001)
    )
    # Use diagonal
    stats = x.similarity(y, perm_type=None, ignore_diagonal=False)
    assert (stats["correlation"] > 0.4) & (stats["correlation"] < 0.85)
    # Error out but make usre TypeError is the reason why
    try:
        x.similarity(y, perm_type="2d")
    except TypeError as _:  # noqa
        pass


def test_distance(sim_adjacency_multiple):
    assert isinstance(sim_adjacency_multiple.distance(), Adjacency)
    assert sim_adjacency_multiple.distance().square_shape()[0] == len(
        sim_adjacency_multiple
    )


def test_ttest(sim_adjacency_multiple):
    out = sim_adjacency_multiple.ttest()
    assert len(out["t"]) == 1
    assert len(out["p"]) == 1
    assert out["t"].shape()[0] == sim_adjacency_multiple.shape()[1]
    assert out["p"].shape()[0] == sim_adjacency_multiple.shape()[1]
    out = sim_adjacency_multiple.ttest(permutation=True, n_permute=1000)
    assert len(out["t"]) == 1
    assert len(out["p"]) == 1
    assert out["t"].shape()[0] == sim_adjacency_multiple.shape()[1]
    assert out["p"].shape()[0] == sim_adjacency_multiple.shape()[1]


def test_threshold(sim_adjacency_directed):
    assert np.sum(sim_adjacency_directed.threshold(upper=0.8).data == 0) == 10
    assert sim_adjacency_directed.threshold(upper=0.8, binarize=True).data[0]
    assert (
        np.sum(sim_adjacency_directed.threshold(upper="70%", binarize=True).data) == 5
    )
    assert np.sum(sim_adjacency_directed.threshold(lower=0.4, binarize=True).data) == 6


def test_graph_directed(sim_adjacency_directed):
    assert isinstance(sim_adjacency_directed.to_graph(), nx.DiGraph)


def test_graph_single(sim_adjacency_single):
    assert isinstance(sim_adjacency_single.to_graph(), nx.Graph)


def test_append(sim_adjacency_single):
    a = Adjacency()
    a = a.append(sim_adjacency_single)
    assert a.shape() == sim_adjacency_single.shape()
    a = a.append(a)
    assert a.shape() == (2, 6)


def test_bootstrap(sim_adjacency_multiple):
    n_samples = 3
    b = sim_adjacency_multiple.bootstrap("mean", n_samples=n_samples)
    assert isinstance(b["Z"], Adjacency)
    b = sim_adjacency_multiple.bootstrap("std", n_samples=n_samples)
    assert isinstance(b["Z"], Adjacency)


def test_plot(sim_adjacency_multiple):
    sim_adjacency_multiple[0].plot()
    sim_adjacency_multiple.plot()


def test_plot_mds(sim_adjacency_single):
    sim_adjacency_single.plot_mds()
    sim_adjacency_single.plot_mds(n_components=3)


def test_similarity_conversion(sim_adjacency_single):
    np.testing.assert_approx_equal(
        -1,
        pearsonr(
            sim_adjacency_single.data,
            sim_adjacency_single.distance_to_similarity().data,
        )[0],
        significant=1,
    )


def test_regression():
    # Test Adjacency Regression
    m1 = block_diag(np.ones((4, 4)), np.zeros((4, 4)), np.zeros((4, 4)))
    m2 = block_diag(np.zeros((4, 4)), np.ones((4, 4)), np.zeros((4, 4)))
    m3 = block_diag(np.zeros((4, 4)), np.zeros((4, 4)), np.ones((4, 4)))
    Y = Adjacency(m1 * 1 + m2 * 2 + m3 * 3, matrix_type="similarity")
    X = Adjacency([m1, m2, m3], matrix_type="similarity")

    stats = Y.regress(X)
    assert np.allclose(stats["beta"], np.array([1, 2, 3]))

    # Test Design_Matrix Regression
    n = 10
    d = Adjacency(
        [
            block_diag(np.ones((4, 4)) + np.random.randn(4, 4) * 0.1, np.zeros((8, 8)))
            for _ in range(n)
        ],
        matrix_type="similarity",
    )
    X = Design_Matrix(np.ones(n))
    stats = d.regress(X)
    out = stats["beta"].cluster_summary(
        clusters=["Group1"] * 4 + ["Group2"] * 8, summary="within"
    )
    assert np.allclose(
        np.array([out["Group1"], out["Group2"]]), np.array([1, 0]), rtol=1e-01
    )  # np.allclose(np.sum(stats['beta']-np.array([1,2,3])),0)


def test_social_relations_model():
    data = Adjacency(
        np.array(
            [
                [np.nan, 8, 5, 10],
                [7, np.nan, 7, 6],
                [8, 7, np.nan, 5],
                [4, 5, 0, np.nan],
            ]
        ),
        matrix_type="directed",
    )
    data2 = data.append(data)
    results1 = data.social_relations_model()
    assert isinstance(data.social_relations_model(), pd.Series)
    assert isinstance(data2.social_relations_model(), pd.DataFrame)
    assert len(results1["actor_effect"]) == data.square_shape()[0]
    assert results1["relationship_effect"].shape == data.square_shape()
    np.testing.assert_approx_equal(results1["actor_variance"], 3.33, significant=2)
    np.testing.assert_approx_equal(results1["partner_variance"], 0.66, significant=2)
    np.testing.assert_approx_equal(
        results1["relationship_variance"], 3.33, significant=2
    )
    np.testing.assert_approx_equal(
        results1["actor_partner_correlation"], 0.22, significant=2
    )
    np.testing.assert_approx_equal(
        results1["dyadic_reciprocity_correlation"], 0.2, significant=2
    )

    # # Test stats_label_distance - FAILED - Need to sort this out
    # labels = np.array(['group1','group1','group2','group2'])
    # stats = dat_multiple[0].stats_label_distance(labels)
    # assert np.isclose(stats['group1']['mean'],-1*stats['group2']['mean'])


def test_isc(sim_adjacency_single):
    n_boot = 100
    for metric in ["median", "mean"]:
        stats = sim_adjacency_single.isc(
            metric=metric, n_bootstraps=n_boot, return_bootstraps=True
        )
        assert (stats["isc"] > -1) & (stats["isc"] < 1)
        assert (stats["p"] > 0) & (stats["p"] < 1)
        assert len(stats["null_distribution"]) == n_boot


def test_fisher_r_to_z(sim_adjacency_single):
    np.testing.assert_almost_equal(
        np.nansum(
            sim_adjacency_single.data - sim_adjacency_single.r_to_z().z_to_r().data
        ),
        0,
        decimal=2,
    )


def test_cluster_summary():
    m1 = block_diag(np.ones((4, 4)), np.zeros((4, 4)), np.zeros((4, 4)))
    m2 = block_diag(np.zeros((4, 4)), np.ones((4, 4)), np.zeros((4, 4)))
    m3 = block_diag(np.zeros((4, 4)), np.zeros((4, 4)), np.ones((4, 4)))
    noisy = (m1 * 1 + m2 * 2 + m3 * 3) + np.random.randn(12, 12) * 0.1
    dat = Adjacency(
        noisy, matrix_type="similarity", labels=["C1"] * 4 + ["C2"] * 4 + ["C3"] * 4
    )

    clusters = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    cluster_mean = dat.cluster_summary(clusters=clusters)
    for i, j in zip(
        np.array([1, 2, 3]), np.array([cluster_mean[x] for x in cluster_mean])
    ):
        np.testing.assert_almost_equal(i, j, decimal=1)

    for i in dat.cluster_summary(clusters=clusters, summary="between").values():
        np.testing.assert_almost_equal(0, i, decimal=1)

    for i in dat.cluster_summary(clusters=clusters, summary="between").values():
        np.testing.assert_almost_equal(0, i, decimal=1)
