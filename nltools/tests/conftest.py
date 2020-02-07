import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from nltools.simulator import Simulator
from nltools.data import Brain_Data, Adjacency, Groupby, Design_Matrix
from nltools.mask import create_sphere


@pytest.fixture(scope="module", params=["2mm"])
def sim_brain_data():
    # MNI_Template["resolution"] = request.params
    sim = Simulator()
    # r = 10
    sigma = 1
    y = [0, 1]
    n_reps = 3
    dat = sim.create_data(y, sigma, reps=n_reps)
    dat.X = pd.DataFrame(
        {"Intercept": np.ones(len(dat.Y)), "X1": np.array(dat.Y).flatten()}, index=None
    )
    return dat


@pytest.fixture(scope="module")
def sim_design_matrix():
    # Design matrices are specified in terms of sampling frequency
    TR = 2.0
    sampling_freq = 1.0 / TR
    return Design_Matrix(
        np.random.randint(2, size=(500, 4)),
        columns=["face_A", "face_B", "house_A", "house_B"],
        sampling_freq=sampling_freq,
    )


@pytest.fixture(scope="module")
def sim_adjacency_single():
    sim = np.random.multivariate_normal(
        [0, 0, 0, 0],
        [
            [1, 0.8, 0.1, 0.4],
            [0.8, 1, 0.6, 0.1],
            [0.1, 0.6, 1, 0.3],
            [0.4, 0.1, 0.3, 1],
        ],
        100,
    )
    data = pairwise_distances(sim.T, metric="correlation")
    labels = ["v_%s" % (x + 1) for x in range(sim.shape[1])]
    return Adjacency(data, labels=labels)


@pytest.fixture(scope="module")
def sim_adjacency_multiple():
    n = 10
    sim = np.random.multivariate_normal(
        [0, 0, 0, 0],
        [
            [1, 0.8, 0.1, 0.4],
            [0.8, 1, 0.6, 0.1],
            [0.1, 0.6, 1, 0.3],
            [0.4, 0.1, 0.3, 1],
        ],
        100,
    )
    data = pairwise_distances(sim.T, metric="correlation")
    dat_all = []
    for t in range(n):
        tmp = data
        dat_all.append(tmp)
    labels = ["v_%s" % (x + 1) for x in range(sim.shape[1])]
    return Adjacency(dat_all, labels=labels)


@pytest.fixture(scope="module")
def sim_adjacency_directed():
    sim_directed = np.array(
        [
            [1, 0.5, 0.3, 0.4],
            [0.8, 1, 0.2, 0.1],
            [0.7, 0.6, 1, 0.5],
            [0.85, 0.4, 0.3, 1],
        ]
    )
    labels = ["v_%s" % (x + 1) for x in range(sim_directed.shape[1])]
    return Adjacency(sim_directed, matrix_type="directed", labels=labels)


@pytest.fixture(scope="module")
def sim_groupby(sim_brain_data):
    r = 10
    s1 = create_sphere([12, 10, -8], radius=r)
    s2 = create_sphere([22, -2, -22], radius=r)
    mask = Brain_Data([s1, s2])
    return Groupby(sim_brain_data, mask)
